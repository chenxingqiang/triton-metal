"""
Triton到MLX的内存布局适配器
处理Triton中特有的内存布局转换到MLX
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional

# 延迟导入MLX以避免不必要的依赖
_mx = None

def _get_mlx():
    """懒加载MLX"""
    global _mx
    if _mx is None:
        import mlx.core as mx
        _mx = mx
    return _mx

class LayoutType:
    """内存布局类型"""
    ROW_MAJOR = "row_major"  # 行优先
    COL_MAJOR = "col_major"  # 列优先
    BLOCK = "block"          # 分块
    SLICED = "sliced"        # 切片
    STRIDED = "strided"      # 步长
    UNKNOWN = "unknown"      # 未知

class MemoryLayout:
    """内存布局描述"""
    
    def __init__(self, 
                 shape: Tuple[int, ...],
                 layout_type: str = LayoutType.ROW_MAJOR,
                 strides: Optional[Tuple[int, ...]] = None,
                 block_shape: Optional[Tuple[int, ...]] = None,
                 block_strides: Optional[Tuple[int, ...]] = None):
        self.shape = shape
        self.layout_type = layout_type
        self.strides = strides if strides else self._default_strides(shape, layout_type)
        self.block_shape = block_shape
        self.block_strides = block_strides
        
    def _default_strides(self, shape: Tuple[int, ...], layout_type: str) -> Tuple[int, ...]:
        """计算默认步长"""
        if layout_type == LayoutType.ROW_MAJOR:
            # 行优先: [shape[1], 1]
            strides = [1]
            for i in range(len(shape) - 1, 0, -1):
                strides.insert(0, strides[0] * shape[i])
            return tuple(strides)
        elif layout_type == LayoutType.COL_MAJOR:
            # 列优先: [1, shape[0]]
            strides = [1]
            for i in range(0, len(shape) - 1):
                strides.append(strides[-1] * shape[i])
            return tuple(strides)
        else:
            # 对于其他布局类型，暂时返回行优先步长
            return self._default_strides(shape, LayoutType.ROW_MAJOR)
    
    def to_mlx_layout(self):
        """转换为MLX支持的布局"""
        # MLX当前主要支持行优先布局，需要通过形状变换来处理不同布局
        return {"strides": self.strides, "shape": self.shape}
        
    def __str__(self):
        """用于调试的字符串表示"""
        layout_str = f"MemoryLayout(shape={self.shape}, type={self.layout_type}, strides={self.strides}"
        if self.block_shape:
            layout_str += f", block_shape={self.block_shape}"
        if self.block_strides:
            layout_str += f", block_strides={self.block_strides}"
        layout_str += ")"
        return layout_str

class LayoutAdapter:
    """内存布局适配器"""
    
    def __init__(self):
        self.mx = _get_mlx()
        
    def adapt_tensor(self, tensor, src_layout, dst_layout=None):
        """将张量从源布局适配到目标布局"""
        # 如果目标布局未指定，默认为行优先
        if dst_layout is None:
            dst_layout = MemoryLayout(src_layout.shape, LayoutType.ROW_MAJOR)
            
        # 如果源布局和目标布局相同，无需转换
        if src_layout.layout_type == dst_layout.layout_type and src_layout.strides == dst_layout.strides:
            return tensor
            
        # 执行适当的布局转换
        if src_layout.layout_type == LayoutType.ROW_MAJOR and dst_layout.layout_type == LayoutType.COL_MAJOR:
            # 行优先到列优先：转置
            return self._transpose_tensor(tensor, len(src_layout.shape))
        elif src_layout.layout_type == LayoutType.COL_MAJOR and dst_layout.layout_type == LayoutType.ROW_MAJOR:
            # 列优先到行优先：转置
            return self._transpose_tensor(tensor, len(src_layout.shape))
        elif src_layout.layout_type == LayoutType.BLOCK:
            # 分块布局转换
            return self._adapt_block_layout(tensor, src_layout, dst_layout)
        elif src_layout.layout_type == LayoutType.SLICED:
            # 切片布局转换
            return self._adapt_sliced_layout(tensor, src_layout, dst_layout)
        elif src_layout.layout_type == LayoutType.STRIDED:
            # 步长布局转换
            return self._adapt_strided_layout(tensor, src_layout, dst_layout)
        else:
            # 默认情况：尝试根据步长进行reshape
            return self._adapt_by_strides(tensor, src_layout, dst_layout)
    
    def _transpose_tensor(self, tensor, ndim):
        """转置张量维度"""
        # 对于多维张量，需要指定转置轴
        if ndim <= 1:
            return tensor  # 无需转置标量或向量
        elif ndim == 2:
            return self.mx.transpose(tensor)  # 二维矩阵简单转置
        else:
            # 多维张量反转轴顺序
            perm = list(range(ndim))
            perm.reverse()
            return self.mx.transpose(tensor, perm)
    
    def _adapt_block_layout(self, tensor, src_layout, dst_layout):
        """处理分块布局转换"""
        # 分块布局通常用于矩阵乘法等操作
        # 基本思路是先重新形状化为块状，再重新排列
        
        if not src_layout.block_shape:
            raise ValueError("源布局缺少block_shape信息")
            
        # 将张量reshape为块状
        block_shape = src_layout.block_shape
        tensor_shape = src_layout.shape
        
        # 计算每个维度上的块数
        blocks_per_dim = [ts // bs for ts, bs in zip(tensor_shape, block_shape)]
        
        # Reshape为多维块状
        reshape_dims = []
        for i in range(len(tensor_shape)):
            reshape_dims.extend([blocks_per_dim[i], block_shape[i]])
        
        # 将tensor重新形状化为块状表示
        tensor = self.mx.reshape(tensor, reshape_dims)
        
        # 调整轴顺序以将块内元素放在一起
        ndim = len(reshape_dims)
        half_ndim = ndim // 2
        perm = []
        for i in range(half_ndim):
            perm.append(i)
            perm.append(i + half_ndim)
        
        # 转置到块状排列
        tensor = self.mx.transpose(tensor, perm)
        
        # 最后reshape回原始形状
        return self.mx.reshape(tensor, dst_layout.shape)
    
    def _adapt_sliced_layout(self, tensor, src_layout, dst_layout):
        """处理切片布局转换"""
        # 切片布局通常用于切分大矩阵
        # 由于MLX中没有直接的切片布局表示，我们使用reshape和transpose模拟
        
        # 对于简单情况，我们可以直接使用reshape
        return self.mx.reshape(tensor, dst_layout.shape)
    
    def _adapt_strided_layout(self, tensor, src_layout, dst_layout):
        """处理步长布局转换"""
        # MLX目前没有直接支持自定义步长的API
        # 我们使用slice和concat操作来模拟所需的步长访问
        
        # 对于简单步长模式，我们可以使用reshape和transpose
        if len(src_layout.shape) == 2:
            # 对于2D张量，我们可以尝试使用转置
            if src_layout.strides[0] < src_layout.strides[1]:
                # 行优先
                return tensor
            else:
                # 列优先
                return self.mx.transpose(tensor)
        
        # 对于更复杂的步长，目前简单返回原始张量
        # 这在实际应用中可能需要更复杂的处理
        return tensor
    
    def _adapt_by_strides(self, tensor, src_layout, dst_layout):
        """通过步长信息适配布局"""
        # 根据源布局和目标布局的步长差异进行适配
        
        # 如果步长相同，直接返回
        if src_layout.strides == dst_layout.strides:
            return tensor
            
        # 尝试判断主要排列顺序
        src_is_row_major = self._is_row_major(src_layout.strides)
        dst_is_row_major = self._is_row_major(dst_layout.strides)
        
        if src_is_row_major != dst_is_row_major:
            # 行列主序不同，需要转置
            return self._transpose_tensor(tensor, len(src_layout.shape))
            
        # 对于其他复杂情况，我们目前简单地reshape
        return self.mx.reshape(tensor, dst_layout.shape)
    
    def _is_row_major(self, strides):
        """判断步长是否对应行优先布局"""
        if len(strides) <= 1:
            return True
        # 对于行优先布局，最后一个维度的步长最小
        return strides[-1] < strides[0]

def detect_layout(tensor, metadata=None):
    """检测张量的内存布局"""
    mx = _get_mlx()
    
    # 尝试从元数据中获取布局信息
    if metadata and "layout" in metadata:
        layout_info = metadata["layout"]
        layout_type = layout_info.get("type", LayoutType.ROW_MAJOR)
        strides = layout_info.get("strides")
        block_shape = layout_info.get("block_shape")
        block_strides = layout_info.get("block_strides")
        return MemoryLayout(tensor.shape, layout_type, strides, block_shape, block_strides)
    
    # 如果没有元数据，尝试从张量属性推断
    if hasattr(tensor, "strides") and tensor.strides:
        # 判断是行优先还是列优先
        if len(tensor.shape) > 1 and tensor.strides[-1] < tensor.strides[0]:
            return MemoryLayout(tensor.shape, LayoutType.ROW_MAJOR, tensor.strides)
        else:
            return MemoryLayout(tensor.shape, LayoutType.COL_MAJOR, tensor.strides)
    
    # 默认假设为行优先
    return MemoryLayout(tensor.shape, LayoutType.ROW_MAJOR)

def adapt_tensor(tensor, src_layout=None, dst_layout=None):
    """将张量从源布局适配到目标布局"""
    adapter = LayoutAdapter()
    
    # 如果源布局未指定，尝试检测
    if src_layout is None:
        src_layout = detect_layout(tensor)
        
    # 如果目标布局未指定，默认为行优先
    if dst_layout is None:
        dst_layout = MemoryLayout(src_layout.shape, LayoutType.ROW_MAJOR)
        
    return adapter.adapt_tensor(tensor, src_layout, dst_layout) 