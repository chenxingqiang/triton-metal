from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget
import platform
import os

class MetalDriver(DriverBase):
    def __init__(self):
        super().__init__()
        # 延迟导入MLX避免不必要的依赖
        self._mx = None
        self._device = None
        
    @property
    def mx(self):
        """懒加载MLX"""
        if self._mx is None:
            import mlx.core as mx
            self._mx = mx
        return self._mx
        
    @property
    def device(self):
        """获取Metal设备"""
        if self._device is None:
            self._device = self.mx.Device("gpu")
        return self._device
        
    @staticmethod
    def is_active():
        """检测是否在Apple Silicon上运行且MLX可用"""
        try:
            # 检查是否为Apple Silicon
            if platform.processor() != 'arm':
                return False
                
            # 检查macOS版本（需要macOS 13.5+）
            mac_ver = platform.mac_ver()[0]
            if tuple(map(int, mac_ver.split('.'))) < (13, 5):
                return False
                
            # 检查MLX可用性
            import mlx.core as mx
            return mx.metal.is_available()
        except ImportError:
            return False
        except AttributeError:
            return False
            
    def get_current_target(self):
        """获取当前Metal目标设备配置"""
        # 这里我们使用固定值，未来可以通过Metal API获取实际值
        return GPUTarget("metal", "apple-silicon", 32)  # warp_size=32作为初始值
        
    def get_active_torch_device(self):
        """兼容PyTorch MPS设备"""
        try:
            import torch
            if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
        except ImportError:
            pass
        return None
        
    def get_benchmarker(self):
        """返回性能基准函数"""
        from triton.testing import do_bench
        return do_bench
        
    def get_empty_cache_for_benchmark(self):
        """清空缓存用于基准测试"""
        # MLX目前没有显式缓存管理API，但我们可以通过创建临时数组来模拟
        cache_size = 256 * 1024 * 1024  # 256MB
        temp = self.mx.zeros((cache_size // 4,), dtype=self.mx.float32)
        self.mx.eval(temp)  # 强制即时计算
        return temp
        
    def clear_cache(self, cache):
        """清空缓存"""
        # MLX中没有直接等价物，可能需要通过MLX API修改
        self.mx.eval(cache * 0)  # 清零并强制执行 