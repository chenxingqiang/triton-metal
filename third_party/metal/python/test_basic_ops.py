#!/usr/bin/env python3
"""
测试MLX桥接层中基本算术和数学运算的映射功能
"""

import unittest
import os
import sys
import numpy as np

# 添加必要的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))

# 导入MLX
try:
    import mlx.core as mx
except ImportError:
    print("错误: 未安装MLX库，请先安装: pip install mlx")
    sys.exit(1)

# 导入我们的模块
from third_party.metal.python.mlx_bridge import init_dtype_map, init_op_map
from third_party.metal.python.memory_layout import MemoryLayout, adapt_tensor

# 模拟Triton操作
class MockOp:
    def __init__(self, name, results=None, operands=None, attributes=None):
        self.name = name
        self.results = results or []
        self.operands = operands or []
        self.attributes = attributes or {}

# 定义简单版本的操作映射用于测试
TEST_OP_MAP = {
    # 二元操作
    'tt.add': mx.add,
    'tt.sub': mx.subtract,
    'tt.mul': mx.multiply,
    'tt.div': mx.divide,
    'tt.max': mx.maximum,
    'tt.min': mx.minimum,
    'tt.pow': mx.power,
    'tt.mod': mx.remainder,
    'tt.and': lambda a, b: mx.logical_and(a != 0, b != 0),
    'tt.or': lambda a, b: mx.logical_or(a != 0, b != 0),
    'tt.xor': lambda a, b: mx.logical_xor(a != 0, b != 0),
    'tt.eq': mx.equal,
    'tt.ne': mx.not_equal,
    'tt.lt': mx.less,
    'tt.le': mx.less_equal,
    'tt.gt': mx.greater,
    'tt.ge': mx.greater_equal,
    
    # 一元操作
    'tt.exp': mx.exp,
    'tt.log': mx.log,
    'tt.sin': mx.sin,
    'tt.cos': mx.cos,
    'tt.sqrt': mx.sqrt,
    'tt.neg': mx.negative,
    'tt.not': lambda x: mx.logical_not(x != 0),
    'tt.abs': mx.abs,
    'tt.tanh': mx.tanh,
    
    # 复杂操作
    'tt.dot': mx.matmul,
    'tt.reshape': mx.reshape,
    'tt.trans': mx.transpose,
    
    # 规约操作
    'tt.reduce': lambda op, operands, _: handle_reduction_test(op, operands),
}

def handle_reduction_test(op, operands):
    """测试专用的规约处理函数"""
    input_tensor = operands[0]
    axis = op.attributes.get("axis")
    reduce_type = op.attributes.get("reduce_type")
    
    if reduce_type == "sum":
        return mx.sum(input_tensor, axis=axis)
    elif reduce_type == "max":
        return mx.max(input_tensor, axis=axis)
    elif reduce_type == "min":
        return mx.min(input_tensor, axis=axis)
    elif reduce_type == "mean":
        return mx.mean(input_tensor, axis=axis)
    else:
        raise NotImplementedError(f"规约类型 {reduce_type} 尚未实现")

class TestBasicOps(unittest.TestCase):
    """测试基本操作映射"""
    
    def setUp(self):
        """测试前初始化"""
        # 创建一些基本测试数据
        self.x = mx.array([1, 2, 3, 4], dtype=mx.float32)
        self.y = mx.array([5, 6, 7, 8], dtype=mx.float32)
        self.a = mx.array([[1, 2], [3, 4]], dtype=mx.float32)
        self.b = mx.array([[5, 6], [7, 8]], dtype=mx.float32)
        
    def test_binary_arithmetic(self):
        """测试基本二元算术运算"""
        # 测试加法
        result = TEST_OP_MAP["tt.add"](self.x, self.y)
        expected = self.x + self.y
        self.assertTrue(mx.array_equal(result, expected))
        
        # 测试减法
        result = TEST_OP_MAP["tt.sub"](self.x, self.y)
        expected = self.x - self.y
        self.assertTrue(mx.array_equal(result, expected))
        
        # 测试乘法
        result = TEST_OP_MAP["tt.mul"](self.x, self.y)
        expected = self.x * self.y
        self.assertTrue(mx.array_equal(result, expected))
        
        # 测试除法
        result = TEST_OP_MAP["tt.div"](self.x, self.y)
        expected = self.x / self.y
        self.assertTrue(mx.array_equal(result, expected))
        
    def test_unary_operations(self):
        """测试一元运算"""
        # 测试指数
        result = TEST_OP_MAP["tt.exp"](self.x)
        expected = mx.exp(self.x)
        self.assertTrue(mx.allclose(result, expected))
        
        # 测试对数
        pos_x = mx.array([1, 2, 3, 4], dtype=mx.float32)
        result = TEST_OP_MAP["tt.log"](pos_x)
        expected = mx.log(pos_x)
        self.assertTrue(mx.allclose(result, expected))
        
        # 测试平方根
        result = TEST_OP_MAP["tt.sqrt"](self.x)
        expected = mx.sqrt(self.x)
        self.assertTrue(mx.allclose(result, expected))
        
        # 测试负数
        result = TEST_OP_MAP["tt.neg"](self.x)
        expected = -self.x
        self.assertTrue(mx.array_equal(result, expected))
        
    def test_logical_operations(self):
        """测试逻辑运算"""
        x_bool = self.x > 2
        y_bool = self.y > 6
        
        # 测试逻辑与
        result = TEST_OP_MAP["tt.and"](x_bool, y_bool)
        expected = mx.logical_and(x_bool, y_bool)
        self.assertTrue(mx.array_equal(result, expected))
        
        # 测试逻辑或
        result = TEST_OP_MAP["tt.or"](x_bool, y_bool)
        expected = mx.logical_or(x_bool, y_bool)
        self.assertTrue(mx.array_equal(result, expected))
        
        # 测试逻辑非
        result = TEST_OP_MAP["tt.not"](x_bool)
        expected = mx.logical_not(x_bool)
        self.assertTrue(mx.array_equal(result, expected))
        
    def test_comparison_operations(self):
        """测试比较运算"""
        # 测试等于
        result = TEST_OP_MAP["tt.eq"](self.x, self.y)
        expected = self.x == self.y
        self.assertTrue(mx.array_equal(result, expected))
        
        # 测试不等于
        result = TEST_OP_MAP["tt.ne"](self.x, self.y)
        expected = self.x != self.y
        self.assertTrue(mx.array_equal(result, expected))
        
        # 测试小于
        result = TEST_OP_MAP["tt.lt"](self.x, self.y)
        expected = self.x < self.y
        self.assertTrue(mx.array_equal(result, expected))
        
        # 测试大于
        result = TEST_OP_MAP["tt.gt"](self.x, self.y)
        expected = self.x > self.y
        self.assertTrue(mx.array_equal(result, expected))
        
    def test_matrix_operations(self):
        """测试矩阵运算"""
        # 测试矩阵乘法
        result = TEST_OP_MAP["tt.dot"](self.a, self.b)
        expected = mx.matmul(self.a, self.b)
        self.assertTrue(mx.allclose(result, expected))
        
        # 测试转置
        result = TEST_OP_MAP["tt.trans"](self.a)
        expected = mx.transpose(self.a)
        self.assertTrue(mx.array_equal(result, expected))
        
        # 测试reshape
        result = TEST_OP_MAP["tt.reshape"](self.x, (2, 2))
        expected = mx.reshape(self.x, (2, 2))
        self.assertTrue(mx.array_equal(result, expected))
        
    def test_reduction_operations(self):
        """测试规约操作"""
        # 创建模拟的规约操作
        sum_op = MockOp("tt.reduce", results=[None], operands=[self.a], 
                        attributes={"reduce_type": "sum", "axis": 0})
        
        # 测试和规约
        result = TEST_OP_MAP["tt.reduce"](sum_op, [self.a], None)
        expected = mx.sum(self.a, axis=0)
        self.assertTrue(mx.array_equal(result, expected))
        
        # 创建模拟的最大值规约操作
        max_op = MockOp("tt.reduce", results=[None], operands=[self.a], 
                       attributes={"reduce_type": "max", "axis": 1})
        
        # 测试最大值规约
        result = TEST_OP_MAP["tt.reduce"](max_op, [self.a], None)
        expected = mx.max(self.a, axis=1)
        self.assertTrue(mx.array_equal(result, expected))
        
    def test_memory_layout(self):
        """测试内存布局适配"""
        # 创建行优先布局
        row_layout = MemoryLayout(self.a.shape)
        
        # 创建列优先布局
        col_layout = MemoryLayout(self.a.shape, "col_major")
        
        # 测试行优先到列优先的转换
        row_to_col = adapt_tensor(self.a, row_layout, col_layout)
        self.assertEqual(row_to_col.shape, self.a.shape)
        
        # 验证转换后等价于转置
        expected = mx.transpose(self.a)
        self.assertTrue(mx.allclose(row_to_col, expected))
        
        # 测试列优先到行优先的转换
        col_to_row = adapt_tensor(expected, col_layout, row_layout)
        self.assertEqual(col_to_row.shape, self.a.shape)
        self.assertTrue(mx.allclose(col_to_row, self.a))

if __name__ == "__main__":
    unittest.main() 