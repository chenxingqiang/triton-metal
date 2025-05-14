#!/usr/bin/env python
"""
Test Metal Memory Manager

This script tests the Metal memory manager functionality with a focus on
M3-specific optimizations.
"""

import os
import sys
import unittest
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import MLX
try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    print("MLX not found. Please install it with 'pip install mlx'")
    MLX_AVAILABLE = False

# Import our modules
try:
    from python.metal_hardware_optimizer import hardware_capabilities, AppleSiliconGeneration
    from python.metal_memory_manager import (
        metal_memory_manager, MetalMemoryManager,
        MemoryLayout, TensorType, MemoryAccessPattern,
        MetalMemoryStrategy, MatrixMemoryStrategy,
        ConvolutionMemoryStrategy, ElementwiseMemoryStrategy,
        ReductionMemoryStrategy
    )
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you're running this from the metal module root directory")
    sys.exit(1)

# Define dummy hardware capabilities to simulate M3 hardware
class DummyEnum(Enum):
    UNKNOWN = 0
    M1 = 1
    M2 = 2
    M3 = 3

class DummyCapabilities:
    def __init__(self, gen=DummyEnum.M3):
        self.chip_generation = gen
        self.shared_memory_size = 65536 if gen == DummyEnum.M3 else 32768

# Create patch for hardware_capabilities
mock_hardware_capabilities = DummyCapabilities(DummyEnum.M3)
mock_m3_generation = DummyEnum.M3

# Mock the AppleSiliconGeneration and hardware_capabilities
sys.modules['metal_hardware_optimizer'] = MagicMock()
sys.modules['metal_hardware_optimizer'].AppleSiliconGeneration = DummyEnum
sys.modules['metal_hardware_optimizer'].hardware_capabilities = mock_hardware_capabilities

# Import modules to test after patching
from metal_memory_manager import MetalMemoryManager, TensorType, MemoryLayout, get_metal_memory_manager

class TestMetalMemoryManager(unittest.TestCase):
    """Test the Metal Memory Manager"""
    
    def setUp(self):
        """Set up test case"""
        # Check if we're running on Apple Silicon
        if hardware_capabilities.chip_generation == AppleSiliconGeneration.UNKNOWN:
            self.skipTest("Not running on Apple Silicon")
        
        # Record the chip generation for test adjustments
        self.chip_generation = hardware_capabilities.chip_generation
        
        # Create a test memory manager
        self.memory_manager = MetalMemoryManager(hardware_capabilities)
        
        # Create test tensors
        self.matrix_tensor = {
            "tensor_type": TensorType.MATRIX,
            "shape": [128, 128]
        }
        
        self.small_matrix_tensor = {
            "tensor_type": TensorType.MATRIX,
            "shape": [32, 32]
        }
        
        self.filter_tensor = {
            "tensor_type": TensorType.FILTER,
            "shape": [3, 3, 64, 64],
            "is_filter": True
        }
        
        self.activation_tensor = {
            "tensor_type": TensorType.FEATURE_MAP,
            "shape": [1, 224, 224, 64],
            "is_filter": False
        }
        
        self.vector_tensor = {
            "tensor_type": TensorType.GENERAL,
            "shape": [1024]
        }
        
        # Create test operations
        self.matmul_op = {
            "id": "op1",
            "type": "tt.matmul",
            "a_id": "a",
            "b_id": "b",
            "a_shape": [128, 64],
            "b_shape": [64, 128],
            "output_shape": [128, 128]
        }
        
        self.conv_op = {
            "id": "op2",
            "type": "tt.conv",
            "input_id": "input",
            "filter_id": "filter",
            "input_shape": [1, 224, 224, 64],
            "filter_shape": [3, 3, 64, 128],
            "output_shape": [1, 224, 224, 128],
            "stride": [1, 1],
            "padding": [1, 1]
        }
        
        self.relu_op = {
            "id": "op3",
            "type": "tt.unary.relu",
            "operand_id": "input",
            "operand_shape": [128, 128],
            "output_shape": [128, 128]
        }
        
        self.reduce_op = {
            "id": "op4",
            "type": "tt.reduce",
            "operand_id": "input",
            "operand_shape": [128, 128],
            "output_shape": [128, 1],
            "dims": [1]  # reduce along axis 1
        }
        
        # Create test graph
        self.test_graph = {
            "ops": [
                self.matmul_op,
                self.conv_op,
                self.relu_op,
                self.reduce_op
            ],
            "inputs": ["a", "b", "input", "filter"],
            "outputs": ["op1", "op2", "op3", "op4"]
        }
    
    def test_matrix_memory_strategy(self):
        """Test matrix memory strategy"""
        strategy = MatrixMemoryStrategy()
        
        # Apply strategy to matrix tensor
        optimized_tensor = strategy.apply(self.matrix_tensor)
        
        # Check that the layout was set
        self.assertIn("layout", optimized_tensor)
        
        # Check hardware-specific optimizations
        if self.chip_generation == AppleSiliconGeneration.M3:
            # M3 should use block layout for large matrices
            self.assertEqual(optimized_tensor["layout"], MemoryLayout.BLOCK_ROW_MAJOR)
            self.assertEqual(optimized_tensor["block_size"], 128)
            self.assertTrue(optimized_tensor["use_tensor_cores"])
        
        # Apply strategy to small matrix tensor
        optimized_small = strategy.apply(self.small_matrix_tensor)
        
        # Small matrices should use different layouts
        self.assertIn("layout", optimized_small)
        
        # Print optimization details
        print(f"\nMatrix optimization for {self.matrix_tensor['shape']}:")
        for key, value in optimized_tensor.items():
            if key not in self.matrix_tensor:
                print(f"  + {key}: {value}")
                
        print(f"\nMatrix optimization for {self.small_matrix_tensor['shape']}:")
        for key, value in optimized_small.items():
            if key not in self.small_matrix_tensor:
                print(f"  + {key}: {value}")
    
    def test_convolution_memory_strategy(self):
        """Test convolution memory strategy"""
        strategy = ConvolutionMemoryStrategy()
        
        # Apply strategy to filter tensor
        optimized_filter = strategy.apply(self.filter_tensor)
        
        # Check that the layout was set
        self.assertIn("layout", optimized_filter)
        
        # Apply strategy to activation tensor
        optimized_activation = strategy.apply(self.activation_tensor)
        
        # Check that the layout was set
        self.assertIn("layout", optimized_activation)
        
        # Check hardware-specific optimizations
        if self.chip_generation == AppleSiliconGeneration.M3:
            # M3 should use texture optimized layout
            self.assertEqual(optimized_filter["layout"], MemoryLayout.TEXTURE_OPTIMIZED)
            self.assertEqual(optimized_filter["tile_size"], 7 if self.filter_tensor["shape"][0] >= 7 else self.filter_tensor["shape"][0])
            self.assertTrue(optimized_filter["use_texture_memory"])
            
            self.assertEqual(optimized_activation["layout"], MemoryLayout.TEXTURE_OPTIMIZED)
            self.assertEqual(optimized_activation["tile_size"], 32)
            self.assertTrue(optimized_activation["interleave_channels"])
        
        # Print optimization details
        print(f"\nFilter optimization for {self.filter_tensor['shape']}:")
        for key, value in optimized_filter.items():
            if key not in self.filter_tensor:
                print(f"  + {key}: {value}")
                
        print(f"\nActivation optimization for {self.activation_tensor['shape']}:")
        for key, value in optimized_activation.items():
            if key not in self.activation_tensor:
                print(f"  + {key}: {value}")
    
    def test_elementwise_memory_strategy(self):
        """Test element-wise memory strategy"""
        strategy = ElementwiseMemoryStrategy()
        
        # Apply strategy to vector tensor
        optimized_tensor = strategy.apply(self.vector_tensor)
        
        # Check that the layout was set
        self.assertIn("layout", optimized_tensor)
        
        # Check hardware-specific optimizations
        if self.chip_generation == AppleSiliconGeneration.M3:
            # M3 should use wider vectorization
            self.assertEqual(optimized_tensor["vectorize_width"], 8)
            self.assertEqual(optimized_tensor["unroll_factor"], 4)
            self.assertTrue(optimized_tensor["use_simdgroup"])
        
        # Print optimization details
        print(f"\nElement-wise optimization for {self.vector_tensor['shape']}:")
        for key, value in optimized_tensor.items():
            if key not in self.vector_tensor:
                print(f"  + {key}: {value}")
    
    def test_reduction_memory_strategy(self):
        """Test reduction memory strategy"""
        strategy = ReductionMemoryStrategy()
        
        # Create a test tensor for reduction
        reduction_tensor = {
            "tensor_type": TensorType.GENERAL,
            "shape": [128, 1024],
            "reduction_axis": 1
        }
        
        # Apply strategy to reduction tensor
        optimized_tensor = strategy.apply(reduction_tensor)
        
        # Check that the strategy was set
        self.assertIn("reduction_strategy", optimized_tensor)
        
        # Check hardware-specific optimizations
        if self.chip_generation == AppleSiliconGeneration.M3:
            # M3 should use hierarchical reduction
            self.assertEqual(optimized_tensor["reduction_strategy"], "hierarchical")
            self.assertTrue(optimized_tensor["use_simdgroup_reduction"])
            self.assertEqual(optimized_tensor["tile_size"], 256)
        
        # Print optimization details
        print(f"\nReduction optimization for {reduction_tensor['shape']} along axis {reduction_tensor['reduction_axis']}:")
        for key, value in optimized_tensor.items():
            if key not in reduction_tensor:
                print(f"  + {key}: {value}")
    
    def test_optimize_tensor_layout(self):
        """Test tensor layout optimization"""
        # Test matrix tensor optimization
        optimized_matrix = self.memory_manager.optimize_tensor_layout(self.matrix_tensor)
        
        # Check that the layout was set
        self.assertIn("layout", optimized_matrix)
        
        # Print the result
        print(f"\nTensor layout optimization for matrix:")
        for key, value in optimized_matrix.items():
            if key not in self.matrix_tensor:
                print(f"  + {key}: {value}")
    
    def test_optimize_operation_memory(self):
        """Test operation memory optimization"""
        # Test matmul operation optimization
        optimized_matmul = self.memory_manager.optimize_operation_memory(self.matmul_op)
        
        # Check that tensors were added
        self.assertIn("tensors", optimized_matmul)
        self.assertIn("a", optimized_matmul["tensors"])
        self.assertIn("b", optimized_matmul["tensors"])
        self.assertIn("output", optimized_matmul["tensors"])
        
        # Test conv operation optimization
        optimized_conv = self.memory_manager.optimize_operation_memory(self.conv_op)
        
        # Check that tensors were added
        self.assertIn("tensors", optimized_conv)
        self.assertIn("input", optimized_conv["tensors"])
        self.assertIn("filter", optimized_conv["tensors"])
        self.assertIn("output", optimized_conv["tensors"])
        
        # Print some details
        print(f"\nOptimized MatMul Operation:")
        print(f"  Input A: {optimized_matmul['tensors']['a'].get('layout')}")
        print(f"  Input B: {optimized_matmul['tensors']['b'].get('layout')}")
        print(f"  Output: {optimized_matmul['tensors']['output'].get('layout')}")
        
        print(f"\nOptimized Conv Operation:")
        print(f"  Input: {optimized_conv['tensors']['input'].get('layout')}")
        print(f"  Filter: {optimized_conv['tensors']['filter'].get('layout')}")
        print(f"  Output: {optimized_conv['tensors']['output'].get('layout')}")
    
    def test_optimize_graph_memory(self):
        """Test graph memory optimization"""
        # Optimize the test graph
        optimized_graph = self.memory_manager.optimize_graph_memory(self.test_graph)
        
        # Check that metadata was added
        self.assertIn("metadata", optimized_graph)
        
        # Check that operations were optimized
        for op in optimized_graph["ops"]:
            self.assertIn("tensors", op)
        
        # Check hardware-specific metadata
        if self.chip_generation == AppleSiliconGeneration.M3:
            self.assertEqual(optimized_graph["metadata"]["memory_optimized_for"], "M3")
            self.assertEqual(optimized_graph["metadata"]["shared_memory_size"], 65536)
        
        # Print metadata
        print(f"\nOptimized Graph Metadata:")
        for key, value in optimized_graph["metadata"].items():
            print(f"  {key}: {value}")

    def test_hardware_detection(self):
        """Test hardware detection"""
        self.assertTrue(self.memory_manager._is_m3_hardware())
        self.assertEqual(self.memory_manager.shared_memory_size, 65536)  # 64KB for M3
        self.assertEqual(self.memory_manager.vector_width, 8)  # 8-wide vectorization for M3
    
    def test_tile_sizes(self):
        """Test tile size configuration"""
        # Check M3-specific tile sizes
        self.assertEqual(self.memory_manager.tile_sizes[TensorType.MATRIX]["default"], (128, 128))
        self.assertEqual(self.memory_manager.tile_sizes[TensorType.MATRIX]["large"], (128, 256))
        
        # Test with M1/M2 hardware
        with patch.object(self.memory_manager, 'is_m3', False):
            self.memory_manager.shared_memory_size = 32768
            tile_sizes = self.memory_manager._configure_tile_sizes()
            self.assertEqual(tile_sizes[TensorType.MATRIX]["default"], (64, 64))
    
    def test_optimize_matmul_memory(self):
        """Test matmul memory optimization"""
        optimized_op = self.memory_manager._optimize_matmul_memory(self.matmul_op.copy())
        
        # Check M3-specific optimizations
        self.assertIn("execution_parameters", optimized_op)
        exec_params = optimized_op["execution_parameters"]
        
        # Check tile sizes
        self.assertEqual(exec_params["tile_m"], 128)
        self.assertEqual(exec_params["tile_n"], 128)
        
        # Check tensor core usage
        self.assertTrue(exec_params["use_tensor_cores"])
        
        # Check M3-specific parameters
        self.assertTrue(exec_params["use_hierarchical_reduction"])
        self.assertTrue(exec_params["use_dynamic_shared_memory"])
        self.assertEqual(exec_params["simdgroup_matrix_size"], 16)
        self.assertEqual(exec_params["vector_width"], 8)
    
    def test_optimize_convolution_memory(self):
        """Test convolution memory optimization"""
        optimized_op = self.memory_manager._optimize_convolution_memory(self.conv_op.copy())
        
        # Check M3-specific optimizations
        self.assertIn("execution_parameters", optimized_op)
        exec_params = optimized_op["execution_parameters"]
        
        # Check memory layout
        self.assertEqual(exec_params["memory_layout"], MemoryLayout.TEXTURE_OPTIMIZED.value)
        
        # Check M3-specific parameters
        self.assertTrue(exec_params["use_texture_for_weights"])
        self.assertTrue(exec_params["use_warp_specialization"])
        self.assertEqual(exec_params["prefetch_mode"], "double_buffer")
        self.assertTrue(exec_params["use_simdgroup_reduction"])
        self.assertEqual(exec_params["vector_width"], 8)
    
    def test_optimize_reduction_memory(self):
        """Test reduction memory optimization"""
        optimized_op = self.memory_manager._optimize_reduction_memory(self.reduce_op.copy())
        
        # Check M3-specific optimizations
        self.assertIn("execution_parameters", optimized_op)
        exec_params = optimized_op["execution_parameters"]
        
        # Check hierarchical reduction
        self.assertTrue(exec_params["use_hierarchical_reduction"])
        
        # Check M3-specific parameters
        self.assertTrue(exec_params["two_stage_reduction"])
        self.assertTrue(exec_params["use_simdgroup_reduction"])
        self.assertEqual(exec_params["vector_width"], 8)
    
    def test_optimize_transpose_memory(self):
        """Test transpose memory optimization"""
        optimized_op = self.memory_manager._optimize_transpose_memory(self.relu_op.copy())
        
        # Check M3-specific optimizations
        self.assertIn("execution_parameters", optimized_op)
        exec_params = optimized_op["execution_parameters"]
        
        # Check memory layout
        self.assertEqual(exec_params["memory_layout"], MemoryLayout.TILED.value)
        
        # Check M3-specific parameters
        self.assertEqual(exec_params["tile_size"], 32)
        self.assertTrue(exec_params["use_simdgroup_matrix"])
        self.assertEqual(exec_params["vector_width"], 8)
    
    def test_optimize_attention_memory(self):
        """Test attention memory optimization"""
        optimized_op = self.memory_manager._optimize_attention_memory(self.activation_tensor.copy())
        
        # Check M3-specific optimizations
        self.assertIn("execution_parameters", optimized_op)
        exec_params = optimized_op["execution_parameters"]
        
        # Check tile size
        self.assertIn("tile_size", exec_params)
        
        # Check M3-specific parameters
        self.assertTrue(exec_params["use_flash_attention"])
        self.assertTrue(exec_params["use_tensor_cores"])
        self.assertEqual(exec_params["block_size"], 128)
        self.assertTrue(exec_params["use_causal_mask_optimization"])
    
    def test_optimize_elementwise_memory(self):
        """Test elementwise memory optimization"""
        optimized_op = self.memory_manager._optimize_elementwise_memory(self.vector_tensor.copy())
        
        # Check M3-specific optimizations
        self.assertIn("execution_parameters", optimized_op)
        exec_params = optimized_op["execution_parameters"]
        
        # Check M3-specific parameters
        self.assertEqual(exec_params["vector_width"], 8)
        self.assertEqual(exec_params["unroll_factor"], 4)
    
    def test_general_memory_optimizations(self):
        """Test general memory optimizations"""
        # Create generic operation
        generic_op = {
            "id": 7,
            "type": "custom_op",
            "inputs": [12],
            "input_shapes": [[512, 512]],
            "output_shape": [512, 512]
        }
        
        optimized_op = self.memory_manager._apply_general_memory_optimizations(generic_op)
        
        # Check M3-specific optimizations
        self.assertIn("execution_parameters", optimized_op)
        exec_params = optimized_op["execution_parameters"]
        
        # Check M3-specific parameters
        self.assertEqual(exec_params["vector_width"], 8)
        self.assertEqual(exec_params["shared_memory_size"], 65536)
        self.assertEqual(exec_params["memory_optimization_level"], 2)
        self.assertTrue(exec_params["use_dynamic_shared_memory"])
    
    def test_singleton_instance(self):
        """Test singleton instance creation"""
        # Get memory manager instance
        memory_manager1 = get_metal_memory_manager()
        memory_manager2 = get_metal_memory_manager()
        
        # Check that both instances are the same
        self.assertIs(memory_manager1, memory_manager2)
    
    def test_m1_vs_m3_optimizations(self):
        """Test differences between M1 and M3 optimizations"""
        # Create M1 hardware capabilities
        m1_capabilities = DummyCapabilities(DummyEnum.M1)
        
        # Patch hardware detection to simulate M1
        with patch.object(self.memory_manager, 'is_m3', False), \
             patch.object(self.memory_manager, 'shared_memory_size', 32768), \
             patch.object(self.memory_manager, 'vector_width', 4):
            
            # Reconfigure tile sizes
            self.memory_manager.tile_sizes = self.memory_manager._configure_tile_sizes()
            
            # Optimize matmul operation for M1
            m1_optimized_op = self.memory_manager._optimize_matmul_memory(self.matmul_op.copy())
            
            # Check M1 optimizations
            self.assertIn("execution_parameters", m1_optimized_op)
            m1_exec_params = m1_optimized_op["execution_parameters"]
            
            # Check tile sizes (should be smaller for M1)
            self.assertEqual(m1_exec_params["tile_m"], 64)
            self.assertEqual(m1_exec_params["tile_n"], 64)
            
            # Check vector width (should be 4 for M1)
            self.assertEqual(m1_exec_params["vector_width"], 4)
            
            # Check shared memory size (should be 32KB for M1)
            self.assertEqual(m1_exec_params["shared_memory_size"], 32768)
            
            # Check for M3-specific parameters that should not be present in M1
            self.assertNotIn("use_hierarchical_reduction", m1_exec_params)
            self.assertNotIn("use_dynamic_shared_memory", m1_exec_params)
        
        # Now optimize the same operation for M3
        m3_optimized_op = self.memory_manager._optimize_matmul_memory(self.matmul_op.copy())
        
        # Check M3 optimizations
        self.assertIn("execution_parameters", m3_optimized_op)
        m3_exec_params = m3_optimized_op["execution_parameters"]
        
        # Compare M1 vs M3 tile sizes
        self.assertNotEqual(m1_exec_params["tile_m"], m3_exec_params["tile_m"])
        self.assertNotEqual(m1_exec_params["tile_n"], m3_exec_params["tile_n"])
        
        # Compare vector width
        self.assertNotEqual(m1_exec_params["vector_width"], m3_exec_params["vector_width"])
        
        # Compare shared memory size
        self.assertNotEqual(m1_exec_params["shared_memory_size"], m3_exec_params["shared_memory_size"])

def main():
    """Run tests"""
    # Print some information
    print("Testing Metal Memory Manager")
    print(f"MLX Available: {MLX_AVAILABLE}")
    print(f"Hardware: {hardware_capabilities.chip_generation.name}")
    
    # Run tests
    unittest.main()

if __name__ == "__main__":
    main() 