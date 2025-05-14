"""
Metal Hardware Optimizer for Triton on Apple Silicon

This module provides hardware detection and optimization for Apple Silicon GPUs,
including M1, M2, and M3 chips.
"""

import sys
import platform
import re
import os
import subprocess
import ctypes
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union

class AppleSiliconGeneration(Enum):
    """Enum for Apple Silicon generations"""
    UNKNOWN = 0
    M1 = 1
    M2 = 2
    M3 = 3

class MetalFeatureSet(Enum):
    """Enum for Metal feature sets"""
    UNKNOWN = 0
    METAL_3_0 = 1
    METAL_3_1 = 2
    METAL_3_2 = 3

class HardwareCapabilities:
    """Detection and representation of Apple Silicon hardware capabilities"""
    
    def __init__(self):
        """Initialize hardware capabilities detector"""
        self.chip_generation = AppleSiliconGeneration.UNKNOWN
        self.feature_set = MetalFeatureSet.UNKNOWN
        self.gpu_family = "unknown"
        self.num_cores = 0
        self.simd_width = 0
        self.max_threads_per_threadgroup = 0
        self.max_threadgroups_per_grid = 0
        self.shared_memory_size = 0
        self.initialize()
    
    def initialize(self):
        """Detect hardware capabilities"""
        # Only run on macOS
        if sys.platform != "darwin":
            return
        
        # Detect Apple Silicon
        if not self._is_apple_silicon():
            return
        
        # Detect chip generation
        self._detect_chip_generation()
        
        # Detect Metal capabilities
        self._detect_metal_capabilities()
        
        # Set hardware-specific parameters
        self._set_hardware_parameters()
    
    def _is_apple_silicon(self) -> bool:
        """
        Check if we're running on Apple Silicon
        
        Returns:
            True if running on Apple Silicon, False otherwise
        """
        try:
            arch = platform.machine()
            return arch == "arm64"
        except:
            return False
    
    def _detect_chip_generation(self):
        """Detect the Apple Silicon chip generation"""
        try:
            # Run sysctl to get CPU info
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True
            )
            
            cpu_info = result.stdout.strip()
            
            # Match the chip family
            if "M1" in cpu_info:
                self.chip_generation = AppleSiliconGeneration.M1
            elif "M2" in cpu_info:
                self.chip_generation = AppleSiliconGeneration.M2
            elif "M3" in cpu_info:
                self.chip_generation = AppleSiliconGeneration.M3
            
            # Parse number of cores
            cores_result = subprocess.run(
                ["sysctl", "-n", "hw.ncpu"],
                capture_output=True,
                text=True
            )
            
            if cores_result.stdout.strip():
                self.num_cores = int(cores_result.stdout.strip())
        
        except Exception as e:
            print(f"Error detecting chip generation: {e}")
    
    def _detect_metal_capabilities(self):
        """Detect Metal capabilities"""
        try:
            # Check for Metal capability programmatically
            # For a proper implementation, we would use PyObjC or a native extension
            # to call Metal APIs directly. For now, we'll infer from the chip generation.
            
            if self.chip_generation == AppleSiliconGeneration.M1:
                self.feature_set = MetalFeatureSet.METAL_3_0
                self.gpu_family = "apple7"
                self.simd_width = 32
                self.max_threads_per_threadgroup = 1024
                self.max_threadgroups_per_grid = 65535
                self.shared_memory_size = 32768  # 32 KB
            
            elif self.chip_generation == AppleSiliconGeneration.M2:
                self.feature_set = MetalFeatureSet.METAL_3_1
                self.gpu_family = "apple8"
                self.simd_width = 32
                self.max_threads_per_threadgroup = 1024
                self.max_threadgroups_per_grid = 65535
                self.shared_memory_size = 32768  # 32 KB
            
            elif self.chip_generation == AppleSiliconGeneration.M3:
                self.feature_set = MetalFeatureSet.METAL_3_2
                self.gpu_family = "apple9"
                self.simd_width = 32
                self.max_threads_per_threadgroup = 1024
                self.max_threadgroups_per_grid = 65535
                self.shared_memory_size = 65536  # 64 KB - increased for M3
        
        except Exception as e:
            print(f"Error detecting Metal capabilities: {e}")
    
    def _set_hardware_parameters(self):
        """Set hardware-specific parameters for optimization"""
        # Set parameters based on detected hardware
        pass
    
    def get_recommended_block_size(self) -> int:
        """
        Get recommended block size for the hardware
        
        Returns:
            Recommended block size
        """
        if self.chip_generation == AppleSiliconGeneration.M3:
            return 256
        elif self.chip_generation == AppleSiliconGeneration.M2:
            return 128
        elif self.chip_generation == AppleSiliconGeneration.M1:
            return 128
        else:
            return 64
    
    def get_recommended_warps(self) -> int:
        """
        Get recommended number of warps (threadgroups)
        
        Returns:
            Recommended number of warps
        """
        if self.chip_generation == AppleSiliconGeneration.M3:
            return 8
        elif self.chip_generation == AppleSiliconGeneration.M2:
            return 4
        elif self.chip_generation == AppleSiliconGeneration.M1:
            return 4
        else:
            return 4
    
    def get_recommended_shared_memory(self) -> int:
        """
        Get recommended shared memory size
        
        Returns:
            Recommended shared memory size
        """
        return self.shared_memory_size
    
    def get_optimized_matmul_tile_sizes(self, m: int, n: int, k: int) -> Tuple[int, int, int]:
        """
        Get optimized tile sizes for matrix multiplication
        
        Args:
            m: First matrix dimension
            n: Second matrix dimension
            k: Common dimension
            
        Returns:
            Tuple of (tile_m, tile_n, tile_k)
        """
        if self.chip_generation == AppleSiliconGeneration.M3:
            # M3-specific tiles
            if m >= 1024 and n >= 1024:
                return (128, 128, 32)
            elif m >= 512 or n >= 512:
                return (64, 64, 32)
            else:
                return (32, 32, 16)
        
        elif self.chip_generation == AppleSiliconGeneration.M2:
            # M2-specific tiles
            if m >= 1024 and n >= 1024:
                return (64, 64, 32)
            else:
                return (32, 32, 16)
        
        elif self.chip_generation == AppleSiliconGeneration.M1:
            # M1-specific tiles
            return (32, 32, 16)
        
        else:
            # Default tiles
            return (16, 16, 8)
    
    def get_auto_tuner_constraints(self, operation_type: str) -> Dict[str, Any]:
        """
        Get hardware-specific constraints for auto-tuning
        
        Args:
            operation_type: Type of operation ('matmul', 'conv', 'general')
            
        Returns:
            Dictionary of constraints for auto-tuning
        """
        constraints = {}
        
        # Common constraints for all operation types
        constraints['max_threads_per_threadgroup'] = self.max_threads_per_threadgroup
        constraints['max_threadgroups_per_grid'] = self.max_threadgroups_per_grid
        constraints['shared_memory_size'] = self.shared_memory_size
        
        # Hardware-specific constraints
        if self.chip_generation == AppleSiliconGeneration.M3:
            # M3-specific constraints
            constraints['simd_width'] = 32
            constraints['max_occupancy'] = 64  # Maximum warps per SM
            constraints['preferred_vector_width'] = 4
            constraints['uses_simdgroup_matrix'] = True
            constraints['optimal_thread_count'] = 512 if operation_type == 'matmul' else 256
            
            # Set operation-specific constraints
            if operation_type == 'matmul':
                constraints['recommended_block_sizes'] = [(128, 128, 32), (64, 64, 32), (128, 64, 32)]
                constraints['recommended_num_warps'] = [8, 6, 4]
                constraints['recommended_num_stages'] = [3, 2]
                constraints['use_mma'] = True
            elif operation_type == 'conv':
                constraints['recommended_block_sizes'] = [(16, 16, 4, 32), (32, 8, 4, 32)]
                constraints['filter_tile_sizes'] = [3, 5, 7]
                constraints['recommended_num_warps'] = [4, 8]
            
        elif self.chip_generation == AppleSiliconGeneration.M2:
            # M2-specific constraints
            constraints['simd_width'] = 32
            constraints['max_occupancy'] = 48  # Maximum warps per SM
            constraints['preferred_vector_width'] = 4
            constraints['uses_simdgroup_matrix'] = True
            constraints['optimal_thread_count'] = 256
            
            # Set operation-specific constraints
            if operation_type == 'matmul':
                constraints['recommended_block_sizes'] = [(64, 64, 32), (32, 128, 16), (128, 32, 16)]
                constraints['recommended_num_warps'] = [4, 6, 8]
                constraints['recommended_num_stages'] = [2, 3]
                constraints['use_mma'] = True
            elif operation_type == 'conv':
                constraints['recommended_block_sizes'] = [(16, 16, 4, 32), (32, 8, 4, 16)]
                constraints['filter_tile_sizes'] = [3, 5]
                constraints['recommended_num_warps'] = [4, 6]
                
        elif self.chip_generation == AppleSiliconGeneration.M1:
            # M1-specific constraints
            constraints['simd_width'] = 32
            constraints['max_occupancy'] = 32  # Maximum warps per SM
            constraints['preferred_vector_width'] = 4
            constraints['uses_simdgroup_matrix'] = False
            constraints['optimal_thread_count'] = 256
            
            # Set operation-specific constraints
            if operation_type == 'matmul':
                constraints['recommended_block_sizes'] = [(32, 32, 16), (64, 32, 16), (32, 64, 16)]
                constraints['recommended_num_warps'] = [4, 6]
                constraints['recommended_num_stages'] = [2]
                constraints['use_mma'] = False
            elif operation_type == 'conv':
                constraints['recommended_block_sizes'] = [(16, 16, 4, 16), (16, 8, 4, 16)]
                constraints['filter_tile_sizes'] = [3]
                constraints['recommended_num_warps'] = [4]
        
        else:
            # Default constraints for unknown hardware
            constraints['simd_width'] = 32
            constraints['max_occupancy'] = 32
            constraints['preferred_vector_width'] = 4
            constraints['uses_simdgroup_matrix'] = False
            constraints['optimal_thread_count'] = 256
            
            # Set operation-specific constraints
            if operation_type == 'matmul':
                constraints['recommended_block_sizes'] = [(32, 32, 16)]
                constraints['recommended_num_warps'] = [4]
                constraints['recommended_num_stages'] = [2]
                constraints['use_mma'] = False
            elif operation_type == 'conv':
                constraints['recommended_block_sizes'] = [(16, 16, 4, 16)]
                constraints['filter_tile_sizes'] = [3]
                constraints['recommended_num_warps'] = [4]
        
        return constraints
    
    def optimize_search_space(self, tunable_params: List, operation_type: str) -> List:
        """
        Optimize the search space for auto-tuning based on hardware capabilities
        
        Args:
            tunable_params: List of tunable parameters
            operation_type: Type of operation ('matmul', 'conv', 'general')
            
        Returns:
            Optimized list of tunable parameters
        """
        # Get hardware constraints
        constraints = self.get_auto_tuner_constraints(operation_type)
        
        # Optimize each parameter based on hardware capabilities
        for param in tunable_params:
            if param.name == "num_warps":
                # Adjust num_warps based on hardware
                if self.chip_generation == AppleSiliconGeneration.M3:
                    param.default_value = 8
                    param.max_value = min(param.max_value, 16)
                elif self.chip_generation == AppleSiliconGeneration.M2:
                    param.default_value = 6
                    param.max_value = min(param.max_value, 12)
                else:
                    param.default_value = 4
                    param.max_value = min(param.max_value, 8)
            
            elif param.name == "num_stages":
                # Adjust num_stages based on hardware
                if self.chip_generation == AppleSiliconGeneration.M3:
                    param.default_value = 3
                    param.max_value = min(param.max_value, 5)
                else:
                    param.default_value = 2
                    param.max_value = min(param.max_value, 4)
            
            elif param.name == "use_simdgroup_matrix" and operation_type == "matmul":
                # M1 doesn't support simdgroup matrix operations as efficiently
                param.default_value = (self.chip_generation != AppleSiliconGeneration.M1)
            
            elif param.name in ["block_m", "block_n"] and operation_type == "matmul":
                if self.chip_generation == AppleSiliconGeneration.M3:
                    param.default_value = 128
                    param.max_value = min(param.max_value, 256)
                elif self.chip_generation == AppleSiliconGeneration.M2:
                    param.default_value = 64
                    param.max_value = min(param.max_value, 128)
                else:
                    param.default_value = 32
                    param.max_value = min(param.max_value, 64)
            
            elif param.name == "block_k" and operation_type == "matmul":
                if self.chip_generation == AppleSiliconGeneration.M3:
                    param.default_value = 32
                else:
                    param.default_value = 16
            
            elif param.name in ["block_x", "block_y"] and operation_type == "conv":
                if self.chip_generation == AppleSiliconGeneration.M3:
                    param.default_value = 32
                    param.max_value = min(param.max_value, 64)
                else:
                    param.default_value = 16
                    param.max_value = min(param.max_value, 32)
        
        return tunable_params
    
    def __str__(self) -> str:
        """String representation"""
        return (
            f"Apple Silicon Generation: {self.chip_generation.name}\n"
            f"Metal Feature Set: {self.feature_set.name}\n"
            f"GPU Family: {self.gpu_family}\n"
            f"Number of Cores: {self.num_cores}\n"
            f"SIMD Width: {self.simd_width}\n"
            f"Max Threads per Threadgroup: {self.max_threads_per_threadgroup}\n"
            f"Max Threadgroups per Grid: {self.max_threadgroups_per_grid}\n"
            f"Shared Memory Size: {self.shared_memory_size} bytes"
        )

# Create global instance
hardware_capabilities = HardwareCapabilities() 