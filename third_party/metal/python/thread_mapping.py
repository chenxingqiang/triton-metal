"""
Thread mapping and shared memory management for Metal backend

This module provides utilities for mapping Triton thread hierarchy to Metal
and managing shared memory in Metal shaders.
"""

import metal_hardware_optimizer
from typing import Dict, List, Tuple, Any, Optional, Union

class ThreadMapping:
    """Mapping of Triton thread hierarchy to Metal thread hierarchy"""
    
    def __init__(self, hardware_capabilities=None):
        """Initialize thread mapping
        
        Args:
            hardware_capabilities: Optional hardware capabilities instance
        """
        self.hardware = hardware_capabilities or getattr(metal_hardware_optimizer, 'hardware_capabilities', None)
        self.max_threads_per_threadgroup = self._get_max_threads_per_threadgroup()
        self.max_threadgroups = self._get_max_threadgroups()
        self.simd_width = self._get_simd_width()
    
    def _get_max_threads_per_threadgroup(self) -> int:
        """Get maximum threads per threadgroup for the current hardware
        
        Returns:
            Maximum threads per threadgroup
        """
        if self.hardware and hasattr(self.hardware, 'max_threads_per_threadgroup'):
            return self.hardware.max_threads_per_threadgroup
        return 1024  # Default for modern Metal GPUs
    
    def _get_max_threadgroups(self) -> Tuple[int, int, int]:
        """Get maximum threadgroups for the current hardware
        
        Returns:
            Tuple of (max_x, max_y, max_z) threadgroups
        """
        if self.hardware and hasattr(self.hardware, 'max_threadgroups'):
            return self.hardware.max_threadgroups
        return (1024, 1024, 64)  # Default for modern Metal GPUs
    
    def _get_simd_width(self) -> int:
        """Get SIMD width for the current hardware
        
        Returns:
            SIMD width in threads
        """
        if self.hardware and hasattr(self.hardware, 'simd_width'):
            return self.hardware.simd_width
        return 32  # Default for most modern GPUs
    
    def get_optimal_block_size(self, total_threads: int) -> Tuple[int, int, int]:
        """Get optimal block size for the given thread count
        
        Args:
            total_threads: Total number of threads
            
        Returns:
            Tuple of (x, y, z) block size
        """
        # For modern Metal GPUs, block sizes that are multiples of SIMD width work best
        if total_threads % self.simd_width == 0:
            block_size = total_threads
        else:
            # Round up to the next multiple of SIMD width
            block_size = ((total_threads + self.simd_width - 1) // self.simd_width) * self.simd_width
        
        # Cap at the maximum threads per threadgroup
        block_size = min(block_size, self.max_threads_per_threadgroup)
        
        # Default to 1D blocks for simplicity
        return (block_size, 1, 1)
    
    def get_grid_dimensions(self, total_blocks: int) -> Tuple[int, int, int]:
        """Get grid dimensions for the given block count
        
        Args:
            total_blocks: Total number of blocks
            
        Returns:
            Tuple of (x, y, z) grid dimensions
        """
        max_x, max_y, max_z = self.max_threadgroups
        
        # Simple 1D grid if it fits
        if total_blocks <= max_x:
            return (total_blocks, 1, 1)
        
        # Try 2D grid
        if total_blocks <= max_x * max_y:
            y = (total_blocks + max_x - 1) // max_x
            return (max_x, y, 1)
        
        # Use 3D grid
        xy = max_x * max_y
        z = (total_blocks + xy - 1) // xy
        z = min(z, max_z)
        return (max_x, max_y, z)
    
    def generate_thread_id_calculation(self, dim: int = 1) -> str:
        """Generate code for thread ID calculation
        
        Args:
            dim: Number of dimensions (1, 2, or 3)
            
        Returns:
            Metal code for thread ID calculation
        """
        if dim == 1:
            return """
            // 1D thread ID calculation
            uint thread_id = threadIdx.x + blockIdx.x * blockDim.x;
            """
        elif dim == 2:
            return """
            // 2D thread ID calculation
            uint thread_id_x = threadIdx.x + blockIdx.x * blockDim.x;
            uint thread_id_y = threadIdx.y + blockIdx.y * blockDim.y;
            uint thread_id = thread_id_y * gridDim.x * blockDim.x + thread_id_x;
            """
        elif dim == 3:
            return """
            // 3D thread ID calculation
            uint thread_id_x = threadIdx.x + blockIdx.x * blockDim.x;
            uint thread_id_y = threadIdx.y + blockIdx.y * blockDim.y;
            uint thread_id_z = threadIdx.z + blockIdx.z * blockDim.z;
            uint thread_id = (thread_id_z * gridDim.y * blockDim.y + thread_id_y) * gridDim.x * blockDim.x + thread_id_x;
            """
        else:
            raise ValueError(f"Unsupported dimension: {dim}")
    
    def generate_thread_mapping_defines(self) -> str:
        """Generate defines for thread mapping
        
        Returns:
            Metal code with thread mapping defines
        """
        return """
        // Thread mapping defines
        #define threadIdx metal::thread_position_in_threadgroup
        #define blockIdx metal::threadgroup_position_in_grid
        #define blockDim metal::threadgroup_size
        #define gridDim metal::grid_size
        
        // Thread ID calculation helpers
        #define get_thread_id() (threadIdx.x + blockIdx.x * blockDim.x)
        #define get_block_id() (blockIdx.x)
        #define get_num_threads() (blockDim.x * gridDim.x)
        #define get_num_blocks() (gridDim.x)
        
        // For kernel launching
        #define launch_triton_kernel(grid, block, kernel, ...) \
            kernel<<<grid, block>>>(__VA_ARGS__)
        """
    
    def get_kernel_attributes(self, threads_per_block: int) -> str:
        """Get kernel attributes for the given thread count
        
        Args:
            threads_per_block: Threads per block
            
        Returns:
            Metal kernel attributes
        """
        # Apple-specific attributes for optimal performance
        if self.hardware:
            # Check hardware version and enable appropriate optimizations
            if hasattr(self.hardware, 'chip_generation'):
                if self.hardware.chip_generation.value >= metal_hardware_optimizer.AppleSiliconGeneration.M3.value:
                    # M3 and newer can use additional optimizations
                    return f"[[thread_position_in_grid]] [[max_total_threads_per_threadgroup({threads_per_block})]]"
                elif self.hardware.chip_generation.value >= metal_hardware_optimizer.AppleSiliconGeneration.M2.value:
                    # M2 optimizations
                    return f"[[thread_position_in_grid]] [[max_total_threads_per_threadgroup({threads_per_block})]]"
                else:
                    # M1 and older
                    return f"[[thread_position_in_grid]]"
        
        # Default attributes
        return "[[thread_position_in_grid]]"

class SharedMemory:
    """Shared memory manager for Metal"""
    
    def __init__(self):
        """Initialize shared memory manager"""
        self.total_size = 0
        self.allocations = {}
        self.next_offset = 0
    
    def allocate(self, size: int, alignment: int = 16) -> int:
        """Allocate shared memory
        
        Args:
            size: Size in bytes
            alignment: Memory alignment
            
        Returns:
            Offset in shared memory
        """
        # Align the next offset
        aligned_offset = (self.next_offset + alignment - 1) & ~(alignment - 1)
        
        # Save the current offset
        offset = aligned_offset
        
        # Add the allocation
        self.allocations[offset] = size
        
        # Update the next offset and total size
        self.next_offset = offset + size
        self.total_size = max(self.total_size, self.next_offset)
        
        return offset
    
    def generate_declaration(self) -> str:
        """Generate shared memory declaration
        
        Returns:
            Metal code for shared memory declaration
        """
        if self.total_size > 0:
            return f"threadgroup char shared_memory[{self.total_size}];"
        else:
            return ""
    
    def generate_access_code(self, offset: int, type_name: str) -> str:
        """Generate code to access shared memory
        
        Args:
            offset: Offset in shared memory
            type_name: Type name for casting
            
        Returns:
            Metal code for accessing shared memory
        """
        return f"(({type_name}*)(&shared_memory[{offset}]))"
    
    def reset(self):
        """Reset the shared memory manager"""
        self.total_size = 0
        self.allocations = {}
        self.next_offset = 0

class SIMDGroupFunctions:
    """SIMD group function utilities for Metal"""
    
    @staticmethod
    def generate_reduce_sum(type_name: str, var_name: str) -> str:
        """Generate code for SIMD group sum reduction
        
        Args:
            type_name: Data type name
            var_name: Variable name
            
        Returns:
            Metal code for SIMD group sum reduction
        """
        return f"simd_sum({var_name})"
    
    @staticmethod
    def generate_reduce_product(type_name: str, var_name: str) -> str:
        """Generate code for SIMD group product reduction
        
        Args:
            type_name: Data type name
            var_name: Variable name
            
        Returns:
            Metal code for SIMD group product reduction
        """
        return f"simd_product({var_name})"
    
    @staticmethod
    def generate_reduce_min(type_name: str, var_name: str) -> str:
        """Generate code for SIMD group min reduction
        
        Args:
            type_name: Data type name
            var_name: Variable name
            
        Returns:
            Metal code for SIMD group min reduction
        """
        return f"simd_min({var_name})"
    
    @staticmethod
    def generate_reduce_max(type_name: str, var_name: str) -> str:
        """Generate code for SIMD group max reduction
        
        Args:
            type_name: Data type name
            var_name: Variable name
            
        Returns:
            Metal code for SIMD group max reduction
        """
        return f"simd_max({var_name})"
    
    @staticmethod
    def generate_broadcast(type_name: str, var_name: str, lane_id: int) -> str:
        """Generate code for SIMD group broadcast
        
        Args:
            type_name: Data type name
            var_name: Variable name
            lane_id: Source lane ID
            
        Returns:
            Metal code for SIMD group broadcast
        """
        return f"simd_broadcast({var_name}, {lane_id})"
    
    @staticmethod
    def generate_shuffle(type_name: str, var_name: str, source_lane: str) -> str:
        """Generate code for SIMD group shuffle
        
        Args:
            type_name: Data type name
            var_name: Variable name
            source_lane: Source lane expression
            
        Returns:
            Metal code for SIMD group shuffle
        """
        return f"simd_shuffle({var_name}, {source_lane})"
    
    @staticmethod
    def generate_shuffle_up(type_name: str, var_name: str, delta: int) -> str:
        """Generate code for SIMD group shuffle up
        
        Args:
            type_name: Data type name
            var_name: Variable name
            delta: Lane delta
            
        Returns:
            Metal code for SIMD group shuffle up
        """
        return f"simd_shuffle_up({var_name}, {delta})"
    
    @staticmethod
    def generate_shuffle_down(type_name: str, var_name: str, delta: int) -> str:
        """Generate code for SIMD group shuffle down
        
        Args:
            type_name: Data type name
            var_name: Variable name
            delta: Lane delta
            
        Returns:
            Metal code for SIMD group shuffle down
        """
        return f"simd_shuffle_down({var_name}, {delta})"
    
    @staticmethod
    def generate_shuffle_xor(type_name: str, var_name: str, mask: int) -> str:
        """Generate code for SIMD group shuffle XOR
        
        Args:
            type_name: Data type name
            var_name: Variable name
            mask: Lane mask
            
        Returns:
            Metal code for SIMD group shuffle XOR
        """
        return f"simd_shuffle_xor({var_name}, {mask})"

# Create global instances for convenience
thread_mapping = ThreadMapping()
shared_memory = SharedMemory()
simd_group_functions = SIMDGroupFunctions()

def map_kernel_launch_params(kernel_params):
    """
    映射Triton内核启动参数到Metal启动参数
    
    参数:
        kernel_params: Triton内核参数，包含网格和块维度
        
    返回:
        Metal内核启动参数
    """
    grid_dim = kernel_params.get("grid", (1, 1, 1))
    block_dim = kernel_params.get("block", (1, 1, 1))
    
    metal_grid_size, metal_threadgroup_size = thread_mapping.map_grid(grid_dim, block_dim)
    
    return {
        "grid_size": metal_grid_size,
        "threadgroup_size": metal_threadgroup_size,
        "shared_memory_size": kernel_params.get("shared_memory", 0)
    } 