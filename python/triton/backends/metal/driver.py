"""Metal driver for Triton

This module implements the driver interface for Metal backend on Apple Silicon GPUs.
It provides a bridge to the Metal device through MLX.
"""

import os
import sys
import ctypes
import importlib
from pathlib import Path
from typing import Dict, Union, Optional, List, Set, Callable, Any

from triton.backends.driver import DriverBase

# Add Metal package to path
metal_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                          '..', '..', '..', '..', 
                                          'third_party', 'metal'))
if metal_path not in sys.path:
    sys.path.insert(0, metal_path)

class MetalDriver(DriverBase):
    """Triton driver for Metal backend on Apple Silicon GPUs"""
    
    def __init__(self):
        """Initialize the Metal driver, loading MLX and Metal dependencies"""
        # Import MLX and Metal-specific modules
        try:
            import mlx.core as mx
            self.mlx = mx
            
            # Import Metal backend components
            from python.metal_hardware_optimizer import hardware_capabilities, AppleSiliconGeneration
            self.hardware_capabilities = hardware_capabilities
            
            # Set device properties
            self.metal_info = self._get_metal_device_info()
            
            # Set default device
            self._current_device = 0
            
            # Flag to check if Metal is available
            self.is_available = True
            
        except ImportError as e:
            print(f"Warning: Failed to initialize Metal driver: {e}")
            self.mlx = None
            self.is_available = False
    
    def _get_metal_device_info(self) -> Dict[str, Any]:
        """Get Metal device information"""
        try:
            info = {
                "name": "Apple Metal GPU",
                "device_count": 1,  # Metal presents a unified view of all GPUs
                "compute_capability": "metal",
                "max_shared_memory": self.hardware_capabilities.shared_memory_size,
                "max_threads_per_block": self.hardware_capabilities.max_threads_per_threadgroup,
                "warp_size": self.hardware_capabilities.simd_width,
                "simd_width": self.hardware_capabilities.simd_width,
                "chip_generation": self.hardware_capabilities.chip_generation.name,
            }
            
            # Add more hardware-specific details
            if hasattr(self.hardware_capabilities, "chip_model"):
                info["chip_model"] = self.hardware_capabilities.chip_model
                
            if hasattr(self.hardware_capabilities, "unified_memory_size"):
                info["unified_memory_size"] = self.hardware_capabilities.unified_memory_size
                
            return info
        except Exception as e:
            print(f"Warning: Failed to get Metal device info: {e}")
            return {
                "name": "Apple Metal GPU",
                "device_count": 1,
                "compute_capability": "metal",
            }
    
    def get_current_device(self) -> int:
        """Get the current device identifier"""
        return self._current_device
    
    def set_current_device(self, device: int) -> None:
        """Set the current device
        
        Args:
            device: Device index (should be 0 for Metal)
        """
        if device != 0:
            raise ValueError("Metal backend only supports device index 0")
        self._current_device = device
    
    def get_current_stream(self, device: Optional[int] = None) -> int:
        """Get the current stream for the specified device
        
        Args:
            device: Device index (default: current device)
            
        Returns:
            Stream ID (always 0 for Metal backend)
        """
        return 0
    
    def get_driver_version(self) -> int:
        """Get the driver version
        
        Returns:
            Driver version as an integer
        """
        try:
            if hasattr(self.mlx, "__version__"):
                # Extract version components from MLX version (e.g., 0.3.0 -> 300)
                version_str = self.mlx.__version__
                components = version_str.split('.')
                if len(components) >= 3:
                    return int(components[0]) * 10000 + int(components[1]) * 100 + int(components[2])
                return 0
            return 0
        except Exception:
            return 0
    
    def get_device_count(self) -> int:
        """Get the number of available devices
        
        Returns:
            Number of devices (always 1 for Metal backend)
        """
        return 1 if self.is_available else 0
    
    def get_device_properties(self, device: int) -> Dict[str, Any]:
        """Get properties of the specified device
        
        Args:
            device: Device index
            
        Returns:
            Dictionary of device properties
        """
        if device != 0:
            raise ValueError("Metal backend only supports device index 0")
        return self.metal_info
    
    def synchronize(self, device: Optional[int] = None) -> None:
        """Synchronize the specified device
        
        Args:
            device: Device index (default: current device)
        """
        # MLX operations are synchronized automatically
        pass
    
    def load_binary(self, binary, name, device) -> int:
        """Load a compiled binary onto the device
        
        Args:
            binary: Compiled binary data
            name: Kernel name
            device: Device index
            
        Returns:
            Handle to the loaded binary
        """
        # The Metal driver will implement this later based on the binary format
        # For now, it's just a placeholder that returns a dummy handle
        return id(binary)

    def unload_binary(self, handle) -> None:
        """Unload a previously loaded binary
        
        Args:
            handle: Handle to the loaded binary
        """
        # No need to implement for now
        pass 