"""Metal compiler for Triton

This module implements the compiler interface for Metal backend on Apple Silicon GPUs.
It provides a bridge to the Metal backend implementation.
"""

import os
import sys
import tempfile
import pathlib
from typing import Dict, Union, Optional, List, Tuple, Any, Type
from types import ModuleType

from triton.backends.compiler import BaseBackend, GPUTarget

# Add Metal package to path
metal_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                          '..', '..', '..', '..', 
                                          'third_party', 'metal'))
if metal_path not in sys.path:
    sys.path.insert(0, metal_path)

class MetalBackend(BaseBackend):
    """Backend implementation for Metal backend on Apple Silicon GPUs"""
    
    @staticmethod
    def supports_target(target: GPUTarget) -> bool:
        """Check if this backend supports the given target
        
        Args:
            target: Target device specification
            
        Returns:
            True if the target is supported, False otherwise
        """
        return target.backend == 'metal'
    
    def __init__(self, target: GPUTarget) -> None:
        """Initialize the Metal backend
        
        Args:
            target: Target device specification
        """
        super().__init__(target)
        
        # Import Metal backend implementation
        try:
            from python.triton_to_metal_converter import TritonToMLXConverter
            self._converter = TritonToMLXConverter()
            
            # Try to import instrumentation module
            try:
                from python.metal_instrumentation import get_metal_instrumentation, get_error_diagnostics
                self.instrumentation = get_metal_instrumentation()
                self.error_diagnostics = get_error_diagnostics()
                self.has_instrumentation = True
            except ImportError:
                print("Warning: metal_instrumentation module not found. Debug and performance tracking will be disabled.")
                self.has_instrumentation = False
                self.instrumentation = None
                self.error_diagnostics = None
            
            # Set file extension for compiled binaries
            self.binary_ext = "metallib"
            
            # Initialize MLX and driver
            self._mlx = None
            self._driver = None
            
        except ImportError as e:
            print(f"Error initializing Metal backend: {e}")
            raise
    
    @property
    def mlx(self):
        """Lazy load MLX"""
        if self._mlx is None:
            try:
                import mlx.core as mx
                self._mlx = mx
            except ImportError:
                raise ImportError("MLX is required for Metal backend. Install it with 'pip install mlx'")
        return self._mlx
    
    @property
    def driver(self):
        """Get Metal driver instance"""
        if self._driver is None:
            from .driver import MetalDriver
            self._driver = MetalDriver()
        return self._driver
    
    def hash(self) -> str:
        """Get unique backend identifier
        
        Returns:
            String identifier for this backend
        """
        try:
            import mlx.core as mx
            version = mx.__version__
            return f'mlx-{version}-metal'
        except ImportError:
            return 'mlx-unknown-metal'
    
    def parse_options(self, options: dict) -> object:
        """Parse compilation options
        
        Args:
            options: Dictionary of options
            
        Returns:
            Parsed options object
        """
        # Import MetalOptions
        from python.metal_backend import MetalOptions
        
        # Get target architecture
        arch = self.target.arch if hasattr(self.target, 'arch') else 'apple-silicon'
        
        # Create options dictionary
        args = {'arch': arch}
        
        # Update with options
        for k, v in options.items():
            if k in [
                'num_warps', 'num_ctas', 'debug_info', 'opt_level', 'max_shared_memory',
                'mlx_shard_size', 'enable_fp_fusion', 'enable_interleaving', 'vectorize',
                'memory_optimization', 'fusion_optimization', 'metal_optimization_level'
            ] and v is not None:
                args[k] = v
        
        # Create and return options object
        return MetalOptions(**args)
    
    def add_stages(self, stages: dict, options: object) -> None:
        """Define compilation stages
        
        Args:
            stages: Dictionary to populate with compilation stages
            options: Parsed options object
        """
        # Import Metal backend
        from python.metal_backend import (
            make_ttir, make_ttgir, make_mlxir, make_metallib
        )
        
        # Define the compilation pipeline stages
        stages["ttir"] = lambda src, metadata: make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: make_ttgir(src, metadata, options)
        stages["mlxir"] = lambda src, metadata: make_mlxir(src, metadata, options)
        stages["metallib"] = lambda src, metadata: make_metallib(src, metadata, options)
    
    def load_dialects(self, context) -> None:
        """Load additional MLIR dialects into the provided context
        
        Args:
            context: MLIR context
        """
        # Load Metal-specific dialects when we have them
        pass
    
    def get_module_map(self) -> Dict[str, ModuleType]:
        """Return module mapping
        
        Returns:
            Dictionary mapping module names to module objects
        """
        try:
            import mlx.core as mx
            import mlx.nn as nn
            return {
                "mlx": mx,
                "mlx.core": mx,
                "mlx.nn": nn
            }
        except ImportError:
            return {} 