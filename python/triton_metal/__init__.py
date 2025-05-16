"""
Triton-Metal: Enhanced Triton with Metal backend for Apple Silicon GPUs.
"""

__version__ = "3.3.0+metal"

# Import redirection to make it easier to migrate from 'triton' to 'triton_metal'
import sys
import importlib.util
import functools
import inspect

# Allow importing triton_metal.language as tl
try:
    from . import language as tl
except ImportError:
    pass

# Set up the backend environment variable if not already set
import os
if "TRITON_BACKEND" not in os.environ:
    os.environ["TRITON_BACKEND"] = "metal"  # Default to metal backend on Apple Silicon

# Check if we're running on Apple Silicon
def is_apple_silicon():
    import platform
    return platform.system() == "Darwin" and platform.machine() in ["arm64", "aarch64"]

# Warn if not on Apple Silicon but trying to use Metal backend
if os.environ.get("TRITON_BACKEND") == "metal" and not is_apple_silicon():
    import warnings
    warnings.warn("Using Metal backend on non-Apple Silicon device. This may not work as expected.")

# Core functionality: JIT compilation decorator
def jit(fn=None, **kwargs):
    """
    Just-In-Time compilation decorator for Triton kernels.

    Args:
        fn: The function to be compiled
        **kwargs: Additional compilation options

    Returns:
        A compiled kernel that can be called with grid and block dimensions
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # In a real implementation, this would compile and run the kernel
            # For now, this is just a placeholder
            return fn(*args, **kwargs)
        
        # Grid launch capability (placeholder implementation)
        @property
        def grid(self):
            class GridLauncher:
                def __getitem__(self, grid_dims):
                    def launcher(*args, **kwargs):
                        # This would actually launch the kernel with the specified grid
                        print(f"Would launch kernel with grid dimensions: {grid_dims}")
                        return wrapper(*args, **kwargs)
                    return launcher
            return GridLauncher()
        
        # Attach the grid property to the wrapper
        wrapper.grid = grid
        
        return wrapper
    
    if fn is None:
        return decorator
    return decorator(fn)

# Create a useful __all__ for wildcard imports
__all__ = ['language', 'runtime', 'backends', '__version__', 'tl', 'jit', 'is_apple_silicon']
