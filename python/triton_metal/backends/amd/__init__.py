"""
AMD ROCm backend for Triton-Metal.
"""

import os
import platform

__all__ = ['is_available', 'activate']

def is_available():
    """Check if AMD backend is available on this system"""
    # This is just a placeholder - in a real implementation,
    # we would check for ROCm/HIP availability
    return False

def activate():
    """Activate the AMD backend"""
    os.environ["TRITON_BACKEND"] = "amd"
    return True

# Context manager for temporarily using the AMD backend
class use_amd:
    def __init__(self):
        self.prev_backend = os.environ.get("TRITON_BACKEND", None)
    
    def __enter__(self):
        activate()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.prev_backend:
            os.environ["TRITON_BACKEND"] = self.prev_backend
        else:
            os.environ.pop("TRITON_BACKEND", None) 