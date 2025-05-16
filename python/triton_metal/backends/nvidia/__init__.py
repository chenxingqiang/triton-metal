"""
NVIDIA CUDA backend for Triton-Metal.
"""

import os
import platform

__all__ = ['is_available', 'activate']

def is_available():
    """Check if NVIDIA backend is available on this system"""
    # This is just a placeholder - in a real implementation,
    # we would check for CUDA availability
    return False

def activate():
    """Activate the NVIDIA backend"""
    os.environ["TRITON_BACKEND"] = "nvidia"
    return True

# Context manager for temporarily using the NVIDIA backend
class use_nvidia:
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