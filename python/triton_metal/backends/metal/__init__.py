"""
Metal backend for Triton on Apple Silicon GPUs.
"""

import os
import platform

__all__ = ['is_available', 'activate']

def is_available():
    """Check if Metal backend is available on this system"""
    return platform.system() == "Darwin" and platform.machine() in ["arm64", "aarch64"]

def activate():
    """Activate the Metal backend"""
    if not is_available():
        import warnings
        warnings.warn("Metal backend is only available on Apple Silicon devices!")
        return False
    
    os.environ["TRITON_BACKEND"] = "metal"
    return True

# Context manager for temporarily using the Metal backend
class use_metal:
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

# Auto-activate if running on Apple Silicon
if is_available():
    activate() 