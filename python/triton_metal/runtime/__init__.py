"""
Runtime module for Triton-Metal package.
"""

import os
import importlib

# Attempt to import driver module
try:
    from . import driver
except ImportError:
    driver = None

# Export common attributes
__all__ = ['driver', 'get_backend']

def get_backend():
    """Get the currently active backend name"""
    return os.environ.get("TRITON_BACKEND", "metal") 