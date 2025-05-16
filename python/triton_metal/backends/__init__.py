"""
Triton-Metal backends module.

This module provides backends for different hardware platforms.
"""

try:
    from . import metal
except ImportError:
    metal = None

try:
    from . import nvidia
except ImportError:
    nvidia = None

try:
    from . import amd
except ImportError:
    amd = None

import os

def get_current_backend():
    """Get the currently active backend name"""
    return os.environ.get("TRITON_BACKEND", "metal")

def list_available_backends():
    """List all available backends"""
    available = []
    if metal is not None and metal.is_available():
        available.append("metal")
    if nvidia is not None:
        available.append("nvidia")
    if amd is not None:
        available.append("amd")
    return available

__all__ = ['metal', 'nvidia', 'amd', 'get_current_backend', 'list_available_backends']
