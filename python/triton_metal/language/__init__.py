"""
Triton-Metal language module.

This module provides the programming constructs for Triton kernels.
"""

# Import common functions and types that should be available in user code
__all__ = ['constexpr', 'program_id', 'num_programs', 'float32', 'float16',
           'int32', 'int64', 'load', 'store', 'atomic_add', 'atomic_max',
           'atomic_min', 'atomic_cas', 'atomic_xchg', 'extern_elementwise']

# Basic type definitions
float32 = "float32"
float16 = "float16"
int32 = "int32"
int64 = "int64"

# Placeholder functions that will be replaced by the actual implementation
# when running a kernel. These are here for documentation and type checking.
def constexpr(value):
    """Declare a compile-time constant"""
    return value

def program_id(axis):
    """Get the program ID along the specified axis"""
    return 0

def num_programs(axis):
    """Get the number of programs along the specified axis"""
    return 1

def load(ptr, offset=0):
    """Load a value from memory"""
    pass

def store(ptr, value, offset=0):
    """Store a value to memory"""
    pass

def atomic_add(ptr, value, offset=0):
    """Atomic addition operation"""
    pass

def atomic_max(ptr, value, offset=0):
    """Atomic maximum operation"""
    pass

def atomic_min(ptr, value, offset=0):
    """Atomic minimum operation"""
    pass

def atomic_cas(ptr, cmp, val, offset=0):
    """Atomic compare-and-swap operation"""
    pass

def atomic_xchg(ptr, value, offset=0):
    """Atomic exchange operation"""
    pass

def extern_elementwise(name, output_types, input_types):
    """Create an extern elementwise function"""
    def decorator(f):
        return f
    return decorator
