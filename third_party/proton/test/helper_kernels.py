import triton_metal.language as tl
import triton_metal


@triton_metal.jit
def custom_add(a_ptr):
    tl.store(a_ptr, 1.0)
