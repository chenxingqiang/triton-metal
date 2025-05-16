#!/usr/bin/env python3
"""
Simple matrix multiplication example using Triton-Metal.

This example demonstrates how to implement a basic matrix multiplication
kernel using the Triton-Metal package on Apple Silicon GPUs.
"""

import numpy as np
import triton_metal
import triton_metal.language as tl

@triton_metal.jit
def matmul_kernel(
    # Pointers to the matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Block sizes (must be compile-time constants)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Matrix multiplication kernel: C = A @ B"""
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block start indices
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    # Pointers to the rows/columns of matrices
    a_ptr = a_ptr + m_start * K
    b_ptr = b_ptr + n_start
    c_ptr = c_ptr + m_start * N + n_start

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over k blocks
    for k in range(0, K, BLOCK_K):
        # Load a block of A
        a = tl.load(a_ptr + k)
        # Load a block of B
        b = tl.load(b_ptr + k * N)
        
        # Matrix multiplication
        acc += tl.dot(a, b)

    # Store the result
    tl.store(c_ptr, acc)

def matmul(a, b):
    """Compute matrix multiplication C = A @ B"""
    # Shape checking
    M, K = a.shape
    K_, N = b.shape
    assert K == K_, f"Incompatible dimensions: {a.shape} vs {b.shape}"
    
    # Allocate output
    c = np.empty((M, N), dtype=np.float32)
    
    # Grid dimensions
    grid = (M // 16, N // 16)
    
    # Launch kernel
    matmul_kernel[grid](a, b, c, M, N, K, BLOCK_M=16, BLOCK_N=16, BLOCK_K=16)
    
    return c

def main():
    """Run a simple benchmark"""
    # Matrix dimensions
    M, N, K = 1024, 1024, 1024
    
    # Create random matrices
    a = np.random.rand(M, K).astype(np.float32)
    b = np.random.rand(K, N).astype(np.float32)
    
    # Compute reference with NumPy
    c_ref = a @ b
    
    # Compute with Triton-Metal
    c = matmul(a, b)
    
    # Check results
    diff = np.abs(c - c_ref).max()
    print(f"Max difference: {diff}")
    
    # Simple benchmark
    import time
    
    iterations = 10
    start = time.time()
    for _ in range(iterations):
        c = matmul(a, b)
    triton_time = (time.time() - start) / iterations
    
    start = time.time()
    for _ in range(iterations):
        c_ref = a @ b
    numpy_time = (time.time() - start) / iterations
    
    print(f"NumPy time: {numpy_time*1000:.2f} ms")
    print(f"Triton-Metal time: {triton_time*1000:.2f} ms")
    print(f"Speedup: {numpy_time/triton_time:.2f}x")

if __name__ == "__main__":
    main() 