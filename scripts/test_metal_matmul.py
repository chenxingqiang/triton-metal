#!/usr/bin/env python
# Test script for Triton Metal backend
# This creates and runs a matrix multiplication kernel

import os
import sys
import time
import numpy as np

# Check if running in CI
is_ci = os.environ.get('CI', 'false').lower() == 'true'

# Print diagnostic info
print(f"Python version: {sys.version}")
print(f"Running in CI: {is_ci}")
print(f"Current directory: {os.getcwd()}")

# Set Metal backend
os.environ["TRITON_BACKEND"] = "metal"

try:
    import triton_metal
    import triton_metal.language as tl
    print(f"Successfully imported triton_metal from {triton_metal.__file__}")
except ImportError as e:
    print(f"Error: triton-metal package not installed properly: {e}")
    sys.exit(1)

# Print Triton-Metal version
print(f"Triton-Metal version: {triton_metal.__version__}")

# Available backends
try:
    from triton_metal.runtime import driver
    backends = driver.get_available_backends()
    print(f"Available backends: {backends}")

    if 'metal' not in backends:
        print("Error: Metal backend not available!")
        sys.exit(1)
        
    print("Metal backend available!")
except Exception as e:
    print(f"Error loading backends: {e}")
    sys.exit(1)

# Define a simple matrix multiplication kernel
@triton_metal.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how to access the next element along a particular dimension
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    """Kernel for computing the matrix multiplication C = A x B.
    
    A has shape (M, K), B has shape (K, N), and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ID to the tiles of output matrix
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create 2D blocks and use the PTX block indices to identify which part
    # of the output this program is responsible for computing
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks to handle the case where the block dimensions don't divide the matrix dimensions
    mask_m = rm < M
    mask_n = rn < N
    
    # Using rm and rn, compute the memory locations for inputs and outputs
    c_ptrs = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate through the inner dimension in blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_block = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        mask_k = k_block < K
        
        # Load inputs for this K block
        a_ptrs = a_ptr + (rm[:, None] * stride_am + k_block[None, :] * stride_ak)
        b_ptrs = b_ptr + (k_block[:, None] * stride_bk + rn[None, :] * stride_bn)
        
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        
        # Compute matrix multiplication for this K block
        acc += tl.dot(a, b)
    
    # Store the result
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])

def matmul(a, b):
    """Computes matrix multiplication using the Metal backend: C = A @ B"""
    # Get the shapes of the input matrices
    M, K = a.shape
    K2, N = b.shape
    
    # Make sure the inner dimensions match
    assert K == K2, f"Inner dimensions must match: {K} vs {K2}"
    
    # Create output matrix
    c = np.empty((M, N), dtype=np.float32)
    
    # Define meta-parameters
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    GROUP_SIZE_M = 8
    
    # Define the grid
    grid = lambda META: (
        triton_metal.cdiv(M, META['BLOCK_SIZE_M']) * triton_metal.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # Run the kernel
    try:
        print(f"Running matmul kernel for matrices of shape ({M}, {K}) x ({K}, {N})")
        start_time = time.time()
        matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.strides[0] // 4, a.strides[1] // 4,
            b.strides[0] // 4, b.strides[1] // 4,
            c.strides[0] // 4, c.strides[1] // 4,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
            GROUP_SIZE_M
        )
        end_time = time.time()
        print(f"Kernel execution time: {(end_time - start_time) * 1000:.2f} ms")
        return c
    except Exception as e:
        print(f"Error running matmul kernel: {e}")
        raise

def run_matmul_test(M=128, N=128, K=128):
    """Run a test for the matrix multiplication kernel."""
    # Create random matrices
    print(f"Creating test matrices A({M}, {K}) and B({K}, {N})")
    try:
        a = np.random.rand(M, K).astype(np.float32)
        b = np.random.rand(K, N).astype(np.float32)
    except Exception as e:
        print(f"Error creating test matrices: {e}")
        return False
    
    # Calculate expected result using numpy
    print("Computing reference result on CPU...")
    try:
        expected = np.matmul(a, b)
    except Exception as e:
        print(f"Error computing reference result: {e}")
        return False
    
    # Calculate using Metal backend
    print("Computing result using Metal backend...")
    try:
        result = matmul(a, b)
    except Exception as e:
        print(f"Error in Metal kernel execution: {e}")
        return False
    
    # Compare results
    try:
        max_diff = np.max(np.abs(result - expected))
        mean_diff = np.mean(np.abs(result - expected))
        rel_diff = np.max(np.abs((result - expected) / (expected + 1e-5)))
        print(f"Max absolute difference: {max_diff}")
        print(f"Mean absolute difference: {mean_diff}")
        print(f"Max relative difference: {rel_diff}")
        
        # Check if the results are close
        if max_diff < 1e-3 and rel_diff < 1e-3:  # Relaxed tolerance for matmul
            print("✅ Test PASSED! Metal matmul is working correctly.")
            return True
        else:
            print("❌ Test FAILED! Results do not match.")
            print(f"  First few expected values:\n{expected[:3, :3]}")
            print(f"  First few actual values:\n{result[:3, :3]}")
            return False
    except Exception as e:
        print(f"Error comparing results: {e}")
        return False

def run_tests():
    """Run a series of matrix multiplication tests with different sizes."""
    print("=" * 50)
    print("Testing Metal Backend Matrix Multiplication")
    print("=" * 50)
    
    # Test small matrices first
    print("\nRunning small matrices test (32x32)...")
    if not run_matmul_test(32, 32, 32):
        print("Small matrices test failed! Exiting.")
        return False
    
    # Test medium matrices
    print("\nRunning medium matrices test (128x128)...")
    if not run_matmul_test(128, 128, 128):
        print("Medium matrices test failed! Exiting.")
        return False
    
    # Test non-square matrices
    print("\nRunning non-square matrices test (64x128x256)...")
    if not run_matmul_test(64, 256, 128):
        print("Non-square matrices test failed! Exiting.")
        return False
    
    # All tests passed
    return True

if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\nAll Metal matmul tests passed successfully!")
        sys.exit(0)
    else:
        print("\nMetal matmul tests failed!")
        sys.exit(1) 