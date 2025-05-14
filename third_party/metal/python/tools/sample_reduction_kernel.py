#!/usr/bin/env python
"""
Sample Triton Kernel with Various Reduction Operations

This file demonstrates different reduction operations in Triton
that would use the COALESCED memory layout in the Metal backend.
"""

import triton
import triton.language as tl
import torch

@triton.jit
def sum_reduction_kernel(
    input_ptr, output_ptr, 
    M, N,
    stride_m, stride_n, 
    BLOCK_SIZE: tl.constexpr
):
    """
    Simple sum reduction kernel that reduces along the N dimension.
    This will use the COALESCED memory layout for optimal performance.
    """
    # Program ID
    pid = tl.program_id(0)
    
    # Offset the input pointer to the current row
    row_start_ptr = input_ptr + pid * stride_m
    
    # Initialize accumulator
    acc = 0.0
    
    # Load and reduce values along the row (N dimension)
    for i in range(0, N, BLOCK_SIZE):
        mask = i + tl.arange(0, BLOCK_SIZE) < N
        values = tl.load(row_start_ptr + i * stride_n, mask=mask, other=0.0)
        acc += tl.sum(values, axis=0)
    
    # Store the result
    tl.store(output_ptr + pid, acc)

@triton.jit
def max_reduction_kernel(
    input_ptr, output_ptr, 
    M, N,
    stride_m, stride_n, 
    BLOCK_SIZE: tl.constexpr
):
    """
    Max reduction kernel that reduces along the N dimension.
    This will use the COALESCED memory layout for optimal performance.
    """
    # Program ID
    pid = tl.program_id(0)
    
    # Offset the input pointer to the current row
    row_start_ptr = input_ptr + pid * stride_m
    
    # Initialize accumulator with minimum value
    acc = -float('inf')
    
    # Load and reduce values along the row (N dimension)
    for i in range(0, N, BLOCK_SIZE):
        mask = i + tl.arange(0, BLOCK_SIZE) < N
        values = tl.load(row_start_ptr + i * stride_n, mask=mask, other=-float('inf'))
        acc = tl.maximum(acc, tl.max(values, axis=0))
    
    # Store the result
    tl.store(output_ptr + pid, acc)

@triton.jit
def mean_reduction_kernel(
    input_ptr, output_ptr, 
    M, N,
    stride_m, stride_n, 
    BLOCK_SIZE: tl.constexpr
):
    """
    Mean reduction kernel that reduces along the N dimension.
    This will use the COALESCED memory layout for optimal performance.
    """
    # Program ID
    pid = tl.program_id(0)
    
    # Offset the input pointer to the current row
    row_start_ptr = input_ptr + pid * stride_m
    
    # Initialize accumulator
    acc = 0.0
    
    # Load and reduce values along the row (N dimension)
    for i in range(0, N, BLOCK_SIZE):
        mask = i + tl.arange(0, BLOCK_SIZE) < N
        values = tl.load(row_start_ptr + i * stride_n, mask=mask, other=0.0)
        acc += tl.sum(values, axis=0)
    
    # Compute mean by dividing by N
    acc = acc / N
    
    # Store the result
    tl.store(output_ptr + pid, acc)

@triton.jit
def multi_axis_reduction_kernel(
    input_ptr, output_ptr, 
    M, N, K,
    stride_m, stride_n, stride_k,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Multi-axis reduction kernel that reduces along the M and N dimensions,
    keeping the K dimension. This will use the COALESCED memory layout.
    """
    # Program ID for the K dimension
    pid = tl.program_id(0)
    
    # Initialize accumulator
    acc = 0.0
    count = 0
    
    # Reduce along M and N dimensions
    for m in range(0, M, BLOCK_M):
        for n in range(0, N, BLOCK_N):
            # Calculate masks for boundary checking
            m_mask = m + tl.arange(0, BLOCK_M) < M
            n_mask = n + tl.arange(0, BLOCK_N) < N
            
            # Combined mask
            mask = tl.reshape(m_mask[:, None] & n_mask[None, :], (-1,))
            
            # Calculate offsets
            offsets = tl.reshape(
                (m + tl.arange(0, BLOCK_M)[:, None]) * stride_m + 
                (n + tl.arange(0, BLOCK_N)[None, :]) * stride_n,
                (-1,)
            )
            
            # Load and accumulate values
            values = tl.load(input_ptr + offsets + pid * stride_k, mask=mask, other=0.0)
            acc += tl.sum(values, axis=0)
            count += tl.sum(mask, axis=0)
    
    # Compute mean 
    acc = acc / count
    
    # Store the result
    tl.store(output_ptr + pid, acc)

def test_sum_reduction():
    """
    Test the sum reduction kernel
    """
    # Create test data
    M, N = 128, 1024
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    y = torch.empty(M, device='cuda', dtype=torch.float32)
    
    # Launch kernel
    grid = (M,)
    sum_reduction_kernel[grid](
        x, y, 
        M, N,
        x.stride(0), x.stride(1),
        BLOCK_SIZE=256
    )
    
    # Verify result
    y_ref = torch.sum(x, dim=1)
    assert torch.allclose(y, y_ref, rtol=1e-2, atol=1e-2)
    print("Sum reduction test passed!")

def test_max_reduction():
    """
    Test the max reduction kernel
    """
    # Create test data
    M, N = 128, 1024
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    y = torch.empty(M, device='cuda', dtype=torch.float32)
    
    # Launch kernel
    grid = (M,)
    max_reduction_kernel[grid](
        x, y, 
        M, N,
        x.stride(0), x.stride(1),
        BLOCK_SIZE=256
    )
    
    # Verify result
    y_ref = torch.max(x, dim=1)[0]
    assert torch.allclose(y, y_ref, rtol=1e-2, atol=1e-2)
    print("Max reduction test passed!")

def test_mean_reduction():
    """
    Test the mean reduction kernel
    """
    # Create test data
    M, N = 128, 1024
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    y = torch.empty(M, device='cuda', dtype=torch.float32)
    
    # Launch kernel
    grid = (M,)
    mean_reduction_kernel[grid](
        x, y, 
        M, N,
        x.stride(0), x.stride(1),
        BLOCK_SIZE=256
    )
    
    # Verify result
    y_ref = torch.mean(x, dim=1)
    assert torch.allclose(y, y_ref, rtol=1e-2, atol=1e-2)
    print("Mean reduction test passed!")

def test_multi_axis_reduction():
    """
    Test the multi-axis reduction kernel
    """
    # Create test data
    M, N, K = 64, 64, 32
    x = torch.randn(M, N, K, device='cuda', dtype=torch.float32)
    y = torch.empty(K, device='cuda', dtype=torch.float32)
    
    # Launch kernel
    grid = (K,)
    multi_axis_reduction_kernel[grid](
        x, y, 
        M, N, K,
        x.stride(0), x.stride(1), x.stride(2),
        BLOCK_M=16, BLOCK_N=16
    )
    
    # Verify result
    y_ref = torch.mean(x, dim=(0, 1))
    assert torch.allclose(y, y_ref, rtol=1e-2, atol=1e-2)
    print("Multi-axis reduction test passed!")

if __name__ == "__main__":
    print("\nTesting reduction kernels that use COALESCED memory layout...")
    
    # Run tests
    test_sum_reduction()
    test_max_reduction()
    test_mean_reduction()
    test_multi_axis_reduction()
    
    print("\nAll tests passed! All these reduction operations use COALESCED memory layout.") 