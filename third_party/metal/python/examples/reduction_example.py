#!/usr/bin/env python
"""
Reduction Operations with COALESCED Memory Layout

This example demonstrates how reduction operations automatically use the COALESCED
memory layout in the Metal backend for optimal performance on Apple Silicon GPUs.
"""

import numpy as np
import torch
import triton
import triton.language as tl
import time
import argparse

# Try to import Metal backend components
try:
    # This would automatically use the COALESCED layout for reductions
    from metal_memory_manager import MemoryLayout
    METAL_BACKEND_AVAILABLE = True
    print(f"Metal backend available. COALESCED layout value: {MemoryLayout.COALESCED.value}")
except ImportError:
    METAL_BACKEND_AVAILABLE = False
    print("Metal backend not available. Running example using default CUDA backend.")


@triton.jit
def row_sum_kernel(
    input_ptr, output_ptr,
    n_rows, n_cols,
    input_row_stride, input_col_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute row-wise sum reduction.
    This will automatically use COALESCED layout in the Metal backend.
    """
    # Get the program ID
    pid = tl.program_id(0)
    
    # Check if we are within bounds
    if pid >= n_rows:
        return
        
    # Offset the input pointer to the start of this row
    input_row_ptr = input_ptr + pid * input_row_stride
    
    # Initialize the accumulator
    acc = 0.0
    
    # Process the row in blocks
    for i in range(0, n_cols, BLOCK_SIZE):
        # Create a mask for the elements in this block
        mask = tl.arange(0, BLOCK_SIZE) + i < n_cols
        
        # Load the elements for this block
        x = tl.load(input_row_ptr + tl.arange(0, BLOCK_SIZE) * input_col_stride, mask=mask, other=0.0)
        
        # Accumulate the sum
        acc += tl.sum(x, axis=0)
    
    # Store the result
    tl.store(output_ptr + pid, acc)


def row_sum_triton(x):
    """
    Compute row-wise sum reduction using Triton.
    
    Args:
        x: Input tensor of shape [batch, features]
        
    Returns:
        Row-wise sum tensor of shape [batch]
    """
    # Input dimensions
    batch, features = x.shape
    
    # Create output tensor
    y = torch.empty(batch, device=x.device, dtype=x.dtype)
    
    # Define grid and block size
    BLOCK_SIZE = 1024
    grid = (batch,)
    
    # Launch kernel
    row_sum_kernel[grid](
        x, y,
        batch, features,
        x.stride(0), x.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return y


def benchmark_reduction(batch_size, feature_size, dtype=torch.float32, num_repeats=100):
    """
    Benchmark reduction operations.
    
    Args:
        batch_size: Number of rows
        feature_size: Number of columns
        dtype: Data type
        num_repeats: Number of repetitions for timing
        
    Returns:
        Dictionary with benchmark results
    """
    # Create input tensor
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    x = torch.randn(batch_size, feature_size, device=device, dtype=dtype)
    
    # Compute reference result
    reference = torch.sum(x, dim=1)
    
    # Warm up
    triton_result = row_sum_triton(x)
    
    # Check correctness
    torch.testing.assert_close(triton_result, reference, rtol=1e-2, atol=1e-2)
    
    # Benchmark PyTorch
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_repeats):
        _ = torch.sum(x, dim=1)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) * 1000 / num_repeats
    
    # Benchmark Triton
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_repeats):
        _ = row_sum_triton(x)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) * 1000 / num_repeats
    
    # Return results
    return {
        "pytorch_ms": pytorch_time,
        "triton_ms": triton_time,
        "speedup": pytorch_time / triton_time,
        "batch_size": batch_size,
        "feature_size": feature_size,
        "metal_backend": METAL_BACKEND_AVAILABLE,
    }


def print_results(results):
    """Print benchmark results"""
    print("\n=== Benchmark Results ===")
    print(f"Batch size: {results['batch_size']}")
    print(f"Feature size: {results['feature_size']}")
    print(f"Metal backend available: {results['metal_backend']}")
    print(f"PyTorch time: {results['pytorch_ms']:.4f} ms")
    print(f"Triton time: {results['triton_ms']:.4f} ms")
    print(f"Speedup: {results['speedup']:.2f}x")
    
    # Print note about COALESCED layout
    if results['metal_backend']:
        print("\nNote: The Metal backend automatically uses COALESCED layout")
        print("for reduction operations like this row-wise sum.")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Benchmark reduction operations")
    parser.add_argument("--batch", type=int, default=1024, help="Batch size")
    parser.add_argument("--features", type=int, default=1024, help="Feature size")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16"], 
                        help="Data type")
    parser.add_argument("--repeats", type=int, default=100, help="Number of repetitions")
    
    args = parser.parse_args()
    
    # Convert dtype string to torch dtype
    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    
    # Run benchmark
    results = benchmark_reduction(args.batch, args.features, dtype, args.repeats)
    
    # Print results
    print_results(results)


if __name__ == "__main__":
    main() 