#!/usr/bin/env python
"""
Reduction Example for Triton Metal Backend

This example demonstrates how to use the Triton Metal backend for reduction operations,
which benefit from the COALESCED memory layout optimization.

The example compares different reduction implementations:
1. NumPy implementation
2. Basic Triton reduction with Metal backend
3. Optimized Triton reduction with Metal backend and auto-tuning
4. MLX implementation (for reference)
"""

import os
import time
import argparse
import numpy as np

# Check if Triton is available
try:
    import triton
    import triton.language as tl
    from triton.runtime.autotuner import autotune
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("Warning: Triton not available. Only NumPy implementation will be run.")

# Check if MLX is available
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("Warning: MLX not available. MLX comparison will be skipped.")

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Triton Metal Reduction Example")
    parser.add_argument("--size", type=int, default=1024*1024, help="Size of input array")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials for timing")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--metal-opt", choices=["none", "basic", "standard", "aggressive", "experimental", "auto"], 
                       default="auto", help="Metal optimization level")
    return parser.parse_args()

def numpy_reduction(x):
    """NumPy implementation of reduction (sum)"""
    return np.sum(x)

@triton.jit
def basic_reduction_kernel(
    x_ptr, 
    out_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    """Basic reduction kernel"""
    # Create block ID and offset
    pid = tl.program_id(0)
    block_offset = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_offset + tl.arange(0, BLOCK_SIZE)
    
    # Bounds check
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Perform basic reduction
    reduced = tl.sum(x, axis=0)
    
    # Store result
    tl.store(out_ptr + pid, reduced)

@triton.jit
def optimized_reduction_kernel(
    x_ptr, 
    out_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr,
    WORK_PER_THREAD: tl.constexpr
):
    """Optimized reduction kernel with work per thread parameter"""
    # Create block ID and offset
    pid = tl.program_id(0)
    
    # Calculate how many elements this block processes
    elements_per_block = BLOCK_SIZE * WORK_PER_THREAD
    block_offset = pid * elements_per_block
    
    # Initialize accumulator
    acc = 0.0
    
    # Each thread processes multiple elements
    for i in range(WORK_PER_THREAD):
        # Calculate offset for this iteration
        offset = block_offset + i * BLOCK_SIZE + tl.program_id(1)
        
        # Bounds check
        if offset < n_elements:
            # Load data and accumulate
            x = tl.load(x_ptr + offset)
            acc += x
    
    # Perform reduction across all threads in the group
    reduced = tl.reduce(acc, 0, op=tl.add)
    
    # Only first thread in group stores the result
    if tl.program_id(1) == 0:
        tl.store(out_ptr + pid, reduced)

@autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64, 'WORK_PER_THREAD': 1}),
        triton.Config({'BLOCK_SIZE': 128, 'WORK_PER_THREAD': 1}),
        triton.Config({'BLOCK_SIZE': 256, 'WORK_PER_THREAD': 1}),
        triton.Config({'BLOCK_SIZE': 512, 'WORK_PER_THREAD': 1}),
        triton.Config({'BLOCK_SIZE': 1024, 'WORK_PER_THREAD': 1}),
        triton.Config({'BLOCK_SIZE': 64, 'WORK_PER_THREAD': 2}),
        triton.Config({'BLOCK_SIZE': 128, 'WORK_PER_THREAD': 2}),
        triton.Config({'BLOCK_SIZE': 256, 'WORK_PER_THREAD': 2}),
        triton.Config({'BLOCK_SIZE': 64, 'WORK_PER_THREAD': 4}),
        triton.Config({'BLOCK_SIZE': 128, 'WORK_PER_THREAD': 4}),
    ],
    key=['n_elements'],
)
def reduction_launcher(x_ptr, out_ptr, n_elements, metal_opt="auto"):
    """Launch the optimized reduction kernel with auto-tuning"""
    # Calculate block size and grid
    meta = {'BLOCK_SIZE': 512, 'WORK_PER_THREAD': 1}  # Default if autotuner hasn't run yet
    
    elements_per_block = meta['BLOCK_SIZE'] * meta['WORK_PER_THREAD']
    grid = (triton.cdiv(n_elements, elements_per_block), meta['BLOCK_SIZE'])
    
    # Launch the kernel with Metal backend and specified optimization level
    optimized_reduction_kernel[grid](
        x_ptr, out_ptr, n_elements, 
        BLOCK_SIZE=meta['BLOCK_SIZE'],
        WORK_PER_THREAD=meta['WORK_PER_THREAD'],
        backend='metal',
        metal_optimization_level=metal_opt,
    )

def run_numpy_reduction(x_np, trials=10):
    """Run and time NumPy reduction"""
    start = time.time()
    
    for _ in range(trials):
        result = numpy_reduction(x_np)
    
    end = time.time()
    elapsed = (end - start) / trials
    
    return result, elapsed

def run_basic_triton_reduction(x_np, trials=10, metal_opt="auto"):
    """Run and time basic Triton reduction with Metal backend"""
    if not HAS_TRITON:
        return None, 0.0
    
    # Send data to device
    x = triton.testing.to_device(x_np)
    n_elements = x_np.shape[0]
    
    # Determine grid size
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Allocate output
    output = triton.testing.to_device(np.zeros((grid[0]), dtype=np.float32))
    
    # Warmup
    basic_reduction_kernel[grid](
        x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE,
        backend='metal',
        metal_optimization_level=metal_opt
    )
    
    # Time execution
    start = time.time()
    
    for _ in range(trials):
        basic_reduction_kernel[grid](
            x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE,
            backend='metal',
            metal_optimization_level=metal_opt
        )
    
    end = time.time()
    elapsed = (end - start) / trials
    
    # Get result and compute final sum
    result_blocks = triton.testing.to_numpy(output)
    result = np.sum(result_blocks)
    
    return result, elapsed

def run_optimized_triton_reduction(x_np, trials=10, metal_opt="auto"):
    """Run and time optimized Triton reduction with Metal backend and auto-tuning"""
    if not HAS_TRITON:
        return None, 0.0
    
    # Send data to device
    x = triton.testing.to_device(x_np)
    n_elements = x_np.shape[0]
    
    # Allocate output (worst-case size with block size 64)
    worst_case_blocks = triton.cdiv(n_elements, 64)
    output = triton.testing.to_device(np.zeros((worst_case_blocks), dtype=np.float32))
    
    # Warmup and allow autotuner to run
    reduction_launcher(x, output, n_elements, metal_opt=metal_opt)
    
    # Time execution
    start = time.time()
    
    for _ in range(trials):
        reduction_launcher(x, output, n_elements, metal_opt=metal_opt)
    
    end = time.time()
    elapsed = (end - start) / trials
    
    # Get result and compute final sum
    result_blocks = triton.testing.to_numpy(output)
    result = np.sum(result_blocks)
    
    return result, elapsed

def run_mlx_reduction(x_np, trials=10):
    """Run and time MLX reduction for comparison"""
    if not HAS_MLX:
        return None, 0.0
    
    # Convert to MLX array
    x_mx = mx.array(x_np)
    
    # Warmup
    result = mx.sum(x_mx)
    mx.eval(result)  # Ensure computation is done
    
    # Time execution
    start = time.time()
    
    for _ in range(trials):
        result = mx.sum(x_mx)
        mx.eval(result)  # Ensure computation is done
    
    end = time.time()
    elapsed = (end - start) / trials
    
    return result.item(), elapsed

def run_benchmark(args):
    """Run all benchmarks and compare performance"""
    # Create input data
    print(f"Creating input array of size {args.size}...")
    x_np = np.random.randn(args.size).astype(np.float32)
    
    # Print debug info if requested
    if args.debug and HAS_TRITON:
        print(f"Triton version: {triton.__version__}")
        print(f"Available backends: {list(triton.runtime.backends.keys())}")
        if 'metal' in triton.runtime.backends:
            print("Metal backend is available!")
        else:
            print("WARNING: Metal backend is not available!")
    
    print("\nRunning benchmarks...")
    print(f"Metal optimization level: {args.metal_opt}")
    print("-" * 60)
    
    # Run NumPy reduction
    numpy_result, numpy_time = run_numpy_reduction(x_np, args.trials)
    print(f"NumPy reduction:                 {numpy_time*1000:.3f} ms")
    
    if HAS_TRITON:
        # Check if Metal backend is available
        if 'metal' not in triton.runtime.backends:
            print("Metal backend not available. Skipping Triton tests.")
        else:
            # Run basic Triton reduction
            basic_result, basic_time = run_basic_triton_reduction(x_np, args.trials, args.metal_opt)
            print(f"Basic Triton reduction (Metal):   {basic_time*1000:.3f} ms")
            print(f"  - Speedup vs NumPy:            {numpy_time/basic_time:.2f}x")
            print(f"  - Result matches:              {np.isclose(numpy_result, basic_result, rtol=1e-3)}")
            
            # Run optimized Triton reduction
            opt_result, opt_time = run_optimized_triton_reduction(x_np, args.trials, args.metal_opt)
            print(f"Optimized Triton reduction:      {opt_time*1000:.3f} ms")
            print(f"  - Speedup vs NumPy:            {numpy_time/opt_time:.2f}x")
            print(f"  - Speedup vs Basic Triton:     {basic_time/opt_time:.2f}x")
            print(f"  - Result matches:              {np.isclose(numpy_result, opt_result, rtol=1e-3)}")
    
    if HAS_MLX:
        # Run MLX reduction for comparison
        mlx_result, mlx_time = run_mlx_reduction(x_np, args.trials)
        print(f"MLX reduction:                  {mlx_time*1000:.3f} ms")
        print(f"  - Speedup vs NumPy:            {numpy_time/mlx_time:.2f}x")
        if HAS_TRITON and 'metal' in triton.runtime.backends:
            print(f"  - Vs Optimized Triton:        {opt_time/mlx_time:.2f}x")
        print(f"  - Result matches:              {np.isclose(numpy_result, mlx_result, rtol=1e-3)}")
    
    print("-" * 60)
    print("Note: The Triton Metal backend uses COALESCED memory layout for reduction operations.")
    print("      This optimizes the memory access pattern for reduction workloads on Apple Silicon.")

if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args) 