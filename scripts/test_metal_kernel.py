#!/usr/bin/env python
# Test script for Triton Metal backend
# This creates and runs a simple vector addition kernel

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

# Define a simple vector addition kernel
@triton_metal.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Define the program ID and the block size
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create a mask to handle the case where the block size doesn't divide the elements
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load the inputs
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Perform the operation
    output = x + y
    
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

def add_vectors(x, y):
    """Add two vectors using the Metal backend."""
    # Define the input and output arrays
    output = np.empty_like(x)
    
    # Define the grid
    n_elements = len(x)
    block_size = 128
    grid = (triton_metal.cdiv(n_elements, block_size),)
    
    # Run the kernel
    try:
        print(f"Launching kernel with grid={grid}")
        start_time = time.time()
        add_kernel[grid](
            x, y, output, n_elements, block_size
        )
        end_time = time.time()
        print(f"Kernel execution time: {(end_time - start_time) * 1000:.2f} ms")
    except Exception as e:
        print(f"Error running kernel: {e}")
        raise
    
    return output

def test_metal_addition(size=1024*1024):
    """Test the Metal backend with a simple addition kernel."""
    # Create two input arrays
    print(f"Creating test arrays with size {size}")
    try:
        x = np.random.rand(size).astype(np.float32)
        y = np.random.rand(size).astype(np.float32)
    except Exception as e:
        print(f"Error creating test arrays: {e}")
        return False
    
    # Calculate the expected result
    print("Computing reference result on CPU...")
    expected = x + y
    
    # Calculate using Metal backend
    print(f"Running vector addition with size {size} on Metal backend...")
    try:
        result = add_vectors(x, y)
    except Exception as e:
        print(f"Error in Metal kernel execution: {e}")
        return False
    
    # Compare results
    try:
        max_diff = np.max(np.abs(result - expected))
        mean_diff = np.mean(np.abs(result - expected))
        print(f"Max difference: {max_diff}")
        print(f"Mean difference: {mean_diff}")
        
        # Check if the results are close
        if max_diff < 1e-5:
            print("✅ Test PASSED! Metal backend is working correctly.")
            return True
        else:
            print("❌ Test FAILED! Results do not match.")
            print(f"  First few expected values: {expected[:5]}")
            print(f"  First few actual values: {result[:5]}")
            return False
    except Exception as e:
        print(f"Error comparing results: {e}")
        return False

def run_tests():
    print("=" * 40)
    print("Testing Triton-Metal with Metal backend")
    print("=" * 40)
    
    # Start with a small test to ensure basic functionality
    print("\nRunning small test (1K elements)...")
    if not test_metal_addition(size=1024):
        print("Small test failed! Exiting.")
        return False
        
    # Now run the full test
    print("\nRunning full test (1M elements)...")
    return test_metal_addition(size=1024*1024)

if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\nMetal backend test completed successfully!")
        sys.exit(0)
    else:
        print("\nMetal backend test failed!")
        sys.exit(1) 