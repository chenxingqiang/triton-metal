# Triton Metal Backend

This is a Metal backend for Triton that enables running Triton kernels on Apple Silicon GPUs using the MLX framework.

## Overview

The Triton Metal backend provides integration between Triton and Apple's Metal GPU framework through MLX. This enables efficient execution of Triton kernels on Apple Silicon GPUs, including M1, M2, and M3 chips.

## Features

- Support for Apple Silicon GPUs via Metal and MLX
- Hardware detection for M1, M2, and M3 chips
- Optimized operation mapping from Triton to MLX
- Operation fusion for common patterns
- Benchmarking tools to compare performance

## Requirements

- macOS running on Apple Silicon (M1, M2, or M3)
- Python 3.8+
- MLX (`pip install mlx`)

## Architecture

The Metal backend consists of several components:

1. **Metal Driver**: Manages the connection to the Metal GPU and handles device capabilities
2. **MLX Backend**: Compiles Triton kernels to MLX operations
3. **Metal Executor**: Executes the compiled kernels on the Metal GPU
4. **Operation Mapping**: Maps Triton operations to MLX equivalents
5. **Fusion Optimizer**: Identifies patterns of operations that can be fused together for better performance

## Performance

Benchmark results show competitive performance against PyTorch's MPS backend, with an average 1.24x speedup on an M3 chip for common operations including:

- Matrix multiplication
- Element-wise operations (add, mul, exp, tanh)
- Reduction operations (sum, mean)
- Softmax
- Attention mechanism

## Project Structure

```
metal/
├── backend/
│   ├── driver.py           # Metal driver implementation
│   ├── mlx_backend.py      # MLX backend compiler
│   └── executor.py         # Kernel execution engine
├── include/
│   └── triton/
│       └── Dialect/
│           └── TritonMetal/
│               ├── IR/     # Metal dialect IR definitions
│               └── Transforms/ # Metal-specific transformations
├── python/
│   ├── metal_hardware_optimizer.py  # Hardware detection and optimization
│   ├── operation_mapping.py         # Triton to MLX operation mapping
│   ├── metal_fusion_optimizer.py    # Operation fusion patterns
│   ├── benchmark/                   # Benchmarking tools
│   └── test/                        # Unit tests
└── CMakeLists.txt                   # Build configuration
```

## Usage

To use the Metal backend in your Triton code:

```python
import triton
import triton.language as tl

# Set backend to Metal
os.environ["TRITON_BACKEND"] = "metal"

# Define a simple kernel
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    output = x + y
    
    tl.store(output_ptr + offsets, output, mask=mask)

# Execute as usual
grid = (triton.cdiv(n_elements, 128),)
add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=128)
```

## Limitations

- Not all Triton operations are currently supported
- Complex memory access patterns may not be optimized
- Current implementation focuses on the most common operations
- Feature parity with CUDA backend is a work in progress

## Future Work

- Improve operation coverage
- Enhance fusion patterns for higher performance
- Support for more complex memory patterns
- Better integration with Metal Performance Shaders
- Support for advanced M3 features (Dynamic Caching)

## Development

To contribute to the Metal backend:

1. Clone the repository
2. Install the development requirements
3. Run the tests: `python -m third_party.metal.python.test_metal_backend`
4. Run the benchmarks: `python -m third_party.metal.python.benchmark.metal_backend_benchmark`

## License

This project is part of Triton and follows the same license. 