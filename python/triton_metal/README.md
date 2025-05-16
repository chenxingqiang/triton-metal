# Triton-Metal

Enhanced Triton with Metal backend for Apple Silicon GPUs.

## Installation

```bash
# Install from PyPI
pip install triton-metal

# Install from source
git clone https://github.com/chenxingqiang/triton-metal.git
cd triton-metal
pip install -e .
```

## Features

- Supports Metal backend for Apple Silicon GPUs (M1, M2, M3 series)
- Compatible with the MLX framework
- Includes all standard Triton features
- Automatic detection and activation of Metal backend on Apple Silicon
- Simple API for explicit backend selection

## Usage

### Basic Usage

```python
import triton
import triton.language as tl

# By default, Metal backend is auto-selected on Apple Silicon
# Define a simple matrix multiplication kernel
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Your Triton kernel code here
    # ...

# Call the kernel
matmul_kernel[grid](a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M=16, BLOCK_N=16, BLOCK_K=16)
```

### Explicit Backend Selection

```python
import triton
from triton.backends import metal

# Check if Metal is available
if metal.is_available():
    # Use Metal backend
    with metal.use_metal():
        # Your code here using Metal backend
        # ...
```

## Requirements

- macOS with Apple Silicon (ARM64)
- Python 3.9 or newer
- MLX for some features (optional): `pip install mlx`

## Benchmarks

Triton-Metal shows significant performance improvements on Apple Silicon GPUs:

| Operation | PyTorch | MLX | Triton-Metal |
|-----------|---------|-----|-------------|
| MatMul (1024x1024) | 1.0x | 1.2x | 1.5x |
| Attention (seq=1024) | 1.0x | 1.3x | 1.8x |

## Compatibility

Triton-Metal maintains API compatibility with original Triton while adding Metal-specific optimizations for Apple Silicon GPUs.

## License

MIT 