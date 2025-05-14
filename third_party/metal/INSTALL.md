# Triton Metal Backend Installation Guide

This guide provides instructions for installing and configuring the Triton Metal backend for running Triton kernels on Apple Silicon GPUs.

## System Requirements

- **macOS:** 13.5 or later (recommended macOS 14.0+)
- **Hardware:** Apple Silicon chip (M1, M2, or M3 series)
- **Python:** 3.9 or later
- **Xcode:** 14.0 or later with Metal command-line tools

## Dependencies

The Metal backend has the following dependencies:

- **MLX:** 0.3.0 or later
- **Triton:** Latest version
- **NumPy:** 1.22.0 or later
- **CMake:** 3.24 or later

## Installation

### 1. Install the Dependencies

```bash
# Install MLX
pip install mlx

# Install Triton
pip install triton

# Install other dependencies
pip install numpy cmake
```

### 2. Build from Source

To build the Triton Metal backend from source:

```bash
# Clone the repository
git clone https://github.com/openai/triton.git
cd triton

# Configure CMake with Metal backend enabled
mkdir build && cd build
cmake .. -DTRITON_BUILD_METAL_BACKEND=ON

# Build
make -j

# Install
make install
```

### 3. Verify Installation

To verify that the Metal backend is installed correctly:

```bash
# Run the Metal capability check
python -c "import triton; print('Metal backend available:', 'metal' in triton.runtime.backends)"

# Run the basic reduction test
cd triton/third_party/metal/python
python test_triton_integration.py -k test_backend_initialization
```

## Configuration Options

The Metal backend supports various configuration options to tune performance:

### Memory Layout Optimization

The backend automatically applies memory layout optimizations, including using the `COALESCED` layout for reduction operations. You can control the optimization level with the following options:

```python
import triton

# Configure compilation options
options = {
    "memory_optimization": "auto",       # Options: "none", "basic", "hardware_specific", "auto"
    "fusion_optimization": "auto",       # Options: "none", "basic", "hardware_specific", "auto"
    "metal_optimization_level": "auto",  # Options: "none", "basic", "standard", "aggressive", "experimental", "auto"
}

# Use options when launching a kernel
kernel[grid](*args, backend='metal', **options)
```

## Hardware-Specific Optimizations

The Metal backend applies hardware-specific optimizations based on the detected Apple Silicon generation:

- **M1 Optimizations:** Basic memory layout and coalescing
- **M2 Optimizations:** Enhanced memory layout with improved memory access patterns
- **M3 Optimizations:** Advanced memory optimizations including tensor cores usage and enhanced reduction operations

## Troubleshooting

### Common Issues

1. **Backend Not Found**

   If the Metal backend is not found, verify that it was built correctly:

   ```bash
   # Check available backends
   python -c "import triton; print(triton.runtime.backends.keys())"
   ```

2. **Compilation Errors**

   For compilation errors, enable debug info:

   ```python
   options = {"debug_info": True}
   kernel[grid](*args, backend='metal', **options)
   ```

3. **Performance Issues**

   If experiencing performance issues, try different optimization levels:

   ```python
   options = {"metal_optimization_level": "aggressive"}
   kernel[grid](*args, backend='metal', **options)
   ```

## Environment Variables

The following environment variables can be used to control the Metal backend behavior:

- `TRITON_METAL_DEBUG=1`: Enable debug output
- `TRITON_METAL_OPT_LEVEL=3`: Set optimization level (0-3)
- `TRITON_METAL_DISABLE_AUTOTUNING=1`: Disable autotuning
- `TRITON_METAL_CACHE_DIR=/path/to/cache`: Set cache directory for compiled kernels

## Testing and Validation

Run the test suite to validate the installation and functionality:

```bash
cd triton/third_party/metal/python
python test_reduction_memory.py
python test_metal_memory_manager.py
python test_integration.py
python test_triton_integration.py
```

## Getting Help

If you encounter issues with the Metal backend, please:

1. Check the error messages and debug output
2. Run the test scripts with verbose output
3. Verify that your Apple Silicon hardware is supported
4. File an issue on the GitHub repository with detailed information about your system and the problem 