# Metal-Specific Unit Tests

This directory contains platform-specific unit tests for the Triton Metal backend. These tests are designed to run on Apple Silicon hardware and may involve actual interaction with the Metal API and MLX framework.

## Test Files

### MetalBackendTest.cpp
Tests the basic functionality of the Metal backend, including:
- Detecting Metal availability
- Initializing the Metal backend
- Compiling simple kernels

### M3OptimizationsTest.cpp
Tests the M3-specific optimizations, including:
- Detecting M3 hardware
- Verifying shared memory size (64KB)
- Verifying vector width (8-wide)
- Verifying SIMD group width (32-wide)
- Testing tensor core support

### MetalMemoryManagerTest.cpp
Tests the Metal memory manager functionality, including:
- Initializing the memory manager
- Allocating and deallocating buffers
- Getting optimal tile sizes
- Getting optimal threadgroup sizes
- Getting optimal vector widths
- Testing memory layout strategies

## Test Requirements

These tests require:
- Apple Silicon hardware (M1, M2, or M3)
- macOS 13.5 or higher
- Metal framework
- MLX framework

## Running the Tests

```bash
cd build
ninja check-triton-unit-tests
```

## Debugging Tests

If tests fail, enable more verbose output:

```bash
cd build
ctest -V -R TestMetal
```

## Adding New Tests

When adding new Metal-specific tests:
1. Create a new test file in this directory
2. Update the CMakeLists.txt file
3. Make sure to conditionally run tests only on Apple Silicon hardware
4. Document the purpose of the test in this README 