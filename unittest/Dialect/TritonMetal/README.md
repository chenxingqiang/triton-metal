# Metal Backend Unit Tests

This directory contains unit tests for the Triton Metal backend, which enables Triton to run on Apple Silicon GPUs.

## Test Organization

The tests are organized as follows:

### IR Tests

Tests in the `IR/` directory verify the Metal dialect's IR components:

- `DialectTest.cpp`: Basic tests for the Metal dialect's operations and types

### Transform Tests

Tests in the `Transforms/` directory verify the Metal-specific transformation passes:

- `TransformsTest.cpp`: Basic tests for Metal dialect transformation passes
- `MemoryOptimizerTest.cpp`: Tests for the Metal memory optimization passes, particularly for M3-specific memory layouts
- `M3OptimizationsTest.cpp`: Tests for M3-specific optimizations like 8-wide vectorization and SIMD group enhancements

### Integration Tests

- `MLXIntegrationTest.cpp`: Tests for integration with the MLX framework
- `HardwareDetectionTest.cpp`: Tests for Apple Silicon hardware detection and capability identification

## Running the Tests

You can run the Metal backend unit tests with the following command:

```bash
cd build
ninja check-triton-unit-tests
```

Or to run a specific test:

```bash
cd build
./unittest/Dialect/TritonMetal/TestTritonMetalDialect
```

## Test Environment

These tests are designed to run on Apple Silicon hardware. Some tests may be skipped when run on non-Apple hardware.

## Adding New Tests

When adding new tests for the Metal backend:

1. Create a new test file in the appropriate directory
2. Update the corresponding CMakeLists.txt file
3. Add an entry to this README to document the test's purpose

## Test Dependencies

The Metal backend tests depend on:

- Google Test framework
- MLIR Test Utilities
- Triton Metal libraries (TritonMetalIR, TritonMetalTransforms)
- MLX framework (for certain tests) 