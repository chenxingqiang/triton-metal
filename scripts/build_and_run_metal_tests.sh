#!/bin/bash
set -e

# Check if running on macOS
if [[ "$(uname)" != "Darwin" ]]; then
  echo "Error: Metal tests can only be built on macOS"
  exit 1
fi

# Check if running on Apple Silicon
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "Warning: Building on non-Apple Silicon Mac. Tests will compile but may not run correctly."
fi

# Create build directory if it doesn't exist
if [ ! -d "build_metal" ]; then
  mkdir build_metal
fi

# Navigate to build directory
cd build_metal

# Configure with CMake
echo "Configuring CMake build with Metal support..."
cmake -DCMAKE_BUILD_TYPE=Release \
      -DTRITON_ENABLE_METAL=ON \
      -DTRITON_BUILD_TESTS=ON \
      ..

# Build Metal tests
echo "Building Metal tests..."
cmake --build . --target run_metal_tests

# Detect Apple Silicon generation for testing
CHIP_INFO=$(system_profiler SPHardwareDataType | grep "Chip")
if [[ $CHIP_INFO == *"Apple M3"* ]]; then
  export TRITON_METAL_IS_M3=1
  export TRITON_METAL_GENERATION=M3
  echo "Detected Apple M3 chip, enabling M3-specific tests"
elif [[ $CHIP_INFO == *"Apple M2"* ]]; then
  export TRITON_METAL_GENERATION=M2
  echo "Detected Apple M2 chip"
elif [[ $CHIP_INFO == *"Apple M1"* ]]; then
  export TRITON_METAL_GENERATION=M1
  echo "Detected Apple M1 chip"
else
  echo "Unknown Apple Silicon generation, defaulting to generic tests"
fi

# Run the tests
echo "Running Metal tests..."
./run_metal_tests

echo "Metal tests completed!" 