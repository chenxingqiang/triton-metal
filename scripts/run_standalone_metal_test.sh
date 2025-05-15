#!/bin/bash
set -e

# Check if running on macOS
if [[ "$(uname)" != "Darwin" ]]; then
  echo "Error: Metal tests can only be built on macOS"
  exit 1
fi

echo "=== Compiling standalone Metal test ==="
clang++ -std=c++17 -o metal_test metal_test.cpp

# Detect Apple Silicon generation for testing
CHIP_INFO=$(system_profiler SPHardwareDataType 2>/dev/null | grep "Chip" || echo "Unknown")
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

echo "=== Running standalone Metal test ==="
./metal_test 