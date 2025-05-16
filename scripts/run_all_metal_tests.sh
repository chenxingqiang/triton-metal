#!/bin/bash
# Script to run all Metal backend tests
# This is useful for developers to quickly test the Metal backend

set -e  # Exit on error

echo "====================================================="
echo "Running all Metal backend tests"
echo "====================================================="

# Print system information
SYSTEM=$(uname -s)
ARCH=$(uname -m)
echo "System: $SYSTEM, Architecture: $ARCH"
echo "Python: $(python --version)"

# Make sure Metal backend is active
export TRITON_BACKEND="metal"

# Verify installation
echo -e "\n\n===== Verifying Triton-Metal installation ====="
python -c "import triton_metal; print('Triton-Metal version:', triton_metal.__version__)"
python -c "from triton_metal.runtime import driver; print('Available backends:', driver.get_available_backends())"

# Make sure Metal backend is available
if ! python -c "from triton_metal.runtime import driver; assert 'metal' in driver.get_available_backends()"; then
  echo "ERROR: Metal backend not available!"
  exit 1
fi

echo -e "\n\n===== Running basic vector addition test ====="
python scripts/test_metal_kernel.py
if [ $? -ne 0 ]; then
  echo "ERROR: Basic vector addition test failed!"
  exit 1
fi

echo -e "\n\n===== Running matrix multiplication test ====="
python scripts/test_metal_matmul.py
if [ $? -ne 0 ]; then
  echo "ERROR: Matrix multiplication test failed!"
  exit 1
fi

# Run unit tests if available
echo -e "\n\n===== Running unit tests (if available) ====="
if [ -d "python/test/unit/metal" ]; then
  echo "Found tests in python/test/unit/metal"
  pytest -xvs python/test/unit/metal
  if [ $? -ne 0 ]; then
    echo "WARNING: Some unit tests failed"
    # Don't exit here, just warn
  fi
elif [ -d "third_party/metal/python/tests" ]; then
  echo "Found tests in third_party/metal/python/tests"
  pytest -xvs third_party/metal/python/tests
  if [ $? -ne 0 ]; then
    echo "WARNING: Some unit tests failed"
    # Don't exit here, just warn
  fi
else
  echo "No unit tests found, skipping"
fi

echo -e "\n\n===== All Metal backend tests completed successfully! =====" 