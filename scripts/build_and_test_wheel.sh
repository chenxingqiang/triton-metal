#!/bin/bash
set -e

# Script to build and test a wheel locally without pushing to PyPI
# Requirements: pip install build twine

echo "=== Cleaning previous builds ==="
rm -rf build/ dist/ *.egg-info/

echo "=== Building triton-metal package wheel ==="
# Ensure correct version without git hash
export TRITON_WHEEL_VERSION_SUFFIX=""

# Build the wheel and source distribution
if ! python -m build; then
  echo ""
  echo "=== Build failed! ==="
  
  # Check if the error contains SSL-related messages
  echo "If you encountered SSL errors (like TLS/SSL connection issues), try:"
  echo "1. Run the LLVM download fallback script:"
  echo "   ./scripts/download_llvm_fallback.sh"
  echo ""
  echo "2. Then build again with the downloaded LLVM path:"
  echo "   LLVM_SYSPATH=\$HOME/.triton/llvm/llvm-<hash>-<system-suffix> python -m build"
  echo ""
  exit 1
fi

echo "=== Verifying the built packages ==="
twine check dist/*

# Create a testing directory if it doesn't exist
TEST_DIR="wheel_test_env"
if [ -d "$TEST_DIR" ]; then
  echo "=== Cleaning previous test environment ==="
  rm -rf "$TEST_DIR"
fi

echo "=== Creating test environment ==="
python -m venv "$TEST_DIR"
source "$TEST_DIR/bin/activate"

# Upgrade pip and install wheel
pip install --upgrade pip wheel

# Find the wheel file
WHEEL_FILE=$(ls dist/*.whl | head -n 1)
if [ -z "$WHEEL_FILE" ]; then
  echo "Error: No wheel file found in dist directory"
  exit 1
fi

echo "=== Installing wheel $WHEEL_FILE ==="
# Install the wheel with required dependencies
pip install "$WHEEL_FILE[metal]"
pip install numpy  # Required for the Metal test

echo "=== Testing the installation ==="
python -c "
import sys
print('Python version:', sys.version)
try:
    import triton_metal
    print('Triton-Metal version:', triton_metal.__version__)
    print('Triton-Metal wheel installed successfully!')
    
    # Show available backends
    from triton_metal.runtime import driver
    backends = driver.get_available_backends()
    print('Available backends:', backends)
    
    # Check if Metal backend is available
    if 'metal' in backends:
        print('Metal backend is available!')
    else:
        print('Warning: Metal backend not found.')
except ImportError as e:
    print('Error importing Triton-Metal:', e)
    exit(1)
"

echo ""
echo "=== Testing wheel locally complete ==="

# Ask if the user wants to run the Metal test
read -p "Do you want to run the Metal backend test? (y/N): " RUN_TEST
if [[ "$RUN_TEST" =~ ^[Yy]$ ]]; then
  echo "=== Running Metal backend test ==="
  python $(dirname "$0")/test_metal_kernel.py
fi

echo ""
echo "The wheel has been built and tested successfully!"
echo "Wheel file: $WHEEL_FILE"
echo ""
echo "To use the test environment again, activate it with:"
echo "source $TEST_DIR/bin/activate"
echo ""
echo "To install the wheel elsewhere, use:"
echo "pip install $WHEEL_FILE[metal]"
echo ""
echo "To test the Metal backend, run:"
echo "python scripts/test_metal_kernel.py"
echo ""
echo "To exit the test environment, run: deactivate" 