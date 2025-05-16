#!/bin/bash
set -e

# Script to build and install triton-metal locally for development testing
# This avoids the need to push to TestPyPI for testing

echo "=== Cleaning previous builds ==="
rm -rf build/ dist/ *.egg-info/

echo "=== Setting up development environment ==="
# Ensure correct version without git hash for consistent testing
export TRITON_WHEEL_VERSION_SUFFIX=""

# Create a virtual environment for testing if it doesn't exist
if [ ! -d ".venv" ]; then
  echo "=== Creating virtual environment ==="
  python -m venv .venv
  echo "Virtual environment created at .venv/"
fi

# Activate the virtual environment
source .venv/bin/activate

# Install development dependencies
echo "=== Installing development dependencies ==="
if ! pip install -e ".[build,metal]"; then
  echo ""
  echo "=== Installation failed! ==="
  
  # Suggest the LLVM download fallback script
  echo "If you encountered SSL errors (like TLS/SSL connection issues), try:"
  echo "1. Run the LLVM download fallback script:"
  echo "   ./scripts/download_llvm_fallback.sh"
  echo ""
  echo "2. Then install again with the downloaded LLVM path:"
  echo "   LLVM_SYSPATH=\$HOME/.triton/llvm/llvm-<hash>-<system-suffix> pip install -e \".[build,metal]\""
  echo ""
  exit 1
fi

pip install numpy  # Required for the Metal test

echo "=== Running basic tests ==="
# Add simple test to verify the installation
python -c "
import sys
print('Python version:', sys.version)
try:
    import triton_metal
    print('Triton-Metal version:', triton_metal.__version__)
    print('Triton-Metal installed successfully!')
    
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
echo "=== Local development installation complete ==="

# Ask if the user wants to run the Metal test
read -p "Do you want to run the Metal backend test? (y/N): " RUN_TEST
if [[ "$RUN_TEST" =~ ^[Yy]$ ]]; then
  echo "=== Running Metal backend test ==="
  python scripts/test_metal_kernel.py
fi

echo ""
echo "The virtual environment is now active. To deactivate, run 'deactivate'"
echo "To run your tests, use the current shell or activate the environment again with:"
echo "source .venv/bin/activate"
echo ""
echo "To test the Metal backend, run:"
echo "python scripts/test_metal_kernel.py" 