#!/bin/bash
# Secure script to set up PyPI environment variables and upload to PyPI
# This avoids hardcoding tokens in scripts

set -e  # Exit on error

# Set environment variables for PyPI upload
export TWINE_USERNAME="__token__"
# Note: TWINE_PASSWORD should be set before running this script
# export TWINE_PASSWORD="your-token-here"

# Verify environment variables
if [ -z "$TWINE_PASSWORD" ]; then
  echo "Error: TWINE_PASSWORD environment variable is not set."
  echo "Please set it before running this script:"
  echo "export TWINE_PASSWORD=your-token-here"
  exit 1
fi

echo "TWINE_USERNAME is set to: $TWINE_USERNAME"
echo "TWINE_PASSWORD is set (hidden)"

# Clean up any existing build artifacts
echo -e "\n===== Cleaning up build artifacts ====="
rm -rf dist/ build/ *.egg-info/

# Set version variables
# Default to no suffix, override with env var if needed
export TRITON_WHEEL_VERSION_SUFFIX=""
echo "Using version suffix: ${TRITON_WHEEL_VERSION_SUFFIX:-(none)}"

# Build the package
echo -e "\n===== Building package ====="
# Use the LLVM_SYSPATH if available
if [ -n "$LLVM_SYSPATH" ]; then
  echo "Using LLVM from: $LLVM_SYSPATH"
  python -m build
else
  echo "No LLVM_SYSPATH specified, attempting to build without it."
  echo "If build fails with SSL errors, try setting LLVM_SYSPATH."
  python -m build
fi

# Verify the built packages
echo -e "\n===== Verifying the built packages ====="
twine check dist/*

# Ask for confirmation before uploading
echo -e "\n===== Ready to upload to PyPI ====="
echo "The following files will be uploaded to PyPI:"
ls -la dist/
read -p "Do you want to proceed with the upload? (y/N): " CONFIRM
if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
  echo -e "\n===== Uploading to PyPI ====="
  twine upload dist/*
  echo -e "\n===== Package published successfully ====="
  echo "To install the published package:"
  echo "pip install triton-metal"
else
  echo -e "\n===== Upload cancelled ====="
fi
