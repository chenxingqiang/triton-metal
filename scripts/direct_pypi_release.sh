#!/bin/bash
# Direct PyPI release script that skips TestPyPI and uploads directly to PyPI
# This script implements the core steps from the PyPI release checklist

set -e  # Exit on error

# Print banner
echo "====================================================="
echo "Direct PyPI Release Process for triton-metal"
echo "====================================================="

# Step 1: Verify package configuration
echo -e "\n===== Step 1: Verifying package configuration ====="
python scripts/verify_package_config.py

if [ $? -ne 0 ]; then
  echo "Package configuration verification failed. Please fix the issues before proceeding."
  exit 1
fi

# Step 2: Set up PyPI credentials
echo -e "\n===== Step 2: Setting up PyPI credentials ====="

# Check if credentials are already set
if [ -z "$TWINE_USERNAME" ]; then
  export TWINE_USERNAME="__token__"
  echo "Set TWINE_USERNAME to __token__"
else
  echo "TWINE_USERNAME is already set to: $TWINE_USERNAME"
fi

# Check if token is set
if [ -z "$TWINE_PASSWORD" ]; then
  echo "TWINE_PASSWORD environment variable is not set."
  echo "Enter your PyPI token (input will be hidden):"
  read -s PYPI_TOKEN
  export TWINE_PASSWORD="$PYPI_TOKEN"
  echo "PyPI token set successfully"
else
  echo "TWINE_PASSWORD is set (hidden)"
fi

# Step 3: Clean and build the source distribution only (sdist)
echo -e "\n===== Step 3: Building source distribution ====="

# Clean up any existing build artifacts
echo "Cleaning up previous build artifacts..."
rm -rf dist/ build/ *.egg-info/

# Set version variables
export TRITON_WHEEL_VERSION_SUFFIX=""
echo "Using version suffix: ${TRITON_WHEEL_VERSION_SUFFIX:-(none)}"

# Build the source distribution only
echo "Building source distribution (sdist)..."
if python -m build --sdist; then
  echo "Source distribution build completed successfully."
else
  echo "Source distribution build failed."
  exit 1
fi

# Verify the built packages
echo -e "\n===== Verifying the built packages ====="
if [ -d "dist" ] && [ "$(ls -A dist)" ]; then
  echo "Found the following files in the dist directory:"
  ls -la dist/
  
  # Check the packages with twine
  echo -e "\nVerifying packages with twine..."
  if twine check dist/*; then
    echo "Package verification successful."
  else
    echo "Package verification failed. There might be issues with the package metadata."
    read -p "Do you want to continue with the upload process anyway? (y/N): " CONTINUE_AFTER_VERIFY_FAIL
    if [[ ! "$CONTINUE_AFTER_VERIFY_FAIL" =~ ^[Yy]$ ]]; then
      echo "Exiting. Please fix the package issues and try again."
      exit 1
    fi
  fi
else
  echo "No files found in the dist directory. The build process did not create any packages."
  exit 1
fi

# Step 4: Upload directly to PyPI
echo -e "\n===== Step 4: Uploading directly to PyPI ====="

echo "The following files will be uploaded to PyPI:"
ls -la dist/

echo -e "\nWARNING: You are about to upload to the production PyPI repository."
echo "This will make the package available to all users worldwide."
read -p "Are you ABSOLUTELY SURE you want to proceed with the upload to PyPI? (yes/N): " CONFIRM

if [ "$CONFIRM" = "yes" ]; then
  echo -e "\n===== Uploading to PyPI ====="
  
  # Try to upload with a timeout
  echo "Uploading packages to PyPI..."
  if timeout 300 twine upload dist/*; then
    echo -e "\n===== Package published to PyPI successfully ====="
    echo "To install the published package from PyPI:"
    echo "pip install triton-metal"
    
    # Remind about post-release tasks
    echo -e "\n===== Post-Release Tasks ====="
    echo "Don't forget to complete the following tasks:"
    echo "1. Create a GitHub release with tag v3.3.0+metal"
    echo "2. Update documentation website"
    echo "3. Plan for the next release"
  else
    echo -e "\nUpload to PyPI failed or timed out."
    echo "Please check your credentials and network connection."
  fi
else
  echo -e "\n===== Upload to PyPI cancelled ====="
fi

echo -e "\n===== PyPI release process completed ====="
