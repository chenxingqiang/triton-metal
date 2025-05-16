#!/bin/bash
# Simplified PyPI release script that focuses on essential tasks
# This script implements the core steps from the PyPI release checklist

set -e  # Exit on error

# Print banner
echo "====================================================="
echo "Simplified PyPI Release Process for triton-metal"
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
  echo "You have the following options:"
  echo "1. Enter your PyPI token now (recommended)"
  echo "2. Set it manually before continuing with: export TWINE_PASSWORD=your-token-here"
  
  read -p "Enter your choice (1/2): " TOKEN_CHOICE
  
  if [ "$TOKEN_CHOICE" = "1" ]; then
    echo "Enter your PyPI token (input will be hidden):"
    read -s PYPI_TOKEN
    export TWINE_PASSWORD="$PYPI_TOKEN"
    echo "PyPI token set successfully"
  elif [ "$TOKEN_CHOICE" = "2" ]; then
    echo "Please set TWINE_PASSWORD manually and then continue."
    read -p "Press Enter when you have set TWINE_PASSWORD..." DUMMY
    if [ -z "$TWINE_PASSWORD" ]; then
      echo "TWINE_PASSWORD is still not set. Exiting."
      exit 1
    else
      echo "TWINE_PASSWORD is now set (hidden)"
    fi
  else
    echo "Invalid choice. Exiting."
    exit 1
  fi
else
  echo "TWINE_PASSWORD is set (hidden)"
fi

# Step 3: Clean and build the source distribution only (sdist)
# This avoids the need for LLVM and other complex dependencies
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
  echo "\nVerifying packages with twine..."
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

# Step 4: Upload to PyPI
echo -e "\n===== Step 4: Uploading to PyPI ====="

echo "The following files will be uploaded to PyPI:"
ls -la dist/

# Skip TestPyPI and proceed directly to PyPI
echo "\nSkipping TestPyPI and proceeding directly to PyPI upload"
PYPI_CHOICE="2"  # Set to PyPI option directly

if [ "$PYPI_CHOICE" = "1" ]; then
  # TestPyPI upload
  REPO_URL="https://test.pypi.org/legacy/"
  REPO_NAME="TestPyPI"
  INSTALL_CMD="pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ triton-metal"
  
  echo "Selected $REPO_NAME for upload"
  read -p "Do you want to proceed with the upload to $REPO_NAME? (y/N): " CONFIRM
  
  if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo -e "\n===== Uploading to $REPO_NAME ====="
    
    # Try to upload with a timeout
    echo "Uploading packages to $REPO_NAME..."
    if timeout 300 twine upload --repository-url $REPO_URL dist/*; then
      echo -e "\n===== Package published to $REPO_NAME successfully ====="
      echo "To install the published package from $REPO_NAME:"
      echo "$INSTALL_CMD"
    else
      echo "\nUpload to $REPO_NAME failed or timed out."
      echo "Please check your credentials and network connection."
    fi
  else
    echo -e "\n===== Upload to $REPO_NAME cancelled ====="
  fi
elif [ "$PYPI_CHOICE" = "2" ]; then
  # PyPI upload
  REPO_NAME="PyPI"
  INSTALL_CMD="pip install triton-metal"
  
  echo "Selected $REPO_NAME for upload"
  echo "\nWARNING: You are about to upload to the production PyPI repository."
  echo "This will make the package available to all users worldwide."
  read -p "Are you ABSOLUTELY SURE you want to proceed with the upload to $REPO_NAME? (yes/N): " CONFIRM
  
  if [ "$CONFIRM" = "yes" ]; then
    echo -e "\n===== Uploading to $REPO_NAME ====="
    
    # Try to upload with a timeout
    echo "Uploading packages to $REPO_NAME..."
    if timeout 300 twine upload dist/*; then
      echo -e "\n===== Package published to $REPO_NAME successfully ====="
      echo "To install the published package from $REPO_NAME:"
      echo "$INSTALL_CMD"
      
      # Remind about post-release tasks
      echo "\n===== Post-Release Tasks ====="
      echo "Don't forget to complete the following tasks:"
      echo "1. Create a GitHub release with tag v3.3.0+metal"
      echo "2. Update documentation website"
      echo "3. Plan for the next release"
    else
      echo "\nUpload to $REPO_NAME failed or timed out."
      echo "Please check your credentials and network connection."
    fi
  else
    echo -e "\n===== Upload to $REPO_NAME cancelled ====="
  fi
elif [ "$PYPI_CHOICE" = "3" ]; then
  echo "Skipping upload. Build completed successfully."
  echo "The built packages are available in the dist directory."
else
  echo "Invalid choice. Exiting without uploading."
  exit 1
fi

echo -e "\n===== PyPI release process completed ====="
