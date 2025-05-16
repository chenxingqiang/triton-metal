#!/bin/bash
# Complete PyPI release script that handles LLVM download and PyPI upload
# This script implements steps 1-5 from the PyPI release checklist

set -e  # Exit on error

# Print banner
echo "====================================================="
echo "Complete PyPI Release Process for triton-metal"
echo "====================================================="

# Step 1: Verify package configuration
echo -e "\n===== Step 1: Verifying package configuration ====="
python scripts/verify_package_config.py

# Get LLVM hash from cmake file
LLVM_HASH=$(head -n 1 "$(dirname "$0")/../cmake/llvm-hash.txt")
LLVM_SHORT_HASH=${LLVM_HASH:0:8}
echo "Using LLVM hash: ${LLVM_SHORT_HASH}"

# Determine system and architecture
SYSTEM=$(uname)
ARCH=$(uname -m)

if [ "$SYSTEM" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
  SYSTEM_SUFFIX="macos-arm64"
  echo "Detected Apple Silicon (ARM64)"
elif [ "$SYSTEM" = "Darwin" ] && [ "$ARCH" = "x86_64" ]; then
  SYSTEM_SUFFIX="macos-x64"
  echo "Detected macOS Intel (x86_64)"
elif [ "$SYSTEM" = "Linux" ] && [ "$ARCH" = "x86_64" ]; then
  SYSTEM_SUFFIX="linux-x64"
  echo "Detected Linux (x86_64)"
else
  echo "Unsupported system: $SYSTEM $ARCH"
  exit 1
fi

# Step 2: Handle LLVM download
echo -e "\n===== Step 2: Setting up LLVM ====="

# Use local directories within the project to avoid permission issues
LOCAL_CACHE_DIR="./build/cache"
LOCAL_LLVM_DIR="./build/llvm"
LLVM_EXTRACT_DIR="$LOCAL_LLVM_DIR/llvm-${LLVM_SHORT_HASH}-${SYSTEM_SUFFIX}"
TARBALL_NAME="llvm-${LLVM_SHORT_HASH}-${SYSTEM_SUFFIX}.tar.gz"
DOWNLOAD_PATH="$LOCAL_CACHE_DIR/$TARBALL_NAME"

# Create local directories
mkdir -p "$LOCAL_CACHE_DIR"
mkdir -p "$LOCAL_LLVM_DIR"

# Check if LLVM is already extracted
if [ -d "$LLVM_EXTRACT_DIR" ] && [ -n "$(ls -A "$LLVM_EXTRACT_DIR")" ]; then
  echo "LLVM already extracted at $LLVM_EXTRACT_DIR"
  export LLVM_SYSPATH="$LLVM_EXTRACT_DIR"
else
  echo "LLVM not found or empty at $LLVM_EXTRACT_DIR"
  
  # Check if we have the tarball
  if [ -f "$DOWNLOAD_PATH" ]; then
    echo "LLVM tarball found at $DOWNLOAD_PATH"
    
    # Try to extract it
    echo "Extracting LLVM tarball to $LLVM_EXTRACT_DIR..."
    mkdir -p "$LLVM_EXTRACT_DIR"
    if tar -xzf "$DOWNLOAD_PATH" -C "$LLVM_EXTRACT_DIR" --strip-components=1; then
      echo "Extraction successful"
      export LLVM_SYSPATH="$LLVM_EXTRACT_DIR"
    else
      echo "Extraction failed. The tarball might be corrupted."
      rm -rf "$LLVM_EXTRACT_DIR"
      echo "Deleted corrupted extraction directory."
    fi
  else
    echo "LLVM tarball not found at $DOWNLOAD_PATH"
    
    # Offer options to the user
    echo "\nOptions for LLVM setup:"
    echo "1. Try to download LLVM using the fallback script"
    echo "2. Specify a custom LLVM path"
    echo "3. Continue without LLVM (build may fail)"
    read -p "Enter your choice (1/2/3): " LLVM_CHOICE
    
    if [ "$LLVM_CHOICE" = "1" ]; then
      echo "\nRunning LLVM download fallback script..."
      if ./scripts/download_llvm_fallback.sh; then
        echo "Download successful. Extracting..."
        mkdir -p "$LLVM_EXTRACT_DIR"
        if tar -xzf "$DOWNLOAD_PATH" -C "$LLVM_EXTRACT_DIR" --strip-components=1; then
          echo "Extraction successful"
          export LLVM_SYSPATH="$LLVM_EXTRACT_DIR"
        else
          echo "Extraction failed."
        fi
      else
        echo "Download failed."
      fi
    elif [ "$LLVM_CHOICE" = "2" ]; then
      read -p "Enter the full path to your LLVM installation: " CUSTOM_LLVM_PATH
      if [ -d "$CUSTOM_LLVM_PATH" ]; then
        export LLVM_SYSPATH="$CUSTOM_LLVM_PATH"
        echo "Using custom LLVM path: $LLVM_SYSPATH"
      else
        echo "The specified path does not exist."
        read -p "Do you want to continue without LLVM? This may fail. (y/N): " CONTINUE_WITHOUT_LLVM
        if [[ ! "$CONTINUE_WITHOUT_LLVM" =~ ^[Yy]$ ]]; then
          echo "Exiting. Please set up LLVM and try again."
          exit 1
        fi
      fi
    elif [ "$LLVM_CHOICE" = "3" ]; then
      echo "Continuing without LLVM. Build may fail."
    else
      echo "Invalid choice. Exiting."
      exit 1
    fi
  fi
fi

# Step 3: Set up PyPI credentials
echo -e "\n===== Step 3: Setting up PyPI credentials ====="

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
  echo "3. Continue without setting it (will be prompted during upload)"
  
  read -p "Enter your choice (1/2/3): " TOKEN_CHOICE
  
  if [ "$TOKEN_CHOICE" = "1" ]; then
    echo "Enter your PyPI token (input will be hidden):"
    read -s PYPI_TOKEN
    export TWINE_PASSWORD="$PYPI_TOKEN"
    echo "PyPI token set successfully"
  elif [ "$TOKEN_CHOICE" = "2" ]; then
    echo "Please set TWINE_PASSWORD manually and then continue."
    read -p "Press Enter when you have set TWINE_PASSWORD..." DUMMY
    if [ -z "$TWINE_PASSWORD" ]; then
      echo "TWINE_PASSWORD is still not set."
      read -p "Do you want to continue anyway? (y/N): " CONTINUE_ANYWAY
      if [[ ! "$CONTINUE_ANYWAY" =~ ^[Yy]$ ]]; then
        echo "Exiting. Please set TWINE_PASSWORD and try again."
        exit 1
      fi
    else
      echo "TWINE_PASSWORD is now set (hidden)"
    fi
  elif [ "$TOKEN_CHOICE" = "3" ]; then
    echo "Continuing without setting TWINE_PASSWORD. You will be prompted during upload."
  else
    echo "Invalid choice. Exiting."
    exit 1
  fi
else
  echo "TWINE_PASSWORD is set (hidden)"
fi

# Step 4: Clean and build the package
echo -e "\n===== Step 4: Building package ====="

# Clean up any existing build artifacts
echo "Cleaning up previous build artifacts..."
rm -rf dist/ build/ *.egg-info/

# Set version variables
export TRITON_WHEEL_VERSION_SUFFIX=""
echo "Using version suffix: ${TRITON_WHEEL_VERSION_SUFFIX:-(none)}"

# Build the package
echo "Starting the build process..."
if [ -n "$LLVM_SYSPATH" ]; then
  echo "Using LLVM from: $LLVM_SYSPATH"
  
  # Try building with LLVM_SYSPATH
  echo "Building with LLVM_SYSPATH set..."
  if LLVM_SYSPATH="$LLVM_SYSPATH" python -m build; then
    echo "Build completed successfully with LLVM_SYSPATH."
    BUILD_SUCCESS=true
  else
    echo "Build failed with LLVM_SYSPATH."
    BUILD_SUCCESS=false
    
    # Ask if user wants to try without LLVM_SYSPATH
    read -p "Do you want to try building without LLVM_SYSPATH? (y/N): " TRY_WITHOUT_LLVM
    if [[ "$TRY_WITHOUT_LLVM" =~ ^[Yy]$ ]]; then
      echo "Attempting build without LLVM_SYSPATH..."
      if python -m build; then
        echo "Build completed successfully without LLVM_SYSPATH."
        BUILD_SUCCESS=true
      else
        echo "Build failed without LLVM_SYSPATH as well."
      fi
    fi
  fi
else
  echo "No LLVM_SYSPATH specified, attempting to build without it."
  if python -m build; then
    echo "Build completed successfully without LLVM_SYSPATH."
    BUILD_SUCCESS=true
  else
    echo "Build failed without LLVM_SYSPATH."
    BUILD_SUCCESS=false
  fi
fi

# Check if build was successful
if [ "$BUILD_SUCCESS" != "true" ]; then
  echo "\nBuild process failed. Please check the error messages above."
  read -p "Do you want to continue with the upload process anyway? (y/N): " CONTINUE_AFTER_FAIL
  if [[ ! "$CONTINUE_AFTER_FAIL" =~ ^[Yy]$ ]]; then
    echo "Exiting. Please fix the build issues and try again."
    exit 1
  fi
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
  read -p "Do you want to continue with the upload process anyway? (y/N): " CONTINUE_WITHOUT_PACKAGES
  if [[ ! "$CONTINUE_WITHOUT_PACKAGES" =~ ^[Yy]$ ]]; then
    echo "Exiting. Please fix the build issues and try again."
    exit 1
  fi
fi

# Step 5: Upload to PyPI
echo -e "\n===== Step 5: Uploading to PyPI ====="

# Check if there are packages to upload
if [ ! -d "dist" ] || [ -z "$(ls -A dist 2>/dev/null)" ]; then
  echo "Error: No packages found in the dist directory."
  echo "The build process did not create any packages or the dist directory doesn't exist."
  exit 1
fi

echo "The following files will be uploaded to PyPI:"
ls -la dist/

# Ask which PyPI to upload to
echo "\nSelect PyPI repository to upload to:"
echo "1. TestPyPI (recommended for testing)"
echo "2. PyPI (production)"
echo "3. Skip upload (build only)"
read -p "Enter your choice (1/2/3): " PYPI_CHOICE

if [ "$PYPI_CHOICE" = "1" ]; then
  # TestPyPI upload
  REPO_URL="https://test.pypi.org/legacy/"
  REPO_NAME="TestPyPI"
  INSTALL_CMD="pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ triton-metal[metal]"
  
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
      
      # Ask if user wants to verify the installation
      read -p "Do you want to verify the installation from $REPO_NAME? (y/N): " VERIFY_INSTALL
      if [[ "$VERIFY_INSTALL" =~ ^[Yy]$ ]]; then
        echo "\nCreating a temporary virtual environment for testing..."
        python -m venv .venv_test
        source .venv_test/bin/activate
        echo "Installing triton-metal from $REPO_NAME..."
        if eval "$INSTALL_CMD"; then
          echo "Installation successful!"
          echo "Testing import..."
          if python -c "import triton_metal; print('Triton-Metal version:', triton_metal.__version__)"; then
            echo "Import successful!"
          else
            echo "Import failed. There might be issues with the package."
          fi
          deactivate
          echo "Removing test environment..."
          rm -rf .venv_test
        else
          echo "Installation failed. There might be issues with the package."
          deactivate
          rm -rf .venv_test
        fi
      fi
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
  INSTALL_CMD="pip install triton-metal[metal]"
  
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
      
      # Ask if user wants to verify the installation
      read -p "Do you want to verify the installation from $REPO_NAME? (y/N): " VERIFY_INSTALL
      if [[ "$VERIFY_INSTALL" =~ ^[Yy]$ ]]; then
        echo "\nCreating a temporary virtual environment for testing..."
        python -m venv .venv_test
        source .venv_test/bin/activate
        echo "Installing triton-metal from $REPO_NAME..."
        if eval "$INSTALL_CMD"; then
          echo "Installation successful!"
          echo "Testing import..."
          if python -c "import triton_metal; print('Triton-Metal version:', triton_metal.__version__)"; then
            echo "Import successful!"
          else
            echo "Import failed. There might be issues with the package."
          fi
          deactivate
          echo "Removing test environment..."
          rm -rf .venv_test
        else
          echo "Installation failed. There might be issues with the package."
          deactivate
          rm -rf .venv_test
        fi
      fi
      
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
