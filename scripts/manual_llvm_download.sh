#!/bin/bash
set -e

# Script to manually download LLVM and prepare for triton-metal build
# This is a workaround for SSL issues

# Get the LLVM hash from the cmake file
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

# Create the triton cache directory if it doesn't exist
TRITON_CACHE_DIR="$HOME/.triton/cache"
TRITON_LLVM_DIR="$HOME/.triton/llvm"
mkdir -p "$TRITON_CACHE_DIR"
mkdir -p "$TRITON_LLVM_DIR"

# Set the target paths
TARBALL_NAME="llvm-${LLVM_SHORT_HASH}-${SYSTEM_SUFFIX}.tar.gz"
DOWNLOAD_PATH="$TRITON_CACHE_DIR/$TARBALL_NAME"
EXTRACT_DIR="$TRITON_LLVM_DIR/llvm-${LLVM_SHORT_HASH}-${SYSTEM_SUFFIX}"

echo "Target tarball: $TARBALL_NAME"
echo "Download path: $DOWNLOAD_PATH"
echo "Extract directory: $EXTRACT_DIR"

# Check if the tarball already exists
if [ -f "$DOWNLOAD_PATH" ]; then
  echo "Tarball already exists at $DOWNLOAD_PATH"
else
  echo "Attempting to download LLVM tarball..."
  # Try different download methods with SSL workarounds
  
  # Method 1: curl with insecure flag
  echo "Trying curl with --insecure flag..."
  if curl --insecure -L -o "$DOWNLOAD_PATH.tmp" "https://oaitriton_metal.blob.core.windows.net/public/llvm-builds/$TARBALL_NAME"; then
    mv "$DOWNLOAD_PATH.tmp" "$DOWNLOAD_PATH"
    echo "Download successful using curl"
  else
    echo "Curl download failed"
    
    # Method 2: wget with no certificate check
    echo "Trying wget with --no-check-certificate..."
    if wget --no-check-certificate -O "$DOWNLOAD_PATH.tmp" "https://oaitriton_metal.blob.core.windows.net/public/llvm-builds/$TARBALL_NAME"; then
      mv "$DOWNLOAD_PATH.tmp" "$DOWNLOAD_PATH"
      echo "Download successful using wget"
    else
      echo "Wget download failed"
      echo "All download methods failed. Please download the LLVM tarball manually."
      exit 1
    fi
  fi
fi

# Extract the tarball if the directory doesn't exist
if [ -d "$EXTRACT_DIR" ]; then
  echo "LLVM already extracted at $EXTRACT_DIR"
else
  echo "Extracting LLVM tarball to $EXTRACT_DIR..."
  mkdir -p "$EXTRACT_DIR"
  tar -xzf "$DOWNLOAD_PATH" -C "$EXTRACT_DIR" --strip-components=1
  echo "Extraction complete"
fi

echo ""
echo "LLVM is ready for use. Build triton-metal with:"
echo "LLVM_SYSPATH=$EXTRACT_DIR python -m build"
echo "or"
echo "LLVM_SYSPATH=$EXTRACT_DIR pip install -e ."
echo ""
