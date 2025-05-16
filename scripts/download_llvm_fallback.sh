#!/bin/bash
set -e

# Script to manually download LLVM builds as a fallback when the standard download method fails
# Especially useful for macOS environments with SSL issues

# Add diagnostic information
echo "Starting LLVM fallback download script"
echo "Current directory: $(pwd)"

# Parse arguments - both URL and destination or just system info
URL=$1
DEST_FILE=$2

# Determine system and architecture if not provided in URL
SYSTEM=$(uname -s)
ARCH=$(uname -m)
echo "Detected system: $SYSTEM, architecture: $ARCH"

if [ -z "$URL" ]; then
  # No URL provided, try to construct it
  if [ "$SYSTEM" == "Darwin" ]; then
    if [ "$ARCH" == "arm64" ]; then
      SYSTEM_SUFFIX="macos-arm64"
      echo "Detected Apple Silicon (ARM64)"
    else
      SYSTEM_SUFFIX="macos-x64"
      echo "Detected macOS (x86_64)"
    fi
  elif [ "$SYSTEM" == "Linux" ]; then
    if [ "$ARCH" == "aarch64" ] || [ "$ARCH" == "arm64" ]; then
      SYSTEM_SUFFIX="ubuntu-arm64"
      echo "Detected Linux (ARM64)"
    else
      SYSTEM_SUFFIX="ubuntu-x64" 
      echo "Detected Linux (x86_64)"
    fi
  else
    echo "Unsupported system: $SYSTEM"
    exit 1
  fi

  # Get LLVM hash from the cmake file
  LLVM_HASH_PATH="cmake/llvm-hash.txt"
  if [ -f "$LLVM_HASH_PATH" ]; then
    LLVM_HASH=$(cat "$LLVM_HASH_PATH" | head -c 8)
    echo "Using LLVM hash: $LLVM_HASH"
  else
    echo "Could not find LLVM hash file"
    exit 1
  fi

  URL="https://oaitriton_metal.blob.core.windows.net/public/llvm-builds/llvm-${LLVM_HASH}-${SYSTEM_SUFFIX}.tar.gz"
  FILENAME="llvm-${LLVM_HASH}-${SYSTEM_SUFFIX}.tar.gz"
else
  # Extract filename from URL if not provided
  FILENAME=$(basename "$URL")
fi

# Set default destination if not provided
if [ -z "$DEST_FILE" ]; then
  # Create cache directory if not exists
  CACHE_DIR="${HOME}/.triton/cache"
  mkdir -p "$CACHE_DIR"
  DEST_FILE="${CACHE_DIR}/${FILENAME}"
fi

echo "URL: $URL"
echo "Destination: $DEST_FILE"

# Create parent directory if it doesn't exist
mkdir -p "$(dirname "$DEST_FILE")"

# Try different download methods
download_success=false

# Method 1: using curl with insecure flag
echo "Trying curl with --insecure flag..."
if curl -L --insecure -o "$DEST_FILE" "$URL"; then
  download_success=true
  echo "Download successful using curl with --insecure"
else
  echo "Curl with --insecure failed"
fi

# Method 2: using wget with no-check-certificate if available and previous method failed
if [ "$download_success" = false ] && command -v wget &>/dev/null; then
  echo "Trying wget with --no-check-certificate..."
  if wget --no-check-certificate -O "$DEST_FILE" "$URL"; then
    download_success=true
    echo "Download successful using wget with --no-check-certificate"
  else
    echo "Wget with --no-check-certificate failed"
  fi
fi

# Method 3: using python if previous methods failed
if [ "$download_success" = false ]; then
  echo "Trying Python urllib..."
  python -c "
import urllib.request
import ssl
import sys
try:
    context = ssl._create_unverified_context()
    urllib.request.urlretrieve('$URL', '$DEST_FILE')
    print('Download successful using Python')
    sys.exit(0)
except Exception as e:
    print(f'Python download failed: {e}')
    sys.exit(1)
  "
  if [ $? -eq 0 ]; then
    download_success=true
  else
    echo "Python download failed"
  fi
fi

# Check if download was successful
if [ "$download_success" = true ]; then
  echo "Download completed successfully to: $DEST_FILE"
  exit 0
else
  echo "All download methods failed for $URL"
  exit 1
fi 