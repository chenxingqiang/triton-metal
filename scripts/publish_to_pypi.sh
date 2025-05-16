#!/bin/bash
# Script to publish the triton-metal package to PyPI
# This script builds the package and uploads it to PyPI

set -e  # Exit on error

MODE="all"  # Default to build and upload

# Check if a mode is specified
if [ "$1" == "build" ]; then
    MODE="build"
    echo "Running in build-only mode"
elif [ "$1" == "upload" ]; then
    MODE="upload"
    echo "Running in upload-only mode"
elif [ "$1" != "" ]; then
    echo "Unknown mode: $1"
    echo "Usage: $0 [build|upload]"
    exit 1
fi

# Print banner
echo "====================================================="
echo "Publishing triton-metal package to PyPI"
echo "====================================================="

# Check if twine is installed
if ! command -v twine &> /dev/null; then
    echo "Error: twine is not installed. Please install it with:"
    echo "pip install twine"
    exit 1
fi

# Print environment information
echo "Python version: $(python --version)"
echo "Current directory: $(pwd)"

# Check for TWINE_USERNAME and TWINE_PASSWORD environment variables
if ([ -z "$TWINE_USERNAME" ] || [ -z "$TWINE_PASSWORD" ]) && [ "$MODE" != "build" ]; then
    echo "Warning: TWINE_USERNAME and/or TWINE_PASSWORD environment variables not set."
    echo "You will be prompted for credentials when uploading to PyPI."
    echo "To avoid this, set them before running this script:"
    echo "export TWINE_USERNAME=__token__"
    echo "export TWINE_PASSWORD=your-token-here"
fi

if [ "$MODE" == "build" ] || [ "$MODE" == "all" ]; then
    # Clean up any existing build artifacts
    echo -e "\n===== Cleaning up build artifacts ====="
    rm -rf dist/ build/ *.egg-info/

    # Set version variables
    # Default to no suffix, override with env var if needed
    TRITON_WHEEL_VERSION_SUFFIX="${TRITON_WHEEL_VERSION_SUFFIX:-}"
    echo "Using version suffix: ${TRITON_WHEEL_VERSION_SUFFIX:-(none)}"

    # Build the package
    echo -e "\n===== Building package ====="
    python -m build

    # List the built distributions
    echo -e "\n===== Built distributions ====="
    ls -la dist/
fi

if [ "$MODE" == "upload" ] || [ "$MODE" == "all" ]; then
    # Verify wheel is built
    if [ ! -d "dist" ] || [ -z "$(ls -A dist)" ]; then
        echo "Error: No distribution found in dist/ directory. Run in build mode first."
        exit 1
    fi

    # Upload to PyPI
    echo -e "\n===== Uploading to PyPI ====="
    if [ -z "$DRY_RUN" ]; then
        twine upload dist/*
    else
        echo "DRY RUN: Would upload the following files to PyPI:"
        ls -la dist/
        echo "To actually upload, run without DRY_RUN environment variable"
    fi

    echo -e "\n===== Package published successfully ====="
    echo "To install the published package:"
    echo "pip install triton-metal" 