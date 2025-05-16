#!/bin/bash
# Script to set up PyPI environment variables securely

# Set TWINE_USERNAME to __token__ for token-based authentication
export TWINE_USERNAME="__token__"

# Prompt for PyPI token (don't hardcode this in scripts)
echo "Please enter your PyPI token when prompted"
read -s TWINE_PASSWORD
export TWINE_PASSWORD

echo "Environment variables set for PyPI upload"
echo "TWINE_USERNAME is set to: $TWINE_USERNAME"
echo "TWINE_PASSWORD is set (hidden)"

# Verify the build before uploading
echo ""
echo "Running package verification..."
python scripts/verify_package_config.py
