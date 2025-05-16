#!/bin/bash
# Post-release tasks script for triton-metal
# This script helps with the tasks that need to be completed after the PyPI release

set -e  # Exit on error

# Print banner
echo "====================================================="
echo "Post-Release Tasks for triton-metal"
echo "====================================================="

# Get the current version from pyproject.toml
VERSION=$(grep -o 'version = "[^"]*"' pyproject.toml | cut -d'"' -f2)
echo "Current version: $VERSION"

# Step 1: Create GitHub release
echo -e "\n===== Step 1: Creating GitHub release ====="
echo "To create a GitHub release, follow these steps:"
echo "1. Go to https://github.com/chengxingqiang/triton-metal/releases/new"
echo "2. Set the tag to 'v$VERSION'"
echo "3. Set the release title to 'Triton-Metal $VERSION'"
echo "4. Include release notes from RELEASE.md"
echo "5. Click 'Publish release'"

read -p "Have you created the GitHub release? (y/N): " GITHUB_RELEASE_DONE
if [[ ! "$GITHUB_RELEASE_DONE" =~ ^[Yy]$ ]]; then
  echo "Please create the GitHub release before continuing."
fi

# Step 2: Update documentation website
echo -e "\n===== Step 2: Updating documentation website ====="
echo "To update the documentation website, follow these steps:"
echo "1. Ensure the documentation reflects the latest release"
echo "2. Update installation instructions if needed"
echo "3. Update any version-specific information"

read -p "Have you updated the documentation website? (y/N): " DOCS_UPDATED
if [[ ! "$DOCS_UPDATED" =~ ^[Yy]$ ]]; then
  echo "Please update the documentation website before continuing."
fi

# Step 3: Plan for next release
echo -e "\n===== Step 3: Planning for next release ====="
echo "To plan for the next release, follow these steps:"
echo "1. Update version numbers for development in setup.py"
echo "2. Create milestone for next version"
echo "3. Plan features and improvements for the next release"

read -p "Do you want to update the version numbers for development now? (y/N): " UPDATE_VERSION
if [[ "$UPDATE_VERSION" =~ ^[Yy]$ ]]; then
  # Prompt for the next development version
  read -p "Enter the next development version (e.g., 3.4.0+metal.dev0): " NEXT_VERSION
  
  if [ -n "$NEXT_VERSION" ]; then
    echo "Updating version in setup.py..."
    sed -i.bak "s/TRITON_VERSION = \"$VERSION\"/TRITON_VERSION = \"$NEXT_VERSION\"/" setup.py
    
    echo "Updating version in pyproject.toml..."
    sed -i.bak "s/version = \"$VERSION\"/version = \"$NEXT_VERSION\"/" pyproject.toml
    
    # Remove backup files
    rm -f setup.py.bak pyproject.toml.bak
    
    echo "Version numbers updated to $NEXT_VERSION for development."
    echo "Remember to commit these changes."
  else
    echo "No version provided. Skipping version update."
  fi
fi

echo -e "\n===== Post-release tasks completed ====="
echo "The PyPI release process is now complete!"
echo "Triton-Metal $VERSION has been released and is available on PyPI."
echo "You can install it with: pip install triton-metal[metal]"
