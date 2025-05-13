#!/bin/bash

# Check if running on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "Error: This script can only be run on macOS."
    exit 1
fi

# Check if running on Apple Silicon
if [[ "$(uname -m)" != "arm64" ]]; then
    echo "Warning: This backend is designed for Apple Silicon Macs and may not work on Intel Macs."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check macOS version
MACOS_VERSION=$(sw_vers -productVersion)
MACOS_MAJOR=$(echo $MACOS_VERSION | cut -d. -f1)
MACOS_MINOR=$(echo $MACOS_VERSION | cut -d. -f2)

if [[ $MACOS_MAJOR -lt 13 ]]; then
    echo "Warning: The Metal backend is designed for macOS 13.0+ and may not work properly on older versions."
    echo "Your macOS version: $MACOS_VERSION"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    
    # For development environments on earlier macOS versions, set force enable
    export TRITON_METAL_FORCE_ENABLE=1
fi

# Get the directory of this script
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRITON_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

# Create a Python virtual environment (optional)
if [[ "$1" == "--venv" ]]; then
    echo "Creating Python virtual environment..."
    python -m venv "$SCRIPT_DIR/venv"
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# Install MLX and other dependencies
if [[ "$(uname -m)" == "arm64" ]]; then
    echo "Installing MLX..."
    pip install mlx>=0.3.0
    
    # Check if installation was successful
    if ! python -c "import mlx.core" &> /dev/null; then
        echo "Failed to install MLX. Please ensure you have the latest version of Python and pip."
        exit 1
    fi
    
    echo "MLX installed successfully!"
fi

# Install other dependencies
echo "Installing additional dependencies..."
pip install numpy>=1.20.0

# Install the Metal backend
echo "Installing Triton Metal backend..."
cd "$SCRIPT_DIR"
pip install -e .

# Set up environment variables
echo "Setting up environment variables..."
export TRITON_METAL_ROOT="$SCRIPT_DIR"
export PYTHONPATH="$TRITON_ROOT/python:$PYTHONPATH"

# Create a sample .env file for future use
cat > "$SCRIPT_DIR/.env" << EOL
# Triton Metal backend environment variables
TRITON_METAL_ROOT="$SCRIPT_DIR"
PYTHONPATH="$TRITON_ROOT/python:\$PYTHONPATH"
# Set to 1 to force enable the Metal backend on older macOS versions
#TRITON_METAL_FORCE_ENABLE=1
# Set to 1 to enable debug logging
#TRITON_METAL_DEBUG=1
# Set to use Metal as the default Triton backend
TRITON_BACKEND="metal"
EOL

# Install the Metal backend as a Triton plugin
echo "Installing Metal backend as a Triton plugin..."
mkdir -p "$TRITON_ROOT/python/triton/backends/plugins"
if [[ ! -e "$TRITON_ROOT/python/triton/backends/plugins/metal" ]]; then
    ln -s "$SCRIPT_DIR" "$TRITON_ROOT/python/triton/backends/plugins/metal"
fi

# Run tests (optional)
if [[ "$1" == "--test" || "$2" == "--test" ]]; then
    echo "Running tests..."
    cd "$SCRIPT_DIR/python"
    python -m test_metal_backend
fi

# Print success message
echo "Triton Metal backend installed successfully!"
echo
echo "To use the Metal backend, add the following to your ~/.bashrc or ~/.zshrc:"
echo "  export TRITON_METAL_ROOT=\"$SCRIPT_DIR\""
echo "  export PYTHONPATH=\"$TRITON_ROOT/python:\$PYTHONPATH\""
echo "  export TRITON_BACKEND=\"metal\""
echo
echo "Or source the provided .env file:"
echo "  source $SCRIPT_DIR/.env"
echo
echo "To run the example and verify your installation:"
echo "  python $SCRIPT_DIR/python/examples/metal_backend_demo.py"
echo
echo "To force enable Metal backend on older macOS versions:"
echo "  export TRITON_METAL_FORCE_ENABLE=1" 