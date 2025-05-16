#!/bin/bash
# Script to publish triton-metal to PyPI using the token provided
# This script is a secure wrapper around publish_to_pypi.sh

# Set token environment variables
export TWINE_USERNAME="__token__"
export TWINE_PASSWORD="pypi-AgEIcHlwaS5vcmcCJGVjNDQ2MWU1LWJjNWItNGU4Yi05MzM5LTg1MTYwN2YwMWQxMAACKlszLCI1NTE1MDVlZS03ZjFhLTQ1YjctYjJlMC0zMjM2ZmQzMTk3NWUiXQAABiANZja89HCH2ememEnbvARnySz3YB-_-YTnDdoFCHefYg"

# Check for flags
DRY_RUN=0
SKIP_TEST=0

for arg in "$@"; do
    case $arg in
        --dry-run)
            echo "Running in dry run mode - will not actually upload to PyPI"
            DRY_RUN=1
            ;;
        --skip-test)
            echo "Skipping test step"
            SKIP_TEST=1
            ;;
    esac
done

# Build the package first
./scripts/publish_to_pypi.sh build

# Test installation if not skipped
if [ $SKIP_TEST -eq 0 ]; then
    echo -e "\n===== Testing package installation and functionality ====="
    
    # Create a temporary directory for testing
    TEST_DIR=$(mktemp -d)
    echo "Testing in temporary directory: $TEST_DIR"
    
    # Copy the wheel to the test directory
    cp dist/*.whl $TEST_DIR/
    
    # Navigate to test directory 
    pushd $TEST_DIR > /dev/null
    
    # Create a virtual environment for testing
    python -m venv test_env
    source test_env/bin/activate
    
    # Install the wheel
    pip install *.whl
    
    # Copy the test script
    cp $OLDPWD/scripts/test_module.py .
    
    # Run the test script
    python test_module.py
    TEST_RESULT=$?
    
    # Deactivate the virtual environment
    deactivate
    
    # Return to original directory
    popd > /dev/null
    
    # Clean up
    rm -rf $TEST_DIR
    
    # Check test result
    if [ $TEST_RESULT -ne 0 ]; then
        echo "Tests failed - aborting upload"
        exit 1
    fi
    
    echo "Tests passed - proceeding with upload"
fi

if [ $DRY_RUN -eq 1 ]; then
    export DRY_RUN=1
    # Call the main publish script in upload mode
    ./scripts/publish_to_pypi.sh upload
else
    # Call the main publish script in upload mode
    ./scripts/publish_to_pypi.sh upload
fi

# Clear the environment variables after use for security
unset TWINE_USERNAME
unset TWINE_PASSWORD 