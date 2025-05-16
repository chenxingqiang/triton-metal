#!/usr/bin/env python3
"""
Simple verification script to check that the Triton-Metal package 
is installed correctly and can be imported.
"""

import sys
import importlib.util

def check_module(module_name):
    """Check if a module can be imported and print its version if available"""
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"✅ {module_name} is available (version: {version})")
        return True
    except ImportError as e:
        print(f"❌ {module_name} could not be imported: {e}")
        return False

def main():
    """Main verification function"""
    print(f"Python version: {sys.version}")
    
    # Check that the main package can be imported
    if not check_module("triton_metal"):
        print("❌ Failed to import triton_metal package")
        return 1

    # Try to import some submodules
    modules_to_check = [
        "triton_metal.language",
        "triton_metal.runtime",
        "triton_metal.backends.metal"
    ]
    
    success = True
    for module in modules_to_check:
        if not check_module(module):
            success = False
    
    if success:
        print("\n✅ Triton-Metal package is correctly installed!")
        return 0
    else:
        print("\n❌ There were issues with some submodules.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 