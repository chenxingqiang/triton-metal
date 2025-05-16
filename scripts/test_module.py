#!/usr/bin/env python3
"""
Simple test script to verify that the Triton-Metal package can be imported
This is useful for confirming that a build is functional before publishing
"""

import os
import sys
import importlib.util

def check_module(module_name):
    """Check if a module can be imported and return its location"""
    spec = importlib.util.find_spec(module_name)
    if spec is not None:
        print(f"✅ {module_name} is available at: {spec.origin}")
        return True
    else:
        print(f"❌ {module_name} is NOT available!")
        return False

def check_backend(backend_name):
    """Check if a backend can be loaded and is recognized by the driver"""
    os.environ["TRITON_BACKEND"] = backend_name
    try:
        import triton_metal
        print(f"✅ Backend {backend_name} loaded successfully")
        print(f"   Triton version: {triton_metal.__version__}")
        
        # Check if runtime driver can recognize the backend
        from triton_metal.runtime import driver
        available_backends = driver.get_available_backends()
        print(f"   Available backends: {available_backends}")
        
        if backend_name in available_backends:
            print(f"✅ Backend {backend_name} is recognized by driver")
        else:
            print(f"❌ Backend {backend_name} is NOT recognized by driver")
            return False
            
        return True
    except Exception as e:
        print(f"❌ Failed to load backend {backend_name}: {e}")
        return False

def main():
    """Main test function"""
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.executable}")
    print(f"Current directory: {os.getcwd()}")
    print("\n--- Checking for required modules ---")
    
    modules_to_check = [
        "triton_metal",
        "triton_metal.language",
        "triton_metal.runtime",
        "triton_metal.backends.metal"
    ]
    
    all_modules_available = True
    for module in modules_to_check:
        result = check_module(module)
        all_modules_available = all_modules_available and result
    
    print("\n--- Checking Metal backend ---")
    metal_available = check_backend("metal")
    
    if all_modules_available and metal_available:
        print("\n✅ All tests passed! The package appears to be working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 