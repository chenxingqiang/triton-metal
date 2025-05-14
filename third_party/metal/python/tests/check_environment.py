#!/usr/bin/env python
"""Environment check for Metal backend tests.

This script checks that the environment is correctly set up for running
Metal backend tests, including required dependencies and hardware.
"""

import os
import sys
import platform
import importlib.util
from typing import Dict, List, Tuple

def check_python_version() -> bool:
    """Check Python version"""
    required_version = (3, 8)
    current_version = sys.version_info[:2]
    
    print(f"Python version: {'.'.join(map(str, current_version))}")
    
    if current_version >= required_version:
        print("✅ Python version is sufficient")
        return True
    else:
        print(f"❌ Python version is too old. Required: {'.'.join(map(str, required_version))}")
        return False

def check_package(package_name: str) -> bool:
    """Check if a package is installed"""
    spec = importlib.util.find_spec(package_name)
    if spec is not None:
        try:
            module = importlib.import_module(package_name)
            if hasattr(module, '__version__'):
                print(f"✅ {package_name} is installed (version {module.__version__})")
            else:
                print(f"✅ {package_name} is installed")
            return True
        except ImportError:
            print(f"❌ {package_name} is installed but could not be imported")
            return False
    else:
        print(f"❌ {package_name} is not installed")
        return False

def check_operating_system() -> bool:
    """Check operating system"""
    system = platform.system()
    version = platform.version()
    machine = platform.machine()
    
    print(f"Operating system: {system} {version} ({machine})")
    
    if system == "Darwin" and machine in ["arm64", "arm"]:
        print("✅ Running on Apple Silicon Mac")
        return True
    else:
        print("❌ Not running on Apple Silicon Mac")
        return False

def check_metal_hardware() -> bool:
    """Check Metal hardware availability"""
    # Try to import PyMetal or MLX to check Metal support
    try:
        import mlx.core as mx
        device = mx.get_default_device()
        device_str = str(device)
        
        print(f"Default device: {device_str}")
        
        if "apple" in device_str.lower():
            print("✅ Metal hardware is available")
            return True
        else:
            print("❌ Metal hardware is not available")
            return False
    except ImportError:
        print("⚠️ Could not check Metal hardware (MLX not installed)")
        return False
    except Exception as e:
        print(f"❌ Error checking Metal hardware: {e}")
        return False

def check_metal_backend() -> bool:
    """Check Metal backend installation"""
    # Check for Metal backend modules
    metal_modules = [
        "metal_backend",
        "triton_to_metal_converter"
    ]
    
    # First try importing directly
    all_found = True
    for module_name in metal_modules:
        if not check_package(module_name):
            all_found = False
    
    # If not found, check if they're within a parent package
    if not all_found:
        print("Attempting to find Metal modules in parent package...")
        
        # Add parent directory to path for imports
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        all_found = True
        for module_name in metal_modules:
            if not check_package(module_name):
                all_found = False
    
    return all_found

def main():
    """Main function"""
    print("=== Metal Backend Environment Check ===\n")
    
    # Check Python version
    python_ok = check_python_version()
    
    # Check required packages
    print("\n=== Required Packages ===")
    numpy_ok = check_package("numpy")
    matplotlib_ok = check_package("matplotlib")
    mlx_ok = check_package("mlx.core")
    
    # Check operating system
    print("\n=== System Information ===")
    os_ok = check_operating_system()
    
    # Check Metal hardware
    metal_hw_ok = check_metal_hardware()
    
    # Check Metal backend
    print("\n=== Metal Backend ===")
    metal_backend_ok = check_metal_backend()
    
    # Print summary
    print("\n=== Summary ===")
    num_passed = sum(1 for x in [python_ok, numpy_ok, matplotlib_ok, mlx_ok, os_ok, metal_hw_ok, metal_backend_ok] if x)
    num_total = 7
    
    print(f"Passed: {num_passed}/{num_total} checks")
    
    if num_passed == num_total:
        print("✅ Environment is correctly set up for Metal backend tests")
        return 0
    else:
        print("❌ Environment is not correctly set up for Metal backend tests")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 