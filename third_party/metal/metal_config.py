#!/usr/bin/env python3
# Copyright (c) 2024, Cheng Xingqiang. All rights reserved.
#
# Metal backend configuration

import os
import platform
import sys

# Metal backend version
METAL_BACKEND_VERSION = "1.0.0"

# Repository information
REPO_AUTHOR = "Cheng Xingqiang"
REPO_URL = "https://github.com/chengxingqiang/triton-metal"

# Metal backend requirements
MIN_MACOS_VERSION = (13, 5)  # macOS 13.5 or higher
MIN_MLX_VERSION = "0.3.0"    # MLX 0.3.0 or higher

# M3-specific optimizations
M3_OPTIMIZATIONS = {
    "shared_memory_size": 65536,    # 64KB shared memory
    "vector_width": 8,              # 8-wide vectorization
    "simdgroup_width": 32,          # 32-wide SIMD groups
    "tensor_cores": True,           # Tensor core support
    "dynamic_caching": True,        # Dynamic register caching
}

# Check if current hardware supports Metal backend
def is_metal_supported():
    """Check if the current hardware supports Metal backend"""
    # Check if running on macOS
    if platform.system() != "Darwin":
        return False
        
    # Check if running on Apple Silicon
    if platform.machine() not in ["arm64", "aarch64"]:
        return False
        
    # Check macOS version
    mac_ver = platform.mac_ver()[0].split(".")
    current_version = tuple(int(v) for v in mac_ver[:2])
    if current_version < MIN_MACOS_VERSION:
        return False
        
    # Check if MLX is available
    try:
        import mlx.core
        return True
    except ImportError:
        return False
        
    return True

# Get chip generation
def get_chip_generation():
    """Detect Apple Silicon chip generation"""
    if not is_metal_supported():
        return None
        
    try:
        # This is a simple heuristic - would need to be improved for actual production use
        import subprocess
        output = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        
        if "M3" in output:
            return "M3"
        elif "M2" in output:
            return "M2"
        elif "M1" in output:
            return "M1"
        else:
            return "Unknown"
    except:
        return "Unknown"

# Print configuration information
if __name__ == "__main__":
    print(f"Metal Backend Configuration by {REPO_AUTHOR}")
    print(f"Version: {METAL_BACKEND_VERSION}")
    print(f"Repository: {REPO_URL}")
    print(f"Metal supported: {is_metal_supported()}")
    print(f"Chip generation: {get_chip_generation()}")
    
    if get_chip_generation() == "M3":
        print("M3-specific optimizations available:")
        for key, value in M3_OPTIMIZATIONS.items():
            print(f"  - {key}: {value}") 