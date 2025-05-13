#!/usr/bin/env python3
"""
Triton Metal Backend 安装脚本
"""

import os
import platform
import sys
from setuptools import setup, find_packages

# Check if running on macOS
is_macos = platform.system() == "Darwin"
is_apple_silicon = is_macos and platform.machine() == "arm64"

# Warn if not running on Apple Silicon Mac
if not is_apple_silicon:
    print("WARNING: This package is designed for Apple Silicon Macs and may not work on other platforms.")

# MLX is only available on macOS with Apple Silicon
mlx_requires = []
if is_apple_silicon:
    mlx_requires = ["mlx>=0.2.0"]

# Determine Triton path
triton_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(triton_path, "python"))

setup(
    name="triton-metal",
    version="0.1.0",
    description="Triton Metal backend for Apple Silicon",
    author="Triton Metal Team",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
    ] + mlx_requires,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-xdist>=2.5.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: MacOS :: MacOS X",
    ],
) 