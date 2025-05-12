#!/usr/bin/env python3
"""
Triton Metal Backend 安装脚本
"""

import os
import platform
import sys
from setuptools import setup, find_packages

# 检查是否在macOS上运行
if not platform.system() == "Darwin":
    print("错误：Metal后端只能在macOS上安装")
    sys.exit(1)

# 检查是否为Apple Silicon
if platform.processor() != "arm":
    print("警告：Metal后端设计用于Apple Silicon芯片，在其他平台上可能无法正常工作")

# 检查MLX可用性
try:
    import mlx.core
    if not mlx.core.metal.is_available():
        print("警告：MLX Metal不可用，后端将无法正常工作")
except ImportError:
    print("错误：未找到MLX库，请先安装MLX: pip install mlx")
    sys.exit(1)

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="triton-metal",
    version="0.1.0",
    description="Triton Metal Backend for Apple Silicon",
    author="Triton Contributors",
    author_email="triton-dev@example.com",
    url="https://github.com/triton-lang/triton",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "mlx>=0.25.0",
    ],
    entry_points={
        "triton.backends": [
            "metal = metal.backend",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: MacOS",
    ],
) 