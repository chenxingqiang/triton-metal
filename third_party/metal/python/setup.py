from setuptools import setup, find_packages

setup(
    name="triton-metal",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "triton.backends": [
            "metal = triton.backends.metal",
        ],
    },
    install_requires=[
        "mlx>=0.3.0",
    ],
    author="Triton Team",
    author_email="triton-team@example.com",
    description="Metal backend for Triton",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 