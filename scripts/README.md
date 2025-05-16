# Triton-Metal Development and Release Scripts

This directory contains scripts to help with the development, testing, and release of the Triton-Metal package.

## Development Scripts

### Local Development Install

`local_dev_install.sh` - Installs Triton-Metal in development mode for quick iteration.

```bash
./scripts/local_dev_install.sh
```

This script:
- Creates a virtual environment in `.venv/`
- Installs Triton-Metal in development mode (`-e` flag)
- Verifies the installation
- Optionally runs a Metal backend test

### Build and Test Wheel

`build_and_test_wheel.sh` - Builds a wheel and tests it in a clean environment without publishing to PyPI.

```bash
./scripts/build_and_test_wheel.sh
```

This script:
- Builds a wheel package
- Creates a test environment in `wheel_test_env/`
- Installs the wheel in the test environment
- Verifies the installation
- Optionally runs a Metal backend test

### Test Metal Backend

`test_metal_kernel.py` - Runs a basic vector addition kernel on the Metal backend.

```bash
python scripts/test_metal_kernel.py
```

This script:
- Initializes the Metal backend
- Defines a simple vector addition kernel
- Runs the kernel and compares the result with NumPy

### LLVM Download Fallback

`download_llvm_fallback.sh` - Downloads LLVM builds directly using curl with SSL verification disabled.

```bash
./scripts/download_llvm_fallback.sh
```

This script:
- Detects your system and architecture
- Downloads the appropriate LLVM package using curl with `-k` flag (SSL verification disabled)
- Extracts it to the correct location
- Sets up the necessary symlinks
- Use this if you encounter SSL errors during the normal installation process

After running this script, you can build triton-metal with:

```bash
LLVM_SYSPATH=$HOME/.triton/llvm/llvm-<hash>-<system-suffix> python -m pip install -e .
```

## Release Scripts

### Publish to PyPI

`publish_to_pypi.sh` - Builds and publishes the package to PyPI.

```bash
./scripts/publish_to_pypi.sh
```

This script:
- Builds the wheel and source distribution
- Verifies the packages
- Prompts you to choose PyPI
- Uploads the packages to the selected repository

## Documentation

### PyPI Release Checklist

`../PYPI_RELEASE_CHECKLIST.md` - A checklist for the PyPI release process.

### Publishing Guide

`README_PUBLISHING.md` - Detailed guide on the publishing process.

## Requirements

These scripts require the following tools:

```bash
pip install build twine numpy
```

## Development Workflow

1. Use `local_dev_install.sh` during active development for quick testing.
2. Use `build_and_test_wheel.sh` to verify the package before release.
3. Use `publish_to_pypi.sh` to publish the package when ready.
4. If you encounter SSL issues during build, use `download_llvm_fallback.sh`.

## Troubleshooting

### SSL Issues

If you encounter SSL errors when building the package (especially on macOS), try:

1. Run the fallback LLVM download script:
   ```bash
   ./scripts/download_llvm_fallback.sh
   ```

2. Build with the downloaded LLVM path:
   ```bash
   LLVM_SYSPATH=$HOME/.triton/llvm/llvm-<hash>-<system-suffix> python -m pip install -e .
   ```

### Git Issues

If you're building from a source distribution and encounter git-related errors, the package should now handle these gracefully. No action is required.

## Notes

- All scripts are designed to be run from the root directory of the repository.
- Virtual environments are created to isolate dependencies.
- The Metal backend test requires NumPy to be installed. 