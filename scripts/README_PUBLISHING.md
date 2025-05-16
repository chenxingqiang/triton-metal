# Publishing triton-metal to PyPI

This document outlines the process for publishing the triton-metal package to PyPI.

## Prerequisites

Before publishing, ensure you have the necessary tools installed:

```bash
pip install build twine
```

You also need an account on PyPI with appropriate permissions for the triton-metal package.

## Publishing Process

1. **Prepare the Release**
   
   Update the version number in the following files:
   - `setup.py` - Update `TRITON_VERSION`
   - `pyproject.toml` - Update `version` field
   
   Ensure all changes are committed and pushed to GitHub.

2. **Use the Publishing Script**

   Run the publishing script from the root directory of the repository:
   
   ```bash
   ./scripts/publish_to_pypi.sh
   ```
   
   The script will:
   - Clean previous builds
   - Build both the wheel and source distribution
   - Verify the packages
   - Prompt you to select whether to upload to TestPyPI or PyPI

3. **Testing the Release on TestPyPI (Recommended)**

   Select option 1 when prompted to upload to TestPyPI first.
   
   After uploading to TestPyPI, test the installation:
   
   ```bash
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ triton-metal[metal]
   ```
   
   Note that the `--extra-index-url` is needed to get dependencies from the regular PyPI.

4. **Publishing to PyPI**

   Once the TestPyPI version is confirmed working, rerun the script and select option 2 to upload to the main PyPI.

## Verifying the Release

After publishing, verify that the package can be installed directly from PyPI:

```bash
pip install triton-metal
pip install "triton-metal[metal]"  # For the full installation with Metal support
```

## Troubleshooting

If you encounter issues with the upload:

1. Check the error messages from Twine
2. Ensure you have the correct credentials configured
3. Verify that the version number has not already been used on PyPI 