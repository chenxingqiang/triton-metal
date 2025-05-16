# PyPI Release Checklist for triton-metal

## Pre-release Tasks
- [ ] Update version numbers:
  - [ ] `setup.py`: Set `TRITON_VERSION = "3.3.0+metal"` (removed git hash suffix)
  - [ ] `pyproject.toml`: Verify version is set to `3.3.0+metal` 
- [ ] Update documentation:
  - [ ] `RELEASE.md`: Add Metal backend support information
  - [ ] `README.md`: Update with PyPI installation instructions
- [ ] Verify package configuration:
  - [ ] Author and email in `setup.py` and `pyproject.toml`
  - [ ] License files are included
  - [ ] Proper Python version requirements
  - [ ] Correct dependencies listed
  - [ ] README.md is properly formatted
- [ ] Test the build locally (choose one approach):
  - [ ] Option 1: Development install:
    - [ ] Run: `./scripts/local_dev_install.sh`
    - [ ] This installs in development mode (-e) for quick testing
  - [ ] Option 2: Test wheel locally:
    - [ ] Run: `./scripts/build_and_test_wheel.sh`
    - [ ] This builds a wheel and installs it in a test environment
  - [ ] Option 3: Manual build check:
    - [ ] Clean previous builds: `rm -rf build/ dist/ *.egg-info/`
    - [ ] Build the package: `python -m build`
    - [ ] Verify built packages: `twine check dist/*`

## Release Tasks
- [ ] Option A: Upload to TestPyPI first (if you have an account):
  - [ ] Run: `./scripts/publish_to_pypi.sh` and select option 1
  - [ ] Verify installation from TestPyPI:
    ```
    pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ triton-metal[metal]
    ```
  - [ ] Test basic functionality

- [ ] Option B: Skip TestPyPI and proceed with local testing:
  - [ ] Test in multiple environments using `./scripts/build_and_test_wheel.sh`
  - [ ] If possible, test on both Intel and Apple Silicon macOS systems

- [ ] Upload to PyPI:
  - [ ] Run: `./scripts/publish_to_pypi.sh` and select option 2
  - [ ] Verify installation from PyPI:
    ```
    pip install triton-metal[metal]
    ```
  - [ ] Test basic functionality

## Post-release Tasks
- [ ] Create a GitHub release:
  - [ ] Tag: `v3.3.0+metal`
  - [ ] Release title: "Triton-Metal 3.3.0+metal"
  - [ ] Include release notes from RELEASE.md
- [ ] Announce the release:
  - [ ] Update documentation website
  - [ ] Share on relevant channels
- [ ] Plan for next release:
  - [ ] Update version numbers for development (`TRITON_VERSION` in setup.py)
  - [ ] Create milestone for next version

## Troubleshooting

### SSL Issues
- [ ] If you encounter SSL errors during the build:
  - [ ] Run the LLVM download fallback script: `./scripts/download_llvm_fallback.sh`
  - [ ] Then build with the downloaded LLVM: `LLVM_SYSPATH=$HOME/.triton/llvm/llvm-<hash>-<system-suffix> python -m build`

### Git Issues
- [ ] If building from a source distribution (sdist) and seeing git errors:
  - [ ] The package will handle this automatically (no action needed)
  - [ ] If issues persist, set environment variables: `TRITON_WHEEL_VERSION_SUFFIX=""`

## Notes
- Remember to use `python -m build` instead of `python setup.py bdist_wheel` for building
- TestPyPI and PyPI require separate accounts and API tokens
- Make sure you have configured `~/.pypirc` correctly if using PyPI/TestPyPI
- For macOS builds, ensure you have tested on both Intel and Apple Silicon
- Using the provided scripts in the `scripts/` directory can streamline the process 