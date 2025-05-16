#!/usr/bin/env python3
"""
Script to verify the package configuration for triton-metal without building it.
This helps ensure the package is ready for PyPI release.
"""

import os
import sys
import tomli
import re
from pathlib import Path

def check_version_consistency():
    """Check if versions in setup.py and pyproject.toml are consistent."""
    # Get version from setup.py
    setup_py_path = Path("setup.py")
    setup_py_content = setup_py_path.read_text()
    setup_py_version_match = re.search(r'TRITON_VERSION\s*=\s*["\']([^"\']+)["\']', setup_py_content)
    
    if not setup_py_version_match:
        print("❌ Could not find TRITON_VERSION in setup.py")
        return False
    
    setup_py_version = setup_py_version_match.group(1)
    
    # Get version from pyproject.toml
    pyproject_path = Path("pyproject.toml")
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomli.load(f)
    
    pyproject_version = pyproject_data.get("project", {}).get("version")
    
    if not pyproject_version:
        print("❌ Could not find version in pyproject.toml")
        return False
    
    # Compare versions
    if setup_py_version == pyproject_version:
        print(f"✅ Version consistency: setup.py and pyproject.toml both have version {setup_py_version}")
        return True
    else:
        print(f"❌ Version mismatch: setup.py has {setup_py_version}, pyproject.toml has {pyproject_version}")
        return False

def check_manifest_includes_licenses():
    """Check if MANIFEST.in includes license files."""
    manifest_path = Path("MANIFEST.in")
    manifest_content = manifest_path.read_text()
    
    has_license = "include LICENSE" in manifest_content
    has_metal_license = "include LICENSE.metal" in manifest_content
    
    if has_license and has_metal_license:
        print("✅ MANIFEST.in includes LICENSE and LICENSE.metal")
        return True
    else:
        missing = []
        if not has_license:
            missing.append("LICENSE")
        if not has_metal_license:
            missing.append("LICENSE.metal")
        print(f"❌ MANIFEST.in is missing: {', '.join(missing)}")
        return False

def check_readme_has_pypi_instructions():
    """Check if README.md has PyPI installation instructions."""
    readme_path = Path("README.md")
    readme_content = readme_path.read_text()
    
    has_pip_install = "pip install triton-metal" in readme_content
    has_extras = "pip install \"triton-metal[metal]\"" in readme_content
    
    if has_pip_install and has_extras:
        print("✅ README.md includes PyPI installation instructions")
        return True
    else:
        missing = []
        if not has_pip_install:
            missing.append("basic pip install command")
        if not has_extras:
            missing.append("pip install with metal extras")
        print(f"❌ README.md is missing: {', '.join(missing)}")
        return False

def check_release_has_metal_info():
    """Check if RELEASE.md has Metal backend information."""
    release_path = Path("RELEASE.md")
    release_content = release_path.read_text()
    
    has_metal_section = "Metal Backend" in release_content
    
    if has_metal_section:
        print("✅ RELEASE.md includes Metal backend information")
        return True
    else:
        print("❌ RELEASE.md is missing Metal backend information")
        return False

def check_python_version_requirements():
    """Check Python version requirements."""
    pyproject_path = Path("pyproject.toml")
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomli.load(f)
    
    requires_python = pyproject_data.get("project", {}).get("requires-python")
    
    if requires_python:
        print(f"✅ Python version requirements: {requires_python}")
        return True
    else:
        print("❌ Missing Python version requirements in pyproject.toml")
        return False

def main():
    """Main function to run all checks."""
    print("=== Verifying triton-metal package configuration ===")
    
    checks = [
        check_version_consistency,
        check_manifest_includes_licenses,
        check_readme_has_pypi_instructions,
        check_release_has_metal_info,
        check_python_version_requirements
    ]
    
    results = [check() for check in checks]
    
    print("\n=== Summary ===")
    if all(results):
        print("✅ All checks passed! The package configuration is ready for PyPI release.")
        return 0
    else:
        print(f"❌ {results.count(False)} check(s) failed. Please fix the issues before proceeding with the PyPI release.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
