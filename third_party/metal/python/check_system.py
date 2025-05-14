#!/usr/bin/env python
"""
Metal Backend System Compatibility Check

This script checks if the current system is compatible with the Triton Metal backend,
including hardware detection, macOS version check, and dependency verification.
"""

import os
import sys
import platform
import subprocess
import pkg_resources
import importlib.util
from enum import Enum
from typing import Dict, List, Tuple, Optional

class AppleSiliconGeneration(Enum):
    """Enum for Apple Silicon generation"""
    UNKNOWN = 0
    M1 = 1
    M2 = 2
    M3 = 3

class CheckResult:
    """Result of a compatibility check"""
    def __init__(self, name: str, passed: bool, message: str, critical: bool = False):
        self.name = name
        self.passed = passed
        self.message = message
        self.critical = critical  # If True, this is a requirement that must be met

def check_macos_version() -> CheckResult:
    """Check if macOS version is compatible"""
    if platform.system() != "Darwin":
        return CheckResult(
            "macOS", False, 
            "Not running on macOS. The Metal backend requires macOS.", 
            critical=True
        )
    
    version = platform.mac_ver()[0]
    major, minor, patch = map(int, version.split('.'))
    
    if (major > 13) or (major == 13 and minor >= 5):
        return CheckResult(
            "macOS Version", True,
            f"macOS version {version} is compatible (>= 13.5 required)."
        )
    else:
        return CheckResult(
            "macOS Version", False,
            f"macOS version {version} is not compatible. Version 13.5 or newer is required.",
            critical=True
        )

def check_apple_silicon() -> CheckResult:
    """Check if running on Apple Silicon"""
    if platform.system() != "Darwin":
        return CheckResult(
            "Apple Silicon", False,
            "Not running on macOS, cannot check for Apple Silicon.",
            critical=True
        )
    
    # Check processor
    processor = platform.processor()
    if processor == "arm":
        # Try to determine which chip generation
        try:
            # Use sysctl to get chip info
            chip_info = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
            
            if "M1" in chip_info:
                gen = AppleSiliconGeneration.M1
                gen_name = "M1"
            elif "M2" in chip_info:
                gen = AppleSiliconGeneration.M2
                gen_name = "M2"
            elif "M3" in chip_info:
                gen = AppleSiliconGeneration.M3
                gen_name = "M3"
            else:
                gen = AppleSiliconGeneration.UNKNOWN
                gen_name = "Unknown"
                
            return CheckResult(
                "Apple Silicon", True,
                f"Running on Apple {gen_name} chip ({chip_info})."
            )
        except:
            return CheckResult(
                "Apple Silicon", True,
                "Running on Apple Silicon (unknown generation)."
            )
    else:
        return CheckResult(
            "Apple Silicon", False,
            f"Not running on Apple Silicon. Detected processor: {processor}",
            critical=True
        )

def check_python_version() -> CheckResult:
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        return CheckResult(
            "Python Version", True,
            f"Python {version.major}.{version.minor}.{version.micro} is compatible (>= 3.9 required)."
        )
    else:
        return CheckResult(
            "Python Version", False,
            f"Python {version.major}.{version.minor}.{version.micro} is not compatible. Version 3.9 or newer is required.",
            critical=True
        )

def check_dependency(package: str, min_version: Optional[str] = None) -> CheckResult:
    """Check if a Python package is installed with minimum version"""
    try:
        # Check if package is installed
        spec = importlib.util.find_spec(package)
        if spec is None:
            return CheckResult(
                f"{package}", False,
                f"{package} is not installed.",
                critical=(package in ["numpy", "mlx"])
            )
        
        # Get version if min_version is specified
        if min_version:
            try:
                installed_version = pkg_resources.get_distribution(package).version
                if pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(min_version):
                    return CheckResult(
                        f"{package} Version", False,
                        f"{package} version {installed_version} is installed, but version {min_version} or newer is required.",
                        critical=(package in ["numpy", "mlx"])
                    )
                else:
                    return CheckResult(
                        f"{package} Version", True,
                        f"{package} version {installed_version} is compatible (>= {min_version} required)."
                    )
            except:
                # If we can't determine the version, assume it's okay
                return CheckResult(
                    f"{package}", True,
                    f"{package} is installed, but version could not be determined."
                )
        
        return CheckResult(
            f"{package}", True,
            f"{package} is installed."
        )
    except:
        return CheckResult(
            f"{package}", False,
            f"Could not check {package} installation.",
            critical=(package in ["numpy", "mlx"])
        )

def check_metal_compiler() -> CheckResult:
    """Check if Metal compiler is available"""
    try:
        # Check if xcrun can find metal
        result = subprocess.run(
            ["xcrun", "--find", "metal"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode == 0:
            metal_path = result.stdout.strip()
            return CheckResult(
                "Metal Compiler", True,
                f"Metal compiler found at {metal_path}."
            )
        else:
            return CheckResult(
                "Metal Compiler", False,
                "Metal compiler not found. Please install Xcode command line tools.",
                critical=True
            )
    except:
        return CheckResult(
            "Metal Compiler", False,
            "Could not check for Metal compiler. Please ensure Xcode command line tools are installed.",
            critical=True
        )

def check_triton_metal_backend() -> CheckResult:
    """Check if Triton Metal backend is available"""
    try:
        import triton
        backends = triton.runtime.backends
        if 'metal' in backends:
            return CheckResult(
                "Triton Metal Backend", True,
                "Triton Metal backend is available."
            )
        else:
            return CheckResult(
                "Triton Metal Backend", False,
                f"Triton Metal backend is not available. Available backends: {list(backends.keys())}",
                critical=False
            )
    except ImportError:
        return CheckResult(
            "Triton Metal Backend", False,
            "Triton not installed. Cannot check for Metal backend.",
            critical=False
        )

def run_all_checks() -> List[CheckResult]:
    """Run all compatibility checks"""
    checks = [
        check_macos_version(),
        check_apple_silicon(),
        check_python_version(),
        check_dependency("numpy", "1.22.0"),
        check_dependency("mlx", "0.3.0"),
        check_dependency("triton"),
        check_metal_compiler(),
        check_triton_metal_backend()
    ]
    
    return checks

def print_summary(checks: List[CheckResult]) -> None:
    """Print a summary of all checks"""
    print("\n" + "=" * 80)
    print(" " * 25 + "METAL BACKEND COMPATIBILITY CHECK")
    print("=" * 80 + "\n")
    
    # Count passed and failed checks
    passed = 0
    critical_failed = 0
    
    # Print all checks
    for check in checks:
        if check.passed:
            status = "✅ PASS"
            passed += 1
        else:
            if check.critical:
                status = "❌ FAIL (REQUIRED)"
                critical_failed += 1
            else:
                status = "⚠️ FAIL (OPTIONAL)"
        
        print(f"{check.name + ':': <25} {status: <20} {check.message}")
    
    # Print summary
    print("\n" + "-" * 80)
    if critical_failed == 0:
        if passed == len(checks):
            print("✅ SUCCESS: All checks passed! Your system is fully compatible with the Metal backend.")
        else:
            print("✅ PARTIAL SUCCESS: Your system meets all critical requirements for the Metal backend.")
            print("   Some optional components are missing, but the backend should work.")
    else:
        print(f"❌ FAILURE: {critical_failed} critical checks failed. Your system is not compatible with the Metal backend.")
        print("   Please address the issues marked as REQUIRED above.")
    
    print("-" * 80)

def main():
    """Main function"""
    checks = run_all_checks()
    print_summary(checks)
    
    # Return non-zero exit code if any critical check failed
    for check in checks:
        if not check.passed and check.critical:
            sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    main() 