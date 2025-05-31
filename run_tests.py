#!/usr/bin/env python3
"""
Comprehensive test runner for Factor Lab.

This script runs both unit tests and setup verification.
"""

import subprocess
import sys
import os


def run_unit_tests():
    """Run pytest unit tests"""
    print("🧪 Running Unit Tests (pytest)")
    print("=" * 50)

    try:
        result = subprocess.run(
            ["poetry", "run", "python", "-m", "pytest", "tests/test_core.py", "-v"],
            cwd=os.path.dirname(__file__),
            capture_output=False,
        )
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running unit tests: {e}")
        return False


def run_setup_verification():
    """Run setup verification"""
    print("\n🔧 Running Setup Verification")
    print("=" * 50)

    try:
        result = subprocess.run(
            ["poetry", "run", "python", "verify_setup.py"],
            cwd=os.path.dirname(__file__),
            capture_output=False,
        )
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running setup verification: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Factor Lab Complete Test Suite")
    print("=" * 60)

    # Run unit tests
    unit_tests_passed = run_unit_tests()

    # Run setup verification
    setup_verification_passed = run_setup_verification()

    # Summary
    print("\n" + "=" * 60)
    print("📊 Complete Test Results:")
    print(f"   Unit Tests: {'✅ PASSED' if unit_tests_passed else '❌ FAILED'}")
    print(
        f"   Setup Verification: {'✅ PASSED' if setup_verification_passed else '❌ FAILED'}"
    )

    if unit_tests_passed and setup_verification_passed:
        print("\n🎉 All tests passed! Factor Lab is fully operational.")
        return True
    else:
        print("\n⚠️  Some tests failed. Please check the output above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
