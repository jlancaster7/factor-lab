#!/usr/bin/env python3
"""
Quick test to verify future date filtering fix
"""

import sys
from pathlib import Path

# Add the src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from factor_lab.data import FMPProvider


def test_fix():
    print("=== Testing Future Date Fix ===")

    fmp = FMPProvider()

    # Test future date (should fail now)
    result_future = fmp._get_trailing_12m_data("AAPL", "2030-01-01")
    print(
        f"Future date (2030-01-01) - Successful: {result_future['metadata']['calculation_successful']}"
    )

    # Test normal date (should work)
    result_normal = fmp._get_trailing_12m_data("AAPL", "2024-01-01")
    print(
        f"Normal date (2024-01-01) - Successful: {result_normal['metadata']['calculation_successful']}"
    )

    # Test current date (should work)
    result_current = fmp._get_trailing_12m_data("AAPL", "2025-06-01")
    print(
        f"Current date (2025-06-01) - Successful: {result_current['metadata']['calculation_successful']}"
    )


if __name__ == "__main__":
    test_fix()
