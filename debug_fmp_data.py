#!/usr/bin/env python3
"""
Quick test to examine FMP income statement data structure and period types
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from factor_lab.data import FMPProvider
import json


def test_fmp_data_structure():
    """Test to examine what data structure FMP returns"""
    print("🔍 Examining FMP Income Statement Data Structure")
    print("=" * 60)

    fmp = FMPProvider()
    symbol = "AAPL"

    # Get raw income statement data
    income_data = fmp._fetch_income_statement(symbol, limit=8)

    if income_data:
        print(f"📊 Found {len(income_data)} income statement records")
        print()

        for i, record in enumerate(income_data):
            period = record.get("period", "Unknown")
            date = record.get("date", "Unknown")
            revenue = record.get("revenue", 0)
            net_income = record.get("netIncome", 0)
            accepted_date = record.get("acceptedDate", "Unknown")

            print(f"Record {i+1}:")
            print(f"  📅 Date: {date}")
            print(f"  🔄 Period: {period}")
            print(f"  💰 Revenue: ${revenue:,.0f}")
            print(f"  💵 Net Income: ${net_income:,.0f}")
            print(f"  📝 Accepted Date: {accepted_date}")
            print()
    else:
        print("❌ No data returned")


if __name__ == "__main__":
    test_fmp_data_structure()
