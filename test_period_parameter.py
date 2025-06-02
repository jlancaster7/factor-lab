#!/usr/bin/env python3
"""
Test script to check what period values FMP accepts for income statement data
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


def test_period_parameter():
    """Test different period parameter values with FMP API"""
    print("🔍 Testing FMP Income Statement Period Parameter")
    print("=" * 60)

    fmp = FMPProvider()
    symbol = "AAPL"

    # Test different period values
    period_tests = [
        "annual",  # Default - should work (we've seen this)
        "quarter",  # What we want to try
        "quarterly",  # Alternative
        "Q1",  # Specific quarter
        "Q2",  # Specific quarter
        "Q3",  # Specific quarter
        "Q4",  # Specific quarter
        "FY",  # Full year (from docs)
    ]

    results = {}

    for period in period_tests:
        print(f"📊 Testing period='{period}'")
        print("-" * 40)

        try:
            income_data = fmp._fetch_income_statement(symbol, limit=3, period=period)

            if income_data and len(income_data) > 0:
                print(f"✅ SUCCESS: Got {len(income_data)} records")

                # Show first record details
                first_record = income_data[0]
                print(f"   📅 First record date: {first_record.get('date', 'N/A')}")
                print(f"   🔄 First record period: {first_record.get('period', 'N/A')}")
                print(f"   💰 Revenue: ${first_record.get('revenue', 0):,.0f}")
                print(f"   💵 Net Income: ${first_record.get('netIncome', 0):,.0f}")

                results[period] = {
                    "success": True,
                    "count": len(income_data),
                    "first_date": first_record.get("date"),
                    "first_period": first_record.get("period"),
                    "sample_revenue": first_record.get("revenue", 0),
                }

            else:
                print(f"❌ FAILED: No data returned")
                results[period] = {"success": False, "error": "No data returned"}

        except Exception as e:
            print(f"❌ ERROR: {e}")
            results[period] = {"success": False, "error": str(e)}

        print()

    # Summary
    print("📋 SUMMARY")
    print("=" * 60)

    successful_periods = []
    failed_periods = []

    for period, result in results.items():
        if result["success"]:
            successful_periods.append(period)
            print(
                f"✅ {period}: {result['count']} records, first period: {result.get('first_period', 'N/A')}"
            )
        else:
            failed_periods.append(period)
            print(f"❌ {period}: {result.get('error', 'Unknown error')}")

    print()
    print(f"✅ Working periods: {successful_periods}")
    print(f"❌ Failed periods: {failed_periods}")

    # Check if we have quarterly data
    quarterly_periods = [
        p
        for p in successful_periods
        if p in ["quarter", "quarterly", "Q1", "Q2", "Q3", "Q4"]
    ]
    if quarterly_periods:
        print(f"🎯 Quarterly data available via: {quarterly_periods}")
    else:
        print(f"⚠️  No quarterly periods worked - may need different approach")


if __name__ == "__main__":
    test_period_parameter()
