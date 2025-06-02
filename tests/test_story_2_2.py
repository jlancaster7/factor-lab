#!/usr/bin/env python3
"""
Test suite for Story 2.2: Trailing 12-Month Calculations + Smart Financial Ratio acceptedDate Mapping

This test validates:
1. Trailing 12-month income statement aggregation
2. Balance sheet point-in-time data retrieval
3. Smart acceptedDate mapping for financial ratios
4. Cross-statement data alignment logic
5. Look-ahead bias prevention in calculated ratios
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path so we can import factor_lab
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from factor_lab.data import FMPProvider
import pandas as pd
from datetime import datetime, timedelta


def test_trailing_12m_calculations():
    """Test trailing 12-month data calculation functionality."""
    print("=" * 70)
    print("📊 TESTING Story 2.2: Trailing 12M Calculations + Smart Ratio Mapping")
    print("=" * 70)

    # Initialize FMP Provider
    try:
        fmp = FMPProvider()
        print(f"✅ FMP Provider initialized successfully")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize FMP Provider: {e}")
        return

    # Test symbol and date
    symbol = "AAPL"
    as_of_date = "2024-01-01"

    print(f"🔍 Testing with symbol: {symbol}")
    print(f"📅 Analysis date (as_of_date): {as_of_date}")
    print()

    # Test 1: Trailing 12M Data Calculation
    print("📋 Test 1: Trailing 12-Month Data Calculation")
    print("-" * 50)

    try:
        trailing_data = fmp._get_trailing_12m_data(symbol, as_of_date)

        metadata = trailing_data["metadata"]
        print(f"✅ Trailing 12M calculation completed")
        print(f"   📊 Calculation successful: {metadata['calculation_successful']}")
        print(f"   🗓️  Quarters used: {metadata['quarters_used']}")
        print(f"   📅 Balance sheet date: {metadata.get('balance_sheet_date', 'N/A')}")

        if trailing_data["trailing_12m"]:
            ttm = trailing_data["trailing_12m"]
            print(f"   💰 TTM Revenue: ${ttm.get('revenue', 0):,.0f}")
            print(f"   💵 TTM Net Income: ${ttm.get('netIncome', 0):,.0f}")
            print(f"   📈 TTM Operating Income: ${ttm.get('operatingIncome', 0):,.0f}")
            print(f"   📊 Net Margin: {ttm.get('net_margin', 0):.2%}")
            print(f"   📊 Operating Margin: {ttm.get('operating_margin', 0):.2%}")

        if trailing_data["balance_sheet"]:
            bs = trailing_data["balance_sheet"]
            print(f"   🏦 Total Assets: ${bs.get('totalAssets', 0):,.0f}")
            print(f"   🏛️  Total Equity: ${bs.get('totalStockholdersEquity', 0):,.0f}")
            print(f"   💳 Total Debt: ${bs.get('totalDebt', 0):,.0f}")

        print()

    except Exception as e:
        print(f"❌ Trailing 12M calculation failed: {e}")
        print()

    # Test 2: Smart acceptedDate Mapping
    print("📋 Test 2: Smart acceptedDate Mapping for Ratios")
    print("-" * 50)

    test_ratios = [
        "debt_to_equity",  # Balance sheet only
        "pe_ratio",  # Income statement only
        "roe",  # Multi-statement (income + balance)
        "current_ratio",  # Balance sheet only
    ]

    for ratio_name in test_ratios:
        try:
            accepted_date = fmp._get_smart_ratio_accepted_date(
                ratio_name, symbol, as_of_date
            )

            if accepted_date:
                print(
                    f"   ✅ {ratio_name}: acceptedDate = {accepted_date.strftime('%Y-%m-%d')}"
                )
            else:
                print(
                    f"   ⚠️  {ratio_name}: No acceptedDate determined (will use fallback)"
                )

        except Exception as e:
            print(f"   ❌ {ratio_name}: Error - {e}")

    print()

    # Test 3: Complete Financial Ratio Calculation with Timing
    print("📋 Test 3: Financial Ratio Calculation with Smart Timing")
    print("-" * 50)

    try:
        ratios_result = fmp._calculate_financial_ratios_with_timing(symbol, as_of_date)

        metadata = ratios_result["metadata"]
        print(f"✅ Financial ratio calculation completed")
        print(f"   🎯 Ratios attempted: {len(metadata['ratios_attempted'])}")
        print(f"   ✅ Ratios successful: {len(metadata['ratios_successful'])}")
        print(f"   ❌ Errors: {len(metadata['errors'])}")
        print()

        # Display successful ratios
        if ratios_result["ratios"]:
            print("   📊 Calculated Ratios:")
            for ratio_name, ratio_data in ratios_result["ratios"].items():
                if isinstance(ratio_data, dict):
                    value = ratio_data.get("value")
                    accepted_date = ratio_data.get("accepted_date", "N/A")
                    note = ratio_data.get("note", "")

                    if value is not None:
                        if ratio_name in ["operating_margin", "net_margin"]:
                            print(
                                f"      {ratio_name}: {value:.2%} (accepted: {accepted_date})"
                            )
                        else:
                            print(
                                f"      {ratio_name}: {value:.3f} (accepted: {accepted_date})"
                            )
                    else:
                        print(f"      {ratio_name}: {note} (accepted: {accepted_date})")

        # Display errors if any
        if metadata["errors"]:
            print("   ⚠️  Errors encountered:")
            for error in metadata["errors"]:
                print(f"      - {error}")

        print()

    except Exception as e:
        print(f"❌ Financial ratio calculation failed: {e}")
        print()

    # Test 4: Look-Ahead Bias Validation
    print("📋 Test 4: Look-Ahead Bias Validation")
    print("-" * 50)

    # Test with a date that should have limited data
    early_date = "2021-01-01"
    print(f"🔍 Testing with early date: {early_date}")

    try:
        early_trailing_data = fmp._get_trailing_12m_data(symbol, early_date)
        early_metadata = early_trailing_data["metadata"]

        print(
            f"   📊 Calculation successful: {early_metadata['calculation_successful']}"
        )
        print(f"   🗓️  Quarters used: {early_metadata['quarters_used']}")

        if early_trailing_data["trailing_12m"]:
            early_ttm = early_trailing_data["trailing_12m"]
            print(f"   💰 TTM Revenue: ${early_ttm.get('revenue', 0):,.0f}")
            print(f"   💵 TTM Net Income: ${early_ttm.get('netIncome', 0):,.0f}")

        # Verify no future data
        if early_trailing_data["balance_sheet"]:
            bs_date = early_trailing_data["balance_sheet"].get("date")
            accepted_date = early_trailing_data["balance_sheet"].get("acceptedDate")

            print(f"   📅 Balance sheet fiscal date: {bs_date}")
            print(f"   📅 Balance sheet accepted date: {accepted_date}")

            # Validate no look-ahead bias
            if accepted_date:
                accepted_dt = pd.to_datetime(accepted_date)
                early_dt = pd.to_datetime(early_date)

                if accepted_dt <= early_dt:
                    print(f"   ✅ No look-ahead bias: accepted <= as_of_date")
                else:
                    print(f"   ❌ Look-ahead bias detected: accepted > as_of_date")

        print()

    except Exception as e:
        print(f"❌ Look-ahead bias validation failed: {e}")
        print()

    # Test 5: Data Quality and Edge Cases
    print("📋 Test 5: Data Quality and Edge Cases")
    print("-" * 50)

    # Test with insufficient data
    future_date = "2030-01-01"
    print(f"🔍 Testing with future date (should have no data): {future_date}")

    try:
        future_data = fmp._get_trailing_12m_data(symbol, future_date)
        future_metadata = future_data["metadata"]

        print(
            f"   📊 Calculation successful: {future_metadata['calculation_successful']}"
        )
        print(f"   🗓️  Quarters used: {future_metadata['quarters_used']}")

        if future_metadata["calculation_successful"]:
            print(f"   ⚠️  Unexpected: Got data for future date")
        else:
            print(f"   ✅ Correctly handled future date (no data available)")

        print()

    except Exception as e:
        print(f"❌ Future date test failed: {e}")
        print()

    # Test with invalid symbol
    invalid_symbol = "INVALID123"
    print(f"🔍 Testing with invalid symbol: {invalid_symbol}")

    try:
        invalid_data = fmp._get_trailing_12m_data(invalid_symbol, as_of_date)
        invalid_metadata = invalid_data["metadata"]

        print(
            f"   📊 Calculation successful: {invalid_metadata['calculation_successful']}"
        )

        if invalid_metadata["calculation_successful"]:
            print(f"   ⚠️  Unexpected: Got data for invalid symbol")
        else:
            print(f"   ✅ Correctly handled invalid symbol")

    except Exception as e:
        print(f"❌ Invalid symbol test failed: {e}")

    print()
    print("=" * 70)
    print("🎉 STORY 2.2 TESTING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    test_trailing_12m_calculations()
