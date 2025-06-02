#!/usr/bin/env python3
"""
Debug script to understand balance sheet data availability and filtering issues
"""

import sys
from pathlib import Path

# Add the src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from factor_lab.data import FMPProvider
import pandas as pd


def debug_balance_sheet_issue():
    """Debug why balance sheet data is not available as of 2024-01-01."""
    print("=" * 70)
    print("🔍 DEBUGGING Balance Sheet Data Availability Issue")
    print("=" * 70)

    fmp = FMPProvider()
    symbol = "AAPL"
    test_date = "2024-01-01"

    print(f"Testing {symbol} balance sheet data as of {test_date}")
    print()

    # Step 1: Fetch raw balance sheet data
    print("📥 Step 1: Fetching raw quarterly balance sheet data...")
    raw_balance_data = fmp._fetch_balance_sheet(symbol, limit=8, period="quarter")

    if raw_balance_data:
        print(f"   ✅ Fetched {len(raw_balance_data)} raw balance sheet records")
        print("   Recent records:")
        for i, record in enumerate(raw_balance_data[:5]):
            date = record.get("date", "Unknown")
            accepted_date = record.get("acceptedDate", "Unknown")
            total_assets = record.get("totalAssets", 0)
            total_equity = record.get("totalEquity", 0)
            print(f"   {i+1}. Date: {date}, AcceptedDate: {accepted_date}")
            print(f"       Assets: ${total_assets:,}, Equity: ${total_equity:,}")
    else:
        print("   ❌ No raw balance sheet data fetched")
        return

    print()

    # Step 2: Validate data
    print("✅ Step 2: Validating balance sheet data...")
    validated_balance = fmp._validate_financial_data(raw_balance_data, "balance_sheet")
    print(f"   Validated {len(validated_balance)}/{len(raw_balance_data)} records")
    print()

    # Step 3: Filter by acceptedDate
    print(f"🔍 Step 3: Filtering by acceptedDate (as_of_date: {test_date})...")
    filtered_balance = fmp._filter_by_accepted_date(validated_balance, test_date)

    print(f"   After filtering: {len(filtered_balance)} records")

    if filtered_balance:
        print("   ✅ Balance sheet records available:")
        for i, record in enumerate(filtered_balance):
            date = record.get("date", "Unknown")
            accepted_date = record.get("acceptedDate", "Unknown")
            total_assets = record.get("totalAssets", 0)
            total_equity = record.get("totalEquity", 0)
            print(f"   {i+1}. Date: {date}, AcceptedDate: {accepted_date}")
            print(f"       Assets: ${total_assets:,}, Equity: ${total_equity:,}")
    else:
        print("   ❌ No balance sheet records after filtering")
        print("   🔍 Checking why records were filtered out...")

        # Analyze each record to see why it was filtered
        test_date_dt = pd.to_datetime(test_date)
        for i, record in enumerate(validated_balance[:3]):
            date = record.get("date", "Unknown")
            accepted_date = record.get("acceptedDate", "Unknown")

            if accepted_date and accepted_date != "Unknown":
                try:
                    accepted_dt = pd.to_datetime(accepted_date)
                    is_valid = accepted_dt <= test_date_dt
                    status = "✅ PASS" if is_valid else "❌ FILTERED"
                    print(
                        f"   Record {i+1}: {accepted_dt} <= {test_date_dt} = {is_valid} ({status})"
                    )
                except Exception as e:
                    print(f"   Record {i+1}: Error parsing date - {e}")
            else:
                print(f"   Record {i+1}: Missing acceptedDate - would be filtered")

    print()

    # Step 4: Test with different dates to find available data
    print(
        "🔍 Step 4: Testing different analysis dates to find available balance sheet data..."
    )

    test_dates = [
        "2023-01-01",
        "2023-06-01",
        "2023-12-01",
        "2024-01-01",
        "2024-06-01",
        "2024-12-01",
        "2025-01-01",
    ]

    for test_date in test_dates:
        try:
            filtered_data = fmp._filter_by_accepted_date(validated_balance, test_date)
            count = len(filtered_data)
            status = f"✅ {count} records" if count > 0 else "❌ No data"
            print(f"   {test_date}: {status}")

            if count > 0:
                latest_record = filtered_data[0]
                latest_date = latest_record.get("date", "Unknown")
                latest_accepted = latest_record.get("acceptedDate", "Unknown")
                print(f"       Latest: Date={latest_date}, Accepted={latest_accepted}")
        except Exception as e:
            print(f"   {test_date}: Error - {e}")

    print()

    # Step 5: Test the full TTM calculation including balance sheet
    print("📊 Step 5: Testing full TTM calculation with balance sheet...")
    result = fmp._get_trailing_12m_data(
        symbol, "2024-01-01", include_balance_sheet=True
    )

    print(
        f"   TTM calculation successful: {result['metadata']['calculation_successful']}"
    )
    print(f"   Balance sheet date: {result['metadata']['balance_sheet_date']}")

    if result["balance_sheet"]:
        print("   ✅ Balance sheet data available:")
        bs = result["balance_sheet"]
        print(f"       Date: {bs.get('date', 'Unknown')}")
        print(f"       Accepted: {bs.get('acceptedDate', 'Unknown')}")
        print(f"       Assets: ${bs.get('totalAssets', 0):,}")
        print(f"       Equity: ${bs.get('totalEquity', 0):,}")
    else:
        print("   ❌ No balance sheet data in TTM result")

    print()
    print("=" * 70)


if __name__ == "__main__":
    debug_balance_sheet_issue()
