#!/usr/bin/env python3
"""
Debug script to understand why future date (2030-01-01) is returning data
when it should be filtered out by acceptedDate.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from factor_lab.data import FMPProvider
import pandas as pd


def debug_future_date_issue():
    """Debug why 2030-01-01 returns data when it shouldn't."""
    print("=== DEBUGGING Future Date Issue (2030-01-01) ===")

    fmp = FMPProvider()
    symbol = "AAPL"
    future_date = "2030-01-01"

    print(f"Testing {symbol} with future date: {future_date}")
    print()

    # Step 1: Fetch raw data
    print("📥 Step 1: Fetching raw quarterly data...")
    raw_data = fmp._fetch_income_statement(symbol, limit=8, period="quarter")

    if raw_data:
        print(f"   Fetched {len(raw_data)} raw records")
        print("   Recent records:")
        for i, record in enumerate(raw_data[:3]):
            date = record.get("date", "Unknown")
            accepted_date = record.get("acceptedDate", "Unknown")
            revenue = record.get("revenue", 0)
            print(
                f"   {i+1}. Date: {date}, AcceptedDate: {accepted_date}, Revenue: ${revenue:,}"
            )
    else:
        print("   ❌ No raw data fetched")
        return

    print()

    # Step 2: Validate data
    print("✅ Step 2: Validating data...")
    validated_data = fmp._validate_financial_data(raw_data, "income_statement")
    print(f"   Validated {len(validated_data)}/{len(raw_data)} records")
    print()

    # Step 3: Filter by acceptedDate - this is where the bug should be caught
    print(f"🔍 Step 3: Filtering by acceptedDate (as_of_date: {future_date})...")
    filtered_data = fmp._filter_by_accepted_date(validated_data, future_date)

    print(f"   After filtering: {len(filtered_data)} records")

    if filtered_data:
        print("   ⚠️  PROBLEM: Got filtered data for future date!")
        print("   Records that passed filter:")
        for i, record in enumerate(filtered_data):
            date = record.get("date", "Unknown")
            accepted_date = record.get("acceptedDate", "Unknown")
            revenue = record.get("revenue", 0)
            print(
                f"   {i+1}. Date: {date}, AcceptedDate: {accepted_date}, Revenue: ${revenue:,}"
            )

            # Check the actual comparison
            if accepted_date and accepted_date != "Unknown":
                try:
                    accepted_dt = pd.to_datetime(accepted_date)
                    future_dt = pd.to_datetime(future_date)
                    comparison_result = accepted_dt <= future_dt
                    print(
                        f"       Comparison: {accepted_dt} <= {future_dt} = {comparison_result}"
                    )
                except Exception as e:
                    print(f"       Error in date comparison: {e}")
    else:
        print("   ✅ Correctly filtered out all data for future date")

    print()

    # Step 4: Test the full TTM calculation
    print("📊 Step 4: Testing full TTM calculation...")
    result = fmp._get_trailing_12m_data(symbol, future_date, min_quarters=4)

    metadata = result["metadata"]
    print(f"   Calculation successful: {metadata['calculation_successful']}")
    print(f"   Quarters used: {metadata['quarters_used']}")

    if metadata["calculation_successful"]:
        print("   ⚠️  PROBLEM: TTM calculation succeeded for future date!")
        ttm = result["trailing_12m"]
        print(f"   TTM Revenue: ${ttm.get('revenue', 0):,}")
    else:
        print("   ✅ TTM calculation correctly failed for future date")


if __name__ == "__main__":
    debug_future_date_issue()
