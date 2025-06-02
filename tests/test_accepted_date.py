#!/usr/bin/env python3
"""
Test script for FMP Provider accepted date handling.
Story 2.1 - Accepted Date Handling validation.
"""

import sys
import os

sys.path.append("/home/jcl/pythonCode/factor_lab_2/factor-lab/src")

from factor_lab.data import FMPProvider
import pandas as pd
from datetime import datetime, timedelta
import json


def test_accepted_date_filtering():
    """Test the accepted date filtering functionality."""
    print("=" * 60)
    print("📅 TESTING FMP Accepted Date Filtering")
    print("=" * 60)

    # Initialize FMP Provider
    try:
        fmp = FMPProvider()
        print(f"✅ FMP Provider initialized successfully")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize FMP Provider: {e}")
        return

    # Test symbol
    symbol = "AAPL"
    print(f"🔍 Testing with symbol: {symbol}")

    # Test 1: Fetch and validate data structure
    print("\n📊 Test 1: Data Structure Analysis")
    print("-" * 40)

    try:
        # Fetch income statement data
        raw_data = fmp._fetch_income_statement(symbol, limit=8)

        if raw_data:
            print(f"✅ Fetched {len(raw_data)} raw income statement records")

            # Validate data to ensure acceptedDate parsing
            validated_data = fmp._validate_financial_data(raw_data, "income_statement")
            print(f"✅ Validated {len(validated_data)} records")

            # Examine acceptedDate availability
            records_with_accepted_date = [
                r for r in validated_data if r.get("acceptedDate")
            ]
            print(
                f"📅 Records with acceptedDate: {len(records_with_accepted_date)}/{len(validated_data)}"
            )

            if records_with_accepted_date:
                # Show sample acceptedDates
                print("\n📋 Sample acceptedDate data:")
                for i, record in enumerate(records_with_accepted_date[:3]):
                    fiscal_date = record.get("date", "Unknown")
                    accepted_date = record.get("acceptedDate")
                    if accepted_date:
                        accepted_str = (
                            accepted_date.strftime("%Y-%m-%d")
                            if hasattr(accepted_date, "strftime")
                            else str(accepted_date)
                        )
                        print(
                            f"   {i+1}. Fiscal: {fiscal_date}, Accepted: {accepted_str}"
                        )
            else:
                print("⚠️  No records found with acceptedDate - this might be an issue")
        else:
            print("❌ No raw data returned")
            return

    except Exception as e:
        print(f"❌ Error in data structure analysis: {e}")
        return

    # Test 2: Point-in-time filtering
    print("\n🎯 Test 2: Point-in-Time Filtering")
    print("-" * 40)

    try:
        # Test various as_of_dates
        test_dates = [
            "2023-01-01",  # Early date - should filter most data
            "2023-06-01",  # Mid-year
            "2024-01-01",  # Recent but not too recent
            "2024-06-01",  # More recent
        ]

        for as_of_date in test_dates:
            print(f"\n📅 Testing as_of_date: {as_of_date}")

            filtered_data = fmp._filter_by_accepted_date(validated_data, as_of_date)
            print(
                f"   📊 Available records: {len(filtered_data)}/{len(validated_data)}"
            )

            if filtered_data:
                # Show latest available data
                latest_record = max(
                    filtered_data, key=lambda x: x.get("acceptedDate", pd.Timestamp.min)
                )
                fiscal_date = latest_record.get("date", "Unknown")
                accepted_date = latest_record.get("acceptedDate")
                if accepted_date:
                    accepted_str = (
                        accepted_date.strftime("%Y-%m-%d")
                        if hasattr(accepted_date, "strftime")
                        else str(accepted_date)
                    )
                    print(
                        f"   📈 Latest available: Fiscal {fiscal_date}, Accepted {accepted_str}"
                    )
            else:
                print(f"   ⚠️  No data available as of {as_of_date}")

    except Exception as e:
        print(f"❌ Error in point-in-time filtering: {e}")
        return

    # Test 3: Look-ahead bias validation
    print("\n🚨 Test 3: Look-Ahead Bias Validation")
    print("-" * 40)

    try:
        # Use a date where we know some data should be filtered
        middle_date = "2023-08-01"
        filtered_data = fmp._filter_by_accepted_date(validated_data, middle_date)

        print(f"📅 Using as_of_date: {middle_date}")
        print(f"📊 Records available: {len(filtered_data)}")

        # Verify no look-ahead bias
        as_of_timestamp = pd.to_datetime(middle_date)
        bias_violations = []

        for record in filtered_data:
            accepted_date = record.get("acceptedDate")
            if accepted_date and accepted_date > as_of_timestamp:
                bias_violations.append(
                    {
                        "fiscal_date": record.get("date"),
                        "accepted_date": accepted_date.strftime("%Y-%m-%d"),
                        "violation": f"accepted > as_of_date ({middle_date})",
                    }
                )

        if bias_violations:
            print(f"❌ LOOK-AHEAD BIAS DETECTED! {len(bias_violations)} violations:")
            for violation in bias_violations:
                print(f"   🚨 {violation}")
        else:
            print(f"✅ No look-ahead bias detected - all acceptedDate <= {middle_date}")

    except Exception as e:
        print(f"❌ Error in bias validation: {e}")
        return

    # Test 4: Edge case testing
    print("\n🧪 Test 4: Edge Case Testing")
    print("-" * 40)

    try:
        # Test with empty data
        empty_result = fmp._filter_by_accepted_date([], "2024-01-01")
        print(f"✅ Empty data handling: {len(empty_result)} records (expected: 0)")

        # Test with invalid date format (should return empty list and log error)
        invalid_result = fmp._filter_by_accepted_date(validated_data, "invalid-date")
        if len(invalid_result) == 0:
            print(
                f"✅ Invalid date format properly handled: returned {len(invalid_result)} records"
            )
        else:
            print(
                f"❌ Invalid date should have returned 0 records but got {len(invalid_result)}"
            )

        # Test with future date (should return all data)
        future_date = "2030-01-01"
        future_result = fmp._filter_by_accepted_date(validated_data, future_date)
        records_with_dates = [r for r in validated_data if r.get("acceptedDate")]
        print(
            f"✅ Future date handling: {len(future_result)}/{len(records_with_dates)} records available"
        )

    except Exception as e:
        print(f"❌ Error in edge case testing: {e}")
        return

    print("\n" + "=" * 60)
    print("🎉 ACCEPTED DATE FILTERING TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    test_accepted_date_filtering()
