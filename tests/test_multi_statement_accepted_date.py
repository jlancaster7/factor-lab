#!/usr/bin/env python3
"""
Comprehensive test for accepted date filtering across all statement types.
Story 2.1 - Multi-statement validation.
"""

import sys

sys.path.append("/home/jcl/pythonCode/factor_lab_2/factor-lab/src")

from factor_lab.data import FMPProvider
import pandas as pd


def test_multi_statement_filtering():
    """Test accepted date filtering across all statement types."""
    print("=" * 60)
    print("📊 TESTING Multi-Statement Accepted Date Filtering")
    print("=" * 60)

    # Initialize FMP Provider
    try:
        fmp = FMPProvider()
        print(f"✅ FMP Provider initialized successfully")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize FMP Provider: {e}")
        return

    symbol = "AAPL"
    as_of_date = "2024-01-01"  # Test date

    # Test all statement types
    statement_types = [
        ("income_statement", fmp._fetch_income_statement),
        ("balance_sheet", fmp._fetch_balance_sheet),
        ("cash_flow", fmp._fetch_cash_flow),
        ("financial_ratios", fmp._fetch_financial_ratios),
    ]

    results_summary = []

    for statement_name, fetch_method in statement_types:
        print(f"\n📋 Testing {statement_name.replace('_', ' ').title()}")
        print("-" * 40)

        try:
            # Fetch raw data
            raw_data = fetch_method(symbol, limit=5)

            if raw_data:
                print(f"✅ Fetched {len(raw_data)} raw {statement_name} records")

                # Validate data
                validated_data = fmp._validate_financial_data(raw_data, statement_name)
                print(f"✅ Validated {len(validated_data)} records")

                # Check acceptedDate availability
                with_accepted_date = [
                    r for r in validated_data if r.get("acceptedDate")
                ]
                print(
                    f"📅 Records with acceptedDate: {len(with_accepted_date)}/{len(validated_data)}"
                )

                # Apply accepted date filtering (with fallback for financial ratios)
                filtered_data = fmp._filter_by_accepted_date(
                    validated_data, as_of_date, use_fiscal_date_fallback=True
                )
                print(f"🎯 Available as of {as_of_date}: {len(filtered_data)} records")

                # Show date range and method used
                if filtered_data:
                    if with_accepted_date:
                        # Using acceptedDate
                        dates = [
                            r.get("acceptedDate")
                            for r in filtered_data
                            if r.get("acceptedDate")
                        ]
                        if dates:
                            earliest = min(dates).strftime("%Y-%m-%d")
                            latest = max(dates).strftime("%Y-%m-%d")
                            print(
                                f"📊 Date range (acceptedDate): {earliest} to {latest}"
                            )
                    else:
                        # Using fiscal date fallback
                        dates = [r.get("date") for r in filtered_data if r.get("date")]
                        if dates:
                            # Convert string dates to datetime if needed
                            import pandas as pd

                            pd_dates = [pd.to_datetime(d) for d in dates]
                            earliest = min(pd_dates).strftime("%Y-%m-%d")
                            latest = max(pd_dates).strftime("%Y-%m-%d")
                            print(
                                f"📊 Date range (fiscal + lag): {earliest} to {latest}"
                            )
                            print(f"🔄 Used fiscal date fallback (75-day lag)")

                # Determine success based on whether we got filtered results
                success = len(filtered_data) > 0
                if not with_accepted_date and success:
                    print(f"✅ Fallback method succeeded for {statement_name}")
                elif not with_accepted_date:
                    print(
                        f"⚠️  No acceptedDate found and fallback failed for {statement_name}"
                    )

                results_summary.append(
                    {
                        "statement_type": statement_name,
                        "total_records": len(validated_data),
                        "with_accepted_date": len(with_accepted_date),
                        "available_as_of": len(filtered_data),
                        "success": success,
                    }
                )
            else:
                print(f"❌ No {statement_name} data available")
                results_summary.append(
                    {
                        "statement_type": statement_name,
                        "total_records": 0,
                        "with_accepted_date": 0,
                        "available_as_of": 0,
                        "success": False,
                    }
                )

        except Exception as e:
            print(f"❌ Error testing {statement_name}: {e}")
            results_summary.append(
                {
                    "statement_type": statement_name,
                    "total_records": 0,
                    "with_accepted_date": 0,
                    "available_as_of": 0,
                    "success": False,
                }
            )

    # Summary report
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY REPORT")
    print(f"{'='*60}")
    print(
        f"{'Statement Type':<20} {'Total':<8} {'w/Date':<8} {'Available':<10} {'Status':<8}"
    )
    print("-" * 60)

    total_success = 0
    for result in results_summary:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        if result["success"]:
            total_success += 1

        print(
            f"{result['statement_type']:<20} {result['total_records']:<8} "
            f"{result['with_accepted_date']:<8} {result['available_as_of']:<10} {status:<8}"
        )

    print("-" * 60)
    print(
        f"Overall Success Rate: {total_success}/{len(statement_types)} ({total_success/len(statement_types)*100:.0f}%)"
    )

    if total_success == len(statement_types):
        print(f"\n🎉 ALL STATEMENT TYPES PASSED ACCEPTED DATE FILTERING!")
    else:
        print(f"\n⚠️  Some statement types failed - review above for details")


if __name__ == "__main__":
    test_multi_statement_filtering()
