#!/usr/bin/env python3
"""
Test script for FMP Provider data validation and cleaning methods.
Story 1.3 - Data Validation and Cleaning validation.
"""

import sys
import os
sys.path.append('/home/jcl/pythonCode/factor_lab_2/factor-lab/src')

from factor_lab.data import FMPProvider
import json
import time
from datetime import datetime
import pandas as pd

def test_data_validation():
    """Test the data validation and cleaning pipeline."""
    print("=" * 60)
    print("🧪 TESTING FMP Data Validation and Cleaning Methods")
    print("=" * 60)
    
    # Initialize FMP Provider
    try:
        fmp = FMPProvider()
        print(f"✅ FMP Provider initialized successfully")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize FMP Provider: {e}")
        return
    
    # Test 1: Parse Date Safely
    print("📅 Test 1: Date Parsing (_parse_date_safely)")
    print("-" * 50)
    
    test_dates = [
        "2024-09-28",
        "2024-09-28T16:30:00",
        "2024-09-28 16:30:00", 
        "2024-12-31T23:59:59",
        "invalid-date",
        "",
        None,
        "2024-02-29",  # Leap year
        "2023-02-29",  # Invalid leap year
    ]
    
    date_results = []
    for test_date in test_dates:
        try:
            result = fmp._parse_date_safely(test_date)
            status = "✅" if result else "⚠️"
            print(f"   {status} '{test_date}' -> {result}")
            date_results.append((test_date, result, True))
        except Exception as e:
            print(f"   ❌ '{test_date}' -> ERROR: {e}")
            date_results.append((test_date, None, False))
    
    print()
    
    # Test 2: Validate Financial Data - Real Data
    print("📊 Test 2: Real Data Validation")
    print("-" * 50)
    
    # Fetch real data to validate
    symbol = "AAPL"
    try:
        real_income_data = fmp._fetch_income_statement(symbol, limit=2)
        if real_income_data:
            print(f"   Fetched {len(real_income_data)} real income statement records")
            
            # Test validation with real data
            validated_real = fmp._validate_financial_data(real_income_data, "income_statement")
            print(f"   ✅ Validated {len(validated_real)}/{len(real_income_data)} real records")
            
            if validated_real:
                sample_record = validated_real[0]
                print(f"   📋 Sample validated record:")
                print(f"      Date: {sample_record.get('date', 'N/A')}")
                print(f"      Symbol: {sample_record.get('symbol', 'N/A')}")
                print(f"      AcceptedDate: {sample_record.get('acceptedDate', 'N/A')}")
                print(f"      Revenue type: {type(sample_record.get('revenue', 'N/A'))}")
                print(f"      Revenue value: {sample_record.get('revenue', 'N/A')}")
        else:
            print("   ❌ Could not fetch real data for validation testing")
    except Exception as e:
        print(f"   ❌ Real data validation test failed: {e}")
    
    print()
    
    # Test 3: Validate Financial Data - Synthetic Edge Cases
    print("🔧 Test 3: Synthetic Edge Case Validation")
    print("-" * 50)
    
    # Create test data with various edge cases
    synthetic_data = [
        # Normal record
        {
            "date": "2024-09-28",
            "symbol": "TEST",
            "acceptedDate": "2024-10-30T16:30:00",
            "revenue": "391035000000",
            "netIncome": "94320000000",
            "totalAssets": "364980000000",
            "period": "FY",
            "fiscalYear": "2024"
        },
        # Missing required fields
        {
            "revenue": "100000000",
            "netIncome": "10000000"
        },
        # Invalid date formats
        {
            "date": "invalid-date",
            "symbol": "TEST2",
            "revenue": "200000000"
        },
        # Null/empty values
        {
            "date": "2024-06-30",
            "symbol": "TEST3",
            "revenue": None,
            "netIncome": "",
            "totalAssets": 0,
            "acceptedDate": ""
        },
        # String numbers (should be converted)
        {
            "date": "2024-03-31",
            "symbol": "TEST4",
            "revenue": "150000000",
            "netIncome": "15000000.50",
            "acceptedDate": "2024-04-15"
        },
        # Non-numeric values in numeric fields
        {
            "date": "2024-12-31",
            "symbol": "TEST5",
            "revenue": "not-a-number",
            "netIncome": "15000000",
            "acceptedDate": "2025-01-15"
        }
    ]
    
    try:
        validated_synthetic = fmp._validate_financial_data(synthetic_data, "test_data")
        print(f"   ✅ Processed {len(synthetic_data)} synthetic records")
        print(f"   ✅ Validated {len(validated_synthetic)} records passed validation")
        
        # Analyze results
        expected_valid = 3  # Records 1, 4, 5 should pass (5 will have revenue=None)
        if len(validated_synthetic) >= expected_valid - 1:  # Allow some flexibility
            print(f"   ✅ Expected ~{expected_valid} valid records, got {len(validated_synthetic)}")
        else:
            print(f"   ⚠️  Expected ~{expected_valid} valid records, got {len(validated_synthetic)}")
        
        # Check specific validations
        for i, record in enumerate(validated_synthetic):
            print(f"   📋 Validated record {i+1}:")
            print(f"      Date: {record.get('date', 'N/A')} (type: {type(record.get('date', 'N/A'))})")
            print(f"      Symbol: {record.get('symbol', 'N/A')}")
            print(f"      Revenue: {record.get('revenue', 'N/A')} (type: {type(record.get('revenue', 'N/A'))})")
            print(f"      AcceptedDate: {record.get('acceptedDate', 'N/A')} (type: {type(record.get('acceptedDate', 'N/A'))})")
            print()
            
    except Exception as e:
        print(f"   ❌ Synthetic data validation test failed: {e}")
    
    print()
    
    # Test 4: Missing Fields Handling
    print("🚫 Test 4: Missing Fields Handling")
    print("-" * 50)
    
    missing_fields_data = [
        {"symbol": "TEST_NO_DATE", "revenue": "100000"},
        {"date": "2024-01-01", "revenue": "200000"},  # Missing symbol
        {"date": "2024-01-01", "symbol": "GOOD", "revenue": "300000"},  # Good record
    ]
    
    try:
        validated_missing = fmp._validate_financial_data(missing_fields_data, "missing_fields_test")
        print(f"   ✅ Processed {len(missing_fields_data)} records with missing fields")
        print(f"   ✅ {len(validated_missing)} records passed validation (expected: 1)")
        
        if len(validated_missing) == 1:
            print(f"   ✅ Correctly filtered out records with missing required fields")
        else:
            print(f"   ⚠️  Expected 1 valid record, got {len(validated_missing)}")
            
    except Exception as e:
        print(f"   ❌ Missing fields test failed: {e}")
    
    print()
    
    # Test 5: Data Type Conversion
    print("🔄 Test 5: Data Type Conversion")
    print("-" * 50)
    
    type_conversion_data = [
        {
            "date": "2024-01-01",
            "symbol": "TYPE_TEST",
            "revenue": "1000000000",      # String -> Float
            "netIncome": 50000000,        # Already numeric
            "totalAssets": "0",           # String zero -> Float
            "operatingIncome": "",        # Empty string -> None
            "grossProfit": None,          # None -> None
            "period": "Q4",               # String (should stay string)
            "fiscalYear": "2024"          # String (should stay string)
        }
    ]
    
    try:
        validated_types = fmp._validate_financial_data(type_conversion_data, "type_conversion_test")
        if validated_types:
            record = validated_types[0]
            print(f"   ✅ Type conversion successful")
            print(f"      revenue: {record['revenue']} (type: {type(record['revenue'])})")
            print(f"      netIncome: {record['netIncome']} (type: {type(record['netIncome'])})")
            print(f"      totalAssets: {record['totalAssets']} (type: {type(record['totalAssets'])})")
            print(f"      operatingIncome: {record['operatingIncome']} (type: {type(record['operatingIncome'])})")
            print(f"      grossProfit: {record['grossProfit']} (type: {type(record['grossProfit'])})")
            print(f"      period: {record['period']} (type: {type(record['period'])})")
            
            # Validate types
            type_checks = [
                (record['revenue'], float, "revenue should be float"),
                (record['netIncome'], (float, int), "netIncome should be numeric"),
                (record['totalAssets'], float, "totalAssets should be float"),
                (record['period'], str, "period should stay string"),
            ]
            
            all_types_correct = True
            for value, expected_type, description in type_checks:
                if isinstance(expected_type, tuple):
                    if not isinstance(value, expected_type):
                        print(f"   ❌ {description}: got {type(value)}")
                        all_types_correct = False
                else:
                    if not isinstance(value, expected_type):
                        print(f"   ❌ {description}: got {type(value)}")
                        all_types_correct = False
            
            if all_types_correct:
                print(f"   ✅ All data types converted correctly")
            else:
                print(f"   ⚠️  Some data type conversions failed")
        else:
            print(f"   ❌ No records passed type conversion validation")
            
    except Exception as e:
        print(f"   ❌ Type conversion test failed: {e}")
    
    print()
    
    # Test 6: Integration with Raw Data Fetching
    print("🔗 Test 6: Integration with Raw Data Fetching")
    print("-" * 50)
    
    try:
        # Test integration with each endpoint
        endpoints = [
            ("income_statement", fmp._fetch_income_statement),
            ("balance_sheet", fmp._fetch_balance_sheet),
            ("cash_flow", fmp._fetch_cash_flow),
            ("financial_ratios", fmp._fetch_financial_ratios)
        ]
        
        for endpoint_name, fetch_method in endpoints:
            print(f"   📊 Testing {endpoint_name} integration...")
            
            # Fetch data
            raw_data = fetch_method(symbol, limit=1)
            if raw_data:
                # Validate data
                validated_data = fmp._validate_financial_data(raw_data, endpoint_name)
                
                if validated_data:
                    print(f"      ✅ {endpoint_name}: {len(validated_data)}/{len(raw_data)} records validated")
                    
                    # Check key fields are properly typed
                    sample = validated_data[0]
                    numeric_fields = [k for k, v in sample.items() 
                                    if k not in ['date', 'symbol', 'acceptedDate', 'period', 'fiscalYear', 'reportedCurrency']
                                    and v is not None]
                    
                    if numeric_fields:
                        sample_field = numeric_fields[0]
                        sample_value = sample[sample_field]
                        print(f"      📋 Sample field '{sample_field}': {sample_value} (type: {type(sample_value)})")
                else:
                    print(f"      ❌ {endpoint_name}: No records passed validation")
            else:
                print(f"      ⚠️  {endpoint_name}: No raw data fetched")
                
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
    
    print()
    print("=" * 60)
    print("🏁 Data Validation and Cleaning Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    test_data_validation()