#!/usr/bin/env python3
"""
Test script for FMP Provider data quality checks.
Story 1.3 - Testing data quality check functionality.
"""

import sys
import os
sys.path.append('/home/jcl/pythonCode/factor_lab_2/factor-lab/src')

from factor_lab.data import FMPProvider
import json

def test_data_quality_checks():
    """Test the data quality check functionality."""
    print("=" * 60)
    print("📊 TESTING FMP Data Quality Checks")
    print("=" * 60)
    
    # Initialize FMP Provider
    try:
        fmp = FMPProvider()
        print(f"✅ FMP Provider initialized successfully")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize FMP Provider: {e}")
        return
    
    # Test 1: Quality check with real data
    print("🔍 Test 1: Real Data Quality Check")
    print("-" * 40)
    
    try:
        # Fetch real income statement data
        symbol = "AAPL"
        real_data = fmp._fetch_income_statement(symbol, limit=3)
        
        if real_data:
            validated_data = fmp._validate_financial_data(real_data, "income_statement")
            quality_stats = fmp._perform_data_quality_checks(validated_data, "income_statement")
            
            print(f"   ✅ Quality check completed for {symbol}")
            print(f"   📊 Total records: {quality_stats['total_records']}")
            print(f"   🎯 Quality score: {quality_stats['quality_score']}%")
            print(f"   ⚠️  Issues found: {quality_stats['issues']}")
            
            if quality_stats['quality_score'] >= 80:
                print(f"   ✅ Good data quality (score >= 80%)")
            else:
                print(f"   ⚠️  Data quality concerns (score < 80%)")
        else:
            print(f"   ❌ Could not fetch real data for quality testing")
            
    except Exception as e:
        print(f"   ❌ Real data quality test failed: {e}")
    
    print()
    
    # Test 2: Quality check with problematic synthetic data
    print("⚠️  Test 2: Problematic Data Quality Check")
    print("-" * 40)
    
    problematic_data = [
        # Good record
        {
            "date": "2024-09-28",
            "symbol": "GOOD",
            "acceptedDate": "2024-10-30",
            "revenue": 1000000000,
            "netIncome": 100000000
        },
        # Missing acceptedDate
        {
            "date": "2024-06-30",
            "symbol": "NO_ACCEPTED",
            "revenue": 800000000,
            "netIncome": 80000000
        },
        # Negative revenue (data error)
        {
            "date": "2024-03-31",
            "symbol": "NEGATIVE",
            "acceptedDate": "2024-04-30",
            "revenue": -500000000,  # Problematic
            "netIncome": 50000000
        },
        # Zero values
        {
            "date": "2024-12-31",
            "symbol": "ZEROS",
            "acceptedDate": "2025-01-30",
            "revenue": 0,  # Potential issue
            "netIncome": 0  # Potential issue
        },
        # Missing key fields
        {
            "date": "2024-12-31",
            "symbol": "MISSING",
            "acceptedDate": "2025-01-30"
            # No revenue or netIncome
        }
    ]
    
    try:
        validated_problematic = fmp._validate_financial_data(problematic_data, "income_statement")
        quality_stats = fmp._perform_data_quality_checks(validated_problematic, "income_statement")
        
        print(f"   📊 Processed {len(problematic_data)} problematic records")
        print(f"   ✅ {len(validated_problematic)} records passed validation")
        print(f"   🎯 Quality score: {quality_stats['quality_score']}%")
        print(f"   ⚠️  Issues breakdown:")
        for issue_type, count in quality_stats['issues'].items():
            if count > 0:
                print(f"      - {issue_type}: {count}")
        
        if quality_stats['quality_score'] < 80:
            print(f"   ✅ Correctly identified data quality issues (score < 80%)")
        else:
            print(f"   ⚠️  Quality score unexpectedly high for problematic data")
            
    except Exception as e:
        print(f"   ❌ Problematic data quality test failed: {e}")
    
    print()
    
    # Test 3: Quality check with different data types
    print("📋 Test 3: Different Data Types Quality Check")
    print("-" * 40)
    
    data_types = ["balance_sheet", "cash_flow", "financial_ratios"]
    
    for data_type in data_types:
        try:
            if data_type == "balance_sheet":
                real_data = fmp._fetch_balance_sheet("AAPL", limit=1)
            elif data_type == "cash_flow":
                real_data = fmp._fetch_cash_flow("AAPL", limit=1)
            else:  # financial_ratios
                real_data = fmp._fetch_financial_ratios("AAPL", limit=1)
            
            if real_data:
                validated_data = fmp._validate_financial_data(real_data, data_type)
                quality_stats = fmp._perform_data_quality_checks(validated_data, data_type)
                
                print(f"   ✅ {data_type}: score={quality_stats['quality_score']}%, records={quality_stats['total_records']}")
            else:
                print(f"   ⚠️  {data_type}: no data available")
                
        except Exception as e:
            print(f"   ❌ {data_type} quality test failed: {e}")
    
    print()
    print("=" * 60)
    print("🏁 Data Quality Checks Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    test_data_quality_checks()
