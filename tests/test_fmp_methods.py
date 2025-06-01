#!/usr/bin/env python3
"""
Test script for FMP Provider raw data fetching methods.
Story 1.2 - Raw Data Fetching Methods validation.
"""

import sys
import os
sys.path.append('/home/jcl/pythonCode/factor_lab_2/factor-lab/src')

from factor_lab.data import FMPProvider
import json
import time

def main():
    print("=" * 60)
    print("🧪 TESTING FMP Provider Raw Data Fetching Methods")
    print("=" * 60)
    
    # Initialize FMP Provider
    try:
        fmp = FMPProvider()
        print(f"✅ FMP Provider initialized successfully")
        print(f"   API Key: {'*' * 10}{fmp.api_key[-4:]}")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize FMP Provider: {e}")
        return
    
    # Test symbol
    symbol = "AAPL"
    print(f"🔍 Testing with symbol: {symbol}")
    print()
    
    # Test 1: Income Statement
    print("📊 Test 1: Income Statement")
    print("-" * 40)
    try:
        start_time = time.time()
        income_data = fmp._fetch_income_statement(symbol, limit=2)
        elapsed = time.time() - start_time
        
        if income_data:
            print(f"✅ Success! Fetched {len(income_data)} records in {elapsed:.2f}s")
            print(f"   Sample keys: {list(income_data[0].keys())[:10]}...")
            print(f"   Latest date: {income_data[0].get('date', 'N/A')}")
            print(f"   Revenue: ${income_data[0].get('revenue', 'N/A'):,}")
            print()
        else:
            print(f"❌ No income statement data returned")
            print()
    except Exception as e:
        print(f"❌ Income statement test failed: {e}")
        print()
    
    # Test 2: Balance Sheet
    print("🏦 Test 2: Balance Sheet")
    print("-" * 40)
    try:
        start_time = time.time()
        balance_data = fmp._fetch_balance_sheet(symbol, limit=2)
        elapsed = time.time() - start_time
        
        if balance_data:
            print(f"✅ Success! Fetched {len(balance_data)} records in {elapsed:.2f}s")
            print(f"   Sample keys: {list(balance_data[0].keys())[:10]}...")
            print(f"   Latest date: {balance_data[0].get('date', 'N/A')}")
            print(f"   Total Assets: ${balance_data[0].get('totalAssets', 'N/A'):,}")
            print()
        else:
            print(f"❌ No balance sheet data returned")
            print()
    except Exception as e:
        print(f"❌ Balance sheet test failed: {e}")
        print()
    
    # Test 3: Cash Flow
    print("💰 Test 3: Cash Flow")
    print("-" * 40)
    try:
        start_time = time.time()
        cashflow_data = fmp._fetch_cash_flow(symbol, limit=2)
        elapsed = time.time() - start_time
        
        if cashflow_data:
            print(f"✅ Success! Fetched {len(cashflow_data)} records in {elapsed:.2f}s")
            print(f"   Sample keys: {list(cashflow_data[0].keys())[:10]}...")
            print(f"   Latest date: {cashflow_data[0].get('date', 'N/A')}")
            print(f"   Operating Cash Flow: ${cashflow_data[0].get('operatingCashFlow', 'N/A'):,}")
            print()
        else:
            print(f"❌ No cash flow data returned")
            print()
    except Exception as e:
        print(f"❌ Cash flow test failed: {e}")
        print()
    
    # Test 4: Financial Ratios
    print("📈 Test 4: Financial Ratios")
    print("-" * 40)
    try:
        start_time = time.time()
        ratios_data = fmp._fetch_financial_ratios(symbol, limit=2)
        elapsed = time.time() - start_time
        
        if ratios_data:
            print(f"✅ Success! Fetched {len(ratios_data)} records in {elapsed:.2f}s")
            print(f"   Sample keys: {list(ratios_data[0].keys())[:10]}...")
            print(f"   Latest date: {ratios_data[0].get('date', 'N/A')}")
            print(f"   P/E Ratio: {ratios_data[0].get('priceEarningsRatio', 'N/A')}")
            print(f"   ROE: {ratios_data[0].get('returnOnEquity', 'N/A')}")
            print()
        else:
            print(f"❌ No financial ratios data returned")
            print()
    except Exception as e:
        print(f"❌ Financial ratios test failed: {e}")
        print()
    
    # Test 5: Error Handling with Invalid Symbol
    print("🚫 Test 5: Error Handling (Invalid Symbol)")
    print("-" * 40)
    try:
        invalid_symbol = "INVALID_SYMBOL_123"
        start_time = time.time()
        invalid_data = fmp._fetch_income_statement(invalid_symbol, limit=1)
        elapsed = time.time() - start_time
        
        if invalid_data is None:
            print(f"✅ Correctly handled invalid symbol in {elapsed:.2f}s")
            print("   Returned None as expected")
            print()
        else:
            print(f"⚠️  Unexpected: Got data for invalid symbol")
            print()
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        print()
    
    # Test 6: Data Validation
    print("🔍 Test 6: Data Validation")
    print("-" * 40)
    try:
        # Get some real data first
        test_data = fmp._fetch_income_statement(symbol, limit=1)
        if test_data:
            # Test validation
            validated_data = fmp._validate_financial_data(test_data, "income_statement")
            print(f"✅ Data validation successful")
            print(f"   Original records: {len(test_data)}")
            print(f"   Validated records: {len(validated_data)}")
            if validated_data:
                print(f"   Sample validated keys: {list(validated_data[0].keys())[:10]}...")
            print()
        else:
            print(f"❌ No data to validate")
            print()
    except Exception as e:
        print(f"❌ Data validation test failed: {e}")
        print()
    
    print("=" * 60)
    print("🏁 FMP Raw Data Fetching Methods Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    main()
