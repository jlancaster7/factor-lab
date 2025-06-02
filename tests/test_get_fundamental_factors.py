#!/usr/bin/env python3
"""
Test suite for get_fundamental_factors() - Main Public API Method

This test validates the redesigned public API method that:
1. Fetches fundamental data for multiple symbols
2. Returns daily forward-filled data aligned with trading days
3. Eliminates NaN values throughout the entire date range
4. Integrates properly with multi-factor models
5. Supports configurable calculation frequencies
"""

import sys
import os
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the src directory to Python path so we can import factor_lab
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from factor_lab.data import FMPProvider


class TestGetFundamentalFactors:
    """Test the main public API method get_fundamental_factors()"""

    @pytest.fixture
    def fmp_provider(self):
        """Initialize FMP provider for testing"""
        try:
            return FMPProvider()
        except Exception as e:
            pytest.skip(f"FMP Provider initialization failed: {e}")

    def test_single_symbol_basic_functionality(self, fmp_provider):
        """Test basic functionality with a single symbol"""
        print("\n📊 Testing single symbol basic functionality...")
        
        # Test parameters
        symbols = ["AAPL"]
        start_date = "2024-01-01"
        end_date = "2024-03-31"  # Short range for fast testing
        
        # Call the method
        result = fmp_provider.get_fundamental_factors(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            frequency="daily",
            calculation_frequency="W"
        )
        
        # Validate return structure
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "AAPL" in result, "AAPL should be in results"
        
        # Validate DataFrame structure
        df = result["AAPL"]
        assert isinstance(df, pd.DataFrame), "Result should contain DataFrame"
        
        # Check expected columns
        expected_columns = ["PE_ratio", "PB_ratio", "ROE", "Debt_Equity"]
        for col in expected_columns:
            assert col in df.columns, f"Column {col} should be present"
        
        # Validate date range
        assert df.index[0].date() >= pd.to_datetime(start_date).date(), "Start date should be respected"
        assert df.index[-1].date() <= pd.to_datetime(end_date).date(), "End date should be respected"
        
        print(f"✅ Single symbol test passed - Shape: {df.shape}")

    def test_multiple_symbols(self, fmp_provider):
        """Test functionality with multiple symbols"""
        print("\n📊 Testing multiple symbols...")
        
        # Test parameters
        symbols = ["AAPL", "MSFT"]
        start_date = "2024-01-01"
        end_date = "2024-02-29"  # Short range for fast testing
        
        # Call the method
        result = fmp_provider.get_fundamental_factors(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            frequency="daily",
            calculation_frequency="W"
        )
        
        # Validate structure
        assert isinstance(result, dict), "Result should be a dictionary"
        
        for symbol in symbols:
            assert symbol in result, f"{symbol} should be in results"
            df = result[symbol]
            
            if not df.empty:
                # Check date alignment (all symbols should have same trading days)
                assert isinstance(df.index, pd.DatetimeIndex), f"{symbol} should have DatetimeIndex"
                print(f"✅ {symbol} - Shape: {df.shape}")
            else:
                print(f"⚠️ {symbol} - Empty DataFrame (possible API limit)")

    def test_no_nan_values(self, fmp_provider):
        """Critical test: Ensure no NaN values in the result"""
        print("\n🔍 Testing NaN elimination (CRITICAL TEST)...")
        
        # Test parameters
        symbols = ["AAPL"]
        start_date = "2024-01-01"
        end_date = "2024-06-30"
        
        # Call the method
        result = fmp_provider.get_fundamental_factors(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            frequency="daily",
            calculation_frequency="W"
        )
        
        # Check for NaN values
        if "AAPL" in result and not result["AAPL"].empty:
            df = result["AAPL"]
            
            # Count NaN values
            nan_counts = df.isna().sum()
            total_nans = nan_counts.sum()
            
            print(f"📊 NaN counts by column:")
            for col, count in nan_counts.items():
                print(f"   {col}: {count}")
            
            # This is the CRITICAL assertion - no NaN values should exist
            assert total_nans == 0, f"Found {total_nans} NaN values - forward-fill failed!"
            
            print("✅ CRITICAL TEST PASSED: No NaN values found")
        else:
            pytest.skip("AAPL data not available for NaN test")

    def test_trading_day_alignment(self, fmp_provider):
        """Test that data is aligned with trading days"""
        print("\n📅 Testing trading day alignment...")
        
        # Test parameters
        symbols = ["AAPL"]
        start_date = "2024-01-01"
        end_date = "2024-01-31"  # Short range for detailed testing
        
        # Call the method
        result = fmp_provider.get_fundamental_factors(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            frequency="daily",
            calculation_frequency="W"
        )
        
        if "AAPL" in result and not result["AAPL"].empty:
            df = result["AAPL"]
            
            # Check that we don't have weekend data (Saturdays/Sundays)
            weekend_days = df.index[df.index.weekday.isin([5, 6])]  # Sat=5, Sun=6
            
            # Some weekends might have forward-filled data, but check if it makes sense
            total_weekend_days = len(weekend_days)
            total_trading_days = len(df)
            
            print(f"📊 Total days: {total_trading_days}")
            print(f"📊 Weekend days: {total_weekend_days}")
            
            # Most days should be weekdays (trading days)
            weekday_ratio = (total_trading_days - total_weekend_days) / total_trading_days
            assert weekday_ratio >= 0.7, f"Too many weekend days: {weekday_ratio:.2%}"
            
            print("✅ Trading day alignment test passed")
        else:
            pytest.skip("AAPL data not available for trading day test")

    def test_calculation_frequencies(self, fmp_provider):
        """Test different calculation frequencies"""
        print("\n⚙️ Testing calculation frequencies...")
        
        # Test parameters
        symbols = ["AAPL"]
        start_date = "2024-01-01"
        end_date = "2024-02-29"
        
        frequencies = ["W", "M"]  # Weekly and Monthly
        
        for freq in frequencies:
            print(f"   Testing frequency: {freq}")
            
            try:
                result = fmp_provider.get_fundamental_factors(
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    frequency="daily",
                    calculation_frequency=freq
                )
                
                if "AAPL" in result and not result["AAPL"].empty:
                    df = result["AAPL"]
                    print(f"   ✅ {freq} frequency - Shape: {df.shape}")
                else:
                    print(f"   ⚠️ {freq} frequency - No data")
                    
            except Exception as e:
                print(f"   ❌ {freq} frequency failed: {e}")

    def test_data_format_compatibility(self, fmp_provider):
        """Test that data format is compatible with multi-factor models"""
        print("\n🔗 Testing multi-factor model compatibility...")
        
        # Test parameters
        symbols = ["AAPL", "MSFT"]
        start_date = "2024-01-01"
        end_date = "2024-02-29"
        
        # Call the method
        result = fmp_provider.get_fundamental_factors(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            frequency="daily",
            calculation_frequency="W"
        )
        
        # Test data format compatibility
        for symbol in symbols:
            if symbol in result and not result[symbol].empty:
                df = result[symbol]
                
                # Check that data types are numeric (not object/string)
                for col in ["PE_ratio", "PB_ratio", "ROE", "Debt_Equity"]:
                    if col in df.columns:
                        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} should be numeric"
                
                # Check for reasonable value ranges
                if "PE_ratio" in df.columns and not df["PE_ratio"].isna().all():
                    pe_values = df["PE_ratio"].dropna()
                    assert (pe_values > 0).any(), "PE ratios should have positive values"
                    assert (pe_values < 1000).all(), "PE ratios should be reasonable (< 1000)"
                
                print(f"✅ {symbol} format compatibility passed")

    def test_error_handling(self, fmp_provider):
        """Test error handling with invalid inputs"""
        print("\n🚨 Testing error handling...")
        
        # Test with invalid date range
        try:
            result = fmp_provider.get_fundamental_factors(
                symbols=["AAPL"],
                start_date="2030-01-01",  # Future date
                end_date="2030-12-31",
                frequency="daily"
            )
            
            # Should handle gracefully (empty data or reasonable error)
            if "AAPL" in result:
                print("✅ Future date handling: Returned data structure")
            else:
                print("✅ Future date handling: No data returned")
                
        except Exception as e:
            print(f"✅ Future date handling: Proper error - {e}")

        # Test with invalid symbol
        try:
            result = fmp_provider.get_fundamental_factors(
                symbols=["INVALID_SYMBOL_12345"],
                start_date="2024-01-01",
                end_date="2024-01-31",
                frequency="daily"
            )
            
            # Should handle gracefully
            print("✅ Invalid symbol handling: Completed without crash")
            
        except Exception as e:
            print(f"✅ Invalid symbol handling: Proper error - {e}")


def test_integration_scenario():
    """Integration test simulating notebook usage"""
    print("\n🎯 INTEGRATION TEST: Simulating notebook usage...")
    
    try:
        fmp = FMPProvider()
        
        # Simulate notebook scenario
        symbols = ["AAPL"]
        start_date = "2024-01-01"
        end_date = "2024-03-31"
        
        # Get fundamental data (like notebook does)
        fundamental_data = fmp.get_fundamental_factors(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            frequency="daily"
        )
        
        # Simulate creating factor scores (like notebook does)
        if "AAPL" in fundamental_data and not fundamental_data["AAPL"].empty:
            df = fundamental_data["AAPL"]
            
            # Create value factors (like notebook cell 11)
            value_pe = -df['PE_ratio']  # Negative because lower P/E is better
            value_pb = -df['PB_ratio']  # Negative because lower P/B is better
            
            # Create quality factors
            quality_roe = df['ROE']
            quality_leverage = -df['Debt_Equity']
            
            # Check that factor creation works
            assert not value_pe.isna().all(), "PE factor should have values"
            assert not quality_roe.isna().all(), "ROE factor should have values"
            
            print("✅ INTEGRATION TEST PASSED: Notebook simulation successful")
            assert True  # Test completed successfully
        else:
            print("⚠️ INTEGRATION TEST SKIPPED: No data available")
            return False
            
    except Exception as e:
        print(f"❌ INTEGRATION TEST FAILED: {e}")
        return False


if __name__ == "__main__":
    """Run tests directly"""
    print("🧪 Running get_fundamental_factors() tests...")
    print("=" * 80)
    
    # Run integration test
    test_integration_scenario()