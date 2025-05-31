#!/usr/bin/env python3
"""
Setup verification script for Factor Lab.

This script verifies that your Factor Lab installation is working correctly
by testing imports, API connectivity, and basic functionality.

Usage:
    python verify_setup.py

This is NOT a pytest test file - it's a standalone verification tool.
For unit tests, see tests/test_core.py
"""

import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_basic_imports():
    """Test that all basic imports work"""
    print("🧪 Testing basic imports...")
    try:
        import factor_lab
        from factor_lab.data import DataManager
        from factor_lab.factors import FactorCalculator
        from factor_lab.portfolio import PortfolioOptimizer
        from factor_lab.backtesting import Backtester
        from factor_lab.visualization import ChartManager
        from factor_lab.utils import load_config

        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_yahoo_finance():
    """Test Yahoo Finance data retrieval"""
    print("\n🧪 Testing Yahoo Finance connection...")
    try:
        # Get some basic stock data
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="5d")
        if len(data) > 0:
            print(f"✅ Yahoo Finance working! Retrieved {len(data)} days of AAPL data")
            print(f"   Latest close: ${data['Close'].iloc[-1]:.2f}")
            return True
        else:
            print("❌ Yahoo Finance returned empty data")
            return False
    except Exception as e:
        print(f"❌ Yahoo Finance error: {e}")
        return False


def test_openbb_import():
    """Test OpenBB Platform import"""
    print("\n🧪 Testing OpenBB Platform import...")
    try:
        from openbb import obb

        print("✅ OpenBB Platform imported successfully!")
        return True
    except ImportError as e:
        print(f"❌ OpenBB Platform import error: {e}")
        return False


def test_config_loading():
    """Test configuration loading"""
    print("\n🧪 Testing configuration loading...")
    try:
        from factor_lab.utils import load_config

        config = load_config()
        if config:
            print("✅ Configuration loaded successfully!")
            print(f"   Environment: {config.get('environment', 'default')}")
            return True
        else:
            print("❌ Configuration loading failed")
            return False
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


def test_factor_calculation():
    """Test basic factor calculation"""
    print("\n🧪 Testing factor calculation...")
    try:
        # Create sample price data
        dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
        np.random.seed(42)
        prices = pd.Series(
            100 * np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates))), index=dates
        )

        from factor_lab.factors import FactorCalculator

        calc = FactorCalculator()

        # Test momentum calculation
        momentum = calc.momentum(prices, lookback=20)
        if len(momentum.dropna()) > 0:
            print("✅ Factor calculation working!")
            print(f"   Sample momentum values: {momentum.dropna().tail(3).values}")
            return True
        else:
            print("❌ Factor calculation returned empty results")
            return False
    except Exception as e:
        print(f"❌ Factor calculation error: {e}")
        return False


def test_data_manager():
    """Test DataManager functionality"""
    print("\n🧪 Testing DataManager...")
    try:
        from factor_lab.data import DataManager

        dm = DataManager()

        # Test getting stock data
        data = dm.get_prices(["AAPL"], start_date="2024-01-01", end_date="2024-01-10")
        if data is not None and len(data) > 0:
            print("✅ DataManager working!")
            print(f"   Retrieved data shape: {data.shape}")
            return True
        else:
            print("❌ DataManager returned no data")
            return False
    except Exception as e:
        print(f"❌ DataManager error: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Factor Lab Setup Test Suite")
    print("=" * 50)

    tests = [
        test_basic_imports,
        test_yahoo_finance,
        test_openbb_import,
        test_config_loading,
        test_factor_calculation,
        test_data_manager,
    ]

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    passed = sum(results)
    total = len(results)
    print(f"   Passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! Factor Lab is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
