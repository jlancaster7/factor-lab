"""
Basic unit tests for Factor Lab core functionality
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestFactorCalculator(unittest.TestCase):
    """Test FactorCalculator functionality"""

    def setUp(self):
        """Set up test data"""
        # Create sample price data
        dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
        np.random.seed(42)
        self.prices = pd.Series(
            100 * np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates))), index=dates
        )

        from factor_lab.factors import FactorCalculator

        self.calc = FactorCalculator()

    def test_momentum_calculation(self):
        """Test momentum factor calculation"""
        momentum = self.calc.momentum(self.prices, lookback=20)

        # Check that we get reasonable results
        self.assertIsInstance(momentum, pd.Series)
        self.assertGreater(len(momentum.dropna()), 0)
        self.assertTrue(np.isfinite(momentum.dropna()).all())

    def test_volatility_calculation(self):
        """Test volatility factor calculation"""
        # Calculate returns from prices for volatility calculation
        returns = self.prices.pct_change()
        volatility = self.calc.volatility(returns, lookback=20)

        # Check that we get reasonable results
        self.assertIsInstance(volatility, pd.Series)
        self.assertGreater(len(volatility.dropna()), 0)
        self.assertTrue(
            (volatility.dropna() >= 0).all()
        )  # Volatility should be non-negative

    def test_rsi_calculation(self):
        """Test RSI calculation"""
        rsi = self.calc.rsi(self.prices, period=14)

        # Check that we get reasonable results
        self.assertIsInstance(rsi, pd.Series)
        self.assertGreater(len(rsi.dropna()), 0)
        # RSI should be between 0 and 100
        self.assertTrue((rsi.dropna() >= 0).all())
        self.assertTrue((rsi.dropna() <= 100).all())


class TestDataManager(unittest.TestCase):
    """Test DataManager functionality"""

    def setUp(self):
        """Set up DataManager"""
        from factor_lab.data import DataManager

        self.dm = DataManager()

    def test_data_manager_initialization(self):
        """Test that DataManager initializes properly"""
        self.assertIsNotNone(self.dm)

    @unittest.skip("Requires internet connection and may be slow")
    def test_stock_data_retrieval(self):
        """Test stock data retrieval (skipped by default)"""
        data = self.dm.get_stock_data(
            ["AAPL"], start_date="2024-01-01", end_date="2024-01-10"
        )
        self.assertIsNotNone(data)
        self.assertGreater(len(data), 0)


class TestConfiguration(unittest.TestCase):
    """Test configuration loading"""

    def test_config_loading(self):
        """Test that configuration loads without errors"""
        try:
            from factor_lab.utils import load_config

            config = load_config()
            self.assertIsInstance(config, dict)
        except Exception as e:
            self.fail(f"Configuration loading failed: {e}")


if __name__ == "__main__":
    unittest.main()
