#!/usr/bin/env python3
"""
Debug portfolio optimization issues
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def debug_portfolio_issue():
    """Debug the portfolio optimization issue"""
    print("🔍 Debugging Portfolio Optimization Issue")
    print("=" * 50)

    try:
        from factor_lab.data import DataManager
        from factor_lab.portfolio import PortfolioOptimizer

        # Create data manager
        dm = DataManager()

        # Get a small universe for testing (same as notebook)
        universe = dm.get_universe("sp500")[:5]  # Even smaller for debugging
        print(f"Test universe: {universe}")

        # Get the same date range as in the notebook
        start_date = "2020-01-01"
        end_date = "2024-01-01"

        # Get returns data
        print(f"\nFetching returns data...")
        returns = dm.get_returns(universe, start_date, end_date)
        print(f"Returns data shape: {returns.shape}")
        print(f"Date range: {returns.index[0]} to {returns.index[-1]}")

        # Examine the returns data
        print(f"\nReturns data statistics:")
        print(f"Mean: {returns.mean().mean():.8f}")
        print(f"Std: {returns.std().mean():.8f}")
        print(f"Min: {returns.min().min():.8f}")
        print(f"Max: {returns.max().max():.8f}")
        print(f"Any NaN: {returns.isna().sum().sum()}")
        print(f"Any infinite: {np.isinf(returns).sum().sum()}")

        # Show sample returns
        print(f"\nSample returns (last 5 days):")
        print(returns.tail())

        # Use 2 years of data like in the notebook
        recent_returns = returns.tail(504)
        print(f"\nUsing {len(recent_returns)} days of data")

        # Try to create optimizer
        print(f"\nCreating PortfolioOptimizer...")
        optimizer = PortfolioOptimizer(recent_returns)

        # Check expected returns
        print(f"\nExpected returns:")
        print(optimizer.expected_returns)
        print(f"Expected returns type: {type(optimizer.expected_returns)}")
        print(f"Any NaN in expected returns: {optimizer.expected_returns.isna().sum()}")

        # Check covariance matrix
        print(f"\nCovariance matrix info:")
        print(f"Shape: {optimizer.cov_matrix.shape}")
        print(f"Diagonal (variances): {np.diag(optimizer.cov_matrix)}")

        eigenvals = np.linalg.eigvals(optimizer.cov_matrix)
        print(f"Eigenvalue range: [{np.min(eigenvals):.2e}, {np.max(eigenvals):.2e}]")
        print(f"Condition number: {np.max(eigenvals) / np.min(eigenvals):.2e}")

        # Try optimization
        print(f"\nTesting mean-variance optimization...")
        result = optimizer.mean_variance_optimization()

        print(f"Result: {result}")

        return True

    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    debug_portfolio_issue()
