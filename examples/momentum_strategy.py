#!/usr/bin/env python3
"""
Factor Lab CLI Example: Basic Momentum Strategy
===============================================

This script demonstrates how to build and backtest a simple momentum-based
factor strategy using the Factor Lab CLI interface.

Usage:
    python examples/momentum_strategy.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from factor_lab import (
    DataManager,
    FactorCalculator,
    PortfolioOptimizer,
    Backtester,
    ChartManager,
    setup_logging,
)


def main():
    """Run momentum strategy example."""

    # Setup logging
    setup_logging()

    print("🚀 Factor Lab - Momentum Strategy Example")
    print("=" * 50)

    # 1. Initialize components
    data_manager = DataManager()
    factor_calc = FactorCalculator()
    optimizer = PortfolioOptimizer()
    backtester = Backtester()
    charts = ChartManager()

    # 2. Define universe
    universe = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "TSLA",
        "META",
        "NVDA",
        "JPM",
        "JNJ",
        "V",
        "PG",
        "UNH",
        "HD",
        "MA",
        "PYPL",
    ]

    print(f"📊 Fetching data for {len(universe)} stocks...")

    # 3. Get price data
    prices = data_manager.get_stock_data(
        symbols=universe, start_date="2020-01-01", end_date="2024-01-01"
    )

    if prices.empty:
        print("❌ Failed to fetch price data")
        return

    print(f"✅ Retrieved {len(prices)} price observations")

    # 4. Calculate momentum factors
    print("🔢 Calculating momentum factors...")

    factors = pd.DataFrame(index=prices.index)

    for symbol in universe:
        if symbol in prices.columns:
            symbol_prices = prices[symbol].dropna()

            # Calculate multiple momentum signals
            mom_12_1 = factor_calc.calculate_momentum(
                symbol_prices, window=252, skip_periods=21
            )  # 12-1 momentum
            mom_6_1 = factor_calc.calculate_momentum(
                symbol_prices, window=126, skip_periods=21
            )  # 6-1 momentum
            mom_3_1 = factor_calc.calculate_momentum(
                symbol_prices, window=63, skip_periods=21
            )  # 3-1 momentum

            # Combine momentum signals (equal weight)
            combined_momentum = (mom_12_1 + mom_6_1 + mom_3_1) / 3
            factors[f"{symbol}_momentum"] = combined_momentum

    # 5. Create factor scores
    print("📈 Creating factor scores...")

    factor_scores = pd.DataFrame(index=prices.index)

    for date in factors.index:
        date_factors = factors.loc[date]

        # Extract momentum values for this date
        momentum_values = {}
        for col in date_factors.index:
            if col.endswith("_momentum") and not pd.isna(date_factors[col]):
                symbol = col.replace("_momentum", "")
                momentum_values[symbol] = date_factors[col]

        if len(momentum_values) >= 5:  # Need at least 5 stocks
            # Rank and normalize
            momentum_df = pd.Series(momentum_values)
            normalized_scores = factor_calc.normalize_factor(momentum_df, method="rank")

            for symbol, score in normalized_scores.items():
                factor_scores.loc[date, symbol] = score

    # 6. Portfolio optimization and backtesting
    print("💼 Running portfolio optimization and backtesting...")

    # Simple strategy: Long top momentum stocks
    def momentum_strategy(date, scores, top_n=8):
        """Select top momentum stocks."""
        if date not in scores.index:
            return {}

        date_scores = scores.loc[date].dropna()
        if len(date_scores) < top_n:
            return {}

        # Select top momentum stocks
        top_stocks = date_scores.nlargest(top_n)

        # Equal weight allocation
        weights = {stock: 1.0 / top_n for stock in top_stocks.index}
        return weights

    # Run backtest
    rebalance_dates = factor_scores.index[::21]  # Monthly rebalancing
    portfolio_weights = {}

    for date in rebalance_dates:
        weights = momentum_strategy(date, factor_scores)
        if weights:
            portfolio_weights[date] = weights

    # Convert to DataFrame format for backtester
    weights_df = pd.DataFrame(index=prices.index, columns=universe).fillna(0.0)

    for date, weights in portfolio_weights.items():
        for symbol, weight in weights.items():
            if symbol in weights_df.columns:
                # Forward fill weights until next rebalance
                next_dates = weights_df.index[weights_df.index >= date]
                if len(next_dates) > 0:
                    end_idx = min(len(next_dates), 21)  # Until next rebalance
                    for i in range(end_idx):
                        if date + pd.Timedelta(days=i * 1) in weights_df.index:
                            weights_df.loc[date + pd.Timedelta(days=i * 1), symbol] = (
                                weight
                            )

    # 7. Run backtest
    backtest_results = backtester.run_backtest(
        prices=prices,
        weights=weights_df,
        transaction_costs=0.001,
        benchmark_symbol="SPY",
    )

    # 8. Display results
    print("\n📊 Backtest Results:")
    print("=" * 30)

    if "portfolio_returns" in backtest_results:
        portfolio_returns = backtest_results["portfolio_returns"]

        # Calculate key metrics
        total_return = (1 + portfolio_returns).prod() - 1
        annual_return = (1 + portfolio_returns.mean()) ** 252 - 1
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        max_drawdown = (
            portfolio_returns.cumsum() - portfolio_returns.cumsum().expanding().max()
        ).min()

        print(f"📈 Total Return: {total_return:.2%}")
        print(f"📊 Annualized Return: {annual_return:.2%}")
        print(f"📉 Annualized Volatility: {annual_vol:.2%}")
        print(f"⚡ Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"📉 Max Drawdown: {max_drawdown:.2%}")

        # Compare to benchmark if available
        if "benchmark_returns" in backtest_results:
            benchmark_returns = backtest_results["benchmark_returns"]
            bench_total_return = (1 + benchmark_returns).prod() - 1
            bench_annual_return = (1 + benchmark_returns.mean()) ** 252 - 1

            print(f"\n🏆 Benchmark Comparison:")
            print(
                f"   Strategy: {annual_return:.2%} vs Benchmark: {bench_annual_return:.2%}"
            )
            print(f"   Excess Return: {annual_return - bench_annual_return:.2%}")

    # 9. Create visualizations
    print("\n📊 Creating visualizations...")

    try:
        # Plot equity curve
        if "portfolio_returns" in backtest_results:
            equity_curve = (1 + backtest_results["portfolio_returns"]).cumprod()

            fig = charts.plot_cumulative_returns(
                equity_curve,
                title="Momentum Strategy - Equity Curve",
                benchmark_returns=(
                    (
                        1 + backtest_results.get("benchmark_returns", pd.Series())
                    ).cumprod()
                    if "benchmark_returns" in backtest_results
                    else None
                ),
            )

            # Save plot
            output_path = Path("data/figures/momentum_strategy_equity_curve.html")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_path))
            print(f"💾 Saved equity curve to: {output_path}")

    except Exception as e:
        print(f"⚠️  Warning: Could not create visualizations: {e}")

    print("\n✅ Momentum strategy example completed successfully!")
    print("💡 Check the 'data/figures/' directory for saved charts")


if __name__ == "__main__":
    main()
