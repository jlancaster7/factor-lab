#!/usr/bin/env python3
"""
Factor Lab CLI Example: Multi-Factor Strategy
=============================================

This script demonstrates building a sophisticated multi-factor strategy
combining momentum, volatility, and mean reversion factors.

Usage:
    python examples/multi_factor_strategy.py
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
    PerformanceAnalyzer,
    setup_logging,
)


def main():
    """Run multi-factor strategy example."""

    # Setup logging
    setup_logging()

    print("🚀 Factor Lab - Multi-Factor Strategy Example")
    print("=" * 55)

    # Initialize components
    data_manager = DataManager()
    factor_calc = FactorCalculator()
    optimizer = PortfolioOptimizer()
    backtester = Backtester()
    performance = PerformanceAnalyzer()
    charts = ChartManager()

    # Define larger universe for better factor diversification
    universe = [
        # Technology
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "NVDA",
        "NFLX",
        "ADBE",
        # Finance
        "JPM",
        "BAC",
        "WFC",
        "GS",
        "MS",
        "C",
        # Healthcare
        "JNJ",
        "PFE",
        "UNH",
        "ABBV",
        "MRK",
        "TMO",
        # Consumer
        "PG",
        "KO",
        "PEP",
        "WMT",
        "HD",
        "DIS",
        # Industrial
        "BA",
        "CAT",
        "GE",
        "MMM",
        "HON",
        "UPS",
        # Energy
        "XOM",
        "CVX",
        "COP",
        "SLB",
    ]

    print(f"📊 Fetching data for {len(universe)} stocks...")

    # Get price data
    prices = data_manager.get_stock_data(
        symbols=universe, start_date="2019-01-01", end_date="2024-01-01"
    )

    if prices.empty:
        print("❌ Failed to fetch price data")
        return

    print(f"✅ Retrieved {len(prices)} price observations")

    # Calculate multiple factors
    print("🔢 Calculating multi-factor signals...")

    all_factors = pd.DataFrame(index=prices.index)

    for symbol in universe:
        if symbol in prices.columns:
            symbol_prices = prices[symbol].dropna()

            # Skip if insufficient data
            if len(symbol_prices) < 100:
                continue

            print(f"   Processing {symbol}...")

            # 1. Momentum factors
            momentum_12_1 = factor_calc.calculate_momentum(
                symbol_prices, window=252, skip_periods=21
            )
            momentum_6_1 = factor_calc.calculate_momentum(
                symbol_prices, window=126, skip_periods=21
            )
            momentum_3_1 = factor_calc.calculate_momentum(
                symbol_prices, window=63, skip_periods=21
            )

            # 2. Volatility factor (negative - we want low volatility)
            volatility = -factor_calc.calculate_volatility(symbol_prices, window=60)

            # 3. Mean reversion factor
            mean_reversion = factor_calc.calculate_mean_reversion(
                symbol_prices, window=20
            )

            # 4. RSI factor (contrarian - buy low RSI)
            rsi = factor_calc.calculate_rsi(symbol_prices, window=14)
            rsi_factor = -(rsi - 50) / 50  # Convert to factor score

            # Store factors
            all_factors[f"{symbol}_momentum_12_1"] = momentum_12_1
            all_factors[f"{symbol}_momentum_6_1"] = momentum_6_1
            all_factors[f"{symbol}_momentum_3_1"] = momentum_3_1
            all_factors[f"{symbol}_volatility"] = volatility
            all_factors[f"{symbol}_mean_reversion"] = mean_reversion
            all_factors[f"{symbol}_rsi"] = rsi_factor

    # Create composite factor scores
    print("📈 Creating composite factor scores...")

    factor_scores = pd.DataFrame(index=prices.index)

    # Define factor weights
    factor_weights = {
        "momentum_12_1": 0.25,
        "momentum_6_1": 0.20,
        "momentum_3_1": 0.15,
        "volatility": 0.20,
        "mean_reversion": 0.10,
        "rsi": 0.10,
    }

    for date in all_factors.index:
        date_factors = all_factors.loc[date]

        # Group factors by symbol
        symbol_scores = {}

        for symbol in universe:
            if symbol in prices.columns:
                # Collect all factor values for this symbol
                factor_values = {}

                for factor_name, weight in factor_weights.items():
                    col_name = f"{symbol}_{factor_name}"
                    if col_name in date_factors.index and not pd.isna(
                        date_factors[col_name]
                    ):
                        factor_values[factor_name] = date_factors[col_name]

                # Calculate composite score if we have enough factors
                if len(factor_values) >= 4:  # Need at least 4 out of 6 factors
                    composite_score = sum(
                        factor_values[factor] * factor_weights[factor]
                        for factor in factor_values.keys()
                    )
                    symbol_scores[symbol] = composite_score

        # Normalize scores across symbols for this date
        if len(symbol_scores) >= 10:  # Need at least 10 stocks
            scores_series = pd.Series(symbol_scores)
            normalized_scores = factor_calc.normalize_factor(
                scores_series, method="z_score"
            )

            for symbol, score in normalized_scores.items():
                factor_scores.loc[date, symbol] = score

    # Portfolio construction and backtesting
    print("💼 Running portfolio optimization...")

    def multi_factor_strategy(date, scores, long_pct=0.3, short_pct=0.0):
        """
        Multi-factor long-short strategy.

        Args:
            date: Rebalancing date
            scores: Factor scores DataFrame
            long_pct: Percentage of universe to go long
            short_pct: Percentage of universe to go short (0 = long-only)
        """
        if date not in scores.index:
            return {}

        date_scores = scores.loc[date].dropna()

        if len(date_scores) < 20:  # Need sufficient universe
            return {}

        n_long = max(1, int(len(date_scores) * long_pct))
        n_short = int(len(date_scores) * short_pct)

        # Select long and short positions
        long_stocks = date_scores.nlargest(n_long)
        short_stocks = date_scores.nsmallest(n_short) if n_short > 0 else pd.Series()

        weights = {}

        # Long positions
        for stock in long_stocks.index:
            weights[stock] = 1.0 / n_long

        # Short positions (if any)
        for stock in short_stocks.index:
            weights[stock] = -1.0 / n_short if n_short > 0 else 0

        return weights

    # Run backtest with monthly rebalancing
    print("⚡ Running backtest...")

    rebalance_dates = factor_scores.index[::21]  # Monthly rebalancing
    portfolio_weights = {}

    for date in rebalance_dates:
        weights = multi_factor_strategy(date, factor_scores)
        if weights:
            portfolio_weights[date] = weights

    # Convert to DataFrame format
    weights_df = pd.DataFrame(index=prices.index, columns=universe).fillna(0.0)

    current_weights = {}
    for date in weights_df.index:
        # Update weights on rebalancing dates
        if date in portfolio_weights:
            current_weights = portfolio_weights[date]

        # Apply current weights
        for symbol in universe:
            if symbol in current_weights:
                weights_df.loc[date, symbol] = current_weights[symbol]

    # Run backtest
    backtest_results = backtester.run_backtest(
        prices=prices,
        weights=weights_df,
        transaction_costs=0.001,
        benchmark_symbol="SPY",
    )

    # Performance analysis
    print("\n📊 Performance Analysis:")
    print("=" * 35)

    if "portfolio_returns" in backtest_results:
        portfolio_returns = backtest_results["portfolio_returns"]

        # Calculate comprehensive metrics
        metrics = performance.calculate_metrics(portfolio_returns)

        print(f"📈 Total Return: {metrics.get('total_return', 0):.2%}")
        print(f"📊 Annualized Return: {metrics.get('annual_return', 0):.2%}")
        print(f"📉 Annualized Volatility: {metrics.get('annual_volatility', 0):.2%}")
        print(f"⚡ Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"📉 Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        print(f"📈 Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")

        # Benchmark comparison
        if "benchmark_returns" in backtest_results:
            benchmark_returns = backtest_results["benchmark_returns"]
            bench_metrics = performance.calculate_metrics(benchmark_returns)

            print(f"\n🏆 vs Benchmark (SPY):")
            print(
                f"   Strategy: {metrics.get('annual_return', 0):.2%} vs Benchmark: {bench_metrics.get('annual_return', 0):.2%}"
            )
            print(
                f"   Excess Return: {metrics.get('annual_return', 0) - bench_metrics.get('annual_return', 0):.2%}"
            )
            print(
                f"   Information Ratio: {(metrics.get('annual_return', 0) - bench_metrics.get('annual_return', 0)) / (portfolio_returns - benchmark_returns).std() / np.sqrt(252):.2f}"
            )

    # Factor exposure analysis
    print("\n🎯 Factor Exposure Analysis:")
    print("=" * 35)

    # Calculate average factor exposures
    non_zero_weights = weights_df[weights_df.abs() > 0.001]
    avg_long_positions = (weights_df > 0.001).sum(axis=1).mean()
    avg_short_positions = (weights_df < -0.001).sum(axis=1).mean()

    print(f"📊 Average Long Positions: {avg_long_positions:.1f}")
    print(f"📊 Average Short Positions: {avg_short_positions:.1f}")
    print(f"📊 Average Turnover: {weights_df.diff().abs().sum(axis=1).mean():.2%}")

    # Create visualizations
    print("\n📊 Creating visualizations...")

    try:
        figures_dir = Path("data/figures")
        figures_dir.mkdir(parents=True, exist_ok=True)

        if "portfolio_returns" in backtest_results:
            # 1. Equity curve comparison
            strategy_curve = (1 + portfolio_returns).cumprod()
            benchmark_curve = (
                (1 + backtest_results.get("benchmark_returns", pd.Series())).cumprod()
                if "benchmark_returns" in backtest_results
                else None
            )

            fig1 = charts.plot_cumulative_returns(
                strategy_curve,
                title="Multi-Factor Strategy vs Benchmark",
                benchmark_returns=benchmark_curve,
            )
            fig1.write_html(str(figures_dir / "multi_factor_equity_curve.html"))

            # 2. Rolling Sharpe ratio
            rolling_sharpe = (
                portfolio_returns.rolling(60).mean()
                / portfolio_returns.rolling(60).std()
                * np.sqrt(252)
            )

            import plotly.graph_objects as go

            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(
                    x=rolling_sharpe.index,
                    y=rolling_sharpe.values,
                    mode="lines",
                    name="60-Day Rolling Sharpe Ratio",
                    line=dict(color="blue", width=2),
                )
            )
            fig2.update_layout(
                title="Multi-Factor Strategy - Rolling Sharpe Ratio",
                xaxis_title="Date",
                yaxis_title="Sharpe Ratio",
                template="plotly_white",
                width=1000,
                height=500,
            )
            fig2.write_html(str(figures_dir / "multi_factor_rolling_sharpe.html"))

            print(f"💾 Saved visualizations to: {figures_dir}")

    except Exception as e:
        print(f"⚠️  Warning: Could not create visualizations: {e}")

    print("\n✅ Multi-factor strategy example completed successfully!")
    print(
        "💡 This strategy combines momentum, volatility, mean reversion, and RSI factors"
    )
    print("💡 Check the 'data/figures/' directory for detailed performance charts")


if __name__ == "__main__":
    main()
