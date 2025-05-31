"""
Backtesting module for the Factor Investing Laboratory.

Implements comprehensive backtesting framework including:
- Strategy simulation
- Performance analysis
- Risk management
- Transaction cost modeling
- Factor strategy backtesting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Callable, Tuple
import logging
from datetime import datetime, timedelta
import warnings

logger = logging.getLogger(__name__)


class Backtester:
    """Main backtesting engine for factor strategies."""

    def __init__(
        self,
        start_date: str,
        end_date: str,
        initial_capital: float = 100000,
        transaction_cost: float = 0.001,
        rebalance_frequency: str = "monthly",
    ):
        """
        Initialize Backtester.

        Parameters:
        -----------
        start_date : str
            Backtest start date (YYYY-MM-DD)
        end_date : str
            Backtest end date (YYYY-MM-DD)
        initial_capital : float
            Initial portfolio capital
        transaction_cost : float
            Transaction cost as fraction of trade value
        rebalance_frequency : str
            Rebalancing frequency ('daily', 'weekly', 'monthly', 'quarterly')
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.rebalance_frequency = rebalance_frequency

        # Results storage
        self.portfolio_values = pd.Series(dtype=float)
        self.positions = pd.DataFrame()
        self.trades = pd.DataFrame()
        self.performance_metrics = {}

    def get_rebalance_dates(self, price_data: pd.DataFrame) -> List[pd.Timestamp]:
        """
        Generate rebalancing dates based on frequency.

        Parameters:
        -----------
        price_data : pd.DataFrame
            Price data to get date range from

        Returns:
        --------
        List[pd.Timestamp]
            List of rebalancing dates
        """
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq="D")
        available_dates = price_data.index.intersection(date_range)

        if self.rebalance_frequency == "daily":
            return list(available_dates)
        elif self.rebalance_frequency == "weekly":
            return list(available_dates[available_dates.weekday == 4])  # Fridays
        elif self.rebalance_frequency == "monthly":
            # Get last date of each month
            monthly_groups = available_dates.to_series().groupby(
                [available_dates.year, available_dates.month]
            )
            return list(monthly_groups.max())
        elif self.rebalance_frequency == "quarterly":
            # Get last date of each quarter
            quarters = available_dates.to_period("Q")
            return list(available_dates.groupby(quarters.astype(str)).max())
        else:
            raise ValueError(
                f"Unknown rebalancing frequency: {self.rebalance_frequency}"
            )

    def calculate_transaction_costs(
        self, old_weights: pd.Series, new_weights: pd.Series, portfolio_value: float
    ) -> float:
        """
        Calculate transaction costs for rebalancing.

        Parameters:
        -----------
        old_weights : pd.Series
            Previous portfolio weights
        new_weights : pd.Series
            New target weights
        portfolio_value : float
            Current portfolio value

        Returns:
        --------
        float
            Total transaction costs
        """
        # Align weights
        all_assets = old_weights.index.union(new_weights.index)
        old_aligned = old_weights.reindex(all_assets).fillna(0)
        new_aligned = new_weights.reindex(all_assets).fillna(0)

        # Calculate turnover (sum of absolute weight changes)
        turnover = (old_aligned - new_aligned).abs().sum()

        # Transaction costs
        return turnover * portfolio_value * self.transaction_cost

    def run_factor_strategy_backtest(
        self,
        price_data: pd.DataFrame,
        factor_data: pd.DataFrame,
        strategy_func: Callable,
        lookback_window: int = 252,
        **strategy_kwargs,
    ) -> Dict:
        """
        Run backtest for a factor-based strategy.

        Parameters:
        -----------
        price_data : pd.DataFrame
            Historical price data
        factor_data : pd.DataFrame
            Factor scores data
        strategy_func : Callable
            Strategy function that takes factor data and returns weights
        lookback_window : int
            Lookback window for factor calculation
        **strategy_kwargs
            Additional arguments for strategy function

        Returns:
        --------
        Dict
            Backtest results
        """
        # Get rebalancing dates
        rebalance_dates = self.get_rebalance_dates(price_data)

        # Initialize tracking variables
        portfolio_value = self.initial_capital
        current_weights = pd.Series(dtype=float)

        portfolio_values = []
        positions_history = []
        trades_history = []

        for i, rebal_date in enumerate(rebalance_dates):
            try:
                # Get available data up to rebalancing date
                available_prices = price_data.loc[:rebal_date]
                available_factors = factor_data.loc[:rebal_date]

                # Skip if insufficient data
                if len(available_prices) < lookback_window:
                    continue

                # Get factor data for strategy
                factor_window = available_factors.tail(lookback_window)

                # Calculate new weights using strategy function
                new_weights = strategy_func(
                    factor_window, available_prices, **strategy_kwargs
                )

                if new_weights is None or len(new_weights) == 0:
                    continue

                # Normalize weights to sum to 1
                if isinstance(new_weights, dict):
                    new_weights = pd.Series(new_weights)
                new_weights = new_weights / new_weights.sum()

                # Calculate transaction costs
                transaction_costs = 0
                if len(current_weights) > 0:
                    transaction_costs = self.calculate_transaction_costs(
                        current_weights, new_weights, portfolio_value
                    )

                # Apply transaction costs
                portfolio_value -= transaction_costs

                # Record trade if significant change
                if len(current_weights) > 0:
                    weight_changes = (
                        (
                            new_weights.reindex(current_weights.index).fillna(0)
                            - current_weights
                        )
                        .abs()
                        .sum()
                    )
                    if weight_changes > 0.01:  # 1% threshold
                        trades_history.append(
                            {
                                "Date": rebal_date,
                                "Transaction_Costs": transaction_costs,
                                "Turnover": weight_changes,
                            }
                        )

                # Update current weights
                current_weights = new_weights.copy()

                # Calculate portfolio performance until next rebalance
                if i < len(rebalance_dates) - 1:
                    next_rebal_date = rebalance_dates[i + 1]
                else:
                    next_rebal_date = self.end_date

                # Get returns for the period
                period_prices = price_data.loc[rebal_date:next_rebal_date]
                if len(period_prices) > 1:
                    period_returns = period_prices.pct_change().dropna()

                    # Calculate daily portfolio values
                    for date, returns in period_returns.iterrows():
                        # Calculate portfolio return
                        portfolio_return = (
                            current_weights.reindex(returns.index).fillna(0) * returns
                        ).sum()
                        portfolio_value *= 1 + portfolio_return

                        portfolio_values.append(
                            {
                                "Date": date,
                                "Portfolio_Value": portfolio_value,
                                "Portfolio_Return": portfolio_return,
                            }
                        )

                # Record positions
                positions_history.append(
                    {
                        "Date": rebal_date,
                        "Portfolio_Value": portfolio_value,
                        **current_weights.to_dict(),
                    }
                )

            except Exception as e:
                logger.warning(f"Error processing rebalance date {rebal_date}: {e}")
                continue

        # Convert results to DataFrames
        self.portfolio_values = pd.DataFrame(portfolio_values).set_index("Date")[
            "Portfolio_Value"
        ]
        self.positions = pd.DataFrame(positions_history).set_index("Date")
        self.trades = (
            pd.DataFrame(trades_history).set_index("Date")
            if trades_history
            else pd.DataFrame()
        )

        # Calculate performance metrics
        returns_series = pd.DataFrame(portfolio_values).set_index("Date")[
            "Portfolio_Return"
        ]
        self.performance_metrics = self._calculate_performance_metrics(returns_series)

        return {
            "portfolio_values": self.portfolio_values,
            "positions": self.positions,
            "trades": self.trades,
            "performance_metrics": self.performance_metrics,
        }

    def run_weights_backtest(
        self,
        prices: pd.DataFrame,
        weights: pd.DataFrame,
        transaction_costs: float = 0.001,
        benchmark_symbol: Optional[str] = None,
    ) -> Dict:
        """
        Run backtest with pre-calculated weights.

        Parameters:
        -----------
        prices : pd.DataFrame
            Historical price data
        weights : pd.DataFrame
            Pre-calculated portfolio weights (dates x symbols)
        transaction_costs : float
            Transaction cost as fraction of trade value
        benchmark_symbol : str, optional
            Benchmark symbol for comparison

        Returns:
        --------
        Dict
            Backtest results including portfolio returns and benchmark comparison
        """
        # Align prices and weights
        common_dates = prices.index.intersection(weights.index)
        common_symbols = prices.columns.intersection(weights.columns)

        if len(common_dates) == 0 or len(common_symbols) == 0:
            raise ValueError("No common dates or symbols between prices and weights")

        aligned_prices = prices.loc[common_dates, common_symbols]
        aligned_weights = weights.loc[common_dates, common_symbols].fillna(0)

        # Calculate daily returns
        price_returns = aligned_prices.pct_change().dropna()

        # Initialize tracking variables
        portfolio_value = self.initial_capital
        portfolio_values = [portfolio_value]
        portfolio_returns = []
        transaction_cost_history = []

        prev_weights = pd.Series(0, index=common_symbols)

        for date in price_returns.index:
            if date in aligned_weights.index:
                # Get current target weights
                current_weights = aligned_weights.loc[date]
                current_weights = (
                    current_weights / current_weights.sum()
                    if current_weights.sum() > 0
                    else current_weights
                )

                # Calculate transaction costs for rebalancing
                weight_changes = (current_weights - prev_weights).abs().sum()
                transaction_cost = weight_changes * portfolio_value * transaction_costs
                portfolio_value -= transaction_cost
                transaction_cost_history.append(transaction_cost)

                prev_weights = current_weights
            else:
                # No rebalancing, use previous weights
                current_weights = prev_weights
                transaction_cost_history.append(0)

            # Calculate portfolio return for this day
            if date in price_returns.index:
                day_returns = price_returns.loc[date]
                portfolio_return = (current_weights * day_returns).sum()
                portfolio_value *= 1 + portfolio_return
                portfolio_returns.append(portfolio_return)
                portfolio_values.append(portfolio_value)

        # Create results series
        portfolio_returns_series = pd.Series(
            portfolio_returns, index=price_returns.index
        )
        portfolio_values_series = pd.Series(
            portfolio_values[1:], index=price_returns.index
        )

        # Get benchmark returns if specified
        benchmark_returns = None
        if benchmark_symbol:
            try:
                from factor_lab.data import DataManager

                data_manager = DataManager()
                benchmark_prices = data_manager.get_prices(
                    symbols=[benchmark_symbol],
                    start_date=str(common_dates[0].date()),
                    end_date=str(common_dates[-1].date()),
                )
                if benchmark_symbol in benchmark_prices.columns:
                    benchmark_returns = (
                        benchmark_prices[benchmark_symbol].pct_change().dropna()
                    )
                    benchmark_returns = benchmark_returns.reindex(
                        portfolio_returns_series.index
                    )
            except Exception as e:
                logger.warning(
                    f"Could not fetch benchmark data for {benchmark_symbol}: {e}"
                )

        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            portfolio_returns_series
        )

        # Prepare results
        results = {
            "portfolio_returns": portfolio_returns_series,
            "portfolio_values": portfolio_values_series,
            "performance_metrics": performance_metrics,
            "transaction_costs": sum(transaction_cost_history),
        }

        if benchmark_returns is not None:
            results["benchmark_returns"] = benchmark_returns
            # Add benchmark comparison metrics if we have benchmark data
            try:
                benchmark_metrics = self._calculate_performance_metrics(
                    benchmark_returns
                )
                results["benchmark_metrics"] = benchmark_metrics
            except Exception as e:
                logger.warning(f"Could not calculate benchmark metrics: {e}")

        return results

    def _calculate_performance_metrics(self, returns: pd.Series) -> Dict:
        """Calculate comprehensive performance metrics."""
        returns_clean = returns.dropna()

        if len(returns_clean) == 0:
            return {}

        # Basic metrics
        total_return = (1 + returns_clean).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns_clean)) - 1
        annualized_vol = returns_clean.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0

        # Drawdown analysis
        cumulative_returns = (1 + returns_clean).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown = drawdown.min()

        # Risk metrics
        var_95 = returns_clean.quantile(0.05)
        var_99 = returns_clean.quantile(0.01)

        # Higher moments
        skewness = returns_clean.skew()
        kurtosis = returns_clean.kurtosis()

        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win/Loss statistics
        positive_returns = returns_clean[returns_clean > 0]
        negative_returns = returns_clean[returns_clean < 0]

        win_rate = (
            len(positive_returns) / len(returns_clean) if len(returns_clean) > 0 else 0
        )
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0

        return {
            "Total Return": total_return,
            "Annualized Return": annualized_return,
            "Annualized Volatility": annualized_vol,
            "Sharpe Ratio": sharpe_ratio,
            "Calmar Ratio": calmar_ratio,
            "Max Drawdown": max_drawdown,
            "VaR 95%": var_95,
            "VaR 99%": var_99,
            "Skewness": skewness,
            "Kurtosis": kurtosis,
            "Win Rate": win_rate,
            "Average Win": avg_win,
            "Average Loss": avg_loss,
        }


class PerformanceAnalyzer:
    """Advanced performance analysis and attribution."""

    def __init__(self):
        """Initialize PerformanceAnalyzer."""
        pass

    def compare_strategies(self, strategy_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple strategy backtest results.

        Parameters:
        -----------
        strategy_results : Dict[str, Dict]
            Dictionary of strategy names and their backtest results

        Returns:
        --------
        pd.DataFrame
            Comparison table of strategy metrics
        """
        comparison_data = {}

        for strategy_name, results in strategy_results.items():
            if "performance_metrics" in results:
                comparison_data[strategy_name] = results["performance_metrics"]

        return pd.DataFrame(comparison_data).T

    def rolling_performance_analysis(
        self, portfolio_values: pd.Series, window: int = 252
    ) -> pd.DataFrame:
        """
        Analyze rolling performance metrics.

        Parameters:
        -----------
        portfolio_values : pd.Series
            Portfolio value time series
        window : int
            Rolling window size

        Returns:
        --------
        pd.DataFrame
            Rolling performance metrics
        """
        returns = portfolio_values.pct_change().dropna()

        rolling_metrics = pd.DataFrame(index=returns.index)

        # Rolling metrics
        rolling_metrics["Rolling_Return"] = returns.rolling(window).mean() * 252
        rolling_metrics["Rolling_Volatility"] = returns.rolling(window).std() * np.sqrt(
            252
        )
        rolling_metrics["Rolling_Sharpe"] = (
            rolling_metrics["Rolling_Return"] / rolling_metrics["Rolling_Volatility"]
        )

        # Rolling max drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.rolling(window).max()
        rolling_metrics["Rolling_Drawdown"] = cumulative / rolling_max - 1
        rolling_metrics["Rolling_Max_Drawdown"] = (
            rolling_metrics["Rolling_Drawdown"].rolling(window).min()
        )

        return rolling_metrics.dropna()

    def factor_exposure_analysis(
        self, positions: pd.DataFrame, factor_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Analyze portfolio factor exposures over time.

        Parameters:
        -----------
        positions : pd.DataFrame
            Portfolio positions over time
        factor_data : pd.DataFrame
            Factor scores data

        Returns:
        --------
        pd.DataFrame
            Portfolio factor exposures over time
        """
        exposures = pd.DataFrame(index=positions.index)

        # Get common assets
        common_assets = set(positions.columns) & set(factor_data.columns)

        if not common_assets:
            logger.warning("No common assets between positions and factor data")
            return exposures

        for factor_name in factor_data.index:
            factor_scores = factor_data.loc[factor_name, common_assets]

            # Calculate weighted exposure for each date
            exposures[factor_name] = (
                positions[common_assets].multiply(factor_scores, axis=1).sum(axis=1)
            )

        return exposures

    def risk_attribution(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
        window: int = 252,
    ) -> Dict:
        """
        Perform risk attribution analysis.

        Parameters:
        -----------
        portfolio_returns : pd.Series
            Portfolio returns
        factor_returns : pd.DataFrame
            Factor returns for attribution
        window : int
            Analysis window

        Returns:
        --------
        Dict
            Risk attribution results
        """
        try:
            # Align data
            aligned_data = pd.concat(
                [portfolio_returns, factor_returns], axis=1, join="inner"
            )

            if len(aligned_data) < window:
                logger.warning("Insufficient data for risk attribution")
                return {}

            y = aligned_data.iloc[:, 0].tail(window)  # Portfolio returns
            X = aligned_data.iloc[:, 1:].tail(window)  # Factor returns

            # Add intercept
            X_with_intercept = np.column_stack([np.ones(len(X)), X])

            # Factor model regression
            coefficients = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]

            # Calculate factor contributions to portfolio variance
            factor_loadings = coefficients[1:]
            factor_covariance = X.cov().values

            # Portfolio variance from factor model
            factor_variance = np.dot(
                factor_loadings, np.dot(factor_covariance, factor_loadings)
            )

            # Individual factor contributions
            factor_contributions = {}
            for i, factor_name in enumerate(X.columns):
                contribution = (factor_loadings[i] ** 2) * factor_covariance[i, i]
                factor_contributions[factor_name] = contribution / factor_variance

            # Specific risk (residual variance)
            residuals = y - X_with_intercept @ coefficients
            specific_variance = residuals.var()

            total_variance = factor_variance + specific_variance

            return {
                "Factor Variance": factor_variance,
                "Specific Variance": specific_variance,
                "Total Variance": total_variance,
                "Factor Contributions": factor_contributions,
                "R-squared": 1 - (specific_variance / total_variance),
            }

        except Exception as e:
            logger.error(f"Error in risk attribution: {e}")
            return {}

    def benchmark_comparison(
        self, portfolio_returns: pd.Series, benchmark_returns: pd.Series
    ) -> Dict:
        """
        Compare portfolio performance against benchmark.

        Parameters:
        -----------
        portfolio_returns : pd.Series
            Portfolio returns
        benchmark_returns : pd.Series
            Benchmark returns

        Returns:
        --------
        Dict
            Benchmark comparison metrics
        """
        # Align data
        aligned_portfolio, aligned_benchmark = portfolio_returns.align(
            benchmark_returns, join="inner"
        )

        if len(aligned_portfolio) == 0:
            return {}

        # Active returns
        active_returns = aligned_portfolio - aligned_benchmark

        # Tracking metrics
        tracking_error = active_returns.std() * np.sqrt(252)
        information_ratio = (
            active_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
        )

        # Beta and alpha
        beta = aligned_portfolio.cov(aligned_benchmark) / aligned_benchmark.var()
        alpha = aligned_portfolio.mean() - beta * aligned_benchmark.mean()

        # Up/down capture ratios
        up_market = aligned_benchmark > 0
        down_market = aligned_benchmark < 0

        up_capture = (
            (aligned_portfolio[up_market].mean() / aligned_benchmark[up_market].mean())
            if up_market.sum() > 0
            else 0
        )
        down_capture = (
            (
                aligned_portfolio[down_market].mean()
                / aligned_benchmark[down_market].mean()
            )
            if down_market.sum() > 0
            else 0
        )

        return {
            "Alpha (Daily)": alpha,
            "Beta": beta,
            "Tracking Error": tracking_error,
            "Information Ratio": information_ratio,
            "Up Capture": up_capture,
            "Down Capture": down_capture,
            "Active Return": active_returns.mean() * 252,
        }
