"""
Factor calculation module for the Factor Investing Laboratory.

Implements various equity factor calculations including:
- Technical factors (momentum, mean reversion)
- Fundamental factors (value, quality, profitability)
- Risk factors (volatility, beta)
- Market microstructure factors
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Callable
import logging
from scipy import stats
from scipy.stats import pearsonr
import warnings

logger = logging.getLogger(__name__)


class FactorCalculator:
    """Main class for calculating equity factors."""

    def __init__(self, data_manager=None):
        """
        Initialize FactorCalculator.

        Parameters:
        -----------
        data_manager : DataManager
            Data manager instance for fetching market data
        """
        self.data_manager = data_manager
        self.factors = {}

    def momentum(self, prices: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
        """
        Calculate momentum factor (price momentum over lookback period).

        Parameters:
        -----------
        prices : pd.DataFrame
            Price data with dates as index and symbols as columns
        lookback : int
            Lookback period in days (default: 252 for 1 year)

        Returns:
        --------
        pd.DataFrame
            Momentum factor scores
        """
        try:
            # Calculate total return over lookback period
            momentum = (prices / prices.shift(lookback) - 1) * 100
            momentum.name = f"momentum_{lookback}d"
            return momentum.dropna()
        except Exception as e:
            logger.error(f"Error calculating momentum: {e}")
            raise

    def short_term_reversal(
        self, prices: pd.DataFrame, lookback: int = 21
    ) -> pd.DataFrame:
        """
        Calculate short-term reversal factor.

        Parameters:
        -----------
        prices : pd.DataFrame
            Price data
        lookback : int
            Lookback period in days (default: 21 for 1 month)

        Returns:
        --------
        pd.DataFrame
            Short-term reversal factor scores (negative of short-term returns)
        """
        try:
            short_term_returns = prices / prices.shift(lookback) - 1
            reversal = -short_term_returns * 100  # Negative for reversal effect
            reversal.name = f"st_reversal_{lookback}d"
            return reversal.dropna()
        except Exception as e:
            logger.error(f"Error calculating short-term reversal: {e}")
            raise

    def volatility(self, returns: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
        """
        Calculate historical volatility factor.

        Parameters:
        -----------
        returns : pd.DataFrame
            Return data
        lookback : int
            Lookback period in days

        Returns:
        --------
        pd.DataFrame
            Volatility factor scores (annualized)
        """
        try:
            vol = returns.rolling(window=lookback).std() * np.sqrt(252)
            vol.name = f"volatility_{lookback}d"
            return vol.dropna()
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            raise

    def beta(
        self, returns: pd.DataFrame, market_returns: pd.Series, lookback: int = 252
    ) -> pd.DataFrame:
        """
        Calculate market beta factor.

        Parameters:
        -----------
        returns : pd.DataFrame
            Stock return data
        market_returns : pd.Series
            Market return data (benchmark)
        lookback : int
            Lookback period in days

        Returns:
        --------
        pd.DataFrame
            Beta factor scores
        """
        try:

            def rolling_beta(y, x, window):
                """Calculate rolling beta using linear regression."""
                return y.rolling(window).corr(x) * (
                    y.rolling(window).std() / x.rolling(window).std()
                )

            # Align data
            aligned_data = pd.concat([returns, market_returns], axis=1, join="inner")
            market_col = aligned_data.columns[-1]

            betas = pd.DataFrame(index=returns.index, columns=returns.columns)

            for col in returns.columns:
                if col in aligned_data.columns:
                    betas[col] = rolling_beta(
                        aligned_data[col], aligned_data[market_col], lookback
                    )

            betas.name = f"beta_{lookback}d"
            return betas.dropna()

        except Exception as e:
            logger.error(f"Error calculating beta: {e}")
            raise

    def rsi(self, prices: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Relative Strength Index (RSI).

        Parameters:
        -----------
        prices : pd.DataFrame
            Price data
        period : int
            RSI period (default: 14)

        Returns:
        --------
        pd.DataFrame
            RSI values (0-100)
        """
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi.name = f"rsi_{period}d"
            return rsi.dropna()

        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            raise

    def bollinger_position(
        self, prices: pd.DataFrame, window: int = 20, num_std: float = 2.0
    ) -> pd.DataFrame:
        """
        Calculate position within Bollinger Bands.

        Parameters:
        -----------
        prices : pd.DataFrame
            Price data
        window : int
            Moving average window
        num_std : float
            Number of standard deviations for bands

        Returns:
        --------
        pd.DataFrame
            Position within bands (0 = lower band, 1 = upper band)
        """
        try:
            rolling_mean = prices.rolling(window=window).mean()
            rolling_std = prices.rolling(window=window).std()

            upper_band = rolling_mean + (rolling_std * num_std)
            lower_band = rolling_mean - (rolling_std * num_std)

            # Calculate position within bands
            bb_position = (prices - lower_band) / (upper_band - lower_band)
            bb_position.name = f"bb_position_{window}d"
            return bb_position.dropna()

        except Exception as e:
            logger.error(f"Error calculating Bollinger Band position: {e}")
            raise

    def price_to_sma(self, prices: pd.DataFrame, window: int = 50) -> pd.DataFrame:
        """
        Calculate price relative to simple moving average.

        Parameters:
        -----------
        prices : pd.DataFrame
            Price data
        window : int
            Moving average window

        Returns:
        --------
        pd.DataFrame
            Price relative to SMA (1.0 = at SMA, >1.0 = above SMA)
        """
        try:
            sma = prices.rolling(window=window).mean()
            price_to_sma = prices / sma
            price_to_sma.name = f"price_to_sma_{window}d"
            return price_to_sma.dropna()

        except Exception as e:
            logger.error(f"Error calculating price to SMA: {e}")
            raise

    def calculate_all_technical_factors(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame = None,
        market_returns: pd.Series = None,
    ) -> pd.DataFrame:
        """
        Calculate all available technical factors.

        Parameters:
        -----------
        prices : pd.DataFrame
            Price data
        returns : pd.DataFrame, optional
            Return data (calculated from prices if not provided)
        market_returns : pd.Series, optional
            Market return data for beta calculation

        Returns:
        --------
        pd.DataFrame
            Combined factor scores
        """
        if returns is None:
            returns = prices.pct_change().dropna()

        factors = {}

        try:
            # Momentum factors
            factors["momentum_252d"] = self.momentum(prices, 252)
            factors["momentum_63d"] = self.momentum(prices, 63)
            factors["momentum_21d"] = self.momentum(prices, 21)

            # Reversal factors
            factors["st_reversal_21d"] = self.short_term_reversal(prices, 21)
            factors["st_reversal_5d"] = self.short_term_reversal(prices, 5)

            # Volatility
            factors["volatility_252d"] = self.volatility(returns, 252)
            factors["volatility_63d"] = self.volatility(returns, 63)

            # Technical indicators
            factors["rsi_14d"] = self.rsi(prices, 14)
            factors["bb_position_20d"] = self.bollinger_position(prices, 20)
            factors["price_to_sma_50d"] = self.price_to_sma(prices, 50)
            factors["price_to_sma_200d"] = self.price_to_sma(prices, 200)

            # Beta (if market returns provided)
            if market_returns is not None:
                factors["beta_252d"] = self.beta(returns, market_returns, 252)

        except Exception as e:
            logger.warning(f"Error calculating some factors: {e}")

        # Combine all factors
        if factors:
            combined = pd.concat(factors.values(), axis=1, keys=factors.keys())
            combined.columns.names = ["factor", "symbol"]
            return combined
        else:
            return pd.DataFrame()


class FactorLibrary:
    """Library of predefined factor calculation functions."""

    @staticmethod
    def zscore_normalize(
        factor_data: pd.DataFrame, window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Z-score normalize factor data.

        Parameters:
        -----------
        factor_data : pd.DataFrame
            Raw factor scores
        window : Optional[int]
            Rolling window for normalization (if None, use full sample)

        Returns:
        --------
        pd.DataFrame
            Z-score normalized factors
        """
        if window is None:
            return (factor_data - factor_data.mean()) / factor_data.std()
        else:
            rolling_mean = factor_data.rolling(window=window).mean()
            rolling_std = factor_data.rolling(window=window).std()
            return (factor_data - rolling_mean) / rolling_std

    @staticmethod
    def rank_normalize(
        factor_data: pd.DataFrame, method: str = "average"
    ) -> pd.DataFrame:
        """
        Rank normalize factor data (cross-sectional ranks).

        Parameters:
        -----------
        factor_data : pd.DataFrame
            Raw factor scores
        method : str
            Ranking method ('average', 'min', 'max', 'first', 'dense')

        Returns:
        --------
        pd.DataFrame
            Rank normalized factors (0-1 scale)
        """
        return factor_data.rank(axis=1, method=method, pct=True)

    @staticmethod
    def winsorize(
        factor_data: pd.DataFrame, limits: tuple = (0.05, 0.05)
    ) -> pd.DataFrame:
        """
        Winsorize factor data to remove outliers.

        Parameters:
        -----------
        factor_data : pd.DataFrame
            Raw factor scores
        limits : tuple
            Lower and upper quantiles for winsorization

        Returns:
        --------
        pd.DataFrame
            Winsorized factor data
        """
        from scipy.stats import mstats

        result = factor_data.copy()
        for col in result.columns:
            if result[col].notna().sum() > 0:  # Only winsorize if there's data
                result[col] = pd.Series(
                    mstats.winsorize(result[col].dropna(), limits=limits),
                    index=result[col].dropna().index,
                )
        return result

    @staticmethod
    def combine_factors(
        factors: Dict[str, pd.DataFrame],
        weights: Optional[Dict[str, float]] = None,
        method: str = "equal_weight",
    ) -> pd.DataFrame:
        """
        Combine multiple factors into a composite factor.

        Parameters:
        -----------
        factors : Dict[str, pd.DataFrame]
            Dictionary of factor DataFrames
        weights : Optional[Dict[str, float]]
            Factor weights (if None, equal weights used)
        method : str
            Combination method ('equal_weight', 'weighted', 'pca')

        Returns:
        --------
        pd.DataFrame
            Combined factor scores
        """
        if not factors:
            return pd.DataFrame()

        # Align all factors
        combined_data = pd.concat(factors.values(), axis=1, keys=factors.keys())

        if method == "equal_weight":
            return combined_data.mean(axis=1, level=1)
        elif method == "weighted":
            if weights is None:
                weights = {name: 1.0 / len(factors) for name in factors.keys()}

            weighted_sum = pd.DataFrame(
                0,
                index=combined_data.index,
                columns=combined_data.columns.get_level_values(1).unique(),
            )

            for factor_name, weight in weights.items():
                if factor_name in factors:
                    weighted_sum += factors[factor_name] * weight

            return weighted_sum
        else:
            raise ValueError(f"Unknown combination method: {method}")

    @staticmethod
    def factor_correlation_matrix(factors: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix between factors."""
        return factors.corr()

    @staticmethod
    def factor_autocorrelation(factor_data: pd.Series, max_lags: int = 20) -> pd.Series:
        """Calculate autocorrelation of a factor time series."""
        from statsmodels.tsa.stattools import acf

        autocorr = acf(factor_data.dropna(), nlags=max_lags, fft=True)
        return pd.Series(autocorr, index=range(max_lags + 1))
