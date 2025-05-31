"""
Data acquisition module for the Factor Investing Laboratory.

Provides unified access to financial data from multiple sources including
OpenBB Platform, Yahoo Finance, and other data providers.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging
from openbb import obb

logger = logging.getLogger(__name__)


class DataProvider:
    """Base class for data providers."""

    def __init__(self, name: str):
        self.name = name

    def get_prices(
        self, symbols: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Get price data for symbols."""
        raise NotImplementedError

    def get_fundamentals(self, symbols: List[str]) -> pd.DataFrame:
        """Get fundamental data for symbols."""
        raise NotImplementedError


class YahooFinanceProvider(DataProvider):
    """Yahoo Finance data provider using yfinance."""

    def __init__(self):
        super().__init__("Yahoo Finance")

    def get_prices(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        price_type: str = "Adj Close",
    ) -> pd.DataFrame:
        """
        Get price data from Yahoo Finance.

        Parameters:
        -----------
        symbols : List[str]
            List of ticker symbols
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
        price_type : str
            Type of price data ('Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume')

        Returns:
        --------
        pd.DataFrame
            DataFrame with dates as index and symbols as columns
        """
        try:
            data = yf.download(
                symbols,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
            )

            # Handle case where auto_adjust=True makes 'Adj Close' unavailable
            if price_type == "Adj Close" and "Adj Close" not in data.columns:
                price_type = "Close"  # Use Close when auto_adjust=True

            if len(symbols) == 1:
                # Single symbol case
                if price_type in data.columns:
                    if isinstance(data[price_type], pd.Series):
                        return data[price_type].to_frame(symbols[0])
                    else:
                        # Already a DataFrame, just rename the column
                        result = data[price_type].copy()
                        result.columns = [symbols[0]]
                        return result
                else:
                    logger.error(
                        f"Price type '{price_type}' not found in data. Available columns: {list(data.columns)}"
                    )
                    if isinstance(data["Close"], pd.Series):
                        return data["Close"].to_frame(symbols[0])
                    else:
                        result = data["Close"].copy()
                        result.columns = [symbols[0]]
                        return result
            else:
                # Multiple symbols case
                if price_type in data.columns:
                    return data[price_type]
                else:
                    logger.error(
                        f"Price type '{price_type}' not found in data. Available columns: {list(data.columns)}"
                    )
                    return data["Close"]  # Default to Close

        except Exception as e:
            logger.error(f"Error fetching data from Yahoo Finance: {e}")
            raise

    def get_info(self, symbol: str) -> Dict:
        """Get company information for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
            return {}


class OpenBBProvider(DataProvider):
    """OpenBB Platform data provider."""

    def __init__(self):
        super().__init__("OpenBB Platform")

    def get_prices(
        self, symbols: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Get price data from OpenBB Platform.

        Parameters:
        -----------
        symbols : List[str]
            List of ticker symbols
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format

        Returns:
        --------
        pd.DataFrame
            DataFrame with dates as index and symbols as columns
        """
        try:
            all_data = []

            for symbol in symbols:
                data = obb.equity.price.historical(
                    symbol=symbol, start_date=start_date, end_date=end_date
                )

                if hasattr(data, "to_df"):
                    df = data.to_df()
                    df[symbol] = df["close"]
                    all_data.append(df[[symbol]])

            if all_data:
                return pd.concat(all_data, axis=1)
            else:
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching data from OpenBB: {e}")
            raise

    def get_fundamentals(self, symbols: List[str]) -> pd.DataFrame:
        """Get fundamental data from OpenBB Platform."""
        try:
            all_data = []

            for symbol in symbols:
                try:
                    # Get key metrics
                    metrics = obb.equity.fundamental.metrics(symbol=symbol)
                    if hasattr(metrics, "to_df"):
                        df = metrics.to_df()
                        df["symbol"] = symbol
                        all_data.append(df)
                except Exception as e:
                    logger.warning(f"Could not fetch fundamentals for {symbol}: {e}")

            if all_data:
                return pd.concat(all_data, ignore_index=True)
            else:
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching fundamentals from OpenBB: {e}")
            raise


class DataManager:
    """Main data manager class that coordinates multiple data providers."""

    def __init__(self, primary_provider: str = "yahoo"):
        """
        Initialize DataManager.

        Parameters:
        -----------
        primary_provider : str
            Primary data provider ('yahoo' or 'openbb')
        """
        self.providers = {"yahoo": YahooFinanceProvider(), "openbb": OpenBBProvider()}
        self.primary_provider = primary_provider
        self.cache = {}

    def get_prices(
        self,
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        provider: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get price data for symbols.

        Parameters:
        -----------
        symbols : Union[str, List[str]]
            Single symbol or list of symbols
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
        provider : Optional[str]
            Data provider to use (defaults to primary_provider)

        Returns:
        --------
        pd.DataFrame
            Price data with dates as index and symbols as columns
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        provider = provider or self.primary_provider

        # Check cache
        cache_key = f"{provider}_{'-'.join(symbols)}_{start_date}_{end_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Fetch data
        data = self.providers[provider].get_prices(symbols, start_date, end_date)

        # Cache result
        self.cache[cache_key] = data

        return data

    def get_returns(
        self,
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        frequency: str = "daily",
        provider: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Calculate returns from price data.

        Parameters:
        -----------
        symbols : Union[str, List[str]]
            Single symbol or list of symbols
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
        frequency : str
            Return frequency ('daily', 'weekly', 'monthly')
        provider : Optional[str]
            Data provider to use

        Returns:
        --------
        pd.DataFrame
            Return data
        """
        prices = self.get_prices(symbols, start_date, end_date, provider)

        if frequency == "daily":
            returns = prices.pct_change().dropna()
        elif frequency == "weekly":
            weekly_prices = prices.resample("W").last()
            returns = weekly_prices.pct_change().dropna()
        elif frequency == "monthly":
            monthly_prices = prices.resample("M").last()
            returns = monthly_prices.pct_change().dropna()
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")

        return returns

    def get_universe(self, universe_type: str = "sp500") -> List[str]:
        """
        Get a predefined universe of stocks.

        Parameters:
        -----------
        universe_type : str
            Type of universe ('sp500', 'nasdaq100', 'dow30')

        Returns:
        --------
        List[str]
            List of ticker symbols
        """
        if universe_type == "sp500":
            # Sample S&P 500 symbols for demo
            return [
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
                "DIS",
                "PYPL",
                "BAC",
                "NFLX",
                "ADBE",
                "CRM",
                "CMCSA",
                "XOM",
                "VZ",
                "KO",
                "PEP",
                "T",
                "ABT",
                "COST",
                "TMO",
                "AVGO",
                "ACN",
                "CVX",
                "NKE",
                "MRK",
                "DHR",
                "LLY",
                "TXN",
                "NEE",
                "WMT",
                "QCOM",
                "MDT",
                "BMY",
                "UNP",
                "IBM",
                "HON",
                "AMGN",
                "PM",
                "RTX",
                "LIN",
                "LOW",
                "SBUX",
            ]
        elif universe_type == "nasdaq100":
            return [
                "AAPL",
                "MSFT",
                "GOOGL",
                "GOOG",
                "AMZN",
                "TSLA",
                "META",
                "NVDA",
                "NFLX",
                "ADBE",
                "CRM",
                "PYPL",
                "INTC",
                "CMCSA",
                "AVGO",
                "TXN",
                "QCOM",
                "COST",
                "TMUS",
                "AMD",
            ]
        elif universe_type == "dow30":
            return [
                "AAPL",
                "MSFT",
                "JPM",
                "JNJ",
                "V",
                "PG",
                "UNH",
                "HD",
                "DIS",
                "BAC",
                "VZ",
                "KO",
                "CVX",
                "NKE",
                "MRK",
                "WMT",
                "IBM",
                "HON",
                "CAT",
                "AXP",
                "GS",
                "TRV",
                "MMM",
                "BA",
                "SHW",
                "MCD",
                "CRM",
                "DOW",
                "WBA",
                "CSCO",
            ]
        else:
            raise ValueError(f"Unknown universe type: {universe_type}")

    def clean_data(
        self, data: pd.DataFrame, method: str = "forward_fill"
    ) -> pd.DataFrame:
        """
        Clean price/return data.

        Parameters:
        -----------
        data : pd.DataFrame
            Raw data to clean
        method : str
            Cleaning method ('forward_fill', 'interpolate', 'drop')

        Returns:
        --------
        pd.DataFrame
            Cleaned data
        """
        if method == "forward_fill":
            return data.fillna(method="ffill")
        elif method == "interpolate":
            return data.interpolate()
        elif method == "drop":
            return data.dropna()
        else:
            raise ValueError(f"Unknown cleaning method: {method}")
