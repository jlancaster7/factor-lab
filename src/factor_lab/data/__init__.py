"""
Data acquisition module for the Factor Investing Laboratory.

Provides unified access to financial data from multiple sources including
OpenBB Platform, Yahoo Finance, and other data providers.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
import logging
from openbb import obb
import json
import time
import requests
from urllib.request import urlopen
import certifi
import yaml
import os
from pathlib import Path

# Import cache components
from ..cache import CacheManager, CacheKey, CacheConfig

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
            # Add retry logic and better error handling
            max_retries = 3
            retry_delay = 2

            # For large symbol lists, batch the downloads
            batch_size = 10
            if len(symbols) > batch_size:
                logger.info(
                    f"Downloading {len(symbols)} symbols in batches of {batch_size}"
                )
                all_data = []

                for i in range(0, len(symbols), batch_size):
                    batch_symbols = symbols[i : i + batch_size]
                    logger.debug(
                        f"Downloading batch {i//batch_size + 1}: {batch_symbols}"
                    )

                    for attempt in range(max_retries):
                        try:
                            batch_data = yf.download(
                                batch_symbols,
                                start=start_date,
                                end=end_date,
                                progress=False,
                                auto_adjust=True,
                                threads=False,
                                timeout=30,
                            )

                            if not batch_data.empty:
                                all_data.append(batch_data)
                                break
                            else:
                                logger.warning(
                                    f"Empty data for batch {batch_symbols} on attempt {attempt + 1}"
                                )

                        except Exception as e:
                            if attempt < max_retries - 1:
                                logger.warning(
                                    f"Batch download attempt {attempt + 1} failed: {e}. Retrying..."
                                )
                                time.sleep(retry_delay)
                            else:
                                logger.error(
                                    f"Failed to download batch {batch_symbols}: {e}"
                                )
                                # Continue with other batches instead of failing completely
                                break

                    # Small delay between batches
                    time.sleep(0.5)

                # Combine all batch data
                if all_data:
                    data = pd.concat(all_data, axis=1)
                    # Remove duplicate columns if any
                    data = data.loc[:, ~data.columns.duplicated()]
                else:
                    data = pd.DataFrame()
            else:
                # Small batch, use original logic
                for attempt in range(max_retries):
                    try:
                        data = yf.download(
                            symbols,
                            start=start_date,
                            end=end_date,
                            progress=False,
                            auto_adjust=True,
                            threads=False,
                            timeout=30,
                        )

                        if not data.empty:
                            break
                        else:
                            logger.warning(
                                f"Empty data returned for {symbols} on attempt {attempt + 1}"
                            )

                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.warning(
                                f"Download attempt {attempt + 1} failed: {e}. Retrying..."
                            )
                            time.sleep(retry_delay)
                            retry_delay *= 2
                        else:
                            raise

            # Handle case where auto_adjust=True makes 'Adj Close' unavailable
            if price_type == "Adj Close" and "Adj Close" not in data.columns:
                price_type = "Close"  # Use Close when auto_adjust=True

            # Handle empty data
            if data.empty:
                logger.warning(f"No data downloaded for symbols: {symbols}")
                return pd.DataFrame(columns=symbols)

            # Handle different data structures from yfinance
            if len(symbols) == 1:
                # Single symbol case
                if isinstance(data.columns, pd.MultiIndex):
                    # Multi-level columns even for single symbol
                    if price_type in data.columns.get_level_values(0):
                        result = data[price_type].copy()
                        if isinstance(result, pd.Series):
                            return result.to_frame(symbols[0])
                        else:
                            result.columns = [symbols[0]]
                            return result
                    else:
                        # Try with 'Close' as fallback
                        result = data["Close"].copy()
                        if isinstance(result, pd.Series):
                            return result.to_frame(symbols[0])
                        else:
                            result.columns = [symbols[0]]
                            return result
                else:
                    # Single-level columns
                    if price_type in data.columns:
                        return data[price_type].to_frame(symbols[0])
                    elif "Close" in data.columns:
                        return data["Close"].to_frame(symbols[0])
                    else:
                        logger.error(f"No price data found for {symbols[0]}")
                        return pd.DataFrame(columns=symbols)
            else:
                # Multiple symbols case
                if isinstance(data.columns, pd.MultiIndex):
                    # Multi-level columns: (price_type, symbol)
                    if price_type in data.columns.get_level_values(0):
                        result = data[price_type]
                        # Ensure column names are just symbols
                        if isinstance(result, pd.DataFrame):
                            return result
                        else:
                            return result.to_frame()
                    else:
                        # Try 'Close' as fallback
                        result = data["Close"]
                        if isinstance(result, pd.DataFrame):
                            return result
                        else:
                            return result.to_frame()
                else:
                    # Single-level columns (shouldn't happen with multiple symbols)
                    if price_type in data.columns:
                        return data[price_type]
                    else:
                        return data["Close"]

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


class FMPProvider(DataProvider):
    """Financial Modeling Prep data provider for fundamental analysis."""

    def __init__(self, api_key: str = "", cache_config: Optional[CacheConfig] = None):
        super().__init__("Financial Modeling Prep")
        self.api_key = api_key or self._get_api_key()
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.stable_url = "https://financialmodelingprep.com/api/v4"

        # Rate limiting: 750 calls per minute = 12.5 calls per second
        self.min_request_interval = 1.0 / 12.5  # ~0.08 seconds between requests
        self.last_request_time = 0

        # Setup session with proper headers
        self.session = requests.Session()
        self.session.headers.update(
            {"Content-Type": "application/json", "User-Agent": "Factor-Lab/1.0"}
        )

        # Initialize cache
        self.cache_config = cache_config or CacheConfig.from_env()
        self.cache = CacheManager(self.cache_config)
        logger.info(f"Initialized cache with directory: {self.cache_config.cache_dir}")

        # Initialize cache optimization
        from ..cache import CachePreloadStrategy, CacheOptimizer

        self.preload_strategy = CachePreloadStrategy(self.cache_config.cache_dir)
        self.cache_optimizer = CacheOptimizer(self.cache)

        logger.info(
            f"Initialized FMP Provider with API key: {'*' * 10}{self.api_key[-4:]}"
        )

    def __del__(self):
        """Cleanup when provider is destroyed."""
        try:
            if hasattr(self, "cache"):
                self.cache.shutdown()
        except Exception:
            pass

    def _get_api_key(self) -> str:
        """Get API key from config file or environment variable."""
        api_key = None

        # Try to load from Factor Lab config
        try:
            config_paths = [
                Path.cwd() / "config" / "environments.yaml",
                Path.cwd().parent / "config" / "environments.yaml",
                Path(__file__).parent.parent.parent.parent
                / "config"
                / "environments.yaml",
            ]

            for config_path in config_paths:
                if config_path.exists():
                    with open(config_path, "r") as f:
                        config = yaml.safe_load(f)
                        api_key = (
                            config.get("api_keys", {})
                            .get("financial_modeling_prep", {})
                            .get("api_key")
                        )
                    if api_key:
                        logger.info(f"Loaded FMP API key from {config_path}")
                        break
        except Exception as e:
            logger.warning(f"Could not load config file: {e}")

        # Fallback to environment variable
        if not api_key:
            api_key = os.getenv("FMP_API_KEY")
            if api_key:
                logger.info("Loaded FMP API key from environment variable")

        if not api_key:
            raise ValueError(
                "FMP API key not found. Please set it in config/environments.yaml "
                "under api_keys.financial_modeling_prep.api_key or as FMP_API_KEY environment variable"
            )
        return api_key

    def _enforce_rate_limit(self):
        """Enforce rate limiting to stay within 750 calls/minute."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.3f} seconds")
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _make_request(self, url: str, params: Dict) -> Optional[Dict]:
        """Make a rate-limited request to FMP API with error handling."""
        # Ensure API key is in params
        params = params.copy()
        params["apikey"] = self.api_key

        # Enforce rate limiting
        self._enforce_rate_limit()

        try:
            logger.debug(f"Making request to: {url}")
            response = self.session.get(url, params=params, timeout=30)

            # Handle different response codes
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                logger.warning("Rate limit exceeded, waiting before retry...")
                retry_after = int(response.headers.get("Retry-After", 60))
                time.sleep(retry_after)
                # Retry once
                response = self.session.get(url, params=params, timeout=30)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(
                        f"Retry failed: {response.status_code} - {response.text}"
                    )
                    return None
            elif response.status_code == 401:
                logger.error("Invalid API key for FMP")
                raise ValueError("Invalid FMP API key")
            elif response.status_code == 404:
                logger.warning(f"Data not found for request: {url}")
                return None
            else:
                logger.error(f"FMP API error: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.Timeout:
            logger.error(f"Request timeout for URL: {url}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for URL {url}: {e}")
            return None

    def get_fundamentals(self, symbols: List[str]) -> pd.DataFrame:
        """Get fundamental data from Financial Modeling Prep."""
        try:
            all_data = []

            for symbol in symbols:
                # Use the new infrastructure to fetch profile data
                url = f"{self.base_url}/profile/{symbol}"
                params = {}

                data = self._make_request(url, params)
                if data and isinstance(data, list) and len(data) > 0:
                    df = pd.json_normalize(data)
                    df["symbol"] = symbol
                    all_data.append(df)
                else:
                    logger.warning(f"No fundamental data found for {symbol}")

            if all_data:
                return pd.concat(all_data, ignore_index=True)
            else:
                return pd.DataFrame()

        except Exception as e:
            logger.error(
                f"Error fetching fundamentals from Financial Modeling Prep: {e}"
            )
            raise

    # === Story 1.2: Raw Data Fetching Methods ===

    def _cached_fetch(
        self,
        statement_type: str,
        symbol: str,
        url: str,
        params: Dict,
        limit: Optional[int] = None,
        period: str = "quarterly",
    ) -> Optional[List[Dict]]:
        """
        Generic method to fetch data with caching support.

        Parameters:
        -----------
        statement_type : str
            Type of statement (income_statement, balance_sheet, etc.)
        symbol : str
            Stock symbol
        url : str
            API endpoint URL
        params : Dict
            Request parameters
        limit : Optional[int]
            Number of records limit
        period : str
            Period type (annual, quarterly)

        Returns:
        --------
        Optional[List[Dict]]
            Cached or fetched data
        """
        # Normalize period - handle 'quarter' as well as 'quarterly'
        normalized_period = (
            "quarterly" if period.lower() in ["quarter", "quarterly"] else "annual"
        )

        # Create cache key
        cache_key = CacheKey(
            symbol=symbol,
            statement_type=statement_type,
            period=normalized_period,
            limit=limit,
            version=self.cache_config.cache_version,
        )

        # Try to get from cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            logger.debug(f"Cache hit for {symbol} {statement_type} ({period})")
            # Track access for preload optimization
            self.preload_strategy.track_access(symbol, statement_type)
            return cached_data

        # Cache miss - fetch from API
        logger.debug(f"Cache miss for {symbol} {statement_type} ({period})")

        data = self._make_request(url, params)
        if data and isinstance(data, list):
            logger.debug(
                f"Fetched {len(data)} {period} {statement_type} records for {symbol}"
            )

            # Cache the data with acceptedDate metadata if available
            if data:
                # Extract acceptedDate from first record for metadata
                accepted_date = data[0].get("acceptedDate") if data else None
                cache_key.accepted_date = accepted_date

                # Cache the response
                self.cache.set(cache_key, data)
                logger.debug(f"Cached {symbol} {statement_type} data")

            return data
        else:
            logger.warning(f"No {period} {statement_type} data found for {symbol}")
            return None

    def _fetch_income_statement(
        self, symbol: str, limit: int = 5, period: str = "annual"
    ) -> Optional[List[Dict]]:
        """Fetch income statement data for a symbol with caching.

        Parameters:
        -----------
        symbol : str
            Stock symbol
        limit : int
            Number of records to fetch
        period : str
            Period type - options may include 'annual', 'quarter', or specific quarters like 'Q1', 'Q2', etc.
        """
        url = f"{self.base_url}/income-statement/{symbol}"
        params = {"limit": limit, "period": period}

        return self._cached_fetch(
            statement_type="income_statement",
            symbol=symbol,
            url=url,
            params=params,
            limit=limit,
            period=period,
        )

    def _fetch_balance_sheet(
        self, symbol: str, limit: int = 5, period: str = "annual"
    ) -> Optional[List[Dict]]:
        """Fetch balance sheet data for a symbol with caching.

        Parameters:
        -----------
        symbol : str
            Stock symbol
        limit : int
            Number of records to fetch
        period : str
            Period type - options may include 'annual', 'quarter', or specific quarters like 'Q1', 'Q2', etc.
        """
        url = f"{self.base_url}/balance-sheet-statement/{symbol}"
        params = {"limit": limit, "period": period}

        return self._cached_fetch(
            statement_type="balance_sheet",
            symbol=symbol,
            url=url,
            params=params,
            limit=limit,
            period=period,
        )

    def _fetch_cash_flow(
        self, symbol: str, limit: int = 5, period: str = "annual"
    ) -> Optional[List[Dict]]:
        """Fetch cash flow statement data for a symbol with caching.

        Parameters:
        -----------
        symbol : str
            Stock symbol
        limit : int
            Number of records to fetch
        period : str
            Period type - options may include 'annual', 'quarter', or specific quarters like 'Q1', 'Q2', etc.
        """
        url = f"{self.base_url}/cash-flow-statement/{symbol}"
        params = {"limit": limit, "period": period}

        return self._cached_fetch(
            statement_type="cash_flow",
            symbol=symbol,
            url=url,
            params=params,
            limit=limit,
            period=period,
        )

    def _fetch_financial_ratios(
        self, symbol: str, limit: int = 5
    ) -> Optional[List[Dict]]:
        """Fetch financial ratios data for a symbol with caching."""
        url = f"{self.base_url}/ratios/{symbol}"
        params = {"limit": limit}

        return self._cached_fetch(
            statement_type="financial_ratios",
            symbol=symbol,
            url=url,
            params=params,
            limit=limit,
            period="quarterly",  # Ratios are typically quarterly
        )

    def _fetch_historical_prices(
        self,
        symbol: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Optional[List[Dict]]:
        """
        Fetch historical end-of-day price data from FMP with optimized caching.

        This method implements full symbol caching to avoid fragmentation.
        Instead of caching each date range separately, it caches the full
        available history per symbol and returns the requested subset.

        Parameters:
        -----------
        symbol : str
            Stock symbol
        from_date : Optional[str]
            Start date in YYYY-MM-DD format
        to_date : Optional[str]
            End date in YYYY-MM-DD format
        limit : Optional[int]
            Maximum number of records to fetch (not used with date range)

        Returns:
        --------
        Optional[List[Dict]]
            List of price records with date, OHLC, volume
        """
        # Create cache key for full history (v2.0 to distinguish from old cache)
        full_history_key = CacheKey(
            symbol=symbol,
            statement_type="price",
            period="full_history",
            fiscal_date="all",
            version="2.0",  # New version for optimized cache
        )

        # Try to get full history from cache
        cached_full_history = self.cache.get(full_history_key)

        if cached_full_history is not None:
            logger.debug(f"Cache hit for {symbol} full price history")

            # Filter to requested date range if needed
            if from_date or to_date:
                filtered_data = []
                from_dt = pd.to_datetime(from_date) if from_date else pd.Timestamp.min
                to_dt = pd.to_datetime(to_date) if to_date else pd.Timestamp.max

                for record in cached_full_history:
                    record_date = pd.to_datetime(record["date"])
                    if from_dt <= record_date <= to_dt:
                        filtered_data.append(record)

                logger.debug(
                    f"Filtered {len(filtered_data)} records from cached {len(cached_full_history)} records"
                )
                return filtered_data
            else:
                return cached_full_history

        # Cache miss - fetch full history from API
        logger.info(f"Cache miss for {symbol} price data - fetching full history")

        url = f"{self.base_url}/historical-price-full/{symbol}"
        params = {}

        # Fetch full available history (no date params)
        # FMP returns max 5 years by default, which is sufficient for most use cases
        data = self._make_request(url, params)

        # FMP returns data in a nested structure
        if data and isinstance(data, dict) and "historical" in data:
            full_history = data["historical"]
            logger.info(
                f"Fetched {len(full_history)} price records for {symbol} (full history)"
            )

            # Cache the full history
            if full_history:
                self.cache.set(full_history_key, full_history)
                logger.info(
                    f"Cached {symbol} full price history ({len(full_history)} records)"
                )

            # Filter to requested date range if needed
            if from_date or to_date:
                filtered_data = []
                from_dt = pd.to_datetime(from_date) if from_date else pd.Timestamp.min
                to_dt = pd.to_datetime(to_date) if to_date else pd.Timestamp.max

                for record in full_history:
                    record_date = pd.to_datetime(record["date"])
                    if from_dt <= record_date <= to_dt:
                        filtered_data.append(record)

                logger.debug(
                    f"Returning {len(filtered_data)} records for requested range"
                )
                return filtered_data
            else:
                return full_history
        else:
            logger.warning(f"No historical price data found for {symbol}")
            return None

    def _update_price_cache_if_stale(self, symbol: str) -> bool:
        """
        Check if price cache needs updating and update if necessary.

        This method checks if the cached price data is up to date.
        If the latest cached date is more than 1 day old, it fetches
        recent data and appends to the cache.

        Parameters:
        -----------
        symbol : str
            Stock symbol

        Returns:
        --------
        bool
            True if cache was updated, False otherwise
        """
        full_history_key = CacheKey(
            symbol=symbol,
            statement_type="price",
            period="full_history",
            fiscal_date="all",
            version="2.0",
        )

        cached_data = self.cache.get(full_history_key)
        if not cached_data:
            return False

        # Check latest date in cache
        latest_cached_date = pd.to_datetime(
            cached_data[0]["date"]
        )  # FMP returns newest first
        today = pd.to_datetime(datetime.now().date())

        # If cache is current (within 1 trading day), no update needed
        if (today - latest_cached_date).days <= 1:
            return False

        logger.info(
            f"Price cache for {symbol} is stale (latest: {latest_cached_date}), updating..."
        )

        # Fetch recent data
        from_date = (latest_cached_date + timedelta(days=1)).strftime("%Y-%m-%d")
        to_date = today.strftime("%Y-%m-%d")

        url = f"{self.base_url}/historical-price-full/{symbol}"
        params = {"from": from_date, "to": to_date}

        data = self._make_request(url, params)

        if data and isinstance(data, dict) and "historical" in data:
            new_records = data["historical"]
            if new_records:
                # Merge new records with existing cache (remove duplicates)
                existing_dates = {record["date"] for record in cached_data}

                # Add new records that aren't already in cache
                records_added = 0
                for record in new_records:
                    if record["date"] not in existing_dates:
                        cached_data.insert(
                            0, record
                        )  # Insert at beginning (newest first)
                        records_added += 1

                if records_added > 0:
                    # Update cache
                    self.cache.set(full_history_key, cached_data)
                    logger.info(
                        f"Updated {symbol} price cache with {records_added} new records"
                    )
                    return True

        return False

    # === Story 3.2: Cache Management Methods ===

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics including hit rate, size, and performance metrics.

        Returns:
        --------
        Dict[str, Any]
            Cache statistics dictionary
        """
        return self.cache.get_stats()

    def clear_cache(
        self, symbol: Optional[str] = None, statement_type: Optional[str] = None
    ):
        """
        Clear cache entries.

        Parameters:
        -----------
        symbol : Optional[str]
            If provided, clear only cache for this symbol
        statement_type : Optional[str]
            If provided with symbol, clear only specific statement type
        """
        if symbol:
            self.cache.invalidate_symbol(symbol, statement_type)
            logger.info(
                f"Cleared cache for {symbol}"
                + (f" {statement_type}" if statement_type else "")
            )
        else:
            self.cache.clear_all()
            logger.info("Cleared all cache")

    def warm_cache(
        self,
        symbols: List[str],
        statement_types: Optional[List[str]] = None,
        include_prices: bool = True,
    ):
        """
        Pre-warm cache for a list of symbols.

        Parameters:
        -----------
        symbols : List[str]
            List of symbols to cache
        statement_types : Optional[List[str]]
            Specific statement types to cache (default: all)
        include_prices : bool
            Whether to cache price data (default: True)
        """
        if statement_types is None:
            statement_types = [
                "income_statement",
                "balance_sheet",
                "cash_flow",
                "financial_ratios",
            ]

        logger.info(f"Warming cache for {len(symbols)} symbols")

        for symbol in symbols:
            # Cache financial statements
            for stmt_type in statement_types:
                try:
                    if stmt_type == "income_statement":
                        self._fetch_income_statement(symbol, limit=20, period="quarter")
                    elif stmt_type == "balance_sheet":
                        self._fetch_balance_sheet(symbol, limit=20, period="quarter")
                    elif stmt_type == "cash_flow":
                        self._fetch_cash_flow(symbol, limit=20, period="quarter")
                    elif stmt_type == "financial_ratios":
                        self._fetch_financial_ratios(symbol, limit=20)
                except Exception as e:
                    logger.error(f"Error warming cache for {symbol} {stmt_type}: {e}")

            # Cache price data (full history)
            if include_prices:
                try:
                    logger.info(f"Caching full price history for {symbol}")
                    self._fetch_historical_prices(
                        symbol
                    )  # No date range = full history
                except Exception as e:
                    logger.error(f"Error caching price data for {symbol}: {e}")

        stats = self.get_cache_stats()
        logger.info(f"Cache warming complete. Hit rate: {stats.get('hit_rate', 0):.2%}")

    def clear_old_price_cache(self):
        """
        Clear old fragmented price cache files (v1.0).
        This removes the thousands of small date-range cache files
        to make room for the new optimized full-history cache.
        """
        try:
            cache_dir = self.cache_config.cache_dir / "price_data"
            if not cache_dir.exists():
                logger.info("No price cache directory found")
                return

            # Count files before cleanup
            old_files = list(cache_dir.glob("*v1.0.json*"))
            file_count = len(old_files)

            if file_count == 0:
                logger.info("No old price cache files found")
                return

            logger.info(f"Found {file_count} old price cache files to remove")

            # Remove old cache files
            removed_count = 0
            for file_path in old_files:
                try:
                    file_path.unlink()
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Could not remove {file_path}: {e}")

            logger.info(f"Removed {removed_count}/{file_count} old price cache files")

            # Also clear any corrupted cache entries from memory
            if hasattr(self.cache, "_cache"):
                # Remove any in-memory entries for old price cache
                keys_to_remove = []
                for key in self.cache._cache.keys():
                    if (
                        hasattr(key, "statement_type")
                        and key.statement_type == "price"
                        and key.version == "1.0"
                    ):
                        keys_to_remove.append(key)

                for key in keys_to_remove:
                    del self.cache._cache[key]

                if keys_to_remove:
                    logger.info(
                        f"Cleared {len(keys_to_remove)} old price cache entries from memory"
                    )

        except Exception as e:
            logger.error(f"Error clearing old price cache: {e}")

    def migrate_price_cache(self, symbols: Optional[List[str]] = None):
        """
        Migrate from old fragmented price cache to new full-history cache.

        Parameters:
        -----------
        symbols : Optional[List[str]]
            Specific symbols to migrate (default: None = migrate all)
        """
        logger.info("Starting price cache migration to optimized format")

        # First, clear old cache to free up space
        self.clear_old_price_cache()

        # If symbols not specified, get from existing cache or use defaults
        if symbols is None:
            # Try to get symbols from existing statement cache
            cache_dir = self.cache_config.cache_dir / "income_statements"
            if cache_dir.exists():
                symbols = []
                for file_path in cache_dir.glob("*_income_statement_*.json*"):
                    symbol = file_path.name.split("_")[0]
                    if symbol not in symbols:
                        symbols.append(symbol)
                logger.info(f"Found {len(symbols)} symbols in existing cache")
            else:
                logger.warning("No symbols found in cache, skipping migration")
                return

        # Warm cache with full price history for each symbol
        logger.info(f"Migrating price cache for {len(symbols)} symbols")
        success_count = 0

        for i, symbol in enumerate(symbols):
            try:
                logger.info(
                    f"[{i+1}/{len(symbols)}] Caching full price history for {symbol}"
                )
                self._fetch_historical_prices(symbol)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to cache price history for {symbol}: {e}")

        logger.info(
            f"Price cache migration complete: {success_count}/{len(symbols)} symbols cached"
        )

    # === Story 3.3: Cache Performance Optimization ===

    def batch_fetch_statements(
        self,
        symbols: List[str],
        statement_types: Optional[List[str]] = None,
        period: str = "quarter",
        limit: int = 20,
        use_cache: bool = True,
    ) -> Dict[str, Dict[str, List[Dict]]]:
        """
        Batch fetch multiple statements for multiple symbols efficiently.

        Parameters:
        -----------
        symbols : List[str]
            List of symbols to fetch
        statement_types : Optional[List[str]]
            Statement types to fetch (default: all)
        period : str
            Period type (annual/quarter)
        limit : int
            Number of records per statement
        use_cache : bool
            Whether to use cache (default: True)

        Returns:
        --------
        Dict[str, Dict[str, List[Dict]]]
            Nested dict: {symbol: {statement_type: data}}
        """
        if statement_types is None:
            statement_types = [
                "income_statement",
                "balance_sheet",
                "cash_flow",
                "financial_ratios",
            ]

        results = {}
        cache_hits = 0
        cache_misses = 0

        # First pass: collect all cache keys and check what's already cached
        cache_keys_to_fetch = []

        for symbol in symbols:
            results[symbol] = {}

            for stmt_type in statement_types:
                if use_cache:
                    # Create cache key
                    cache_key = CacheKey(
                        symbol=symbol,
                        statement_type=stmt_type,
                        period=(
                            "quarterly"
                            if period.lower() in ["quarter", "quarterly"]
                            else "annual"
                        ),
                        limit=limit if stmt_type != "financial_ratios" else None,
                        version=self.cache_config.cache_version,
                    )

                    # Try cache first
                    cached_data = self.cache.get(cache_key)
                    if cached_data is not None:
                        results[symbol][stmt_type] = cached_data
                        cache_hits += 1
                    else:
                        cache_keys_to_fetch.append((symbol, stmt_type, cache_key))
                        cache_misses += 1
                else:
                    cache_keys_to_fetch.append((symbol, stmt_type, None))

        logger.info(
            f"Batch fetch: {cache_hits} cache hits, {cache_misses} cache misses"
        )

        # Second pass: fetch missing data
        for symbol, stmt_type, cache_key in cache_keys_to_fetch:
            try:
                data = None
                if stmt_type == "income_statement":
                    data = self._fetch_income_statement(
                        symbol, limit=limit, period=period
                    )
                elif stmt_type == "balance_sheet":
                    data = self._fetch_balance_sheet(symbol, limit=limit, period=period)
                elif stmt_type == "cash_flow":
                    data = self._fetch_cash_flow(symbol, limit=limit, period=period)
                elif stmt_type == "financial_ratios":
                    data = self._fetch_financial_ratios(symbol, limit=limit)

                if data:
                    results[symbol][stmt_type] = data
            except Exception as e:
                logger.error(f"Error fetching {stmt_type} for {symbol}: {e}")
                results[symbol][stmt_type] = None

        return results

    def preload_cache_from_results(
        self,
        results: Dict[str, Dict[str, List[Dict]]],
        statement_types: Optional[List[str]] = None,
    ):
        """
        Preload cache from previously fetched results.
        Useful for warming cache from batch operations.

        Parameters:
        -----------
        results : Dict[str, Dict[str, List[Dict]]]
            Results from batch_fetch_statements or similar
        statement_types : Optional[List[str]]
            Specific statement types to cache (default: all available)
        """
        if statement_types is None:
            statement_types = [
                "income_statement",
                "balance_sheet",
                "cash_flow",
                "financial_ratios",
            ]

        cached_count = 0

        for symbol, symbol_data in results.items():
            for stmt_type, data in symbol_data.items():
                if stmt_type in statement_types and data:
                    # Create cache key
                    cache_key = CacheKey(
                        symbol=symbol,
                        statement_type=stmt_type,
                        period="quarterly",  # Default to quarterly
                        limit=20 if stmt_type != "financial_ratios" else None,
                        version=self.cache_config.cache_version,
                    )

                    # Extract acceptedDate if available
                    if isinstance(data, list) and len(data) > 0:
                        accepted_date = data[0].get("acceptedDate")
                        cache_key.accepted_date = accepted_date

                    # Cache the data
                    self.cache.set(cache_key, data)
                    cached_count += 1

        logger.info(f"Preloaded {cached_count} entries into cache")

    def get_cache_memory_usage(self) -> Dict[str, Any]:
        """
        Get detailed memory usage statistics for the cache.

        Returns:
        --------
        Dict[str, Any]
            Memory usage statistics by statement type
        """
        stats = {
            "total_size_mb": 0,
            "by_statement_type": {},
            "by_symbol": {},
            "compression_ratio": 0,
            "entry_count": 0,
        }

        # Get cache directory
        cache_dir = self.cache_config.cache_dir

        # Calculate sizes by statement type
        for stmt_type in [
            "income_statements",
            "balance_sheets",
            "cash_flows",
            "financial_ratios",
            "price_data",
        ]:
            type_dir = cache_dir / stmt_type
            if type_dir.exists():
                type_size = sum(
                    f.stat().st_size for f in type_dir.rglob("*") if f.is_file()
                )
                stats["by_statement_type"][stmt_type] = type_size / (1024 * 1024)  # MB
                stats["total_size_mb"] += type_size / (1024 * 1024)
                stats["entry_count"] += len(list(type_dir.glob("*.json*")))

        # Estimate compression ratio if enabled
        if self.cache_config.enable_compression:
            # Rough estimate: gzip typically achieves 60-80% compression on JSON
            stats["compression_ratio"] = 0.7  # 70% compression estimate

        return stats

    def smart_preload_cache(self) -> Dict[str, Any]:
        """
        Intelligently preload cache based on usage patterns.

        Returns:
        --------
        Dict[str, Any]
            Preload results and statistics
        """
        import time

        start_time = time.time()

        # Get preload recommendations
        recommendations = self.preload_strategy.get_preload_recommendations()

        # Determine symbols to preload
        symbols_to_preload = list(
            set(
                recommendations.get("high_frequency", [])[:30]
                + recommendations.get("recent", [])[:20]
                + recommendations.get("critical", [])[:10]
            )
        )

        if not symbols_to_preload:
            # Fallback to top symbols from access history
            symbols_to_preload = self.preload_strategy.get_preload_symbols(top_n=50)

        logger.info(f"Smart preload: Loading {len(symbols_to_preload)} symbols")

        # Get optimal batch size
        batch_size = self.preload_strategy.get_optimal_batch_size()

        # Batch fetch with optimized settings
        all_results = {}
        success_count = 0

        for i in range(0, len(symbols_to_preload), batch_size):
            batch = symbols_to_preload[i : i + batch_size]

            try:
                # Use batch fetch for efficiency
                results = self.batch_fetch_statements(
                    symbols=batch,
                    statement_types=[
                        "income_statement",
                        "balance_sheet",
                        "financial_ratios",
                    ],
                    period="quarter",
                    limit=20,
                    use_cache=True,
                )

                all_results.update(results)

                # Count successes
                for symbol, data in results.items():
                    if any(v for v in data.values() if v):
                        success_count += 1

            except Exception as e:
                logger.error(f"Error in smart preload batch: {e}")

        # Record preload operation
        duration = time.time() - start_time
        self.preload_strategy.record_preload(
            symbols_to_preload, duration, success_count
        )

        # Get cache health
        cache_analysis = self.cache_optimizer.analyze_cache_performance()

        return {
            "symbols_loaded": len(symbols_to_preload),
            "success_count": success_count,
            "duration": duration,
            "rate": success_count / duration if duration > 0 else 0,
            "cache_health": cache_analysis["health"],
            "recommendations": cache_analysis["recommendations"],
        }

    def optimize_cache_settings(self) -> Dict[str, Any]:
        """
        Optimize cache settings based on usage patterns.

        Returns:
        --------
        Dict[str, Any]
            Optimization results
        """
        # Get current performance
        analysis = self.cache_optimizer.analyze_cache_performance()

        # Get TTL recommendations
        ttl_recommendations = self.cache_optimizer.optimize_ttl_settings()

        # Apply recommendations if needed
        applied_changes = []

        if analysis["health"] < 80:
            # Apply TTL optimizations
            for stmt_type, recommended_ttl in ttl_recommendations.items():
                current_ttl = getattr(self.cache_config, f"{stmt_type}_ttl", None)
                if (
                    current_ttl and abs(current_ttl - recommended_ttl) > 3600
                ):  # > 1 hour difference
                    setattr(self.cache_config, f"{stmt_type}_ttl", recommended_ttl)
                    applied_changes.append(
                        f"Adjusted {stmt_type} TTL to {recommended_ttl/3600:.1f} hours"
                    )

        return {
            "cache_health": analysis["health"],
            "performance": analysis["performance"],
            "recommendations": analysis["recommendations"],
            "applied_changes": applied_changes,
            "ttl_settings": ttl_recommendations,
        }

    # === Story 1.3: Data Validation and Cleaning ===

    def _validate_financial_data(self, data: List[Dict], data_type: str) -> List[Dict]:
        """Validate and clean financial statement data."""
        if not data:
            return []

        validated_data = []
        for record in data:
            try:
                # Validate required fields
                if not all(key in record for key in ["date", "symbol"]):
                    logger.warning(
                        f"Missing required fields in {data_type} data: {record}"
                    )
                    continue

                # Validate and parse date
                if isinstance(record["date"], str):
                    try:
                        record["date"] = pd.to_datetime(record["date"]).strftime(
                            "%Y-%m-%d"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Invalid date format in {data_type}: {record['date']}"
                        )
                        continue

                # Validate acceptedDate if present
                if "acceptedDate" in record and record["acceptedDate"]:
                    try:
                        record["acceptedDate"] = pd.to_datetime(record["acceptedDate"])
                    except Exception as e:
                        logger.warning(
                            f"Invalid acceptedDate format: {record['acceptedDate']}"
                        )
                        record["acceptedDate"] = None

                # Convert numeric fields
                for key, value in record.items():
                    if key not in [
                        "date",
                        "symbol",
                        "acceptedDate",
                        "period",
                        "fiscalYear",
                        "reportedCurrency",
                        "cik",  # Keep CIK as string
                        "calendarYear",  # Keep calendar year
                        "fillingDate",  # Keep as string
                    ]:
                        # Handle empty strings and None values
                        if value is None or value == "":
                            record[key] = None
                        else:
                            try:
                                record[key] = float(value) if value != 0 else 0.0
                            except (ValueError, TypeError):
                                record[key] = None

                validated_data.append(record)

            except Exception as e:
                logger.warning(f"Error validating {data_type} record: {e}")
                continue

        logger.debug(f"Validated {len(validated_data)}/{len(data)} {data_type} records")
        return validated_data

    def _parse_date_safely(self, date_str: str) -> Optional[datetime]:
        """Safely parse date string to datetime object."""
        if not date_str or date_str == "None":
            return None

        try:
            # Handle different date formats consistently
            # Strip time component for consistency
            date_part = (
                date_str.split("T")[0].split(" ")[0]
                if isinstance(date_str, str)
                else str(date_str)
            )

            # Try standard date format first
            try:
                return datetime.strptime(date_part, "%Y-%m-%d")
            except ValueError:
                pass

            # Fallback to pandas for more complex parsing
            parsed_date = pd.to_datetime(date_str)
            return parsed_date.to_pydatetime().replace(
                hour=0, minute=0, second=0, microsecond=0
            )

        except Exception as e:
            logger.warning(f"Could not parse date: {date_str} - {e}")
            return None

    def _perform_data_quality_checks(
        self, data: List[Dict], data_type: str
    ) -> Dict[str, Any]:
        """Perform basic data quality checks and return statistics."""
        if not data:
            return {"total_records": 0, "quality_score": 0.0, "issues": {}}

        total_records = len(data)
        quality_issues = {
            "missing_acceptedDate": 0,
            "negative_revenue": 0,
            "missing_key_fields": 0,
            "invalid_dates": 0,
            "zero_values": 0,
        }

        key_fields_by_type = {
            "income_statement": ["revenue", "netIncome"],
            "balance_sheet": ["totalAssets", "totalEquity"],
            "cash_flow": ["operatingCashFlow", "netCashFlow"],
            "financial_ratios": ["currentRatio", "returnOnEquity"],
        }

        key_fields = key_fields_by_type.get(data_type, ["revenue"])

        # Track which records have critical issues
        problematic_records = set()

        for i, record in enumerate(data):
            record_has_critical_issue = False

            # Check for missing acceptedDate (critical for preventing look-ahead bias)
            if not record.get("acceptedDate"):
                quality_issues["missing_acceptedDate"] += 1
                record_has_critical_issue = True

            # Check for negative revenue (critical data quality issue)
            if record.get("revenue") and record["revenue"] < 0:
                quality_issues["negative_revenue"] += 1
                record_has_critical_issue = True

            # Check for missing key fields (critical for calculations)
            missing_key_fields = sum(
                1 for field in key_fields if record.get(field) is None
            )
            if missing_key_fields > 0:
                quality_issues["missing_key_fields"] += 1
                record_has_critical_issue = True

            # Check for invalid dates (critical for time-series analysis)
            try:
                if record.get("date"):
                    pd.to_datetime(record["date"])
            except:
                quality_issues["invalid_dates"] += 1
                record_has_critical_issue = True

            # Check for zero values in key financial metrics (minor issue)
            zero_values = sum(1 for field in key_fields if record.get(field) == 0.0)
            if zero_values > 0:
                quality_issues["zero_values"] += 1
                # Zero values are not critical issues - could be legitimate

            # Mark record as problematic if it has any critical issue
            if record_has_critical_issue:
                problematic_records.add(i)

        # Calculate quality score based on percentage of clean records
        # Only critical issues (not zero_values) count towards quality score
        clean_records = total_records - len(problematic_records)
        quality_score = (
            (clean_records / total_records) * 100 if total_records > 0 else 100
        )

        quality_stats = {
            "total_records": total_records,
            "quality_score": round(quality_score, 2),
            "issues": quality_issues,
            "data_type": data_type,
        }

        # Log quality issues if score is low
        if quality_score < 80:
            logger.warning(
                f"Data quality concerns for {data_type}: score={quality_score:.1f}%, issues={quality_issues}"
            )
        else:
            logger.debug(
                f"Data quality check for {data_type}: score={quality_score:.1f}%"
            )

        return quality_stats

    # === Story 2.1: Accepted Date Handling ===
    def _filter_by_accepted_date(
        self,
        data: List[Dict],
        as_of_date: Union[str, datetime],
        use_fiscal_date_fallback: bool = True,
        fiscal_lag_days: int = 75,
    ) -> List[Dict]:
        """
        Filter financial statements by acceptedDate to prevent look-ahead bias.

        This is critical for historical factor analysis - we can only use data that
        was publicly available (accepted/filed) as of the analysis date.

        For data without acceptedDate (e.g., calculated ratios), optionally uses
        fiscal period date + lag as a conservative estimate.

        Parameters:
        -----------
        data : List[Dict]
            List of financial statement records (validated)
        as_of_date : Union[str, datetime]
            The point-in-time date for analysis (YYYY-MM-DD or datetime)
        use_fiscal_date_fallback : bool
            If True, use fiscal date + lag for records without acceptedDate
        fiscal_lag_days : int
            Days to add to fiscal date as filing lag estimate (default: 75 days)

        Returns:
        --------
        List[Dict]
            Filtered records where acceptedDate <= as_of_date (or fiscal_date + lag <= as_of_date)
        """
        if not data:
            return []

        # Parse as_of_date to datetime for comparison
        if isinstance(as_of_date, str):
            try:
                as_of_date = pd.to_datetime(as_of_date)
            except Exception as e:
                logger.error(f"Invalid as_of_date format: {as_of_date} - {e}")
                return []
        elif isinstance(as_of_date, datetime):
            as_of_date = pd.to_datetime(as_of_date)
        else:
            logger.error(
                f"as_of_date must be string or datetime, got {type(as_of_date)}"
            )
            return []
        filtered_data = []
        filtered_count = 0
        no_accepted_date_count = 0
        fallback_used_count = 0

        for record in data:
            # Check if acceptedDate exists and is valid
            accepted_date = record.get("acceptedDate")

            if not accepted_date:
                no_accepted_date_count += 1

                # Try fallback using fiscal date + lag
                if use_fiscal_date_fallback and record.get("date"):
                    try:
                        fiscal_date = pd.to_datetime(record["date"])
                        estimated_accepted_date = fiscal_date + pd.Timedelta(
                            days=fiscal_lag_days
                        )

                        if estimated_accepted_date <= as_of_date:
                            filtered_data.append(record)
                            fallback_used_count += 1
                            logger.debug(
                                f"Used fiscal date fallback: {record.get('symbol', 'Unknown')} "
                                f"fiscal={fiscal_date.strftime('%Y-%m-%d')} + {fiscal_lag_days}d = "
                                f"estimated_accepted={estimated_accepted_date.strftime('%Y-%m-%d')}"
                            )
                        else:
                            filtered_count += 1
                            logger.debug(
                                f"Filtered by fallback: estimated_accepted={estimated_accepted_date.strftime('%Y-%m-%d')} > "
                                f"as_of_date={as_of_date.strftime('%Y-%m-%d')}"
                            )
                        continue
                    except Exception as e:
                        logger.warning(f"Could not apply fiscal date fallback: {e}")

                logger.debug(
                    f"Record missing acceptedDate and no fallback: {record.get('symbol', 'Unknown')} {record.get('date', 'Unknown')}"
                )
                continue

            # Ensure acceptedDate is pandas Timestamp for comparison
            if not isinstance(accepted_date, pd.Timestamp):
                try:
                    accepted_date = pd.to_datetime(accepted_date)
                except Exception as e:
                    logger.warning(
                        f"Could not parse acceptedDate: {accepted_date} - {e}"
                    )
                    continue

            # Apply look-ahead bias filter: acceptedDate <= as_of_date
            if accepted_date <= as_of_date:
                filtered_data.append(record)
            else:
                filtered_count += 1
                logger.debug(
                    f"Filtered out look-ahead data: {record.get('symbol', 'Unknown')} "
                    f"fiscal_period={record.get('date', 'Unknown')} "
                    f"acceptedDate={accepted_date.strftime('%Y-%m-%d')} > "
                    f"as_of_date={as_of_date.strftime('%Y-%m-%d')}"
                )

        # Log filtering results
        original_count = len(data)
        final_count = len(filtered_data)
        logger.info(
            f"Look-ahead bias filtering: {original_count} -> {final_count} records "
            f"(filtered: {filtered_count}, missing acceptedDate: {no_accepted_date_count})"
        )

        if final_count == 0 and original_count > 0:
            logger.warning(
                f"No records available as of {as_of_date.strftime('%Y-%m-%d')}. "
                f"Earliest acceptedDate might be after as_of_date."
            )

        return filtered_data

    # === Story 2.2: Trailing 12-Month Calculations + Smart acceptedDate Mapping ===

    def _get_trailing_12m_data(
        self,
        symbol: str,
        as_of_date: Union[str, datetime],
        include_balance_sheet: bool = True,
        min_quarters: int = 4,
    ) -> Dict[str, Any]:
        """
        Get trailing 12-month financial data for a symbol as of a specific date.

        This method fetches and aggregates 4 quarters of income statement data
        and the most recent balance sheet data, all filtered by acceptedDate
        to prevent look-ahead bias.

        Parameters:
        -----------
        symbol : str
            Stock symbol (e.g., 'AAPL')
        as_of_date : Union[str, datetime]
            Point-in-time analysis date
        include_balance_sheet : bool
            Whether to include balance sheet data for point-in-time metrics
        min_quarters : int
            Minimum number of quarters required (default: 4)

        Returns:
        --------
        Dict[str, Any]
            Dictionary containing:
            - 'trailing_12m': Dict with summed income statement items
            - 'balance_sheet': Dict with most recent balance sheet data (if requested)
            - 'metadata': Dict with calculation metadata
        """
        logger.info(f"Calculating trailing 12M data for {symbol} as of {as_of_date}")

        result = {
            "trailing_12m": {},
            "balance_sheet": {},
            "metadata": {
                "symbol": symbol,
                "as_of_date": str(as_of_date),
                "quarters_used": 0,
                "balance_sheet_date": None,
                "calculation_successful": False,
            },
        }

        try:
            # Parse as_of_date for validation
            if isinstance(as_of_date, str):
                as_of_dt = pd.to_datetime(as_of_date)
            elif isinstance(as_of_date, datetime):
                as_of_dt = pd.to_datetime(as_of_date)
            else:
                logger.error(f"Invalid as_of_date type: {type(as_of_date)}")
                return result

            # Prevent future analysis dates (no factor analysis in the future!)
            current_date = pd.to_datetime(datetime.now().date())
            if as_of_dt > current_date:
                logger.warning(
                    f"Future analysis date detected: {as_of_dt.strftime('%Y-%m-%d')} > "
                    f"current date {current_date.strftime('%Y-%m-%d')}. "
                    f"Factor analysis cannot be performed for future dates."
                )
                result["metadata"]["calculation_successful"] = False
                return result
            # Fetch income statement data (get more quarters to ensure sufficient historical data)
            # API returns most recent quarters regardless of as_of_date, so we need more data
            # to ensure we have enough quarters after acceptedDate filtering
            income_data = self._fetch_income_statement(
                symbol, limit=20, period="quarter"
            )
            if not income_data:
                logger.warning(f"No quarter income statement data found for {symbol}")
                return result

            logger.debug(
                f"Fetched {len(income_data)} raw quarterly records for {symbol}"
            )

            # Validate and filter by acceptedDate
            validated_income = self._validate_financial_data(
                income_data, "income_statement"
            )
            logger.debug(
                f"Validated {len(validated_income)} quarterly records for {symbol}"
            )

            filtered_income = self._filter_by_accepted_date(
                validated_income, as_of_date
            )
            logger.debug(
                f"After acceptedDate filtering: {len(filtered_income)} quarterly records for {symbol}"
            )

            # Check if we have sufficient quarters after filtering
            if len(filtered_income) < min_quarters:
                logger.warning(
                    f"Insufficient quarterly data for {symbol}: {len(filtered_income)}/{min_quarters} quarters available as of {as_of_date}"
                )
                result["metadata"]["quarters_used"] = len(filtered_income)
                result["metadata"]["calculation_successful"] = False
                return result

            # Take the 4 most recent quarters (already sorted by date descending from API)
            recent_quarters = filtered_income[:min_quarters]
            result["metadata"]["quarters_used"] = len(recent_quarters)

            # Sum income statement items over 4 quarters
            trailing_12m = self._sum_income_statement_quarters(recent_quarters)
            result["trailing_12m"] = trailing_12m

            # Get balance sheet data if requested
            if include_balance_sheet:
                # Fetch more balance sheet records to account for acceptedDate filtering
                # Use same limit as income statement to ensure sufficient data after filtering
                balance_data = self._fetch_balance_sheet(
                    symbol, limit=20, period="quarter"
                )
                if balance_data:
                    validated_balance = self._validate_financial_data(
                        balance_data, "balance_sheet"
                    )
                    filtered_balance = self._filter_by_accepted_date(
                        validated_balance, as_of_date
                    )

                    if filtered_balance:
                        # Use most recent balance sheet
                        result["balance_sheet"] = filtered_balance[0]
                        result["metadata"]["balance_sheet_date"] = filtered_balance[
                            0
                        ].get("date")
                        logger.debug(
                            f"Using balance sheet dated {filtered_balance[0].get('date')} "
                            f"(accepted: {filtered_balance[0].get('acceptedDate')}) for {symbol}"
                        )
                    else:
                        logger.warning(
                            f"No balance sheet data available for {symbol} as of {as_of_date}"
                        )
                else:
                    logger.warning(f"Failed to fetch balance sheet data for {symbol}")

            result["metadata"]["calculation_successful"] = True
            logger.info(f"Successfully calculated trailing 12M data for {symbol}")

        except Exception as e:
            logger.error(f"Error calculating trailing 12M data for {symbol}: {e}")

        return result

    def _sum_income_statement_quarters(self, quarters: List[Dict]) -> Dict[str, float]:
        """
        Sum income statement items across quarters for trailing 12-month calculation.

        Parameters:
        -----------
        quarters : List[Dict]
            List of quarterly income statement records (validated)

        Returns:
        --------
        Dict[str, float]
            Dictionary with summed financial metrics
        """
        # Define which fields to sum (income statement flow items)
        summable_fields = [
            "revenue",
            "costOfRevenue",
            "grossProfit",
            "operatingExpenses",
            "operatingIncome",
            "totalOtherIncomeExpensesNet",
            "ebitda",
            "depreciationAndAmortization",
            "incomeBeforeTax",
            "incomeTaxExpense",
            "netIncome",
            "netIncomeRatio",
            "eps",
            "epsdiluted",
        ]

        trailing_12m = {}

        for field in summable_fields:
            total = 0
            quarters_with_data = 0

            for quarter in quarters:
                value = quarter.get(field)
                if value is not None and not pd.isna(value):
                    total += value
                    quarters_with_data += 1

            # Only include fields that have data in at least 3 quarters
            if quarters_with_data >= 3:
                trailing_12m[field] = total
            else:
                trailing_12m[field] = None

        # Calculate some derived metrics
        if trailing_12m.get("revenue") and trailing_12m.get("netIncome"):
            trailing_12m["net_margin"] = (
                trailing_12m["netIncome"] / trailing_12m["revenue"]
            )

        if trailing_12m.get("revenue") and trailing_12m.get("operatingIncome"):
            trailing_12m["operating_margin"] = (
                trailing_12m["operatingIncome"] / trailing_12m["revenue"]
            )

        # Extract shares outstanding from most recent quarter
        # Use basic shares (not diluted) for market cap calculation
        if quarters and quarters[0].get("weightedAverageShsOut"):
            trailing_12m["shares_outstanding"] = quarters[0]["weightedAverageShsOut"]
            logger.debug(
                f"Shares outstanding: {trailing_12m['shares_outstanding']:,.0f}"
            )

        return trailing_12m

    def _get_smart_ratio_accepted_date(
        self, ratio_name: str, symbol: str, as_of_date: Union[str, datetime]
    ) -> Optional[datetime]:
        """
        Get the appropriate acceptedDate for a financial ratio based on its underlying statements.

        This implements smart acceptedDate mapping to provide precise timing for ratios
        that depend on multiple financial statements.

        Parameters:
        -----------
        ratio_name : str
            Name of the financial ratio (e.g., 'debt_to_equity', 'roe', 'pe_ratio')
        symbol : str
            Stock symbol
        as_of_date : Union[str, datetime]
            Analysis date for filtering

        Returns:
        --------
        Optional[datetime]
            The appropriate acceptedDate for the ratio, or None if not determinable
        """
        # Define ratio-to-statement mapping
        ratio_statement_mapping = {
            # Balance sheet ratios (use balance sheet acceptedDate)
            "debt_to_equity": ["balance_sheet"],
            "current_ratio": ["balance_sheet"],
            "pb_ratio": ["balance_sheet"],
            "book_value_per_share": ["balance_sheet"],
            "debt_ratio": ["balance_sheet"],
            "asset_turnover": ["balance_sheet"],
            # Income statement ratios (use income statement acceptedDate)
            "pe_ratio": ["income_statement"],
            "earnings_yield": ["income_statement"],
            "net_margin": ["income_statement"],
            "operating_margin": ["income_statement"],
            "gross_margin": ["income_statement"],
            # Multi-statement ratios (use latest acceptedDate)
            "roe": ["income_statement", "balance_sheet"],
            "roa": ["income_statement", "balance_sheet"],
            "roic": ["income_statement", "balance_sheet"],
            "interest_coverage": ["income_statement", "balance_sheet"],
        }

        required_statements = ratio_statement_mapping.get(ratio_name.lower())
        if not required_statements:
            logger.warning(
                f"Unknown ratio type: {ratio_name}. Using fiscal date fallback."
            )
            return None

        accepted_dates = []

        try:
            # Fetch required statement data (get more records to ensure we have data after filtering)
            for statement_type in required_statements:
                if statement_type == "income_statement":
                    data = self._fetch_income_statement(symbol, limit=4)
                elif statement_type == "balance_sheet":
                    data = self._fetch_balance_sheet(symbol, limit=4)
                else:
                    continue

                if data:
                    validated_data = self._validate_financial_data(data, statement_type)
                    filtered_data = self._filter_by_accepted_date(
                        validated_data, as_of_date
                    )

                    if filtered_data and filtered_data[0].get("acceptedDate"):
                        accepted_dates.append(
                            pd.to_datetime(filtered_data[0]["acceptedDate"])
                        )
                    else:
                        logger.debug(
                            f"No {statement_type} data with acceptedDate available for {symbol} as of {as_of_date}"
                        )
                else:
                    logger.debug(f"Failed to fetch {statement_type} data for {symbol}")
            if accepted_dates:
                # For multi-statement ratios, use the latest acceptedDate
                # This ensures all component data was available
                return max(accepted_dates)
            else:
                logger.warning(
                    f"No acceptedDate found for {ratio_name} components for {symbol}"
                )
                return None

        except Exception as e:
            logger.error(f"Error determining acceptedDate for {ratio_name}: {e}")
            return None

    def _calculate_financial_ratios_with_timing(
        self,
        symbol: str,
        as_of_date: Union[str, datetime],
        ratios_to_calculate: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate financial ratios with proper acceptedDate timing.

        This method combines trailing 12M data with balance sheet data to calculate
        key financial ratios, using smart acceptedDate mapping for precise timing.

        Parameters:
        -----------
        symbol : str
            Stock symbol
        as_of_date : Union[str, datetime]
            Analysis date
        ratios_to_calculate : Optional[List[str]]
            Specific ratios to calculate. If None, calculates all supported ratios.

        Returns:
        --------
        Dict[str, Any]
            Dictionary containing calculated ratios and metadata
        """
        if ratios_to_calculate is None:
            ratios_to_calculate = [
                "pe_ratio",
                "pb_ratio",
                "debt_to_equity",
                "roe",
                "roa",
                "current_ratio",
                "operating_margin",
                "net_margin",
            ]

        logger.info(
            f"Calculating ratios for {symbol} as of {as_of_date}: {ratios_to_calculate}"
        )

        result = {
            "ratios": {},
            "metadata": {
                "symbol": symbol,
                "as_of_date": str(as_of_date),
                "calculation_date": datetime.now().isoformat(),
                "ratios_attempted": ratios_to_calculate,
                "ratios_successful": [],
                "errors": [],
            },
        }

        try:
            # Get trailing 12M data with balance sheet
            trailing_data = self._get_trailing_12m_data(
                symbol, as_of_date, include_balance_sheet=True
            )

            if not trailing_data["metadata"]["calculation_successful"]:
                result["metadata"]["errors"].append("Failed to get trailing 12M data")
                return result

            ttm = trailing_data["trailing_12m"]
            balance_sheet = trailing_data["balance_sheet"]

            # Calculate each requested ratio
            for ratio_name in ratios_to_calculate:
                try:
                    ratio_value = None
                    accepted_date = self._get_smart_ratio_accepted_date(
                        ratio_name, symbol, as_of_date
                    )

                    # Calculate ratio based on type
                    if ratio_name == "pe_ratio":
                        net_income = ttm.get("netIncome")
                        shares_out = ttm.get("shares_outstanding")

                        if net_income and shares_out and net_income > 0:
                            market_cap = self._calculate_market_cap(
                                symbol, as_of_date, shares_out
                            )
                            if market_cap:
                                ratio_value = market_cap / net_income
                                logger.debug(
                                    f"PE calculation: Market Cap ${market_cap:,.0f} / Net Income ${net_income:,.0f} = {ratio_value:.2f}"
                                )

                    elif ratio_name == "pb_ratio":
                        book_value = balance_sheet.get("totalStockholdersEquity")
                        shares_out = ttm.get("shares_outstanding")

                        if book_value and shares_out and book_value > 0:
                            market_cap = self._calculate_market_cap(
                                symbol, as_of_date, shares_out
                            )
                            if market_cap:
                                ratio_value = market_cap / book_value
                                logger.debug(
                                    f"PB calculation: Market Cap ${market_cap:,.0f} / Book Value ${book_value:,.0f} = {ratio_value:.2f}"
                                )

                    elif ratio_name == "debt_to_equity":
                        total_debt = balance_sheet.get("totalDebt", 0)
                        equity = balance_sheet.get("totalStockholdersEquity")
                        if equity and equity != 0:
                            ratio_value = total_debt / equity

                    elif ratio_name == "roe":
                        net_income = ttm.get("netIncome")
                        equity = balance_sheet.get("totalStockholdersEquity")
                        if net_income and equity and equity > 0:
                            ratio_value = net_income / equity
                            logger.debug(
                                f"ROE calculation: Net Income ${net_income:,.0f} / Equity ${equity:,.0f} = {ratio_value:.4f}"
                            )

                    elif ratio_name == "roa":
                        net_income = ttm.get("netIncome")
                        total_assets = balance_sheet.get("totalAssets")
                        if net_income and total_assets and total_assets > 0:
                            ratio_value = net_income / total_assets

                    elif ratio_name == "current_ratio":
                        current_assets = balance_sheet.get("totalCurrentAssets")
                        current_liabilities = balance_sheet.get(
                            "totalCurrentLiabilities"
                        )
                        if (
                            current_assets
                            and current_liabilities
                            and current_liabilities != 0
                        ):
                            ratio_value = current_assets / current_liabilities

                    elif ratio_name == "operating_margin":
                        ratio_value = ttm.get("operating_margin")

                    elif ratio_name == "net_margin":
                        ratio_value = ttm.get("net_margin")

                    # Store calculated ratio
                    if ratio_value is not None:
                        result["ratios"][ratio_name] = {
                            "value": ratio_value,
                            "accepted_date": (
                                accepted_date.isoformat() if accepted_date else None
                            ),
                        }
                        result["metadata"]["ratios_successful"].append(ratio_name)
                    else:
                        result["metadata"]["errors"].append(
                            f"Could not calculate {ratio_name}: missing data"
                        )

                except Exception as e:
                    error_msg = f"Error calculating {ratio_name}: {e}"
                    result["metadata"]["errors"].append(error_msg)
                    logger.warning(error_msg)

            logger.info(
                f"Calculated {len(result['metadata']['ratios_successful'])}/{len(ratios_to_calculate)} ratios for {symbol}"
            )

        except Exception as e:
            error_msg = f"Error in ratio calculation pipeline: {e}"
            result["metadata"]["errors"].append(error_msg)
            logger.error(error_msg)

        return result

    def _calculate_market_cap(
        self, symbol: str, as_of_date: Union[str, datetime], shares_outstanding: float
    ) -> Optional[float]:
        """
        Calculate market capitalization using price and shares outstanding.

        This method fetches the stock price as of the given date and multiplies
        by shares outstanding to get market cap. Handles weekends/holidays by
        using the most recent trading day.

        Parameters:
        -----------
        symbol : str
            Stock symbol
        as_of_date : Union[str, datetime]
            Date for market cap calculation
        shares_outstanding : float
            Number of shares outstanding

        Returns:
        --------
        Optional[float]
            Market capitalization or None if price unavailable
        """
        try:
            # Convert as_of_date to string format for API
            if isinstance(as_of_date, datetime):
                as_of_date_str = as_of_date.strftime("%Y-%m-%d")
            else:
                as_of_date_str = str(as_of_date)

            # Fetch price for a small window around the date (handle weekends)
            # Get 5 days of data to ensure we have at least one trading day
            from_date = pd.to_datetime(as_of_date_str) - timedelta(days=5)
            from_date_str = from_date.strftime("%Y-%m-%d")

            logger.debug(f"Fetching price for {symbol} around {as_of_date_str}")

            # Fetch historical prices
            price_data = self._fetch_historical_prices(
                symbol, from_date=from_date_str, to_date=as_of_date_str
            )

            if not price_data:
                logger.warning(
                    f"No price data available for {symbol} as of {as_of_date_str}"
                )
                return None

            # Convert to DataFrame for easier manipulation
            price_df = pd.DataFrame(price_data)
            price_df["date"] = pd.to_datetime(price_df["date"])
            price_df.set_index("date", inplace=True)
            price_df.sort_index(inplace=True)

            # Get the most recent price up to and including as_of_date
            as_of_dt = pd.to_datetime(as_of_date_str)
            valid_prices = price_df[price_df.index <= as_of_dt]

            if valid_prices.empty:
                logger.warning(
                    f"No price data available for {symbol} on or before {as_of_date_str}"
                )
                return None

            # Use the most recent available price (handles weekends/holidays)
            latest_price_row = valid_prices.iloc[-1]
            price_date = valid_prices.index[-1]
            price = latest_price_row["close"]  # Use closing price

            # Calculate market cap
            market_cap = price * shares_outstanding

            logger.debug(
                f"Market cap for {symbol}: ${price:.2f} × {shares_outstanding:,.0f} = ${market_cap:,.0f} "
                f"(price date: {price_date.strftime('%Y-%m-%d')})"
            )

            return market_cap

        except Exception as e:
            logger.error(f"Error calculating market cap for {symbol}: {e}")
            return None

    # === Epic 5: Public API for Notebook Integration ===

    def get_fundamental_factors(
        self,
        symbols: Union[str, List[str]],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        frequency: str = "daily",
        calculation_frequency: str = "W",
    ) -> Dict[str, pd.DataFrame]:
        """
        Get fundamental factor data for multiple symbols in notebook-compatible format.

        This method fetches fundamental data and calculates key financial ratios,
        returning them in a format suitable for factor analysis notebooks.

        Parameters:
        -----------
        symbols : Union[str, List[str]]
            Single symbol or list of symbols
        start_date : Union[str, datetime]
            Start date for data fetch
        end_date : Union[str, datetime]
            End date for data fetch
        frequency : str
            Data frequency ('daily' or 'quarterly'). Daily will forward-fill quarterly data.
        calculation_frequency : str
            How often to calculate ratios: 'D' (daily), 'W' (weekly), 'M' (monthly)
            Default is 'W' for weekly calculations to balance accuracy and efficiency

        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary with symbol as key and DataFrame with fundamental factors as value.
            Each DataFrame has columns:
            - PE_ratio, PB_ratio, ROE, Debt_Equity: Financial ratios
            Index is DatetimeIndex with trading days only.
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        # Parse dates
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        logger.info(
            f"Fetching fundamental factors for {len(symbols)} symbols from {start_date} to {end_date}"
        )

        fundamental_data = {}

        for symbol in symbols:
            try:
                logger.info(f"Processing fundamental data for {symbol}")

                # Step 1: Get trading days from price data
                logger.debug(
                    f"Fetching price data to determine trading days for {symbol}"
                )
                price_data = self.get_prices(
                    symbols=[symbol],
                    start_date=str(start_date),
                    end_date=str(end_date),
                    price_type="close",
                )

                if price_data.empty:
                    logger.warning(f"No price data available for {symbol}")
                    fundamental_data[symbol] = pd.DataFrame()
                    continue

                trading_days = price_data.index
                logger.debug(f"Found {len(trading_days)} trading days for {symbol}")

                # Step 2: Get quarterly reporting dates and build fundamental timeline
                # PERFORMANCE FIX: Calculate how many quarters we actually need based on date range
                from datetime import timedelta

                years_needed = (
                    (end_date - start_date).days / 365
                ) + 1  # Add 1 year for TTM lookback
                quarters_needed = max(
                    8, int(years_needed * 4)
                )  # Minimum 8 quarters, or calculated amount
                quarters_needed = min(
                    quarters_needed, 20
                )  # Cap at 20 to avoid over-fetching

                logger.debug(
                    f"Fetching {quarters_needed} quarters for {symbol} (date range: {years_needed:.1f} years)"
                )

                sample_statements = self._fetch_income_statement(
                    symbol, limit=quarters_needed, period="quarter"
                )

                if not sample_statements:
                    logger.warning(f"No quarterly statements found for {symbol}")
                    fundamental_data[symbol] = pd.DataFrame()
                    continue

                # Build fundamental data timeline
                fundamental_timeline = []
                validated_statements = self._validate_financial_data(
                    sample_statements, "income_statement"
                )

                # Get all relevant quarters
                for statement in validated_statements:
                    if "date" in statement and "acceptedDate" in statement:
                        quarter_end = pd.to_datetime(statement["date"])
                        report_date = pd.to_datetime(statement["acceptedDate"])

                        # Only include quarters that could affect our date range
                        if report_date <= end_date + timedelta(days=90):
                            fundamental_timeline.append(
                                {"quarter_end": quarter_end, "report_date": report_date}
                            )

                # Sort by report date
                fundamental_timeline.sort(key=lambda x: x["report_date"])
                logger.info(
                    f"Found {len(fundamental_timeline)} quarters in timeline for {symbol}"
                )

                # Step 3: Determine calculation dates
                calculation_dates = set()

                # Add regular intervals based on calculation_frequency
                if calculation_frequency == "D":
                    # Daily calculations (expensive!)
                    calculation_dates.update(trading_days)
                elif calculation_frequency == "W":
                    # Weekly - every Friday that's a trading day
                    weekly_dates = trading_days[trading_days.weekday == 4]
                    calculation_dates.update(weekly_dates)
                elif calculation_frequency == "M":
                    # Monthly - last trading day of each month
                    monthly_dates = price_data.resample("ME").last().index
                    calculation_dates.update(monthly_dates)

                # Include key dates for more accurate transitions
                for item in fundamental_timeline:
                    # Add quarter end dates if they're trading days
                    if item["quarter_end"] in trading_days:
                        calculation_dates.add(item["quarter_end"])

                    # Add report dates if they're trading days
                    if item["report_date"] in trading_days:
                        calculation_dates.add(item["report_date"])

                calculation_dates = sorted(calculation_dates)
                logger.debug(
                    f"Will calculate ratios for {len(calculation_dates)} dates"
                )

                # Step 4: Get TTM data for each quarter
                quarter_fundamentals = {}
                for timeline_item in fundamental_timeline:
                    quarter_date = timeline_item["quarter_end"]
                    if start_date - timedelta(days=365) <= quarter_date <= end_date:
                        try:
                            ttm_data = self._get_trailing_12m_data(symbol, quarter_date)
                            if ttm_data["metadata"]["calculation_successful"]:
                                quarter_fundamentals[quarter_date] = {
                                    "ttm_data": ttm_data,
                                    "report_date": timeline_item["report_date"],
                                }
                        except Exception as e:
                            logger.warning(
                                f"Could not get TTM data for {symbol} Q{quarter_date}: {e}"
                            )

                # Step 5: Calculate ratios for each calculation date
                daily_data = []

                for calc_date in calculation_dates:
                    # Find applicable quarter (most recent reported as of calc_date)
                    applicable_quarter = None
                    applicable_quarter_date = None

                    for quarter_date, quarter_info in quarter_fundamentals.items():
                        if quarter_info["report_date"].date() <= calc_date.date():
                            if (
                                applicable_quarter is None
                                or quarter_info["report_date"]
                                > applicable_quarter["report_date"]
                            ):
                                applicable_quarter = quarter_info
                                applicable_quarter_date = quarter_date

                    if applicable_quarter:
                        try:
                            # Calculate ratios using this quarter's fundamentals
                            ttm = applicable_quarter["ttm_data"]["trailing_12m"]
                            bs = applicable_quarter["ttm_data"]["balance_sheet"]

                            # Get shares outstanding
                            shares = ttm.get("shares_outstanding")

                            # Calculate market cap for current date
                            market_cap = None
                            if shares and not pd.isna(shares):
                                market_cap = self._calculate_market_cap(
                                    symbol, calc_date, shares
                                )

                            # Calculate ratios
                            pe_ratio = None
                            if (
                                market_cap
                                and ttm.get("netIncome")
                                and ttm["netIncome"] > 0
                            ):
                                pe_ratio = market_cap / ttm["netIncome"]

                            pb_ratio = None
                            if (
                                market_cap
                                and bs.get("totalStockholdersEquity")
                                and bs["totalStockholdersEquity"] > 0
                            ):
                                pb_ratio = market_cap / bs["totalStockholdersEquity"]

                            roe = None
                            if (
                                ttm.get("netIncome")
                                and bs.get("totalStockholdersEquity")
                                and bs["totalStockholdersEquity"] > 0
                            ):
                                roe = ttm["netIncome"] / bs["totalStockholdersEquity"]

                            debt_equity = None
                            if (
                                bs.get("totalStockholdersEquity")
                                and bs["totalStockholdersEquity"] != 0
                            ):
                                debt_equity = (
                                    bs.get("totalDebt", 0)
                                    / bs["totalStockholdersEquity"]
                                )

                            # Build row
                            row = {
                                "date": calc_date,
                                "PE_ratio": pe_ratio,
                                "PB_ratio": pb_ratio,
                                "ROE": roe,
                                "Debt_Equity": debt_equity,
                            }

                            daily_data.append(row)

                        except Exception as e:
                            logger.debug(
                                f"Could not calculate ratios for {symbol} on {calc_date}: {e}"
                            )

                # Step 6: Convert to DataFrame and reindex to all trading days
                if daily_data:
                    df = pd.DataFrame(daily_data)
                    df.set_index("date", inplace=True)

                    # Reindex to ALL trading days and forward-fill
                    df = df.reindex(trading_days, method="ffill")

                    # Also backward fill for any leading NaNs
                    df = df.bfill()

                    fundamental_data[symbol] = df
                else:
                    logger.warning(f"No fundamental data calculated for {symbol}")
                    fundamental_data[symbol] = pd.DataFrame()

            except Exception as e:
                logger.error(f"Error fetching fundamental data for {symbol}: {e}")
                fundamental_data[symbol] = pd.DataFrame()

        logger.info(
            f"Completed fundamental factor fetch for {len(fundamental_data)} symbols"
        )
        return fundamental_data

    def get_prices(
        self,
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        price_type: str = "close",
    ) -> pd.DataFrame:
        """
        Get historical price data in DataManager-compatible format.

        Parameters:
        -----------
        symbols : Union[str, List[str]]
            Single symbol or list of symbols
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
        price_type : str
            Type of price ('open', 'high', 'low', 'close', 'adjClose', 'vwap')

        Returns:
        --------
        pd.DataFrame
            Price data with dates as index and symbols as columns
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        logger.info(
            f"Fetching price data for {len(symbols)} symbols from {start_date} to {end_date}"
        )

        price_data = {}

        for symbol in symbols:
            try:
                # Fetch historical prices
                historical_data = self._fetch_historical_prices(
                    symbol, from_date=start_date, to_date=end_date
                )

                if historical_data:
                    # Convert to DataFrame
                    df = pd.DataFrame(historical_data)

                    # Parse dates and set as index
                    df["date"] = pd.to_datetime(df["date"])
                    df.set_index("date", inplace=True)

                    # Sort by date (FMP returns newest first)
                    df.sort_index(inplace=True)

                    # Extract requested price type
                    if price_type in df.columns:
                        price_data[symbol] = df[price_type]
                    else:
                        logger.warning(
                            f"Price type '{price_type}' not found for {symbol}, using 'close'"
                        )
                        price_data[symbol] = df["close"]
                else:
                    logger.warning(f"No price data found for {symbol}")

            except Exception as e:
                logger.error(f"Error fetching price data for {symbol}: {e}")

        # Combine all price series into a single DataFrame
        if price_data:
            prices_df = pd.DataFrame(price_data)

            # Fill missing dates with NaN (align all series)
            # Use business days for financial data (exclude weekends)
            all_dates = pd.date_range(start=start_date, end=end_date, freq="B")
            prices_df = prices_df.reindex(all_dates)

            # Forward fill missing values (weekends/holidays)
            prices_df = prices_df.ffill()

            logger.info(f"Successfully fetched price data: {prices_df.shape}")
            return prices_df
        else:
            logger.warning("No price data fetched")
            return pd.DataFrame()


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
        self.providers = {
            "yahoo": YahooFinanceProvider(),
            "openbb": OpenBBProvider(),
            "fmp": FMPProvider(),
        }
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
