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


class FMPProvider(DataProvider):
    """Financial Modeling Prep data provider for fundamental analysis."""

    def __init__(self, api_key: str = ""):
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

        logger.info(
            f"Initialized FMP Provider with API key: {'*' * 10}{self.api_key[-4:]}"
        )

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

    def _fetch_income_statement(
        self, symbol: str, limit: int = 5
    ) -> Optional[List[Dict]]:
        """Fetch income statement data for a symbol."""
        url = f"{self.base_url}/income-statement/{symbol}"
        params = {"limit": limit}

        data = self._make_request(url, params)
        if data and isinstance(data, list):
            logger.debug(f"Fetched {len(data)} income statement records for {symbol}")
            return data
        else:
            logger.warning(f"No income statement data found for {symbol}")
            return None

    def _fetch_balance_sheet(self, symbol: str, limit: int = 5) -> Optional[List[Dict]]:
        """Fetch balance sheet data for a symbol."""
        url = f"{self.base_url}/balance-sheet-statement/{symbol}"
        params = {"limit": limit}

        data = self._make_request(url, params)
        if data and isinstance(data, list):
            logger.debug(f"Fetched {len(data)} balance sheet records for {symbol}")
            return data
        else:
            logger.warning(f"No balance sheet data found for {symbol}")
            return None

    def _fetch_cash_flow(self, symbol: str, limit: int = 5) -> Optional[List[Dict]]:
        """Fetch cash flow statement data for a symbol."""
        url = f"{self.base_url}/cash-flow-statement/{symbol}"
        params = {"limit": limit}

        data = self._make_request(url, params)
        if data and isinstance(data, list):
            logger.debug(f"Fetched {len(data)} cash flow records for {symbol}")
            return data
        else:
            logger.warning(f"No cash flow data found for {symbol}")
            return None

    def _fetch_financial_ratios(
        self, symbol: str, limit: int = 5
    ) -> Optional[List[Dict]]:
        """Fetch financial ratios data for a symbol."""
        url = f"{self.base_url}/ratios/{symbol}"
        params = {"limit": limit}

        data = self._make_request(url, params)
        if data and isinstance(data, list):
            logger.debug(f"Fetched {len(data)} financial ratio records for {symbol}")
            return data
        else:
            logger.warning(f"No financial ratios data found for {symbol}")
            return None

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
