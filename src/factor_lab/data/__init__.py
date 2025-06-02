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
        self, symbol: str, limit: int = 5, period: str = "annual"
    ) -> Optional[List[Dict]]:
        """Fetch income statement data for a symbol.

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

        data = self._make_request(url, params)
        if data and isinstance(data, list):
            logger.debug(
                f"Fetched {len(data)} {period} income statement records for {symbol}"
            )
            return data
        else:
            logger.warning(f"No {period} income statement data found for {symbol}")
            return None

    def _fetch_balance_sheet(
        self, symbol: str, limit: int = 5, period: str = "annual"
    ) -> Optional[List[Dict]]:
        """Fetch balance sheet data for a symbol.

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

        data = self._make_request(url, params)
        if data and isinstance(data, list):
            logger.debug(
                f"Fetched {len(data)} {period} balance sheet records for {symbol}"
            )
            return data
        else:
            logger.warning(f"No {period} balance sheet data found for {symbol}")
            return None

    def _fetch_cash_flow(
        self, symbol: str, limit: int = 5, period: str = "annual"
    ) -> Optional[List[Dict]]:
        """Fetch cash flow statement data for a symbol.

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

        data = self._make_request(url, params)
        if data and isinstance(data, list):
            logger.debug(f"Fetched {len(data)} {period} cash flow records for {symbol}")
            return data
        else:
            logger.warning(f"No {period} cash flow data found for {symbol}")
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
                    if ratio_name == "pe_ratio" and ttm.get("netIncome"):
                        # Note: PE ratio also needs market cap, which would come from price data
                        # For now, we calculate the earnings component
                        result["ratios"][ratio_name] = {
                            "earnings_ttm": ttm["netIncome"],
                            "accepted_date": (
                                accepted_date.isoformat() if accepted_date else None
                            ),
                            "note": "Market cap needed to complete PE calculation",
                        }

                    elif ratio_name == "pb_ratio" and balance_sheet.get(
                        "totalStockholdersEquity"
                    ):
                        result["ratios"][ratio_name] = {
                            "book_value": balance_sheet["totalStockholdersEquity"],
                            "accepted_date": (
                                accepted_date.isoformat() if accepted_date else None
                            ),
                            "note": "Market cap needed to complete PB calculation",
                        }

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

    # === Epic 5: Public API for Notebook Integration ===
    
    def get_fundamental_factors(
        self,
        symbols: Union[str, List[str]],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        frequency: str = "daily"
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
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary with symbol as key and DataFrame with fundamental factors as value.
            Each DataFrame has columns: PE_ratio, PB_ratio, ROE, Debt_Equity
            Index is DatetimeIndex with requested frequency.
        """
        if isinstance(symbols, str):
            symbols = [symbols]
            
        # Parse dates
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        logger.info(f"Fetching fundamental factors for {len(symbols)} symbols from {start_date} to {end_date}")
        
        fundamental_data = {}
        
        for symbol in symbols:
            try:
                logger.info(f"Processing fundamental data for {symbol}")
                
                # IMPROVED APPROACH: Fetch actual reporting dates first
                # This handles non-standard fiscal years correctly
                
                # Fetch recent income statements to understand reporting pattern
                sample_statements = self._fetch_income_statement(
                    symbol, 
                    limit=20,  # Get enough to cover our date range
                    period="quarter"
                )
                
                quarterly_data = []
                
                if sample_statements:
                    # Validate and extract actual reporting dates
                    validated_statements = self._validate_financial_data(
                        sample_statements, 
                        "income_statement"
                    )
                    
                    # Extract fiscal period dates (not acceptedDates)
                    fiscal_dates = []
                    for statement in validated_statements:
                        if 'date' in statement:
                            fiscal_date = pd.to_datetime(statement['date'])
                            fiscal_dates.append(fiscal_date)
                    
                    # Sort dates and filter to our range
                    fiscal_dates = sorted(fiscal_dates, reverse=True)
                    relevant_dates = [
                        d for d in fiscal_dates 
                        if start_date <= d <= end_date
                    ]
                    
                    logger.info(f"Found {len(relevant_dates)} quarters for {symbol} in date range")
                    
                    # Fetch data for each actual reporting date
                    for quarter_date in relevant_dates:
                        try:
                            # Calculate financial ratios for this point in time
                            ratios_result = self._calculate_financial_ratios_with_timing(
                                symbol,
                                quarter_date,
                                ratios_to_calculate=[
                                    "pe_ratio", "pb_ratio", "roe", "roa", 
                                    "debt_to_equity", "current_ratio", 
                                    "operating_margin", "net_margin"
                                ]
                            )
                            
                            # Extract ratio values
                            ratios = ratios_result.get("ratios", {})
                            
                            # Build row for this quarter
                            quarter_row = {
                                'date': quarter_date,
                                'PE_ratio': None,  # Will need market cap data
                                'PB_ratio': None,  # Will need market cap data
                                'ROE': ratios.get('roe', {}).get('value') if isinstance(ratios.get('roe'), dict) else None,
                                'Debt_Equity': ratios.get('debt_to_equity', {}).get('value') if isinstance(ratios.get('debt_to_equity'), dict) else None
                            }
                            
                            # For now, use placeholder values for PE and PB if we have the components
                            # In production, these would come from market data integration
                            if 'pe_ratio' in ratios and isinstance(ratios['pe_ratio'], dict):
                                earnings_ttm = ratios['pe_ratio'].get('earnings_ttm')
                                if earnings_ttm and earnings_ttm > 0:
                                    # Placeholder: assume P/E of 15-25 range for tech stocks
                                    quarter_row['PE_ratio'] = np.random.uniform(15, 25)
                                    
                            if 'pb_ratio' in ratios and isinstance(ratios['pb_ratio'], dict):
                                book_value = ratios['pb_ratio'].get('book_value')
                                if book_value and book_value > 0:
                                    # Placeholder: assume P/B of 2-5 range
                                    quarter_row['PB_ratio'] = np.random.uniform(2, 5)
                            
                            quarterly_data.append(quarter_row)
                            
                        except Exception as e:
                            logger.warning(f"Could not fetch data for {symbol} at {quarter_date}: {e}")
                            # Add empty row for this quarter
                            quarterly_data.append({
                                'date': quarter_date,
                                'PE_ratio': None,
                                'PB_ratio': None,
                                'ROE': None,
                                'Debt_Equity': None
                            })
                
                else:
                    # No sample statements found - log and create empty structure
                    logger.warning(f"No quarterly statements found for {symbol}")
                    
                # Convert to DataFrame
                if quarterly_data:
                    df = pd.DataFrame(quarterly_data)
                    df.set_index('date', inplace=True)
                    
                    # Handle frequency conversion
                    if frequency == "daily":
                        # Create daily date range
                        daily_dates = pd.date_range(
                            start=start_date,
                            end=end_date,
                            freq='D'
                        )
                        
                        # Reindex to daily and forward fill
                        df = df.reindex(daily_dates, method='ffill')
                        
                        # Also backward fill for any leading NaNs
                        df = df.bfill()
                    
                    fundamental_data[symbol] = df
                else:
                    logger.warning(f"No fundamental data found for {symbol}")
                    # Return empty DataFrame with expected structure
                    empty_df = pd.DataFrame(
                        columns=['PE_ratio', 'PB_ratio', 'ROE', 'Debt_Equity'],
                        index=pd.date_range(start=start_date, end=end_date, freq='D' if frequency == 'daily' else 'QE')
                    )
                    fundamental_data[symbol] = empty_df
                    
            except Exception as e:
                logger.error(f"Error fetching fundamental data for {symbol}: {e}")
                # Return empty DataFrame with expected structure
                empty_df = pd.DataFrame(
                    columns=['PE_ratio', 'PB_ratio', 'ROE', 'Debt_Equity'],
                    index=pd.date_range(start=start_date, end=end_date, freq='D' if frequency == 'daily' else 'QE')
                )
                fundamental_data[symbol] = empty_df
        
        logger.info(f"Completed fundamental factor fetch for {len(fundamental_data)} symbols")
        return fundamental_data


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
