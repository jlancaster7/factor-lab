"""
Utility functions for the Factor Investing Laboratory.

Provides helper functions and utilities for:
- Data processing and validation
- Mathematical calculations
- Configuration management
- Logging setup
- Common operations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import logging
import warnings
from datetime import datetime, timedelta
import os
import json
import yaml
from pathlib import Path


def setup_logging(
    level: str = "INFO", log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration.

    Parameters:
    -----------
    level : str
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    log_file : Optional[str]
        Log file path (if None, logs to console only)

    Returns:
    --------
    logging.Logger
        Configured logger
    """
    # Create logger
    logger = logging.getLogger("factor_lab")
    logger.setLevel(getattr(logging, level.upper()))

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class ConfigManager:
    """Configuration management class."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ConfigManager.

        Parameters:
        -----------
        config_path : Optional[str]
            Path to configuration file
        """
        self.config_path = config_path
        self.config = {}

        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            self.config = self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "data": {
                "primary_provider": "yahoo",
                "cache_enabled": True,
                "cache_duration_days": 1,
            },
            "factors": {
                "normalization_method": "zscore",
                "winsorize_limits": [0.05, 0.05],
                "min_history_days": 252,
            },
            "portfolio": {
                "default_optimization": "mean_variance",
                "transaction_cost": 0.001,
                "rebalance_frequency": "monthly",
                "max_weight": 0.1,
                "min_weight": 0.0,
            },
            "backtesting": {
                "initial_capital": 100000,
                "benchmark": "SPY",
                "lookback_window": 252,
            },
            "visualization": {
                "default_theme": "plotly_white",
                "figure_size": [12, 8],
                "color_palette": "Set1",
            },
        }

    def load_config(self, config_path: str) -> None:
        """Load configuration from file."""
        try:
            with open(config_path, "r") as f:
                if config_path.endswith(".json"):
                    self.config = json.load(f)
                elif config_path.endswith(".yaml") or config_path.endswith(".yml"):
                    self.config = yaml.safe_load(f)
                else:
                    raise ValueError(
                        "Unsupported config file format. Use .json or .yaml"
                    )
        except Exception as e:
            logging.warning(f"Could not load config from {config_path}: {e}")
            self.config = self._get_default_config()

    def save_config(self, config_path: str) -> None:
        """Save configuration to file."""
        try:
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)

            with open(config_path, "w") as f:
                if config_path.endswith(".json"):
                    json.dump(self.config, f, indent=2)
                elif config_path.endswith(".yaml") or config_path.endswith(".yml"):
                    yaml.dump(self.config, f, default_flow_style=False)
                else:
                    raise ValueError(
                        "Unsupported config file format. Use .json or .yaml"
                    )
        except Exception as e:
            logging.error(f"Could not save config to {config_path}: {e}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Parameters:
        -----------
        key_path : str
            Dot-separated key path (e.g., 'data.primary_provider')
        default : Any
            Default value if key not found

        Returns:
        --------
        Any
            Configuration value
        """
        keys = key_path.split(".")
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation.

        Parameters:
        -----------
        key_path : str
            Dot-separated key path
        value : Any
            Value to set
        """
        keys = key_path.split(".")
        config_ref = self.config

        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]

        config_ref[keys[-1]] = value


class DataValidator:
    """Data validation utilities."""

    @staticmethod
    def validate_price_data(data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate price data.

        Parameters:
        -----------
        data : pd.DataFrame
            Price data to validate

        Returns:
        --------
        Tuple[bool, List[str]]
            (is_valid, list_of_issues)
        """
        issues = []

        # Check if data is empty
        if data.empty:
            issues.append("Data is empty")
            return False, issues

        # Check for negative prices
        if (data < 0).any().any():
            issues.append("Negative prices detected")

        # Check for missing data
        missing_pct = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        if missing_pct > 0.1:
            issues.append(f"High missing data percentage: {missing_pct:.2%}")

        # Check for price gaps (>50% moves)
        returns = data.pct_change().abs()
        large_moves = returns > 0.5
        if large_moves.any().any():
            issues.append("Large price gaps detected (>50% moves)")

        # Check date index
        if not isinstance(data.index, pd.DatetimeIndex):
            issues.append("Index is not DatetimeIndex")

        is_valid = len(issues) == 0
        return is_valid, issues

    @staticmethod
    def validate_returns_data(data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate returns data.

        Parameters:
        -----------
        data : pd.DataFrame
            Returns data to validate

        Returns:
        --------
        Tuple[bool, List[str]]
            (is_valid, list_of_issues)
        """
        issues = []

        # Check if data is empty
        if data.empty:
            issues.append("Data is empty")
            return False, issues

        # Check for extreme returns
        extreme_returns = data.abs() > 1.0  # >100% daily returns
        if extreme_returns.any().any():
            issues.append("Extreme returns detected (>100%)")

        # Check for missing data
        missing_pct = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        if missing_pct > 0.2:
            issues.append(f"High missing data percentage: {missing_pct:.2%}")

        # Check for constant returns (suspicious)
        for col in data.columns:
            if data[col].std() == 0:
                issues.append(f"Constant returns detected for {col}")

        is_valid = len(issues) == 0
        return is_valid, issues


class MathUtils:
    """Mathematical utility functions."""

    @staticmethod
    def safe_divide(
        numerator: Union[float, np.ndarray],
        denominator: Union[float, np.ndarray],
        default_value: float = 0.0,
    ) -> Union[float, np.ndarray]:
        """
        Safe division that handles division by zero.

        Parameters:
        -----------
        numerator : Union[float, np.ndarray]
            Numerator
        denominator : Union[float, np.ndarray]
            Denominator
        default_value : float
            Value to return when denominator is zero

        Returns:
        --------
        Union[float, np.ndarray]
            Division result
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = np.divide(
                numerator,
                denominator,
                out=np.full_like(numerator, default_value, dtype=float),
                where=(denominator != 0),
            )
        return result

    @staticmethod
    def zscore_normalize(
        data: pd.DataFrame, window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Z-score normalize data.

        Parameters:
        -----------
        data : pd.DataFrame
            Data to normalize
        window : Optional[int]
            Rolling window (if None, use full sample)

        Returns:
        --------
        pd.DataFrame
            Normalized data
        """
        if window is None:
            return (data - data.mean()) / data.std()
        else:
            rolling_mean = data.rolling(window=window).mean()
            rolling_std = data.rolling(window=window).std()
            return (data - rolling_mean) / rolling_std

    @staticmethod
    def rank_normalize(data: pd.DataFrame, axis: int = 1) -> pd.DataFrame:
        """
        Rank normalize data (percentile ranks).

        Parameters:
        -----------
        data : pd.DataFrame
            Data to normalize
        axis : int
            Axis along which to rank (0=columns, 1=rows)

        Returns:
        --------
        pd.DataFrame
            Rank normalized data
        """
        return data.rank(axis=axis, pct=True)

    @staticmethod
    def winsorize_data(
        data: pd.DataFrame, limits: Tuple[float, float] = (0.05, 0.05)
    ) -> pd.DataFrame:
        """
        Winsorize data to handle outliers.

        Parameters:
        -----------
        data : pd.DataFrame
            Data to winsorize
        limits : Tuple[float, float]
            Lower and upper quantile limits

        Returns:
        --------
        pd.DataFrame
            Winsorized data
        """
        from scipy.stats import mstats

        result = data.copy()
        for col in result.columns:
            if result[col].notna().sum() > 0:
                result[col] = pd.Series(
                    mstats.winsorize(result[col].dropna(), limits=limits),
                    index=result[col].dropna().index,
                )
        return result

    @staticmethod
    def calculate_information_coefficient(
        factor_scores: pd.DataFrame,
        forward_returns: pd.DataFrame,
        periods: List[int] = [1, 5, 21],
    ) -> Dict[int, float]:
        """
        Calculate Information Coefficient (IC) between factors and forward returns.

        Parameters:
        -----------
        factor_scores : pd.DataFrame
            Factor scores
        forward_returns : pd.DataFrame
            Forward returns
        periods : List[int]
            Forward return periods to calculate

        Returns:
        --------
        Dict[int, float]
            IC by period
        """
        ics = {}

        for period in periods:
            # Calculate period forward returns
            forward_ret = forward_returns.shift(-period)

            # Calculate correlations
            correlations = []
            for date in factor_scores.index:
                if date in forward_ret.index:
                    factor_cross_section = factor_scores.loc[date]
                    return_cross_section = forward_ret.loc[date]

                    # Get common assets
                    common_assets = factor_cross_section.index.intersection(
                        return_cross_section.index
                    )

                    if len(common_assets) > 10:  # Minimum number of assets
                        corr = factor_cross_section[common_assets].corr(
                            return_cross_section[common_assets]
                        )
                        if not np.isnan(corr):
                            correlations.append(corr)

            ics[period] = np.mean(correlations) if correlations else np.nan

        return ics


class DateUtils:
    """Date utility functions."""

    @staticmethod
    def get_business_days(start_date: str, end_date: str) -> pd.DatetimeIndex:
        """
        Get business days between start and end dates.

        Parameters:
        -----------
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)

        Returns:
        --------
        pd.DatetimeIndex
            Business days
        """
        return pd.bdate_range(start=start_date, end=end_date)

    @staticmethod
    def get_month_ends(start_date: str, end_date: str) -> pd.DatetimeIndex:
        """
        Get month end dates between start and end dates.

        Parameters:
        -----------
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)

        Returns:
        --------
        pd.DatetimeIndex
            Month end dates
        """
        return pd.date_range(start=start_date, end=end_date, freq="M")

    @staticmethod
    def add_business_days(date: datetime, days: int) -> datetime:
        """
        Add business days to a date.

        Parameters:
        -----------
        date : datetime
            Starting date
        days : int
            Number of business days to add

        Returns:
        --------
        datetime
            Resulting date
        """
        return pd.bdate_range(start=date, periods=days + 1)[-1].to_pydatetime()


class FileUtils:
    """File utility functions."""

    @staticmethod
    def ensure_directory(path: str) -> None:
        """Ensure directory exists, create if not."""
        Path(path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def save_dataframe(
        df: pd.DataFrame, filepath: str, format: str = "parquet"
    ) -> None:
        """
        Save DataFrame to file.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to save
        filepath : str
            Output file path
        format : str
            File format ('parquet', 'csv', 'pickle')
        """
        FileUtils.ensure_directory(os.path.dirname(filepath))

        if format == "parquet":
            df.to_parquet(filepath)
        elif format == "csv":
            df.to_csv(filepath)
        elif format == "pickle":
            df.to_pickle(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")

    @staticmethod
    def load_dataframe(filepath: str, format: Optional[str] = None) -> pd.DataFrame:
        """
        Load DataFrame from file.

        Parameters:
        -----------
        filepath : str
            Input file path
        format : Optional[str]
            File format (auto-detected if None)

        Returns:
        --------
        pd.DataFrame
            Loaded DataFrame
        """
        if format is None:
            # Auto-detect format from extension
            ext = os.path.splitext(filepath)[1].lower()
            if ext == ".parquet":
                format = "parquet"
            elif ext == ".csv":
                format = "csv"
            elif ext == ".pickle" or ext == ".pkl":
                format = "pickle"
            else:
                raise ValueError(f"Cannot auto-detect format for {filepath}")

        if format == "parquet":
            return pd.read_parquet(filepath)
        elif format == "csv":
            return pd.read_csv(filepath, index_col=0, parse_dates=True)
        elif format == "pickle":
            return pd.read_pickle(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Global configuration instance
config = ConfigManager()


def load_config(config_path: Optional[str] = None) -> Dict:
    """
    Load configuration from file or return default configuration.

    Parameters:
    -----------
    config_path : Optional[str]
        Path to configuration file. If None, uses default config locations.

    Returns:
    --------
    Dict
        Configuration dictionary
    """
    if config_path is None:
        # Try default config locations
        possible_paths = [
            "config/settings.yaml",
            "config/environments.yaml",
            "../config/settings.yaml",
            "../config/environments.yaml",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                if config_path.endswith(".json"):
                    return json.load(f)
                elif config_path.endswith(".yaml") or config_path.endswith(".yml"):
                    return yaml.safe_load(f)
                else:
                    raise ValueError(
                        "Unsupported config file format. Use .json or .yaml"
                    )
        except Exception as e:
            logging.warning(f"Could not load config from {config_path}: {e}")

    # Return default config if file loading fails
    return {
        "environment": "development",
        "data": {
            "primary_provider": "yahoo",
            "cache_enabled": True,
            "cache_duration_days": 1,
        },
        "factors": {
            "normalization_method": "zscore",
            "winsorize_limits": [0.05, 0.05],
            "min_history_days": 252,
        },
        "portfolio": {
            "default_optimization": "mean_variance",
            "transaction_cost": 0.001,
            "rebalance_frequency": "monthly",
            "max_weight": 0.1,
            "min_weight": 0.0,
        },
        "backtesting": {
            "initial_capital": 100000,
            "benchmark": "SPY",
            "lookback_window": 252,
        },
        "visualization": {
            "default_theme": "plotly_white",
            "figure_size": [12, 8],
            "color_palette": "Set1",
        },
    }
