"""
Cache key generation and management for FMP data.
"""

from dataclasses import dataclass
from typing import Optional, Union, List
from datetime import datetime
import hashlib
import json


@dataclass
class CacheKey:
    """
    Structured cache key for FMP data.
    
    Key format: {symbol}_{statement_type}_{period}_{fiscal_date}_{version}
    
    Examples:
        AAPL_income_statement_quarterly_2024-09-30_v1.0
        MSFT_balance_sheet_annual_2024-06-30_v1.0
        GOOGL_financial_ratios_quarterly_2024-03-31_v1.0
        NVDA_price_daily_2024-01-15_v1.0
    """
    
    symbol: str
    statement_type: str  # income_statement, balance_sheet, cash_flow, financial_ratios, price
    period: str  # quarterly, annual, daily
    fiscal_date: Optional[Union[str, datetime]] = None
    version: str = "1.0"
    
    # Additional metadata for cache management
    accepted_date: Optional[Union[str, datetime]] = None
    limit: Optional[int] = None
    
    def __post_init__(self):
        """Validate and normalize cache key components."""
        # Normalize symbol to uppercase
        self.symbol = self.symbol.upper()
        
        # Validate statement type
        valid_types = ["income_statement", "balance_sheet", "cash_flow", 
                      "financial_ratios", "price"]
        if self.statement_type not in valid_types:
            raise ValueError(f"Invalid statement type: {self.statement_type}")
        
        # Validate period
        valid_periods = ["quarterly", "annual", "daily"]
        if self.period not in valid_periods:
            raise ValueError(f"Invalid period: {self.period}")
        
        # Convert dates to strings
        if isinstance(self.fiscal_date, datetime):
            self.fiscal_date = self.fiscal_date.strftime("%Y-%m-%d")
        if isinstance(self.accepted_date, datetime):
            self.accepted_date = self.accepted_date.strftime("%Y-%m-%d")
    
    def to_string(self) -> str:
        """Generate the cache key string."""
        parts = [
            self.symbol,
            self.statement_type,
            self.period
        ]
        
        if self.fiscal_date:
            parts.append(self.fiscal_date)
        
        if self.limit:
            parts.append(f"limit{self.limit}")
        
        parts.append(f"v{self.version}")
        
        return "_".join(parts)
    
    def to_filename(self) -> str:
        """Generate a safe filename for the cache entry."""
        # Use the key string but replace problematic characters
        filename = self.to_string().replace("/", "-").replace(":", "-")
        return f"{filename}.json.gz" if self.version else f"{filename}.json"
    
    def get_subdir(self) -> str:
        """Get the subdirectory for this cache entry."""
        subdir_map = {
            "income_statement": "income_statements",
            "balance_sheet": "balance_sheets",
            "cash_flow": "cash_flows",
            "financial_ratios": "financial_ratios",
            "price": "price_data"
        }
        return subdir_map.get(self.statement_type, "metadata")
    
    def to_hash(self) -> str:
        """Generate a hash of the cache key for quick lookups."""
        key_data = {
            "symbol": self.symbol,
            "statement_type": self.statement_type,
            "period": self.period,
            "fiscal_date": self.fiscal_date,
            "limit": self.limit,
            "version": self.version
        }
        key_json = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_json.encode()).hexdigest()[:16]
    
    @classmethod
    def from_string(cls, key_string: str) -> "CacheKey":
        """Parse a cache key string back into a CacheKey object."""
        parts = key_string.split("_")
        
        if len(parts) < 3:
            raise ValueError(f"Invalid cache key string: {key_string}")
        
        symbol = parts[0]
        statement_type = parts[1]
        
        # Handle multi-word statement types
        if parts[1] == "balance" and len(parts) > 2 and parts[2] == "sheet":
            statement_type = "balance_sheet"
            period = parts[3] if len(parts) > 3 else "quarterly"
            remaining_parts = parts[4:]
        elif parts[1] == "income" and len(parts) > 2 and parts[2] == "statement":
            statement_type = "income_statement"
            period = parts[3] if len(parts) > 3 else "quarterly"
            remaining_parts = parts[4:]
        elif parts[1] == "cash" and len(parts) > 2 and parts[2] == "flow":
            statement_type = "cash_flow"
            period = parts[3] if len(parts) > 3 else "quarterly"
            remaining_parts = parts[4:]
        elif parts[1] == "financial" and len(parts) > 2 and parts[2] == "ratios":
            statement_type = "financial_ratios"
            period = parts[3] if len(parts) > 3 else "quarterly"
            remaining_parts = parts[4:]
        else:
            period = parts[2] if len(parts) > 2 else "quarterly"
            remaining_parts = parts[3:]
        
        # Parse optional components
        fiscal_date = None
        limit = None
        version = "1.0"
        
        for part in remaining_parts:
            if part.startswith("v"):
                version = part[1:]
            elif part.startswith("limit"):
                limit = int(part[5:])
            elif len(part) >= 10:  # Date format or date range
                fiscal_date = part
        
        return cls(
            symbol=symbol,
            statement_type=statement_type,
            period=period,
            fiscal_date=fiscal_date,
            limit=limit,
            version=version
        )
    
    @classmethod
    def for_batch(cls, symbols: List[str], statement_type: str, 
                  period: str = "quarterly", version: str = "1.0") -> List["CacheKey"]:
        """Generate cache keys for a batch of symbols."""
        return [
            cls(symbol=symbol, statement_type=statement_type, 
                period=period, version=version)
            for symbol in symbols
        ]