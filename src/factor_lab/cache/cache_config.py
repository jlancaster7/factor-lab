"""
Cache configuration settings for FMP data provider.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os


@dataclass
class CacheConfig:
    """Configuration settings for the FMP cache system."""
    
    # Cache directory path
    cache_dir: Path = Path("data/cache/fmp")
    
    # Cache expiration settings (in seconds)
    default_ttl: int = 86400  # 24 hours default
    income_statement_ttl: int = 7 * 86400  # 7 days for quarterly reports
    balance_sheet_ttl: int = 7 * 86400  # 7 days for quarterly reports
    cash_flow_ttl: int = 7 * 86400  # 7 days for quarterly reports
    financial_ratios_ttl: int = 86400  # 1 day for calculated ratios
    price_data_ttl: int = 3600  # 1 hour for price data
    
    # Cache size management
    max_cache_size_mb: int = 1000  # 1GB max cache size
    cleanup_threshold_mb: int = 800  # Start cleanup at 800MB
    
    # Cache behavior
    enable_compression: bool = True  # Use gzip compression
    cache_version: str = "1.0"  # Increment when schema changes
    
    # Performance settings
    batch_size: int = 100  # Number of items to cache in batch operations
    async_writes: bool = True  # Write to cache asynchronously
    
    # Debugging and monitoring
    log_cache_hits: bool = True
    log_cache_misses: bool = True
    collect_statistics: bool = True
    
    def __post_init__(self):
        """Create cache directory if it doesn't exist."""
        self.cache_dir = Path(self.cache_dir)
        if not self.cache_dir.is_absolute():
            # Make it relative to the project root
            project_root = Path(__file__).parent.parent.parent.parent
            self.cache_dir = project_root / self.cache_dir
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different data types
        for subdir in ["income_statements", "balance_sheets", "cash_flows", 
                       "financial_ratios", "price_data", "metadata"]:
            (self.cache_dir / subdir).mkdir(exist_ok=True)
    
    @classmethod
    def from_env(cls) -> "CacheConfig":
        """Create cache config from environment variables."""
        kwargs = {}
        
        # Override from environment if available
        if cache_dir := os.getenv("FMP_CACHE_DIR"):
            kwargs["cache_dir"] = Path(cache_dir)
        
        if default_ttl := os.getenv("FMP_CACHE_TTL"):
            kwargs["default_ttl"] = int(default_ttl)
        
        if max_size := os.getenv("FMP_CACHE_MAX_SIZE_MB"):
            kwargs["max_cache_size_mb"] = int(max_size)
        
        if compression := os.getenv("FMP_CACHE_COMPRESSION"):
            kwargs["enable_compression"] = compression.lower() == "true"
        
        return cls(**kwargs)