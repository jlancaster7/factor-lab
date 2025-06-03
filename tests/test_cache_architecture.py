"""
Test cache architecture design for FMP provider.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import time

from factor_lab.cache import CacheManager, CacheKey, CacheConfig


class TestCacheArchitecture:
    """Test the cache architecture components."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def cache_config(self, temp_cache_dir):
        """Create test cache configuration."""
        config = CacheConfig(
            cache_dir=temp_cache_dir,
            default_ttl=60,  # 1 minute for testing
            enable_compression=True,
            async_writes=False,  # Synchronous for testing
            log_cache_hits=True,
            log_cache_misses=True
        )
        return config
    
    @pytest.fixture
    def cache_manager(self, cache_config):
        """Create cache manager instance."""
        return CacheManager(config=cache_config)
    
    def test_cache_key_creation(self):
        """Test cache key creation and string generation."""
        # Test basic cache key
        key = CacheKey(
            symbol="AAPL",
            statement_type="income_statement",
            period="quarterly",
            fiscal_date="2024-09-30",
            version="1.0"
        )
        
        assert key.symbol == "AAPL"
        assert key.statement_type == "income_statement"
        assert key.period == "quarterly"
        assert key.fiscal_date == "2024-09-30"
        
        # Test key string generation
        key_string = key.to_string()
        assert key_string == "AAPL_income_statement_quarterly_2024-09-30_v1.0"
        
        # Test filename generation
        filename = key.to_filename()
        assert filename == "AAPL_income_statement_quarterly_2024-09-30_v1.0.json.gz"
        
        # Test subdirectory
        assert key.get_subdir() == "income_statements"
    
    def test_cache_key_parsing(self):
        """Test parsing cache key from string."""
        key_string = "MSFT_balance_sheet_annual_2024-06-30_v1.0"
        key = CacheKey.from_string(key_string)
        
        assert key.symbol == "MSFT"
        assert key.statement_type == "balance_sheet"
        assert key.period == "annual"
        assert key.fiscal_date == "2024-06-30"
        assert key.version == "1.0"
    
    def test_cache_key_with_limit(self):
        """Test cache key with limit parameter."""
        key = CacheKey(
            symbol="GOOGL",
            statement_type="financial_ratios",
            period="quarterly",
            limit=20,
            version="1.0"
        )
        
        key_string = key.to_string()
        assert "limit20" in key_string
    
    def test_cache_key_validation(self):
        """Test cache key validation."""
        # Invalid statement type
        with pytest.raises(ValueError):
            CacheKey(
                symbol="AAPL",
                statement_type="invalid_type",
                period="quarterly"
            )
        
        # Invalid period
        with pytest.raises(ValueError):
            CacheKey(
                symbol="AAPL",
                statement_type="income_statement",
                period="invalid_period"
            )
    
    def test_cache_manager_set_and_get(self, cache_manager):
        """Test basic cache set and get operations."""
        key = CacheKey(
            symbol="AAPL",
            statement_type="income_statement",
            period="quarterly",
            fiscal_date="2024-09-30"
        )
        
        test_data = {
            "revenue": 94930000000,
            "netIncome": 14736000000,
            "date": "2024-09-30",
            "acceptedDate": "2024-11-01"
        }
        
        # Set data in cache
        cache_manager.set(key, test_data)
        
        # Retrieve data from cache
        cached_data = cache_manager.get(key)
        assert cached_data is not None
        assert cached_data["revenue"] == 94930000000
        assert cached_data["netIncome"] == 14736000000
        
        # Check cache hit statistics
        stats = cache_manager.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0
        assert stats["writes"] == 1
    
    def test_cache_expiration(self, cache_manager):
        """Test cache expiration based on TTL."""
        # Create key with very short TTL
        key = CacheKey(
            symbol="MSFT",
            statement_type="balance_sheet",
            period="quarterly",
            fiscal_date="2024-06-30"
        )
        
        test_data = {"totalAssets": 500000000000}
        
        # Set with 1 second TTL
        cache_manager.set(key, test_data, ttl_override=1)
        
        # Should get data immediately
        assert cache_manager.get(key) is not None
        
        # Wait for expiration
        time.sleep(2)
        
        # Should return None after expiration
        assert cache_manager.get(key) is None
        
        # Check miss was recorded
        stats = cache_manager.get_stats()
        assert stats["misses"] >= 1
    
    def test_cache_version_invalidation(self, cache_manager):
        """Test cache invalidation based on version mismatch."""
        key_v1 = CacheKey(
            symbol="GOOGL",
            statement_type="cash_flow",
            period="annual",
            fiscal_date="2023-12-31",
            version="1.0"
        )
        
        test_data = {"operatingCashFlow": 110000000000}
        cache_manager.set(key_v1, test_data)
        
        # Create key with different version
        key_v2 = CacheKey(
            symbol="GOOGL",
            statement_type="cash_flow",
            period="annual",
            fiscal_date="2023-12-31",
            version="2.0"
        )
        
        # Should not get data due to version mismatch
        assert cache_manager.get(key_v2) is None
    
    def test_batch_operations(self, cache_manager):
        """Test batch set and get operations."""
        # Create multiple keys
        keys = CacheKey.for_batch(
            symbols=["AAPL", "MSFT", "GOOGL"],
            statement_type="financial_ratios",
            period="quarterly"
        )
        
        # Create test data
        entries = {}
        for i, key in enumerate(keys):
            # Set fiscal date for each key
            key.fiscal_date = "2024-09-30"
            entries[key] = {
                "peRatio": 25.0 + i,
                "pbRatio": 5.0 + i,
                "date": "2024-09-30"
            }
        
        # Batch set
        cache_manager.set_batch(entries)
        
        # Batch get
        results = cache_manager.get_batch(keys)
        assert len(results) == 3
        
        # Verify data
        for key in keys:
            key_string = key.to_string()
            assert key_string in results
            assert "peRatio" in results[key_string]
    
    def test_cache_invalidation(self, cache_manager):
        """Test cache invalidation methods."""
        # Set multiple entries for a symbol
        for quarter in ["2024-03-31", "2024-06-30", "2024-09-30"]:
            key = CacheKey(
                symbol="NVDA",
                statement_type="income_statement",
                period="quarterly",
                fiscal_date=quarter
            )
            cache_manager.set(key, {"revenue": 1000000000})
        
        # Verify all entries exist
        for quarter in ["2024-03-31", "2024-06-30", "2024-09-30"]:
            key = CacheKey(
                symbol="NVDA",
                statement_type="income_statement",
                period="quarterly",
                fiscal_date=quarter
            )
            assert cache_manager.get(key) is not None
        
        # Invalidate all entries for symbol
        cache_manager.invalidate_symbol("NVDA")
        
        # Verify all entries are gone
        for quarter in ["2024-03-31", "2024-06-30", "2024-09-30"]:
            key = CacheKey(
                symbol="NVDA",
                statement_type="income_statement",
                period="quarterly",
                fiscal_date=quarter
            )
            assert cache_manager.get(key) is None
    
    def test_cache_statistics(self, cache_manager):
        """Test cache statistics collection."""
        # Perform various operations
        key = CacheKey(
            symbol="TSLA",
            statement_type="balance_sheet",
            period="annual",
            fiscal_date="2023-12-31"
        )
        
        # Miss
        cache_manager.get(key)
        
        # Write
        cache_manager.set(key, {"data": "test"})
        
        # Hit
        cache_manager.get(key)
        
        # Get statistics
        stats = cache_manager.get_stats()
        
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1
        assert stats["writes"] >= 1
        assert stats["hit_rate"] > 0
        assert "cache_size_mb" in stats
        assert "cache_dir" in stats
    
    def test_cache_subdirectories(self, cache_config):
        """Test that cache subdirectories are created properly."""
        expected_subdirs = [
            "income_statements",
            "balance_sheets",
            "cash_flows",
            "financial_ratios",
            "price_data",
            "metadata"
        ]
        
        for subdir in expected_subdirs:
            subdir_path = cache_config.cache_dir / subdir
            assert subdir_path.exists()
            assert subdir_path.is_dir()
    
    def test_cache_config_from_env(self, monkeypatch, temp_cache_dir):
        """Test cache configuration from environment variables."""
        monkeypatch.setenv("FMP_CACHE_DIR", str(temp_cache_dir))
        monkeypatch.setenv("FMP_CACHE_TTL", "3600")
        monkeypatch.setenv("FMP_CACHE_MAX_SIZE_MB", "500")
        monkeypatch.setenv("FMP_CACHE_COMPRESSION", "false")
        
        config = CacheConfig.from_env()
        
        assert config.cache_dir == temp_cache_dir
        assert config.default_ttl == 3600
        assert config.max_cache_size_mb == 500
        assert config.enable_compression is False
    
    def test_cache_cleanup(self, temp_cache_dir):
        """Test cache cleanup when size threshold is exceeded."""
        # Create config with very low threshold
        config = CacheConfig(
            cache_dir=temp_cache_dir,
            max_cache_size_mb=1,  # 1MB max
            cleanup_threshold_mb=0.8,  # 800KB threshold
            enable_compression=False  # No compression for predictable sizes
        )
        
        manager = CacheManager(config=config)
        
        # Create many cache entries to exceed threshold
        for i in range(100):
            key = CacheKey(
                symbol=f"TEST{i}",
                statement_type="income_statement",
                period="quarterly",
                fiscal_date="2024-01-01"
            )
            # Large data to exceed threshold quickly
            large_data = {"data": "x" * 10000}  # ~10KB per entry
            manager.set(key, large_data)
        
        # Check that cleanup was triggered
        stats = manager.get_stats()
        assert stats["evictions"] > 0
        
        # Cache size should be below max
        assert stats["cache_size_mb"] < config.max_cache_size_mb