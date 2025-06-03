"""
Test cache integration with FMP provider.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import time

from factor_lab.data import FMPProvider
from factor_lab.cache import CacheConfig


class TestCacheIntegration:
    """Test cache integration in FMP provider."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def cache_config(self, temp_cache_dir):
        """Create test cache configuration."""
        return CacheConfig(
            cache_dir=temp_cache_dir,
            default_ttl=3600,  # 1 hour
            income_statement_ttl=7200,  # 2 hours for testing
            enable_compression=True,
            log_cache_hits=True,
            log_cache_misses=True
        )
    
    @pytest.fixture
    def mock_fmp_provider(self, cache_config):
        """Create FMP provider with mocked API key and cache config."""
        with patch.object(FMPProvider, '_get_api_key', return_value='test_api_key'):
            provider = FMPProvider(cache_config=cache_config)
            return provider
    
    def test_cache_initialization(self, mock_fmp_provider, temp_cache_dir):
        """Test that cache is properly initialized."""
        assert mock_fmp_provider.cache is not None
        assert mock_fmp_provider.cache_config.cache_dir == temp_cache_dir
        
        # Check cache subdirectories exist
        expected_subdirs = ["income_statements", "balance_sheets", "cash_flows", 
                           "financial_ratios", "price_data", "metadata"]
        for subdir in expected_subdirs:
            assert (temp_cache_dir / subdir).exists()
    
    @patch.object(FMPProvider, '_make_request')
    def test_income_statement_caching(self, mock_request, mock_fmp_provider):
        """Test income statement data is cached and retrieved."""
        # Mock API response
        mock_data = [
            {
                "date": "2024-09-30",
                "symbol": "AAPL",
                "revenue": 94930000000,
                "netIncome": 14736000000,
                "acceptedDate": "2024-11-01"
            }
        ]
        mock_request.return_value = mock_data
        
        # First call - should hit API
        result1 = mock_fmp_provider._fetch_income_statement("AAPL", limit=5, period="quarter")
        assert result1 == mock_data
        assert mock_request.call_count == 1
        
        # Check cache stats to ensure data was written
        stats = mock_fmp_provider.get_cache_stats()
        print(f"Cache stats after first call: {stats}")
        
        # Second call - should hit cache
        result2 = mock_fmp_provider._fetch_income_statement("AAPL", limit=5, period="quarter")
        assert result2 == mock_data
        
        # Check cache stats again
        stats2 = mock_fmp_provider.get_cache_stats()
        print(f"Cache stats after second call: {stats2}")
        
        assert mock_request.call_count == 1  # No additional API call
        
        # Check cache statistics
        stats = mock_fmp_provider.get_cache_stats()
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1
        assert stats["writes"] >= 1
    
    @patch.object(FMPProvider, '_make_request')
    def test_all_statement_types_cached(self, mock_request, mock_fmp_provider):
        """Test all statement types are properly cached."""
        # Mock different responses for each statement type
        mock_responses = {
            "income": [{"type": "income", "revenue": 1000000}],
            "balance": [{"type": "balance", "assets": 2000000}],
            "cash": [{"type": "cash", "operatingCashFlow": 500000}],
            "ratios": [{"type": "ratios", "peRatio": 25.5}]
        }
        
        def mock_response(url, params):
            if "income-statement" in url:
                return mock_responses["income"]
            elif "balance-sheet" in url:
                return mock_responses["balance"]
            elif "cash-flow" in url:
                return mock_responses["cash"]
            elif "ratios" in url:
                return mock_responses["ratios"]
            return None
        
        mock_request.side_effect = mock_response
        
        # Fetch each statement type
        income = mock_fmp_provider._fetch_income_statement("AAPL", period="quarter")
        balance = mock_fmp_provider._fetch_balance_sheet("AAPL", period="quarter")
        cash = mock_fmp_provider._fetch_cash_flow("AAPL", period="quarter")
        ratios = mock_fmp_provider._fetch_financial_ratios("AAPL")
        
        assert income == mock_responses["income"]
        assert balance == mock_responses["balance"]
        assert cash == mock_responses["cash"]
        assert ratios == mock_responses["ratios"]
        
        # All should be cached now
        assert mock_request.call_count == 4
        
        # Fetch again - all should come from cache
        mock_request.reset_mock()
        
        income2 = mock_fmp_provider._fetch_income_statement("AAPL", period="quarter")
        balance2 = mock_fmp_provider._fetch_balance_sheet("AAPL", period="quarter")
        cash2 = mock_fmp_provider._fetch_cash_flow("AAPL", period="quarter")
        ratios2 = mock_fmp_provider._fetch_financial_ratios("AAPL")
        
        assert mock_request.call_count == 0  # No API calls
        assert income2 == income
        assert balance2 == balance
        assert cash2 == cash
        assert ratios2 == ratios
    
    @patch.object(FMPProvider, '_make_request')
    def test_price_data_caching(self, mock_request, mock_fmp_provider):
        """Test historical price data caching with date ranges."""
        mock_price_data = {
            "historical": [
                {"date": "2024-01-01", "close": 185.50},
                {"date": "2024-01-02", "close": 186.25}
            ]
        }
        mock_request.return_value = mock_price_data
        
        # First call with date range
        prices1 = mock_fmp_provider._fetch_historical_prices(
            "AAPL", from_date="2024-01-01", to_date="2024-01-31"
        )
        assert prices1 == mock_price_data["historical"]
        assert mock_request.call_count == 1
        
        # Same date range - should hit cache
        prices2 = mock_fmp_provider._fetch_historical_prices(
            "AAPL", from_date="2024-01-01", to_date="2024-01-31"
        )
        assert prices2 == mock_price_data["historical"]
        assert mock_request.call_count == 1  # No additional call
        
        # Different date range - should hit API
        prices3 = mock_fmp_provider._fetch_historical_prices(
            "AAPL", from_date="2024-02-01", to_date="2024-02-28"
        )
        assert mock_request.call_count == 2  # New API call
    
    @patch.object(FMPProvider, '_make_request')
    def test_cache_invalidation(self, mock_request, mock_fmp_provider):
        """Test cache invalidation functionality."""
        mock_data = [{"symbol": "AAPL", "revenue": 1000000}]
        mock_request.return_value = mock_data
        
        # Cache some data
        mock_fmp_provider._fetch_income_statement("AAPL", period="quarter")
        mock_fmp_provider._fetch_balance_sheet("AAPL", period="quarter")
        
        # Clear cache for AAPL
        mock_fmp_provider.clear_cache(symbol="AAPL")
        
        # Next fetch should hit API again
        mock_request.reset_mock()
        result = mock_fmp_provider._fetch_income_statement("AAPL", period="quarter")
        assert mock_request.call_count == 1
    
    @patch.object(FMPProvider, '_make_request')
    def test_cache_statistics(self, mock_request, mock_fmp_provider):
        """Test cache statistics collection."""
        mock_data = [{"symbol": "AAPL", "data": "test"}]
        mock_request.return_value = mock_data
        
        # Initial stats
        stats = mock_fmp_provider.get_cache_stats()
        initial_hits = stats.get("hits", 0)
        initial_misses = stats.get("misses", 0)
        
        # Make some requests
        mock_fmp_provider._fetch_income_statement("AAPL", period="quarter")  # Miss
        mock_fmp_provider._fetch_income_statement("AAPL", period="quarter")  # Hit
        mock_fmp_provider._fetch_income_statement("MSFT", period="quarter")  # Miss
        
        # Check updated stats
        stats = mock_fmp_provider.get_cache_stats()
        assert stats["hits"] == initial_hits + 1
        assert stats["misses"] == initial_misses + 2
        assert stats["writes"] >= 2
        assert "hit_rate" in stats
        assert "cache_size_mb" in stats
    
    @patch.object(FMPProvider, '_make_request')
    def test_cache_warming(self, mock_request, mock_fmp_provider):
        """Test cache warming functionality."""
        mock_request.return_value = [{"data": "test"}]
        
        # Warm cache for multiple symbols
        symbols = ["AAPL", "MSFT", "GOOGL"]
        mock_fmp_provider.warm_cache(symbols, statement_types=["income_statement", "balance_sheet"])
        
        # Should have made 6 API calls (3 symbols × 2 statement types)
        assert mock_request.call_count == 6
        
        # All subsequent requests should hit cache
        mock_request.reset_mock()
        for symbol in symbols:
            mock_fmp_provider._fetch_income_statement(symbol, limit=20, period="quarter")
            mock_fmp_provider._fetch_balance_sheet(symbol, limit=20, period="quarter")
        
        assert mock_request.call_count == 0  # All from cache
    
    def test_cache_expiration(self, mock_fmp_provider):
        """Test that cache respects TTL settings."""
        with patch.object(FMPProvider, '_make_request') as mock_request:
            mock_data = [{"symbol": "AAPL", "data": "test"}]
            mock_request.return_value = mock_data
            
            # Set very short TTL for testing
            mock_fmp_provider.cache_config.income_statement_ttl = 1  # 1 second
            
            # First fetch
            result1 = mock_fmp_provider._fetch_income_statement("AAPL", period="quarter")
            assert mock_request.call_count == 1
            
            # Immediate second fetch - should hit cache
            result2 = mock_fmp_provider._fetch_income_statement("AAPL", period="quarter")
            assert mock_request.call_count == 1
            
            # Wait for expiration
            time.sleep(2)
            
            # Should hit API again due to expiration
            result3 = mock_fmp_provider._fetch_income_statement("AAPL", period="quarter")
            assert mock_request.call_count == 2
    
    @patch.object(FMPProvider, '_make_request')
    def test_different_parameters_cached_separately(self, mock_request, mock_fmp_provider):
        """Test that different parameters create different cache entries."""
        mock_request.return_value = [{"data": "test"}]
        
        # Fetch with different parameters
        mock_fmp_provider._fetch_income_statement("AAPL", limit=5, period="quarter")
        mock_fmp_provider._fetch_income_statement("AAPL", limit=10, period="quarter")
        mock_fmp_provider._fetch_income_statement("AAPL", limit=5, period="annual")
        
        # Should have made 3 API calls for different parameter combinations
        assert mock_request.call_count == 3
        
        # Same parameters should hit cache
        mock_request.reset_mock()
        mock_fmp_provider._fetch_income_statement("AAPL", limit=5, period="quarter")
        assert mock_request.call_count == 0