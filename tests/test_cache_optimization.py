"""
Test cache performance optimization features.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import time

from factor_lab.data import FMPProvider
from factor_lab.cache import CacheConfig, CachePreloadStrategy, CacheOptimizer


class TestCacheOptimization:
    """Test cache optimization features."""
    
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
            default_ttl=3600,
            enable_compression=True,
            max_cache_size_mb=100,
            cleanup_threshold_mb=80
        )
    
    @pytest.fixture
    def mock_fmp_provider(self, cache_config):
        """Create FMP provider with mocked API key."""
        with patch.object(FMPProvider, '_get_api_key', return_value='test_api_key'):
            provider = FMPProvider(cache_config=cache_config)
            return provider
    
    @patch.object(FMPProvider, '_make_request')
    def test_batch_fetch_statements(self, mock_request, mock_fmp_provider):
        """Test batch fetching of statements."""
        # Mock different responses for different symbols
        def mock_response(url, params):
            symbol = url.split('/')[-1]
            if "income-statement" in url:
                return [{"symbol": symbol, "type": "income", "revenue": 1000000}]
            elif "balance-sheet" in url:
                return [{"symbol": symbol, "type": "balance", "assets": 2000000}]
            elif "ratios" in url:
                return [{"symbol": symbol, "type": "ratios", "peRatio": 25.5}]
            return None
        
        mock_request.side_effect = mock_response
        
        # Batch fetch for multiple symbols
        symbols = ["AAPL", "MSFT", "GOOGL"]
        results = mock_fmp_provider.batch_fetch_statements(
            symbols=symbols,
            statement_types=["income_statement", "balance_sheet", "financial_ratios"],
            period="quarter",
            limit=20
        )
        
        # Verify results structure
        assert len(results) == 3
        for symbol in symbols:
            assert symbol in results
            assert "income_statement" in results[symbol]
            assert "balance_sheet" in results[symbol]
            assert "financial_ratios" in results[symbol]
        
        # Second batch fetch should use cache
        mock_request.reset_mock()
        results2 = mock_fmp_provider.batch_fetch_statements(
            symbols=symbols,
            statement_types=["income_statement", "balance_sheet"],
            period="quarter",
            limit=20
        )
        
        # Should have minimal API calls (only for financial ratios if not cached)
        assert mock_request.call_count < len(symbols) * 2
    
    def test_preload_cache_from_results(self, mock_fmp_provider):
        """Test preloading cache from batch results."""
        # Create sample results
        results = {
            "AAPL": {
                "income_statement": [{"symbol": "AAPL", "revenue": 1000000, "acceptedDate": "2024-01-01"}],
                "balance_sheet": [{"symbol": "AAPL", "assets": 2000000}]
            },
            "MSFT": {
                "income_statement": [{"symbol": "MSFT", "revenue": 1500000}],
                "financial_ratios": [{"symbol": "MSFT", "peRatio": 30}]
            }
        }
        
        # Preload cache
        mock_fmp_provider.preload_cache_from_results(results)
        
        # Check cache stats
        stats = mock_fmp_provider.get_cache_stats()
        assert stats["writes"] >= 4  # 4 entries should be cached
    
    def test_cache_memory_usage(self, mock_fmp_provider):
        """Test cache memory usage reporting."""
        # Add some data to cache
        with patch.object(FMPProvider, '_make_request') as mock_request:
            mock_request.return_value = [{"symbol": "AAPL", "data": "x" * 1000}]
            
            # Fetch some data to populate cache
            mock_fmp_provider._fetch_income_statement("AAPL", period="quarter")
            mock_fmp_provider._fetch_balance_sheet("AAPL", period="quarter")
        
        # Get memory usage
        usage = mock_fmp_provider.get_cache_memory_usage()
        
        assert "total_size_mb" in usage
        assert "by_statement_type" in usage
        assert "entry_count" in usage
        assert usage["entry_count"] >= 2
    
    def test_preload_strategy(self, temp_cache_dir):
        """Test cache preload strategy."""
        strategy = CachePreloadStrategy(temp_cache_dir)
        
        # Track some accesses
        for i in range(15):
            strategy.track_access("AAPL", "income_statement")
        
        for i in range(10):
            strategy.track_access("MSFT", "balance_sheet")
        
        for i in range(5):
            strategy.track_access("GOOGL", "financial_ratios")
        
        # Get preload symbols
        symbols = strategy.get_preload_symbols(top_n=3)
        assert "AAPL" in symbols
        assert "MSFT" in symbols
        
        # Get recommendations
        recommendations = strategy.get_preload_recommendations()
        assert "high_frequency" in recommendations
        assert len(recommendations["high_frequency"]) > 0
    
    def test_cache_optimizer(self, mock_fmp_provider):
        """Test cache optimizer functionality."""
        optimizer = mock_fmp_provider.cache_optimizer
        
        # Analyze performance
        analysis = optimizer.analyze_cache_performance()
        
        assert "performance" in analysis
        assert "health" in analysis
        assert "recommendations" in analysis
        assert analysis["health"] >= 0
        assert analysis["health"] <= 100
        
        # Get TTL recommendations
        ttl_recommendations = optimizer.optimize_ttl_settings()
        
        assert "income_statement" in ttl_recommendations
        assert "balance_sheet" in ttl_recommendations
        assert "price_data" in ttl_recommendations
        assert all(ttl > 0 for ttl in ttl_recommendations.values())
    
    @patch.object(FMPProvider, '_make_request')
    def test_smart_preload_cache(self, mock_request, mock_fmp_provider):
        """Test smart cache preloading."""
        mock_request.return_value = [{"symbol": "TEST", "data": "test"}]
        
        # Simulate some access history
        for symbol in ["AAPL", "MSFT", "GOOGL"]:
            for i in range(5):
                mock_fmp_provider.preload_strategy.track_access(symbol, "income_statement")
        
        # Run smart preload
        with patch.object(mock_fmp_provider.preload_strategy, 'get_preload_recommendations') as mock_rec:
            mock_rec.return_value = {
                "high_frequency": ["AAPL", "MSFT"],
                "recent": ["GOOGL"],
                "critical": ["NVDA"]
            }
            
            results = mock_fmp_provider.smart_preload_cache()
        
        assert "symbols_loaded" in results
        assert "success_count" in results
        assert "duration" in results
        assert "cache_health" in results
        assert results["symbols_loaded"] > 0
    
    def test_optimize_cache_settings(self, mock_fmp_provider):
        """Test cache settings optimization."""
        # Simulate poor cache health
        with patch.object(mock_fmp_provider.cache_optimizer, 'analyze_cache_performance') as mock_analysis:
            mock_analysis.return_value = {
                "health": 70,  # Below 80 threshold
                "performance": {"hit_rate": 0.4},
                "recommendations": [{"type": "low_hit_rate", "message": "Test"}]
            }
            
            results = mock_fmp_provider.optimize_cache_settings()
        
        assert "cache_health" in results
        assert "ttl_settings" in results
        assert results["cache_health"] == 70
    
    def test_batch_fetch_with_cache_disabled(self, mock_fmp_provider):
        """Test batch fetch with cache disabled."""
        with patch.object(FMPProvider, '_make_request') as mock_request:
            mock_request.return_value = [{"data": "test"}]
            
            # Fetch with cache disabled
            results = mock_fmp_provider.batch_fetch_statements(
                symbols=["AAPL", "MSFT"],
                statement_types=["income_statement"],
                use_cache=False
            )
            
            # Should make API calls for all requests
            assert mock_request.call_count == 2
            
            # Fetch again with cache disabled
            mock_request.reset_mock()
            results2 = mock_fmp_provider.batch_fetch_statements(
                symbols=["AAPL", "MSFT"],
                statement_types=["income_statement"],
                use_cache=False
            )
            
            # Should make API calls again (not using cache)
            assert mock_request.call_count == 2
    
    def test_preload_strategy_optimal_batch_size(self, temp_cache_dir):
        """Test optimal batch size calculation."""
        strategy = CachePreloadStrategy(temp_cache_dir)
        
        # Default batch size
        assert strategy.get_optimal_batch_size() == 50
        
        # Record some preload operations
        strategy.record_preload(["AAPL", "MSFT"], duration=2.0, success_count=10)
        strategy.record_preload(["GOOGL", "NVDA", "TSLA"], duration=3.0, success_count=12)
        
        # Should calculate based on rate
        optimal_size = strategy.get_optimal_batch_size()
        assert optimal_size >= 20
        assert optimal_size <= 200