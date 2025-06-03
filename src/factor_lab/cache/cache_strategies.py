"""
Cache preloading and optimization strategies for FMP provider.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)


class CachePreloadStrategy:
    """
    Strategies for preloading cache to optimize performance.
    """
    
    def __init__(self, cache_dir: Path):
        """
        Initialize cache preload strategy.
        
        Parameters:
        -----------
        cache_dir : Path
            Cache directory path
        """
        self.cache_dir = cache_dir
        self.metadata_file = cache_dir / "metadata" / "preload_strategy.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load preload metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading preload metadata: {e}")
        return {
            "frequently_accessed": {},
            "last_preload": None,
            "preload_history": []
        }
    
    def _save_metadata(self):
        """Save preload metadata."""
        try:
            self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving preload metadata: {e}")
    
    def track_access(self, symbol: str, statement_type: str):
        """
        Track cache access for preload optimization.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        statement_type : str
            Type of statement accessed
        """
        key = f"{symbol}_{statement_type}"
        if key not in self.metadata["frequently_accessed"]:
            self.metadata["frequently_accessed"][key] = {
                "count": 0,
                "last_access": None
            }
        
        self.metadata["frequently_accessed"][key]["count"] += 1
        self.metadata["frequently_accessed"][key]["last_access"] = datetime.now().isoformat()
        
        # Periodically save metadata
        if sum(item["count"] for item in self.metadata["frequently_accessed"].values()) % 100 == 0:
            self._save_metadata()
    
    def get_preload_symbols(self, top_n: int = 50) -> List[str]:
        """
        Get list of symbols to preload based on access patterns.
        
        Parameters:
        -----------
        top_n : int
            Number of top symbols to preload
            
        Returns:
        --------
        List[str]
            List of symbols to preload
        """
        # Get frequently accessed symbols
        symbol_counts = {}
        for key, data in self.metadata["frequently_accessed"].items():
            symbol = key.split("_")[0]
            if symbol not in symbol_counts:
                symbol_counts[symbol] = 0
            symbol_counts[symbol] += data["count"]
        
        # Sort by access count
        sorted_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N symbols
        return [symbol for symbol, _ in sorted_symbols[:top_n]]
    
    def get_preload_recommendations(self) -> Dict[str, List[str]]:
        """
        Get recommendations for cache preloading.
        
        Returns:
        --------
        Dict[str, List[str]]
            Recommendations by category
        """
        recommendations = {
            "high_frequency": [],  # Frequently accessed
            "recent": [],         # Recently accessed
            "critical": [],       # Critical for performance
            "batch_candidates": []  # Good for batch loading
        }
        
        # High frequency symbols (accessed > 10 times)
        for key, data in self.metadata["frequently_accessed"].items():
            if data["count"] > 10:
                symbol = key.split("_")[0]
                if symbol not in recommendations["high_frequency"]:
                    recommendations["high_frequency"].append(symbol)
        
        # Recently accessed (within last 24 hours)
        cutoff = datetime.now() - timedelta(days=1)
        for key, data in self.metadata["frequently_accessed"].items():
            if data["last_access"]:
                last_access = datetime.fromisoformat(data["last_access"])
                if last_access > cutoff:
                    symbol = key.split("_")[0]
                    if symbol not in recommendations["recent"]:
                        recommendations["recent"].append(symbol)
        
        # Critical symbols (S&P 500 components, major indices)
        critical_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", 
                           "BRK.B", "JPM", "JNJ", "V", "PG", "UNH", "MA", "HD"]
        recommendations["critical"] = [s for s in critical_symbols 
                                      if s in self.metadata["frequently_accessed"]]
        
        # Batch candidates (symbols often accessed together)
        # This is a simplified version - could be enhanced with more sophisticated analysis
        recommendations["batch_candidates"] = list(set(
            recommendations["high_frequency"][:20] + 
            recommendations["recent"][:10]
        ))
        
        return recommendations
    
    def record_preload(self, symbols: List[str], duration: float, success_count: int):
        """
        Record preload operation for analysis.
        
        Parameters:
        -----------
        symbols : List[str]
            Symbols that were preloaded
        duration : float
            Time taken for preload (seconds)
        success_count : int
            Number of successfully preloaded entries
        """
        self.metadata["last_preload"] = {
            "timestamp": datetime.now().isoformat(),
            "symbol_count": len(symbols),
            "duration": duration,
            "success_count": success_count,
            "rate": success_count / duration if duration > 0 else 0
        }
        
        # Keep history of last 10 preloads
        self.metadata["preload_history"].append(self.metadata["last_preload"])
        if len(self.metadata["preload_history"]) > 10:
            self.metadata["preload_history"] = self.metadata["preload_history"][-10:]
        
        self._save_metadata()
    
    def get_optimal_batch_size(self) -> int:
        """
        Calculate optimal batch size based on historical performance.
        
        Returns:
        --------
        int
            Optimal batch size for preloading
        """
        if not self.metadata["preload_history"]:
            return 50  # Default
        
        # Analyze historical rates
        rates = [h["rate"] for h in self.metadata["preload_history"] if h.get("rate", 0) > 0]
        if not rates:
            return 50
        
        avg_rate = sum(rates) / len(rates)
        
        # Target 30 second preload time
        target_duration = 30
        optimal_size = int(avg_rate * target_duration)
        
        # Bound between 20 and 200
        return max(20, min(200, optimal_size))


class CacheOptimizer:
    """
    Optimize cache performance through various strategies.
    """
    
    def __init__(self, cache_manager):
        """
        Initialize cache optimizer.
        
        Parameters:
        -----------
        cache_manager : CacheManager
            Cache manager instance
        """
        self.cache_manager = cache_manager
        self.config = cache_manager.config
    
    def analyze_cache_performance(self) -> Dict[str, Any]:
        """
        Analyze cache performance and provide optimization recommendations.
        
        Returns:
        --------
        Dict[str, Any]
            Performance analysis and recommendations
        """
        stats = self.cache_manager.get_stats()
        
        analysis = {
            "performance": {
                "hit_rate": stats.get("hit_rate", 0),
                "total_requests": stats.get("hits", 0) + stats.get("misses", 0),
                "cache_size_mb": stats.get("cache_size_mb", 0),
                "errors": stats.get("errors", 0)
            },
            "health": self._calculate_health_score(stats),
            "recommendations": []
        }
        
        # Generate recommendations
        if analysis["performance"]["hit_rate"] < 0.5:
            analysis["recommendations"].append({
                "type": "low_hit_rate",
                "message": "Hit rate below 50%. Consider preloading frequently accessed symbols.",
                "action": "warm_cache"
            })
        
        if analysis["performance"]["cache_size_mb"] > self.config.max_cache_size_mb * 0.9:
            analysis["recommendations"].append({
                "type": "cache_full",
                "message": "Cache approaching size limit. Consider increasing limit or reducing TTL.",
                "action": "adjust_ttl"
            })
        
        if analysis["performance"]["errors"] > 10:
            analysis["recommendations"].append({
                "type": "high_errors",
                "message": f"High error count ({analysis['performance']['errors']}). Check cache integrity.",
                "action": "verify_cache"
            })
        
        return analysis
    
    def _calculate_health_score(self, stats: Dict[str, Any]) -> float:
        """
        Calculate overall cache health score (0-100).
        
        Parameters:
        -----------
        stats : Dict[str, Any]
            Cache statistics
            
        Returns:
        --------
        float
            Health score
        """
        score = 100.0
        
        # Deduct for low hit rate
        hit_rate = stats.get("hit_rate", 0)
        if hit_rate < 0.8:
            score -= (0.8 - hit_rate) * 50
        
        # Deduct for errors
        error_rate = stats.get("errors", 0) / max(1, stats.get("writes", 1))
        score -= error_rate * 20
        
        # Deduct for cache pressure
        cache_size = stats.get("cache_size_mb", 0)
        if cache_size > self.config.max_cache_size_mb * 0.8:
            pressure = (cache_size - self.config.max_cache_size_mb * 0.8) / (self.config.max_cache_size_mb * 0.2)
            score -= pressure * 10
        
        return max(0, min(100, score))
    
    def optimize_ttl_settings(self) -> Dict[str, int]:
        """
        Suggest optimized TTL settings based on usage patterns.
        
        Returns:
        --------
        Dict[str, int]
            Recommended TTL settings by statement type
        """
        # This is a simplified version - could analyze actual usage patterns
        recommendations = {
            "income_statement": 7 * 24 * 3600,      # 7 days
            "balance_sheet": 7 * 24 * 3600,         # 7 days  
            "cash_flow": 7 * 24 * 3600,             # 7 days
            "financial_ratios": 24 * 3600,          # 1 day
            "price_data": 3600                      # 1 hour
        }
        
        # Adjust based on cache pressure
        stats = self.cache_manager.get_stats()
        cache_size = stats.get("cache_size_mb", 0)
        
        if cache_size > self.config.max_cache_size_mb * 0.8:
            # Reduce TTLs by 50% if cache is under pressure
            for key in recommendations:
                recommendations[key] = recommendations[key] // 2
        
        return recommendations