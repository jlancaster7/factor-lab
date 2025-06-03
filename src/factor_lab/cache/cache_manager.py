"""
Cache manager for FMP data provider.
"""

import json
import gzip
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import threading

from .cache_key import CacheKey
from .cache_config import CacheConfig

logger = logging.getLogger(__name__)


class CacheStatistics:
    """Track cache performance statistics."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.evictions = 0
        self.errors = 0
        self._lock = threading.Lock()
    
    def record_hit(self):
        with self._lock:
            self.hits += 1
    
    def record_miss(self):
        with self._lock:
            self.misses += 1
    
    def record_write(self):
        with self._lock:
            self.writes += 1
    
    def record_eviction(self):
        with self._lock:
            self.evictions += 1
    
    def record_error(self):
        with self._lock:
            self.errors += 1
    
    def get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get all statistics."""
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "writes": self.writes,
                "evictions": self.evictions,
                "errors": self.errors,
                "hit_rate": self.get_hit_rate()
            }


class CacheManager:
    """
    Manages caching for FMP data with intelligent expiration and versioning.
    
    Features:
    - Statement-level caching with acceptedDate awareness
    - Automatic cache expiration based on data type
    - Version-based cache invalidation
    - Size-based cache cleanup
    - Asynchronous write operations
    - Compression support
    - Performance statistics
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize cache manager with configuration."""
        self.config = config or CacheConfig()
        self.stats = CacheStatistics()
        
        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=4) if self.config.async_writes else None
        
        # Cache metadata tracking
        self._metadata_file = self.config.cache_dir / "metadata" / "cache_metadata.json"
        self._metadata = self._load_metadata()
        
        logger.info(f"Initialized CacheManager with cache dir: {self.config.cache_dir}")
    
    def get(self, key: CacheKey) -> Optional[Any]:
        """
        Retrieve data from cache if available and not expired.
        
        Args:
            key: Cache key for the data
            
        Returns:
            Cached data if available and valid, None otherwise
        """
        try:
            file_path = self._get_file_path(key)
            
            if not file_path.exists():
                self.stats.record_miss()
                if self.config.log_cache_misses:
                    logger.debug(f"Cache miss: {key.to_string()}")
                return None
            
            # Check if cache is expired
            if self._is_expired(key, file_path):
                self.stats.record_miss()
                if self.config.log_cache_misses:
                    logger.debug(f"Cache expired: {key.to_string()}")
                return None
            
            # Read cached data
            data = self._read_cache_file(file_path)
            
            if data is not None:
                self.stats.record_hit()
                if self.config.log_cache_hits:
                    logger.debug(f"Cache hit: {key.to_string()}")
                
                # Validate version
                cached_version = data.get("_cache_version", "0.0")
                if cached_version != key.version:
                    logger.debug(f"Cache version mismatch: {cached_version} != {key.version}")
                    return None
                
                # Return the actual data
                return data.get("data")
            
            return None
            
        except Exception as e:
            self.stats.record_error()
            logger.error(f"Error reading cache for {key.to_string()}: {e}")
            return None
    
    def set(self, key: CacheKey, data: Any, ttl_override: Optional[int] = None):
        """
        Store data in cache with metadata.
        
        Args:
            key: Cache key for the data
            data: Data to cache
            ttl_override: Optional TTL override in seconds
        """
        try:
            # Prepare cache entry with metadata
            cache_entry = {
                "data": data,
                "_cache_version": key.version,
                "_cached_at": time.time(),
                "_cache_key": key.to_string(),
                "_accepted_date": key.accepted_date,
                "_ttl": ttl_override or self._get_ttl_for_type(key.statement_type)
            }
            
            if self.config.async_writes and self._executor:
                # Write asynchronously
                self._executor.submit(self._write_cache_entry, key, cache_entry)
            else:
                # Write synchronously
                self._write_cache_entry(key, cache_entry)
                
        except Exception as e:
            self.stats.record_error()
            logger.error(f"Error writing cache for {key.to_string()}: {e}")
    
    def _write_cache_entry(self, key: CacheKey, cache_entry: Dict):
        """Write cache entry to disk."""
        try:
            file_path = self._get_file_path(key)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write data (returns actual path used)
            actual_path = self._write_cache_file(file_path, cache_entry)
            
            self.stats.record_write()
            
            # Update metadata with actual path
            self._update_metadata(key, actual_path)
            
            # Check if cleanup is needed
            if self._should_cleanup():
                self._cleanup_cache()
                
        except Exception as e:
            self.stats.record_error()
            logger.error(f"Error in _write_cache_entry: {e}")
    
    def invalidate(self, key: CacheKey):
        """Invalidate a specific cache entry."""
        try:
            file_path = self._get_file_path(key)
            if file_path.exists():
                file_path.unlink()
                self.stats.record_eviction()
                logger.debug(f"Invalidated cache: {key.to_string()}")
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
    
    def invalidate_symbol(self, symbol: str, statement_type: Optional[str] = None):
        """Invalidate all cache entries for a symbol."""
        count = 0
        try:
            if statement_type:
                # Invalidate specific statement type
                subdir = self.config.cache_dir / self._get_subdir_for_type(statement_type)
                pattern = f"{symbol.upper()}_*"
            else:
                # Invalidate all data for symbol
                pattern = f"{symbol.upper()}_*"
                subdir = self.config.cache_dir
            
            for file_path in subdir.rglob(pattern):
                if file_path.is_file():
                    file_path.unlink()
                    count += 1
            
            if count > 0:
                self.stats.evictions += count
                logger.info(f"Invalidated {count} cache entries for {symbol}")
                
        except Exception as e:
            logger.error(f"Error invalidating symbol cache: {e}")
    
    def get_batch(self, keys: List[CacheKey]) -> Dict[str, Any]:
        """
        Retrieve multiple cache entries in batch.
        
        Returns:
            Dictionary mapping cache key strings to data
        """
        results = {}
        for key in keys:
            data = self.get(key)
            if data is not None:
                results[key.to_string()] = data
        return results
    
    def set_batch(self, entries: Dict[CacheKey, Any], ttl_override: Optional[int] = None):
        """Store multiple cache entries in batch."""
        for key, data in entries.items():
            self.set(key, data, ttl_override)
    
    def clear_all(self):
        """Clear entire cache."""
        try:
            shutil.rmtree(self.config.cache_dir)
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Recreate subdirectories
            for subdir in ["income_statements", "balance_sheets", "cash_flows", 
                          "financial_ratios", "price_data", "metadata"]:
                (self.config.cache_dir / subdir).mkdir(exist_ok=True)
            
            # Reset metadata
            self._metadata = {}
            self._save_metadata()
            
            logger.info("Cleared all cache")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.stats.get_stats()
        
        # Add size information
        cache_size_mb = self._get_cache_size_mb()
        stats["cache_size_mb"] = cache_size_mb
        stats["cache_dir"] = str(self.config.cache_dir)
        
        return stats
    
    def _get_file_path(self, key: CacheKey) -> Path:
        """Get the file path for a cache key."""
        subdir = self.config.cache_dir / key.get_subdir()
        filename = key.to_filename()
        
        if not self.config.enable_compression:
            filename = filename.replace(".json.gz", ".json")
        
        return subdir / filename
    
    def _read_cache_file(self, file_path: Path) -> Optional[Dict]:
        """Read a cache file (with compression support)."""
        try:
            if file_path.suffix == ".gz":
                with gzip.open(file_path, "rt", encoding="utf-8") as f:
                    return json.load(f)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error reading cache file {file_path}: {e}")
            return None
    
    def _write_cache_file(self, file_path: Path, data: Dict) -> Path:
        """Write a cache file (with compression support)."""
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.config.enable_compression and file_path.suffix != ".gz":
            # Create the actual file path with .gz extension
            actual_path = file_path.parent / (file_path.stem + ".json.gz")
        else:
            actual_path = file_path
        
        if actual_path.suffix == ".gz":
            with gzip.open(actual_path, "wt", encoding="utf-8") as f:
                json.dump(data, f, separators=(",", ":"))
        else:
            with open(actual_path, "w", encoding="utf-8") as f:
                json.dump(data, f, separators=(",", ":"))
        
        return actual_path
    
    def _is_expired(self, key: CacheKey, file_path: Path) -> bool:
        """Check if a cache entry is expired."""
        try:
            # Read just the metadata
            data = self._read_cache_file(file_path)
            if not data:
                return True
            
            cached_at = data.get("_cached_at", 0)
            ttl = data.get("_ttl", self._get_ttl_for_type(key.statement_type))
            
            age = time.time() - cached_at
            return age > ttl
            
        except Exception:
            return True
    
    def _get_ttl_for_type(self, statement_type: str) -> int:
        """Get TTL for a specific statement type."""
        ttl_map = {
            "income_statement": self.config.income_statement_ttl,
            "balance_sheet": self.config.balance_sheet_ttl,
            "cash_flow": self.config.cash_flow_ttl,
            "financial_ratios": self.config.financial_ratios_ttl,
            "price": self.config.price_data_ttl
        }
        return ttl_map.get(statement_type, self.config.default_ttl)
    
    def _get_subdir_for_type(self, statement_type: str) -> str:
        """Get subdirectory for a statement type."""
        subdir_map = {
            "income_statement": "income_statements",
            "balance_sheet": "balance_sheets",
            "cash_flow": "cash_flows",
            "financial_ratios": "financial_ratios",
            "price": "price_data"
        }
        return subdir_map.get(statement_type, "metadata")
    
    def _get_cache_size_mb(self) -> float:
        """Calculate total cache size in MB."""
        total_size = 0
        for file_path in self.config.cache_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size / (1024 * 1024)
    
    def _should_cleanup(self) -> bool:
        """Check if cache cleanup is needed."""
        cache_size_mb = self._get_cache_size_mb()
        return cache_size_mb > self.config.cleanup_threshold_mb
    
    def _cleanup_cache(self):
        """Clean up old cache entries to free space."""
        logger.info("Starting cache cleanup...")
        
        try:
            # Get all cache files with their modification times
            cache_files = []
            for pattern in ["*.json", "*.json.gz"]:
                for file_path in self.config.cache_dir.rglob(pattern):
                    if file_path.is_file() and file_path.parent.name != "metadata":
                        try:
                            stat = file_path.stat()
                            cache_files.append((file_path, stat.st_mtime, stat.st_size))
                        except FileNotFoundError:
                            # File was removed by another process
                            continue
            
            # Sort by modification time (oldest first)
            cache_files.sort(key=lambda x: x[1])
            
            # Calculate how much to remove
            current_size = sum(size for _, _, size in cache_files)
            target_size = self.config.cleanup_threshold_mb * 0.8 * 1024 * 1024  # 80% of threshold
            to_remove = current_size - target_size
            
            removed_size = 0
            removed_count = 0
            
            for file_path, _, size in cache_files:
                if removed_size >= to_remove:
                    break
                
                try:
                    if file_path.exists():  # Double check file still exists
                        file_path.unlink()
                        removed_size += size
                        removed_count += 1
                        self.stats.record_eviction()
                except FileNotFoundError:
                    # File already removed
                    pass
                except Exception as e:
                    logger.debug(f"Error removing cache file {file_path}: {e}")
            
            logger.info(f"Cache cleanup complete: removed {removed_count} files, freed {removed_size / (1024 * 1024):.1f} MB")
            
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        try:
            if self._metadata_file.exists():
                with open(self._metadata_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading cache metadata: {e}")
        
        return {}
    
    def _save_metadata(self):
        """Save cache metadata."""
        try:
            self._metadata_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._metadata_file, "w") as f:
                json.dump(self._metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache metadata: {e}")
    
    def _update_metadata(self, key: CacheKey, file_path: Path):
        """Update metadata for a cache entry."""
        self._metadata[key.to_string()] = {
            "file_path": str(file_path),
            "cached_at": time.time(),
            "symbol": key.symbol,
            "statement_type": key.statement_type,
            "period": key.period
        }
        
        # Periodically save metadata
        if len(self._metadata) % 100 == 0:
            self._save_metadata()
    
    def shutdown(self):
        """Shutdown cache manager and cleanup resources."""
        try:
            # Save metadata
            self._save_metadata()
            
            # Shutdown thread pool
            if self._executor:
                self._executor.shutdown(wait=True)
            
            logger.info("Cache manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during cache manager shutdown: {e}")