"""Search result caching functionality."""

import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import aiofiles
import aiofiles.os

from .base import SearchQuery, SearchResults


class SearchCache:
    """File-based cache for search results with TTL support."""

    def __init__(self, cache_dir: Path, ttl_hours: int = 24):
        """
        Initialize search cache.

        Args:
            cache_dir: Directory to store cache files
            ttl_hours: Time-to-live for cached results in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)

    def _get_cache_key(
        self, query: SearchQuery, backend: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate cache key for search query."""
        # Create a deterministic key from query parameters
        key_data = {
            "query": query.query,
            "max_results": query.max_results,
            "expanded_terms": sorted(query.expanded_terms)
            if query.expanded_terms
            else [],
            "backend": backend,
            "context": context or {},
        }

        # Create hash from serialized data
        serialized = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[
            :16
        ]  # 16 chars should be enough

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cache entry."""
        return self.cache_dir / f"search_{cache_key}.json"

    async def get(
        self, query: SearchQuery, backend: str, context: Optional[Dict[str, Any]] = None
    ) -> Optional[SearchResults]:
        """
        Get cached search results if available and not expired.

        Args:
            query: Search query
            backend: Search backend name
            context: Additional context for caching

        Returns:
            Cached SearchResults if available and fresh, None otherwise
        """
        cache_key = self._get_cache_key(query, backend, context)
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        try:
            async with aiofiles.open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.loads(await f.read())

            # Check if cache entry is expired
            cached_at = datetime.fromisoformat(cache_data["cached_at"])
            if datetime.now() - cached_at > self.ttl:
                # Remove expired cache entry
                await aiofiles.os.remove(cache_path)
                return None

            # Deserialize search results
            results_data = cache_data["results"]
            return SearchResults.model_validate(results_data)

        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            # If cache is corrupted, remove it
            try:
                await aiofiles.os.remove(cache_path)
            except OSError:
                pass
            return None

    async def set(
        self,
        query: SearchQuery,
        backend: str,
        results: SearchResults,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store search results in cache.

        Args:
            query: Search query
            backend: Search backend name
            results: Search results to cache
            context: Additional context for caching
        """
        cache_key = self._get_cache_key(query, backend, context)
        cache_path = self._get_cache_path(cache_key)

        cache_data = {
            "cached_at": datetime.now().isoformat(),
            "query": query.model_dump(),
            "backend": backend,
            "context": context or {},
            "results": results.model_dump(
                mode="json"
            ),  # Use JSON mode to handle datetime
        }

        # Write to temporary file first for atomic operation
        temp_path = cache_path.with_suffix(".tmp")
        try:
            async with aiofiles.open(temp_path, "w", encoding="utf-8") as f:
                await f.write(
                    json.dumps(cache_data, indent=2, ensure_ascii=False, default=str)
                )
            await aiofiles.os.rename(temp_path, cache_path)
        except Exception:
            # Clean up temp file if write failed
            try:
                await aiofiles.os.remove(temp_path)
            except OSError:
                pass
            raise

    async def clear_expired(self) -> int:
        """
        Clear expired cache entries.

        Returns:
            Number of entries removed
        """
        removed_count = 0

        if not self.cache_dir.exists():
            return removed_count

        for cache_file in self.cache_dir.glob("search_*.json"):
            try:
                async with aiofiles.open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.loads(await f.read())

                cached_at = datetime.fromisoformat(cache_data["cached_at"])
                if datetime.now() - cached_at > self.ttl:
                    await aiofiles.os.remove(cache_file)
                    removed_count += 1

            except (json.JSONDecodeError, KeyError, ValueError, OSError):
                # Remove corrupted cache files
                try:
                    await aiofiles.os.remove(cache_file)
                    removed_count += 1
                except OSError:
                    pass

        return removed_count

    async def clear_all(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries removed
        """
        removed_count = 0

        if not self.cache_dir.exists():
            return removed_count

        for cache_file in self.cache_dir.glob("search_*.json"):
            try:
                await aiofiles.os.remove(cache_file)
                removed_count += 1
            except OSError:
                pass

        return removed_count

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "total_entries": 0,
            "expired_entries": 0,
            "valid_entries": 0,
            "cache_size_bytes": 0,
        }

        if not self.cache_dir.exists():
            return stats

        for cache_file in self.cache_dir.glob("search_*.json"):
            try:
                stats["cache_size_bytes"] += cache_file.stat().st_size
                stats["total_entries"] += 1

                async with aiofiles.open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.loads(await f.read())

                cached_at = datetime.fromisoformat(cache_data["cached_at"])
                if datetime.now() - cached_at > self.ttl:
                    stats["expired_entries"] += 1
                else:
                    stats["valid_entries"] += 1

            except (json.JSONDecodeError, KeyError, ValueError, OSError):
                # Count corrupted files as expired
                stats["expired_entries"] += 1

        return stats
