"""Tests for search result caching."""

import pytest
import tempfile
import asyncio
from pathlib import Path
from datetime import datetime, timedelta

from interaction_finder.search.cache import SearchCache
from interaction_finder.search.base import SearchQuery, SearchResult, SearchResults


@pytest.fixture
def temp_cache_dir():
    """Create temporary directory for cache testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def cache(temp_cache_dir):
    """Create SearchCache instance for testing."""
    return SearchCache(temp_cache_dir, ttl_hours=1)


@pytest.fixture
def sample_query():
    """Sample search query for testing."""
    return SearchQuery(query="BRCA1 mutations", max_results=20)


@pytest.fixture
def sample_results():
    """Sample search results for testing."""
    query = SearchQuery(query="BRCA1 mutations")
    return SearchResults(
        query=query,
        backend="pubmed",
        results=[
            SearchResult(
                title="Paper 1", url="https://example.com/1", backend="pubmed"
            ),
            SearchResult(
                title="Paper 2", url="https://example.com/2", backend="pubmed"
            ),
        ],
        total_found=100,
        search_time=1.5,
    )


class TestSearchCache:
    """Test SearchCache functionality."""

    @pytest.mark.asyncio
    async def test_cache_key_generation(self, cache, sample_query):
        """Test cache key generation is consistent."""
        key1 = cache._get_cache_key(sample_query, "pubmed")
        key2 = cache._get_cache_key(sample_query, "pubmed")
        assert key1 == key2

        # Different backend should give different key
        key3 = cache._get_cache_key(sample_query, "perplexica")
        assert key1 != key3

        # Different query should give different key
        different_query = SearchQuery(query="different query")
        key4 = cache._get_cache_key(different_query, "pubmed")
        assert key1 != key4

    @pytest.mark.asyncio
    async def test_cache_miss(self, cache, sample_query):
        """Test cache miss returns None."""
        result = await cache.get(sample_query, "pubmed")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_set_and_get(self, cache, sample_query, sample_results):
        """Test setting and getting cached results."""
        # Cache miss initially
        result = await cache.get(sample_query, "pubmed")
        assert result is None

        # Set cache
        await cache.set(sample_query, "pubmed", sample_results)

        # Cache hit
        result = await cache.get(sample_query, "pubmed")
        assert result is not None
        assert result.query == sample_results.query
        assert result.backend == sample_results.backend
        assert len(result.results) == len(sample_results.results)
        assert result.total_found == sample_results.total_found

    @pytest.mark.asyncio
    async def test_cache_with_context(self, cache, sample_query, sample_results):
        """Test caching with additional context."""
        context = {"expansion_enabled": True, "max_terms": 10}

        # Set with context
        await cache.set(sample_query, "pubmed", sample_results, context)

        # Get with same context
        result = await cache.get(sample_query, "pubmed", context)
        assert result is not None

        # Get with different context should miss
        different_context = {"expansion_enabled": False}
        result = await cache.get(sample_query, "pubmed", different_context)
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_expiration(self, temp_cache_dir):
        """Test cache expiration with short TTL."""
        # Create cache with very short TTL
        short_ttl_cache = SearchCache(temp_cache_dir, ttl_hours=0.001)  # ~3.6 seconds

        query = SearchQuery(query="test query")
        results = SearchResults(
            query=query,
            backend="pubmed",
            results=[],
            total_found=0,
        )

        # Set cache
        await short_ttl_cache.set(query, "pubmed", results)

        # Should hit immediately
        result = await short_ttl_cache.get(query, "pubmed")
        assert result is not None

        # Wait for expiration (add some buffer time)
        await asyncio.sleep(5)

        # Should miss after expiration
        result = await short_ttl_cache.get(query, "pubmed")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_stats(self, cache, sample_query, sample_results):
        """Test cache statistics."""
        # Empty cache stats
        stats = await cache.get_stats()
        assert stats["total_entries"] == 0
        assert stats["valid_entries"] == 0
        assert stats["expired_entries"] == 0
        assert stats["cache_size_bytes"] == 0

        # Add some entries
        await cache.set(sample_query, "pubmed", sample_results)

        different_query = SearchQuery(query="different query")
        await cache.set(different_query, "pubmed", sample_results)

        stats = await cache.get_stats()
        assert stats["total_entries"] == 2
        assert stats["valid_entries"] == 2
        assert stats["expired_entries"] == 0
        assert stats["cache_size_bytes"] > 0

    @pytest.mark.asyncio
    async def test_clear_all(self, cache, sample_query, sample_results):
        """Test clearing all cache entries."""
        # Add entry
        await cache.set(sample_query, "pubmed", sample_results)

        # Verify it exists
        result = await cache.get(sample_query, "pubmed")
        assert result is not None

        # Clear all
        removed = await cache.clear_all()
        assert removed == 1

        # Verify it's gone
        result = await cache.get(sample_query, "pubmed")
        assert result is None

    @pytest.mark.asyncio
    async def test_clear_expired(self, temp_cache_dir):
        """Test clearing only expired entries."""
        # Create cache with mixed TTLs by manipulating cache files
        cache = SearchCache(temp_cache_dir, ttl_hours=1)

        # Add some entries
        query1 = SearchQuery(query="query 1")
        query2 = SearchQuery(query="query 2")
        test_query = SearchQuery(query="test")
        results = SearchResults(
            query=test_query, backend="pubmed", results=[], total_found=0
        )

        await cache.set(query1, "pubmed", results)
        await cache.set(query2, "pubmed", results)

        # Initially no expired entries
        removed = await cache.clear_expired()
        assert removed == 0

        # Mock expiration by creating cache with very short TTL
        short_cache = SearchCache(temp_cache_dir, ttl_hours=0.001)
        await asyncio.sleep(5)  # Wait for expiration

        removed = await short_cache.clear_expired()
        assert removed == 2

    @pytest.mark.asyncio
    async def test_corrupted_cache_handling(self, temp_cache_dir):
        """Test handling of corrupted cache files."""
        cache = SearchCache(temp_cache_dir)

        # Create a corrupted cache file manually
        corrupt_file = temp_cache_dir / "search_corrupted.json"
        corrupt_file.write_text("invalid json {")

        # Should handle gracefully
        stats = await cache.get_stats()
        assert stats["expired_entries"] >= 1  # Corrupted files counted as expired

        # Clear expired should remove corrupted files
        removed = await cache.clear_expired()
        assert removed >= 1
        assert not corrupt_file.exists()

    @pytest.mark.asyncio
    async def test_concurrent_cache_access(self, cache, sample_results):
        """Test concurrent cache access."""
        queries = [SearchQuery(query=f"query {i}") for i in range(10)]

        # Concurrent sets
        await asyncio.gather(
            *[cache.set(query, "pubmed", sample_results) for query in queries]
        )

        # Concurrent gets
        results = await asyncio.gather(
            *[cache.get(query, "pubmed") for query in queries]
        )

        # All should succeed
        assert all(result is not None for result in results)
        assert len(results) == 10
