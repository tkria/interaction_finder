"""
Tests for the PageFetcher high-level interface.

Tests cover:
- PageFetcher initialization and configuration
- get_chunks_with_embeddings functionality
- get_chunks text extraction
- Error handling and edge cases
- Integration with WebClient and URLCache
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from interaction_finder.fetcher.page_fetcher import PageFetcher
from interaction_finder.settings import IfetcherConfig


class TestPageFetcherInitialization:
    """Test PageFetcher initialization and configuration."""

    def test_pagefetcher_creation_basic(self, tmp_path):
        """Test basic PageFetcher creation."""
        cache_dir = tmp_path / "test_cache"
        fetcher = PageFetcher(cache_dir=cache_dir)

        assert fetcher.cache_dir == cache_dir
        assert fetcher.timeout == 30  # Default
        assert fetcher.verbose is False  # Default
        assert fetcher.web_client is not None
        assert fetcher.cache is not None

    def test_pagefetcher_creation_with_options(self, tmp_path):
        """Test PageFetcher creation with custom options."""
        cache_dir = tmp_path / "custom_cache"
        fetcher = PageFetcher(
            cache_dir=cache_dir, timeout=60, show_status=False, verbose=True
        )

        assert fetcher.cache_dir == cache_dir
        assert fetcher.timeout == 60
        assert fetcher.verbose is True


class TestGetChunksWithEmbeddings:
    """Test get_chunks_with_embeddings functionality."""

    @pytest.fixture
    def fetcher(self, tmp_path):
        cache_dir = tmp_path / "cache"
        return PageFetcher(cache_dir=cache_dir, show_status=False)

    @pytest.mark.asyncio
    async def test_get_chunks_with_embeddings_cache_hit(self, fetcher):
        """Test getting chunks when data is in cache."""
        url = "http://example.com/test"
        cached_data = {
            "chunks": [
                {"text": "chunk1", "wordcount": 5, "embedding": [1.0, 2.0, 3.0]},
                {"text": "chunk2", "wordcount": 7, "embedding": [4.0, 5.0, 6.0]},
            ]
        }

        # Mock cache to return data
        fetcher.cache.has_path = AsyncMock(return_value=True)
        fetcher.cache.get_content = AsyncMock(return_value=cached_data)

        result = await fetcher.get_chunks_with_embeddings(url)

        assert len(result) == 2
        assert result[0]["text"] == "chunk1"
        assert result[1]["text"] == "chunk2"
        fetcher.cache.has_path.assert_called_once_with(url, "chunks")
        fetcher.cache.get_content.assert_called_once_with(url, "chunks")

    @pytest.mark.asyncio
    async def test_get_chunks_with_embeddings_cache_miss(self, fetcher):
        """Test getting chunks when not in cache (requires web fetching)."""
        url = "http://example.com/test"

        # Mock cache miss
        fetcher.cache.has_path = AsyncMock(return_value=False)

        # Mock the batch operations method that's actually called
        mock_chunks = [
            {"text": "chunk1", "wordcount": 5, "embedding": [1.0, 2.0, 3.0]},
        ]
        fetcher.batch_ops._fetch_chunks_and_cache = AsyncMock(return_value=mock_chunks)

        result = await fetcher.get_chunks_with_embeddings(url)

        # Check result
        assert len(result) == 1
        assert result[0]["text"] == "chunk1"

        # Check that batch_ops method was called with correct parameters
        fetcher.batch_ops._fetch_chunks_and_cache.assert_called_once_with(
            url, retry=False
        )

    @pytest.mark.asyncio
    async def test_get_chunks_with_embeddings_retry(self, fetcher):
        """Test retry functionality."""
        url = "http://example.com/test"

        fetcher.cache.has_path = AsyncMock(return_value=False)
        fetcher.batch_ops._fetch_chunks_and_cache = AsyncMock(return_value=[])

        await fetcher.get_chunks_with_embeddings(url, retry=True)

        # Check that batch_ops method was called with retry=True
        fetcher.batch_ops._fetch_chunks_and_cache.assert_called_once_with(
            url, retry=True
        )


class TestGetChunks:
    """Test get_chunks text extraction functionality."""

    @pytest.fixture
    def fetcher(self, tmp_path):
        cache_dir = tmp_path / "cache"
        return PageFetcher(cache_dir=cache_dir, show_status=False)

    @pytest.mark.asyncio
    async def test_get_chunks_extracts_text(self, fetcher):
        """Test that get_chunks extracts text from chunks with embeddings."""
        chunks_with_embeddings = [
            {"text": "First chunk text", "wordcount": 3, "embedding": [1.0, 2.0]},
            {"text": "Second chunk text", "wordcount": 3, "embedding": [3.0, 4.0]},
        ]

        fetcher.get_chunks_with_embeddings = AsyncMock(
            return_value=chunks_with_embeddings
        )

        result = await fetcher.get_chunks("http://example.com/test")

        assert result == ["First chunk text", "Second chunk text"]
        fetcher.get_chunks_with_embeddings.assert_called_once_with(
            "http://example.com/test", retry=False
        )

    @pytest.mark.asyncio
    async def test_get_chunks_empty_result(self, fetcher):
        """Test get_chunks with empty chunks."""
        fetcher.get_chunks_with_embeddings = AsyncMock(return_value=[])

        result = await fetcher.get_chunks("http://example.com/test")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_chunks_with_retry(self, fetcher):
        """Test get_chunks retry parameter."""
        fetcher.get_chunks_with_embeddings = AsyncMock(return_value=[])

        await fetcher.get_chunks("http://example.com/test", retry=True)

        fetcher.get_chunks_with_embeddings.assert_called_once_with(
            "http://example.com/test", retry=True
        )


class TestErrorHandling:
    """Test error handling in PageFetcher."""

    @pytest.fixture
    def fetcher(self, tmp_path):
        cache_dir = tmp_path / "cache"
        return PageFetcher(cache_dir=cache_dir, show_status=False)

    @pytest.mark.asyncio
    async def test_get_chunks_web_client_error(self, fetcher):
        """Test error handling when batch operations fail."""
        fetcher.cache.has_path = AsyncMock(return_value=False)
        fetcher.batch_ops._fetch_chunks_and_cache = AsyncMock(
            side_effect=RuntimeError("Network error")
        )

        with pytest.raises(RuntimeError, match="Network error"):
            await fetcher.get_chunks_with_embeddings("http://example.com/test")

    @pytest.mark.asyncio
    async def test_get_chunks_cache_error(self, fetcher):
        """Test handling of cache errors."""
        fetcher.cache.has_path = AsyncMock(side_effect=RuntimeError("Cache error"))

        with pytest.raises(RuntimeError, match="Cache error"):
            await fetcher.get_chunks_with_embeddings("http://example.com/test")


class TestIntegration:
    """Test integration between PageFetcher components."""

    def test_pagefetcher_creates_web_client(self, tmp_path):
        """Test that PageFetcher creates WebClient with correct parameters."""
        with patch(
            "interaction_finder.fetcher.page_fetcher.WebClient"
        ) as mock_web_client_class:
            mock_web_client_instance = Mock()
            mock_web_client_class.return_value = mock_web_client_instance

            cache_dir = tmp_path / "cache"
            fetcher = PageFetcher(cache_dir=cache_dir, timeout=45, show_status=False)

            # Should create WebClient with timeout and verbose parameters
            mock_web_client_class.assert_called_once_with(45, False)
            assert fetcher.web_client == mock_web_client_instance

    def test_pagefetcher_creates_cache(self, tmp_path):
        """Test that PageFetcher creates URLCache."""
        with patch(
            "interaction_finder.fetcher.page_fetcher.URLCache"
        ) as mock_cache_class:
            mock_cache_instance = Mock()
            mock_cache_class.return_value = mock_cache_instance

            cache_dir = tmp_path / "cache"
            fetcher = PageFetcher(cache_dir=cache_dir, show_status=False)

            # Should create URLCache with cache_dir
            mock_cache_class.assert_called_once_with(cache_dir)
            assert fetcher.cache == mock_cache_instance
