"""
Tests for batch operations for concurrent URL fetching.

Tests cover:
- BatchOperations class initialization and configuration
- fetch_multiple functionality for different content types
- Concurrent fetching with semaphore limiting
- Smart caching with cache hit/miss scenarios
- Progress display integration
- Error handling (fail_fast and graceful error handling)
- Prefetch operations
- Domain width calculation and formatting
- Batch utility functions
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from interaction_finder.fetcher.batch_operations import (
    BatchOperations,
    fetch_urls_with_progress,
    fetch_urls_concurrent_with_progress,
    DEFAULT_MAX_CONCURRENT,
)

# Configure async tests for anyio with asyncio backend only
pytest_plugins = ["anyio"]


@pytest.fixture(params=["asyncio"], scope="session")
def anyio_backend(request):
    return request.param


class TestBatchOperationsInitialization:
    """Test BatchOperations initialization and configuration."""

    def test_batch_operations_creation(self):
        """Test basic BatchOperations creation."""
        mock_cache = Mock()
        mock_web_client = Mock()

        batch_ops = BatchOperations(mock_cache, mock_web_client, show_status=True)

        assert batch_ops.cache == mock_cache
        assert batch_ops.web_client == mock_web_client
        assert batch_ops.progress_display is not None
        assert batch_ops._domain_width == 15  # Default

    def test_batch_operations_no_status(self):
        """Test BatchOperations creation without status display."""
        mock_cache = Mock()
        mock_web_client = Mock()

        batch_ops = BatchOperations(mock_cache, mock_web_client, show_status=False)

        assert batch_ops.progress_display is not None


class TestFetcherMapping:
    """Test fetcher function mapping for different content types."""

    def test_get_fetcher_for_content_type_html(self):
        """Test HTML content type mapping."""
        batch_ops = BatchOperations(Mock(), Mock())

        fetcher = batch_ops._get_fetcher_for_content_type("html")

        assert fetcher == batch_ops._fetch_html_and_cache

    def test_get_fetcher_for_content_type_pdf(self):
        """Test PDF content type mapping."""
        batch_ops = BatchOperations(Mock(), Mock())

        fetcher = batch_ops._get_fetcher_for_content_type("pdf")

        assert fetcher == batch_ops._fetch_pdf_and_cache

    def test_get_fetcher_for_content_type_markdown(self):
        """Test Markdown content type mapping."""
        batch_ops = BatchOperations(Mock(), Mock())

        fetcher = batch_ops._get_fetcher_for_content_type("markdown")

        assert fetcher == batch_ops._fetch_markdown_and_cache

    def test_get_fetcher_for_content_type_chunks(self):
        """Test chunks content type mapping."""
        batch_ops = BatchOperations(Mock(), Mock())

        fetcher = batch_ops._get_fetcher_for_content_type("chunks")

        assert fetcher == batch_ops._fetch_chunks_and_cache

    def test_get_fetcher_for_content_type_invalid(self):
        """Test invalid content type raises ValueError."""
        batch_ops = BatchOperations(Mock(), Mock())

        with pytest.raises(ValueError, match="Unknown content type: invalid"):
            batch_ops._get_fetcher_for_content_type("invalid")


class TestFetchMultiple:
    """Test fetch_multiple functionality."""

    @pytest.fixture
    def batch_ops(self):
        mock_cache = Mock()
        mock_web_client = Mock()
        return BatchOperations(mock_cache, mock_web_client, show_status=False)

    @pytest.mark.anyio
    async def test_fetch_multiple_empty_urls(self, batch_ops):
        """Test fetch_multiple with empty URL list."""
        result = await batch_ops.fetch_multiple([], "html")

        assert result == []

    @pytest.mark.anyio
    async def test_fetch_multiple_all_cached(self, batch_ops):
        """Test fetch_multiple when all URLs are cached."""
        urls = ["http://example1.com", "http://example2.com"]
        cached_content = ["cached1", "cached2"]

        # Mock cache to return all URLs as cached
        batch_ops.cache.get_content_batch = AsyncMock(
            return_value=[
                ("http://example1.com", "cached1"),
                ("http://example2.com", "cached2"),
            ]
        )

        result = await batch_ops.fetch_multiple(urls, "html")

        assert result == cached_content
        batch_ops.cache.get_content_batch.assert_called_once_with(urls, "html")

    @pytest.mark.anyio
    async def test_fetch_multiple_none_cached(self, batch_ops):
        """Test fetch_multiple when no URLs are cached."""
        urls = ["http://example1.com", "http://example2.com"]

        # Mock cache to return no cached content
        batch_ops.cache.get_content_batch = AsyncMock(
            return_value=[
                ("http://example1.com", None),
                ("http://example2.com", None),
            ]
        )

        # Mock fetcher functions
        batch_ops._fetch_html_and_cache = AsyncMock(
            side_effect=["fetched1", "fetched2"]
        )

        # Mock progress display
        mock_progress = Mock()
        mock_progress.__enter__ = Mock(return_value=mock_progress)
        mock_progress.__exit__ = Mock(return_value=None)
        mock_progress.update = Mock()
        batch_ops.progress_display.batch_progress = Mock(return_value=mock_progress)

        result = await batch_ops.fetch_multiple(urls, "html")

        # Results may come back in any order due to concurrency
        assert set(result) == {"fetched1", "fetched2"}
        assert len(result) == 2

    @pytest.mark.anyio
    async def test_fetch_multiple_mixed_cache_status(self, batch_ops):
        """Test fetch_multiple with mixed cache status."""
        urls = ["http://example1.com", "http://example2.com", "http://example3.com"]

        # Mock cache: first URL cached, others not
        batch_ops.cache.get_content_batch = AsyncMock(
            return_value=[
                ("http://example1.com", "cached1"),
                ("http://example2.com", None),
                ("http://example3.com", None),
            ]
        )

        # Mock fetcher functions for uncached URLs
        batch_ops._fetch_html_and_cache = AsyncMock(
            side_effect=["fetched2", "fetched3"]
        )

        # Mock progress display
        mock_progress = Mock()
        mock_progress.__enter__ = Mock(return_value=mock_progress)
        mock_progress.__exit__ = Mock(return_value=None)
        mock_progress.update = Mock()
        batch_ops.progress_display.batch_progress = Mock(return_value=mock_progress)

        result = await batch_ops.fetch_multiple(urls, "html")

        # Should return in original order: cached, then fetched
        # The specific order of fetched items may vary due to async execution
        assert len(result) == 3
        assert result[0] == "cached1"  # First item is cached
        assert set(result[1:]) == {
            "fetched2",
            "fetched3",
        }  # Fetched items are both present

    @pytest.mark.anyio
    async def test_fetch_multiple_with_options(self, batch_ops):
        """Test fetch_multiple with custom options."""
        urls = ["http://example.com"]

        batch_ops.cache.get_content_batch = AsyncMock(
            return_value=[("http://example.com", None)]
        )
        batch_ops._fetch_html_and_cache = AsyncMock(return_value="fetched")

        # Mock progress display
        mock_progress = Mock()
        mock_progress.__enter__ = Mock(return_value=mock_progress)
        mock_progress.__exit__ = Mock(return_value=None)
        mock_progress.update = Mock()
        batch_ops.progress_display.batch_progress = Mock(return_value=mock_progress)

        await batch_ops.fetch_multiple(urls, "html", max_concurrent=10, retry=True)

        # Should pass filtered options to fetcher
        batch_ops._fetch_html_and_cache.assert_called_once_with(
            "http://example.com", retry=True
        )


class TestConcurrentFetchWithProgress:
    """Test concurrent fetching with progress display."""

    @pytest.fixture
    def batch_ops(self):
        mock_cache = Mock()
        mock_web_client = Mock()
        return BatchOperations(mock_cache, mock_web_client, show_status=False)

    @pytest.mark.anyio
    async def test_concurrent_fetch_success(self, batch_ops):
        """Test successful concurrent fetching."""
        urls = ["http://example1.com", "http://example2.com"]

        # Mock fetcher function
        async def mock_fetcher(url, **kwargs):
            return f"content_{url.split('.')[-2][-1]}"  # Return content_1, content_2

        # Mock progress display
        mock_progress = Mock()
        mock_progress.__enter__ = Mock(return_value=mock_progress)
        mock_progress.__exit__ = Mock(return_value=None)
        mock_progress.update = Mock()
        batch_ops.progress_display.batch_progress = Mock(return_value=mock_progress)

        result = await batch_ops._concurrent_fetch_with_progress(
            urls, mock_fetcher, "html", max_concurrent=2, fail_fast=False
        )

        assert len(result) == 2
        assert "http://example1.com" in result
        assert "http://example2.com" in result

    @pytest.mark.anyio
    async def test_concurrent_fetch_with_error_fail_fast(self, batch_ops):
        """Test concurrent fetching with fail_fast=True."""
        urls = ["http://example1.com", "http://example2.com"]

        # Mock fetcher function that raises exception
        async def mock_fetcher_error(url, **kwargs):
            if url == "http://example1.com":
                raise RuntimeError("Fetch failed")
            return "success"

        # Mock progress display
        mock_progress = Mock()
        mock_progress.__enter__ = Mock(return_value=mock_progress)
        mock_progress.__exit__ = Mock(return_value=None)
        batch_ops.progress_display.batch_progress = Mock(return_value=mock_progress)

        with pytest.raises(RuntimeError, match="Fetch failed"):
            await batch_ops._concurrent_fetch_with_progress(
                urls, mock_fetcher_error, "html", max_concurrent=2, fail_fast=True
            )

    @pytest.mark.anyio
    async def test_concurrent_fetch_with_error_no_fail_fast(self, batch_ops):
        """Test concurrent fetching with fail_fast=False (graceful error handling)."""
        urls = ["http://example1.com", "http://example2.com"]

        # Mock fetcher function that raises exception for first URL
        async def mock_fetcher_error(url, **kwargs):
            if url == "http://example1.com":
                raise RuntimeError("Fetch failed")
            return "success"

        # Mock progress display
        mock_progress = Mock()
        mock_progress.__enter__ = Mock(return_value=mock_progress)
        mock_progress.__exit__ = Mock(return_value=None)
        mock_progress.update = Mock()
        batch_ops.progress_display.batch_progress = Mock(return_value=mock_progress)

        result = await batch_ops._concurrent_fetch_with_progress(
            urls, mock_fetcher_error, "html", max_concurrent=2, fail_fast=False
        )

        # Should return error for first URL and success for second
        assert len(result) == 2
        assert isinstance(result["http://example1.com"], RuntimeError)
        assert result["http://example2.com"] == "success"


class TestFetcherFunctions:
    """Test individual fetcher functions for different content types."""

    @pytest.fixture
    def batch_ops(self):
        mock_cache = Mock()
        mock_web_client = Mock()
        return BatchOperations(mock_cache, mock_web_client, show_status=False)

    @pytest.mark.anyio
    async def test_fetch_html_and_cache_cache_hit(self, batch_ops):
        """Test HTML fetching when content is cached."""
        url = "http://example.com"
        cached_content = "cached html content"

        batch_ops.cache.has_path = AsyncMock(return_value=True)
        batch_ops.cache.get_content = AsyncMock(return_value=cached_content)

        result = await batch_ops._fetch_html_and_cache(url)

        assert result == cached_content
        batch_ops.cache.has_path.assert_called_once_with(url, "html")
        batch_ops.cache.get_content.assert_called_once_with(url, "html")

    @pytest.mark.anyio
    async def test_fetch_html_and_cache_cache_miss(self, batch_ops):
        """Test HTML fetching when content is not cached."""
        url = "http://example.com"
        fetch_result = {
            "raw_content": "html content",
            "markdown_content": "markdown content",
            "doi": "10.1234/example",
            "final_url": "http://example.com/final",
        }

        batch_ops.cache.has_path = AsyncMock(return_value=False)
        batch_ops.web_client.fetch_html = AsyncMock(return_value=fetch_result)
        batch_ops.cache.set_content = AsyncMock()
        batch_ops.cache.clear_failed = AsyncMock()
        batch_ops.cache.mark_failed = AsyncMock()

        result = await batch_ops._fetch_html_and_cache(url, retry=True)

        assert result == "html content"
        batch_ops.web_client.fetch_html.assert_called_once_with(url, retry=True)

        # Should cache all extracted content
        assert batch_ops.cache.set_content.call_count == 3  # html, markdown, doi

    @pytest.mark.anyio
    async def test_fetch_pdf_and_cache(self, batch_ops):
        """Test PDF fetching and caching."""
        url = "http://example.com/paper.pdf"
        fetch_result = {
            "raw_content": "pdf content",
            "markdown_content": "markdown content",
            "final_url": "http://example.com/paper.pdf",
        }

        batch_ops.cache.has_path = AsyncMock(return_value=False)
        batch_ops.cache.is_failed = AsyncMock(return_value=False)
        batch_ops.cache.get_redirect_info = AsyncMock(return_value=None)
        batch_ops.web_client.fetch_pdf = AsyncMock(return_value=fetch_result)
        batch_ops.cache.set_content = AsyncMock()
        batch_ops.cache.clear_failed = AsyncMock()
        batch_ops.cache.mark_failed = AsyncMock()

        result = await batch_ops._fetch_pdf_and_cache(url)

        assert result == "pdf content"
        batch_ops.web_client.fetch_pdf.assert_called_once_with(url, retry=False)

        # Should cache both PDF and markdown content
        assert batch_ops.cache.set_content.call_count == 2

    @pytest.mark.anyio
    async def test_fetch_markdown_and_cache(self, batch_ops):
        """Test markdown fetching and caching."""
        url = "http://example.com"
        raw_markdown = "raw markdown content"
        fetch_result = {
            "raw_content": "html content",
            "markdown_content": raw_markdown,
            "final_url": "http://example.com/final",
        }

        batch_ops.cache.has_path = AsyncMock(return_value=False)
        batch_ops.cache.is_failed = AsyncMock(return_value=False)
        batch_ops.web_client._is_pdf_url = Mock(return_value=False)
        batch_ops.web_client.fetch_html = AsyncMock(return_value=fetch_result)
        batch_ops.cache.set_content = AsyncMock()
        batch_ops.cache.clear_failed = AsyncMock()
        batch_ops.cache.mark_failed = AsyncMock()
        batch_ops.cache.get_redirect_info = AsyncMock(return_value=None)

        # Mock the content processor
        with patch(
            "interaction_finder.fetcher.content_processor.ContentProcessor"
        ) as mock_processor_class:
            mock_processor = mock_processor_class.return_value
            processed_markdown = "processed markdown content"
            mock_processor.refine_article.return_value = processed_markdown

            result = await batch_ops._fetch_markdown_and_cache(url)

            assert result == processed_markdown
            batch_ops.web_client.fetch_html.assert_called_once_with(url, retry=False)
            # Should cache both raw_markdown and processed markdown
            assert batch_ops.cache.set_content.call_count == 2

    @pytest.mark.anyio
    async def test_fetch_chunks_and_cache(self, batch_ops):
        """Test chunks fetching and caching."""
        url = "http://example.com"
        markdown_content = "markdown content"
        chunks = ["chunk1", "chunk2", "chunk3"]

        batch_ops.cache.has_path = AsyncMock(return_value=False)
        batch_ops.web_client._is_pdf_url = Mock(return_value=False)
        batch_ops._fetch_html_and_cache = AsyncMock()
        batch_ops._fetch_markdown_and_cache = AsyncMock(return_value=markdown_content)
        batch_ops.web_client.create_chunks = Mock(return_value=chunks)
        batch_ops.cache.set_content = AsyncMock()
        batch_ops.cache.get_redirect_info = AsyncMock(return_value=None)

        result = await batch_ops._fetch_chunks_and_cache(url, retry=True)

        assert result == chunks
        # Should first fetch HTML, then markdown, then create chunks
        batch_ops._fetch_html_and_cache.assert_called_once_with(url, retry=True)
        batch_ops._fetch_markdown_and_cache.assert_called_once_with(url, retry=True)
        batch_ops.web_client.create_chunks.assert_called_once_with(markdown_content)
        batch_ops.cache.set_content.assert_called_once_with(url, "chunks", chunks, None)


class TestPrefetchOperations:
    """Test prefetch functionality."""

    @pytest.fixture
    def batch_ops(self):
        mock_cache = Mock()
        mock_web_client = Mock()
        return BatchOperations(mock_cache, mock_web_client, show_status=False)

    @pytest.mark.anyio
    async def test_prefetch_urls_empty(self, batch_ops):
        """Test prefetch with empty URL list."""
        await batch_ops.prefetch_urls([])

        # Should return without error

    @pytest.mark.anyio
    async def test_prefetch_urls_all_cached(self, batch_ops):
        """Test prefetch when all URLs are already cached."""
        urls = ["http://example.com"]

        batch_ops.cache.has_path = AsyncMock(return_value=True)

        await batch_ops.prefetch_urls(urls, ["html"])

        # Should check cache but not fetch anything

    @pytest.mark.anyio
    async def test_prefetch_urls_with_uncached(self, batch_ops):
        """Test prefetch with uncached URLs."""
        urls = ["http://example.com"]

        batch_ops.cache.has_path = AsyncMock(return_value=False)
        batch_ops._fetch_html_and_cache = AsyncMock(return_value="content")

        # Mock progress display
        mock_progress = Mock()
        mock_progress.__enter__ = Mock(return_value=mock_progress)
        mock_progress.__exit__ = Mock(return_value=None)
        mock_progress.update = Mock()
        batch_ops.progress_display.batch_progress = Mock(return_value=mock_progress)

        await batch_ops.prefetch_urls(urls, ["html"])

        # Should attempt to fetch uncached content
        batch_ops.cache.has_path.assert_called_with("http://example.com", "html")

    @pytest.mark.anyio
    async def test_safe_fetch_single_success(self, batch_ops):
        """Test safe fetch single URL success."""

        async def mock_fetcher(url):
            return "content"

        result = await batch_ops._safe_fetch_single("http://example.com", mock_fetcher)

        assert result == "content"

    @pytest.mark.anyio
    async def test_safe_fetch_single_error(self, batch_ops):
        """Test safe fetch single URL with error (should return None)."""

        async def mock_fetcher_error(url):
            raise RuntimeError("Fetch failed")

        result = await batch_ops._safe_fetch_single(
            "http://example.com", mock_fetcher_error
        )

        assert result is None


class TestUtilityFunctions:
    """Test utility functions for batch operations."""

    @pytest.fixture
    def batch_ops(self):
        return BatchOperations(Mock(), Mock(), show_status=False)

    @pytest.mark.anyio
    async def test_clear_cache_for_urls(self, batch_ops):
        """Test clearing cache for specified URLs."""
        urls = ["http://example1.com", "http://example2.com"]

        batch_ops.cache.clear_url = AsyncMock()

        await batch_ops.clear_cache_for_urls(urls)

        assert batch_ops.cache.clear_url.call_count == 2

    def test_calculate_batch_content_size_strings(self, batch_ops):
        """Test calculating content size for string results."""
        results = ["short", "longer content", "medium"]

        size = batch_ops.calculate_batch_content_size(results, "html")

        expected = len("short") + len("longer content") + len("medium")
        assert size == expected

    def test_calculate_batch_content_size_chunks(self, batch_ops):
        """Test calculating content size for chunk results."""
        results = [
            ["chunk1", "chunk2"],
            ["chunk3", "chunk4", "chunk5"],
            "string result",
        ]

        size = batch_ops.calculate_batch_content_size(results, "chunks")

        expected = (
            len("chunk1")
            + len("chunk2")
            + len("chunk3")
            + len("chunk4")
            + len("chunk5")
            + len("string result")
        )
        assert size == expected

    def test_calculate_batch_content_size_with_errors(self, batch_ops):
        """Test calculating content size with error results."""
        results = ["content", RuntimeError("error"), None, "more content"]

        size = batch_ops.calculate_batch_content_size(results, "html")

        expected = len("content") + len("more content")
        assert size == expected

    def test_calculate_optimal_width_basic(self, batch_ops):
        """Test optimal width calculation."""
        urls = [
            "http://example.com",
            "http://verylongdomainname.com",
            "http://short.co",
            "http://medium-domain.org",
        ]

        width = batch_ops._calculate_optimal_width(urls)

        assert isinstance(width, int)
        assert 10 <= width <= 20

    def test_calculate_optimal_width_single_url(self, batch_ops):
        """Test optimal width calculation with single URL."""
        urls = ["http://example.com"]

        width = batch_ops._calculate_optimal_width(urls)

        assert width == len("example.com")

    def test_calculate_optimal_width_empty(self, batch_ops):
        """Test optimal width calculation with empty URLs."""
        urls = []

        width = batch_ops._calculate_optimal_width(urls)

        assert width == 15  # Default

    def test_format_domain_normal(self, batch_ops):
        """Test domain formatting with normal length."""
        batch_ops._domain_width = 20

        formatted = batch_ops.format_domain("http://example.com")

        assert formatted == "example.com".ljust(20)

    def test_format_domain_too_long(self, batch_ops):
        """Test domain formatting when domain is too long."""
        batch_ops._domain_width = 10

        formatted = batch_ops.format_domain("http://verylongdomainname.com")

        assert len(formatted) == 10
        assert formatted.endswith("…")

    def test_format_domain_invalid_url(self, batch_ops):
        """Test domain formatting with invalid URL."""
        batch_ops._domain_width = 15

        formatted = batch_ops.format_domain("not-a-valid-url")

        assert len(formatted) == 15


class TestStandaloneFunctions:
    """Test standalone batch functions for backward compatibility."""

    @pytest.mark.anyio
    async def test_fetch_urls_with_progress(self):
        """Test standalone sequential fetch function."""
        urls = ["http://example.com"]

        # Mock all dependencies that the standalone function creates
        # The function imports these locally, so we need to patch the original modules
        with (
            patch("interaction_finder.fetcher.cache.URLCache") as mock_cache_class,
            patch(
                "interaction_finder.fetcher.web_client.WebClient"
            ) as mock_web_client_class,
            patch(
                "interaction_finder.fetcher.batch_operations.BatchOperations"
            ) as mock_batch_class,
        ):
            # Create mock instances
            mock_cache = Mock()
            mock_web_client = Mock()
            mock_batch = Mock()

            mock_cache_class.return_value = mock_cache
            mock_web_client_class.return_value = mock_web_client
            mock_batch_class.return_value = mock_batch
            mock_batch.fetch_multiple = AsyncMock(return_value=["result"])

            result = await fetch_urls_with_progress(
                urls, Mock(), "html", max_concurrent=5, progress=True
            )

            assert result == ["result"]
            mock_batch.fetch_multiple.assert_called_once_with(
                urls,
                "html",
                max_concurrent=1,  # Sequential
            )

    @pytest.mark.anyio
    async def test_fetch_urls_concurrent_with_progress(self):
        """Test standalone concurrent fetch function."""
        urls = ["http://example.com"]

        # Mock all dependencies that the standalone function creates
        # The function imports these locally, so we need to patch the original modules
        with (
            patch("interaction_finder.fetcher.cache.URLCache") as mock_cache_class,
            patch(
                "interaction_finder.fetcher.web_client.WebClient"
            ) as mock_web_client_class,
            patch(
                "interaction_finder.fetcher.batch_operations.BatchOperations"
            ) as mock_batch_class,
        ):
            # Create mock instances
            mock_cache = Mock()
            mock_web_client = Mock()
            mock_batch = Mock()

            mock_cache_class.return_value = mock_cache
            mock_web_client_class.return_value = mock_web_client
            mock_batch_class.return_value = mock_batch
            mock_batch.fetch_multiple = AsyncMock(return_value=["result"])

            result = await fetch_urls_concurrent_with_progress(
                urls, Mock(), "html", max_concurrent=10, progress=False
            )

            assert result == ["result"]
            mock_batch.fetch_multiple.assert_called_once_with(
                urls,
                "html",
                max_concurrent=10,  # Concurrent
            )


class TestErrorHandling:
    """Test error handling in batch operations."""

    @pytest.fixture
    def batch_ops(self):
        mock_cache = Mock()
        mock_web_client = Mock()
        return BatchOperations(mock_cache, mock_web_client, show_status=False)

    @pytest.mark.anyio
    async def test_fetch_multiple_cache_error(self, batch_ops):
        """Test fetch_multiple when cache operations fail."""
        urls = ["http://example.com"]

        batch_ops.cache.get_content_batch = AsyncMock(
            side_effect=RuntimeError("Cache error")
        )

        with pytest.raises(RuntimeError, match="Cache error"):
            await batch_ops.fetch_multiple(urls, "html")

    @pytest.mark.anyio
    async def test_fetch_multiple_invalid_content_type(self, batch_ops):
        """Test fetch_multiple with invalid content type."""
        urls = ["http://example.com"]

        batch_ops.cache.get_content_batch = AsyncMock(
            return_value=[("http://example.com", None)]
        )

        with pytest.raises(ValueError, match="Unknown content type: invalid"):
            await batch_ops.fetch_multiple(urls, "invalid")
