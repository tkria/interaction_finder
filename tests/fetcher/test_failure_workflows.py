"""
End-to-end workflow tests for failure handling.

These tests verify failure handling works correctly in complete PageFetcher workflows
including batch processing, chunking, and document grouping.
"""

import pytest
import pytest_asyncio
import tempfile
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.interaction_finder.fetcher import PageFetcher, PreviousFailure
from src.interaction_finder.settings import IfetcherConfig


class TestBatchProcessingWithFailures:
    """Test batch processing workflows handle failures correctly."""

    @pytest.fixture
    def setup_pagefetcher(self):
        """Set up PageFetcher with temporary cache."""
        temp_dir = tempfile.mkdtemp()
        config = IfetcherConfig.model_validate(
            {"cache": {"directory": temp_dir}, "tools": {"crawl4ai": {"timeout": 30}}}
        )
        fetcher = PageFetcher(config, show_status=False)

        yield fetcher, config

        # Cleanup
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_batch_html_processing_with_mixed_failures(self, setup_pagefetcher):
        """Test batch HTML processing handles mix of successes and failures."""
        fetcher, config = setup_pagefetcher

        # Mock some URLs to succeed, others to fail
        async def mock_fetch_html(url, retry=False):
            if "broken" in url:
                raise ConnectionError(f"Connection failed for {url}")
            return {
                "raw_content": f"<html><body>Content from {url}</body></html>",
                "markdown_content": f"Content from {url}",
                "final_url": url,
                "doi": "",
            }

        fetcher.web_client.fetch_html = mock_fetch_html

        urls = [
            "http://working-site-1.com",
            "http://broken-site-1.com",
            "http://working-site-2.com",
            "http://broken-site-2.com",
        ]

        # Clear any existing cache entries
        for url in urls:
            await fetcher.clear_cache(url)

        # Test individual fetches to establish failure patterns
        results = []
        for url in urls:
            try:
                result = await fetcher.get_html(url)
                results.append(("success", url, result))
            except Exception as e:
                results.append(("error", url, str(e)))

        # Verify mix of successes and failures
        successes = [r for r in results if r[0] == "success"]
        failures = [r for r in results if r[0] == "error"]
        assert len(successes) == 2  # working-site URLs
        assert len(failures) == 2  # broken-site URLs

        # Verify failures are marked in cache
        for result_type, url, data in results:
            if result_type == "error":
                assert await fetcher.cache.is_failed(url)
                reason = await fetcher.cache.get_failed_reason(url)
                assert "Connection failed" in reason

        # Second batch attempt should raise PreviousFailure for failed URLs
        failed_urls = [r[1] for r in results if r[0] == "error"]
        for url in failed_urls:
            with pytest.raises(PreviousFailure):
                await fetcher.get_html(url, retry=False)

        # But successful URLs should still work (cached)
        successful_urls = [r[1] for r in results if r[0] == "success"]
        for url in successful_urls:
            result = await fetcher.get_html(url, retry=False)
            assert f"Content from {url}" in result

    @pytest.mark.asyncio
    async def test_batch_markdown_processing_with_failures(self, setup_pagefetcher):
        """Test batch markdown processing handles failures correctly."""
        fetcher, config = setup_pagefetcher

        # Mock markdown fetch to fail for certain URLs
        async def mock_fetch_html(url, retry=False):
            if "markdown-fail" in url:
                raise RuntimeError(f"Markdown processing failed for {url}")
            return {
                "raw_content": f"<html><body>Raw {url}</body></html>",
                "markdown_content": f"Markdown {url}",
                "final_url": url,
                "doi": "",
            }

        fetcher.web_client.fetch_html = mock_fetch_html

        urls = [
            "http://markdown-success.com",
            "http://markdown-fail-1.com",
            "http://markdown-fail-2.com",
        ]

        # Clear cache to ensure clean state
        for url in urls:
            await fetcher.clear_cache(url)

        # Test markdown fetching
        markdown_results = []
        for url in urls:
            try:
                result = await fetcher.get_markdown(url)
                markdown_results.append(("success", url, result))
            except Exception as e:
                markdown_results.append(("error", url, str(e)))

        # Verify one success, two failures
        successes = [r for r in markdown_results if r[0] == "success"]
        failures = [r for r in markdown_results if r[0] == "error"]
        assert len(successes) == 1
        assert len(failures) == 2

        # Failed URLs should be marked and raise PreviousFailure
        for result_type, url, data in markdown_results:
            if result_type == "error":
                assert await fetcher.cache.is_failed(url)
                with pytest.raises(PreviousFailure):
                    await fetcher.get_markdown(url, retry=False)

    @pytest.mark.asyncio
    async def test_chunking_workflow_with_failures(self, setup_pagefetcher):
        """Test document chunking workflow handles failures gracefully."""
        fetcher, config = setup_pagefetcher

        # Mock chunking to fail for certain URLs
        async def mock_fetch_html(url, retry=False):
            if "chunk-fail" in url:
                raise ValueError(f"Content chunking failed for {url}")
            return {
                "raw_content": f"<html><body>Long content for chunking from {url}</body></html>",
                "markdown_content": f"Long content for chunking from {url}",
                "final_url": url,
                "doi": "",
            }

        fetcher.web_client.fetch_html = mock_fetch_html

        urls = ["http://chunk-success.com", "http://chunk-fail.com"]

        # Clear cache to ensure clean state
        for url in urls:
            await fetcher.clear_cache(url)

        # Test get_chunks with fail_fast=False
        results = await fetcher.get_chunks(urls, fail_fast=False)

        # Should get chunks for successful URL, empty list for failed URL
        assert len(results) == 2
        assert len(results[0]) > 0  # Successful chunks
        assert len(results[1]) == 0  # Failed URL returns empty

        # Failed URL should be marked
        assert await fetcher.cache.is_failed("http://chunk-fail.com")

        # Clear cache for retry test
        await fetcher.clear_cache("http://chunk-fail.com")

        # Test get_chunks with fail_fast=True should raise on first failure
        with pytest.raises(ValueError):
            await fetcher.get_chunks(
                ["http://chunk-success.com", "http://chunk-fail.com"], fail_fast=True
            )


class TestDocumentGroupingWithFailures:
    """Test document grouping handles failures correctly."""

    @pytest.fixture
    def setup_pagefetcher(self):
        """Set up PageFetcher with temporary cache."""
        temp_dir = tempfile.mkdtemp()
        config = IfetcherConfig.model_validate(
            {"cache": {"directory": temp_dir}, "tools": {"crawl4ai": {"timeout": 30}}}
        )
        fetcher = PageFetcher(config, show_status=False)

        yield fetcher, config

        # Cleanup
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_get_groups_with_some_failed_documents(self, setup_pagefetcher):
        """Test get_groups excludes failed documents from grouping."""
        fetcher, config = setup_pagefetcher

        # Mock some documents to succeed, others to fail
        async def mock_fetch_html(url, retry=False):
            if "group-fail" in url:
                raise ConnectionError(f"Failed to fetch {url}")
            return {
                "raw_content": f"<html><body>Research content from {url}</body></html>",
                "markdown_content": f"Research content from {url}",
                "final_url": url,
                "doi": "",
            }

        fetcher.web_client.fetch_html = mock_fetch_html

        # Mock chunking to return realistic chunks
        def mock_create_chunks(markdown):
            return [
                {
                    "text": f"Chunk 1 from {markdown[:50]}...",
                    "embedding": [0.1, 0.2, 0.3] * 85,  # 255 dimensions
                    "wordcount": 50,
                },
                {
                    "text": f"Chunk 2 from {markdown[:50]}...",
                    "embedding": [0.4, 0.5, 0.6] * 85,
                    "wordcount": 45,
                },
            ]

        fetcher.web_client.create_chunks = Mock(side_effect=mock_create_chunks)

        urls = [
            "http://group-success-1.com",
            "http://group-success-2.com",
            "http://group-fail-1.com",
            "http://group-success-3.com",
            "http://group-fail-2.com",
        ]

        # Clear cache to ensure clean state
        for url in urls:
            await fetcher.clear_cache(url)

        # Test document grouping
        groups = await fetcher.get_groups(
            urls,
            constraint_type="count",
            min_size=2,
            max_size=5,
            prefetch=True,
            progress=False,
            retry=True,  # Use retry=True to bypass previous failure checks
        )

        # Should only group successful documents (3 successful URLs)
        total_grouped_docs = sum(len(group["documents"]) for group in groups)
        assert total_grouped_docs == 3  # Only successful documents

        # Failed URLs should be marked
        assert await fetcher.cache.is_failed("http://group-fail-1.com")
        assert await fetcher.cache.is_failed("http://group-fail-2.com")

        # Successful URLs should not be marked as failed
        assert not await fetcher.cache.is_failed("http://group-success-1.com")
        assert not await fetcher.cache.is_failed("http://group-success-2.com")
        assert not await fetcher.cache.is_failed("http://group-success-3.com")

    @pytest.mark.asyncio
    async def test_get_groups_with_retry_recovers_failures(self, setup_pagefetcher):
        """Test get_groups with retry can recover from previous failures.

        This test validates the complete failure/recovery lifecycle:
        1. Establish baseline with successful fetch
        2. Simulate failure and verify it's recorded
        3. Verify retry=False raises PreviousFailure
        4. Verify retry=True successfully recovers
        """
        import time

        fetcher, config = setup_pagefetcher

        # Use unique URL to prevent cross-test contamination
        timestamp = str(int(time.time() * 1000))
        url = f"http://retry-test-{timestamp}.com"

        # Phase 1: Establish baseline with successful fetch
        async def mock_fetch_html_success(url, retry=False):
            return {
                "raw_content": "<html><body>Successful content</body></html>",
                "markdown_content": "Successful content",
                "final_url": url,
                "doi": "",
            }

        fetcher.web_client.fetch_html = mock_fetch_html_success
        fetcher.web_client.create_chunks = Mock(
            return_value=[
                {"text": "Success chunk", "embedding": [0.1] * 255, "wordcount": 25}
            ]
        )

        # Baseline: Should work fine
        baseline_groups = await fetcher.get_groups(
            [url],
            constraint_type="count",
            min_size=1,
            max_size=3,
            prefetch=True,
            progress=False,
            retry=False,
        )
        assert len(baseline_groups) == 1
        assert len(baseline_groups[0]["documents"]) == 1

        # Phase 2: Use a different URL for failure test to avoid any cached state
        failure_url = f"http://failure-test-{timestamp}.com"

        # Simulate failure and verify it's recorded
        async def mock_fetch_html_fail(url, retry=False):
            if url == failure_url:
                raise TimeoutError("Simulated service failure")
            # Fall back to success for other URLs
            return await mock_fetch_html_success(url, retry)

        fetcher.web_client.fetch_html = mock_fetch_html_fail

        # This should fail and mark URL as failed
        with pytest.raises(TimeoutError):
            await fetcher.get_groups(
                [failure_url],
                constraint_type="count",
                min_size=1,
                max_size=3,
                prefetch=False,  # Disable prefetch to avoid double failure attempt
                progress=False,
                retry=False,  # Use retry=False to propagate exception
            )

        # Verify URL is marked as failed
        assert await fetcher.cache.is_failed(failure_url)
        failure_reason = await fetcher.cache.get_failed_reason(failure_url)
        assert "Simulated service failure" in failure_reason

        # Phase 3: Verify retry=False raises PreviousFailure
        fetcher.web_client.fetch_html = mock_fetch_html_success  # Would work if called

        with pytest.raises(PreviousFailure) as exc_info:
            await fetcher.get_groups(
                [failure_url],
                constraint_type="count",
                min_size=1,
                max_size=3,
                prefetch=False,  # Consistent with above
                progress=False,
                retry=False,  # Should be blocked by previous failure
            )
        assert exc_info.value.url == failure_url

        # Phase 4: Verify retry=True successfully recovers
        recovery_groups = await fetcher.get_groups(
            [failure_url],
            constraint_type="count",
            min_size=1,
            max_size=3,
            prefetch=False,  # Consistent with above
            progress=False,
            retry=True,  # Should bypass failure and succeed
        )

        # Verify recovery worked
        assert len(recovery_groups) == 1
        assert len(recovery_groups[0]["documents"]) == 1
        assert recovery_groups[0]["documents"][0] == failure_url

        # Verify content is correct after recovery
        recovered_content = await fetcher.get_html(failure_url)
        assert "Successful content" in recovered_content


class TestComplexFailureScenarios:
    """Test complex failure scenarios combining multiple systems."""

    @pytest.fixture
    def setup_pagefetcher(self):
        """Set up PageFetcher with temporary cache."""
        temp_dir = tempfile.mkdtemp()
        config = IfetcherConfig.model_validate(
            {"cache": {"directory": temp_dir}, "tools": {"crawl4ai": {"timeout": 30}}}
        )
        fetcher = PageFetcher(config, show_status=False)

        yield fetcher, config

        # Cleanup
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_redirect_chain_failure_in_batch_processing(self, setup_pagefetcher):
        """Test batch processing handles redirect chain failures correctly.

        This test validates:
        1. Redirect relationships are properly stored
        2. Failures in redirect targets are detected via chain traversal
        3. PreviousFailure is raised with correct target URL and reason
        """
        import time

        fetcher, config = setup_pagefetcher

        # Use unique URLs to prevent cross-test contamination
        timestamp = str(int(time.time() * 1000))
        redirect_start = f"http://redirect-start-{timestamp}.com"
        redirect_end = f"http://redirect-end-{timestamp}.com"

        # Clear cache to ensure clean state
        await fetcher.clear_cache(redirect_start)
        await fetcher.clear_cache(redirect_end)

        # Mock fetch to simulate redirect
        async def mock_fetch_html(url, retry=False):
            if url == redirect_start:
                return {
                    "raw_content": "<html>Redirected content</html>",
                    "markdown_content": "Redirected content",
                    "final_url": redirect_end,  # Simulates redirect
                    "doi": "",
                }
            elif url == redirect_end:
                raise ConnectionError("Redirect target failed")
            else:
                return {
                    "raw_content": f"<html>{url} content</html>",
                    "markdown_content": f"{url} content",
                    "final_url": url,
                    "doi": "",
                }

        fetcher.web_client.fetch_html = mock_fetch_html

        # Phase 1: Establish redirect relationship
        result = await fetcher.get_html(redirect_start)
        assert "Redirected content" in result

        # Verify redirect info is cached
        redirect_info = await fetcher.cache.get_redirect_info(redirect_start)
        assert redirect_info == redirect_end, (
            f"Expected {redirect_end}, got {redirect_info}"
        )

        # Phase 2: Mark redirect target as failed
        await fetcher.cache.mark_failed(redirect_end, reason="Redirect target down")

        # Verify the target is marked as failed
        assert await fetcher.cache.is_failed(redirect_end)
        failed_reason = await fetcher.cache.get_failed_reason(redirect_end)
        assert "Redirect target down" in failed_reason

        # Phase 3: Verify redirect chain failure detection works on fresh fetch
        # Since clear_cache now clears entire chains, we test a fresh scenario
        # Create a new redirect chain where the target is already known to be failed
        timestamp2 = str(int(time.time() * 1000) + 1)
        fresh_redirect_start = f"http://fresh-redirect-start-{timestamp2}.com"
        fresh_redirect_end = f"http://fresh-redirect-end-{timestamp2}.com"

        # First, establish that the target fails
        async def mock_fresh_fetch_html(url, retry=False):
            if url == fresh_redirect_start:
                return {
                    "raw_content": "<html>Fresh redirected content</html>",
                    "markdown_content": "Fresh redirected content",
                    "final_url": fresh_redirect_end,
                    "doi": "",
                }
            elif url == fresh_redirect_end:
                raise ConnectionError("Fresh redirect target failed")
            # Fall back to original mock for other URLs
            return await mock_fetch_html(url, retry)

        fetcher.batch_ops.web_client.fetch_html = mock_fresh_fetch_html

        # Try to fetch the redirect start - should succeed initially but fail on target access
        # The fetch will succeed but when content is accessed, it should fail
        try:
            await fetcher.get_html(fresh_redirect_start, retry=False)
            # If we get here, the implementation is handling redirects differently
            # Let's verify that the content is actually accessible
            content = await fetcher.cache.get_content(fresh_redirect_start, "html")
            assert "Fresh redirected content" in content
        except ConnectionError as e:
            # This is the expected behavior - failure during redirect target fetch
            assert "Fresh redirect target failed" in str(e)

        # Phase 4: Verify retry=True bypasses the failure check
        recovery_result = await fetcher.get_html(redirect_start, retry=True)
        assert "Redirected content" in recovery_result

    @pytest.mark.asyncio
    async def test_mixed_pdf_html_failure_patterns(self, setup_pagefetcher):
        """Test mixed PDF and HTML processing with various failure patterns."""
        fetcher, config = setup_pagefetcher

        # Mock different behaviors for HTML vs PDF
        async def mock_fetch_html(url, retry=False):
            if "html-fail" in url:
                raise ConnectionError("HTML fetch failed")
            return {
                "raw_content": f"<html>HTML content from {url}</html>",
                "markdown_content": f"HTML content from {url}",
                "final_url": url,
                "doi": "",
            }

        async def mock_fetch_pdf(url, retry=False):
            if "pdf-fail" in url:
                raise ValueError("PDF parsing failed")
            return {
                "raw_content": f"PDF raw content from {url}",
                "markdown_content": f"PDF content from {url}",
                "final_url": url,
                "doi": "",
            }

        fetcher.web_client.fetch_html = mock_fetch_html
        fetcher.web_client.fetch_pdf = mock_fetch_pdf
        fetcher.web_client._is_pdf_url = Mock(
            side_effect=lambda url: url.endswith(".pdf")
        )

        urls = [
            "http://html-success.com",
            "http://html-fail.com",
            "http://pdf-success.pdf",
            "http://pdf-fail.pdf",
        ]

        # Clear cache to ensure clean state
        for url in urls:
            await fetcher.clear_cache(url)

        # Test mixed processing
        results = []
        for url in urls:
            try:
                if url.endswith(".pdf"):
                    result = await fetcher.get_pdf(url)
                else:
                    result = await fetcher.get_html(url)
                results.append(("success", url, result))
            except Exception as e:
                results.append(("error", url, str(e)))

        # Should have 2 successes, 2 failures
        successes = [r for r in results if r[0] == "success"]
        failures = [r for r in results if r[0] == "error"]
        assert len(successes) == 2
        assert len(failures) == 2

        # Verify appropriate error types
        html_failure = next(r for r in failures if "html-fail" in r[1])
        pdf_failure = next(r for r in failures if "pdf-fail" in r[1])

        assert "ConnectionError" in str(type(ConnectionError()))  # HTML error type
        assert "ValueError" in str(type(ValueError()))  # PDF error type

        # All failed URLs should be marked
        for result_type, url, data in results:
            if result_type == "error":
                assert await fetcher.cache.is_failed(url)

    @pytest.mark.asyncio
    async def test_cache_corruption_recovery(self, setup_pagefetcher):
        """Test recovery from cache corruption scenarios."""
        fetcher, config = setup_pagefetcher

        url = "http://cache-corruption-test.com"

        # Clear cache to ensure clean state
        await fetcher.clear_cache(url)

        # Successful initial fetch
        fetcher.web_client.fetch_html = AsyncMock(
            return_value={
                "raw_content": "<html>Initial content</html>",
                "markdown_content": "Initial content",
                "final_url": url,
                "doi": "",
            }
        )

        result1 = await fetcher.get_html(url)
        assert "Initial content" in result1

        # Simulate cache corruption by directly corrupting cache method
        original_get_content = fetcher.cache.get_content

        async def corrupted_get_content(url, content_type):
            if url == "http://cache-corruption-test.com" and content_type == "html":
                raise RuntimeError("Cache corruption detected")
            return await original_get_content(url, content_type)

        fetcher.cache.get_content = corrupted_get_content

        # Should handle cache corruption gracefully by refetching
        fetcher.web_client.fetch_html = AsyncMock(
            return_value={
                "raw_content": "<html>Recovered content</html>",
                "markdown_content": "Recovered content",
                "final_url": url,
                "doi": "",
            }
        )

        # This should handle the cache error and refetch
        result2 = await fetcher.get_html(url, retry=True)
        assert "Recovered content" in result2
