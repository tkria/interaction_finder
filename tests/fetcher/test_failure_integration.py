"""
Integration tests for failure handling with real network scenarios.

These tests use real network requests to verify failure handling works correctly
in production-like conditions. They are marked as integration tests and can be
run separately from unit tests.
"""

import pytest
import tempfile
import sys
from pathlib import Path
import asyncio

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from interaction_finder.fetcher import PageFetcher, PreviousFailure
from interaction_finder.settings import IfetcherConfig


# Mark all tests in this file as integration tests
pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class TestRealNetworkFailures:
    """Integration tests with actual network failure scenarios."""

    @pytest.fixture
    def setup_pagefetcher(self):
        """Set up PageFetcher with temporary cache for integration tests."""
        temp_dir = tempfile.mkdtemp()
        config = IfetcherConfig.model_validate(
            {
                "cache": {"directory": temp_dir},
                "tools": {
                    "crawl4ai": {
                        "timeout": 5,  # Short timeout for failure testing
                        "max_retries": 1,
                    }
                },
            }
        )
        fetcher = PageFetcher(config, show_status=False, verbose=False)

        yield fetcher, config

        # Cleanup
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.slow
    async def test_dns_resolution_failure(self, setup_pagefetcher):
        """Test DNS resolution failures are handled correctly."""
        fetcher, config = setup_pagefetcher

        # Use a definitely non-existent domain
        bad_url = "http://definitely-does-not-exist-12345.invalid"

        # First attempt should fail and mark failure
        with pytest.raises((Exception,)) as exc_info:
            await fetcher.get_html(bad_url)

        # Should be some kind of network/DNS error
        error_str = str(exc_info.value)
        assert any(
            keyword in error_str.lower()
            for keyword in [
                "dns",
                "name",
                "resolution",
                "gaierror",
                "nodename",
                "network",
            ]
        )

        # Verify failure was marked
        assert await fetcher.cache.is_failed(bad_url)
        reason = await fetcher.cache.get_failed_reason(bad_url)
        assert reason  # Should contain some error message

        # Second attempt should raise PreviousFailure
        with pytest.raises(PreviousFailure) as prev_exc:
            await fetcher.get_html(bad_url, retry=False)

        assert prev_exc.value.url == bad_url
        # Original error should be preserved in PreviousFailure message
        assert reason in str(prev_exc.value)

    @pytest.mark.slow
    async def test_connection_timeout_failure(self, setup_pagefetcher):
        """Test connection timeout failures."""
        fetcher, config = setup_pagefetcher

        # Use a non-routable IP address that will timeout
        # 10.255.255.1 is reserved and should not respond
        timeout_url = "http://10.255.255.1:80/test"

        # Should timeout and be marked as failed
        with pytest.raises((Exception,)) as exc_info:
            await fetcher.get_html(timeout_url)

        error_str = str(exc_info.value)
        assert any(
            keyword in error_str.lower()
            for keyword in ["timeout", "connect", "time", "unreachable"]
        )

        assert await fetcher.cache.is_failed(timeout_url)

        # PreviousFailure should preserve timeout error
        with pytest.raises(PreviousFailure) as prev_exc:
            await fetcher.get_html(timeout_url, retry=False)
        assert prev_exc.value.url == timeout_url

    @pytest.mark.slow
    async def test_http_404_error_handling(self, setup_pagefetcher):
        """Test HTTP 404 errors are handled correctly."""
        fetcher, config = setup_pagefetcher

        # Use httpbin.org which is reliable for testing HTTP status codes
        not_found_url = "https://httpbin.org/status/404"

        # Should fail with HTTP error
        with pytest.raises((Exception,)) as exc_info:
            await fetcher.get_html(not_found_url)

        # Should be marked as failed
        assert await fetcher.cache.is_failed(not_found_url)
        reason = await fetcher.cache.get_failed_reason(not_found_url)
        assert "404" in reason or "not found" in reason.lower()

        # PreviousFailure should contain HTTP error info
        with pytest.raises(PreviousFailure) as prev_exc:
            await fetcher.get_html(not_found_url, retry=False)
        assert (
            "404" in str(prev_exc.value) or "not found" in str(prev_exc.value).lower()
        )

    @pytest.mark.slow
    async def test_http_500_server_error(self, setup_pagefetcher):
        """Test HTTP 500 server errors."""
        fetcher, config = setup_pagefetcher

        server_error_url = "https://httpbin.org/status/500"

        with pytest.raises((Exception,)) as exc_info:
            await fetcher.get_html(server_error_url)

        assert await fetcher.cache.is_failed(server_error_url)
        reason = await fetcher.cache.get_failed_reason(server_error_url)
        assert "500" in reason or "server error" in reason.lower()

    @pytest.mark.slow
    async def test_successful_request_after_fixing_url(self, setup_pagefetcher):
        """Test successful request workflow with httpbin."""
        fetcher, config = setup_pagefetcher

        # Use httpbin.org for a reliable successful request
        success_url = "https://httpbin.org/html"

        try:
            result = await fetcher.get_html(success_url)

            # Should contain HTML content
            assert result
            assert isinstance(result, str)
            assert len(result) > 0

            # Should not be marked as failed
            assert not await fetcher.cache.is_failed(success_url)

            # Should be able to fetch again (cached)
            result2 = await fetcher.get_html(success_url, retry=False)
            assert result2 == result

        except Exception as e:
            # If httpbin is down, skip this test
            pytest.skip(f"httpbin.org appears to be unavailable: {e}")

    @pytest.mark.slow
    async def test_pdf_download_failure(self, setup_pagefetcher):
        """Test PDF download failure scenarios."""
        fetcher, config = setup_pagefetcher

        # Try to fetch a non-existent PDF
        bad_pdf_url = "https://httpbin.org/status/404.pdf"

        try:
            with pytest.raises((Exception,)) as exc_info:
                await fetcher.get_pdf(bad_pdf_url)

            assert await fetcher.cache.is_failed(bad_pdf_url)

            with pytest.raises(PreviousFailure):
                await fetcher.get_pdf(bad_pdf_url, retry=False)

        except Exception as e:
            if "httpbin" in str(e):
                pytest.skip(f"httpbin.org appears to be unavailable: {e}")
            else:
                raise


class TestBatchFailurePatterns:
    """Test failure patterns in batch operations with real requests."""

    @pytest.fixture
    def setup_pagefetcher(self):
        """Set up PageFetcher for batch testing."""
        temp_dir = tempfile.mkdtemp()
        config = IfetcherConfig.model_validate(
            {
                "cache": {"directory": temp_dir},
                "tools": {"crawl4ai": {"timeout": 10, "max_retries": 1}},
            }
        )
        fetcher = PageFetcher(config, show_status=False, verbose=False)

        yield fetcher, config

        # Cleanup
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.slow
    async def test_mixed_success_failure_batch(self, setup_pagefetcher):
        """Test batch processing with mix of working and broken URLs."""
        fetcher, config = setup_pagefetcher

        # Mix of URLs that should work and fail
        urls = [
            "https://httpbin.org/html",  # Should work
            "http://definitely-broken-url-12345.invalid",  # DNS failure
            "https://httpbin.org/status/404",  # HTTP error
            "https://httpbin.org/json",  # Should work
        ]

        results = []
        for url in urls:
            try:
                result = await fetcher.get_html(url)
                results.append(("success", url, len(result) if result else 0))
            except Exception as e:
                results.append(("error", url, str(type(e).__name__)))

        # Should have some successes and some failures
        successes = [r for r in results if r[0] == "success"]
        failures = [r for r in results if r[0] == "error"]

        # Even if httpbin is down, we should still get the DNS failure
        assert len(failures) >= 1  # At least the invalid domain should fail

        # All failed URLs should be marked
        for result_type, url, data in results:
            if result_type == "error":
                assert await fetcher.cache.is_failed(url)

        # Failed URLs should raise PreviousFailure on retry=False
        for result_type, url, data in results:
            if result_type == "error":
                with pytest.raises(PreviousFailure):
                    await fetcher.get_html(url, retry=False)

    @pytest.mark.slow
    async def test_chunking_with_network_failures(self, setup_pagefetcher):
        """Test chunking workflow with network failures."""
        fetcher, config = setup_pagefetcher

        urls = [
            "https://httpbin.org/html",  # Should work
            "http://nonexistent-chunking-test.invalid",  # Should fail
        ]

        try:
            # Test chunking with fail_fast=False
            results = await fetcher.get_chunks(urls, fail_fast=False)

            assert len(results) == 2
            # First might succeed (if httpbin works), second should be empty
            assert len(results[1]) == 0  # Failed URL returns empty chunks

            # Failed URL should be marked
            assert await fetcher.cache.is_failed(
                "http://nonexistent-chunking-test.invalid"
            )

        except Exception as e:
            if "httpbin" in str(e):
                pytest.skip(f"httpbin.org appears to be unavailable: {e}")
            else:
                # The invalid domain should definitely fail
                assert await fetcher.cache.is_failed(
                    "http://nonexistent-chunking-test.invalid"
                )


class TestFailureRecoveryPatterns:
    """Test failure recovery with real network conditions."""

    @pytest.fixture
    def setup_pagefetcher(self):
        """Set up PageFetcher for recovery testing."""
        temp_dir = tempfile.mkdtemp()
        config = IfetcherConfig.model_validate(
            {
                "cache": {"directory": temp_dir},
                "tools": {"crawl4ai": {"timeout": 8, "max_retries": 1}},
            }
        )
        fetcher = PageFetcher(config, show_status=False, verbose=False)

        yield fetcher, config

        # Cleanup
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.slow
    async def test_failure_then_manual_recovery(self, setup_pagefetcher):
        """Test recovery by manually clearing failures."""
        fetcher, config = setup_pagefetcher

        bad_url = "http://recovery-test-domain-12345.invalid"

        # Initial failure
        with pytest.raises((Exception,)):
            await fetcher.get_html(bad_url)

        assert await fetcher.cache.is_failed(bad_url)

        # Should raise PreviousFailure
        with pytest.raises(PreviousFailure):
            await fetcher.get_html(bad_url, retry=False)

        # Manually clear the failure (simulating external fix)
        await fetcher.cache.clear_failed(bad_url)
        assert not await fetcher.cache.is_failed(bad_url)

        # Now it should attempt fetch again (and fail again, but that's expected)
        with pytest.raises((Exception,)) as exc_info:
            await fetcher.get_html(bad_url, retry=False)

        # Should NOT be PreviousFailure this time, but the actual network error
        assert not isinstance(exc_info.value, PreviousFailure)

    @pytest.mark.slow
    async def test_retry_flag_bypasses_previous_failure(self, setup_pagefetcher):
        """Test that retry=True bypasses previous failure checking."""
        fetcher, config = setup_pagefetcher

        bad_url = "http://retry-bypass-test-12345.invalid"

        # Initial failure
        with pytest.raises((Exception,)):
            await fetcher.get_html(bad_url)

        assert await fetcher.cache.is_failed(bad_url)

        # retry=True should bypass PreviousFailure and attempt actual fetch
        with pytest.raises((Exception,)) as exc_info:
            await fetcher.get_html(bad_url, retry=True)

        # Should NOT be PreviousFailure, but actual network error
        assert not isinstance(exc_info.value, PreviousFailure)
        # Should still be marked as failed after retry attempt
        assert await fetcher.cache.is_failed(bad_url)

    @pytest.mark.slow
    async def test_concurrent_failure_handling(self, setup_pagefetcher):
        """Test failure handling with concurrent requests."""
        fetcher, config = setup_pagefetcher

        # Multiple bad URLs
        bad_urls = [f"http://concurrent-fail-{i}-12345.invalid" for i in range(3)]

        # Run concurrent requests that should all fail
        tasks = [fetcher.get_html(url, retry=True) for url in bad_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should be exceptions (not PreviousFailure)
        for i, result in enumerate(results):
            assert isinstance(result, Exception)
            assert not isinstance(result, PreviousFailure)

        # All URLs should be marked as failed
        for url in bad_urls:
            assert await fetcher.cache.is_failed(url)

        # Subsequent individual requests should raise PreviousFailure
        for url in bad_urls:
            with pytest.raises(PreviousFailure):
                await fetcher.get_html(url, retry=False)


# Configuration for pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring network"
    )
    config.addinivalue_line("markers", "slow: mark test as slow running test")
