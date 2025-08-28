"""
Regression tests for issues discovered during fetcher refactoring.

These tests specifically target bugs that were found and fixed during the
refactoring process to prevent them from reoccurring.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from interaction_finder.fetcher import PageFetcher, URLCache
from interaction_finder.fetcher.web_client import (
    _get_granular_logger,
    _get_quiet_logger,
)
from interaction_finder.settings import IfetcherConfig


class TestCachePathInitialization:
    """Test for cache path initialization bug found during refactoring."""

    def test_cache_accepts_path_objects(self):
        """Test that URLCache properly handles Path objects from config.abspath()."""
        # This would have caught the bug where cache expected string but got Path
        config = Mock()
        config.output = Mock()
        config.output.cache = "test_cache"
        config.abspath = Mock(return_value=Path("test_cache_path"))

        # This should not raise AttributeError: 'Path' object has no attribute 'mkdir'
        with tempfile.TemporaryDirectory() as temp_dir:
            config.abspath.return_value = Path(temp_dir) / "cache"
            cache = URLCache(config)
            assert cache.base_path.exists()
            assert isinstance(cache.base_path, Path)

    def test_cache_accepts_string_paths_for_backward_compatibility(self):
        """Test that URLCache can handle string paths (edge case)."""
        config = Mock()
        config.output = Mock()
        config.output.cache = "test_cache"

        # Test with string return (shouldn't happen with real config, but test anyway)
        with tempfile.TemporaryDirectory() as temp_dir:
            config.abspath = Mock(return_value=str(Path(temp_dir) / "cache"))
            cache = URLCache(config)
            assert cache.base_path.exists()
            assert isinstance(cache.base_path, Path)


class TestDOIExtractionJSONParsing:
    """Test for DOI extraction JSON parsing bug found during refactoring."""

    def test_doi_extraction_with_json_string(self):
        """Test DOI extraction when crawl4ai returns JSON string (original format)."""
        from interaction_finder.fetcher.web_client import WebClient

        config = Mock()
        config.tools = Mock()
        config.tools.crawl4ai = Mock()
        config.tools.crawl4ai.timeout = 30

        client = WebClient(config, verbose=False)

        # Mock crawl4ai result with JSON string format (what actually gets returned)
        result = Mock()
        result.extracted_content = json.dumps([{"doi_meta_cite": "10.1234/test.doi"}])

        doi = client._extract_doi(result)
        assert doi == "10.1234/test.doi"

    def test_doi_extraction_with_parsed_dict(self):
        """Test DOI extraction when data is already parsed (edge case)."""
        from interaction_finder.fetcher.web_client import WebClient

        config = Mock()
        config.tools = Mock()
        config.tools.crawl4ai = Mock()
        config.tools.crawl4ai.timeout = 30

        client = WebClient(config, verbose=False)

        # Mock result with already parsed dict (fallback case)
        result = Mock()
        result.extracted_content = [{"doi_meta_cite": "10.1234/parsed.doi"}]

        doi = client._extract_doi(result)
        assert doi == "10.1234/parsed.doi"

    def test_doi_extraction_field_priority(self):
        """Test that DOI extraction follows correct field priority."""
        from interaction_finder.fetcher.web_client import WebClient

        config = Mock()
        config.tools = Mock()
        config.tools.crawl4ai = Mock()
        config.tools.crawl4ai.timeout = 30

        client = WebClient(config, verbose=False)

        # Mock result with multiple DOI fields - should prefer doi_meta_pub
        result = Mock()
        result.extracted_content = json.dumps(
            [
                {
                    "doi_meta_cite": "10.1234/cite.doi",  # Lower priority
                    "doi_meta_pub": "10.1234/pub.doi",  # Higher priority
                    "doi_dc": "10.1234/dc.doi",  # Lower priority
                }
            ]
        )

        doi = client._extract_doi(result)
        assert doi == "10.1234/pub.doi"  # Should pick the highest priority

    def test_doi_extraction_handles_list_values(self):
        """Test that DOI extraction handles list values correctly."""
        from interaction_finder.fetcher.web_client import WebClient

        config = Mock()
        config.tools = Mock()
        config.tools.crawl4ai = Mock()
        config.tools.crawl4ai.timeout = 30

        client = WebClient(config, verbose=False)

        # Mock result where DOI field contains a list
        result = Mock()
        result.extracted_content = json.dumps(
            [{"doi_meta_cite": ["10.1234/first.doi", "10.1234/second.doi"]}]
        )

        doi = client._extract_doi(result)
        assert doi == "10.1234/first.doi"  # Should take first element

    def test_doi_extraction_handles_malformed_json(self):
        """Test that DOI extraction gracefully handles malformed JSON."""
        from interaction_finder.fetcher.web_client import WebClient

        config = Mock()
        config.tools = Mock()
        config.tools.crawl4ai = Mock()
        config.tools.crawl4ai.timeout = 30

        client = WebClient(config, verbose=False)

        # Mock result with malformed JSON
        result = Mock()
        result.extracted_content = "{ invalid json }"

        doi = client._extract_doi(result)
        assert doi == ""  # Should return empty string, not crash


class TestRedirectURLExtraction:
    """Test for redirect URL extraction bug found during refactoring."""

    def test_redirect_url_extraction_from_crawl4ai_results(self):
        """Test redirect URL extraction from crawl4ai _results structure."""
        from interaction_finder.fetcher.web_client import WebClient

        config = Mock()
        config.tools = Mock()
        config.tools.crawl4ai = Mock()
        config.tools.crawl4ai.timeout = 30

        client = WebClient(config, verbose=False)

        # Mock crawl4ai result with redirect info in _results
        result = Mock()
        first_result = Mock()
        first_result.redirected_url = "https://final.url/after/redirect"
        first_result.url = "https://intermediate.url"
        result._results = [first_result]

        final_url = client._extract_final_url(result, "https://original.url")
        assert final_url == "https://final.url/after/redirect"

    def test_redirect_url_extraction_without_redirect(self):
        """Test URL extraction when no redirect occurred."""
        from interaction_finder.fetcher.web_client import WebClient

        config = Mock()
        config.tools = Mock()
        config.tools.crawl4ai = Mock()
        config.tools.crawl4ai.timeout = 30

        client = WebClient(config, verbose=False)

        # Mock result with no redirect - need to mock both result.url and first_result.url
        result = Mock()
        result.url = "https://original.url"  # This gets checked at line 562

        class MockResult:
            def __init__(self):
                self.redirected_url = None
                self.url = "https://original.url"

        first_result = MockResult()
        result._results = [first_result]

        final_url = client._extract_final_url(result, "https://original.url")
        assert final_url == "https://original.url"

    def test_redirect_url_extraction_fallback_to_result_url(self):
        """Test URL extraction falls back to result.url when _results unavailable."""
        from interaction_finder.fetcher.web_client import WebClient

        config = Mock()
        config.tools = Mock()
        config.tools.crawl4ai = Mock()
        config.tools.crawl4ai.timeout = 30

        client = WebClient(config, verbose=False)

        # Mock result without _results but with url attribute
        result = Mock()
        result._results = None
        result.url = "https://fallback.url"

        final_url = client._extract_final_url(result, "https://original.url")
        assert final_url == "https://fallback.url"


class TestCLIBackwardCompatibility:
    """Test for CLI _fetch_multiple method compatibility."""

    @pytest.mark.asyncio
    async def test_fetch_multiple_method_exists(self):
        """Test that _fetch_multiple method exists for CLI compatibility."""
        config = Mock()
        config.output = Mock()
        config.output.cache = "test_cache"
        config.abspath = Mock(return_value=Path("test_cache"))
        config.tools = Mock()
        config.tools.crawl4ai = Mock()
        config.tools.crawl4ai.timeout = 30

        fetcher = PageFetcher(config, show_status=False)

        # This would have caught the missing method error
        assert hasattr(fetcher, "_fetch_multiple")
        assert callable(getattr(fetcher, "_fetch_multiple"))

    @pytest.mark.asyncio
    async def test_fetch_multiple_method_signature(self):
        """Test that _fetch_multiple has correct method signature."""
        config = Mock()
        config.output = Mock()
        config.output.cache = "test_cache"
        config.abspath = Mock(return_value=Path("test_cache"))
        config.tools = Mock()
        config.tools.crawl4ai = Mock()
        config.tools.crawl4ai.timeout = 30

        fetcher = PageFetcher(config, show_status=False)

        # Mock the batch_ops.fetch_multiple to avoid actual network calls
        from unittest.mock import AsyncMock

        fetcher.batch_ops.fetch_multiple = AsyncMock(return_value=["test_result"])

        # Test that we can call it with CLI parameters
        result = await fetcher._fetch_multiple(
            ["http://test.url"], "chunks", progress=True, fail_fast=False, retry=False
        )

        # Should delegate to batch_ops properly
        fetcher.batch_ops.fetch_multiple.assert_called_once()
        assert result == ["test_result"]


class TestLoggerInstantiation:
    """Test for GranularLogger abstract method implementation."""

    def test_granular_logger_instantiation(self):
        """Test that GranularLogger can be instantiated without abstract method errors."""
        # This would have caught the abstract method implementation issue
        logger = _get_granular_logger()
        assert logger is not None

        # Test that all required methods exist
        required_methods = [
            "debug",
            "info",
            "success",
            "warning",
            "error",
            "url_status",
        ]
        for method_name in required_methods:
            assert hasattr(logger, method_name)
            assert callable(getattr(logger, method_name))

    def test_quiet_logger_instantiation(self):
        """Test that QuietLogger continues to work correctly."""
        logger = _get_quiet_logger()
        assert logger is not None

        # Test that all required methods exist
        required_methods = [
            "debug",
            "info",
            "success",
            "warning",
            "error",
            "url_status",
        ]
        for method_name in required_methods:
            assert hasattr(logger, method_name)
            assert callable(getattr(logger, method_name))

    def test_logger_methods_dont_crash(self):
        """Test that logger methods can be called without crashing."""
        logger = _get_granular_logger()

        # These should all complete without errors
        logger.debug("test message")
        logger.info("test message")
        logger.success("test message")
        logger.warning("test message")
        logger.error("test message")
        logger.url_status("http://test.url", success=True, timing=1.0)


class TestModuleStructureIntegrity:
    """Test that the refactored module structure maintains expected interfaces."""

    def test_fetcher_module_exports(self):
        """Test that fetcher module exports all expected classes and functions."""
        from interaction_finder.fetcher import (
            PageFetcher,
            URLCache,
            PreviousFailure,
            fetch_urls_with_progress,
            fetch_urls_concurrent_with_progress,
            refine_article_content,
            extract_headings,
            classify_heading_relevance,
            url_to_hash,
            normalize_url,
            url_to_hash_base36,
        )

        # All these imports should succeed without ImportError
        assert PageFetcher is not None
        assert URLCache is not None
        assert PreviousFailure is not None
        assert fetch_urls_with_progress is not None
        assert fetch_urls_concurrent_with_progress is not None

    def test_page_fetcher_has_legacy_methods(self):
        """Test that PageFetcher maintains legacy method compatibility."""
        config = Mock()
        config.output = Mock()
        config.output.cache = "test_cache"
        config.abspath = Mock(return_value=Path("test_cache"))
        config.tools = Mock()
        config.tools.crawl4ai = Mock()
        config.tools.crawl4ai.timeout = 30

        fetcher = PageFetcher(config, show_status=False)

        # Test that legacy methods exist (these are used by existing code)
        legacy_methods = [
            "_extract_final_url",
            "_extract_doi",
            "_refine_article_content",
            "_is_pdf_url",
            "_create_chunks",
            "_fetch_multiple",
        ]

        for method_name in legacy_methods:
            assert hasattr(fetcher, method_name), (
                f"Missing legacy method: {method_name}"
            )
            assert callable(getattr(fetcher, method_name))
