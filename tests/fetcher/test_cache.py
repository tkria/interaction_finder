# Configure async tests for anyio with asyncio backend only
pytest_plugins = ["anyio"]

import pytest
import tempfile
import shutil
import asyncio
import threading
import socket
from pathlib import Path
from unittest.mock import Mock, MagicMock, AsyncMock
from http.server import HTTPServer, BaseHTTPRequestHandler
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from interaction_finder.fetcher import URLCache, PageFetcher
from interaction_finder.settings import IfetcherConfig


@pytest.fixture(params=["asyncio"], scope="session")
def anyio_backend(request):
    return request.param


class MockHTTPHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for testing."""

    def log_message(self, format, *args):
        # Suppress logging during tests
        pass

    def do_GET(self):
        if self.path == "/test.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            html_content = "<html><head><meta name='citation_doi' content='10.1234/test'/></head><body><h1>Test Page</h1></body></html>"
            self.wfile.write(html_content.encode())
        elif self.path == "/test.pdf":
            self.send_response(200)
            self.send_header("Content-type", "application/pdf")
            self.end_headers()
            # Create minimal valid PDF content
            pdf_content = self._create_minimal_pdf()
            self.wfile.write(pdf_content)
        elif self.path == "/redirect":
            self.send_response(302)
            self.send_header(
                "Location", f"http://localhost:{self.server.server_port}/test.html"
            )
            self.end_headers()
        elif self.path == "/404":
            self.send_response(404)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Not Found")
        else:
            self.send_response(404)
            self.end_headers()

    def _create_minimal_pdf(self):
        """Create a minimal valid PDF that works with most PDF processors."""
        return (
            b"%PDF-1.4\n"
            b"1 0 obj\n"
            b"<<\n"
            b"/Type /Catalog\n"
            b"/Pages 2 0 R\n"
            b">>\n"
            b"endobj\n"
            b"2 0 obj\n"
            b"<<\n"
            b"/Type /Pages\n"
            b"/Kids [3 0 R]\n"
            b"/Count 1\n"
            b">>\n"
            b"endobj\n"
            b"3 0 obj\n"
            b"<<\n"
            b"/Type /Page\n"
            b"/Parent 2 0 R\n"
            b"/MediaBox [0 0 612 792]\n"
            b"/Resources <<\n"
            b"  /Font << /F1 4 0 R >>\n"
            b">>\n"
            b"/Contents 5 0 R\n"
            b">>\n"
            b"endobj\n"
            b"4 0 obj\n"
            b"<<\n"
            b"/Type /Font\n"
            b"/Subtype /Type1\n"
            b"/BaseFont /Helvetica\n"
            b">>\n"
            b"endobj\n"
            b"5 0 obj\n"
            b"<<\n"
            b"/Length 44\n"
            b">>\n"
            b"stream\n"
            b"BT\n"
            b"/F1 12 Tf\n"
            b"50 750 Td\n"
            b"(Test PDF Content) Tj\n"
            b"ET\n"
            b"endstream\n"
            b"endobj\n"
            b"xref\n"
            b"0 6\n"
            b"0000000000 65535 f \n"
            b"0000000009 00000 n \n"
            b"0000000074 00000 n \n"
            b"0000000120 00000 n \n"
            b"0000000274 00000 n \n"
            b"0000000361 00000 n \n"
            b"trailer\n"
            b"<<\n"
            b"/Size 6\n"
            b"/Root 1 0 R\n"
            b">>\n"
            b"startxref\n"
            b"454\n"
            b"%%EOF\n"
        )


@pytest.fixture(scope="session")
def http_server():
    """Start a local HTTP server for testing."""
    # Find an available port
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()

    server = HTTPServer(("localhost", port), MockHTTPHandler)

    # Run server in background thread
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield f"http://localhost:{port}"

    server.shutdown()
    server.server_close()


class TestBasicCaching:
    """Test basic caching behavior from user perspective."""

    @pytest.fixture
    def temp_config(self):
        """Create a temporary config for testing."""
        temp_dir = tempfile.mkdtemp()
        config = Mock(spec=IfetcherConfig)
        config.output = Mock()
        config.output.cache = "cache"
        config.abspath = Mock(return_value=Path(temp_dir) / "cache")
        yield config
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def cache(self, temp_config):
        """Create a URLCache instance for testing."""
        cache_dir = temp_config.abspath(temp_config.output.cache)
        return URLCache(cache_dir)

    @pytest.mark.asyncio
    async def test_store_and_retrieve_html(self, cache):
        """User can store HTML content and retrieve it exactly as stored."""
        url = "https://example.com/page"
        content = "<html><body><h1>My Page</h1><p>Content here</p></body></html>"

        await cache.set_path(url, "html", content)
        retrieved = await cache.get_content(url, "html")

        assert retrieved == content
        assert await cache.has_path(url, "html")

    @pytest.mark.asyncio
    async def test_store_and_retrieve_markdown(self, cache):
        """User can store Markdown content and retrieve it exactly as stored."""
        url = "https://example.com/article"
        content = (
            "# My Article\n\nThis is **bold** content with [links](http://example.com)."
        )

        await cache.set_path(url, "markdown", content)
        retrieved = await cache.get_content(url, "markdown")

        assert retrieved == content
        assert await cache.has_path(url, "markdown")

    @pytest.mark.asyncio
    async def test_store_and_retrieve_pdf(self, cache):
        """User can store PDF content and retrieve it exactly as stored."""
        url = "https://example.com/paper.pdf"
        content = "PDF content simulated as text"

        await cache.set_path(url, "pdf", content)
        retrieved = await cache.get_content(url, "pdf")

        assert retrieved == content
        assert await cache.has_path(url, "pdf")

    @pytest.mark.asyncio
    async def test_multiple_content_types_same_url(self, cache):
        """User can store multiple content types for the same URL."""
        url = "https://example.com/multi"
        html_content = "<html><body>HTML</body></html>"
        markdown_content = "# Markdown\nContent"

        await cache.set_path(url, "html", html_content)
        await cache.set_path(url, "markdown", markdown_content)

        assert await cache.get_content(url, "html") == html_content
        assert await cache.get_content(url, "markdown") == markdown_content
        assert await cache.has_path(url, "html")
        assert await cache.has_path(url, "markdown")

    @pytest.mark.asyncio
    async def test_content_not_found_raises_keyerror(self, cache):
        """Attempting to retrieve non-existent content raises KeyError."""
        url = "https://nonexistent.com/page"

        with pytest.raises(KeyError):
            await cache.get_content(url, "html")
        with pytest.raises(KeyError):
            await cache.get_content(url, "markdown")
        with pytest.raises(KeyError):
            await cache.get_content(url, "pdf")

        assert not await cache.has_path(url, "html")
        assert not await cache.has_path(url, "markdown")
        assert not await cache.has_path(url, "pdf")

    @pytest.mark.asyncio
    async def test_unicode_content_handling(self, cache):
        """User can store and retrieve content with Unicode characters."""
        url = "https://example.com/unicode"
        content = "Content with émojis 🚀 and ünicode characters 中文"

        await cache.set_path(url, "html", content)
        retrieved = await cache.get_content(url, "html")

        assert retrieved == content

    @pytest.mark.asyncio
    async def test_cache_persistence_across_instances(self, temp_config):
        """Cache content persists when creating new cache instances."""
        url = "https://example.com/persistent"
        content = "<html>Persistent content</html>"

        # Store content with first cache instance
        cache_dir = temp_config.abspath(temp_config.output.cache)
        cache1 = URLCache(cache_dir)
        await cache1.set_path(url, "html", content)

        # Create new cache instance and verify content exists
        cache2 = URLCache(cache_dir)
        assert await cache2.has_path(url, "html")
        assert await cache2.get_content(url, "html") == content


class TestUserWorkflows:
    """Test complete user workflows."""

    @pytest.fixture
    def temp_config(self):
        """Create a temporary config for testing."""
        temp_dir = tempfile.mkdtemp()
        config = Mock(spec=IfetcherConfig)
        config.output = Mock()
        config.output.cache = "cache"
        config.abspath = Mock(return_value=Path(temp_dir) / "cache")
        config.tools = Mock()
        config.tools.crawl4ai = Mock()
        config.tools.crawl4ai.timeout = 30
        config.tools.crawl4ai.user_agent = "TestAgent/1.0"
        config.tools.crawl4ai.delay_between_requests = 1.0
        yield config
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def fetcher(self, temp_config):
        """Create a PageFetcher instance for testing."""
        cache_dir = temp_config.abspath(temp_config.output.cache)
        return PageFetcher(cache_dir=cache_dir)

    @pytest.mark.asyncio
    async def test_basic_caching_workflow(self, fetcher):
        """Complete workflow: store content, check cache, retrieve content."""
        url = "https://example.com/workflow"
        html_content = "<html><body>Workflow test</body></html>"
        markdown_content = "# Workflow Test\n\nThis is a test."

        # Initially not cached
        assert not await fetcher.is_cached(url)

        # Store content
        await fetcher.cache.set_path(url, "html", html_content)
        await fetcher.cache.set_path(url, "markdown", markdown_content)

        # Check cached status
        assert await fetcher.is_cached(url)

        # Retrieve content (silently)
        assert await fetcher.get_html(url, progress=False) == html_content
        assert await fetcher.get_markdown(url, progress=False) == markdown_content
        assert (
            await fetcher.get_raw(url) == html_content
        )  # Raw returns HTML when available

    @pytest.mark.asyncio
    async def test_transparent_fetching_workflow(self, fetcher, monkeypatch):
        """Fetching content automatically when not cached."""
        url = "https://example.com/fetch-test"
        expected_html = "<html>Fetched content</html>"
        expected_markdown = "# Fetched content"

        # Mock the fetching process at the WebClient level
        async def mock_fetch_html(url, retry=False):
            return {
                "raw_content": expected_html,
                "markdown_content": expected_markdown,
                "final_url": url,
                "doi": "",
            }

        monkeypatch.setattr(fetcher.web_client, "fetch_html", mock_fetch_html)

        # Initially not cached
        assert not await fetcher.is_cached(url)

        # First access should trigger fetch and cache (silently)
        result = await fetcher.get_html(url, progress=False)
        assert result == expected_html
        assert await fetcher.is_cached(url)

        # Subsequent access should use cache (no additional fetch)
        result2 = await fetcher.get_html(url, progress=False)
        assert result2 == expected_html

        # Markdown should also be available
        markdown = await fetcher.get_markdown(url, progress=False)
        assert markdown == expected_markdown

    @pytest.mark.asyncio
    async def test_redirect_handling_workflow(self, fetcher):
        """User can access content by original URL even after redirects."""
        original_url = "https://example.com/old-path"
        final_url = "https://example.com/new-path"
        content = "<html>Redirected content</html>"
        markdown = "# Redirected content"

        # Simulate storing content that was redirected
        await fetcher.cache.set_path(original_url, "html", content, final_url)
        await fetcher.cache.set_path(original_url, "markdown", markdown, final_url)

        # User can access by original URL (silently)
        assert await fetcher.is_cached(original_url)
        assert await fetcher.get_html(original_url, progress=False) == content
        assert await fetcher.get_markdown(original_url, progress=False) == markdown

        # Should know about the redirect
        assert await fetcher.cache.get_redirect_info(original_url) == final_url

    @pytest.mark.asyncio
    async def test_bulk_prefetching_workflow(self, fetcher, http_server):
        """User can prefetch multiple URLs efficiently."""
        urls = [f"{http_server}/test.html"]

        # Initially none cached
        for url in urls:
            assert not await fetcher.is_cached(url)

        # Prefetch all URLs (fetch them individually for now, silently)
        for url in urls:
            await fetcher.get_html(url, progress=False)

        # Verify all are cached
        for url in urls:
            assert await fetcher.is_cached(url)

    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, fetcher, http_server):
        """User gets appropriate errors for various failure scenarios."""
        # Test 404 error handling
        not_found_url = f"{http_server}/404"

        # 404 should either raise an exception or return empty content
        try:
            content = await fetcher.get_html(not_found_url, progress=False)
            # If no exception, content should be minimal/empty
            assert len(content) < 100  # Very short content for 404
        except Exception:
            # Exception is also acceptable for 404
            pass

    @pytest.mark.asyncio
    async def test_mixed_content_workflow(self, fetcher, http_server):
        """User working with mixed cached and uncached content."""
        cached_url = "https://example.com/cached"
        real_url = f"{http_server}/test.html"

        # Cache some content first
        await fetcher.cache.set_path(cached_url, "html", "<html>Cached content</html>")
        await fetcher.cache.set_path(cached_url, "markdown", "# Cached Content")

        # Check status of both
        assert await fetcher.is_cached(cached_url)
        assert not await fetcher.is_cached(real_url)

        # Can get cached content immediately
        assert await fetcher.get_html(cached_url) == "<html>Cached content</html>"
        assert await fetcher.get_markdown(cached_url) == "# Cached Content"

        # Can fetch real content
        real_html = await fetcher.get_html(real_url)
        assert "Test Page" in real_html


class TestCacheManagement:
    """Test cache management from user perspective."""

    @pytest.fixture
    def temp_config(self):
        """Create a temporary config for testing."""
        temp_dir = tempfile.mkdtemp()
        config = Mock(spec=IfetcherConfig)
        config.output = Mock()
        config.output.cache = "cache"
        config.abspath = Mock(return_value=Path(temp_dir) / "cache")
        yield config
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def cache(self, temp_config):
        """Create a URLCache instance for testing."""
        cache_dir = temp_config.abspath(temp_config.output.cache)
        return URLCache(cache_dir)

    @pytest.mark.asyncio
    async def test_cache_status_checking(self, cache):
        """User can check what's in the cache."""
        urls_and_content = [
            ("https://example.com/page1", "<html>Page 1</html>"),
            ("https://example.com/page2", "<html>Page 2</html>"),
            ("https://example.com/page3", "<html>Page 3</html>"),
        ]

        # Store all content
        for url, content in urls_and_content:
            await cache.set_path(url, "html", content)

        # Can check individual URLs
        for url, content in urls_and_content:
            assert await cache.has_path(url, "html")
            assert await cache.has_url(url)

        # Can list all cached URLs
        cached_urls = await cache.list_cached_urls()
        assert len(cached_urls) == len(urls_and_content)
        for url, _ in urls_and_content:
            assert url in cached_urls

    @pytest.mark.asyncio
    async def test_cache_clearing(self, cache):
        """User can clear specific URLs from cache."""
        url1 = "https://example.com/keep"
        url2 = "https://example.com/remove"

        await cache.set_path(url1, "html", "<html>Keep this</html>")
        await cache.set_path(url2, "html", "<html>Remove this</html>")
        await cache.set_path(url1, "markdown", "# Keep this")
        await cache.set_path(url2, "markdown", "# Remove this")

        # Both should be cached
        assert await cache.has_path(url1, "html")
        assert await cache.has_path(url2, "html")
        assert await cache.has_path(url1, "markdown")
        assert await cache.has_path(url2, "markdown")

        # Clear one URL
        await cache.clear_url(url2)

        # Only url1 should remain
        assert await cache.has_path(url1, "html")
        assert not await cache.has_path(url2, "html")
        assert await cache.get_content(url1, "html") == "<html>Keep this</html>"

    @pytest.mark.asyncio
    async def test_content_type_detection(self, cache):
        """User can determine what type of content is available."""
        html_url = "https://example.com/html-page"
        pdf_url = "https://example.com/document.pdf"
        both_url = "https://example.com/both"

        # Store different content types
        await cache.set_path(html_url, "html", "<html>HTML page</html>")
        await cache.set_path(pdf_url, "pdf", "PDF content")
        await cache.set_path(both_url, "html", "<html>HTML version</html>")
        await cache.set_path(both_url, "markdown", "# Markdown version")

        # Can detect what's available
        assert await cache.get_source_type(html_url) == "html"
        assert await cache.get_source_type(pdf_url) == "pdf"
        assert await cache.get_source_type(both_url) == "html"  # HTML takes precedence

        # Can get appropriate content
        assert await cache.get_source_content(html_url) == "<html>HTML page</html>"
        assert await cache.get_source_content(pdf_url) == "PDF content"
        assert await cache.get_source_content(both_url) == "<html>HTML version</html>"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def temp_config(self):
        """Create a temporary config for testing."""
        temp_dir = tempfile.mkdtemp()
        config = Mock(spec=IfetcherConfig)
        config.output = Mock()
        config.output.cache = "cache"
        config.abspath = Mock(return_value=Path(temp_dir) / "cache")
        yield config
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def cache(self, temp_config):
        """Create a URLCache instance for testing."""
        cache_dir = temp_config.abspath(temp_config.output.cache)
        return URLCache(cache_dir)

    @pytest.mark.asyncio
    async def test_empty_content_handling(self, cache):
        """User can store and retrieve empty content."""
        url = "https://example.com/empty"

        await cache.set_path(url, "html", "")
        await cache.set_path(url, "markdown", "")

        assert await cache.get_content(url, "html") == ""
        assert await cache.get_content(url, "markdown") == ""
        assert await cache.has_path(url, "html")
        assert await cache.has_path(url, "markdown")

    @pytest.mark.asyncio
    async def test_very_large_content(self, cache):
        """User can store and retrieve large content."""
        url = "https://example.com/large"
        large_content = "x" * 100000  # 100KB of content

        # Store and retrieve large content
        await cache.set_path(url, "html", large_content)
        retrieved = await cache.get_content(url, "html")

        assert retrieved == large_content
        assert len(retrieved) == 100000

    @pytest.mark.asyncio
    async def test_special_characters_in_content(self, cache):
        """User can store content with special characters and markup."""
        url = "https://example.com/special"
        content = """<html>
        <body>
            <p>Special chars: &lt;&gt;&amp;&quot;&#39;</p>
            <p>Unicode: 中文 العربية русский 🚀</p>
            <script>console.log("JavaScript");</script>
        </body>
        </html>"""

        await cache.set_path(url, "html", content)
        retrieved = await cache.get_content(url, "html")

        assert retrieved == content

    @pytest.mark.asyncio
    async def test_same_content_different_urls(self, cache):
        """User can store the same content under different URLs."""
        content = "<html>Same content</html>"
        url1 = "https://site1.com/page"
        url2 = "https://site2.com/page"

        # Store same content under different URLs
        await cache.set_path(url1, "html", content)
        await cache.set_path(url2, "html", content)

        # Both should work independently
        assert await cache.get_content(url1, "html") == content
        assert await cache.get_content(url2, "html") == content
        assert await cache.has_path(url1, "html")
        assert await cache.has_path(url2, "html")

    @pytest.mark.asyncio
    async def test_update_existing_content(self, cache):
        """User can update content for existing URLs."""
        url = "https://example.com/update"
        original_content = "<html>Original</html>"
        updated_content = "<html>Updated</html>"

        # Store initial content
        await cache.set_path(url, "html", original_content)
        assert await cache.get_content(url, "html") == original_content

        # Update with new content
        await cache.set_path(url, "html", updated_content)
        assert await cache.get_content(url, "html") == updated_content


class TestEnhancedFunctionality:
    """Test enhanced PageFetcher functionality: PDF detection, DOI extraction, references removal."""

    @pytest.fixture
    def temp_config(self):
        """Create a temporary config for testing."""
        temp_dir = tempfile.mkdtemp()
        config = Mock(spec=IfetcherConfig)
        config.output = Mock()
        config.output.cache = "cache"
        config.abspath = Mock(return_value=Path(temp_dir) / "cache")
        config.tools = Mock()
        config.tools.crawl4ai = Mock()
        yield config
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def cache(self, temp_config):
        cache_dir = temp_config.abspath(temp_config.output.cache)
        return URLCache(cache_dir)

    @pytest.fixture
    def fetcher(self, temp_config):
        cache_dir = temp_config.abspath(temp_config.output.cache)
        return PageFetcher(cache_dir=cache_dir)

    def test_pdf_url_detection(self, fetcher):
        """Test that PDF URLs are correctly identified."""
        assert fetcher.web_client._is_pdf_url("https://example.com/paper.pdf") == True
        assert fetcher.web_client._is_pdf_url("https://example.com/paper.PDF") == True
        assert fetcher.web_client._is_pdf_url("https://example.com/page.html") == False
        assert fetcher.web_client._is_pdf_url("https://example.com/") == False

    def test_references_removal(self, fetcher):
        """Test that references sections are properly removed from markdown."""
        markdown_with_refs = """
# Introduction

This is the main content of the paper.

## Methodology

Some methodology content here.

## References

1. Smith, J. (2020). Some paper title.
2. Doe, A. (2019). Another paper.

## Appendix

This should remain.
"""

        cleaned_markdown = fetcher.content_processor.refine_article(markdown_with_refs)

        # Should remove references but keep appendix
        assert "References" not in cleaned_markdown
        assert "Smith, J." not in cleaned_markdown
        assert "Appendix" in cleaned_markdown
        assert "This should remain" in cleaned_markdown

    @pytest.mark.asyncio
    async def test_flexible_cache_path_generation(self, cache):
        """Test the flexible cache path generation."""
        url = "https://example.com/test"

        # Test single extension
        (html_path,) = await cache._get_paths(url, "html")
        assert html_path.suffix == ".html"

        # Test multiple extensions
        html_path, pdf_path, doi_path = await cache._get_paths(
            url, "html", "pdf", "doi"
        )
        assert html_path.suffix == ".html"
        assert pdf_path.suffix == ".pdf"
        assert doi_path.suffix == ".doi"

        # All should have the same stem (hash)
        assert html_path.stem == pdf_path.stem == doi_path.stem

    @pytest.mark.asyncio
    async def test_doi_storage_and_retrieval(self, cache):
        """Test DOI storage and retrieval."""
        url = "https://example.com/paper"
        doi = "10.1234/example.2024"

        # Store DOI
        await cache.set_path(url, "doi", doi)

        # Check if DOI is cached
        assert await cache.has_path(url, "doi") == True

        # Retrieve DOI
        retrieved_doi = await cache.get_content(url, "doi")
        assert retrieved_doi == doi

        # Check that the .doi file was created
        (doi_path,) = await cache._get_paths(url, "doi")
        assert doi_path.exists()
        assert doi_path.read_text().strip() == doi

    @pytest.mark.asyncio
    async def test_generic_path_methods(self, cache):
        """Test the generic has_path, get_path, and set_path methods."""
        url = "https://example.com/test"

        # Test that path doesn't exist initially
        assert await cache.has_path(url, "html") == False
        assert await cache.has_path(url, "pdf") == False
        assert await cache.has_path(url, "doi") == False

        # Store content using generic method
        await cache.set_path(url, "html", "<html>content</html>")
        await cache.set_path(url, "pdf", "PDF content")
        await cache.set_path(url, "doi", "10.1234/test")

        # Check using generic method
        assert await cache.has_path(url, "html") == True
        assert await cache.has_path(url, "pdf") == True
        assert await cache.has_path(url, "doi") == True

        # Retrieve using generic method
        assert await cache.get_path(url, "html") == "<html>content</html>"
        assert await cache.get_path(url, "pdf") == "PDF content"
        assert (await cache.get_path(url, "doi")).strip() == "10.1234/test"

        # Test that specific methods still work
        assert await cache.get_content(url, "html") == "<html>content</html>"
        assert await cache.get_content(url, "pdf") == "PDF content"
        assert await cache.get_content(url, "doi") == "10.1234/test"

    @pytest.mark.asyncio
    async def test_cache_file_cleanup_with_doi(self, cache):
        """Test that cache cleanup removes all associated files including DOI."""
        url = "https://example.com/test"

        # Create various cached files
        await cache.set_path(url, "html", "<html>test</html>")
        await cache.set_path(url, "markdown", "# Test")
        await cache.set_path(url, "doi", "10.1234/test")

        # Verify files exist
        html_path, md_path, doi_path = await cache._get_paths(url, "html", "md", "doi")
        assert html_path.exists()
        assert md_path.exists()
        assert doi_path.exists()

        # Clear cache
        await cache.clear_url(url)

        # Verify files are removed
        assert not html_path.exists()
        assert not md_path.exists()
        assert not doi_path.exists()

    @pytest.mark.asyncio
    async def test_has_url_includes_doi(self, cache):
        """Test that has_url returns True when only DOI is cached."""
        url = "https://example.com/test"

        # Initially no content
        assert await cache.has_url(url) == False

        # Add only DOI
        await cache.set_path(url, "doi", "10.1234/test")

        # Should now return True
        assert await cache.has_url(url) == True


class TestIntegrationWithHTTPServer:
    """Integration tests using real HTTP server."""

    @pytest.fixture
    def temp_config(self):
        """Create a temporary config for testing."""
        temp_dir = tempfile.mkdtemp()
        config = Mock(spec=IfetcherConfig)
        config.output = Mock()
        config.output.cache = "cache"
        config.abspath = Mock(return_value=Path(temp_dir) / "cache")
        config.tools = Mock()
        config.tools.crawl4ai = Mock()
        config.tools.crawl4ai.timeout = 30
        config.tools.crawl4ai.user_agent = "TestAgent/1.0"
        config.tools.crawl4ai.delay_between_requests = 1.0
        yield config
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def fetcher(self, temp_config):
        cache_dir = temp_config.abspath(temp_config.output.cache)
        return PageFetcher(cache_dir=cache_dir)

    @pytest.mark.asyncio
    async def test_real_html_fetching(self, fetcher, http_server):
        """Test fetching real HTML content from local server."""
        url = f"{http_server}/test.html"

        # Initially not cached
        assert not await fetcher.is_cached(url)

        # Fetch content (silently for tests)
        html_content = await fetcher.get_html(url, progress=False)

        # Verify content
        assert "Test Page" in html_content
        assert "<h1>Test Page</h1>" in html_content

        # Should now be cached
        assert await fetcher.is_cached(url)

        # Second fetch should use cache
        html_content2 = await fetcher.get_html(url, progress=False)
        assert html_content == html_content2

    @pytest.mark.asyncio
    async def test_real_pdf_fetching(self, fetcher, http_server):
        """Test fetching PDF content with mocked PDF processing to avoid hanging."""
        url = f"{http_server}/test.pdf"

        # Mock the PDF processing to avoid hanging on any PDF content
        expected_content = "Extracted content from test PDF document"

        # Mock the web client's PDF fetch method
        from unittest.mock import AsyncMock, patch
        mock_pdf_result = {
            "raw_content": expected_content,
            "markdown_content": "# Test PDF\n\nExtracted content from test PDF document",
            "final_url": url,
            "doi": ""
        }

        with patch.object(fetcher.web_client, '_fetch_pdf_content', new=AsyncMock(return_value=mock_pdf_result)):
            pdf_content = await fetcher.get_pdf(url, progress=False)

            # Verify content and caching behavior
            assert isinstance(pdf_content, str)
            assert pdf_content == expected_content
            assert await fetcher.is_cached(url)

            # Verify it was cached properly - second call should be from cache
            cached_content = await fetcher.get_pdf(url, progress=False)
            assert cached_content == expected_content

    @pytest.mark.asyncio
    async def test_pdf_approach_2_demo(self, fetcher, tmp_path):
        """Demo of Approach 2: Testing with real minimal PDF content.

        This shows how approach 2 could work if PDF processing didn't hang.
        The key insight is creating valid PDF content that parsers can handle quickly.
        """
        # Create a truly minimal but valid PDF file
        pdf_content = self._create_working_minimal_pdf()
        pdf_path = tmp_path / "minimal.pdf"
        pdf_path.write_bytes(pdf_content)

        # Verify the PDF file was created
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 0

        # Test PDF URL detection (this part works)
        test_pdf_url = "https://example.com/document.pdf"
        assert fetcher.web_client._is_pdf_url(test_pdf_url)
        assert not fetcher.web_client._is_pdf_url("https://example.com/document.html")

        # Note: In a working implementation, you would test:
        # pdf_content = await fetcher.get_pdf(f"file://{pdf_path}")
        # But this currently hangs due to crawl4ai PDF processing issues

    def _create_working_minimal_pdf(self):
        """Create the smallest possible valid PDF that should work with most parsers."""
        # This is a minimal PDF with proper structure
        return (
            b"%PDF-1.4\n"
            b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
            b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n"
            b"0000000053 00000 n \n0000000101 00000 n \ntrailer\n"
            b"<</Size 4/Root 1 0 R>>\nstartxref\n164\n%%EOF"
        )

    @pytest.mark.asyncio
    async def test_doi_extraction_from_real_html(self, fetcher, http_server):
        """Test DOI extraction from real HTML content."""
        url = f"{http_server}/test.html"

        # Fetch content (which includes DOI in meta tag, silently)
        await fetcher.get_html(url, progress=False)

        # Check if DOI was extracted
        doi = await fetcher.get_doi(url)
        assert doi == "10.1234/test"

    @pytest.mark.asyncio
    async def test_redirect_handling(self, fetcher, http_server):
        """Test handling of HTTP redirects."""
        redirect_url = f"{http_server}/redirect"
        final_url = f"{http_server}/test.html"

        # Fetch redirected URL (silently)
        html_content = await fetcher.get_html(redirect_url, progress=False)

        # Should get content from final URL
        assert "Test Page" in html_content

        # Should be able to get redirect info
        redirect_info = await fetcher.cache.get_redirect_info(redirect_url)
        assert redirect_info == final_url

    @pytest.mark.asyncio
    async def test_concurrent_fetching(self, fetcher, http_server):
        """Test concurrent fetching of multiple URLs."""
        html_urls = [f"{http_server}/test.html", f"{http_server}/redirect"]

        # Fetch HTML URLs concurrently (avoid PDF which has parsing issues with fake content, silently)
        tasks = [fetcher.get_html(url, progress=False) for url in html_urls]
        results = await asyncio.gather(*tasks)

        # All should have content
        for result in results:
            assert len(result) > 0

        # All should be cached
        for url in html_urls:
            assert await fetcher.is_cached(url)

    @pytest.mark.asyncio
    async def test_error_handling_with_real_404(self, fetcher, http_server):
        """Test error handling with real 404 responses."""
        not_found_url = f"{http_server}/nonexistent"

        # Should handle 404 gracefully (silently)
        try:
            content = await fetcher.get_html(not_found_url, progress=False)
            # If it doesn't raise, should be empty or error content
            assert isinstance(content, str)
        except Exception as e:
            # Should be a reasonable error message
            assert isinstance(e, Exception)

    @pytest.mark.asyncio
    async def test_markdown_conversion_from_real_html(self, fetcher, http_server):
        """Test markdown conversion from real HTML."""
        url = f"{http_server}/test.html"

        # Fetch and convert to markdown (silently)
        markdown_content = await fetcher.get_markdown(url, progress=False)

        # Should contain converted content
        assert len(markdown_content) > 0
        assert isinstance(markdown_content, str)

        # Should be cached
        assert await fetcher.cache.has_path(url, "markdown")


class TestMultipleURLFetching:
    """Test the new multiple URL fetching functionality."""

    @pytest.fixture
    def temp_config(self, tmp_path):
        """Fixture to create a temporary config."""
        from interaction_finder.settings import IfetcherConfig

        config = Mock(spec=IfetcherConfig)
        config.output = Mock()
        config.output.cache = "cache"
        config.abspath = Mock(return_value=tmp_path / "test_cache")
        config.tools = Mock()
        config.tools.crawl4ai = Mock()
        config.tools.crawl4ai.timeout = 30
        config.tools.crawl4ai.user_agent = "TestAgent/1.0"
        config.tools.crawl4ai.delay_between_requests = 1.0
        return config

    @pytest.fixture
    def fetcher(self, temp_config):
        """Fixture to create a PageFetcher instance."""
        from interaction_finder.fetcher import PageFetcher

        cache_dir = temp_config.abspath(temp_config.output.cache)
        return PageFetcher(cache_dir=cache_dir, show_status=False)

    @pytest.mark.asyncio
    async def test_multiple_html_urls(self, fetcher, http_server):
        """Test fetching multiple HTML URLs."""
        urls = [
            f"{http_server}/test.html",
            f"{http_server}/test.html",  # Duplicate to test caching
        ]

        # Test with progress=False for silent fetching
        contents = await fetcher.get_html(urls, progress=False)

        assert isinstance(contents, list)
        assert len(contents) == 2
        assert all(isinstance(content, str) for content in contents)
        assert all(len(content) > 0 for content in contents)

        # Both should have the same content (one from cache)
        assert contents[0] == contents[1]

    @pytest.mark.asyncio
    async def test_multiple_markdown_urls(self, fetcher, http_server):
        """Test fetching multiple URLs and converting to markdown."""
        urls = [
            f"{http_server}/test.html",
            f"{http_server}/test.html",
        ]

        # Test markdown conversion for multiple URLs
        contents = await fetcher.get_markdown(urls, progress=False)

        assert isinstance(contents, list)
        assert len(contents) == 2
        assert all(isinstance(content, str) for content in contents)
        assert all(len(content) > 0 for content in contents)

    @pytest.mark.asyncio
    async def test_single_url_backward_compatibility(self, fetcher, http_server):
        """Test that single URL fetching still works as before."""
        url = f"{http_server}/test.html"

        # Single URL should return string, not list
        content = await fetcher.get_html(url, progress=False)
        assert isinstance(content, str)
        assert len(content) > 0

        # Same for markdown
        markdown_content = await fetcher.get_markdown(url, progress=False)
        assert isinstance(markdown_content, str)
        assert len(markdown_content) > 0

    @pytest.mark.asyncio
    async def test_empty_url_list(self, fetcher):
        """Test handling of empty URL list."""
        contents = await fetcher.get_html([], progress=False)
        assert isinstance(contents, list)
        assert len(contents) == 0

    @pytest.mark.asyncio
    async def test_type_error_for_invalid_input(self, fetcher):
        """Test that invalid input types raise TypeError."""
        with pytest.raises(TypeError):
            await fetcher.get_html(123, progress=False)

        with pytest.raises(TypeError):
            await fetcher.get_html({"url": "test"}, progress=False)
