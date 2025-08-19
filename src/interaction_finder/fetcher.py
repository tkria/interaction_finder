"""
Enhanced PageFetcher with intelligent URL handling and granular status updates.

Features:
- Single URL and batch URL fetching with automatic concurrency
- Granular status updates showing detailed crawl4ai stages
- Automatic PDF vs HTML detection based on URL extension
- DOI extraction from HTML pages using XPath selectors
- References section removal from markdown content
- Efficient lazy imports for faster startup
- Progress bars for batch operations, detailed status for single URLs
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Union, List
import aiofiles
import aiofiles.os
from rich.console import Console

if TYPE_CHECKING:
    from .settings import IfetcherConfig

_crawl4ai_html_imports = None
_crawl4ai_pdf_imports = None
_quiet_logger = None

def _get_crawl4ai_imports():
    """Lazy import crawl4ai modules with simple caching."""
    global _crawl4ai_html_imports
    if _crawl4ai_html_imports is None:
        try:
            from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
            from crawl4ai.extraction_strategy import JsonXPathExtractionStrategy
            from crawl4ai.async_logger import AsyncLoggerBase
            _crawl4ai_html_imports = (AsyncWebCrawler, CrawlerRunConfig, JsonXPathExtractionStrategy, AsyncLoggerBase)
        except ImportError:
            raise RuntimeError("crawl4ai package is required for fetching. Install with: pip install crawl4ai")
    return _crawl4ai_html_imports

def _get_crawl4ai_pdf_imports():
    """Lazy import crawl4ai PDF modules with simple caching."""
    global _crawl4ai_pdf_imports
    if _crawl4ai_pdf_imports is None:
        try:
            from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
            from crawl4ai.processors.pdf import PDFCrawlerStrategy, PDFContentScrapingStrategy
            from crawl4ai.async_logger import AsyncLoggerBase
            _crawl4ai_pdf_imports = (AsyncWebCrawler, CrawlerRunConfig, PDFCrawlerStrategy, PDFContentScrapingStrategy, AsyncLoggerBase)
        except ImportError:
            raise RuntimeError("crawl4ai package is required for fetching. Install with: pip install crawl4ai")
    return _crawl4ai_pdf_imports

def _get_granular_logger(status_display=None):
    """
    Create a logger that provides granular stage information to Rich status displays.

    Args:
        status_display: Rich status object to update with progress information

    Returns:
        GranularLogger instance that updates the status display with crawl4ai stages
    """
    _, _, _, AsyncLoggerBase = _get_crawl4ai_imports()

    class GranularLogger(AsyncLoggerBase):
        """Logger that translates crawl4ai stages into Rich status updates."""

        def __init__(self, status_display=None):
            self.status_display = status_display
        def debug(self, message: str, tag: str = "DEBUG", **kwargs): pass
        def info(self, message: str, tag: str = "INFO", **kwargs): pass
        def success(self, message: str, tag: str = "SUCCESS", **kwargs): pass
        def warning(self, message: str, tag: str = "WARNING", **kwargs): pass
        def error(self, message: str, tag: str = "ERROR", **kwargs): pass
        def error_status(self, url: str, error: str, tag: str = "ERROR", url_length: int = 100): pass

        def url_status(self, url: str, success: bool, timing: float,
                       tag: str = "FETCH", url_length: int = 100) -> None:
            if self.status_display is None:
                return

            domain = url.split('//')[1].split('/')[0] if '//' in url else url[:20]

            stage_messages = {
                "FETCH": f"[cyan]Fetching from[/cyan] [bold]{domain}[/bold]",
                "SCRAPE": f"[blue]Processing content from[/blue] [bold]{domain}[/bold]",
                "EXTRACT": f"[yellow]Extracting data from[/yellow] [bold]{domain}[/bold]",
                "COMPLETE": f"[green]✓ Completed[/green] [bold]{domain}[/bold] [dim]({timing:.1f}s)[/dim]"
            }

            if tag in stage_messages and hasattr(self.status_display, 'update'):
                self.status_display.update(stage_messages[tag])

    return GranularLogger(status_display)

def _get_quiet_logger():
    """Get a silent logger for crawl4ai when no status updates are needed."""
    global _quiet_logger
    if _quiet_logger is None:
        _quiet_logger = _get_granular_logger(status_display=None)
    return _quiet_logger

class StatusDisplay:
    """Manages Rich status indicators for fetch operations."""

    def __init__(self, show_status: bool = True):
        self.show_status = show_status
        self.console = Console() if show_status else None

    def create_status(self, message: str, progress_info=None):
        """
        Create appropriate status indicator based on context.

        Args:
            message: Initial status message to display
            progress_info: Optional tuple of (progress_instance, task_id) for progress bars

        Returns:
            Rich status object or DummyStatus for no-op
        """
        if progress_info is not None:
            progress_instance, task_id = progress_info
            progress_instance.update(task_id, description=message)
            return DummyStatus()
        elif self.show_status and self.console:
            return self.console.status(message, spinner="dots")
        else:
            return DummyStatus()

class DummyStatus:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def update(self, text: str):
        pass
def url_to_hash(url: str) -> str:
    """Convert URL to a filesystem-safe hash (backward compatibility)."""
    return hashlib.sha256(url.encode('utf-8')).hexdigest()[:16]

def url_to_hash_base36(url: str) -> str:
    """Convert URL to a base36 hash for compact representation."""
    hash_int = hash(url) & ((1 << 63) - 1)  # Make positive
    # Manual base36 conversion
    if hash_int == 0:
        return "0"

    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    result = ""
    while hash_int:
        result = digits[hash_int % 36] + result
        hash_int //= 36
    return result

class URLCache:
    """File-based cache for URLs with support for HTML, PDF, and Markdown content."""

    def __init__(self, config: IfetcherConfig):
        self.config = config
        self.base_path = config.abspath(config.output.cache)
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def _get_paths(self, url: str, *extensions: str) -> tuple[Path, ...]:
        """Get file paths for specific extensions with collision resolution."""
        base_hash = url_to_hash_base36(url)
        probe = 0

        while probe < 1000:  # Safety limit
            if probe == 0:
                hash_str = base_hash
            else:
                hash_str = f"{base_hash}_{probe}"

            url_path = self.base_path / f"{hash_str}.url"

            # If no URL file exists, this slot is available
            if not url_path.exists():
                return tuple(self.base_path / f"{hash_str}.{ext}" for ext in extensions)

            # URL file exists, check if it's for the same URL
            try:
                async with aiofiles.open(url_path, 'r', encoding='utf-8') as f:
                    stored_url = (await f.read()).strip()
                if stored_url == url:
                    # Found existing entry for this URL
                    return tuple(self.base_path / f"{hash_str}.{ext}" for ext in extensions)
            except (OSError, UnicodeDecodeError):
                # Corrupted file, skip this slot
                pass

            probe += 1

        raise RuntimeError(f"Too many hash collisions for URL: {url}")

    async def _store_url_mapping(self, url: str, final_url: Optional[str] = None) -> None:
        """Store the URL mapping and redirect info in sidecar files with atomic operations."""
        url_path, redir_path = await self._get_paths(url, "url", "redir")

        # Atomic file creation: write to temp file then rename
        temp_url_path = url_path.with_suffix('.url.tmp')
        try:
            async with aiofiles.open(temp_url_path, 'w', encoding='utf-8') as f:
                await f.write(url)
            # Atomic rename - prevents race conditions
            await aiofiles.os.rename(temp_url_path, url_path)
        except Exception:
            # Clean up temp file on failure
            if temp_url_path.exists():
                temp_url_path.unlink()
            raise

        # Store redirect mapping if final URL differs from original
        if final_url and final_url != url:
            temp_redir_path = redir_path.with_suffix('.redir.tmp')
            try:
                async with aiofiles.open(temp_redir_path, 'w', encoding='utf-8') as f:
                    await f.write(final_url)
                await aiofiles.os.rename(temp_redir_path, redir_path)
            except Exception:
                if temp_redir_path.exists():
                    temp_redir_path.unlink()
                raise
        elif redir_path.exists():
            # Remove stale redirect file if URLs now match
            redir_path.unlink()

    async def _verify_url_mapping(self, url: str) -> bool:
        """Verify the URL mapping matches what's stored."""
        url_path, = await self._get_paths(url, "url")
        if url_path.exists():
            try:
                async with aiofiles.open(url_path, 'r', encoding='utf-8') as f:
                    stored_url = (await f.read()).strip()
                return stored_url == url
            except (OSError, UnicodeDecodeError):
                return False
        return False

    async def has_path(self, url: str, extension: str) -> bool:
        """Check if content with given extension is cached for URL (including redirected URLs)."""
        content_path, redir_path = await self._get_paths(url, extension, "redir")
        if content_path.exists():
            return True
        # Check if this URL redirected to another URL that has content
        if redir_path.exists():
            try:
                final_url = redir_path.read_text(encoding='utf-8').strip()
                final_content_path, = self._get_paths(final_url, extension)
                return final_content_path.exists()
            except (OSError, UnicodeDecodeError):
                pass
        return False

    async def get_path(self, url: str, extension: str) -> str:
        """Get cached content for URL with given extension (following redirects if needed)."""
        content_path, redir_path = await self._get_paths(url, extension, "redir")

        # Check if content exists
        if content_path.exists():
            if await self._verify_url_mapping(url):
                async with aiofiles.open(content_path, 'r', encoding='utf-8') as f:
                    return await f.read()

        # Check for redirected content
        if redir_path.exists():
            try:
                async with aiofiles.open(redir_path, 'r', encoding='utf-8') as f:
                    final_url = (await f.read()).strip()
                final_content_path, = await self._get_paths(final_url, extension)
                if final_content_path.exists():
                    async with aiofiles.open(final_content_path, 'r', encoding='utf-8') as f:
                        return await f.read()
            except (OSError, UnicodeDecodeError):
                pass

        raise KeyError(f"Content with extension '{extension}' not cached for URL: {url}")

    async def set_path(self, url: str, extension: str, content: str, final_url: Optional[str] = None) -> None:
        """Store content for URL and extension with atomic operations."""
        path, = await self._get_paths(url, extension)

        # Atomic content write
        temp_path = path.with_suffix(f'.{extension}.tmp')
        try:
            async with aiofiles.open(temp_path, 'w', encoding='utf-8') as f:
                await f.write(content)
            await aiofiles.os.rename(temp_path, path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

        await self._store_url_mapping(url, final_url)

    async def get_html(self, url: str) -> str:
        """Get cached HTML content for URL (following redirects if needed)."""
        return await self.get_path(url, "html")

    async def set_html(self, url: str, content: str, final_url: Optional[str] = None) -> None:
        """Store HTML content for URL."""
        await self.set_path(url, "html", content, final_url)

    async def get_pdf(self, url: str) -> str:
        """Get cached PDF content for URL (following redirects if needed)."""
        return await self.get_path(url, "pdf")

    async def set_pdf(self, url: str, content: str, final_url: Optional[str] = None) -> None:
        """Store PDF content for URL."""
        await self.set_path(url, "pdf", content, final_url)

    async def get_markdown(self, url: str) -> str:
        """Get cached Markdown content for URL (following redirects if needed)."""
        return await self.get_path(url, "md")

    async def set_markdown(self, url: str, content: str, final_url: Optional[str] = None) -> None:
        """Store Markdown content for URL."""
        await self.set_path(url, "md", content, final_url)

    async def get_doi(self, url: str) -> Optional[str]:
        """Get cached DOI for URL (following redirects if needed)."""
        try:
            content = await self.get_path(url, "doi")
            return content.strip()
        except KeyError:
            return None

    async def set_doi(self, url: str, doi: str, final_url: Optional[str] = None) -> None:
        """Store DOI for URL."""
        await self.set_path(url, "doi", doi, final_url)

    async def has_url(self, url: str) -> bool:
        """Check if URL is cached in any format."""
        return (await self.has_path(url, "html") or
                await self.has_path(url, "pdf") or
                await self.has_path(url, "md") or
                await self.has_path(url, "doi"))

    async def get_source_type(self, url: str) -> Optional[str]:
        """Get the source content type for a URL ('html' or 'pdf')."""
        if await self.has_path(url, "html"):
            return "html"
        elif await self.has_path(url, "pdf"):
            return "pdf"
        return None

    async def get_source_content(self, url: str) -> Optional[str]:
        """Get the raw source content (HTML or PDF) for a URL."""
        if await self.has_path(url, "html"):
            return await self.get_html(url)
        elif await self.has_path(url, "pdf"):
            return await self.get_pdf(url)
        return None

    async def clear_url(self, url: str) -> None:
        """Remove all cached content for a URL."""
        paths = await self._get_paths(url, "html", "pdf", "md", "url", "doi", "redir")
        for path in paths:
            if path.exists():
                path.unlink()

    async def get_url_hash(self, url: str) -> str:
        """Get the actual hash string used for a URL (including probe suffix if any)."""
        url_path, = await self._get_paths(url, "url")
        # Extract hash from the path name
        return url_path.stem  # Remove .url extension to get the hash

    async def get_original_url(self, hash_str: str) -> Optional[str]:
        """Get the original URL from a hash string."""
        url_path = self.base_path / f"{hash_str}.url"
        if url_path.exists():
            try:
                async with aiofiles.open(url_path, 'r', encoding='utf-8') as f:
                    return (await f.read()).strip()
            except (OSError, UnicodeDecodeError):
                return None
        return None

    async def get_redirect_info(self, url: str) -> Optional[str]:
        """Get the final URL if this URL redirected, None otherwise."""
        redir_path, = await self._get_paths(url, "redir")
        if redir_path.exists():
            try:
                async with aiofiles.open(redir_path, 'r', encoding='utf-8') as f:
                    content = (await f.read()).strip()
                # Validate that the content looks like a URL
                if content and '://' in content:
                    return content
            except (OSError, UnicodeDecodeError):
                pass
        return None

    async def list_cached_urls(self) -> list[str]:
        """Get a list of all cached URLs."""
        urls = []
        for url_file in self.base_path.glob("*.url"):
            try:
                async with aiofiles.open(url_file, 'r', encoding='utf-8') as f:
                    url = (await f.read()).strip()
                # Verify at least one content file exists
                if await self.has_url(url):
                    urls.append(url)
            except (OSError, UnicodeDecodeError):
                continue
        return urls

class PageFetcher:
    """High-level interface for fetching and caching web pages."""

    def __init__(self, config: 'IfetcherConfig', show_status: bool = True):
        self.cache = URLCache(config)
        self.config = config
        self.status_display = StatusDisplay(show_status=show_status)

    async def _fetch_html_url(self, url: str, progress_info=None) -> dict[str, str]:
        """
        Fetch HTML URL using Crawl4AI with DOI extraction.

        Args:
            url: URL to fetch
            progress_info: Optional tuple of (progress_instance, task_id) for progress bars

        Returns:
            Dict containing raw_content, markdown_content, final_url, and doi
        """
        AsyncWebCrawler, CrawlerRunConfig, JsonXPathExtractionStrategy, _ = _get_crawl4ai_imports()
        domain = url.split('//')[1].split('/')[0] if '//' in url else url

        doi_schema = {
            "name": "DOI extractor (XPath)",
            "baseSelector": "/html",
            "fields": [
                {
                    "name": "doi_meta_cite",
                    "selector": "//meta[@name='citation_doi']",
                    "type": "attribute",
                    "attribute": "content"
                },
                {
                    "name": "doi_meta_pub",
                    "selector": "//meta[@name='publication_doi']",
                    "type": "attribute",
                    "attribute": "content"
                },
                {
                    "name": "doi_dc",
                    "selector": "//meta[translate(@name,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz')='dc.identifier' and contains(@content, 'doi.org/')]",
                    "type": "attribute",
                    "attribute": "content"
                },
                {
                    "name": "doi_dc_doi",
                    "selector": "//meta[translate(@name,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz')='dc.identifier.doi']",
                    "type": "attribute",
                    "attribute": "content"
                },
                {
                    "name": "doi_canonical",
                    "selector": "//link[@rel='canonical' and contains(@href, 'doi.org/')]",
                    "type": "attribute",
                    "attribute": "href"
                }
            ]
        }

        # Create crawler configuration with DOI extraction and proper timeouts
        cfg = CrawlerRunConfig(
            extraction_strategy=JsonXPathExtractionStrategy(doi_schema, verbose=False),
            page_timeout=self.config.tools.crawl4ai.timeout * 1000,
            verbose=False
        )

        if progress_info is not None:
            progress_instance, task_id = progress_info
            progress_instance.update(task_id, description=f"[green]Fetching HTML from[/green] [bold]{domain}[/bold]")
            granular_logger = _get_quiet_logger()
            async with AsyncWebCrawler(logger=granular_logger) as crawler:
                result = await crawler.arun(url=url, config=cfg)
                if not result.success:
                    raise RuntimeError(f"{result.status_code} error fetching {url}: {result.error_message}")
        else:
            with self.status_display.create_status(f"[dim]Initializing...[/dim]") as status:
                granular_logger = _get_granular_logger(status_display=status)
                async with AsyncWebCrawler(logger=granular_logger) as crawler:
                    result = await crawler.arun(url=url, config=cfg)
                    if not result.success:
                        raise RuntimeError(f"{result.status_code} error fetching {url}: {result.error_message}")

        final_url = self._extract_final_url(result, url)
        raw_content = result.html or ''
        markdown_content = self._extract_and_clean_markdown(result)
        doi = self._extract_doi(result)

        return {
            'raw_content': raw_content,
            'markdown_content': markdown_content,
            'final_url': final_url,
            'doi': doi
        }

    async def _fetch_pdf_url(self, url: str, progress_info=None) -> dict[str, str]:
        """
        Fetch PDF URL using Crawl4AI PDF processing strategy.

        Args:
            url: PDF URL to fetch
            progress_info: Optional tuple of (progress_instance, task_id) for progress bars

        Returns:
            Dict containing raw_content, markdown_content, final_url (no DOI extraction for PDFs yet)
        """
        AsyncWebCrawler, CrawlerRunConfig, PDFCrawlerStrategy, PDFContentScrapingStrategy, _ = _get_crawl4ai_pdf_imports()
        domain = url.split('//')[1].split('/')[0] if '//' in url else url

        pdf_crawler_cfg = PDFCrawlerStrategy()
        pdf_scraping_cfg = PDFContentScrapingStrategy()

        cfg = CrawlerRunConfig(
            scraping_strategy=pdf_scraping_cfg,
            page_timeout=self.config.tools.crawl4ai.timeout * 1000,
            verbose=False
        )

        if progress_info is not None:
            progress_instance, task_id = progress_info
            progress_instance.update(task_id, description=f"[red]Fetching PDF from[/red] [bold]{domain}[/bold]")
            granular_logger = _get_quiet_logger()
            async with AsyncWebCrawler(crawler_strategy=pdf_crawler_cfg, logger=granular_logger) as crawler:
                result = await crawler.arun(url=url, config=cfg)
                if not result.success:
                    raise RuntimeError(f"{result.status_code} error fetching PDF {url}: {result.error_message}")
        else:
            with self.status_display.create_status(f"[dim]Initializing...[/dim]") as status:
                granular_logger = _get_granular_logger(status_display=status)
                async with AsyncWebCrawler(crawler_strategy=pdf_crawler_cfg, logger=granular_logger) as crawler:
                    result = await crawler.arun(url=url, config=cfg)
                    if not result.success:
                        raise RuntimeError(f"{result.status_code} error fetching PDF {url}: {result.error_message}")

        final_url = result.url
        markdown_content = self._extract_and_clean_markdown(result)
        raw_content = markdown_content

        return {
            'raw_content': raw_content,
            'markdown_content': markdown_content,
            'content_type': 'application/pdf',
            'final_url': final_url,
            'doi': ''
        }

    def _extract_final_url(self, result, original_url: str) -> str:
        """Extract the final URL after any redirects."""
        final_url = original_url
        if hasattr(result, '_results') and result._results:
            first_result = result._results[0]
            if hasattr(first_result, 'redirected_url') and first_result.redirected_url:
                final_url = first_result.redirected_url
            elif hasattr(first_result, 'url'):
                final_url = first_result.url

        if final_url == original_url and hasattr(result, 'url'):
            final_url = result.url

        return final_url

    def _extract_and_clean_markdown(self, result) -> str:
        """Extract markdown content from crawl result and clean it."""
        markdown_content = ''
        if hasattr(result, 'markdown') and result.markdown:
            if hasattr(result.markdown, 'raw_markdown'):
                markdown_content = result.markdown.raw_markdown or ''
            else:
                markdown_content = str(result.markdown)

        return self._remove_references_section(markdown_content)

    def _extract_doi(self, result) -> str:
        """Extract DOI from crawl4ai extracted content."""
        doi = ''
        try:
            if hasattr(result, 'extracted_content') and result.extracted_content:
                extracted_data = json.loads(result.extracted_content)
                for doi_data in extracted_data:

                    for field in ["doi_meta_pub", "doi_meta_cite", "doi_dc_doi", "doi_dc"]:
                        if field in doi_data and doi_data[field]:
                            doi = doi_data[field]
                            break

                    if not doi and "doi_canonical" in doi_data and doi_data["doi_canonical"]:
                        canonical_url = doi_data["doi_canonical"]
                        if 'doi.org/' in canonical_url:
                            doi = canonical_url.split('doi.org/')[1]

                    if doi:
                        break
        except (json.JSONDecodeError, AttributeError, KeyError, IndexError):
            pass
        return doi

    def _remove_references_section(self, markdown: str) -> str:
        """
        Remove 'References' sections from markdown content.

        Identifies any heading containing 'References' and removes all content
        from that heading up to the next heading or end of document.
        """
        pattern = r'(^|\n)(#{1,6}\s+references.*?)(\n#{1,6}\s+|\Z)'

        def remove_section(match):
            start, heading, end = match.groups()
            return start + end if end.startswith('\n') else start

        return re.sub(pattern, remove_section, markdown, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)

    def _is_pdf_url(self, url: str) -> bool:
        """Check if URL points to a PDF file based on extension."""
        from urllib.parse import urlparse
        return urlparse(url).path.lower().endswith('.pdf')

    async def _fetch_and_cache(self, url: str, progress=True, progress_info=None) -> dict[str, str]:
        """
        Fetch content from URL and cache it based on content type.

        Args:
            url: URL to fetch
            progress: True for status display, False for silent, tuple for progress bar integration
            progress_info: Optional tuple of (progress_instance, task_id) for progress bars

        Returns:
            Dict containing content information
        """
        is_pdf = self._is_pdf_url(url)

        if progress is True:
            if is_pdf:
                content_info = await self._fetch_pdf_url(url)
                content_info['source_type'] = 'pdf'
            else:
                content_info = await self._fetch_html_url(url)
                content_info['source_type'] = 'html'
        elif progress is False:
            old_show_status = self.status_display.show_status
            self.status_display.show_status = False
            try:
                if is_pdf:
                    content_info = await self._fetch_pdf_url(url)
                    content_info['source_type'] = 'pdf'
                else:
                    content_info = await self._fetch_html_url(url)
                    content_info['source_type'] = 'html'
            finally:
                self.status_display.show_status = old_show_status
        else:
            if is_pdf:
                content_info = await self._fetch_pdf_url(url, progress_info=progress)
                content_info['source_type'] = 'pdf'
            else:
                content_info = await self._fetch_html_url(url, progress_info=progress)
                content_info['source_type'] = 'html'

        final_url = content_info.get('final_url', url)

        with self.status_display.create_status(f"[yellow]Caching content[/yellow]", progress_info):
            if is_pdf:
                await self.cache.set_path(url, "pdf", content_info['raw_content'], final_url)
            else:
                await self.cache.set_path(url, "html", content_info['raw_content'], final_url)

            await self.cache.set_path(url, "md", content_info['markdown_content'], final_url)

            if content_info.get('doi'):
                await self.cache.set_path(url, "doi", content_info['doi'], final_url)

        return content_info

    async def get_html(self, url: Union[str, List[str]], progress=True) -> Union[str, List[str]]:
        """
        Get HTML content for single URL or multiple URLs with concurrent fetching.

        Args:
            url: Single URL string or list of URLs to fetch
            progress: True for status display, False for silent, tuple for progress bar integration

        Returns:
            Single content string for single URL, list of content strings for multiple URLs
        """
        if isinstance(url, str):
            if not await self.cache.has_path(url, "html"):
                await self._fetch_and_cache(url, progress=progress)
            return await self.cache.get_path(url, "html")

        elif isinstance(url, list):
            if len(url) == 0:
                return []

            if progress is True:
                results = await self._fetch_multiple_concurrent(url, "html", show_progress=True)
            elif progress is False:
                results = await self._fetch_multiple_concurrent(url, "html", show_progress=False)
            else:
                return await self._fetch_multiple_with_custom_progress(url, "html", progress)

            return [result["content"] for result in results if result["status"] == "success"]

        else:
            raise TypeError(f"url must be str or list[str], got {type(url)}")

    async def get_pdf(self, url: Union[str, List[str]], progress=True) -> Union[str, List[str]]:
        """
        Get PDF content for single URL or multiple URLs with concurrent fetching.

        Args:
            url: Single URL string or list of URLs to fetch
            progress: True for status display, False for silent, tuple for progress bar integration

        Returns:
            Single content string for single URL, list of content strings for multiple URLs
        """
        if isinstance(url, str):
            if not await self.cache.has_path(url, "pdf"):
                await self._fetch_and_cache(url, progress=progress)
            return await self.cache.get_path(url, "pdf")

        elif isinstance(url, list):
            if len(url) == 0:
                return []

            if progress is True:
                results = await self._fetch_multiple_concurrent(url, "pdf", show_progress=True)
            elif progress is False:
                results = await self._fetch_multiple_concurrent(url, "pdf", show_progress=False)
            else:
                return await self._fetch_multiple_with_custom_progress(url, "pdf", progress)

            return [result["content"] for result in results if result["status"] == "success"]

        else:
            raise TypeError(f"url must be str or list[str], got {type(url)}")

    async def get_markdown(self, url: Union[str, List[str]], progress=True) -> Union[str, List[str]]:
        """
        Get Markdown content for single URL or multiple URLs with concurrent fetching.
        Automatically detects and handles both HTML and PDF URLs.

        Args:
            url: Single URL string or list of URLs to fetch
            progress: True for status display, False for silent, tuple for progress bar integration

        Returns:
            Single content string for single URL, list of content strings for multiple URLs
        """
        if isinstance(url, str):
            if await self.cache.has_path(url, "md"):
                return await self.cache.get_path(url, "md")

            await self._fetch_and_cache(url, progress=progress)
            return await self.cache.get_path(url, "md")

        elif isinstance(url, list):
            if len(url) == 0:
                return []

            if progress is True:
                results = await self._fetch_multiple_concurrent(url, "markdown", show_progress=True)
            elif progress is False:
                results = await self._fetch_multiple_concurrent(url, "markdown", show_progress=False)
            else:
                return await self._fetch_multiple_with_custom_progress(url, "markdown", progress)

            return [result["content"] for result in results if result["status"] == "success"]

        else:
            raise TypeError(f"url must be str or list[str], got {type(url)}")

    async def get_raw(self, url: str) -> str:
        """Get raw content (HTML or PDF) for URL, fetching if necessary."""
        # Check cache first
        source_content = await self.cache.get_source_content(url)
        if source_content is not None:
            return source_content

        # Not cached, fetch it
        content_info = await self._fetch_and_cache(url)
        return content_info['raw_content']

    async def is_cached(self, url: str) -> bool:
        """Check if URL is cached (any content type)."""
        return await self.cache.has_url(url)

    async def get_source_type(self, url: str) -> Optional[str]:
        """Get the source content type for a cached URL."""
        return await self.cache.get_source_type(url)

    async def clear_cache(self, url: str) -> None:
        """Clear all cached content for a URL."""
        await self.cache.clear_url(url)

    async def prefetch(self, urls: list[str], progress=True) -> None:
        """
        Prefetch multiple URLs for faster subsequent access.

        Args:
            urls: List of URLs to prefetch
            progress: True for status display, False for silent mode
        """
        for url in urls:
            try:
                await self.get_markdown(url, progress=progress)
            except Exception as e:
                print(f"Failed to prefetch {url}: {e}")

    async def get_doi(self, url: str) -> Optional[str]:
        """
        Get DOI for URL if available.

        Args:
            url: URL to get DOI for

        Returns:
            DOI string if found, None otherwise
        """
        if not await self.cache.has_path(url, "doi") and not await self.cache.has_url(url):
            await self._fetch_and_cache(url)
        return await self.cache.get_doi(url)

    async def _fetch_multiple_concurrent(self, urls: List[str], content_type: str, show_progress: bool = True, max_concurrent: int = 5) -> List[dict]:
        """
        Fetch multiple URLs concurrently, reusing existing batch logic.

        Args:
            urls: List of URLs to fetch
            content_type: Type of content ("html", "pdf", "markdown")
            show_progress: Whether to show progress bar
            max_concurrent: Maximum concurrent fetches

        Returns:
            List of result dictionaries
        """
        if show_progress:
            return await fetch_urls_concurrent_with_progress(
                urls,
                self.config,
                content_type=content_type,
                max_concurrent=max_concurrent
            )
        else:
            # Implement silent concurrent fetching
            import asyncio

            semaphore = asyncio.Semaphore(max_concurrent)

            async def fetch_one(url: str):
                async with semaphore:
                    try:
                        if content_type == "html":
                            if not await self.cache.has_path(url, "html"):
                                await self._fetch_and_cache(url)
                            content = await self.cache.get_path(url, "html")
                        elif content_type == "pdf":
                            if not await self.cache.has_path(url, "pdf"):
                                await self._fetch_and_cache(url)
                            content = await self.cache.get_path(url, "pdf")
                        elif content_type == "markdown":
                            if await self.cache.has_path(url, "md"):
                                content = await self.cache.get_path(url, "md")
                            else:
                                # Fetch appropriate content type to generate markdown
                                if self._is_pdf_url(url):
                                    if not await self.cache.has_path(url, "pdf"):
                                        await self._fetch_and_cache(url)
                                else:
                                    if not await self.cache.has_path(url, "html"):
                                        await self._fetch_and_cache(url)
                                content = await self.cache.get_path(url, "md")
                        else:
                            content = await self.get_raw(url)

                        return {
                            "url": url,
                            "content": content,
                            "size": len(content),
                            "status": "success"
                        }
                    except Exception as e:
                        return {
                            "url": url,
                            "error": str(e),
                            "status": "error"
                        }

            tasks = [fetch_one(url) for url in urls]
            return await asyncio.gather(*tasks)

    async def _fetch_multiple_with_custom_progress(self, urls: List[str], content_type: str, progress_info) -> List[str]:
        """
        Fetch multiple URLs with custom progress bar integration.

        Args:
            urls: List of URLs to fetch
            content_type: Type of content ("html", "pdf", "markdown")
            progress_info: Tuple of (progress_instance, task_id)

        Returns:
            List of content strings for successful fetches
        """
        progress_instance, task_id = progress_info
        results = []

        for i, url in enumerate(urls):
            try:
                domain = url.split('//')[1].split('/')[0] if '//' in url else url
                progress_instance.update(task_id, description=f"[blue]Fetching {content_type} from[/blue] [bold]{domain}[/bold] ({i+1}/{len(urls)})")
                if content_type == "html":
                    if not await self.cache.has_path(url, "html"):
                        await self._fetch_and_cache(url)
                    content = await self.cache.get_path(url, "html")
                elif content_type == "pdf":
                    if not await self.cache.has_path(url, "pdf"):
                        await self._fetch_and_cache(url)
                    content = await self.cache.get_path(url, "pdf")
                elif content_type == "markdown":
                    if await self.cache.has_path(url, "md"):
                        content = await self.cache.get_path(url, "md")
                    else:
                        # Fetch appropriate content type to generate markdown
                        if self._is_pdf_url(url):
                            if not await self.cache.has_path(url, "pdf"):
                                await self._fetch_and_cache(url)
                        else:
                            if not await self.cache.has_path(url, "html"):
                                await self._fetch_and_cache(url)
                        content = await self.cache.get_path(url, "md")
                else:
                    content = await self.get_raw(url)

                results.append(content)

            except Exception as e:
                print(f"Failed to fetch {url}: {e}")
                results.append("")

        return results
async def fetch_urls_with_progress(
    urls: list[str],
    config: 'IfetcherConfig',
    content_type: str = "html",
    progress_description: str = "Fetching URLs..."
) -> list[dict]:
    """
    Fetch multiple URLs with clean progress display.

    Args:
        urls: List of URLs to fetch
        config: Configuration for PageFetcher
        content_type: Type of content to fetch ("html", "pdf", "markdown")
        progress_description: Description for progress bar

    Returns:
        List of result dicts with url, content, and status
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
    content_methods = {
        "html": "get_html",
        "pdf": "get_pdf",
        "markdown": "get_markdown"
    }

    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:

        task = progress.add_task(progress_description, total=len(urls))
        fetcher = PageFetcher(config, show_status=False)
        method_name = content_methods.get(content_type, "get_raw")

        for url in urls:
            try:
                method = getattr(fetcher, method_name)
                if method_name == "get_raw":
                    content = await method(url)
                else:
                    content = await method(url, progress=(progress, task))

                results.append({
                    "url": url,
                    "content": content,
                    "size": len(content),
                    "status": "success"
                })

                domain = url.split('//')[1].split('/')[0] if '//' in url else url
                progress.update(task, description=f"[green]✓ {domain} ({len(content):,} chars)")

            except Exception as e:
                results.append({
                    "url": url,
                    "error": str(e),
                    "status": "error"
                })
                domain = url.split('//')[1].split('/')[0] if '//' in url else url
                progress.update(task, description=f"[red]✗ {domain} (failed)")

            progress.advance(task)

    return results

async def fetch_urls_concurrent_with_progress(
    urls: list[str],
    config: 'IfetcherConfig',
    content_type: str = "html",
    max_concurrent: int = 5
) -> list[dict]:
    """
    Fetch multiple URLs concurrently with progress display.

    Args:
        urls: List of URLs to fetch
        config: Configuration for PageFetcher
        content_type: Type of content to fetch
        max_concurrent: Maximum concurrent fetches

    Returns:
        List of result dicts
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
    import asyncio
    content_methods = {
        "html": "get_html",
        "pdf": "get_pdf",
        "markdown": "get_markdown"
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:

        task = progress.add_task("[blue]Concurrent fetch...", total=len(urls))
        completed = {"count": 0}
        method_name = content_methods.get(content_type, "get_raw")

        async def fetch_one(url: str):
            fetcher = PageFetcher(config, show_status=False)
            domain = url.split('//')[1].split('/')[0] if '//' in url else url

            try:
                method = getattr(fetcher, method_name)
                if method_name == "get_raw":
                    content = await method(url)
                else:
                    content = await method(url, progress=False)

                completed["count"] += 1
                progress.update(
                    task,
                    advance=1,
                    description=f"[blue]Fetching... ({completed['count']}/{len(urls)}) - {domain}"
                )

                return {
                    "url": url,
                    "content": content,
                    "size": len(content),
                    "status": "success"
                }

            except Exception as e:
                completed["count"] += 1
                progress.update(
                    task,
                    advance=1,
                    description=f"[blue]Fetching... ({completed['count']}/{len(urls)}) - Failed: {domain}"
                )

                return {
                    "url": url,
                    "error": str(e),
                    "status": "error"
                }

        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_with_limit(url):
            async with semaphore:
                return await fetch_one(url)

        tasks = [fetch_with_limit(url) for url in urls]
        results = await asyncio.gather(*tasks)

    return results

    async def list_cached_urls(self) -> list[str]:
        """Get a list of all cached URLs."""
        return await self.cache.list_cached_urls()

    async def get_doi(self, url: str) -> Optional[str]:
        """Get DOI for URL if available."""
        if not await self.cache.has_path(url, "doi") and not await self.cache.has_url(url):
            # Try to fetch and extract DOI
            await self._fetch_and_cache(url)

        return await self.cache.get_doi(url)
