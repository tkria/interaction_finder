"""High-level PageFetcher interface with eliminated duplication using higher-order functions."""

from typing import Union, List, Optional, Callable, Any, TYPE_CHECKING
from .cache import URLCache
from .web_client import WebClient
from .batch_operations import BatchOperations
from .content_processor import ContentProcessor

if TYPE_CHECKING:
    from ..settings import IfetcherConfig


def _create_content_getter(content_type: str, single_fetcher: Callable):
    """Higher-order function that creates get_* methods with unified logic."""

    async def get_content(
        self, url: Union[str, List[str]], progress=True, fail_fast=False, *, retry=False
    ):
        if isinstance(url, str):
            return await single_fetcher(self, url, retry=retry)
        elif isinstance(url, list):
            return (
                []
                if not url
                else await self.batch_ops.fetch_multiple(
                    url,
                    content_type,
                    progress=progress,
                    fail_fast=fail_fast,
                    retry=retry,
                )
            )
        else:
            raise TypeError(f"Expected str or List[str], got {type(url)}")

    return get_content


class PageFetcher:
    """High-level interface for fetching and caching web pages with eliminated boilerplate."""

    def __init__(
        self, config: "IfetcherConfig", show_status: bool = True, verbose: bool = False
    ):
        self.cache = URLCache(config)
        self.web_client = WebClient(config, verbose)
        self.content_processor = ContentProcessor()
        self.batch_ops = BatchOperations(self.cache, self.web_client, show_status)
        self.config = config
        self.verbose = verbose

    # Auto-generate all get_* methods using higher-order functions
    get_html = _create_content_getter(
        "html", lambda self, url, **kw: self.batch_ops._fetch_html_and_cache(url, **kw)
    )

    get_pdf = _create_content_getter(
        "pdf", lambda self, url, **kw: self.batch_ops._fetch_pdf_and_cache(url, **kw)
    )

    get_markdown = _create_content_getter(
        "markdown",
        lambda self, url, **kw: self.batch_ops._fetch_markdown_and_cache(url, **kw),
    )

    get_chunks = _create_content_getter(
        "chunks",
        lambda self, url, **kw: self.batch_ops._fetch_chunks_and_cache(url, **kw),
    )

    async def get_raw(self, url: str, *, retry: bool = False) -> str:
        """Get raw content (HTML or PDF) for URL, fetching if necessary."""
        # Check cache first
        if await self.cache.has_path(url, "html"):
            return await self.cache.get_content(url, "html")
        elif await self.cache.has_path(url, "pdf"):
            return await self.cache.get_content(url, "pdf")

        # Fetch based on URL type
        if self.web_client._is_pdf_url(url):
            return await self.get_pdf(url, retry=retry)
        else:
            return await self.get_html(url, retry=retry)

    async def get_raw_markdown(self, url: str, *, retry: bool = False) -> str:
        """Get raw (pre-cleaned) markdown content for URL, fetching if necessary."""
        if await self.cache.has_path(url, "raw_markdown"):
            return await self.cache.get_content(url, "raw_markdown")

        # Fetch will populate raw_markdown cache
        if self.web_client._is_pdf_url(url):
            await self.get_pdf(url, retry=retry)
        else:
            await self.get_html(url, retry=retry)

        # Try again after fetching
        return await self.cache.get_content(url, "raw_markdown")

    async def is_cached(self, url: str) -> bool:
        """Check if URL is cached (any content type)."""
        return await self.cache.has_url(url)

    async def get_source_type(self, url: str) -> Optional[str]:
        """Get the source content type for a cached URL."""
        return await self.cache.get_source_type(url)

    async def clear_cache(self, url: str) -> None:
        """Clear cached content for URL."""
        await self.cache.clear_url(url)

    async def prefetch(
        self, urls: List[str], content_types: List[str] = None, max_concurrent: int = 5
    ) -> None:
        """Prefetch multiple URLs for multiple content types."""
        await self.batch_ops.prefetch_urls(urls, content_types, max_concurrent)

    async def get_doi(self, url: str, *, retry: bool = False) -> Optional[str]:
        """Get DOI for URL if available."""
        if await self.cache.has_path(url, "doi"):
            return await self.cache.get_content(url, "doi")

        # Fetch HTML to populate DOI cache (PDFs don't have DOI extraction)
        if not self.web_client._is_pdf_url(url):
            await self.get_html(url, retry=retry)
            # Try again after fetching
            if await self.cache.has_path(url, "doi"):
                return await self.cache.get_content(url, "doi")

        return None

    async def list_cached_urls(self) -> List[str]:
        """Get a list of all cached URLs."""
        return await self.cache.list_cached_urls()

    def _process_multiple_results(
        self, results: List[Any], progress_display
    ) -> List[Any]:
        """Process results from batch operations, handling exceptions."""
        processed = []
        for result in results:
            if isinstance(result, Exception):
                if progress_display and self.verbose:
                    progress_display.console.print(f"[red]Error: {result}[/red]")
                processed.append(None)
            else:
                processed.append(result)
        return processed

    # Legacy methods for backward compatibility (delegating to respective components)

    async def _fetch_and_cache(
        self, url: str, progress=True, progress_info=None, *, retry: bool = False
    ):
        """Legacy method - use appropriate get_* method instead."""
        if self.web_client._is_pdf_url(url):
            return {
                "raw_content": await self.get_pdf(url, retry=retry),
                "markdown_content": await self.get_raw_markdown(url, retry=retry),
                "final_url": url,  # Simplified for legacy compatibility
                "doi": "",
            }
        else:
            return {
                "raw_content": await self.get_html(url, retry=retry),
                "markdown_content": await self.get_raw_markdown(url, retry=retry),
                "final_url": url,  # Simplified for legacy compatibility
                "doi": await self.get_doi(url, retry=retry) or "",
            }

    def _extract_final_url(self, result, original_url: str) -> str:
        """Legacy method - functionality moved to WebClient."""
        return self.web_client._extract_final_url(result, original_url)

    def _extract_doi(self, result) -> str:
        """Legacy method - functionality moved to WebClient."""
        return self.web_client._extract_doi(result)

    def _refine_article_content(self, markdown: str) -> str:
        """Legacy method - functionality moved to ContentProcessor."""
        return self.content_processor.refine_article(markdown)

    def _is_pdf_url(self, url: str) -> bool:
        """Legacy method - functionality moved to WebClient."""
        return self.web_client._is_pdf_url(url)

    def _create_chunks(self, markdown_content: str) -> List[str]:
        """Legacy method - functionality moved to WebClient."""
        return self.web_client.create_chunks(markdown_content)

    def _calculate_content_size(self, content: Any, content_type: str) -> int:
        """Calculate the size of content for display purposes."""
        return self.batch_ops.calculate_batch_content_size([content], content_type)

    async def _fetch_multiple(
        self,
        urls: List[str],
        content_type: str,
        progress=None,
        max_concurrent: int = 5,
        fail_fast: bool = False,
        *,
        retry: bool = False,
    ) -> List:
        """
        Legacy method for backward compatibility with CLI.

        Unified method to fetch multiple URLs with different progress modes.
        Delegates to batch_ops.fetch_multiple with appropriate parameters.
        """
        # Convert legacy progress parameter to new format
        show_progress = progress if progress is not None else True

        return await self.batch_ops.fetch_multiple(
            urls,
            content_type,
            max_concurrent=max_concurrent,
            fail_fast=fail_fast,
            progress=show_progress,
            retry=retry,
        )


# Export the class for backward compatibility
__all__ = ["PageFetcher"]
