"""High-level PageFetcher interface with eliminated duplication using higher-order functions."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Union, List, Optional, Callable, Any, Dict, Tuple, Literal
from .cache import URLCache
from .web_client import WebClient
from .batch_operations import BatchOperations
from .content_processor import ContentProcessor
from interaction_finder.logging import logfire

logger = logging.getLogger(__name__)


@dataclass
class FetchedDocument:
    """Document fetched from URL with markdown content and metadata."""

    url: str
    content_markdown: str
    source_type: Optional[str] = None  # "html" or "pdf"
    doi: Optional[str] = None
    publication_date: Optional[str] = None  # YYYY-MM-DD format


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
        self,
        cache_dir: Path | str = "cache",
        timeout: int = 30,
        show_status: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize PageFetcher.

        Parameters:
            cache_dir: Directory path for cache storage (default: "cache")
            timeout: Request timeout in seconds (default: 30)
            show_status: Show progress indicators (default: True)
            verbose: Enable verbose logging (default: False)
        """
        self.cache = URLCache(cache_dir)
        self.web_client = WebClient(timeout, verbose)
        self.content_processor = ContentProcessor()
        self.batch_ops = BatchOperations(self.cache, self.web_client, show_status)
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
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

    async def get_chunks_with_embeddings(
        self,
        url: Union[str, List[str]],
        *,
        progress: bool = True,
        fail_fast: bool = True,
        retry: bool = False,
    ):
        """Get chunks with embeddings for URL(s)."""
        if isinstance(url, str):
            # Single URL - check cache first
            if not retry and await self.cache.has_path(url, "chunks"):
                cached_data = await self.cache.get_content(url, "chunks")
                # Handle both formats: direct chunks list or {"chunks": [...]}
                if isinstance(cached_data, dict) and "chunks" in cached_data:
                    return cached_data["chunks"]
                else:
                    return cached_data
            # Cache miss - fetch using batch operations
            return await self.batch_ops._fetch_chunks_and_cache(url, retry=retry)
        else:
            # Multiple URLs - use batch operations
            results = await self.batch_ops.fetch_multiple(
                url, "chunks", progress=progress, fail_fast=fail_fast, retry=retry
            )
            return results

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

    async def fetch_documents(
        self,
        urls: List[str],
        *,
        progress: bool = False,
        fail_fast: bool = False,
        retry: bool = False,
    ) -> Dict[str, Optional[FetchedDocument]]:
        """
        Fetch multiple URLs and return markdown content with metadata.

        Parameters:
            urls: List[str] - URLs to fetch
            progress: bool - Show progress bar
            fail_fast: bool - Stop on first error
            retry: bool - Retry failed fetches

        Returns:
            Dict[url, FetchedDocument] - Mapping of URL to document
                (None value if fetch failed)

        Raises:
            Exception - If fail_fast=True and a fetch fails
        """
        # Fetch markdown for all URLs using existing batch operation
        markdown_results = await self.get_markdown(
            urls, progress=progress, fail_fast=False, retry=retry
        )

        # Build document dict with metadata
        documents = {}
        for url, markdown in zip(urls, markdown_results):
            if markdown and not isinstance(markdown, Exception):
                # Fetch metadata from cache (DOI and publication_date are cached during initial fetch)
                source_type = await self.get_source_type(url)
                doi = None
                pub_date = None

                if source_type == "html":
                    doi = await self.get_doi(url, retry=False)
                    if doi:
                        # Publication date should be cached from initial fetch
                        pub_date = await self.get_publication_date(url, doi=doi)

                documents[url] = FetchedDocument(
                    url=url,
                    content_markdown=markdown,
                    source_type=source_type,
                    doi=doi,
                    publication_date=pub_date,
                )
            else:
                documents[url] = None
                if fail_fast:
                    error = (
                        markdown
                        if isinstance(markdown, Exception)
                        else Exception(f"Failed to fetch {url}")
                    )
                    raise error

        return documents

    async def get_doi(self, url: str, *, retry: bool = False) -> Optional[str]:
        """Get DOI for URL if available."""
        if await self.cache.has_path(url, "doi"):
            return await self.cache.get_content(url, "doi")

        # Try OpenAlex direct lookup for PubMed/PMC URLs
        if metadata := await self._fetch_metadata_from_openalex(
            url, ["doi", "publication_date"]
        ):
            return metadata.get("doi")

        # Fall back to HTML fetching for non-PubMed URLs (PDFs don't have DOI extraction)
        if not self.web_client._is_pdf_url(url):
            await self.get_html(url, retry=retry)
            if await self.cache.has_path(url, "doi"):
                return await self.cache.get_content(url, "doi")

        return None

    async def get_publication_date(
        self, url: str, doi: Optional[str] = None
    ) -> Optional[str]:
        """
        Get publication date for URL (YYYY-MM-DD format).

        For PubMed/PMC URLs, queries OpenAlex directly via PMID/PMCID.
        Otherwise uses DOI-based lookup.

        Parameters:
            url: URL to get publication date for
            doi: Optional DOI to use for lookup (avoids re-fetching)

        Returns:
            Publication date string or None if not available
        """
        if await self.cache.has_path(url, "publication_date"):
            return await self.cache.get_content(url, "publication_date")

        # Try OpenAlex direct lookup for PubMed/PMC URLs
        if metadata := await self._fetch_metadata_from_openalex(
            url, ["publication_date", "doi"]
        ):
            return metadata.get("publication_date")

        # Fall back to DOI-based lookup
        if doi is None:
            doi = await self.get_doi(url, retry=False)

        if doi:
            try:
                from .doi_metadata import fetch_doi_metadata

                metadata = await fetch_doi_metadata(doi)
                if metadata and metadata.get("publication_date"):
                    pub_date = metadata["publication_date"]
                    await self.cache.set_content(url, "publication_date", pub_date)
                    return pub_date
            except Exception as exc:
                logger.warning(
                    "Failed to fetch DOI metadata",
                    extra={
                        "url": url,
                        "doi": doi,
                        "error": str(exc),
                    },
                )

        return None

    async def list_cached_urls(self) -> List[str]:
        """Get a list of all cached URLs."""
        return await self.cache.list_cached_urls()

    def _extract_pubmed_identifier(
        self, url: str
    ) -> Tuple[Optional[str], Optional[Literal["pmid", "pmcid"]]]:
        """Extract PubMed or PMC identifier from URL."""
        # Try PMID: pubmed.ncbi.nlm.nih.gov/{PMID}/
        if match := re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", url):
            return match.group(1), "pmid"
        # Try PMCID: ncbi.nlm.nih.gov/pmc/articles/PMC{ID}/
        if match := re.search(r"ncbi\.nlm\.nih\.gov/pmc/articles/PMC(\d+)", url):
            return match.group(1), "pmcid"
        return None, None

    async def _fetch_metadata_from_openalex(
        self, url: str, fields: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch metadata from OpenAlex using PMID/PMCID if URL is PubMed, otherwise return None.

        Caches all fetched fields for the URL.

        Parameters:
            url: URL to fetch metadata for
            fields: Fields to request from OpenAlex

        Returns:
            Dict with requested fields, or None if not a PubMed URL or fetch failed
        """
        identifier, id_type = self._extract_pubmed_identifier(url)
        if not identifier or not id_type:
            return None

        try:
            from .doi_metadata import fetch_work_metadata

            metadata = await fetch_work_metadata(
                identifier, id_type=id_type, select=fields
            )
            if metadata:
                # Cache all available fields
                for field in fields:
                    if value := metadata.get(field):
                        await self.cache.set_content(url, field, value)
                return metadata
        except Exception as exc:
            logger.warning(
                "OpenAlex metadata fetch failed",
                extra={
                    "url": url,
                    "id_type": id_type,
                    "identifier": identifier,
                    "error": str(exc),
                },
            )

        return None

    async def get_chunks(
        self,
        url: Union[str, List[str]],
        progress=True,
        fail_fast=False,
        *,
        retry: bool = False,
    ) -> Union[List[str], List[List[str]]]:
        """Get chunk texts only."""
        if isinstance(url, str):
            chunks_data = await self.get_chunks_with_embeddings(url, retry=retry)
            return [chunk["text"] for chunk in chunks_data]
        elif isinstance(url, list):
            results = []
            for single_url in url:
                try:
                    chunks_data = await self.get_chunks_with_embeddings(
                        single_url, retry=retry
                    )
                    results.append([chunk["text"] for chunk in chunks_data])
                except Exception as exc:
                    if fail_fast:
                        raise
                    logger.warning(
                        "Failed to create chunks for URL",
                        extra={
                            "url": single_url,
                            "error": str(exc),
                        },
                    )
                    results.append([])
            return results
        else:
            raise TypeError(f"Expected str or List[str], got {type(url)}")


# Export the class and dataclass
__all__ = ["PageFetcher", "FetchedDocument"]
