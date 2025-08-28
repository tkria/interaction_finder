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

    # get_chunks method is defined manually below for backward compatibility
    # get_chunks = _create_content_getter(
    #     "chunks",
    #     lambda self, url, **kw: self.batch_ops._fetch_chunks_and_cache(url, **kw),
    # )

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

    async def get_groups(
        self,
        urls: List[str],
        constraint_type: str = "count",
        min_size: int = 3,
        max_size: int = 8,
        linkage_method: str = "average",
        prefetch: bool = True,
        progress: bool = True,
        retry: bool = False,
    ) -> List[dict]:
        """
        Group documents by semantic similarity with size constraints.

        Args:
            urls: List of URLs to group
            constraint_type: Either "count" or "words"
            min_size: Minimum group size (documents or words)
            max_size: Maximum group size (documents or words)
            linkage_method: Clustering linkage method ("average", "complete", "single")
            prefetch: Whether to fetch missing documents
            progress: Show progress bar
            retry: Retry failed fetches

        Returns:
            List of group dictionaries with:
            - documents: List of URLs in group
            - total_words: Combined word count
            - cohesion_score: Average pairwise similarity
        """
        from .document_grouper import DocumentGrouper, compute_group_cohesion

        if not urls:
            return []

        # Validate constraint parameters
        if constraint_type not in ("count", "words"):
            raise ValueError("constraint_type must be 'count' or 'words'")
        if min_size < 1:
            raise ValueError("min_size must be >= 1")
        if min_size > max_size:
            raise ValueError("min_size must be <= max_size")

        # Ensure all documents are cached with chunks
        if prefetch:
            await self.prefetch(urls, ["chunks"], max_concurrent=5)

        # Get chunk data for all documents
        doc_chunks = {}
        doc_word_counts = {}

        for url in urls:
            try:
                chunks_data = await self.get_chunks_with_embeddings(url, retry=retry)
                doc_chunks[url] = chunks_data
                doc_word_counts[url] = sum(chunk["wordcount"] for chunk in chunks_data)
            except Exception as e:
                if retry:
                    # Second attempt failed, skip this document
                    if self.verbose:
                        from rich.console import Console

                        console = Console()
                        console.print(f"[red]Failed to get chunks for {url}: {e}[/red]")
                    continue
                else:
                    raise

        # Filter out documents without chunks
        valid_urls = [url for url in urls if url in doc_chunks]

        if not valid_urls:
            return []

        if len(valid_urls) == 1:
            # Single document - return as single group
            url = valid_urls[0]
            doc_embeddings = self._compute_document_embeddings({url: doc_chunks[url]})
            return [
                {
                    "documents": [url],
                    "total_words": doc_word_counts[url],
                    "cohesion_score": 1.0,
                }
            ]

        # Group documents using constrained agglomerative clustering
        grouper = DocumentGrouper(linkage_method=linkage_method)
        groups = grouper.group_documents(
            documents=valid_urls,
            chunk_data=doc_chunks,
            constraint_type=constraint_type,
            min_size=min_size,
            max_size=max_size,
        )

        # Compute document embeddings for cohesion calculation
        doc_embeddings = self._compute_document_embeddings(doc_chunks)

        # Format results
        result = []
        for group in groups:
            total_words = sum(doc_word_counts.get(url, 0) for url in group)
            cohesion = compute_group_cohesion(group, doc_embeddings)

            result.append(
                {
                    "documents": group,
                    "total_words": total_words,
                    "cohesion_score": cohesion,
                }
            )

        return result

    async def get_chunks_with_embeddings(
        self, url: str, retry: bool = False
    ) -> List[dict]:
        """Get chunks with embeddings."""
        if await self.cache.has_path(url, "chunks"):
            content = await self.cache.get_content(url, "chunks")
            if isinstance(content, dict) and "chunks" in content:
                return content["chunks"]

        # Fetch and compute
        markdown = await self.get_markdown(url, retry=retry)
        chunks_data = self.web_client.create_chunks(markdown)

        # Save in new format
        from datetime import datetime

        await self.cache.set_content(
            url,
            "chunks",
            {
                "version": 2,
                "chunks": chunks_data,
                "metadata": {
                    "model": "minishlab/potion-base-8M",
                    "chunk_method": "SemanticChunker",
                    "created_at": datetime.now().isoformat(),
                    "total_wordcount": sum(c["wordcount"] for c in chunks_data),
                },
            },
        )

        return chunks_data

    def _compute_document_embeddings(self, doc_chunks: dict) -> dict:
        """Compute document-level embeddings as averages of chunk embeddings.

        Returns:
            Dict mapping URLs to numpy arrays of document embeddings.
            Documents without embeddings are excluded from the result.
        """
        import numpy as np

        doc_embeddings = {}
        for url, chunks in doc_chunks.items():
            chunk_embeddings = []
            for chunk in chunks:
                if chunk.get("embedding"):
                    chunk_embeddings.append(np.array(chunk["embedding"]))

            if chunk_embeddings:
                # Compute mean embedding as numpy array
                doc_embeddings[url] = np.mean(chunk_embeddings, axis=0)
            # Skip documents without embeddings rather than using zeros

        return doc_embeddings

    def _parse_constraint(self, constraint: str) -> tuple:
        """Parse constraint string - delegated to DocumentGrouper."""
        from .document_grouper import DocumentGrouper

        return DocumentGrouper.parse_constraint(constraint)

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
                except Exception:
                    if fail_fast:
                        raise
                    results.append([])
            return results
        else:
            raise TypeError(f"Expected str or List[str], got {type(url)}")


# Export the class for backward compatibility
__all__ = ["PageFetcher"]
