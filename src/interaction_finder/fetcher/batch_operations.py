"""Batch operations for concurrent URL fetching."""

import asyncio
from typing import List, Dict, Any, Callable, Optional, Tuple
from urllib.parse import urlparse
from .cache import URLCache
from .web_client import WebClient
from .progress_display import StatusDisplay


class PreviousFailure(Exception):
    """Raised when a previous fetch failure sentinel is present for a URL."""

    def __init__(self, url: str, message: str | None = None):
        super().__init__(message or f"Previous failure recorded for URL: {url}")
        self.url = url


# Batch operation constants
DEFAULT_MAX_CONCURRENT = 5


class BatchOperations:
    """Handles batch URL processing with concurrent fetching and smart caching."""

    def __init__(
        self, cache: URLCache, web_client: WebClient, show_status: bool = True
    ):
        self.cache = cache
        self.web_client = web_client
        self.progress_display = StatusDisplay(show_status)
        self._domain_width = 15  # Default width, will be calculated dynamically

    async def fetch_multiple(
        self,
        urls: List[str],
        content_type: str,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        fail_fast: bool = False,
        **fetch_options,
    ) -> List[Any]:
        """Generic multi-URL fetcher using functional composition."""

        if not urls:
            return []

        # Calculate optimal domain width for consistent progress display
        self._domain_width = self._calculate_optimal_width(urls)

        # Smart caching check - separate cached from uncached URLs
        cached_results = await self.cache.get_content_batch(urls, content_type)
        cached_urls = {
            url: content for url, content in cached_results if content is not None
        }
        uncached_urls = [url for url, content in cached_results if content is None]

        # Fetch uncached URLs
        fetched_results = {}
        if uncached_urls:
            # Use optimized batch path for chunks when fetching multiple URLs
            if content_type == "chunks" and len(uncached_urls) > 1:
                fetched_results = await self._fetch_chunks_batch(
                    uncached_urls,
                    max_concurrent,
                    fail_fast,
                    **fetch_options,
                )
            else:
                # Standard path for other content types or single URL
                fetcher_func = self._get_fetcher_for_content_type(content_type)
                fetched_results = await self._concurrent_fetch_with_progress(
                    uncached_urls,
                    fetcher_func,
                    content_type,
                    max_concurrent,
                    fail_fast,
                    **fetch_options,
                )

        # Combine and return in original order
        return [cached_urls.get(url) or fetched_results.get(url) for url in urls]

    def _get_fetcher_for_content_type(self, content_type: str) -> Callable:
        """Get the appropriate fetcher function for a content type."""
        fetcher_map = {
            "html": self._fetch_html_and_cache,
            "pdf": self._fetch_pdf_and_cache,
            "markdown": self._fetch_markdown_and_cache,
            "chunks": self._fetch_chunks_and_cache,
        }

        fetcher_func = fetcher_map.get(content_type)
        if not fetcher_func:
            raise ValueError(f"Unknown content type: {content_type}")
        return fetcher_func

    async def _concurrent_fetch_with_progress(
        self,
        urls: List[str],
        fetcher_func: Callable,
        content_type: str,
        max_concurrent: int,
        fail_fast: bool,
        **options,
    ) -> Dict[str, Any]:
        """Concurrent fetching with progress display and error handling."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_with_semaphore(url: str) -> Tuple[str, Any]:
            async with semaphore:
                try:
                    # Filter out options that the fetcher functions don't expect
                    filtered_options = {
                        k: v for k, v in options.items() if k in ["retry"]
                    }  # Only pass retry parameter
                    result = await fetcher_func(url, **filtered_options)
                    return url, result
                except Exception as e:
                    if fail_fast:
                        raise
                    return url, e

        # Execute with progress display
        with self.progress_display.batch_progress(
            len(urls), f"Fetching {content_type}"
        ) as progress:
            tasks = [fetch_with_semaphore(url) for url in urls]

            # Process results as they complete
            results = {}
            completed = 0
            for coro in asyncio.as_completed(tasks):
                url, result = await coro
                results[url] = result
                completed += 1
                # Format domain for consistent width display
                domain = self.format_domain(url)
                progress.update(1, f"Fetching {domain}")

            # Update final status when all complete
            if completed > 0:
                progress.update(0, f"Fetched {completed} sites")

        return results

    async def _check_previous_failures(self, url: str) -> None:
        """Check for previous failures and raise PreviousFailure if found."""
        # Check original URL
        if await self.cache.is_failed(url):
            reason = await self.cache.get_failed_reason(url)
            raise PreviousFailure(
                url, reason or f"Previous failure recorded for URL: {url}"
            )

        # Follow redirect chain and check each URL for failures
        try:
            visited = set()
            current = url
            for _ in range(3):  # Safety cap to avoid infinite loops
                redir = await self.cache.get_redirect_info(current)
                if not redir or redir in visited:
                    break
                visited.add(redir)
                if await self.cache.is_failed(redir):
                    reason = await self.cache.get_failed_reason(redir)
                    raise PreviousFailure(
                        redir,
                        reason or f"Previous failure recorded for URL: {redir}",
                    )
                current = redir
        except PreviousFailure:
            # Re-raise PreviousFailure exceptions
            raise
        except (FileNotFoundError, KeyError, OSError):
            # Only catch expected cache miss/access errors, continue processing
            pass

    async def _fetch_html_and_cache(self, url: str, retry: bool = False) -> str:
        """Fetch HTML content and cache it."""
        # When retrying, bypass cache and fetch fresh content
        if not retry and await self.cache.has_path(url, "html"):
            return await self.cache.get_content(url, "html")

        # Check for previous failures unless retrying
        if not retry:
            await self._check_previous_failures(url)

        # Fetch operation - mark as failed only on fetch errors
        try:
            fetch_result = await self.web_client.fetch_html(url, retry=retry)
        except Exception as e:
            # Only mark as failed for actual fetch/network errors
            await self.cache.mark_failed(url, reason=str(e))
            raise

        # Post-processing - don't mark as failed on processing errors
        try:
            # Cache all extracted data
            final_url = fetch_result["final_url"]
            await self.cache.set_content(
                url, "html", fetch_result["raw_content"], final_url
            )

            if fetch_result["markdown_content"]:
                await self.cache.set_content(
                    url, "raw_markdown", fetch_result["markdown_content"], final_url
                )
            if fetch_result["doi"]:
                await self.cache.set_content(url, "doi", fetch_result["doi"], final_url)

            # Clear any previous failure markers on success
            await self.cache.clear_failed(url)
            if final_url and final_url != url:
                await self.cache.clear_failed(final_url)

            return fetch_result["raw_content"]

        except Exception:
            # Processing errors propagate but don't mark as failed
            # The content was successfully fetched, just post-processing failed
            raise

    async def _fetch_pdf_and_cache(self, url: str, retry: bool = False) -> str:
        """Fetch PDF content and cache it."""
        # When retrying, bypass cache and fetch fresh content
        if not retry and await self.cache.has_path(url, "pdf"):
            return await self.cache.get_content(url, "pdf")

        # Check for previous failures unless retrying
        if not retry:
            await self._check_previous_failures(url)

        # Fetch operation - mark as failed only on fetch errors
        try:
            fetch_result = await self.web_client.fetch_pdf(url, retry=retry)
        except Exception as e:
            # Only mark as failed for actual fetch/network errors
            await self.cache.mark_failed(url, reason=str(e))
            raise

        # Post-processing - don't mark as failed on processing errors
        try:
            # Cache all extracted data
            final_url = fetch_result["final_url"]
            await self.cache.set_content(
                url, "pdf", fetch_result["raw_content"], final_url
            )

            if fetch_result["markdown_content"]:
                await self.cache.set_content(
                    url, "raw_markdown", fetch_result["markdown_content"], final_url
                )

            # Clear any previous failure markers on success
            await self.cache.clear_failed(url)
            if final_url and final_url != url:
                await self.cache.clear_failed(final_url)

            return fetch_result["raw_content"]

        except Exception:
            # Processing errors propagate but don't mark as failed
            # The content was successfully fetched, just post-processing failed
            raise

    async def _fetch_markdown_and_cache(self, url: str, retry: bool = False) -> str:
        """Fetch processed markdown content and cache it."""
        # When retrying, bypass cache and fetch fresh content
        if not retry and await self.cache.has_path(url, "markdown"):
            return await self.cache.get_content(url, "markdown")

        # Check if we already have raw markdown cached (from HTML/PDF fetch)
        # Skip cache check if retrying
        if not retry and await self.cache.has_path(url, "raw_markdown"):
            raw_markdown = await self.cache.get_content(url, "raw_markdown")
            final_url = await self.cache.get_redirect_info(url)
        # Check if we can regenerate raw_markdown from cached HTML/PDF content
        elif not retry and (
            await self.cache.has_path(url, "html")
            or await self.cache.has_path(url, "pdf")
        ):
            # Regenerate raw_markdown from cached content using file:// URI
            if await self.cache.has_path(url, "html"):
                html_path = await self.cache.get_file_path(url, "html")
                if html_path is None:
                    raise RuntimeError(f"HTML file path not found for URL: {url}")
                file_uri = html_path.absolute().as_uri()
                fetch_result = await self.web_client.fetch_html(file_uri, retry=False)
            else:  # PDF case
                pdf_path = await self.cache.get_file_path(url, "pdf")
                if pdf_path is None:
                    raise RuntimeError(f"PDF file path not found for URL: {url}")
                file_uri = pdf_path.absolute().as_uri()
                fetch_result = await self.web_client.fetch_pdf(file_uri, retry=False)

            raw_markdown = fetch_result["markdown_content"]
            final_url = await self.cache.get_redirect_info(url)

            # Cache the regenerated raw markdown
            await self.cache.set_content(url, "raw_markdown", raw_markdown, final_url)
        else:
            # Check for previous failures unless retrying
            if not retry:
                await self._check_previous_failures(url)

            # Fetch operation - mark as failed only on fetch errors
            try:
                # Do fresh fetch to get all data
                if self.web_client._is_pdf_url(url):
                    fetch_result = await self.web_client.fetch_pdf(url, retry=retry)
                else:
                    fetch_result = await self.web_client.fetch_html(url, retry=retry)
            except Exception as e:
                # Only mark as failed for actual fetch/network errors
                await self.cache.mark_failed(url, reason=str(e))
                raise

            # Post-processing - don't mark as failed on processing errors
            try:
                raw_markdown = fetch_result["markdown_content"]
                final_url = fetch_result["final_url"]

                # Cache the raw markdown
                await self.cache.set_content(
                    url, "raw_markdown", raw_markdown, final_url
                )

                # Clear any previous failure markers on success
                await self.cache.clear_failed(url)
                if final_url and final_url != url:
                    await self.cache.clear_failed(final_url)

            except Exception:
                # Processing errors propagate but don't mark as failed
                # The content was successfully fetched, just post-processing failed
                raise

        # Process the raw markdown content
        from .content_processor import ContentProcessor

        processor = ContentProcessor()
        processed_markdown = processor.refine_article(raw_markdown)

        # Cache the processed markdown
        await self.cache.set_content(url, "markdown", processed_markdown, final_url)

        return processed_markdown

    async def _fetch_chunks_and_cache(self, url: str, retry: bool = False) -> List[str]:
        """Fetch chunked content and cache it."""
        # When retrying, bypass cache and fetch fresh content
        if not retry and await self.cache.has_path(url, "chunks"):
            return await self.cache.get_content(url, "chunks")

        # Ensure we have both raw content (HTML/PDF) and processed markdown
        if self.web_client._is_pdf_url(url):
            await self._fetch_pdf_and_cache(url, retry=retry)
        else:
            await self._fetch_html_and_cache(url, retry=retry)

        # Get processed markdown (this will use the cached raw_markdown)
        processed_markdown = await self._fetch_markdown_and_cache(url, retry=retry)
        chunks = self.web_client.create_chunks(processed_markdown)

        # Get final URL from redirect info and cache chunks there
        final_url = await self.cache.get_redirect_info(url)
        await self.cache.set_content(url, "chunks", chunks, final_url)

        return chunks

    async def _fetch_chunks_batch(
        self,
        urls: List[str],
        max_concurrent: int,
        fail_fast: bool,
        **options,
    ) -> Dict[str, List[dict]]:
        """
        Batch chunk multiple URLs efficiently using chunk_batch().

        Fetches all markdown in parallel, then processes all documents together
        using chonkie's batch chunking for optimal embedding generation.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        # Step 1: Fetch all markdown in parallel
        async def fetch_markdown_with_semaphore(
            url: str,
        ) -> Tuple[str, str | Exception]:
            async with semaphore:
                try:
                    retry = options.get("retry", False)
                    # Ensure we have raw content
                    if self.web_client._is_pdf_url(url):
                        await self._fetch_pdf_and_cache(url, retry=retry)
                    else:
                        await self._fetch_html_and_cache(url, retry=retry)
                    # Get processed markdown
                    markdown = await self._fetch_markdown_and_cache(url, retry=retry)
                    return url, markdown
                except Exception as e:
                    if fail_fast:
                        raise
                    return url, e

        with self.progress_display.batch_progress(
            len(urls), "Fetching content"
        ) as progress:
            tasks = [fetch_markdown_with_semaphore(url) for url in urls]

            # Collect markdown results as they complete
            markdown_results = {}
            completed = 0
            for coro in asyncio.as_completed(tasks):
                url, result = await coro
                markdown_results[url] = result
                completed += 1
                domain = self.format_domain(url)
                progress.update(1, f"Fetching {domain}")

            if completed > 0:
                progress.update(0, f"Fetched {completed} sites")

        # Step 2: Separate successful fetches from errors
        successful_urls = []
        successful_markdowns = []
        error_results = {}

        for url in urls:
            result = markdown_results[url]
            if isinstance(result, Exception):
                error_results[url] = result
            else:
                successful_urls.append(url)
                successful_markdowns.append(result)

        # Step 3: Batch chunk all successful documents at once
        chunk_results = {}
        if successful_markdowns:
            with self.progress_display.single_status("Chunking documents") as status:
                status.update(
                    f"Chunking {len(successful_markdowns)} documents with embeddings"
                )

                # Run batch chunking in thread pool to avoid blocking event loop
                batch_chunks = await asyncio.to_thread(
                    self.web_client.create_chunks, successful_markdowns
                )

                status.update(f"Chunked {len(successful_markdowns)} documents")

            # Step 4: Cache individual results
            for url, chunks in zip(successful_urls, batch_chunks):
                final_url = await self.cache.get_redirect_info(url)
                await self.cache.set_content(url, "chunks", chunks, final_url)
                chunk_results[url] = chunks

        # Merge error results
        chunk_results.update(error_results)

        return chunk_results

    async def prefetch_urls(
        self,
        urls: List[str],
        content_types: List[str] = None,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    ) -> None:
        """Prefetch multiple URLs for multiple content types."""
        if not urls:
            return

        if content_types is None:
            content_types = ["html", "markdown"]  # Default prefetch types

        # Calculate optimal domain width for consistent progress display
        self._domain_width = self._calculate_optimal_width(urls)

        # Create tasks for all URL/content type combinations
        tasks = []
        for content_type in content_types:
            for url in urls:
                # Only fetch if not already cached
                if not await self.cache.has_path(url, content_type):
                    fetcher_func = self._get_fetcher_for_content_type(content_type)
                    tasks.append(self._safe_fetch_single(url, fetcher_func))

        if not tasks:
            return  # Everything already cached

        # Execute with semaphore limiting
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_with_limit(task):
            async with semaphore:
                return await task

        limited_tasks = [fetch_with_limit(task) for task in tasks]

        with self.progress_display.batch_progress(
            len(limited_tasks), "Fetching and chunking documents"
        ) as progress:
            completed = 0
            for coro in asyncio.as_completed(limited_tasks):
                await coro  # Wait for completion but ignore result
                completed += 1
                progress.update(1, f"Prefetching ({completed}/{len(limited_tasks)})")

            # Update final status when all complete
            if completed > 0:
                progress.update(0, f"Prefetched {completed} documents")

    async def _safe_fetch_single(
        self, url: str, fetcher_func: Callable
    ) -> Optional[Any]:
        """Safely fetch a single URL, returning None on error."""
        try:
            return await fetcher_func(url)
        except Exception:
            return None  # Ignore errors in prefetch

    def get_cached_urls_by_type(self, content_type: str) -> List[str]:
        """Get all URLs that have cached content of the specified type."""
        # This would need to be implemented as an async method in practice
        # For now, returning empty list as this is primarily for diagnostics
        return []

    async def clear_cache_for_urls(self, urls: List[str]) -> None:
        """Clear cache for specified URLs."""
        for url in urls:
            await self.cache.clear_url(url)

    def calculate_batch_content_size(
        self, results: List[Any], content_type: str
    ) -> int:
        """Calculate total size of batch results."""
        total_size = 0
        for result in results:
            if result is not None and not isinstance(result, Exception):
                if content_type == "chunks" and isinstance(result, list):
                    total_size += sum(
                        len(chunk) if isinstance(chunk, str) else 0 for chunk in result
                    )
                elif isinstance(result, str):
                    total_size += len(result)
        return total_size

    def _calculate_optimal_width(self, urls: List[str]) -> int:
        """Calculate optimal domain width using 75th percentile of domain lengths."""
        domains = [urlparse(url).netloc for url in urls]
        domain_lengths = [len(domain) for domain in domains if domain]

        if domain_lengths:
            import statistics

            percentile_75 = (
                statistics.quantiles(domain_lengths, n=4)[2]
                if len(domain_lengths) > 1
                else domain_lengths[0]
            )
            return min(20, int(percentile_75))
        else:
            return 15

    def format_domain(self, url: str, width: Optional[int] = None) -> str:
        """Format domain name with constant width for progress display."""
        try:
            domain = urlparse(url).netloc
            if not domain:
                domain = url[:50]

            # Use provided width or instance width
            target_width = width or self._domain_width

            if len(domain) <= target_width:
                return domain.ljust(target_width)  # Pad with spaces
            else:
                return domain[: target_width - 1] + "…"  # Truncate with ellipsis
        except Exception:
            fallback = url[:50]
            target_width = width or self._domain_width
            return fallback[:target_width].ljust(target_width)


# Standalone batch functions for backward compatibility and direct use
async def fetch_urls_with_progress(
    urls: List[str],
    config,
    content_type: str,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    progress: bool = True,
) -> List[Any]:
    """Sequential batch fetching with progress bars."""
    from .cache import URLCache
    from .web_client import WebClient

    cache = URLCache(config)
    web_client = WebClient(config)
    batch_ops = BatchOperations(cache, web_client, show_status=progress)

    return await batch_ops.fetch_multiple(
        urls, content_type, max_concurrent=1
    )  # Sequential


async def fetch_urls_concurrent_with_progress(
    urls: List[str],
    config,
    content_type: str,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    progress: bool = True,
) -> List[Any]:
    """Concurrent batch fetching with semaphore limiting."""
    from .cache import URLCache
    from .web_client import WebClient

    cache = URLCache(config)
    web_client = WebClient(config)
    batch_ops = BatchOperations(cache, web_client, show_status=progress)

    return await batch_ops.fetch_multiple(
        urls, content_type, max_concurrent=max_concurrent
    )
