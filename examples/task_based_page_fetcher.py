"""
Example PageFetcher implementation using the three-stage TaskPlanner system.

This demonstrates how the PageFetcher would be dramatically simplified by using:
1. Configuration: Define content processing rules once
2. Planning: Compute optimal execution DAG for requested content
3. Execution: Run the plan with automatic parallelization and caching

The result is ultra-clean public APIs with no duplication.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Union, Optional, Any
from pathlib import Path

from src.interaction_finder.task_planner import TaskPlanner
from src.interaction_finder.fetcher import URLCache
from src.interaction_finder.fetcher.web_client import (
    _get_crawl4ai_imports,
    _get_chunker,
)
from src.interaction_finder.settings import IfetcherConfig


class TaskBasedPageFetcher:
    """
    Clean PageFetcher implementation using TaskPlanner for dependency management.

    This approach eliminates all complex single/batch logic and replaces it with
    a declarative task system that automatically handles optimization.
    """

    def __init__(self, config: IfetcherConfig, show_status: bool = True):
        self.config = config
        self.cache = URLCache(config.abspath(config.output.cache))
        self.show_status = show_status

        # Stage 1: Configuration - set up the content processing pipeline
        self.planner = TaskPlanner(max_concurrent=config.tools.crawl4ai.max_concurrent)
        self._configure_content_pipeline()

    def _configure_content_pipeline(self):
        """
        Stage 1: Configuration - Define all content processing rules.

        This defines the complete DAG of how content types depend on each other.
        Each rule specifies dependencies, execution logic, and cache checking.
        """

        # Raw content fetching
        self.planner.add_rule(
            name="html",
            dependencies=[],
            executor=self._fetch_html_task,
            output_pattern="cache/{url_hash}.html",
            description="Fetch HTML content from URL",
        )

        self.planner.add_rule(
            name="pdf",
            dependencies=[],
            executor=self._fetch_pdf_task,
            output_pattern="cache/{url_hash}.pdf",
            description="Fetch PDF content from URL",
        )

        # Processed content
        self.planner.add_rule(
            name="markdown",
            dependencies=["html"],  # Will auto-resolve to pdf for PDF URLs
            executor=self._markdown_task,
            output_pattern="cache/{url_hash}.md",
            description="Convert HTML/PDF to markdown",
        )

        self.planner.add_rule(
            name="chunks",
            dependencies=["markdown"],
            executor=self._chunks_task,
            output_pattern="cache/{url_hash}.chunks",
            description="Split markdown into semantic chunks",
        )

        # Metadata extraction
        self.planner.add_rule(
            name="doi",
            dependencies=["html"],
            executor=self._doi_task,
            output_pattern="cache/{url_hash}.doi",
            description="Extract DOI from HTML content",
        )

        # Raw content selector
        self.planner.add_rule(
            name="raw",
            dependencies=[
                "html",
                "pdf",
            ],  # Depends on both, will select appropriate one
            executor=self._raw_task,
            output_pattern=None,  # Raw is computed, not cached
            description="Select raw HTML or PDF content based on URL type",
        )

    # =============================================================================
    # Task Executors - Simple, focused functions
    # =============================================================================

    async def _fetch_html_task(self, context: Dict[str, Any]) -> str:
        """Fetch HTML content for URL and return path to cached file."""
        url = context["url"]
        url_hash = self._url_to_hash(url)

        # Check if already cached
        if await self.cache.has_path(url, "html"):
            cache_dir = self.config.abspath(self.config.output.cache)
            return str(cache_dir / f"{url_hash}.html")

        # Fetch and cache
        await self._fetch_and_cache(url)
        cache_dir = self.config.abspath(self.config.output.cache)
        return str(cache_dir / f"{url_hash}.html")

    async def _fetch_pdf_task(self, context: Dict[str, Any]) -> str:
        """Fetch PDF content for URL and return path to cached file."""
        url = context["url"]
        url_hash = self._url_to_hash(url)

        # Check if already cached
        if await self.cache.has_path(url, "pdf"):
            cache_dir = self.config.abspath(self.config.output.cache)
            return str(cache_dir / f"{url_hash}.pdf")

        # Fetch and cache
        await self._fetch_and_cache(url)
        cache_dir = self.config.abspath(self.config.output.cache)
        return str(cache_dir / f"{url_hash}.pdf")

    async def _markdown_task(self, context: Dict[str, Any]) -> str:
        """Convert HTML/PDF to markdown and return path to cached file."""
        url = context["url"]
        url_hash = self._url_to_hash(url)

        cache_dir = self.config.abspath(self.config.output.cache)
        md_path = cache_dir / f"{url_hash}.md"

        if await self.cache.has_path(url, "markdown"):
            return str(md_path)

        # Markdown generation happens during fetch_and_cache
        # This executor is mainly for dependency ordering
        await self._fetch_and_cache(url)
        return str(md_path)

    async def _chunks_task(self, context: Dict[str, Any]) -> str:
        """Create chunks from markdown content and return path to cached file."""
        url = context["url"]
        url_hash = self._url_to_hash(url)

        cache_dir = self.config.abspath(self.config.output.cache)
        chunks_path = cache_dir / f"{url_hash}.chunks"

        if await self.cache.has_path(url, "chunks"):
            return str(chunks_path)

        # Get markdown from dependencies (file-based)
        dependency_paths = context["dependency_paths"]
        markdown_path = dependency_paths.get("markdown")

        if markdown_path and Path(markdown_path).exists():
            markdown_content = Path(markdown_path).read_text()
        else:
            # Fallback to loaded dependency
            markdown_content = context["dependencies"]["markdown"]

        # Create chunks using existing chunker
        chunker = _get_chunker()
        chunks = chunker(markdown_content)
        chunk_texts = [chunk.text for chunk in chunks]

        # Cache the result
        await self.cache.set_chunks(url, chunk_texts)
        return str(chunks_path)

    async def _doi_task(self, context: Dict[str, Any]) -> Optional[str]:
        """Extract DOI from HTML content and return path to cached file."""
        url = context["url"]
        url_hash = self._url_to_hash(url)

        cache_dir = self.config.abspath(self.config.output.cache)
        doi_path = cache_dir / f"{url_hash}.doi"

        if await self.cache.has_path(url, "doi"):
            return str(doi_path) if doi_path.exists() else None

        # DOI extraction happens during fetch_and_cache
        doi = await self.cache.get_doi(url)
        if doi:
            # Save DOI to file for consistency
            doi_path.write_text(doi)
            return str(doi_path)
        return None

    async def _raw_task(self, context: Dict[str, Any]) -> str:
        """Select appropriate raw content path based on URL type."""
        url = context["url"]
        dependency_paths = context["dependency_paths"]

        if self._is_pdf_url(url):
            pdf_path = dependency_paths.get("pdf")
            return pdf_path if pdf_path else ""
        else:
            html_path = dependency_paths.get("html")
            return html_path if html_path else ""

    # =============================================================================
    # Helper methods
    # =============================================================================

    # =============================================================================
    # Public API - Ultra-clean implementations
    # =============================================================================

    async def get_html(
        self, url: Union[str, List[str]], progress: bool = True
    ) -> Union[str, List[str]]:
        """Get HTML content for single URL or multiple URLs."""
        return await self._get_content(url, "html", progress)

    async def get_pdf(
        self, url: Union[str, List[str]], progress: bool = True
    ) -> Union[str, List[str]]:
        """Get PDF content for single URL or multiple URLs."""
        return await self._get_content(url, "pdf", progress)

    async def get_markdown(
        self, url: Union[str, List[str]], progress: bool = True
    ) -> Union[str, List[str]]:
        """Get markdown content for single URL or multiple URLs."""
        return await self._get_content(url, "markdown", progress)

    async def get_chunks(
        self, url: Union[str, List[str]], progress: bool = True
    ) -> Union[List[str], List[List[str]]]:
        """Get chunked content for single URL or multiple URLs."""
        return await self._get_content(url, "chunks", progress)

    async def get_raw(self, url: Union[str, List[str]]) -> Union[str, List[str]]:
        """Get raw content for single URL or multiple URLs."""
        return await self._get_content(url, "raw", False)

    async def get_doi(
        self, url: Union[str, List[str]]
    ) -> Union[Optional[str], List[Optional[str]]]:
        """Get DOI for single URL or multiple URLs."""
        return await self._get_content(url, "doi", False)

    async def _get_content(
        self, url: Union[str, List[str]], content_type: str, progress: bool
    ):
        """
        Generic content getter using TaskPlanner.

        This is the only method that handles single vs. batch logic.
        Everything else is handled by the task planning system.
        """
        if isinstance(url, str):
            # Stage 2: Planning - create execution plan for single URL
            context = {"url": url, "url_hash": self._url_to_hash(url)}
            task_list = self.planner.create_task_list([content_type], context)

            # Stage 3: Execution
            progress_callback = self._progress_callback if progress else None
            if progress:
                results = await task_list.execute_parallel(progress_callback)
            else:
                results = await task_list.execute_parallel()

            if content_type not in results:
                failed = task_list.get_failed_tasks()
                if content_type in failed:
                    raise RuntimeError(
                        f"Failed to get {content_type} for {url}: {failed[content_type]}"
                    )
                else:
                    raise RuntimeError(f"No result for {content_type}")

            return results[content_type]

        elif isinstance(url, list):
            # Batch processing: create tasks for each URL and run concurrently
            tasks = [
                self._get_content(single_url, content_type, False) for single_url in url
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions and return successful results
            return [r for r in results if not isinstance(r, Exception)]

        else:
            raise TypeError(f"url must be str or list[str], got {type(url)}")

    # =============================================================================
    # Advanced Features
    # =============================================================================

    async def prefetch_multiple(
        self, urls: List[str], content_types: List[str]
    ) -> Dict[str, List[str]]:
        """
        Efficiently prefetch multiple content types for multiple URLs.

        This creates an optimal execution plan across all URL/content-type combinations.
        Returns paths to cached files instead of content.
        """
        all_tasks = []
        url_content_pairs = []

        for url in urls:
            for content_type in content_types:
                all_tasks.append(self._get_content(url, content_type, False))
                url_content_pairs.append((url, content_type))

        # Execute all combinations concurrently
        results = await asyncio.gather(*all_tasks, return_exceptions=True)

        # Organize results by content type (now file paths)
        organized = {ct: [] for ct in content_types}
        for i, result in enumerate(results):
            _, content_type = url_content_pairs[i]
            if not isinstance(result, Exception):
                organized[content_type].append(result)

        return organized

    def explain_dependencies(self, content_type: str) -> str:
        """Show what will be executed to get the requested content type."""
        context = {"url": "https://example.com", "url_hash": "example"}
        task_list = self.planner.create_task_list([content_type], context)
        return task_list.visualize()

    def get_available_content_types(self) -> List[str]:
        """Get list of all available content types."""
        return self.planner.list_rules()

    # =============================================================================
    # Helper methods
    # =============================================================================

    def _url_to_hash(self, url: str) -> str:
        """Create a hash for URL (for cache file naming)."""
        import hashlib

        return hashlib.md5(url.encode()).hexdigest()[:10]

    def _is_pdf_url(self, url: str) -> bool:
        """Check if URL points to a PDF file."""
        return url.lower().endswith(".pdf")

    async def _progress_callback(
        self, task_name: str, status: str, completed: int, total: int
    ):
        """Progress callback for task execution."""
        if self.show_status:
            status_emoji = {"cached": "💾", "completed": "✅", "failed": "❌"}.get(
                status, "🔄"
            )
            print(f"  {status_emoji} {task_name} ({completed}/{total})")

    async def _fetch_and_cache_raw(self, url: str, context: Dict[str, Any]):
        """
        Fetch raw content and generate all derived formats.

        This reuses the existing crawl4ai integration logic.
        """
        # Import crawl4ai components
        crawl4ai_module = _get_crawl4ai_imports()

        # Use existing fetch logic (simplified for example)
        # In real implementation, this would use the full crawl4ai pipeline
        # from the original fetcher.py

        # Mock implementation for demonstration
        if self._is_pdf_url(url):
            # Would fetch PDF and convert to text
            await self.cache.set_pdf(url, f"PDF content for {url}")
        else:
            # Would fetch HTML and convert to markdown
            await self.cache.set_html(url, f"<html>Content for {url}</html>")
            await self.cache.set_markdown(url, f"# Content for {url}")


# =============================================================================
# Usage Examples
# =============================================================================


async def example_usage():
    """Examples showing how clean the new API is."""

    config = IfetcherConfig()
    fetcher = TaskBasedPageFetcher(config)

    # Same external API as before - but now returns file paths for better memory efficiency
    chunks_path = await fetcher.get_chunks("https://example.com")
    all_chunk_paths = await fetcher.get_chunks(
        ["https://example.com", "https://test.com"]
    )

    # Load content from files when needed
    if isinstance(chunks_path, str):
        chunks_content = Path(chunks_path).read_text()
        print("Chunks loaded from:", chunks_path)

    # 1. Explain what will happen
    print(fetcher.explain_dependencies("chunks"))
    # Output:
    # Execution Plan:
    #   Level 1: html (no dependencies)
    #   Level 2: markdown
    #   Level 3: chunks

    # 2. Efficient multi-content prefetching - returns file paths
    file_paths = await fetcher.prefetch_multiple(
        urls=["https://example.com", "https://test.com"],
        content_types=["html", "markdown", "chunks"],
    )
    print("Cached files:", file_paths)

    # 3. See what content types are available
    print("Available:", fetcher.get_available_content_types())
    # Output: ['html', 'pdf', 'markdown', 'chunks', 'doi', 'raw']


# =============================================================================
# Key Benefits of This Architecture
# =============================================================================

"""
1. **Declarative Configuration**: Define the content pipeline once in _configure_content_pipeline()

2. **Automatic Optimization**: TaskPlanner handles:
   - Dependency resolution (chunks needs markdown needs html)
   - Parallel execution of independent tasks
   - Minimal work (only computes what's needed)
   - Intelligent caching

3. **Ultra-Clean Public API**: Every get_X method is now 1 line:
   return await self._get_content(url, "content_type", progress)

4. **No Duplication**: Each content type has exactly one implementation

5. **Type Safety**: No string-based method dispatch, everything is function references

6. **Extensible**: Add new content types by just adding a rule to the planner

7. **Debuggable**: Can visualize execution plans and see exactly what will run

8. **Flexible**: Can easily request any combination of content types for any URLs

9. **Efficient**: Automatically finds optimal execution order and parallelization

10. **Consistent**: Single and batch operations use identical logic

11. **Memory Efficient**: Returns file paths instead of content, avoiding memory bloat

12. **Persistent**: Results cached to disk survive process restarts

The result is a PageFetcher that's dramatically simpler but much more powerful.
The complex dependency management is handled by the TaskPlanner, and the file-based
approach ensures memory efficiency and persistence.
"""

if __name__ == "__main__":
    asyncio.run(example_usage())
