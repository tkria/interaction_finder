"""High-level PageFetcher interface with eliminated duplication using higher-order functions."""

from dataclasses import dataclass
from typing import Union, List, Optional, Callable, Any, TYPE_CHECKING, Dict
from .cache import URLCache
from .web_client import WebClient
from .batch_operations import BatchOperations
from .content_processor import ContentProcessor

if TYPE_CHECKING:
    from ..settings import IfetcherConfig


@dataclass
class FetchedDocument:
    """Document fetched from URL with markdown content and metadata."""

    url: str
    content_markdown: str
    source_type: Optional[str] = None  # "html" or "pdf"
    doi: Optional[str] = None


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
                # Fetch additional metadata
                source_type = await self.get_source_type(url)
                doi = None
                if source_type == "html":
                    doi = await self.get_doi(url, retry=False)

                documents[url] = FetchedDocument(
                    url=url,
                    content_markdown=markdown,
                    source_type=source_type,
                    doi=doi,
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

    def get_clustering_log(self) -> Optional[List[Dict]]:
        """Get the operation log from the last clustering operation."""
        return getattr(self, "_last_clustering_log", None)

    def _print_clustering_log(self, log: List[Dict]) -> None:
        """Print detailed clustering operation log for debugging."""
        print("\n=== CLUSTERING OPERATION LOG ===")

        for entry in log:
            entry_type = entry["type"]

            if entry_type == "initialization":
                print(
                    f"Initialization: {entry['total_docs']} docs, avg similarity: {entry['avg_similarity']:.3f}"
                )
                print(
                    f"Constraints: {entry['constraints']['min_size']}-{entry['constraints']['max_size']}"
                )

                # Display similarity distribution and adaptive thresholds
                if entry.get("similarity_stats"):
                    stats = entry["similarity_stats"]
                    print(f"Similarity distribution:")
                    print(
                        f"  Min: {stats['min']:.3f}, Q1: {stats['q25']:.3f}, Median: {stats['median']:.3f}"
                    )
                    print(
                        f"  Q3: {stats['q75']:.3f}, Max: {stats['max']:.3f}, Std: {stats['std']:.3f}"
                    )
                    if entry.get("adaptive_thresholds"):
                        thresholds = entry["adaptive_thresholds"]
                        print(
                            f"Adaptive thresholds: Q3={thresholds[0]:.3f}, Median={thresholds[1]:.3f}, Q1={thresholds[2]:.3f}"
                        )

            elif entry_type == "phase1_merge":
                threshold_info = (
                    f" @{entry.get('threshold_level', 'unknown')}"
                    if entry.get("threshold_level")
                    else ""
                )
                helps_info = (
                    " (helps undersized)" if entry.get("helps_undersized") else ""
                )
                print(
                    f"  Merge {entry['clusters'][0]}+{entry['clusters'][1]}: [{entry['sizes'][0]}]+[{entry['sizes'][1]}] → [{sum(entry['sizes'])}], sim={entry['similarity_score']:.3f}{threshold_info}{helps_info}"
                )

            elif entry_type == "threshold_lowered":
                print(
                    f"  → Lowering threshold from {entry['from_level']} ({entry['from_threshold']:.3f}) - {entry['remaining_undersized']} clusters still undersized"
                )

            elif entry_type == "phase1_emergency_fallback":
                print(
                    f"  → Emergency fallback: {len(entry['remaining_undersized'])} clusters still below min_size"
                )

            elif entry_type == "phase1_emergency_merge":
                print(
                    f"  Emergency merge {entry['clusters'][0]}+{entry['clusters'][1]}: [{entry['sizes'][0]}]+[{entry['sizes'][1]}] → [{sum(entry['sizes'])}], sim={entry['similarity_score']:.3f}"
                )

            elif entry_type == "phase1_complete":
                print(f"\nPhase 1 Complete: {entry['merges_performed']} merges")
                print(
                    f"Clusters: {entry['total_clusters']} total, {entry['clusters_at_min_size']} at min_size"
                )
                print(f"Cluster sizes: {entry['cluster_sizes']}")

            elif entry_type == "phase2_iteration":
                print(f"\nPhase 2 Iteration {entry['iteration']}:")
                print(
                    f"  Clusters: {entry['total_clusters']}, Sizes: {entry['cluster_sizes']}"
                )
                print(
                    f"  Candidates: {entry['candidates_evaluated']} total, {entry['candidates_valid']} valid"
                )
                print(
                    f"  Rejected: {entry['candidates_rejected_size']} size, {entry['candidates_rejected_modularity']} modularity"
                )
                print(
                    f"  Best ΔQ: {entry['best_delta_q']:.4f}, pair sizes: {entry['best_pair_sizes']}"
                )
                if entry.get("modularity_distribution"):
                    mod_dist = entry["modularity_distribution"]
                    print(
                        f"  Modularity range: [{min(mod_dist):.4f}, {max(mod_dist):.4f}]"
                    )

                # Show top modularity candidates for detailed insight
                if entry.get("modularity_values"):
                    mod_values = entry["modularity_values"]
                    # Sort by delta_q descending and show top 3
                    sorted_mods = sorted(
                        mod_values, key=lambda x: x["delta_q"], reverse=True
                    )[:3]
                    print("  Top modularity candidates:")
                    for mv in sorted_mods:
                        status = (
                            "✓" if not mv.get("rejected") else f"✗ ({mv['rejected']})"
                        )
                        print(
                            f"    [{mv['sizes'][0]}]+[{mv['sizes'][1]}]: ΔQ={mv['delta_q']:.4f} {status}"
                        )

            elif entry_type == "phase2_stopping":
                print(f"\nPhase 2 Stopped: {entry['reason']}")
                print(
                    f"Final: {entry['final_clusters']} clusters, ΔQ: {entry['final_delta_q']:.4f}"
                )
                print(f"Final sizes: {entry['final_cluster_sizes']}")

            elif entry_type == "clustering_complete":
                print(f"\n=== CLUSTERING SUMMARY ===")
                print(f"Phase 1 merges: {entry['total_phase1_merges']}")
                print(f"Phase 2 merges: {entry['total_phase2_merges']}")
                print(f"Final groups: {entry['final_groups']}")
                print(
                    f"Groups at max size ({entry.get('max_size', 'N/A')}): {entry['groups_at_max_size']}"
                )
                print(
                    f"Groups at min size ({entry.get('min_size', 'N/A')}): {entry['groups_at_min_size']}"
                )
                print(f"Group sizes: {entry['final_group_sizes']}")

        print("=== END CLUSTERING LOG ===\n")

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
        clustering_method: str = "agglomerative",
        embedding_weights: str = "uniform",
        seeding_method: str = "kmeans",
        refinement_method: str = "hierarchical",
        prefetch: bool = True,
        progress: bool = True,
        retry: bool = False,
        fail_fast: bool = True,
        include_chunks: bool = False,
        return_metadata: bool = False,
        **kwargs,
    ):
        """
        Group documents by semantic similarity with size constraints.

        Args:
            urls: List of URLs to group
            constraint_type: Either "count" or "words"
            min_size: Minimum group size (documents or words)
            max_size: Maximum group size (documents or words)
            linkage_method: Clustering linkage method ("average", "complete", "single")
            clustering_method: Clustering algorithm ("agglomerative", "spectral", "hybrid", "random")
            embedding_weights: Document embedding weights ("uniform", "idf")
            seeding_method: Spectral seeding method for hybrid clustering ("kmeans", "fiedler")
            refinement_method: Refinement method for hybrid clustering ("hierarchical", "agglomerative")
            prefetch: Whether to fetch missing documents
            progress: Show progress bar
            retry: Retry failed fetches
            return_metadata: If True, returns (groups, metadata) tuple instead of just groups

        Returns:
            If return_metadata=False: List of group dictionaries with:
            - documents: List of URLs in group
            - total_words: Combined word count
            - cohesion_score: Average pairwise similarity

            If return_metadata=True: Tuple of (groups, clustering_metadata) where
            clustering_metadata contains comprehensive metrics
        """
        if not urls:
            return []

        # Validate constraint parameters
        if constraint_type not in ("count", "words"):
            raise ValueError("constraint_type must be 'count' or 'words'")
        if min_size < 1:
            raise ValueError("min_size must be >= 1")
        if min_size > max_size:
            raise ValueError("min_size must be <= max_size")

        # Get chunk data for all documents using batch operations
        # Always use fail_fast=False for chunk fetching to allow partial processing
        # No separate prefetch needed - get_chunks_with_embeddings handles caching internally
        chunks_results = await self.get_chunks_with_embeddings(
            urls, progress=progress, fail_fast=False, retry=retry
        )

        # Build doc_chunks dict from results
        doc_chunks = {}
        doc_word_counts = {}
        for url, chunks_data in zip(urls, chunks_results):
            # Skip failed results (exceptions) and None results
            if chunks_data is not None and not isinstance(chunks_data, Exception):
                if isinstance(chunks_data, list):
                    # Direct list of chunks
                    doc_chunks[url] = chunks_data
                    doc_word_counts[url] = sum(
                        chunk["wordcount"] for chunk in chunks_data
                    )
                elif isinstance(chunks_data, dict) and "chunks" in chunks_data:
                    # Wrapped in dict with "chunks" key
                    chunks_list = chunks_data["chunks"]
                    doc_chunks[url] = chunks_list
                    doc_word_counts[url] = sum(
                        chunk["wordcount"] for chunk in chunks_list
                    )

        # Filter out documents without chunks
        valid_urls = [url for url in urls if url in doc_chunks]

        if not valid_urls:
            if return_metadata:
                return [], {}
            return []

        if len(valid_urls) == 1:
            # Single document - return as single group
            url = valid_urls[0]
            doc_embeddings = self._compute_document_embeddings({url: doc_chunks[url]})
            formatted_groups = [
                {
                    "documents": [url],
                    "total_words": doc_word_counts[url],
                    "cohesion_score": 1.0,
                }
            ]
            if return_metadata:
                return formatted_groups, {}
            return formatted_groups

        # Group documents using specified clustering method
        grouper = DocumentGrouper(
            linkage_method=linkage_method,
            clustering_method=clustering_method,
            embedding_weights=embedding_weights,
            seeding_method=seeding_method,
            refinement_method=refinement_method,
            **kwargs,
        )

        if len(valid_urls) > 1:
            # Create single status object for entire process
            if progress:
                with self.batch_ops.progress_display.create_status(
                    "Computing document similarities and grouping..."
                ) as status:
                    result = grouper.group_documents_with_details(
                        documents=valid_urls,
                        chunk_data=doc_chunks,
                        constraint_type=constraint_type,
                        min_size=min_size,
                        max_size=max_size,
                        clustering_method=clustering_method,
                        embedding_weights=embedding_weights,
                        seeding_method=seeding_method,
                        refinement_method=refinement_method,
                        status=status,
                        **kwargs,
                    )
                groups = result.groups
                clustering_result = result

                # Compute embeddings for cohesion (reuse existing computation)
                from .content_processor import convert_legacy_chunk_data

                typed_chunks = convert_legacy_chunk_data(doc_chunks)
                doc_embeddings = grouper.embedder.compute_embeddings(
                    valid_urls, typed_chunks
                )
            else:
                result = grouper.group_documents_with_details(
                    documents=valid_urls,
                    chunk_data=doc_chunks,
                    constraint_type=constraint_type,
                    min_size=min_size,
                    max_size=max_size,
                    clustering_method=clustering_method,
                    embedding_weights=embedding_weights,
                    seeding_method=seeding_method,
                    refinement_method=refinement_method,
                    **kwargs,
                )
                groups = result.groups
                clustering_result = result

                # Compute embeddings for cohesion
                from .content_processor import convert_legacy_chunk_data

                typed_chunks = convert_legacy_chunk_data(doc_chunks)
                doc_embeddings = grouper.embedder.compute_embeddings(
                    valid_urls, typed_chunks
                )
        else:
            result = grouper.group_documents_with_details(
                documents=valid_urls,
                chunk_data=doc_chunks,
                constraint_type=constraint_type,
                min_size=min_size,
                max_size=max_size,
                clustering_method=clustering_method,
                embedding_weights=embedding_weights,
                seeding_method=seeding_method,
                refinement_method=refinement_method,
                **kwargs,
            )
            groups = result.groups
            clustering_result = result
            from .content_processor import convert_legacy_chunk_data

            typed_chunks = convert_legacy_chunk_data(doc_chunks)
            doc_embeddings = grouper.embedder.compute_embeddings(
                valid_urls, typed_chunks
            )

        # Format results
        formatted_groups = []
        for group in groups:
            total_words = sum(doc_word_counts.get(url, 0) for url in group)
            cohesion = compute_group_cohesion(group, doc_embeddings)

            group_data = {
                "documents": group,
                "total_words": total_words,
                "cohesion_score": cohesion,
            }

            # Include chunks if requested
            if include_chunks:
                group_data["chunks"] = {
                    url: doc_chunks[url] for url in group if url in doc_chunks
                }

            formatted_groups.append(group_data)

        # Add operation log to result for debugging clustering behavior
        if "clustering_result" in locals() and hasattr(
            clustering_result, "operation_log"
        ):
            # Store the operation log in a way that can be accessed later
            self._last_clustering_log = clustering_result.operation_log
            # Print the log for debugging (only in verbose mode)
            if self.verbose:
                self._print_clustering_log(clustering_result.operation_log)

        # Return groups with optional metadata
        if return_metadata and "clustering_result" in locals():
            return formatted_groups, clustering_result.metadata
        else:
            return formatted_groups

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


# Export the class and dataclass
__all__ = ["PageFetcher", "FetchedDocument"]
