"""Convenient entry point for running widesearch pipeline.

Provides a high-level async function that handles all the setup and
orchestration, making it easy to run widesearch with minimal boilerplate.
"""

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

from interaction_finder.checkpoint import PipelineCheckpoint, SearchStageData
from interaction_finder.fetcher import PageFetcher
from interaction_finder.logging import logfire
from interaction_finder.resources import ResourcePool, compute_chunk_spans
from interaction_finder.search.models import SearchBackend, SearchResult
from interaction_finder.settings import IfetcherConfig
from interaction_finder.widesearch.deps import Deps
from interaction_finder.widesearch.graph import graph
from interaction_finder.widesearch.nodes import PlanGoalsNode
from interaction_finder.widesearch.reranker import Reranker
from interaction_finder.widesearch.state import State


async def run_widesearch(
    topic: str,
    keyphrases: list[str],
    search_backend: SearchBackend,
    *,
    resource_pool: ResourcePool | None = None,
    config: IfetcherConfig | None = None,
    max_rounds: int | None = None,
    reranker: Reranker | None = None,
    http_client: httpx.AsyncClient | None = None,
    progress: Any | None = None,
) -> list[SearchResult]:
    """Run widesearch pipeline with convenient defaults.

    High-level entry point that handles all setup and orchestration.
    Returns a list of unique SearchResult objects with URLs selected
    during the search process.

    Parameters:
        topic: str — research topic to search for
        keyphrases: list[str] — keyphrases to incorporate in queries
        search_backend: SearchBackend — search backend to use (PubMed, Perplexica, etc.)
        resource_pool: ResourcePool | None — existing resource pool (creates new if None)
        config: IfetcherConfig | None — configuration object (uses defaults if None)
        max_rounds: int | None — override max_rounds from config
        reranker: Reranker | None — pre-initialized reranker (creates new if None and rerank_top_k > 0)
        http_client: httpx.AsyncClient | None — HTTP client (creates temporary if None)
        progress: Any | None — optional progress counter for live display

    Returns:
        list[SearchResult] — unique results with URLs selected during search

    Example:
        >>> from interaction_finder.search.backends import PubMedBackend
        >>> from interaction_finder.widesearch import run_widesearch
        >>>
        >>> backend = PubMedBackend()
        >>> results = await run_widesearch(
        ...     topic="diabetes treatment",
        ...     keyphrases=["insulin", "glucose"],
        ...     search_backend=backend,
        ...     max_rounds=3,
        ... )
        >>> print(f"Found {len(results)} unique URLs")
    """
    # Load config or use defaults
    if config is None:
        config = IfetcherConfig()

    # Extract widesearch config
    ws_config = config.tools.widesearch

    # Resolve effective values (parameter overrides take precedence)
    effective_max_rounds = (
        max_rounds if max_rounds is not None else ws_config.max_rounds
    )

    # Create or use existing resource pool
    if resource_pool is None:
        resource_pool = ResourcePool()

    # Create reranker if enabled (rerank_top_k > 0) and not provided
    if reranker is None and ws_config.rerank_top_k > 0:
        reranker = Reranker(
            model_name=ws_config.reranker_model,
            device=ws_config.reranker_device,
        )

    # Handle HTTP client
    own_client = http_client is None
    if own_client:
        http_client = httpx.AsyncClient(timeout=30.0)

    try:
        # Create dependencies
        deps = Deps(
            http_client=http_client,
            search_backend=search_backend,
            reranker=reranker,
            resource_pool=resource_pool,
            config=config,
            progress=progress,
        )

        # Create initial state
        state = State(
            topic=topic,
            keyphrases=keyphrases,
            max_rounds=effective_max_rounds,
        )

        # Run the graph
        result = await graph.run(PlanGoalsNode(), state=state, deps=deps)

        return result.output

    finally:
        # Clean up HTTP client if we created it
        if own_client and http_client is not None:
            await http_client.aclose()


async def run_widesearch_with_checkpoint(
    input_checkpoint: PipelineCheckpoint,
    search_backend: SearchBackend,
    *,
    config: IfetcherConfig | None = None,
    max_rounds: int | None = None,
    reranker: Reranker | None = None,
    http_client: httpx.AsyncClient | None = None,
    progress: Any | None = None,
) -> PipelineCheckpoint:
    """Run widesearch pipeline preserving keywords data.

    Takes an input checkpoint (typically from keywords stage) and adds
    search stage data while preserving all existing data. Returns a
    unified checkpoint with both keywords and search stages.

    Parameters:
        input_checkpoint: PipelineCheckpoint — checkpoint with keywords stage data
        search_backend: SearchBackend — search backend to use (PubMed, Perplexica, etc.)
        config: IfetcherConfig | None — configuration object (uses defaults if None)
        max_rounds: int | None — override max_rounds from config
        reranker: Reranker | None — pre-initialized reranker (creates new if None and rerank_top_k > 0)
        http_client: httpx.AsyncClient | None — HTTP client (creates temporary if None)
        progress: Any | None — optional progress counter for live display

    Returns:
        PipelineCheckpoint — checkpoint with keywords + search stage data

    Example:
        >>> from interaction_finder.search.backends import PubMedBackend
        >>> backend = PubMedBackend()
        >>> # Assume keywords_checkpoint from keywords stage
        >>> checkpoint = await run_widesearch_with_checkpoint(
        ...     input_checkpoint=keywords_checkpoint,
        ...     search_backend=backend,
        ...     max_rounds=3
        ... )
        >>> print(f"Keywords: {len(checkpoint.keywords.terms)}")
        >>> print(f"Search queries: {len(checkpoint.search.queries)}")
    """
    # Extract data from input checkpoint
    topic = input_checkpoint.topic
    keyphrases = input_checkpoint.keywords.terms if input_checkpoint.keywords else []
    resource_pool = input_checkpoint.resources

    # Track starting URLs for delta calculation
    {rid.url for rid in resource_pool.resource_map.keys()}

    # Load config or use defaults
    if config is None:
        config = IfetcherConfig()

    # Extract widesearch config
    ws_config = config.tools.widesearch

    # Resolve effective values (parameter overrides take precedence)
    effective_max_rounds = (
        max_rounds if max_rounds is not None else ws_config.max_rounds
    )

    # Create reranker if enabled (rerank_top_k > 0) and not provided
    if reranker is None and ws_config.rerank_top_k > 0:
        reranker = Reranker(
            model_name=ws_config.reranker_model,
            device=ws_config.reranker_device,
        )

    # Handle HTTP client
    own_client = http_client is None
    if own_client:
        http_client = httpx.AsyncClient(timeout=30.0)

    try:
        # Create dependencies
        deps = Deps(
            http_client=http_client,
            search_backend=search_backend,
            reranker=reranker,
            resource_pool=resource_pool,
            config=config,
            progress=progress,
        )

        # Create initial state
        state = State(
            topic=topic,
            keyphrases=keyphrases,
            max_rounds=effective_max_rounds,
        )

        # Run the graph
        result = await graph.run(PlanGoalsNode(), state=state, deps=deps)

        # Build unified checkpoint preserving keywords data
        return PipelineCheckpoint(
            topic=topic,
            resources=resource_pool,  # Accumulated pool
            keywords=input_checkpoint.keywords,  # PRESERVED from input
            search=SearchStageData(
                results=result.output,
                queries=state.all_queries,
                query_results=state.selected_results,
                keyphrases=keyphrases,
                rounds_completed=state.current_round,
            ),
        )

    finally:
        # Clean up HTTP client if we created it
        if own_client and http_client is not None:
            await http_client.aclose()


async def fetch_and_populate_results(
    checkpoint: PipelineCheckpoint,
    config: IfetcherConfig,
) -> dict[str, int]:
    """Fetch content for selected results and add to ResourcePool.

    Parameters:
        checkpoint: PipelineCheckpoint — checkpoint with search results
        config: IfetcherConfig — configuration with cache directory

    Returns:
        dict[str, int] — statistics: total, fetched, cached, failed
    """
    fetcher = PageFetcher(
        cache_dir=config.abspath(config.output.cache), show_status=False
    )

    # Check if search stage data is present
    if not checkpoint.search:
        raise ValueError("Checkpoint missing search stage data")

    # Identify URLs needing content
    urls_to_fetch = []
    for result in checkpoint.search.results:
        rid = checkpoint.resources._find_resource_id(result.url)
        if rid is None:
            rid = checkpoint.resources.register(result.url)
        if checkpoint.resources.get(rid) is None:
            urls_to_fetch.append((result.url, result.title, rid))

    if not urls_to_fetch:
        return {
            "total": len(checkpoint.search.results),
            "fetched": 0,
            "cached": len(checkpoint.search.results),
            "failed": 0,
        }

    # Fetch and populate
    urls = [url for url, _, _ in urls_to_fetch]
    # Fetch documents with DOI metadata
    documents = await fetcher.fetch_documents(urls, progress=False, fail_fast=False)
    # Fetch chunks for all URLs in batch
    chunk_lists = await fetcher.get_chunks(urls, progress=False, fail_fast=False)

    fetched = 0
    for (url, title, rid), chunks in zip(urls_to_fetch, chunk_lists):
        doc = documents.get(url)
        if doc:
            # Compute chunk spans from full text and chunk texts
            chunk_spans = (
                compute_chunk_spans(doc.content_markdown, chunks) if chunks else None
            )
            checkpoint.resources.add_content(
                rid,
                title,
                doc.content_markdown,
                chunks=chunk_spans,
                doi=doc.doi,
                publication_date=doc.publication_date,
            )
            fetched += 1
        else:
            logger.warning(f"Failed to fetch content for {url}")

    return {
        "total": len(checkpoint.search.results),
        "fetched": fetched,
        "cached": len(checkpoint.search.results) - len(urls_to_fetch),
        "failed": len(urls_to_fetch) - fetched,
    }
