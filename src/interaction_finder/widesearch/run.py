"""Convenient entry point for running widesearch pipeline.

Provides a high-level async function that handles all the setup and
orchestration, making it easy to run widesearch with minimal boilerplate.
"""

from typing import Any

import httpx

from interaction_finder.resources import ResourcePool
from interaction_finder.search.models import SearchBackend, SearchResult
from interaction_finder.settings import IfetcherConfig
from interaction_finder.widesearch.deps import Deps
from interaction_finder.widesearch.graph import graph
from interaction_finder.widesearch.models import WidesearchCheckpoint
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
    enable_reranking: bool | None = None,
    http_client: httpx.AsyncClient | None = None,
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
        enable_reranking: bool | None — override enable_reranking from config
        http_client: httpx.AsyncClient | None — HTTP client (creates temporary if None)

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
        ...     enable_reranking=False
        ... )
        >>> print(f"Found {len(results)} unique URLs")
    """
    # Load config or use defaults
    if config is None:
        config = IfetcherConfig()

    # Extract widesearch config
    ws_config = config.tools.widesearch

    # Build config dict for pipeline
    pipeline_config: dict[str, Any] = {
        "results_per_query": ws_config.results_per_query,
        "rerank_top_k": ws_config.rerank_top_k,
        "enable_reranking": (
            enable_reranking
            if enable_reranking is not None
            else ws_config.enable_reranking
        ),
    }

    # Determine max_rounds
    effective_max_rounds = (
        max_rounds if max_rounds is not None else ws_config.max_rounds
    )

    # Create or use existing resource pool
    if resource_pool is None:
        resource_pool = ResourcePool()

    # Create reranker only if reranking is enabled
    # This avoids loading the model when not needed and allows running
    # without torch/sentence-transformers dependencies
    reranker = None
    if pipeline_config["enable_reranking"]:
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
            config=pipeline_config,
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
    topic: str,
    keyphrases: list[str],
    search_backend: SearchBackend,
    *,
    resource_pool: ResourcePool | None = None,
    config: IfetcherConfig | None = None,
    max_rounds: int | None = None,
    enable_reranking: bool | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> WidesearchCheckpoint:
    """Run widesearch pipeline and return complete checkpoint.

    Identical interface to run_widesearch() but returns a WidesearchCheckpoint
    containing full state including results, queries, query-to-URL mappings,
    and the complete resource pool. Useful for checkpointing searches or
    passing complete state to downstream processes.

    Parameters:
        topic: str — research topic to search for
        keyphrases: list[str] — keyphrases to incorporate in queries
        search_backend: SearchBackend — search backend to use (PubMed, Perplexica, etc.)
        resource_pool: ResourcePool | None — existing resource pool (creates new if None)
        config: IfetcherConfig | None — configuration object (uses defaults if None)
        max_rounds: int | None — override max_rounds from config
        enable_reranking: bool | None — override enable_reranking from config
        http_client: httpx.AsyncClient | None — HTTP client (creates temporary if None)

    Returns:
        WidesearchCheckpoint — complete checkpoint with results, queries, and resource pool

    Example:
        >>> from interaction_finder.search.backends import PubMedBackend
        >>> from interaction_finder.widesearch import run_widesearch_with_checkpoint
        >>>
        >>> backend = PubMedBackend()
        >>> checkpoint = await run_widesearch_with_checkpoint(
        ...     topic="diabetes treatment",
        ...     keyphrases=["insulin", "glucose"],
        ...     search_backend=backend,
        ...     max_rounds=3
        ... )
        >>> print(f"Executed {len(checkpoint.queries)} queries")
        >>> print(f"Found {len(checkpoint.results)} unique results")
    """
    # Load config or use defaults
    if config is None:
        config = IfetcherConfig()

    # Extract widesearch config
    ws_config = config.tools.widesearch

    # Build config dict for pipeline
    pipeline_config: dict[str, Any] = {
        "results_per_query": ws_config.results_per_query,
        "rerank_top_k": ws_config.rerank_top_k,
        "enable_reranking": (
            enable_reranking
            if enable_reranking is not None
            else ws_config.enable_reranking
        ),
    }

    # Determine max_rounds
    effective_max_rounds = (
        max_rounds if max_rounds is not None else ws_config.max_rounds
    )

    # Create or use existing resource pool
    if resource_pool is None:
        resource_pool = ResourcePool()

    # Create reranker only if reranking is enabled
    reranker = None
    if pipeline_config["enable_reranking"]:
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
            config=pipeline_config,
        )

        # Create initial state
        state = State(
            topic=topic,
            keyphrases=keyphrases,
            max_rounds=effective_max_rounds,
        )

        # Run the graph
        result = await graph.run(PlanGoalsNode(), state=state, deps=deps)

        # Build checkpoint from final state
        checkpoint = WidesearchCheckpoint(
            results=result.output,
            queries=state.all_queries,
            query_results=state.selected_results,
            resources=resource_pool,
            topic=state.topic,
            keyphrases=state.keyphrases,
            rounds_completed=state.current_round,
        )

        return checkpoint

    finally:
        # Clean up HTTP client if we created it
        if own_client and http_client is not None:
            await http_client.aclose()
