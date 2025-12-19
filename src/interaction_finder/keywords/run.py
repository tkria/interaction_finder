"""Main entry point for keyword research pipeline."""

from typing import Any

import httpx

from interaction_finder.logging import get_logger

logger = get_logger(__name__)

from interaction_finder.checkpoint import KeywordsStageData, PipelineCheckpoint
from interaction_finder.usage import PipelineUsage
from interaction_finder.version import get_version_string
from interaction_finder.fetcher import PageFetcher
from interaction_finder.keywords.deps import Deps
from interaction_finder.keywords.extractors import (
    KeyBERTExtractor,
    RAKEExtractor,
    TFIDFExtractor,
    YAKEExtractor,
)
from interaction_finder.keywords.graph import graph
from interaction_finder.keywords.nodes import ExpandQueryNode
from interaction_finder.keywords.reranker import Reranker
from interaction_finder.keywords.state import State
from interaction_finder.logging import logfire
from interaction_finder.resources import ResourcePool
from interaction_finder.search.backends.pubmed import PubMedBackend
from interaction_finder.settings import IfetcherConfig


async def run_keyword_research(
    topic: str,
    config: IfetcherConfig,
    search_backend: Any | None = None,
    verbose: bool = False,
    progress: Any | None = None,
) -> PipelineCheckpoint:
    """Run keyword research pipeline for a topic.

    Executes the complete keyword research graph: query expansion, search,
    reranking, document fetching, keyword extraction, evaluation, and reflection.

    Parameters:
        topic: str — research topic to find bridging terms for
        config: IfetcherConfig — configuration object
        search_backend: SearchBackend | None — search backend to use (creates PubMedBackend if None)
        verbose: bool — enable verbose logging (default: False)
        progress: StatusTable | None — progress tracking object

    Returns:
        PipelineCheckpoint — checkpoint with keywords stage data

    Example:
        >>> config = IfetcherConfig.from_path("config.toml")
        >>> checkpoint = await run_keyword_research("pulmonary arterial hypertension", config)
        >>> print(f"Found {len(checkpoint.keywords.terms)} bridging terms")
    """
    with logfire.span(
        "run_keyword_research",
        topic=topic,
        max_rounds=config.stage.keywords.max_rounds,
    ):
        # Extract configuration (stage settings and tool/algorithm settings)
        stage_config = config.stage.keywords
        tool_config = config.tools.keywords
        # Create async HTTP client
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            # Initialize fetcher
            fetcher = PageFetcher(
                cache_dir=str(config.abspath(config.output.cache)),
                timeout=30,
                show_status=verbose,
                verbose=verbose,
            )
            # Initialize search backend (use provided or create PubMed default)
            if search_backend is None:
                search_backend = PubMedBackend(
                    config={"timeout": config.tools.search.timeout}
                )
            # Initialize reranker if enabled (rerank_top_k > 0)
            reranker = None
            if stage_config.rerank_top_k > 0:
                reranker = Reranker(
                    model_name=stage_config.reranker_model,
                    device=stage_config.reranker_device,
                )
            # Initialize extractors (algorithm configs from tools.keywords)
            extractors = {
                "rake": RAKEExtractor(
                    min_length=tool_config.rake.min_length,
                    max_length=tool_config.rake.max_length,
                ),
                "yake": YAKEExtractor(
                    n_grams=tool_config.yake.n_grams,
                    deduplication_threshold=tool_config.yake.deduplication_threshold,
                    window_size=tool_config.yake.window_size,
                ),
                "tfidf": TFIDFExtractor(
                    max_features=tool_config.tfidf.max_features,
                    ngram_range=tool_config.tfidf.ngram_range,
                    min_df=tool_config.tfidf.min_df,
                ),
                "keybert": KeyBERTExtractor(
                    model_name=tool_config.keybert.model_name,
                    diversity=tool_config.keybert.diversity,
                    top_n=tool_config.keybert.top_n,
                    device=tool_config.keybert.device,
                ),
            }
            # Initialize resource pool
            resource_pool = ResourcePool()
            # Create dependencies
            deps = Deps(
                http_client=http_client,
                fetcher=fetcher,
                search_backend=search_backend,
                reranker=reranker,
                extractors=extractors,
                resource_pool=resource_pool,
                config=config,
                progress=progress,
            )
            # Create state
            state = State(topic=topic, max_rounds=stage_config.max_rounds)
            # Run graph
            result = await graph.run(ExpandQueryNode(), state=state, deps=deps)
            logger.info(
                f"Completed: {len(result.output.terms)} bridging terms from {result.output.total_documents_processed} documents in {result.output.rounds_completed} rounds",
                terms=result.output.terms,
            )

            # Snapshot resource URLs at end of keywords stage
            resource_urls = [rid.url for rid in resource_pool.resource_map.keys()]

            # Build usage from deps
            usage = PipelineUsage(keywords=deps.usage)

            # Build unified checkpoint with keywords stage data
            return PipelineCheckpoint(
                topic=topic,
                resources=resource_pool,
                created_by=get_version_string(),
                usage=usage,
                keywords=KeywordsStageData(
                    terms=result.output.terms,
                    scores=result.output.scores,
                    total_documents_processed=result.output.total_documents_processed,
                    rounds_completed=result.output.rounds_completed,
                    coverage_assessment=result.output.coverage_assessment,
                    resource_urls=resource_urls,
                ),
            )
