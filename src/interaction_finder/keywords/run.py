"""Main entry point for keyword research pipeline."""

import httpx

from interaction_finder.logging import logfire
from interaction_finder.fetcher import PageFetcher
from interaction_finder.keywords.deps import Deps
from interaction_finder.keywords.extractors import (
    KeyBERTExtractor,
    RAKEExtractor,
    TFIDFExtractor,
    YAKEExtractor,
)
from interaction_finder.keywords.graph import graph
from interaction_finder.keywords.models import BridgingTermsOut
from interaction_finder.keywords.nodes import ExpandQueryNode
from interaction_finder.keywords.reranker import Reranker
from interaction_finder.keywords.state import State
from interaction_finder.resources import ResourcePool
from interaction_finder.search.backends.pubmed import PubMedBackend
from interaction_finder.settings import IfetcherConfig


async def run_keyword_research(
    topic: str,
    config: IfetcherConfig,
    verbose: bool = False,
) -> BridgingTermsOut:
    """Run keyword research pipeline for a topic.

    Executes the complete keyword research graph: query expansion, search,
    reranking, document fetching, keyword extraction, evaluation, and reflection.

    Parameters:
        topic: str — research topic to find bridging terms for
        config: IfetcherConfig — configuration object
        verbose: bool — enable verbose logging (default: False)

    Returns:
        BridgingTermsOut — final bridging terms with metadata

    Example:
        >>> config = IfetcherConfig.from_path("config.toml")
        >>> result = await run_keyword_research("pulmonary arterial hypertension", config)
        >>> print(f"Found {len(result.terms)} bridging terms")
    """
    with logfire.span(
        "run_keyword_research",
        topic=topic,
        max_rounds=config.tools.keywords.max_rounds,
    ):
        # Extract configuration
        kw_config = config.tools.keywords
        # Create async HTTP client
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            # Initialize fetcher
            fetcher = PageFetcher(
                cache_dir=str(config.abspath(config.output.cache)),
                timeout=30,
                show_status=verbose,
                verbose=verbose,
            )
            # Initialize search backend (currently only PubMed is supported)
            # In future, add backend factory to support multiple backends
            search_backend = PubMedBackend(config={})
            # Initialize reranker if enabled (rerank_top_k > 0)
            reranker = None
            if kw_config.rerank_top_k > 0:
                reranker = Reranker(
                    model_name=kw_config.reranker_model,
                    device=kw_config.reranker_device,
                )
            # Initialize extractors
            extractors = {
                "rake": RAKEExtractor(
                    min_length=kw_config.rake.min_length,
                    max_length=kw_config.rake.max_length,
                ),
                "yake": YAKEExtractor(
                    n_grams=kw_config.yake.n_grams,
                    deduplication_threshold=kw_config.yake.deduplication_threshold,
                    window_size=kw_config.yake.window_size,
                ),
                "tfidf": TFIDFExtractor(
                    max_features=kw_config.tfidf.max_features,
                    ngram_range=kw_config.tfidf.ngram_range,
                    min_df=kw_config.tfidf.min_df,
                ),
                "keybert": KeyBERTExtractor(
                    model_name=kw_config.keybert.model_name,
                    diversity=kw_config.keybert.diversity,
                    top_n=kw_config.keybert.top_n,
                    device=kw_config.keybert.device,
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
            )
            # Create state
            state = State(topic=topic, max_rounds=kw_config.max_rounds)
            # Run graph
            result = await graph.run(ExpandQueryNode(), state=state, deps=deps)
            logfire.info(
                f"Completed: {len(result.output.terms)} bridging terms from {result.output.total_documents_processed} documents in {result.output.rounds_completed} rounds",
                terms=result.output.terms,
            )
            return result.output
