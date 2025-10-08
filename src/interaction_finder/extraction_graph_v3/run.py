"""
Main pipeline entry point for extraction graph V3.

Provides run_extraction_v3() as the public API for running the complete
pipeline with checkpoint save/resume, resource loading, and result formatting.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Callable

from pydantic_graph import GraphRunContext

from .graph import extraction_graph_v3
from .state import ExtractionStateV3
from .deps import ExtractionDepsV3
from .models import BatchExtractionResultV3, ExtractionMetadata, PairCandidate
from .cache import SemanticCacheManager
from ..extraction_graph_v2.models import EntityWithQuotes, IndividualAssessment
from ..settings import IfetcherConfig
from ..fetcher import PageFetcher
from ..resources import ResourcePool, compute_chunk_spans

logger = logging.getLogger(__name__)


async def _load_documents(urls: List[str], page_fetcher: PageFetcher) -> ResourcePool:
    """
    Load documents from URLs into ResourcePool.

    Fetches chunks and full document for each URL, computes chunk spans,
    and adds to resource pool for downstream processing.

    Args:
        urls: List of document URLs to fetch
        page_fetcher: PageFetcher instance for content retrieval

    Returns:
        ResourcePool with loaded documents
    """
    resource_pool = ResourcePool()

    for url in urls:
        try:
            # Get chunks
            chunks = await page_fetcher.get_chunks(url)
            if not chunks:
                logger.warning(f"No chunks retrieved from {url}")
                continue

            # Get full document and title
            markdown, title = await page_fetcher.get_markdown(url)
            if not markdown:
                logger.warning(f"No markdown content retrieved from {url}")
                # Fallback to joined chunks
                markdown = "\n\n".join(chunks)
                title = f"Document from {url}"

            # Compute chunk spans in full document
            chunk_texts = (
                chunks if isinstance(chunks, list) else [c["text"] for c in chunks]
            )
            chunk_spans = compute_chunk_spans(markdown, chunk_texts)

            # Add to pool
            resource = resource_pool.add(url, title, markdown, chunks=chunk_spans)
            logger.info(
                f"Loaded document {resource.id.id}: {title} "
                f"({len(chunk_spans)} chunks, {len(markdown)} chars)"
            )

        except Exception as e:
            logger.error(f"Failed to load {url}: {e}")
            continue

    logger.info(
        f"Document loading complete: {len(resource_pool.resources)}/{len(urls)} successful"
    )
    return resource_pool


async def save_checkpoint(
    stage: str, state: ExtractionStateV3, checkpoint_dir: Path
) -> Path:
    """
    Save pipeline state to checkpoint file.

    Serializes state to JSON for resumability. Checkpoint includes stage name,
    timestamp, and all intermediate results (entities, assessments, candidates, pairs).

    Args:
        stage: Current pipeline stage name ('extraction', 'assessment', 'candidates', 'evaluation')
        state: Current extraction state
        checkpoint_dir: Directory to save checkpoint file

    Returns:
        Path to saved checkpoint file
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        checkpoint_dir / f"checkpoint_{stage}_{datetime.now().isoformat()}.json"
    )

    # Serialize state to dict
    checkpoint = {
        "stage": stage,
        "timestamp": datetime.now().isoformat(),
        "entities_found": {
            name: e.model_dump() for name, e in state.entities_found.items()
        },
        "assessments": {
            name: a.model_dump() for name, a in state.individual_assessments.items()
        },
        "candidates": {
            f"{k[0]}::{k[1]}": c.model_dump() for k, c in state.pair_candidates.items()
        },
        "pairs": [p.model_dump() for p in state.final_pairs],
        "metrics": {
            "cache_hits_extraction": state.metrics.cache_hits_extraction,
            "cache_misses_extraction": state.metrics.cache_misses_extraction,
            "cache_hits_assessment": state.metrics.cache_hits_assessment,
            "cache_misses_assessment": state.metrics.cache_misses_assessment,
            "candidates_generated": state.metrics.candidates_generated,
            "pairs_accepted": state.metrics.pairs_accepted,
            "pairs_rejected": state.metrics.pairs_rejected,
        },
    }

    # Write to file
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2))
    logger.info(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


async def resume_from_checkpoint(
    checkpoint_path: Path, resource_pool: ResourcePool
) -> tuple[ExtractionStateV3, str]:
    """
    Resume pipeline from saved checkpoint.

    Deserializes state from JSON checkpoint and determines start stage
    for resuming execution.

    Args:
        checkpoint_path: Path to checkpoint file
        resource_pool: ResourcePool with loaded documents (must match checkpoint)

    Returns:
        Tuple of (restored state, start stage name)
    """
    data = json.loads(checkpoint_path.read_text())

    # Create state with resource pool
    state = ExtractionStateV3(resource_pool=resource_pool)

    # Helper to reconstruct ResourceQuote from JSON
    def _reconstruct_quote(quote_data: dict):
        """Reconstruct a ResourceQuote by looking up resource and recreating."""
        resource_id = quote_data["resource_id"]
        phrase = quote_data["phrase"]
        # Find resource in pool
        resource = next(
            (r for r in resource_pool.resources if r.id.id == resource_id), None
        )
        if resource:
            # Reconstruct quote by searching for phrase in resource
            return resource.quote(phrase)
        else:
            logger.warning(
                f"Resource {resource_id} not found in pool for quote: {phrase}"
            )
            return None

    # Restore entities with reconstructed quotes
    state.entities_found = {}
    for name, entity_data in data["entities_found"].items():
        quotes = []
        for quote_data in entity_data.get("quotes", []):
            quote = _reconstruct_quote(quote_data)
            if quote:
                quotes.append(quote)

        state.entities_found[name] = EntityWithQuotes(
            name=entity_data["name"],
            kind=entity_data["kind"],
            quotes=quotes,
        )

    # Restore assessments with reconstructed quotes
    state.individual_assessments = {}
    for name, assessment_data in data["assessments"].items():
        # Reconstruct entity
        entity_data = assessment_data["entity"]
        entity_quotes = []
        for quote_data in entity_data.get("quotes", []):
            quote = _reconstruct_quote(quote_data)
            if quote:
                entity_quotes.append(quote)

        entity = EntityWithQuotes(
            name=entity_data["name"],
            kind=entity_data["kind"],
            quotes=entity_quotes,
        )

        # Reconstruct evidence quotes
        evidence_quotes = []
        for quote_data in assessment_data.get("evidence_quotes", []):
            quote = _reconstruct_quote(quote_data)
            if quote:
                evidence_quotes.append(quote)

        state.individual_assessments[name] = IndividualAssessment(
            entity=entity,
            relationship_potential=assessment_data["relationship_potential"],
            related_entities=assessment_data.get("related_entities", []),
            evidence_quotes=evidence_quotes,
            reasoning=assessment_data.get("reasoning", ""),
        )

    # Restore candidates
    state.pair_candidates = {
        tuple(k.split("::")): PairCandidate.model_validate(c)
        for k, c in data["candidates"].items()
    }

    # Restore pairs (already EntityPairOut from V2)
    from ..extraction_graph_v2.models import EntityPairOut

    state.final_pairs = [EntityPairOut.model_validate(p) for p in data["pairs"]]

    # Restore metrics
    metrics_data = data.get("metrics", {})
    state.metrics.cache_hits_extraction = metrics_data.get("cache_hits_extraction", 0)
    state.metrics.cache_misses_extraction = metrics_data.get(
        "cache_misses_extraction", 0
    )
    state.metrics.cache_hits_assessment = metrics_data.get("cache_hits_assessment", 0)
    state.metrics.cache_misses_assessment = metrics_data.get(
        "cache_misses_assessment", 0
    )
    state.metrics.candidates_generated = metrics_data.get("candidates_generated", 0)
    state.metrics.pairs_accepted = metrics_data.get("pairs_accepted", 0)
    state.metrics.pairs_rejected = metrics_data.get("pairs_rejected", 0)

    # Determine start stage
    start_stage = data["stage"]
    logger.info(
        f"Checkpoint restored from stage '{start_stage}': "
        f"{len(state.entities_found)} entities, "
        f"{len(state.individual_assessments)} assessments, "
        f"{len(state.pair_candidates)} candidates, "
        f"{len(state.final_pairs)} pairs"
    )

    return state, start_stage


async def run_extraction_v3(
    urls: List[str],
    config: IfetcherConfig,
    page_fetcher: PageFetcher,
    *,
    model: Optional[str] = None,
    verbose: bool = False,
    checkpoint_path: Optional[Path] = None,
    save_callback: Optional[Callable] = None,
) -> BatchExtractionResultV3:
    """
    Run the V3 extraction pipeline.

    Main entry point for running the complete 4-node extraction pipeline
    with checkpoint save/resume, semantic caching, and comprehensive metrics.

    Args:
        urls: Document URLs to process
        config: Configuration object with task and agent settings
        page_fetcher: PageFetcher instance for document retrieval
        model: Override model from config (e.g., 'openai:gpt-4o')
        verbose: Enable Rich progress displays
        checkpoint_path: Path to resume from (if exists) or save to (if new run)
        save_callback: Callback for incremental saves (not currently used)

    Returns:
        BatchExtractionResultV3 with pairs, metadata, cache stats, and stage metrics
    """
    logger.info(f"Starting extraction V3 for {len(urls)} documents")

    # Resume from checkpoint if provided and exists
    if checkpoint_path and checkpoint_path.exists():
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        # Must load documents first to restore state
        resource_pool = await _load_documents(urls, page_fetcher)
        state, start_stage = await resume_from_checkpoint(
            checkpoint_path, resource_pool
        )
    else:
        # Load documents from scratch
        resource_pool = await _load_documents(urls, page_fetcher)
        if not resource_pool.resources:
            logger.error("No documents loaded successfully, aborting")
            return BatchExtractionResultV3(
                total_pairs=0,
                entity_pairs=[],
                errors=[{"error": "No documents loaded successfully"}],
            )

        state = ExtractionStateV3(resource_pool=resource_pool)
        start_stage = "ExtractEntities"  # First node name

    # Initialize dependencies
    deps = ExtractionDepsV3.from_config(
        config,
        page_fetcher,
        model=model,
        checkpoint_callback=(
            lambda stage, state_arg: save_checkpoint(
                stage, state_arg, checkpoint_path.parent
            )
            if checkpoint_path
            else None
        ),
    )

    # Initialize cache if enabled
    if deps.semantic_cache_enabled:
        cache_dir = config.paths.get("cache_dir", ".cache/semantic")
        state.cache = SemanticCacheManager(Path(cache_dir))
        logger.info(f"Semantic cache enabled: {cache_dir}")

    # Start timing
    state.metrics.start_timing()

    # Run graph
    logger.info(f"Running extraction graph from stage: {start_stage}")
    ctx = GraphRunContext(state=state, deps=deps)

    try:
        # Run the graph from the start stage
        # pydantic-graph 1.0.0 API uses node names as strings
        await extraction_graph_v3.run(start_stage, ctx)

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        return BatchExtractionResultV3(
            total_pairs=len(state.final_pairs),
            entity_pairs=state.final_pairs,
            errors=[{"error": f"Pipeline execution failed: {e}"}],
        )

    # Build result
    entity_counts = {}
    for kind in deps.get_entity_kinds():
        entity_counts[kind] = len(
            [e for e in state.entities_found.values() if e.kind == kind]
        )

    result = BatchExtractionResultV3(
        total_pairs=len(state.final_pairs),
        entity_pairs=state.final_pairs,
        total_entities=entity_counts,
        metadata=ExtractionMetadata(
            timestamp=datetime.now(),
            model=str(deps.model) if hasattr(deps.model, "__str__") else deps.model,
            pipeline_version="v3",
            prompt_version="2025-10-v3",
            entity_kinds=deps.get_entity_kinds(),
        ),
        cache_stats={
            "extraction_hits": state.metrics.cache_hits_extraction,
            "extraction_misses": state.metrics.cache_misses_extraction,
            "assessment_hits": state.metrics.cache_hits_assessment,
            "assessment_misses": state.metrics.cache_misses_assessment,
            "extraction_hit_rate": state.metrics.get_cache_hit_rate()[
                "extraction_hit_rate"
            ],
            "assessment_hit_rate": state.metrics.get_cache_hit_rate()[
                "assessment_hit_rate"
            ],
            "overall_hit_rate": state.metrics.get_cache_hit_rate()["overall_hit_rate"],
        },
        stage_metrics={
            "extraction": {
                "calls": state.metrics.entities_extraction_calls,
                "successes": state.metrics.entities_extraction_successes,
                "duration": state.metrics.extraction_time,
            },
            "assessment": {
                "calls": state.metrics.assessment_calls,
                "successes": state.metrics.assessment_successes,
                "duration": state.metrics.assessment_time,
            },
            "candidate_generation": {
                "candidates_generated": state.metrics.candidates_generated,
                "from_cooccurrence": state.metrics.candidates_from_cooccurrence,
                "from_assessment": state.metrics.candidates_from_assessment,
                "duration": state.metrics.candidate_generation_time,
            },
            "evaluation": {
                "calls": state.metrics.pair_evaluation_calls,
                "successes": state.metrics.pair_evaluation_successes,
                "pairs_accepted": state.metrics.pairs_accepted,
                "pairs_rejected": state.metrics.pairs_rejected,
                "duration": state.metrics.pair_evaluation_time,
            },
        },
        quote_errors=deps.quote_error_log,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
    )

    logger.info(
        f"Extraction V3 complete: {result.total_pairs} pairs from "
        f"{len(state.entities_found)} entities "
        f"(cache hit rate: {result.cache_stats['overall_hit_rate']:.1f}%)"
    )

    return result


__all__ = [
    "run_extraction_v3",
    "save_checkpoint",
    "resume_from_checkpoint",
]
