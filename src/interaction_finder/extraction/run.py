"""Entry point for association extraction pipeline.

Provides the main run_extraction function that orchestrates the entire
extraction process from resources to final pairs with provenance.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from interaction_finder.checkpoint import (
    ExtractionStageData,
    PipelineCheckpoint,
    check_config_consistency,
)
from interaction_finder.extraction.deps import Deps
from interaction_finder.usage import PipelineUsage
from interaction_finder.extraction.shared import empty_result
from interaction_finder.extraction.stages import (
    consolidate_by_neighbours,
    consolidate_entities,
    consolidate_new_relationships,
    consolidate_relationships,
    finalize,
    judge_cross_document,
    process_documents,
    sweep_co_mentions,
)
from interaction_finder.extraction.state import State
from interaction_finder.extraction.utils import build_permitted_pairs
from interaction_finder.logging import get_logger, logfire
from interaction_finder.settings import IfetcherConfig
from interaction_finder.version import get_version_string

# Pipeline stages in execution order. All return bool; False aborts pipeline.
StageFunc = Callable[[State, Deps], Awaitable[bool]]
STAGES: list[StageFunc] = [
    process_documents,
    consolidate_entities,
    consolidate_relationships,
    sweep_co_mentions,
    consolidate_new_relationships,
    consolidate_by_neighbours,
    judge_cross_document,
]
STAGE_NAMES = [fn.__name__ for fn in STAGES]


def get_resume_stage_index(stage_name: str) -> int:
    """Get the index of the stage to resume from (the one after the completed stage)."""
    try:
        return STAGE_NAMES.index(stage_name) + 1
    except ValueError:
        return 0


async def run_extraction(
    input_checkpoint: PipelineCheckpoint,
    target_entity_types: list[str],
    config: IfetcherConfig | None = None,
    logger: logging.Logger | None = None,
    progress=None,
    checkpoint_path: str | None = None,
) -> PipelineCheckpoint:
    """Run the association extraction pipeline preserving all prior data.

    Automatically resumes from partial checkpoints if extraction was interrupted.

    Parameters:
        input_checkpoint: PipelineCheckpoint with search stage data
        target_entity_types: Types of entities to extract (e.g., ["gene", "disease"])
        config: Configuration object (creates default if None)
        logger: Logger for warnings and debugging (optional)
        progress: Progress counter for live display (optional)
        checkpoint_path: Path for saving partial checkpoints (optional)

    Returns:
        PipelineCheckpoint with keywords + search + extraction stage data

    Example:
        >>> # Assume search_checkpoint from widesearch stage
        >>> checkpoint = await run_extraction(
        ...     input_checkpoint=search_checkpoint,
        ...     target_entity_types=["gene", "disease"],
        ... )
        >>> print(f"Keywords: {len(checkpoint.keywords.terms)}")
        >>> print(f"Searches: {len(checkpoint.search.queries)}")
        >>> print(f"Pairs: {len(checkpoint.extraction.judgments)}")
    """
    topic = input_checkpoint.topic
    resource_pool = input_checkpoint.resources
    with logfire.span("Extraction: {topic}", topic=topic):
        if config is None:
            config = IfetcherConfig()
        if logger is None:
            logger = get_logger(__name__)
        # Check config consistency and get updated config for checkpoint
        updated_config, config_warnings = check_config_consistency(
            input_checkpoint.config, config, "extraction"
        )
        for warning in config_warnings:
            logger.warning(warning)
        agent_semaphore = asyncio.Semaphore(
            config.stage.extraction.agent_concurrency_limit
        )
        permitted_pairs = build_permitted_pairs(target_entity_types)
        # Determine starting stage index (0 for fresh start, or index after last completed)
        start_stage_idx = 0
        if (
            input_checkpoint.extraction
            and input_checkpoint.extraction.metadata
            and input_checkpoint.extraction.metadata.is_resumable
        ):
            meta = input_checkpoint.extraction.metadata
            logger.info(f"Resuming extraction from: {meta.resume_from}")
            state = State.from_dict(
                meta.resume_state,
                topic=topic,
                target_entity_types=target_entity_types,
                permitted_pairs=permitted_pairs,
                resource_pool=resource_pool,
            )
            start_stage_idx = get_resume_stage_index(meta.resume_from)
        else:
            state = State(
                topic=topic,
                target_entity_types=target_entity_types,
                permitted_pairs=permitted_pairs,
            )
        # Initialize usage from input checkpoint (deep copy to avoid mutation)
        input_usage = input_checkpoint.usage
        stage_usage = (
            {k: v.copy_deep() for k, v in input_usage.extraction.items()}
            if input_usage
            else {}
        )
        deps = Deps(
            resource_pool=resource_pool,
            config=config,
            logger=logger,
            progress=progress,
            agent_semaphore=agent_semaphore,
            checkpoint_path=checkpoint_path,
            input_checkpoint=input_checkpoint,
            usage=stage_usage,
        )
        # Run pipeline stages
        result = await _run_pipeline(state, deps, start_stage_idx)
        # Build usage preserving other stages
        usage = PipelineUsage(
            keywords=input_usage.keywords if input_usage else {},
            search=input_usage.search if input_usage else {},
            extraction=deps.usage,
        )
        # Build unified checkpoint preserving all prior data
        return PipelineCheckpoint(
            topic=topic,
            resources=resource_pool,
            created_by=get_version_string(),
            usage=usage,
            config=updated_config,
            keywords=input_checkpoint.keywords,
            search=input_checkpoint.search,
            extraction=ExtractionStageData(
                target_entity_types=target_entity_types,
                permitted_pairs=permitted_pairs,
                judgments=result.judgments,
                metadata=result.metadata,
                consolidated=result.consolidated,
                paper_quality=result.paper_quality,
            ),
        )


async def _run_pipeline(state: State, deps: Deps, start_stage_idx: int):
    """Execute extraction pipeline stages sequentially."""
    for stage in STAGES[start_stage_idx:]:
        if not await stage(state, deps):
            return empty_result(state, deps)
    return finalize(state, deps)
