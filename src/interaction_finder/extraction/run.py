"""Entry point for association extraction pipeline.

Provides the main run_extraction function that orchestrates the entire
extraction process from resources to final pairs with provenance.
"""

import asyncio
import logging

from interaction_finder.checkpoint import ExtractionStageData, PipelineCheckpoint
from interaction_finder.version import get_version_string
from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.graph import graph
from interaction_finder.extraction.nodes import (
    ProcessDocumentsNode,
    ConsolidateEntitiesNode,
    ConsolidateRelationshipsNode,
    SweepCoMentionsNode,
    ConsolidateNewRelationshipsNode,
    JudgeCrossDocumentNode,
)
from interaction_finder.extraction.state import State
from interaction_finder.extraction.utils import build_permitted_pairs
from interaction_finder.logging import logfire, get_logger
from interaction_finder.settings import IfetcherConfig

# Mapping from stage name to next node
_RESUME_NODE_MAP = {
    "process_documents": ConsolidateEntitiesNode,
    "consolidate_entities": ConsolidateRelationshipsNode,
    "consolidate_relationships": SweepCoMentionsNode,
    "sweep_co_mentions": ConsolidateNewRelationshipsNode,
    "consolidate_new_relationships": JudgeCrossDocumentNode,
}


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
    # Extract data from input checkpoint
    topic = input_checkpoint.topic
    resource_pool = input_checkpoint.resources

    with logfire.span(f"Extraction: {topic}"):
        # Load config or use defaults
        if config is None:
            config = IfetcherConfig()

        # Initialize logger
        if logger is None:
            logger = get_logger(__name__)

        # Create agent concurrency semaphore
        agent_semaphore = asyncio.Semaphore(
            config.tools.extraction.agent_concurrency_limit
        )

        # Build permitted pairs map from target entity types
        permitted_pairs = build_permitted_pairs(target_entity_types)

        # Check for resumable extraction
        if (
            input_checkpoint.extraction
            and input_checkpoint.extraction.metadata
            and input_checkpoint.extraction.metadata.is_resumable
        ):
            meta = input_checkpoint.extraction.metadata
            logger.info(f"Resuming extraction from: {meta.resume_from}")

            # Deserialize state from checkpoint
            state = State.from_dict(
                meta.resume_state,
                topic=topic,
                target_entity_types=target_entity_types,
                permitted_pairs=permitted_pairs,
                resource_pool=resource_pool,
            )

            # Determine starting node (next after completed stage)
            start_node = _RESUME_NODE_MAP[meta.resume_from]()

        else:
            # Fresh start
            state = State(
                topic=topic,
                target_entity_types=target_entity_types,
                permitted_pairs=permitted_pairs,
            )
            start_node = ProcessDocumentsNode()

        # Create dependencies (after state is initialized)
        deps = Deps(
            resource_pool=resource_pool,
            config=config,
            logger=logger,
            progress=progress,
            agent_semaphore=agent_semaphore,
            checkpoint_path=checkpoint_path,
            input_checkpoint=input_checkpoint,
        )

        # Run graph
        result = await graph.run(start_node, state=state, deps=deps)

        # Build unified checkpoint preserving all prior data
        return PipelineCheckpoint(
            topic=topic,
            resources=resource_pool,
            created_by=get_version_string(),
            keywords=input_checkpoint.keywords,  # PRESERVED
            search=input_checkpoint.search,  # PRESERVED
            extraction=ExtractionStageData(
                target_entity_types=target_entity_types,
                permitted_pairs=permitted_pairs,
                judgments=result.output.judgments,
                metadata=result.output.metadata,
                consolidation_rules=result.output.consolidation_rules,
                relationship_oppositions={
                    k: list(v)
                    for k, v in result.output.relationship_oppositions.items()
                },
            ),
        )
