"""
Simplified runner for extraction graph V2 pipeline.

Simple entry point that runs the 3-node pipeline without grouping complexity.
"""

import logging
from typing import List

from pydantic_graph import GraphRunContext

from ..resources import ResourcePool
from .state import ExtractionState
from .deps import ExtractionDeps
from .models import EntityPairOut

logger = logging.getLogger(__name__)


async def run_extraction_v2_simple(
    resource_pool: ResourcePool,
    deps: ExtractionDeps,
) -> List[EntityPairOut]:
    """
    Run the complete extraction pipeline with simplified logic.

    Args:
        resource_pool: ResourcePool with loaded documents
        deps: External dependencies (model, config, etc.)

    Returns:
        List of EntityPairOut objects with complete provenance
    """
    logger.info("Starting simplified extraction graph V2 pipeline")

    # Create initial state
    state = ExtractionState(resource_pool=resource_pool)

    # Log initial state
    summary = state.get_summary()
    logger.info(f"Initial state: {summary}")

    try:
        state.metrics.start_timing()

        # Phase 1: Extract entities from all documents
        logger.info("Phase 1: Extracting entities from all documents")
        from .nodes import ExtractEntities

        extract_node = ExtractEntities()
        result = await extract_node.run(GraphRunContext(state=state, deps=deps))

        if result is not None:
            # Node ended early, return empty results
            logger.info(f"Extraction ended: {result}")
            return []

        logger.info(f"Extraction complete: {len(state.entities_found)} entities found")

        # Phase 2: Individual assessment
        logger.info("Phase 2: Assessing entities individually")
        from .nodes import AssessIndividually

        assess_node = AssessIndividually()
        await assess_node.run(GraphRunContext(state=state, deps=deps))

        logger.info(
            f"Assessment complete: {len(state.individual_assessments)} assessments"
        )

        # Phase 3: Aggregate into pairs
        logger.info("Phase 3: Aggregating into entity pairs")
        from .nodes import AggregateIntoPairs

        aggregate_node = AggregateIntoPairs()
        await aggregate_node.run(GraphRunContext(state=state, deps=deps))

        logger.info(f"Pipeline complete: {len(state.final_pairs)} pairs generated")

        # Validate provenance chain
        if state.validate_provenance_chain():
            logger.info("✓ Complete provenance chain validated")
        else:
            logger.warning("⚠ Provenance chain validation failed")

        return state.final_pairs

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return []
