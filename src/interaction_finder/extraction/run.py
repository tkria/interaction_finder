"""Entry point for association extraction pipeline.

Provides the main run_extraction function that orchestrates the entire
extraction process from resources to final pairs with provenance.
"""

import logging

from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.graph import graph
from interaction_finder.extraction.models import ExtractionResult
from interaction_finder.extraction.nodes import ExtractFromDocumentsNode
from interaction_finder.extraction.state import State
from interaction_finder.logging import logfire
from interaction_finder.resources import ResourcePool


async def run_extraction(
    topic: str,
    target_entity_types: list[str],
    resource_pool: ResourcePool,
    config: dict | None = None,
    logger: logging.Logger | None = None,
) -> ExtractionResult:
    """Run the association extraction pipeline.

    Parameters:
        topic: Research topic for context (e.g., "genes associated with breast cancer")
        target_entity_types: Types of entities to extract (e.g., ["gene", "disease"])
        resource_pool: ResourcePool containing documents to process
        config: Configuration dict with model names and settings (optional)
        logger: Logger for warnings and debugging (optional)

    Returns:
        ExtractionResult with accepted pairs and metadata

    Example:
        >>> pool = ResourcePool()
        >>> pool.add(url="...", title="...", document_text="...")
        >>> result = await run_extraction(
        ...     topic="BRCA1 and breast cancer",
        ...     target_entity_types=["gene", "disease"],
        ...     resource_pool=pool
        ... )
        >>> print(f"Found {len(result.accepted_pairs)} associations")
    """
    with logfire.span("run_extraction", topic=topic):
        # Initialize deps
        if config is None:
            config = {}

        if logger is None:
            logger = logging.getLogger(__name__)

        deps = Deps(resource_pool=resource_pool, config=config, logger=logger)

        # Create initial state
        state = State(topic=topic, target_entity_types=target_entity_types)

        # Run graph
        result = await graph.run(ExtractFromDocumentsNode(), state=state, deps=deps)

        return result.output
