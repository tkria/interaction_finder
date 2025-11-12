"""Entry point for association extraction pipeline.

Provides the main run_extraction function that orchestrates the entire
extraction process from resources to final pairs with provenance.
"""

import logging

from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.graph import graph
from interaction_finder.extraction.models import ExtractionResult
from interaction_finder.extraction.nodes import ExtractEntitiesNode
from interaction_finder.extraction.state import State
from interaction_finder.extraction.utils import build_permitted_pairs
from interaction_finder.logging import logfire
from interaction_finder.resources import ResourcePool
from interaction_finder.settings import IfetcherConfig


async def run_extraction(
    topic: str,
    target_entity_types: list[str],
    resource_pool: ResourcePool,
    config: IfetcherConfig | None = None,
    logger: logging.Logger | None = None,
) -> ExtractionResult:
    """Run the association extraction pipeline.

    Parameters:
        topic: Research topic for context (e.g., "genes associated with breast cancer")
        target_entity_types: Types of entities to extract (e.g., ["gene", "disease"])
        resource_pool: ResourcePool containing documents to process
        config: Configuration object (creates default if None)
        logger: Logger for warnings and debugging (optional)

    Returns:
        ExtractionResult with accepted pairs and metadata

    Example:
        >>> from interaction_finder import IfetcherConfig
        >>> config = IfetcherConfig.from_path("config.toml")
        >>> pool = ResourcePool()
        >>> pool.add(url="...", title="...", document_text="...")
        >>> result = await run_extraction(
        ...     topic="BRCA1 and breast cancer",
        ...     target_entity_types=["gene", "disease"],
        ...     resource_pool=pool,
        ...     config=config,
        ... )
        >>> accepted = [j for j in result.judgments if j.accepted]
        >>> print(f"Found {len(accepted)} associations")
    """
    with logfire.span("run_extraction", topic=topic):
        # Load config or use defaults
        if config is None:
            config = IfetcherConfig()

        # Initialize logger
        if logger is None:
            logger = logging.getLogger(__name__)

        # Create dependencies
        deps = Deps(
            resource_pool=resource_pool,
            config=config,
            logger=logger,
        )

        # Build permitted pairs map from target entity types
        permitted_pairs = build_permitted_pairs(target_entity_types)

        # Create initial state
        state = State(
            topic=topic,
            target_entity_types=target_entity_types,
            permitted_pairs=permitted_pairs,
        )

        # Run graph
        result = await graph.run(ExtractEntitiesNode(), state=state, deps=deps)

        return result.output
