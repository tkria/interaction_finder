"""Association extraction module.

Extracts binary entity-entity associations from scientific literature with
comprehensive quote-level provenance tracking and multi-stage evidence assessment.

Main entry point:
    run_extraction() - Run the full extraction pipeline

Key models:
    ExtractionResult - Final output with accepted pairs and metadata
    PairWithProvenance - Individual association with complete provenance
    ExtractionMetadata - Summary statistics

Example:
    >>> from interaction_finder.extraction import run_extraction
    >>> from interaction_finder.resources import ResourcePool
    >>>
    >>> pool = ResourcePool()
    >>> pool.add(url="...", title="...", document_text="...")
    >>>
    >>> result = await run_extraction(
    ...     topic="BRCA1 and breast cancer",
    ...     target_entity_types=["gene", "disease"],
    ...     resource_pool=pool
    ... )
    >>>
    >>> for pair in result.accepted_pairs:
    ...     print(f"{pair.entity1} {pair.relationship_type} {pair.entity2}")
"""

from interaction_finder.extraction.graph import graph
from interaction_finder.extraction.models import (
    ExtractionMetadata,
    ExtractionResult,
    PairWithProvenance,
)
from interaction_finder.extraction.run import run_extraction

__all__ = [
    "run_extraction",
    "ExtractionResult",
    "PairWithProvenance",
    "ExtractionMetadata",
    "graph",
]
