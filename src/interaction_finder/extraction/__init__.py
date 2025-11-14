"""Association extraction module.

Extracts binary entity-entity associations from scientific literature with
comprehensive quote-level provenance tracking and multi-stage evidence assessment.

Main entry point:
    run_extraction() - Run the full extraction pipeline

Key models:
    ExtractionResult - Final output with all pair judgments and metadata
    PairJudgment - Individual association judgment with accept/reject decision
    PairAssessment - Per-document evidence assessment
    ExtractionMetadata - Summary statistics

Entity kind pair filtering:
    Pairs are filtered based on the target_entity_types list. A kind must appear
    at least twice in the list to allow same-kind pairs (e.g., gene-gene).

    Examples:
        ["gene", "disease"] → only gene-disease pairs allowed
        ["gene", "gene", "disease"] → gene-gene and gene-disease pairs allowed
        ["gene", "gene", "disease", "disease"] → all combinations allowed

Example:
    >>> from interaction_finder.extraction import run_extraction
    >>> from interaction_finder.resources import ResourcePool
    >>>
    >>> pool = ResourcePool()
    >>> pool.add(url="...", title="...", document_text="...")
    >>>
    >>> # Only extract gene-disease associations (no gene-gene or disease-disease)
    >>> result = await run_extraction(
    ...     topic="BRCA1 and breast cancer",
    ...     target_entity_types=["gene", "disease"],
    ...     resource_pool=pool
    ... )
    >>>
    >>> accepted = [j for j in result.judgments if j.accepted]
    >>> for judgment in accepted:
    ...     print(f"{judgment.entity1.name} {judgment.relationship} {judgment.entity2.name}")
"""

from interaction_finder.extraction.graph import graph
from interaction_finder.extraction.models import (
    ExtractionMetadata,
    PairAssessment,
    PairJudgment,
)
from interaction_finder.extraction.run import run_extraction

__all__ = [
    "run_extraction",
    "PairJudgment",
    "PairAssessment",
    "ExtractionMetadata",
    "graph",
]
