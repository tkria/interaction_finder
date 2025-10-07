"""
Extraction Graph - A pydantic-graph based entity pair extraction pipeline.

This submodule processes document groups from PageFetcher to extract typed entity pairs
(genes, diseases, celltypes, biomarkers, etc.) using a structured graph workflow.

Key features:
- Configurable entity types via task.kinds
- Document attribution for all extracted entities
- Parallel processing with batch operations
- Quality validation and self-correction
- Full provenance tracking

Example usage:
    from interaction_finder.extraction_graph import extract_entity_pairs

    pairs = await extract_entity_pairs(
        document_groups=groups,
        config=config
    )
"""

# Compatibility fix for dependencies that may still expect RunResult
# (pydantic-ai renamed RunResult to FinalResult in recent versions)
try:
    from pydantic_ai import RunResult  # Try the old name first
except ImportError:
    from pydantic_ai.result import FinalResult
    import pydantic_ai

    pydantic_ai.RunResult = FinalResult  # Make the old name available

from .run import (
    extract_entity_pairs,
    prepare_document_groups,
    extract_from_urls,
    save_results,
)
from .models import EntityInfo, EntityPairOut, BatchExtractionResult, ExtractionSummary
from .state import DocumentGroup, EntityExtractionState
from .graph import extraction_graph, get_graph_visualization, describe_graph_flow

__all__ = [
    "extract_entity_pairs",
    "prepare_document_groups",
    "extract_from_urls",
    "save_results",
    "EntityInfo",
    "EntityPairOut",
    "BatchExtractionResult",
    "ExtractionSummary",
    "DocumentGroup",
    "EntityExtractionState",
    "extraction_graph",
    "get_graph_visualization",
    "describe_graph_flow",
]
