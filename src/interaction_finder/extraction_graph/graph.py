"""
Graph assembly for the entity extraction pipeline.

This module assembles the graph nodes into a complete workflow,
following the pydantic-graph pattern.
"""

from pydantic_graph import Graph

from .nodes import (
    InitialRouter,
    EntityExtraction,
    ParallelEntityAssessment,
    ParallelEntityProcessor,
    EntityAggregation,
    EntityPairGeneration,
    QualityValidation,
)


# Assemble the graph with all nodes (using direct extraction)
extraction_graph = Graph(
    nodes=[
        InitialRouter,
        EntityExtraction,
        ParallelEntityProcessor,
        EntityAggregation,
        EntityPairGeneration,
        QualityValidation,
    ]
)

# Note: Legacy graph removed due to node incompatibility
# EntityExtraction now returns ParallelEntityProcessor by default
# To use the old behavior, use the legacy nodes separately


def get_graph_visualization() -> str:
    """
    Get Mermaid diagram code for the extraction graph.

    Returns:
        Mermaid diagram code as string
    """
    try:
        return extraction_graph.mermaid_code()
    except Exception:
        # Fallback if mermaid generation fails
        return """
graph TD
    A[InitialRouter] --> B[EntityExtraction]
    A --> END1[Non-scientific End]
    B --> C[ParallelEntityProcessor]
    B --> END2[No Entities End]
    C --> D[EntityAggregation]
    C --> END3[No Individual Processing End]
    D --> E[EntityPairGeneration]
    D --> END4[No Aggregation End]
    E --> F[QualityValidation]
    E --> END5[No Pairs End]
    F --> G[Approved Pairs End]
    F --> E[Revision Loop]
"""


def describe_graph_flow() -> str:
    """
    Provide a text description of the graph flow.

    Returns:
        Human-readable description of the workflow
    """
    return """
Individual Entity Processing Graph Workflow:

1. InitialRouter: Classifies document group content
   - Scientific → EntityExtraction
   - Non-scientific → End

2. EntityExtraction: Extracts entities from document group using direct functional approach
   - Entities found → ParallelEntityProcessor
   - No entities → End

3. ParallelEntityProcessor: Processes entities individually in batches
   - For each entity: extracts context and assesses relationship potential
   - Always → EntityAggregation

4. EntityAggregation: Combines individual assessments into pairs
   - Pairs created → EntityPairGeneration
   - No pairs → End

5. EntityPairGeneration: Generates final pairs from aggregation results
   - Pairs created → QualityValidation
   - No pairs → End

6. QualityValidation: Validates pair quality
   - Approve → End (success)
   - Revise → EntityPairGeneration (up to max_revisions)
   - Reject → End (filtered out)

The graph implements individual entity processing where each entity gets
its own context extraction and assessment before aggregation into pairs.
"""
