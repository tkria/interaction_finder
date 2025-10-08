"""
Extraction graph V3 pipeline definition.

Assembles the four-node linear pipeline:
1. ExtractEntities - Extract entities from documents with semantic caching
2. AssessIndividually - Assess each entity's relationship potential
3. GeneratePairCandidates - Generate candidates using hybrid co-occurrence + assessment
4. EvaluatePairs - Evaluate candidates with evidence-based assessment

Linear flow: all documents processed through all stages without routing.
"""

from pydantic_graph import Graph

from .nodes import (
    ExtractEntities,
    AssessIndividually,
    GeneratePairCandidates,
    EvaluatePairs,
)

# Linear 4-node extraction pipeline
extraction_graph_v3 = Graph(
    nodes=[
        ExtractEntities,
        AssessIndividually,
        GeneratePairCandidates,
        EvaluatePairs,
    ]
)

__all__ = ["extraction_graph_v3"]
