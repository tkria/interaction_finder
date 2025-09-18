"""
Graph assembly for extraction graph V2.

Simple 3-node graph following pydantic-graph patterns.
"""

from pydantic_graph import Graph

from .nodes import ExtractEntities, AssessIndividually, AggregateIntoPairs


# Assemble the simplified 3-node graph
extraction_graph_v2 = Graph(
    nodes=[
        ExtractEntities,
        AssessIndividually,
        AggregateIntoPairs,
    ]
)


def get_graph_description() -> str:
    """
    Get human-readable description of the graph flow.

    Returns:
        Description of the 3-node workflow
    """
    return """
Simplified Extraction Graph V2 Workflow:

1. ExtractEntities: Hard-coded entity extraction with ResourceQuote creation
   - Finds BRCA1 and breast cancer in documents
   - Creates ResourceQuotes for each entity occurrence
   - Stores EntityWithQuotes in state

2. AssessIndividually: Individual entity assessment (key insight preserved)
   - Processes each entity with its specific contexts
   - Mock assessment logic for testing
   - Creates IndividualAssessment with evidence quotes

3. AggregateIntoPairs: Combine assessments into pairs
   - Matches entities based on assessment potential
   - Creates EntityPairOut with complete provenance
   - Validates all pairs have evidence quotes

The graph maintains the original insight (individual processing) while
dramatically simplifying the implementation and ensuring complete
ResourceQuote provenance throughout.
"""
