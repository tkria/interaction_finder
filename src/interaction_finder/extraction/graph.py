"""Graph assembly for association extraction pipeline.

The graph defines the flow from extraction through assessment to final judgment.
"""

from pydantic_graph import Graph

from interaction_finder.extraction.nodes import (
    AssessEntitiesNode,
    AssessPairsNode,
    ExtractFromDocumentsNode,
    FinalizeNode,
    JudgePairsNode,
)

# Assemble the extraction graph
graph = Graph(
    nodes=[
        ExtractFromDocumentsNode,
        AssessEntitiesNode,
        AssessPairsNode,
        JudgePairsNode,
        FinalizeNode,
    ]
)


def save_diagram(output_path: str = "extraction_graph.md"):
    """Generate and save a mermaid diagram of the extraction pipeline.

    Parameters:
        output_path: Where to save the markdown file with diagram
    """
    with open(output_path, "w") as f:
        f.write("# Association Extraction Pipeline\n\n")
        f.write("```mermaid\n")
        f.write(graph.mermaid_code())
        f.write("\n```\n")
