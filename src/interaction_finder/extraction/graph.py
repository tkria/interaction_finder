"""Graph assembly for association extraction pipeline.

The graph defines the flow:
1. Process all documents concurrently (entities → pairs → assessments)
2. Merge entities globally and update pair references
3. Merge relationship labels globally
4. Make cross-document judgments
5. Finalize results
"""

from pydantic_graph import Graph

from interaction_finder.extraction.nodes import (
    FinalizeNode,
    JudgeCrossDocumentNode,
    ConsolidateEntitiesNode,
    ConsolidateRelationshipsNode,
    ProcessDocumentsNode,
)

# Assemble the extraction graph with concurrent per-document processing
graph = Graph(
    nodes=[
        ProcessDocumentsNode,
        ConsolidateEntitiesNode,
        ConsolidateRelationshipsNode,
        JudgeCrossDocumentNode,
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
