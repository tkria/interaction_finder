"""Graph assembly for widesearch pipeline.

Assembles all nodes into a complete Pydantic Graph for execution.
The graph defines the control flow and valid transitions between nodes.
"""

from pydantic_graph import Graph

from interaction_finder.widesearch.nodes import (
    GenerateQueriesNode,
    PlanGoalsNode,
    ReflectNode,
    RerankNode,
    SearchNode,
    SelectResultsNode,
)

# Assemble the graph with all node classes
graph = Graph(
    nodes=[
        PlanGoalsNode,
        GenerateQueriesNode,
        SearchNode,
        RerankNode,
        SelectResultsNode,
        ReflectNode,
    ]
)


def save_diagram(path: str = "widesearch_diagram.md") -> None:
    """Save Mermaid diagram of the graph to a file.

    Parameters:
        path: str — output file path (default: widesearch_diagram.md)
    """
    with open(path, "w") as f:
        f.write("# Widesearch Pipeline Graph\n\n")
        f.write(graph.mermaid_code())
        f.write("\n")
