"""Graph assembly for keyword research pipeline.

Assembles all nodes into a complete graph and provides utilities for
visualization.
"""

from pydantic_graph import Graph

from interaction_finder.keywords.nodes import (
    ExpandQueryNode,
    EvaluateKeywordsNode,
    ExtractKeywordsNode,
    FetchDocumentsNode,
    FinalizeNode,
    ReflectNode,
    RerankNode,
    SearchNode,
    SelectResultsNode,
)

# Assemble the graph with all node classes
graph = Graph(
    nodes=[
        ExpandQueryNode,
        SearchNode,
        RerankNode,
        SelectResultsNode,
        FetchDocumentsNode,
        ExtractKeywordsNode,
        EvaluateKeywordsNode,
        ReflectNode,
        FinalizeNode,
    ]
)


def save_diagram(output_path: str = "keywords_graph.md") -> None:
    """Generate and save mermaid diagram of the graph.

    Parameters:
        output_path: str — path to save mermaid diagram (default: keywords_graph.md)
    """
    with open(output_path, "w") as f:
        f.write("# Keywords Research Pipeline Graph\n\n")
        f.write(
            "This graph shows the control flow of the keyword research pipeline.\n\n"
        )
        f.write("## Pipeline Overview\n\n")
        f.write("1. **ExpandQuery**: Generate review-focused search queries\n")
        f.write("2. **Search**: Execute searches via backend\n")
        f.write("3. **Rerank**: Semantic reranking of results\n")
        f.write("4. **SelectResults**: LLM selects promising results\n")
        f.write("5. **FetchDocuments**: Retrieve document content\n")
        f.write("6. **ExtractKeywords**: Run all keyword extractors\n")
        f.write("7. **EvaluateKeywords**: LLM evaluates and summarizes\n")
        f.write("8. **Reflect**: Decide continue or stop\n")
        f.write("9. **Finalize**: Deduplicate and return results\n\n")
        f.write("## Control Flow Diagram\n\n")
        f.write("```mermaid\n")
        f.write(graph.mermaid_code())
        f.write("\n```\n")
        f.write("\n## Notes\n\n")
        f.write(
            "- **Reflection Loop**: ReflectNode can return to ExpandQueryNode for another round\n"
        )
        f.write(
            "- **Early Termination**: Multiple nodes can skip to FinalizeNode if conditions aren't met\n"
        )
        f.write("- **Iteration Limit**: Max rounds enforced in ReflectNode\n")
