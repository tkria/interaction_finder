"""Tests for extraction pipeline graph assembly."""

from interaction_finder.extraction.graph import graph


class TestGraphAssembly:
    """Test graph is properly assembled."""

    def test_graph_exists(self):
        """Test graph is created."""
        assert graph is not None

    def test_graph_has_nodes(self):
        """Test graph has nodes."""
        assert len(graph.node_defs) > 0

    def test_graph_has_expected_nodes(self):
        """Test graph contains all expected node types."""
        node_names = list(graph.node_defs.keys())

        expected_nodes = [
            "ProcessDocumentsNode",
            "ConsolidateEntitiesNode",
            "JudgeCrossDocumentNode",
            "FinalizeNode",
        ]

        for expected in expected_nodes:
            assert expected in node_names, f"Missing node: {expected}"

    def test_graph_generates_mermaid(self):
        """Test mermaid diagram generation."""
        mermaid = graph.mermaid_code()

        assert isinstance(mermaid, str)
        assert len(mermaid) > 0
        assert "ProcessDocumentsNode" in mermaid
        assert "FinalizeNode" in mermaid
