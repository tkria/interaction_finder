"""Tests for graph assembly."""

from interaction_finder.keywords.graph import graph, save_diagram


class TestGraphAssembly:
    """Test graph assembly and configuration."""

    def test_graph_is_defined(self):
        """Test graph is defined."""
        assert graph is not None

    def test_graph_has_nodes(self):
        """Test graph has nodes registered."""
        # Graph should have nodes
        assert hasattr(graph, "node_defs")
        assert len(graph.node_defs) > 0

    def test_save_diagram_function_exists(self):
        """Test save_diagram function exists."""
        assert callable(save_diagram)

    def test_graph_can_generate_mermaid(self):
        """Test graph can generate mermaid code."""
        mermaid = graph.mermaid_code()
        assert isinstance(mermaid, str)
        assert len(mermaid) > 0
        # Should contain node names
        assert "ExpandQueryNode" in mermaid or "Expand" in mermaid
