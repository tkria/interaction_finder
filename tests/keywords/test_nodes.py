"""Tests for graph nodes."""

import pytest

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


class TestNodeStructure:
    """Test node structure and configuration."""

    def test_all_nodes_exist(self):
        """Test all expected nodes are defined."""
        nodes = [
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
        # All should be classes
        for node in nodes:
            assert isinstance(node, type)

    def test_expand_query_node_has_run_method(self):
        """Test ExpandQueryNode has run method."""
        node = ExpandQueryNode()
        assert hasattr(node, "run")
        assert callable(node.run)

    def test_search_node_has_run_method(self):
        """Test SearchNode has run method."""
        node = SearchNode()
        assert hasattr(node, "run")
        assert callable(node.run)

    def test_rerank_node_has_run_method(self):
        """Test RerankNode has run method."""
        node = RerankNode()
        assert hasattr(node, "run")
        assert callable(node.run)

    def test_select_results_node_has_run_method(self):
        """Test SelectResultsNode has run method."""
        node = SelectResultsNode()
        assert hasattr(node, "run")
        assert callable(node.run)

    def test_fetch_documents_node_has_run_method(self):
        """Test FetchDocumentsNode has run method."""
        node = FetchDocumentsNode()
        assert hasattr(node, "run")
        assert callable(node.run)

    def test_extract_keywords_node_has_run_method(self):
        """Test ExtractKeywordsNode has run method."""
        node = ExtractKeywordsNode()
        assert hasattr(node, "run")
        assert callable(node.run)

    def test_evaluate_keywords_node_has_run_method(self):
        """Test EvaluateKeywordsNode has run method."""
        node = EvaluateKeywordsNode()
        assert hasattr(node, "run")
        assert callable(node.run)

    def test_reflect_node_has_run_method(self):
        """Test ReflectNode has run method."""
        node = ReflectNode()
        assert hasattr(node, "run")
        assert callable(node.run)

    def test_finalize_node_has_run_method(self):
        """Test FinalizeNode has run method."""
        node = FinalizeNode()
        assert hasattr(node, "run")
        assert callable(node.run)


class TestNodeReturnTypes:
    """Test that nodes have correct return type annotations."""

    def test_expand_query_returns_search_node(self):
        """Test ExpandQueryNode run signature."""
        import inspect

        sig = inspect.signature(ExpandQueryNode.run)
        # Should return SearchNode (as string annotation)
        assert sig.return_annotation is not None

    def test_finalize_returns_end(self):
        """Test FinalizeNode returns End type."""
        import inspect

        sig = inspect.signature(FinalizeNode.run)
        # Should return End[BridgingTermsOut]
        assert sig.return_annotation is not None


class TestNodeDataclasses:
    """Test nodes are properly configured as dataclasses."""

    def test_nodes_are_dataclasses(self):
        """Test all nodes are dataclasses."""
        from dataclasses import is_dataclass

        nodes = [
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
        for node in nodes:
            assert is_dataclass(node), f"{node.__name__} should be a dataclass"
