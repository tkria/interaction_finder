"""Tests for graph nodes."""

import pytest
from unittest.mock import Mock
from pydantic_graph import GraphRunContext

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
from interaction_finder.keywords.deps import Deps
from interaction_finder.keywords.state import State
from interaction_finder.keywords.models import DocumentSummaryOut
from interaction_finder.resources import ResourcePool


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


class TestFinalizeNode:
    """Test FinalizeNode deduplication and finalization logic."""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_case_insensitive_deduplication(self):
        """Test that FinalizeNode deduplicates terms case-insensitively."""
        # Create mock dependencies
        mock_reranker = Mock()
        mock_reranker.rerank_terms = Mock(
            return_value=[
                ("machine learning", 0.9),
                ("deep learning", 0.8),
                ("neural networks", 0.7),
            ]
        )
        deps = Deps(
            http_client=Mock(),
            fetcher=Mock(),
            search_backend=Mock(),
            reranker=mock_reranker,
            extractors={},
            resource_pool=ResourcePool(),
            config={},
        )
        # Create state with duplicate terms in different cases
        state = State(topic="artificial intelligence", max_rounds=1)
        state.current_round = 1
        state.document_summaries = [
            DocumentSummaryOut(
                summary="First document discusses machine learning and deep learning approaches to AI.",
                related_areas=[],
                bridging_terms=["Machine Learning", "Deep Learning"],
                coverage_contribution="Provides foundational ML concepts and terminology",
            ),
            DocumentSummaryOut(
                summary="Second document covers neural networks and their applications in various domains.",
                related_areas=[],
                bridging_terms=["machine learning", "Neural Networks"],
                coverage_contribution="Adds detail on neural network architectures and applications",
            ),
            DocumentSummaryOut(
                summary="Third document reviews state-of-the-art neural network training techniques.",
                related_areas=[],
                bridging_terms=["MACHINE LEARNING", "neural networks"],
                coverage_contribution="Provides comprehensive training methodology coverage",
            ),
        ]
        # Create context
        ctx = GraphRunContext(state=state, deps=deps)
        # Run FinalizeNode
        node = FinalizeNode()
        result = await node.run(ctx)
        # Check that rerank_terms was called with deduplicated terms
        called_terms = mock_reranker.rerank_terms.call_args[0][1]
        # Should have 3 unique terms (case-insensitive)
        assert len(called_terms) == 3
        # Check that we preserved original casing (first occurrence)
        assert set(called_terms) == {
            "Machine Learning",
            "Deep Learning",
            "Neural Networks",
        }
        # Check final output (End wraps the BridgingTermsOut)
        output = result.data
        assert output.terms == [
            "machine learning",
            "deep learning",
            "neural networks",
        ]
        assert output.scores == [0.9, 0.8, 0.7]

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_empty_bridging_terms(self):
        """Test FinalizeNode handles empty bridging terms gracefully."""
        # Create mock dependencies
        deps = Deps(
            http_client=Mock(),
            fetcher=Mock(),
            search_backend=Mock(),
            reranker=Mock(),
            extractors={},
            resource_pool=ResourcePool(),
            config={},
        )
        # Create state with no bridging terms
        state = State(topic="test topic", max_rounds=1)
        state.current_round = 1
        state.document_summaries = [
            DocumentSummaryOut(
                summary="This document discusses general concepts without specific bridging terms identified.",
                related_areas=[],
                bridging_terms=[],
                coverage_contribution="Provides general background but no specific bridging terms",
            )
        ]
        # Create context
        ctx = GraphRunContext(state=state, deps=deps)
        # Run FinalizeNode
        node = FinalizeNode()
        result = await node.run(ctx)
        # Check output
        output = result.data
        assert output.terms == []
        assert output.scores == []
        assert output.total_documents_processed == 1

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_preserves_first_occurrence_casing(self):
        """Test that case-insensitive dedup preserves first occurrence casing."""
        # Create mock dependencies
        mock_reranker = Mock()
        mock_reranker.rerank_terms = Mock(
            return_value=[("First Case", 0.9), ("second case", 0.8)]
        )
        deps = Deps(
            http_client=Mock(),
            fetcher=Mock(),
            search_backend=Mock(),
            reranker=mock_reranker,
            extractors={},
            resource_pool=ResourcePool(),
            config={},
        )
        # Create state with terms in specific order
        state = State(topic="test", max_rounds=1)
        state.current_round = 1
        state.document_summaries = [
            DocumentSummaryOut(
                summary="First document introduces the topic with specific terminology and case usage patterns.",
                related_areas=[],
                bridging_terms=["First Case"],
                coverage_contribution="Establishes initial terminology and case conventions",
            ),
            DocumentSummaryOut(
                summary="Second document builds on the first with additional terms and different case styles.",
                related_areas=[],
                bridging_terms=["FIRST CASE", "second case"],
                coverage_contribution="Expands terminology with alternative case representations",
            ),
            DocumentSummaryOut(
                summary="Third document concludes with final terms maintaining consistent terminology usage.",
                related_areas=[],
                bridging_terms=["SECOND CASE"],
                coverage_contribution="Completes terminology coverage with final case variations",
            ),
        ]
        # Create context
        ctx = GraphRunContext(state=state, deps=deps)
        # Run FinalizeNode
        node = FinalizeNode()
        await node.run(ctx)
        # Check that first occurrence casing was preserved
        called_terms = mock_reranker.rerank_terms.call_args[0][1]
        assert "First Case" in called_terms
        assert "second case" in called_terms
        # Should not have duplicates
        assert len(called_terms) == 2

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_advanced_normalization_deduplication(self):
        """Test advanced normalization (plurals, abbreviations, suffixes)."""
        # Create mock dependencies
        mock_reranker = Mock()
        mock_reranker.rerank_terms = Mock(
            return_value=[("genetic risk factor", 0.95), ("bmp", 0.90)]
        )
        deps = Deps(
            http_client=Mock(),
            fetcher=Mock(),
            search_backend=Mock(),
            reranker=mock_reranker,
            extractors={},
            resource_pool=ResourcePool(),
            config={},
        )
        # Create state with terms that should deduplicate via advanced normalization
        state = State(topic="test", max_rounds=1)
        state.current_round = 1
        state.document_summaries = [
            DocumentSummaryOut(
                summary="Document one discusses risk factors and signaling pathways in detail with examples.",
                related_areas=[],
                bridging_terms=[
                    "Genetic risk factor",
                    "BMP signaling pathway",
                ],
                coverage_contribution="Establishes foundational terminology for genetic factors",
            ),
            DocumentSummaryOut(
                summary="Document two covers similar topics using plural forms and abbreviations for clarity.",
                related_areas=[],
                bridging_terms=[
                    "genetic risk factors",  # plural
                    "BMP signaling",  # without "pathway"
                ],
                coverage_contribution="Expands on genetic factors using alternative terminology",
            ),
            DocumentSummaryOut(
                summary="Document three adds parenthetical abbreviations and pathway suffix variations.",
                related_areas=[],
                bridging_terms=[
                    "Genetic risk factors (GRF)",  # plural + abbreviation
                    "BMP pathway",  # different suffix
                ],
                coverage_contribution="Provides comprehensive coverage with abbreviations",
            ),
        ]
        # Create context
        ctx = GraphRunContext(state=state, deps=deps)
        # Run FinalizeNode
        node = FinalizeNode()
        await node.run(ctx)
        # Check that advanced normalization deduplicated terms
        called_terms = mock_reranker.rerank_terms.call_args[0][1]
        # Should have only 2 unique terms despite 6 original terms
        assert len(called_terms) == 2
        # Verify original casing preserved (first occurrences)
        assert "Genetic risk factor" in called_terms
        assert "BMP signaling pathway" in called_terms
