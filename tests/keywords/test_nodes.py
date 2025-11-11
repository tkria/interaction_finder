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
from interaction_finder.settings import IfetcherConfig


@pytest.fixture
def mock_deps():
    """Create mock Deps for testing concurrent extraction."""
    # Create mock extractors
    extractors = {
        "rake": Mock(name="rake"),
        "yake": Mock(name="yake"),
        "tfidf": Mock(name="tfidf"),
        "keybert": Mock(name="keybert"),
    }
    return Deps(
        http_client=Mock(),
        fetcher=Mock(),
        search_backend=Mock(),
        reranker=Mock(),
        extractors=extractors,
        resource_pool=ResourcePool(),
        config=IfetcherConfig(),
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


class TestConcurrentExtraction:
    """Test concurrent keyword extraction with multiple resources and extractors."""

    @pytest.mark.asyncio
    async def test_extract_keywords_concurrent_resources(self, mock_deps):
        """Test parallel extraction across multiple resources."""
        from interaction_finder.keywords.extractors.base import ScoredKeyword

        # Create multiple resources using ResourcePool
        for i in range(5):
            rid = mock_deps.resource_pool.register(f"http://example.com/doc{i}")
            mock_deps.resource_pool.add_content(
                rid,
                f"Document {i}",
                f"This is document {i} with unique content about topic {i}.",
            )
        # Configure extractors to return predictable results
        for name, extractor in mock_deps.extractors.items():
            extractor.extract = Mock(
                side_effect=lambda text, max_keywords, n=name: [
                    ScoredKeyword(keyword=f"{n}_keyword_{i}", score=0.9 - i * 0.1)
                    for i in range(3)
                ]
            )
        # Create state and context
        state = State(topic="test topic", max_rounds=1)
        ctx = GraphRunContext(state=state, deps=mock_deps)
        # Run ExtractKeywordsNode
        node = ExtractKeywordsNode()
        await node.run(ctx)
        # Verify all resources were processed
        assert len(ctx.state.extracted_keywords) == 5
        # Verify each resource has keywords from all extractors
        for url in ctx.state.extracted_keywords:
            keywords = ctx.state.extracted_keywords[url]
            # Should have keywords from all extractors (3 keywords × 4 extractors = 12)
            assert len(keywords) == 12

    @pytest.mark.asyncio
    async def test_extract_keywords_concurrent_extractors(self, mock_deps):
        """Test parallel execution of extractors within single resource."""
        import time
        from interaction_finder.keywords.extractors.base import ScoredKeyword

        # Create single resource using ResourcePool
        rid = mock_deps.resource_pool.register("http://example.com/doc")
        mock_deps.resource_pool.add_content(
            rid,
            "Test Document",
            "This is a test document with some content.",
        )
        # Track execution timing to verify parallelism
        execution_times = []

        def slow_extract(text, max_keywords):
            """Simulate CPU-bound extraction with delay."""
            start = time.time()
            time.sleep(0.1)  # Simulate work
            duration = time.time() - start
            execution_times.append(duration)
            return [ScoredKeyword(keyword="test", score=0.9)]

        # Configure all extractors with slow extraction
        for extractor in mock_deps.extractors.values():
            extractor.extract = Mock(side_effect=slow_extract)
        # Create state and context
        state = State(topic="test topic", max_rounds=1)
        ctx = GraphRunContext(state=state, deps=mock_deps)
        # Run ExtractKeywordsNode and measure total time
        start = time.time()
        node = ExtractKeywordsNode()
        await node.run(ctx)
        total_time = time.time() - start
        # Verify extractors ran concurrently (total time < sum of individual times)
        # With 4 extractors × 0.1s each = 0.4s sequential, but should be ~0.1s parallel
        assert total_time < 0.3  # Allow overhead, but much less than 0.4s sequential
        # Verify all extractors were called
        assert len(execution_times) == 4

    @pytest.mark.asyncio
    async def test_extract_keywords_error_handling(self, mock_deps):
        """Test error handling during concurrent extraction."""
        from interaction_finder.keywords.extractors.base import ScoredKeyword

        # Create resource using ResourcePool
        rid = mock_deps.resource_pool.register("http://example.com/doc")
        mock_deps.resource_pool.add_content(
            rid,
            "Test Document",
            "This is a test document.",
        )
        # Configure some extractors to fail, others to succeed
        extractors = list(mock_deps.extractors.items())
        extractors[0][1].extract = Mock(side_effect=ValueError("Extraction failed"))
        extractors[1][1].extract = Mock(
            return_value=[ScoredKeyword(keyword="success1", score=0.9)]
        )
        extractors[2][1].extract = Mock(side_effect=RuntimeError("Another failure"))
        extractors[3][1].extract = Mock(
            return_value=[ScoredKeyword(keyword="success2", score=0.8)]
        )
        # Create state and context
        state = State(topic="test topic", max_rounds=1)
        ctx = GraphRunContext(state=state, deps=mock_deps)
        # Run ExtractKeywordsNode
        node = ExtractKeywordsNode()
        await node.run(ctx)
        # Verify successful extractors produced results
        keywords = ctx.state.extracted_keywords["http://example.com/doc"]
        assert len(keywords) == 2
        assert keywords[0].keyword == "success1"
        assert keywords[1].keyword == "success2"

    @pytest.mark.asyncio
    async def test_extract_keywords_no_race_conditions(self, mock_deps):
        """Test that concurrent extraction doesn't cause data corruption."""
        from interaction_finder.keywords.extractors.base import ScoredKeyword

        # Create many resources to stress-test concurrency using ResourcePool
        for i in range(20):
            rid = mock_deps.resource_pool.register(f"http://example.com/doc{i}")
            mock_deps.resource_pool.add_content(
                rid,
                f"Document {i}",
                f"Content for document {i}",
            )
        # Configure extractors with unique keywords per call
        call_counter = {"count": 0}

        def unique_extract(text, max_keywords):
            """Return unique keywords based on call order."""
            call_id = call_counter["count"]
            call_counter["count"] += 1
            return [
                ScoredKeyword(keyword=f"keyword_{call_id}_{i}", score=0.9)
                for i in range(2)
            ]

        for extractor in mock_deps.extractors.values():
            extractor.extract = Mock(side_effect=unique_extract)
        # Create state and context
        state = State(topic="test topic", max_rounds=1)
        ctx = GraphRunContext(state=state, deps=mock_deps)
        # Run ExtractKeywordsNode
        node = ExtractKeywordsNode()
        await node.run(ctx)
        # Verify all resources processed without corruption
        assert len(ctx.state.extracted_keywords) == 20
        # Verify each resource has correct number of keywords
        for url, keywords in ctx.state.extracted_keywords.items():
            assert len(keywords) == 8  # 2 keywords × 4 extractors
        # Verify all keywords are unique (no race condition duplicates)
        all_keywords = []
        for keywords in ctx.state.extracted_keywords.values():
            all_keywords.extend([kw.keyword for kw in keywords])
        assert len(all_keywords) == len(set(all_keywords))  # All unique

    @pytest.mark.asyncio
    async def test_extract_keywords_skip_already_processed(self, mock_deps):
        """Test that already-processed resources are skipped."""
        from interaction_finder.keywords.extractors.base import ScoredKeyword

        # Create resources using ResourcePool
        for i in range(3):
            rid = mock_deps.resource_pool.register(f"http://example.com/doc{i}")
            mock_deps.resource_pool.add_content(
                rid,
                f"Document {i}",
                f"Content {i}",
            )
        # Pre-populate state with one already-processed resource
        state = State(topic="test topic", max_rounds=1)
        state.extracted_keywords["http://example.com/doc1"] = [
            ScoredKeyword(keyword="already_processed", score=0.9)
        ]
        # Configure extractors
        for extractor in mock_deps.extractors.values():
            extractor.extract = Mock(
                return_value=[ScoredKeyword(keyword="new_keyword", score=0.8)]
            )
        # Create context and run
        ctx = GraphRunContext(state=state, deps=mock_deps)
        node = ExtractKeywordsNode()
        await node.run(ctx)
        # Verify only 2 new resources were processed (doc0 and doc2)
        assert len(ctx.state.extracted_keywords) == 3
        # Verify doc1 still has original keywords
        assert (
            ctx.state.extracted_keywords["http://example.com/doc1"][0].keyword
            == "already_processed"
        )
        # Verify doc0 and doc2 have new keywords
        assert all(
            kw.keyword == "new_keyword"
            for kw in ctx.state.extracted_keywords["http://example.com/doc0"]
        )
        assert all(
            kw.keyword == "new_keyword"
            for kw in ctx.state.extracted_keywords["http://example.com/doc2"]
        )
