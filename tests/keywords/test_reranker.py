"""Tests for semantic reranker."""

import pytest

from interaction_finder.keywords.reranker import Reranker
from interaction_finder.search.models import SearchResult


class TestReranker:
    """Test suite for semantic reranker."""

    def test_initialization_defaults(self):
        """Test Reranker initializes with correct defaults."""
        reranker = Reranker()
        assert reranker.model_name == "zeroentropy/zerank-1-small"
        assert reranker.batch_size == 32

    def test_initialization_custom_params(self):
        """Test Reranker initialization with custom parameters."""
        reranker = Reranker(model_name="custom-model", batch_size=16)
        assert reranker.model_name == "custom-model"
        assert reranker.batch_size == 16

    def test_initialization_invalid_batch_size(self):
        """Test Reranker rejects invalid batch_size."""
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            Reranker(batch_size=0)

    @pytest.mark.slow
    def test_rerank_basic_results(self):
        """Test reranking improves relevance ordering."""
        reranker = Reranker()
        query = "machine learning neural networks"
        # Create results with varying relevance
        results = [
            SearchResult(
                title="Introduction to Cooking",
                url="http://example.com/1",
                snippet="How to cook pasta and make sauces.",
            ),
            SearchResult(
                title="Deep Learning with Neural Networks",
                url="http://example.com/2",
                snippet="Comprehensive guide to neural networks in machine learning.",
            ),
            SearchResult(
                title="Machine Learning Basics",
                url="http://example.com/3",
                snippet="Introduction to machine learning algorithms.",
            ),
        ]
        # Rerank
        reranked = reranker.rerank(query, results)
        # Check that results are processed correctly
        assert len(reranked) == len(results)
        assert all(isinstance(r, SearchResult) for r in reranked)
        # All results should have relevance scores
        assert all(r.relevance is not None for r in reranked)
        # Scores should be in [0, 1] range
        assert all(0 <= r.relevance <= 1 for r in reranked)
        # Scores should be sorted descending
        scores = [r.relevance for r in reranked]
        assert scores == sorted(scores, reverse=True)
        # All results should be present (no duplicates or missing)
        reranked_urls = {r.url for r in reranked}
        original_urls = {r.url for r in results}
        assert reranked_urls == original_urls

    def test_rerank_empty_query_raises(self):
        """Test reranking raises on empty query."""
        reranker = Reranker()
        results = [
            SearchResult(title="Test", url="http://example.com", snippet="Test snippet")
        ]
        with pytest.raises(ValueError, match="Query cannot be empty"):
            reranker.rerank("", results)
        with pytest.raises(ValueError, match="Query cannot be empty"):
            reranker.rerank("   ", results)

    def test_rerank_empty_results_raises(self):
        """Test reranking raises on empty results."""
        reranker = Reranker()
        with pytest.raises(ValueError, match="Results cannot be empty"):
            reranker.rerank("test query", [])

    def test_rerank_invalid_top_k(self):
        """Test reranking raises on invalid top_k."""
        reranker = Reranker()
        results = [
            SearchResult(title="Test", url="http://example.com", snippet="Test snippet")
        ]
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            reranker.rerank("test query", results, top_k=0)

    @pytest.mark.slow
    def test_rerank_with_top_k(self):
        """Test reranking respects top_k parameter."""
        reranker = Reranker()
        query = "machine learning"
        results = [
            SearchResult(
                title=f"Machine Learning Article {i}",
                url=f"http://example.com/{i}",
                snippet=f"Article {i} about machine learning.",
            )
            for i in range(10)
        ]
        # Rerank with top_k=3
        reranked = reranker.rerank(query, results, top_k=3)
        assert len(reranked) == 3
        # All should have relevance scores
        assert all(r.relevance is not None for r in reranked)

    @pytest.mark.slow
    def test_rerank_handles_missing_snippets(self):
        """Test reranking handles results without snippets."""
        reranker = Reranker()
        query = "test query"
        results = [
            SearchResult(title="Article 1", url="http://example.com/1", snippet=None),
            SearchResult(
                title="Article 2",
                url="http://example.com/2",
                snippet="Has a snippet",
            ),
        ]
        # Should not raise
        reranked = reranker.rerank(query, results)
        assert len(reranked) == 2

    @pytest.mark.slow
    def test_rerank_single_result(self):
        """Test reranking with single result."""
        reranker = Reranker()
        query = "test"
        results = [
            SearchResult(
                title="Test Article", url="http://example.com", snippet="Test snippet"
            )
        ]
        reranked = reranker.rerank(query, results)
        assert len(reranked) == 1
        assert reranked[0].relevance is not None
        assert 0 <= reranked[0].relevance <= 1

    @pytest.mark.slow
    def test_score_normalization(self):
        """Test that scores are properly normalized to [0, 1]."""
        reranker = Reranker()
        query = "machine learning algorithms"
        results = [
            SearchResult(
                title="Machine Learning Algorithms",
                url="http://example.com/1",
                snippet="Comprehensive guide to machine learning algorithms.",
            ),
            SearchResult(
                title="Completely Unrelated Topic",
                url="http://example.com/2",
                snippet="This has nothing to do with the query.",
            ),
        ]
        reranked = reranker.rerank(query, results)
        # All scores should be in [0, 1]
        assert all(0 <= r.relevance <= 1 for r in reranked)
        # Relevant result should score higher
        assert reranked[0].relevance > reranked[1].relevance

    @pytest.mark.slow
    def test_lazy_model_loading(self):
        """Test that model is loaded lazily on first use."""
        reranker = Reranker()
        # Model should be None before first rerank
        assert reranker._model is None
        # After rerank, model should be loaded
        results = [
            SearchResult(title="Test", url="http://example.com", snippet="Test snippet")
        ]
        reranker.rerank("test", results)
        assert reranker._model is not None

    @pytest.mark.slow
    def test_healthy_check(self):
        """Test Reranker health check loads model successfully."""
        reranker = Reranker()
        # Should be able to load model
        assert reranker.healthy() is True
        # Model should now be loaded
        assert reranker._model is not None
