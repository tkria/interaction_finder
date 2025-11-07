"""Tests for widesearch Reranker."""

import pytest

from interaction_finder.search.models import SearchResult
from interaction_finder.widesearch.reranker import Reranker


class TestReranker:
    """Tests for Reranker class."""

    def test_initialization(self):
        """Test reranker initialization."""
        reranker = Reranker()
        assert reranker.model_name == "zeroentropy/zerank-1-small"
        assert reranker.batch_size == 1
        assert reranker.device is None

    def test_custom_initialization(self):
        """Test reranker with custom parameters."""
        reranker = Reranker(model_name="custom/model", batch_size=4, device="cpu")
        assert reranker.model_name == "custom/model"
        assert reranker.batch_size == 4
        assert reranker.device == "cpu"

    def test_invalid_batch_size(self):
        """Test that invalid batch size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            Reranker(batch_size=0)

    def test_rerank_empty_query(self):
        """Test that empty query raises ValueError."""
        reranker = Reranker()
        results = [
            SearchResult(title="Test", url="https://example.com", snippet="snippet")
        ]
        with pytest.raises(ValueError, match="Query cannot be empty"):
            reranker.rerank("", results)

    def test_rerank_empty_results(self):
        """Test that empty results raises ValueError."""
        reranker = Reranker()
        with pytest.raises(ValueError, match="Results cannot be empty"):
            reranker.rerank("test query", [])

    def test_rerank_invalid_top_k(self):
        """Test that invalid top_k raises ValueError."""
        reranker = Reranker()
        results = [
            SearchResult(title="Test", url="https://example.com", snippet="snippet")
        ]
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            reranker.rerank("test query", results, top_k=0)

    def test_rerank_basic(self):
        """Test basic reranking functionality."""
        reranker = Reranker(device="cpu")
        results = [
            SearchResult(
                title="Diabetes and insulin",
                url="https://example.com/1",
                snippet="Study on diabetes treatment",
            ),
            SearchResult(
                title="Cancer therapy",
                url="https://example.com/2",
                snippet="New cancer treatment approach",
            ),
            SearchResult(
                title="Diabetes management strategies",
                url="https://example.com/3",
                snippet="Comprehensive diabetes care",
            ),
        ]

        reranked = reranker.rerank("diabetes treatment", results)

        # Check that results were returned
        assert len(reranked) == 3
        # Check that relevance scores were added
        assert all(r.relevance is not None for r in reranked)
        # Check that scores are in [0, 1] range
        assert all(0.0 <= r.relevance <= 1.0 for r in reranked)
        # Check that results are sorted by relevance
        scores = [r.relevance for r in reranked]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_with_top_k(self):
        """Test reranking with top_k limit."""
        reranker = Reranker(device="cpu")
        results = [
            SearchResult(
                title=f"Paper {i}",
                url=f"https://example.com/{i}",
                snippet="test",
            )
            for i in range(10)
        ]

        reranked = reranker.rerank("test query", results, top_k=3)

        assert len(reranked) == 3
        assert all(r.relevance is not None for r in reranked)

    def test_rerank_without_snippet(self):
        """Test reranking works with results without snippets."""
        reranker = Reranker(device="cpu")
        results = [
            SearchResult(
                title="Paper without snippet",
                url="https://example.com/1",
                snippet=None,
            ),
            SearchResult(
                title="Paper with snippet",
                url="https://example.com/2",
                snippet="This has content",
            ),
        ]

        reranked = reranker.rerank("test query", results)

        assert len(reranked) == 2
        assert all(r.relevance is not None for r in reranked)

    def test_healthy(self):
        """Test healthy method."""
        reranker = Reranker(device="cpu")
        # Should be able to load model
        assert reranker.healthy() is True
