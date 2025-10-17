"""Tests for search evaluation framework."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from interaction_finder.search.evaluation.metrics import (
    SearchEvaluator,
    EvaluationConfig,
    QueryMetrics,
    ComparisonMetrics,
    calculate_aggregate_metrics,
)
from interaction_finder.search.base import SearchQuery, SearchResult, SearchResults


@pytest.fixture
def eval_config():
    """Create evaluation configuration for testing."""
    return EvaluationConfig(
        relevance_threshold=0.7,
        max_results_to_evaluate=20,
        timeout_seconds=30,
    )


@pytest.fixture
def evaluator(eval_config):
    """Create SearchEvaluator for testing."""
    return SearchEvaluator(eval_config)


@pytest.fixture
def sample_results():
    """Sample search results for testing."""
    query = SearchQuery(query="BRCA1 mutations")
    return SearchResults(
        query=query,
        backend="pubmed",
        results=[
            SearchResult(
                title="BRCA1 mutations in breast cancer",
                url="https://pubmed.ncbi.nlm.nih.gov/123456",
                backend="pubmed",
                metadata={"pmid": "123456"},
            ),
            SearchResult(
                title="Another BRCA1 paper",
                url="https://pubmed.ncbi.nlm.nih.gov/789012",
                backend="pubmed",
                metadata={"pmid": "789012"},
            ),
            SearchResult(
                title="Unrelated paper",
                url="https://pubmed.ncbi.nlm.nih.gov/345678",
                backend="pubmed",
                metadata={"pmid": "345678"},
            ),
        ],
        total_found=100,
        search_time=1.5,
    )


class TestEvaluationConfig:
    """Test EvaluationConfig validation."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = EvaluationConfig()
        assert config.relevance_threshold == 0.7
        assert config.max_results_to_evaluate == 50
        assert config.timeout_seconds == 60

    def test_custom_config(self):
        """Test custom configuration values."""
        config = EvaluationConfig(
            relevance_threshold=0.8,
            max_results_to_evaluate=100,
            timeout_seconds=120,
        )
        assert config.relevance_threshold == 0.8
        assert config.max_results_to_evaluate == 100
        assert config.timeout_seconds == 120

    def test_invalid_relevance_threshold(self):
        """Test invalid relevance threshold."""
        with pytest.raises(ValueError):
            EvaluationConfig(relevance_threshold=1.5)

        with pytest.raises(ValueError):
            EvaluationConfig(relevance_threshold=-0.1)


class TestSearchEvaluator:
    """Test SearchEvaluator functionality."""

    @pytest.mark.asyncio
    async def test_evaluate_query_basic(self, evaluator, sample_results):
        """Test basic query evaluation without ground truth."""
        # Mock backend
        mock_backend = AsyncMock()
        mock_backend.search.return_value = sample_results

        metrics = await evaluator.evaluate_query(
            query="BRCA1 mutations",
            backend_name="pubmed",
            search_backend=mock_backend,
        )

        assert metrics.query == "BRCA1 mutations"
        assert metrics.backend == "pubmed"
        assert metrics.total_results == 3
        assert metrics.unique_results == 3
        assert metrics.search_time_seconds > 0
        assert metrics.precision == 0.0  # No ground truth provided
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_query_with_ground_truth(self, evaluator, sample_results):
        """Test query evaluation with ground truth."""
        # Mock backend
        mock_backend = AsyncMock()
        mock_backend.search.return_value = sample_results

        # Define ground truth (first two results are relevant)
        ground_truth = {"123456", "789012"}

        metrics = await evaluator.evaluate_query(
            query="BRCA1 mutations",
            backend_name="pubmed",
            search_backend=mock_backend,
            ground_truth=ground_truth,
        )

        assert metrics.relevant_results == 2
        assert metrics.precision == 2 / 3  # 2 relevant out of 3 results
        assert metrics.recall == 2 / 2  # 2 relevant out of 2 total relevant
        assert metrics.f1_score == 2 * (metrics.precision * metrics.recall) / (
            metrics.precision + metrics.recall
        )

    @pytest.mark.asyncio
    async def test_evaluate_query_with_relevance_scores(
        self, evaluator, sample_results
    ):
        """Test query evaluation with relevance scores."""
        # Mock backend
        mock_backend = AsyncMock()
        mock_backend.search.return_value = sample_results

        # Define relevance scores
        relevance_scores = {
            "123456": 0.9,  # Highly relevant
            "789012": 0.8,  # Relevant
            "345678": 0.5,  # Below threshold (0.7)
        }

        metrics = await evaluator.evaluate_query(
            query="BRCA1 mutations",
            backend_name="pubmed",
            search_backend=mock_backend,
            relevance_scores=relevance_scores,
        )

        assert metrics.average_relevance_score == (0.9 + 0.8 + 0.5) / 3
        assert metrics.relevant_results == 2  # Only scores >= 0.7

    @pytest.mark.asyncio
    async def test_evaluate_query_error_handling(self, evaluator):
        """Test error handling in query evaluation."""
        # Mock backend that raises exception
        mock_backend = AsyncMock()
        mock_backend.search.side_effect = Exception("Search failed")

        metrics = await evaluator.evaluate_query(
            query="test query",
            backend_name="pubmed",
            search_backend=mock_backend,
        )

        assert len(metrics.errors) == 1
        assert "Search failed" in metrics.errors[0]
        assert metrics.total_results == 0

    @pytest.mark.asyncio
    async def test_compare_backends(self, evaluator):
        """Test backend comparison."""
        # Create mock results for two backends
        query_a = SearchQuery(query="test query")
        results_a = SearchResults(
            query=query_a,
            backend="pubmed",
            results=[
                SearchResult(
                    title="Paper 1", url="https://example.com/1", backend="pubmed"
                ),
                SearchResult(
                    title="Paper 2", url="https://example.com/2", backend="pubmed"
                ),
                SearchResult(
                    title="Paper 3", url="https://example.com/3", backend="pubmed"
                ),
            ],
            total_found=50,
        )

        query_b = SearchQuery(query="test query")
        results_b = SearchResults(
            query=query_b,
            backend="perplexica",
            results=[
                SearchResult(
                    title="Paper 2", url="https://example.com/2", backend="perplexica"
                ),  # Overlap
                SearchResult(
                    title="Paper 4", url="https://example.com/4", backend="perplexica"
                ),
                SearchResult(
                    title="Paper 5", url="https://example.com/5", backend="perplexica"
                ),
            ],
            total_found=30,
        )

        # Mock backends
        mock_backend_a = AsyncMock()
        mock_backend_a.search.return_value = results_a

        mock_backend_b = AsyncMock()
        mock_backend_b.search.return_value = results_b

        comparison = await evaluator.compare_backends(
            query="test query",
            backend_a_name="pubmed",
            backend_a=mock_backend_a,
            backend_b_name="perplexica",
            backend_b=mock_backend_b,
        )

        assert comparison.query == "test query"
        assert comparison.backend_a == "pubmed"
        assert comparison.backend_b == "perplexica"
        assert comparison.overlap_count == 1  # Paper 2 is in both
        assert comparison.unique_to_a == 2  # Papers 1 and 3
        assert comparison.unique_to_b == 2  # Papers 4 and 5
        assert comparison.jaccard_similarity == 1 / 5  # 1 overlap, 5 total unique

    @pytest.mark.asyncio
    async def test_evaluate_multiple_queries(self, evaluator, sample_results):
        """Test evaluating multiple queries."""
        queries = ["query 1", "query 2", "query 3"]

        # Mock backend
        mock_backend = AsyncMock()
        mock_backend.search.return_value = sample_results

        metrics_list = await evaluator.evaluate_multiple_queries(
            queries=queries,
            backend_name="pubmed",
            search_backend=mock_backend,
        )

        assert len(metrics_list) == 3
        for i, metrics in enumerate(metrics_list):
            assert metrics.query == queries[i]
            assert metrics.backend == "pubmed"
            assert metrics.total_results == 3


class TestAggregateMetrics:
    """Test aggregate metrics calculation."""

    def test_calculate_aggregate_metrics_empty(self):
        """Test aggregate calculation with empty list."""
        aggregates = calculate_aggregate_metrics([])
        assert aggregates == {}

    def test_calculate_aggregate_metrics_with_errors(self):
        """Test aggregate calculation with all errors."""
        metrics_with_errors = [
            QueryMetrics(query="query 1", backend="pubmed", errors=["Error 1"]),
            QueryMetrics(query="query 2", backend="pubmed", errors=["Error 2"]),
        ]

        aggregates = calculate_aggregate_metrics(metrics_with_errors)
        assert aggregates["error_rate"] == 1.0

    def test_calculate_aggregate_metrics_success(self):
        """Test aggregate calculation with successful metrics."""
        metrics_list = [
            QueryMetrics(
                query="query 1",
                backend="pubmed",
                total_results=10,
                search_time_seconds=1.0,
                precision=0.8,
                recall=0.6,
                f1_score=0.68,
            ),
            QueryMetrics(
                query="query 2",
                backend="pubmed",
                total_results=15,
                search_time_seconds=1.5,
                precision=0.9,
                recall=0.7,
                f1_score=0.78,
            ),
        ]

        aggregates = calculate_aggregate_metrics(metrics_list)

        assert aggregates["total_queries"] == 2
        assert aggregates["successful_queries"] == 2
        assert aggregates["error_rate"] == 0.0
        assert aggregates["avg_total_results"] == 12.5
        assert aggregates["avg_search_time"] == 1.25
        assert abs(aggregates["avg_precision"] - 0.85) < 1e-10
        assert abs(aggregates["avg_recall"] - 0.65) < 1e-10
        assert abs(aggregates["avg_f1_score"] - 0.73) < 1e-10

    def test_calculate_aggregate_metrics_mixed(self):
        """Test aggregate calculation with mixed success/error metrics."""
        metrics_list = [
            QueryMetrics(
                query="query 1",
                backend="pubmed",
                total_results=10,
                search_time_seconds=1.0,
                precision=0.8,
            ),
            QueryMetrics(query="query 2", backend="pubmed", errors=["Search failed"]),
            QueryMetrics(
                query="query 3",
                backend="pubmed",
                total_results=20,
                search_time_seconds=2.0,
                precision=0.6,
            ),
        ]

        aggregates = calculate_aggregate_metrics(metrics_list)

        assert aggregates["total_queries"] == 3
        assert aggregates["successful_queries"] == 2
        assert aggregates["error_rate"] == 1 / 3
        assert aggregates["avg_total_results"] == 15.0  # (10 + 20) / 2
        assert aggregates["avg_search_time"] == 1.5  # (1.0 + 2.0) / 2
