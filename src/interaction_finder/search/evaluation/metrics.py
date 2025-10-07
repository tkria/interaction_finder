"""
Evaluation metrics for search backends and approaches.

This module provides metrics for comparing search effectiveness including
precision, recall, relevance scoring, and comparative analysis.
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple
from datetime import datetime
import statistics

from pydantic import BaseModel, Field

from ..base import SearchQuery, SearchResults, SearchResult


@dataclass
class QueryMetrics:
    """Metrics for a single query evaluation."""

    query: str
    backend: str
    total_results: int = 0
    relevant_results: int = 0
    search_time_seconds: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    average_relevance_score: float = 0.0
    unique_results: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class ComparisonMetrics:
    """Comparative metrics between different search approaches."""

    query: str
    backend_a: str
    backend_b: str
    overlap_count: int = 0
    unique_to_a: int = 0
    unique_to_b: int = 0
    jaccard_similarity: float = 0.0
    rank_correlation: float = 0.0
    avg_relevance_difference: float = 0.0


class EvaluationConfig(BaseModel):
    """Configuration for search evaluation."""

    relevance_threshold: float = Field(
        0.7, description="Threshold for considering results relevant", ge=0.0, le=1.0
    )
    max_results_to_evaluate: int = Field(
        50, description="Maximum number of results to evaluate per query", ge=1, le=1000
    )
    timeout_seconds: int = Field(
        60, description="Timeout for search operations", ge=10, le=300
    )
    calculate_manual_relevance: bool = Field(
        False, description="Whether to request manual relevance judgments"
    )


class SearchEvaluator:
    """Evaluator for comparing search backends and approaches."""

    def __init__(self, config: EvaluationConfig):
        self.config = config

    async def evaluate_query(
        self,
        query: str,
        backend_name: str,
        search_backend,
        ground_truth: Optional[Set[str]] = None,
        relevance_scores: Optional[Dict[str, float]] = None,
    ) -> QueryMetrics:
        """
        Evaluate a single query against a search backend.

        Args:
            query: Search query to evaluate
            backend_name: Name of the search backend
            search_backend: Search backend instance
            ground_truth: Set of known relevant document URLs/IDs
            relevance_scores: Manual relevance scores for documents

        Returns:
            QueryMetrics with evaluation results
        """
        metrics = QueryMetrics(query=query, backend=backend_name)

        try:
            # Time the search operation
            start_time = datetime.now()

            search_query = SearchQuery(
                query=query, max_results=self.config.max_results_to_evaluate
            )

            results = await search_backend.search(search_query)
            end_time = datetime.now()

            metrics.search_time_seconds = (end_time - start_time).total_seconds()
            metrics.total_results = len(results.results)

            # Calculate unique results (by URL)
            unique_urls = set()
            for result in results.results:
                if result.url:
                    unique_urls.add(result.url)
            metrics.unique_results = len(unique_urls)

            # If we have ground truth, calculate precision and recall
            if ground_truth:
                relevant_found = set()
                for result in results.results[: self.config.max_results_to_evaluate]:
                    if result.url in ground_truth or (
                        result.pmid and result.pmid in ground_truth
                    ):
                        relevant_found.add(result.url or result.pmid)

                metrics.relevant_results = len(relevant_found)

                if metrics.total_results > 0:
                    metrics.precision = metrics.relevant_results / min(
                        metrics.total_results, self.config.max_results_to_evaluate
                    )

                if len(ground_truth) > 0:
                    metrics.recall = len(relevant_found) / len(ground_truth)

                if metrics.precision + metrics.recall > 0:
                    metrics.f1_score = (
                        2
                        * (metrics.precision * metrics.recall)
                        / (metrics.precision + metrics.recall)
                    )

            # If we have relevance scores, calculate average relevance
            if relevance_scores:
                relevance_values = []
                for result in results.results[: self.config.max_results_to_evaluate]:
                    key = result.url or result.pmid
                    if key and key in relevance_scores:
                        relevance_values.append(relevance_scores[key])

                if relevance_values:
                    metrics.average_relevance_score = statistics.mean(relevance_values)
                    metrics.relevant_results = sum(
                        1
                        for score in relevance_values
                        if score >= self.config.relevance_threshold
                    )

        except Exception as e:
            metrics.errors.append(str(e))

        return metrics

    async def compare_backends(
        self,
        query: str,
        backend_a_name: str,
        backend_a,
        backend_b_name: str,
        backend_b,
    ) -> ComparisonMetrics:
        """
        Compare two search backends for a given query.

        Args:
            query: Search query to compare
            backend_a_name: Name of first backend
            backend_a: First search backend instance
            backend_b_name: Name of second backend
            backend_b: Second search backend instance

        Returns:
            ComparisonMetrics with comparison results
        """
        comparison = ComparisonMetrics(
            query=query, backend_a=backend_a_name, backend_b=backend_b_name
        )

        try:
            # Get results from both backends
            search_query = SearchQuery(
                query=query, max_results=self.config.max_results_to_evaluate
            )

            results_a, results_b = await asyncio.gather(
                backend_a.search(search_query),
                backend_b.search(search_query),
                return_exceptions=True,
            )

            if isinstance(results_a, Exception) or isinstance(results_b, Exception):
                return comparison

            # Extract result URLs/IDs for comparison
            def get_result_ids(results: SearchResults) -> Set[str]:
                ids = set()
                for result in results.results:
                    if result.url:
                        ids.add(result.url)
                    elif result.pmid:
                        ids.add(result.pmid)
                return ids

            ids_a = get_result_ids(results_a)
            ids_b = get_result_ids(results_b)

            # Calculate overlap metrics
            overlap = ids_a & ids_b
            comparison.overlap_count = len(overlap)
            comparison.unique_to_a = len(ids_a - ids_b)
            comparison.unique_to_b = len(ids_b - ids_a)

            # Jaccard similarity
            union = ids_a | ids_b
            if union:
                comparison.jaccard_similarity = len(overlap) / len(union)

            # Simple rank correlation (if results have scores)
            common_results = []
            for result_a in results_a.results:
                key_a = result_a.url or result_a.pmid
                if key_a in ids_b:
                    for i, result_b in enumerate(results_b.results):
                        key_b = result_b.url or result_b.pmid
                        if key_a == key_b:
                            rank_a = results_a.results.index(result_a)
                            rank_b = i
                            common_results.append((rank_a, rank_b))
                            break

            if len(common_results) > 1:
                ranks_a = [r[0] for r in common_results]
                ranks_b = [r[1] for r in common_results]

                # Simple correlation coefficient
                if len(set(ranks_a)) > 1 and len(set(ranks_b)) > 1:
                    try:
                        correlation = statistics.correlation(ranks_a, ranks_b)
                        comparison.rank_correlation = correlation
                    except statistics.StatisticsError:
                        pass

        except Exception:
            pass  # Return empty comparison on error

        return comparison

    async def evaluate_multiple_queries(
        self,
        queries: List[str],
        backend_name: str,
        search_backend,
        ground_truth_map: Optional[Dict[str, Set[str]]] = None,
        relevance_scores_map: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> List[QueryMetrics]:
        """
        Evaluate multiple queries against a search backend.

        Args:
            queries: List of search queries
            backend_name: Name of the search backend
            search_backend: Search backend instance
            ground_truth_map: Map of query -> set of relevant document URLs/IDs
            relevance_scores_map: Map of query -> map of document -> relevance score

        Returns:
            List of QueryMetrics for each query
        """
        results = []

        # Run evaluations concurrently with limited concurrency
        semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent evaluations

        async def evaluate_single(query: str) -> QueryMetrics:
            async with semaphore:
                ground_truth = ground_truth_map.get(query) if ground_truth_map else None
                relevance_scores = (
                    relevance_scores_map.get(query) if relevance_scores_map else None
                )
                return await self.evaluate_query(
                    query, backend_name, search_backend, ground_truth, relevance_scores
                )

        tasks = [evaluate_single(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        return [r for r in results if isinstance(r, QueryMetrics)]


def calculate_aggregate_metrics(query_metrics: List[QueryMetrics]) -> Dict[str, float]:
    """
    Calculate aggregate metrics across multiple query evaluations.

    Args:
        query_metrics: List of QueryMetrics from multiple queries

    Returns:
        Dictionary with aggregate metrics
    """
    if not query_metrics:
        return {}

    valid_metrics = [m for m in query_metrics if not m.errors]

    if not valid_metrics:
        return {"error_rate": 1.0}

    aggregates = {
        "total_queries": len(query_metrics),
        "successful_queries": len(valid_metrics),
        "error_rate": (len(query_metrics) - len(valid_metrics)) / len(query_metrics),
        "avg_total_results": statistics.mean(m.total_results for m in valid_metrics),
        "avg_search_time": statistics.mean(
            m.search_time_seconds for m in valid_metrics
        ),
    }

    # Add optional metrics only if data is available
    precision_values = [m.precision for m in valid_metrics if m.precision > 0]
    if precision_values:
        aggregates["avg_precision"] = statistics.mean(precision_values)

    recall_values = [m.recall for m in valid_metrics if m.recall > 0]
    if recall_values:
        aggregates["avg_recall"] = statistics.mean(recall_values)

    f1_values = [m.f1_score for m in valid_metrics if m.f1_score > 0]
    if f1_values:
        aggregates["avg_f1_score"] = statistics.mean(f1_values)

    relevance_values = [
        m.average_relevance_score
        for m in valid_metrics
        if m.average_relevance_score > 0
    ]
    if relevance_values:
        aggregates["avg_relevance_score"] = statistics.mean(relevance_values)

    # Add median values for key metrics
    if valid_metrics:
        aggregates.update(
            {
                "median_search_time": statistics.median(
                    m.search_time_seconds for m in valid_metrics
                ),
                "median_total_results": statistics.median(
                    m.total_results for m in valid_metrics
                ),
            }
        )

    return aggregates
