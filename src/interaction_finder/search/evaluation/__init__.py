"""
Evaluation and comparison tools for search backends.

This package provides metrics, benchmarks, and comparison utilities for
evaluating the effectiveness of different search approaches and backends.
"""

from .metrics import (
    SearchEvaluator,
    EvaluationConfig,
    QueryMetrics,
    ComparisonMetrics,
    calculate_aggregate_metrics,
)
from .runner import (
    EvaluationRunner,
    DEFAULT_BIOMEDICAL_QUERIES,
    DEFAULT_GENERAL_QUERIES,
)

__all__ = [
    # Core evaluation
    "SearchEvaluator",
    "EvaluationConfig",
    "QueryMetrics",
    "ComparisonMetrics",
    "calculate_aggregate_metrics",
    # Evaluation runner
    "EvaluationRunner",
    "DEFAULT_BIOMEDICAL_QUERIES",
    "DEFAULT_GENERAL_QUERIES",
]
