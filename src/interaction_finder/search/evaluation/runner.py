"""
Evaluation runner for search backend comparisons.

This module provides tools for running comprehensive evaluations of search
backends with predefined query sets and generating comparison reports.
"""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

import aiofiles
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .metrics import (
    SearchEvaluator,
    EvaluationConfig,
    QueryMetrics,
    ComparisonMetrics,
    calculate_aggregate_metrics,
)
from ..backends.pubmed import PubMedBackend
from ..backends.perplexica import PerplexicaBackend

console = Console()


class EvaluationRunner:
    """Runner for comprehensive search backend evaluations."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.evaluator = SearchEvaluator(config)

    async def run_backend_evaluation(
        self,
        queries: List[str],
        backend_name: str,
        backend_config: Dict[str, Any],
        ground_truth_map: Optional[Dict[str, set]] = None,
        output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run evaluation for a single backend.

        Args:
            queries: List of test queries
            backend_name: Name of the backend to test
            backend_config: Configuration for the backend
            ground_truth_map: Optional ground truth data
            output_file: Optional file to save results

        Returns:
            Dictionary with evaluation results
        """
        console.print(f"[blue]Evaluating {backend_name} backend...[/blue]")

        # Create backend instance
        if backend_name == "pubmed":
            backend = PubMedBackend(backend_config)
        elif backend_name == "perplexica":
            backend = PerplexicaBackend(backend_config)
        else:
            raise ValueError(f"Unsupported backend: {backend_name}")

        results = []

        async with backend:
            with console.status(f"Running {len(queries)} queries..."):
                query_metrics = await self.evaluator.evaluate_multiple_queries(
                    queries, backend_name, backend, ground_truth_map
                )
                results.extend(query_metrics)

        # Calculate aggregate metrics
        aggregates = calculate_aggregate_metrics(results)

        evaluation_results = {
            "timestamp": datetime.now().isoformat(),
            "backend": backend_name,
            "config": backend_config,
            "queries": queries,
            "individual_results": [
                {
                    "query": m.query,
                    "total_results": m.total_results,
                    "relevant_results": m.relevant_results,
                    "search_time_seconds": m.search_time_seconds,
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1_score": m.f1_score,
                    "average_relevance_score": m.average_relevance_score,
                    "unique_results": m.unique_results,
                    "errors": m.errors,
                }
                for m in results
            ],
            "aggregate_metrics": aggregates,
        }

        # Save results if requested
        if output_file:
            await self._save_results(evaluation_results, output_file)

        return evaluation_results

    async def run_comparison_evaluation(
        self,
        queries: List[str],
        backend_configs: Dict[str, Dict[str, Any]],
        output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run comparative evaluation between multiple backends.

        Args:
            queries: List of test queries
            backend_configs: Map of backend name -> configuration
            output_file: Optional file to save results

        Returns:
            Dictionary with comparison results
        """
        console.print("[blue]Running comparative evaluation...[/blue]")

        if len(backend_configs) < 2:
            raise ValueError("Need at least 2 backends for comparison")

        # Run individual evaluations
        backend_results = {}
        for backend_name, config in backend_configs.items():
            backend_results[backend_name] = await self.run_backend_evaluation(
                queries, backend_name, config
            )

        # Run pairwise comparisons
        comparison_results = []
        backend_names = list(backend_configs.keys())

        for i in range(len(backend_names)):
            for j in range(i + 1, len(backend_names)):
                backend_a = backend_names[i]
                backend_b = backend_names[j]

                console.print(f"[dim]Comparing {backend_a} vs {backend_b}...[/dim]")

                # Create backend instances for comparison
                backend_a_instance = self._create_backend(
                    backend_a, backend_configs[backend_a]
                )
                backend_b_instance = self._create_backend(
                    backend_b, backend_configs[backend_b]
                )

                comparisons = []
                async with backend_a_instance, backend_b_instance:
                    for query in queries:
                        comparison = await self.evaluator.compare_backends(
                            query,
                            backend_a,
                            backend_a_instance,
                            backend_b,
                            backend_b_instance,
                        )
                        comparisons.append(comparison)

                comparison_results.append(
                    {
                        "backend_a": backend_a,
                        "backend_b": backend_b,
                        "comparisons": [
                            {
                                "query": c.query,
                                "overlap_count": c.overlap_count,
                                "unique_to_a": c.unique_to_a,
                                "unique_to_b": c.unique_to_b,
                                "jaccard_similarity": c.jaccard_similarity,
                                "rank_correlation": c.rank_correlation,
                            }
                            for c in comparisons
                        ],
                        "average_jaccard_similarity": sum(
                            c.jaccard_similarity for c in comparisons
                        )
                        / len(comparisons)
                        if comparisons
                        else 0,
                        "average_overlap": sum(c.overlap_count for c in comparisons)
                        / len(comparisons)
                        if comparisons
                        else 0,
                    }
                )

        evaluation_results = {
            "timestamp": datetime.now().isoformat(),
            "queries": queries,
            "backend_results": backend_results,
            "comparisons": comparison_results,
        }

        # Save results if requested
        if output_file:
            await self._save_results(evaluation_results, output_file)

        return evaluation_results

    def _create_backend(self, backend_name: str, config: Dict[str, Any]):
        """Create backend instance from name and config."""
        if backend_name == "pubmed":
            return PubMedBackend(config)
        elif backend_name == "perplexica":
            return PerplexicaBackend(config)
        else:
            raise ValueError(f"Unsupported backend: {backend_name}")

    async def _save_results(self, results: Dict[str, Any], output_file: str) -> None:
        """Save evaluation results to file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(results, indent=2, ensure_ascii=False))

        console.print(f"[green]Results saved to {output_path}[/green]")

    def display_evaluation_results(self, results: Dict[str, Any]) -> None:
        """Display evaluation results in a formatted table."""
        if "backend_results" in results:
            # Comparison results
            self._display_comparison_results(results)
        else:
            # Single backend results
            self._display_single_backend_results(results)

    def _display_single_backend_results(self, results: Dict[str, Any]) -> None:
        """Display results for a single backend evaluation."""
        backend = results["backend"]
        aggregates = results["aggregate_metrics"]

        console.print(Panel(f"[bold]Evaluation Results: {backend}[/bold]"))

        # Summary table
        summary_table = Table(title="Summary Metrics")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", justify="right")

        summary_table.add_row(
            "Total Queries", str(aggregates.get("total_queries", "N/A"))
        )
        summary_table.add_row(
            "Successful Queries", str(aggregates.get("successful_queries", "N/A"))
        )
        summary_table.add_row(
            "Error Rate", f"{aggregates.get('error_rate', 0) * 100:.1f}%"
        )
        summary_table.add_row(
            "Avg Results per Query", f"{aggregates.get('avg_total_results', 0):.1f}"
        )
        summary_table.add_row(
            "Avg Search Time", f"{aggregates.get('avg_search_time', 0):.2f}s"
        )

        if "avg_precision" in aggregates:
            summary_table.add_row("Avg Precision", f"{aggregates['avg_precision']:.3f}")
        if "avg_recall" in aggregates:
            summary_table.add_row("Avg Recall", f"{aggregates['avg_recall']:.3f}")
        if "avg_f1_score" in aggregates:
            summary_table.add_row("Avg F1 Score", f"{aggregates['avg_f1_score']:.3f}")

        console.print(summary_table)

        # Individual query results
        if results.get("individual_results"):
            query_table = Table(title="Individual Query Results")
            query_table.add_column("Query", style="cyan", max_width=30)
            query_table.add_column("Results", justify="right")
            query_table.add_column("Time (s)", justify="right")
            query_table.add_column("Precision", justify="right")
            query_table.add_column("Recall", justify="right")

            for result in results["individual_results"][:10]:  # Show first 10
                query_table.add_row(
                    result["query"][:30] + "..."
                    if len(result["query"]) > 30
                    else result["query"],
                    str(result["total_results"]),
                    f"{result['search_time_seconds']:.2f}",
                    f"{result['precision']:.3f}" if result["precision"] > 0 else "N/A",
                    f"{result['recall']:.3f}" if result["recall"] > 0 else "N/A",
                )

            console.print(query_table)

    def _display_comparison_results(self, results: Dict[str, Any]) -> None:
        """Display comparative evaluation results."""
        console.print(Panel("[bold]Comparative Evaluation Results[/bold]"))

        # Backend performance table
        backend_table = Table(title="Backend Performance Summary")
        backend_table.add_column("Backend", style="cyan")
        backend_table.add_column("Avg Results", justify="right")
        backend_table.add_column("Avg Time (s)", justify="right")
        backend_table.add_column("Error Rate", justify="right")
        backend_table.add_column("Avg Precision", justify="right")

        for backend_name, backend_result in results["backend_results"].items():
            aggregates = backend_result["aggregate_metrics"]
            backend_table.add_row(
                backend_name,
                f"{aggregates.get('avg_total_results', 0):.1f}",
                f"{aggregates.get('avg_search_time', 0):.2f}",
                f"{aggregates.get('error_rate', 0) * 100:.1f}%",
                f"{aggregates.get('avg_precision', 0):.3f}"
                if aggregates.get("avg_precision", 0) > 0
                else "N/A",
            )

        console.print(backend_table)

        # Comparison table
        if results.get("comparisons"):
            comparison_table = Table(title="Backend Comparisons")
            comparison_table.add_column("Comparison", style="cyan")
            comparison_table.add_column("Avg Overlap", justify="right")
            comparison_table.add_column("Jaccard Similarity", justify="right")

            for comp in results["comparisons"]:
                comparison_table.add_row(
                    f"{comp['backend_a']} vs {comp['backend_b']}",
                    f"{comp['average_overlap']:.1f}",
                    f"{comp['average_jaccard_similarity']:.3f}",
                )

            console.print(comparison_table)


# Default query sets for evaluation
DEFAULT_BIOMEDICAL_QUERIES = [
    "BRCA1 mutations breast cancer",
    "p53 tumor suppressor pathways",
    "diabetes insulin resistance",
    "COVID-19 immune response",
    "Alzheimer disease amyloid beta",
    "cancer immunotherapy PD-1",
    "CRISPR gene editing",
    "heart disease genetic markers",
    "depression serotonin receptors",
    "arthritis inflammatory cytokines",
]

DEFAULT_GENERAL_QUERIES = [
    "machine learning algorithms",
    "climate change impacts",
    "renewable energy systems",
    "artificial intelligence ethics",
    "quantum computing applications",
    "sustainable agriculture methods",
    "urban planning strategies",
    "educational technology tools",
    "cybersecurity best practices",
    "social media influence behavior",
]
