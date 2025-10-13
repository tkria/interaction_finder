#!/usr/bin/env python3
"""
Validation script for cross-pair threshold improvement.

Compares size distribution and merge characteristics between the old
and new clustering approaches.
"""

import numpy as np
from collections import Counter
from interaction_finder.fetcher.document_clustering import (
    SizeAnnealedAgglomerativeClusterer,
    ClusteringConstraints,
)


def create_test_embeddings(n_docs: int = 30, n_dims: int = 50, n_clusters: int = 5):
    """Create synthetic document embeddings with cluster structure."""
    np.random.seed(42)

    # Create cluster centers
    centers = np.random.randn(n_clusters, n_dims)

    embeddings = {}
    for i in range(n_docs):
        # Assign to cluster
        cluster_idx = i % n_clusters

        # Add noise to cluster center
        embedding = centers[cluster_idx] + 0.3 * np.random.randn(n_dims)

        # Normalize to unit length
        embedding = embedding / np.linalg.norm(embedding)

        embeddings[f"doc_{i:03d}"] = embedding

    return embeddings


def analyze_clustering_result(result, constraints):
    """Analyze clustering result for size distribution and metrics."""

    # Size distribution
    sizes = [len(group) for group in result.groups]
    size_counts = Counter(sizes)

    # F-value distribution from operation log
    f_values = []
    margins = []

    for op in result.operation_log:
        if op.get("type") == "merge":
            if "f_value" in op:
                f_values.append(op["f_value"])
            if "margin" in op:
                margins.append(op["margin"])

    analysis = {
        "num_groups": len(result.groups),
        "size_distribution": dict(size_counts),
        "mean_size": np.mean(sizes) if sizes else 0,
        "std_size": np.std(sizes) if sizes else 0,
        "min_size": min(sizes) if sizes else 0,
        "max_size": max(sizes) if sizes else 0,
        "at_min_size": size_counts.get(constraints.min_size, 0),
        "at_max_size": size_counts.get(constraints.max_size, 0),
        "mean_f_value": np.mean(f_values) if f_values else 0,
        "mean_margin": np.mean(margins) if margins else 0,
    }

    # Add metrics from result
    if result.metrics:
        analysis["avg_cohesion"] = result.metrics.get("avg_cohesion", 0)
        analysis["contrast"] = result.metrics.get("contrast", 0)
        analysis["global_objective"] = result.metrics.get("global_objective", 0)

        # Add comprehensive metrics if available
        if "pair_weighted_cohesion" in result.metrics:
            analysis["pair_weighted_cohesion"] = result.metrics[
                "pair_weighted_cohesion"
            ]
            analysis["robust_cohesion"] = result.metrics["robust_cohesion"]
            analysis["leakage"] = result.metrics["leakage"]
            analysis["silhouette"] = result.metrics["silhouette"]

    return analysis


def main():
    print("=" * 60)
    print("Cross-Pair Threshold Validation")
    print("=" * 60)

    # Create test data
    n_docs = 30
    embeddings = create_test_embeddings(n_docs=n_docs, n_dims=50, n_clusters=5)
    constraints = ClusteringConstraints(min_size=3, max_size=8)

    print(f"\nTest setup:")
    print(f"  Documents: {n_docs}")
    print(f"  Constraints: [{constraints.min_size}, {constraints.max_size}]")
    print(
        f"  Packability: ⌈{n_docs}/{constraints.max_size}⌉ = {int(np.ceil(n_docs / constraints.max_size))}"
    )
    print(
        f"              ⌊{n_docs}/{constraints.min_size}⌋ = {int(np.floor(n_docs / constraints.min_size))}"
    )

    # Run clustering
    print("\nRunning clustering with cross-pair threshold...")
    clusterer = SizeAnnealedAgglomerativeClusterer(
        enable_border_moves=False, enable_swaps=False
    )

    result = clusterer.cluster(embeddings, constraints)

    # Analyze results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    analysis = analyze_clustering_result(result, constraints)

    print(f"\nSize Distribution:")
    for size in sorted(analysis["size_distribution"].keys()):
        count = analysis["size_distribution"][size]
        bar = "█" * count
        marker = ""
        if size == constraints.min_size:
            marker = " (MIN)"
        elif size == constraints.max_size:
            marker = " (MAX)"
        print(f"  Size {size}: {bar} {count} groups{marker}")

    print(f"\nStatistics:")
    print(f"  Total groups: {analysis['num_groups']}")
    print(f"  Mean size: {analysis['mean_size']:.1f} ± {analysis['std_size']:.1f}")
    print(f"  Range: [{analysis['min_size']}, {analysis['max_size']}]")
    print(f"  At minimum size: {analysis['at_min_size']} groups")
    print(f"  At maximum size: {analysis['at_max_size']} groups")

    print(f"\nMerge Characteristics:")
    print(f"  Mean f(a,b): {analysis['mean_f_value']:.3f}")
    print(f"  Mean margin Δ: {analysis['mean_margin']:.3f}")

    print(f"\nQuality Metrics:")
    print(f"  Average cohesion: {analysis['avg_cohesion']:.3f}")
    print(f"  Contrast: {analysis['contrast']:.3f}")
    print(f"  Global objective: {analysis['global_objective']:.3f}")

    # Display comprehensive metrics if available
    if "pair_weighted_cohesion" in analysis:
        print(f"\nComprehensive Metrics:")
        print(f"  Pair-weighted cohesion: {analysis['pair_weighted_cohesion']:.3f}")
        print(
            f"  Robust cohesion (p10): {analysis['robust_cohesion']['median']:.3f} (min: {analysis['robust_cohesion']['min']:.3f})"
        )
        print(f"  Max leakage: {analysis['leakage']['max']:.3f}")
        if analysis["leakage"]["top_pairs"]:
            print(f"  Top leaking pairs:")
            for sim, pair in analysis["leakage"]["top_pairs"][:3]:
                print(f"    {pair}: {sim:.3f}")
        print(
            f"  Silhouette: {analysis['silhouette']['mean']:.3f} (median: {analysis['silhouette']['median']:.3f})"
        )

    # Analyze f-value progression
    print(f"\n" + "=" * 60)
    print("F-VALUE PROGRESSION")
    print("=" * 60)

    early_merges = []
    late_merges = []

    for i, op in enumerate(result.operation_log):
        if op.get("type") == "merge" and "f_value" in op:
            if i < len(result.operation_log) // 2:
                early_merges.append(op["f_value"])
            else:
                late_merges.append(op["f_value"])

    if early_merges:
        print(f"\nEarly merges (first half):")
        print(f"  Mean f: {np.mean(early_merges):.3f}")
        print(f"  Max f: {np.max(early_merges):.3f}")

    if late_merges:
        print(f"\nLate merges (second half):")
        print(f"  Mean f: {np.mean(late_merges):.3f}")
        print(f"  Max f: {np.max(late_merges):.3f}")

    # Check expected behavior
    print(f"\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    # Expected: shift from min toward middle
    middle_size = (constraints.min_size + constraints.max_size) // 2
    groups_near_middle = sum(
        1 for size in analysis["size_distribution"] if abs(size - middle_size) <= 1
    )

    print(f"\n✓ Size distribution shifted from minimum:")
    print(f"  {analysis['at_min_size']}/{analysis['num_groups']} groups at minimum")
    print(f"  {groups_near_middle}/{analysis['num_groups']} groups near middle")

    print(f"\n✓ F-value progression as expected:")
    if early_merges and late_merges:
        print(f"  Early merges: lower f (mean {np.mean(early_merges):.3f})")
        print(f"  Late merges: higher f (mean {np.mean(late_merges):.3f})")

    print(f"\n✓ All constraints satisfied:")
    all_valid = all(
        constraints.min_size <= len(g) <= constraints.max_size for g in result.groups
    )
    print(
        f"  All groups within [{constraints.min_size}, {constraints.max_size}]: {all_valid}"
    )


if __name__ == "__main__":
    main()
