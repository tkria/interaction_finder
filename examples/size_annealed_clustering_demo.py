#!/usr/bin/env python3
"""
Demo script for size-annealed agglomerative clustering.

This script demonstrates the parameter-free size-annealed clustering algorithm
on synthetic document embeddings.
"""

import numpy as np
from interaction_finder.fetcher.document_clustering import (
    SizeAnnealedAgglomerativeClusterer,
    ClusteringConstraints,
)


def create_synthetic_embeddings(n_docs: int = 20, n_dims: int = 50, n_clusters: int = 4):
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


def main():
    print("=" * 60)
    print("Size-annealed Agglomerative Clustering Demo")
    print("=" * 60)
    
    # Create synthetic embeddings
    n_docs = 30
    embeddings = create_synthetic_embeddings(n_docs=n_docs, n_dims=50, n_clusters=5)
    print(f"\nCreated {len(embeddings)} document embeddings")
    
    # Define size constraints
    constraints = ClusteringConstraints(min_size=4, max_size=8)
    print(f"Constraints: min_size={constraints.min_size}, max_size={constraints.max_size}")
    
    # Check packability
    min_groups = int(np.ceil(n_docs / constraints.max_size))
    max_groups = int(np.floor(n_docs / constraints.min_size))
    print(f"Packability: need {min_groups}-{max_groups} groups")
    
    # Initialize clusterer
    clusterer = SizeAnnealedAgglomerativeClusterer(
        cohesion_epsilon=0.0,  # Strict cohesion preservation
        enable_border_moves=True,  # Allow optimization
        enable_swaps=False  # Skip swaps for speed
    )
    
    print("\nPerforming clustering...")
    result = clusterer.cluster(embeddings, constraints)
    
    # Display results
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    
    print(f"\nFound {len(result.groups)} groups:")
    for i, group in enumerate(result.groups):
        print(f"  Group {i+1}: {len(group)} documents")
        print(f"    Members: {', '.join(group[:5])}", end="")
        if len(group) > 5:
            print(f"... ({len(group)-5} more)")
        else:
            print()
    
    # Display metrics
    print(f"\nMetrics:")
    print(f"  Algorithm: {result.metrics.get('algorithm')}")
    print(f"  Iterations: {result.metrics.get('iterations')}")
    print(f"  Average cohesion: {result.metrics.get('avg_cohesion', 0):.3f}")
    print(f"  Cohesion std: {result.metrics.get('cohesion_std', 0):.3f}")
    print(f"  Global objective: {result.metrics.get('global_objective', 0):.3f}")
    
    # Check constraint satisfaction
    print(f"\nConstraint verification:")
    all_satisfied = True
    for i, group in enumerate(result.groups):
        size = len(group)
        satisfied = constraints.min_size <= size <= constraints.max_size
        status = "✓" if satisfied else "✗"
        print(f"  Group {i+1}: size={size} {status}")
        all_satisfied = all_satisfied and satisfied
    
    if all_satisfied:
        print("\n✓ All constraints satisfied!")
    else:
        print("\n✗ Some constraints violated")
    
    # Show operation statistics
    print(f"\nOperation log summary:")
    op_types = {}
    for op in result.operation_log:
        op_type = op.get("type", "unknown")
        op_types[op_type] = op_types.get(op_type, 0) + 1
    
    for op_type, count in sorted(op_types.items()):
        print(f"  {op_type}: {count}")
    
    # Show stall resolutions if any
    stall_ops = [op for op in result.operation_log if op.get("type") == "stall_resolution"]
    if stall_ops:
        print(f"\nStall resolutions: {len(stall_ops)}")
        for op in stall_ops[:3]:  # Show first 3
            print(f"  - Merged undersized cluster {op.get('undersized_cluster')} "
                  f"(size {op.get('undersized_size')}) with cluster {op.get('partner')}")


if __name__ == "__main__":
    main()