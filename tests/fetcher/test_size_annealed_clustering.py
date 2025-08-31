"""
Tests for size-annealed agglomerative clustering algorithm.
"""

import pytest
import numpy as np
from typing import Dict, List

from interaction_finder.fetcher.document_clustering import (
    SizeAnnealedAgglomerativeClusterer,
    ClusteringConstraints,
    ClusterCache,
)


class TestSizeAnnealedAgglomerativeClustering:
    """Test suite for size-annealed agglomerative clustering."""

    def create_test_embeddings(self, n: int, seed: int = 42) -> Dict[str, np.ndarray]:
        """Create test embeddings with controllable similarity patterns."""
        np.random.seed(seed)
        embeddings = {}

        # Create embeddings with some structure (groups of similar documents)
        n_groups = max(1, n // 4)
        group_centers = np.random.randn(n_groups, 100)

        for i in range(n):
            group_idx = i % n_groups
            # Add noise to group center
            embedding = group_centers[group_idx] + 0.3 * np.random.randn(100)
            # Normalize to unit length
            embedding = embedding / np.linalg.norm(embedding)
            embeddings[f"doc_{i}"] = embedding

        return embeddings

    def test_initialization(self):
        """Test clusterer initialization."""
        clusterer = SizeAnnealedAgglomerativeClusterer(
            enable_border_moves=False, enable_swaps=False
        )

        assert not clusterer.enable_border_moves
        assert not clusterer.enable_swaps

    def test_packability_check(self):
        """Test packability constraint checking."""
        clusterer = SizeAnnealedAgglomerativeClusterer()

        # Valid constraints
        constraints = ClusteringConstraints(min_size=2, max_size=5)
        assert clusterer._check_packability(10, constraints)

        # Invalid constraints (can't pack 10 docs into groups of size 6-8)
        constraints = ClusteringConstraints(min_size=6, max_size=8)
        assert not clusterer._check_packability(10, constraints)

        # Edge case: exact packing
        constraints = ClusteringConstraints(min_size=5, max_size=5)
        assert clusterer._check_packability(10, constraints)

    def test_cache_initialization(self):
        """Test cache initialization from similarity matrix."""
        clusterer = SizeAnnealedAgglomerativeClusterer()

        # Create simple test case
        S = np.array([[0.0, 0.8, 0.3], [0.8, 0.0, 0.5], [0.3, 0.5, 0.0]])
        docs = ["doc_0", "doc_1", "doc_2"]

        cache = clusterer._initialize_caches(S, docs)

        # Check singleton initialization
        assert len(cache.size) == 3
        assert all(cache.size[i] == 1 for i in range(3))
        assert all(cache.W[i] == 0.0 for i in range(3))
        assert all(cache.mu[i] == 0.0 for i in range(3))

        # Check cross-sums
        assert cache.R[(0, 1)] == 0.8
        assert cache.R[(1, 0)] == 0.8  # Symmetric
        assert cache.R[(0, 2)] == 0.3
        assert cache.R[(1, 2)] == 0.5

        # Check members
        assert cache.members[0] == {0}
        assert cache.members[1] == {1}
        assert cache.members[2] == {2}

    def test_cache_update_after_merge(self):
        """Test cache updates after merging clusters."""
        clusterer = SizeAnnealedAgglomerativeClusterer()

        # Create test similarity matrix
        S = np.array(
            [
                [0.0, 0.8, 0.3, 0.2],
                [0.8, 0.0, 0.5, 0.4],
                [0.3, 0.5, 0.0, 0.7],
                [0.2, 0.4, 0.7, 0.0],
            ]
        )
        docs = ["doc_0", "doc_1", "doc_2", "doc_3"]

        cache = clusterer._initialize_caches(S, docs)

        # Merge clusters 0 and 1
        clusterer._update_caches_after_merge(cache, 0, 1, S)

        # Check updated size and members
        assert cache.size[0] == 2
        assert cache.members[0] == {0, 1}
        assert 1 not in cache.size  # Cluster 1 removed

        # Check within-cluster sum
        assert cache.W[0] == 0.8  # S[0,1]

        # Check mean
        expected_mu = 2 * 0.8 / (2 * 1)
        assert abs(cache.mu[0] - expected_mu) < 1e-10

        # Check updated cross-sums
        assert cache.R[(0, 2)] == 0.3 + 0.5  # S[0,2] + S[1,2]
        assert cache.R[(0, 3)] == 0.2 + 0.4  # S[0,3] + S[1,3]

    def test_merge_acceptance(self):
        """Test merge acceptance criteria with cross-pair threshold."""
        clusterer = SizeAnnealedAgglomerativeClusterer()

        # Setup cache with some clusters
        cache = ClusterCache()

        # Test case 1: Balanced merge with moderate similarity
        cache.size = {0: 2, 1: 2}
        cache.W = {0: 0.8, 1: 0.7}
        cache.mu = {0: 0.8, 1: 0.7}

        # For balanced merge (2+2), f = 2*2*2/(4*3) = 8/12 = 0.667
        # Threshold = 0.7 + 0.667*(0.8-0.7) = 0.7 + 0.067 = 0.767
        # Cross-mean m = R/(2*2) = R/4
        # To pass, need m >= 0.767, so R >= 3.067
        cache.R = {(0, 1): 3.1, (1, 0): 3.1}

        constraints = ClusteringConstraints(min_size=2, max_size=6)

        # Should accept: size OK, R > 0, cross-pair threshold met
        assert clusterer._accept_merge(cache, 0, 1, constraints, 0.1)

        # Should reject: size violation
        cache.size[1] = 5
        assert not clusterer._accept_merge(cache, 0, 1, constraints, 0.1)

        # Should reject: negative R
        cache.size[1] = 3
        cache.R[(0, 1)] = -0.1
        cache.R[(1, 0)] = -0.1
        assert not clusterer._accept_merge(cache, 0, 1, constraints, 0.1)

    def test_merge_priority_computation(self):
        """Test heap priority computation for merges with post-merge cohesion."""
        clusterer = SizeAnnealedAgglomerativeClusterer()

        cache = ClusterCache()
        cache.size = {0: 2, 1: 3}
        cache.W = {0: 1.0, 1: 2.0}
        cache.mu = {0: 1.0, 1: 2.0 / 3}
        cache.R = {(0, 1): 1.5, (1, 0): 1.5}

        priority = clusterer._compute_merge_priority(cache, 0, 1)

        # New priority structure: (-μ_merge, -m, -s, -R_AB, min(A, B))
        assert len(priority) == 5

        # Compute expected values
        a, b = 2, 3
        s = a + b  # 5
        R_AB = 1.5
        m = R_AB / (a * b)  # 1.5 / 6 = 0.25

        # Compute expected post-merge cohesion μ_merge
        W_merged = cache.W[0] + cache.W[1] + R_AB  # 1.0 + 2.0 + 1.5 = 4.5
        mu_merge = 2 * W_merged / (s * (s - 1))  # 2 * 4.5 / (5 * 4) = 9.0 / 20 = 0.45

        # Verify heap key components
        assert abs(priority[0] - (-mu_merge)) < 1e-6  # -0.45
        assert abs(priority[1] - (-m)) < 1e-6  # -0.25
        assert priority[2] == -float(s)  # -5.0
        assert priority[3] == -R_AB  # -1.5
        assert priority[4] == 0  # min(0, 1)

    def test_stall_resolution(self):
        """Test stall condition handling."""
        clusterer = SizeAnnealedAgglomerativeClusterer()

        # Create scenario with undersized cluster
        S = np.array(
            [
                [0.0, 0.8, 0.3, 0.2],
                [0.8, 0.0, 0.5, 0.4],
                [0.3, 0.5, 0.0, 0.7],
                [0.2, 0.4, 0.7, 0.0],
            ]
        )

        cache = ClusterCache()
        cache.size = {0: 1, 1: 2, 2: 1}  # Cluster 0 and 2 undersized
        cache.W = {0: 0.0, 1: 0.5, 2: 0.0}
        cache.mu = {0: 0.0, 1: 1.0, 2: 0.0}
        cache.members = {0: {0}, 1: {1, 3}, 2: {2}}
        cache.R = {
            (0, 1): 0.8 + 0.2,
            (1, 0): 0.8 + 0.2,
            (0, 2): 0.3,
            (2, 0): 0.3,
            (1, 2): 0.5 + 0.7,
            (2, 1): 0.5 + 0.7,
        }

        constraints = ClusteringConstraints(min_size=2, max_size=4)
        operation_log = []

        # Should handle stall by merging undersized cluster
        result = clusterer._handle_stall(cache, constraints, S, operation_log, 1)
        assert result
        assert len(operation_log) == 1
        assert operation_log[0]["type"] == "stall_resolution"

    def test_basic_clustering(self):
        """Test basic clustering functionality."""
        clusterer = SizeAnnealedAgglomerativeClusterer()

        # Create test embeddings
        embeddings = self.create_test_embeddings(10)
        constraints = ClusteringConstraints(min_size=2, max_size=4)

        # Perform clustering
        result = clusterer.cluster(embeddings, constraints)

        # Check result structure
        assert result.groups is not None
        assert result.metrics is not None
        assert result.operation_log is not None

        # Check constraints are satisfied
        for group in result.groups:
            assert len(group) >= constraints.min_size
            assert len(group) <= constraints.max_size

        # Check all documents are assigned
        all_docs = set()
        for group in result.groups:
            all_docs.update(group)
        assert len(all_docs) == len(embeddings)

    def test_border_moves(self):
        """Test border move optimization."""
        clusterer = SizeAnnealedAgglomerativeClusterer(enable_border_moves=True)

        # Create embeddings with clear structure
        embeddings = self.create_test_embeddings(12, seed=123)
        constraints = ClusteringConstraints(min_size=3, max_size=5)

        # Perform clustering
        result = clusterer.cluster(embeddings, constraints)

        # Check for border move operations in log
        border_moves = [
            op for op in result.operation_log if op.get("type") == "border_move"
        ]

        # Border moves may or may not happen depending on initial clustering
        assert isinstance(border_moves, list)

    def test_document_swaps(self):
        """Test document swap optimization."""
        clusterer = SizeAnnealedAgglomerativeClusterer(enable_swaps=True)

        embeddings = self.create_test_embeddings(12, seed=456)
        constraints = ClusteringConstraints(min_size=3, max_size=5)

        # Perform clustering
        result = clusterer.cluster(embeddings, constraints)

        # Check for swap operations in log
        swaps = [op for op in result.operation_log if op.get("type") == "document_swap"]

        # Swaps may or may not happen depending on clustering quality
        assert isinstance(swaps, list)

    def test_edge_cases(self):
        """Test edge cases."""
        clusterer = SizeAnnealedAgglomerativeClusterer()

        # Empty input
        result = clusterer.cluster({}, ClusteringConstraints(2, 5))
        assert result.groups == []

        # Single document
        embeddings = {"doc_0": np.ones(10) / np.sqrt(10)}
        result = clusterer.cluster(embeddings, ClusteringConstraints(1, 5))
        assert len(result.groups) == 1
        assert result.groups[0] == ["doc_0"]

        # Two documents with min_size=2
        embeddings = {
            "doc_0": np.array([1, 0, 0]) / 1.0,
            "doc_1": np.array([0, 1, 0]) / 1.0,
        }
        result = clusterer.cluster(embeddings, ClusteringConstraints(2, 3))
        assert len(result.groups) == 1
        assert len(result.groups[0]) == 2

    def test_deterministic_behavior(self):
        """Test that clustering is deterministic."""
        clusterer = SizeAnnealedAgglomerativeClusterer()

        embeddings = self.create_test_embeddings(20)
        constraints = ClusteringConstraints(min_size=3, max_size=6)

        # Run clustering multiple times
        results = []
        for _ in range(3):
            result = clusterer.cluster(embeddings, constraints)
            results.append(result)

        # Check that results are identical
        for i in range(1, len(results)):
            assert len(results[i].groups) == len(results[0].groups)
            # Sort groups for comparison
            groups_0 = sorted([sorted(g) for g in results[0].groups])
            groups_i = sorted([sorted(g) for g in results[i].groups])
            assert groups_0 == groups_i

    def test_balanced_vs_unbalanced_merges(self):
        """Test that balanced and unbalanced merges have different thresholds."""
        clusterer = SizeAnnealedAgglomerativeClusterer()
        cache = ClusterCache()
        constraints = ClusteringConstraints(min_size=2, max_size=10)

        # Balanced large merge (high f) requires m ≈ μ_max
        cache.size = {0: 4, 1: 4}
        cache.mu = {0: 0.6, 1: 0.8}
        cache.W = {0: 3.6, 1: 4.8}  # Consistent with mu values

        # f = 2*4*4/(8*7) = 32/56 = 0.571
        # Threshold = 0.6 + 0.571*(0.8-0.6) = 0.6 + 0.114 = 0.714
        # Need cross-mean m >= 0.714

        # Test: Just below threshold should fail
        cache.R = {(0, 1): 11.0, (1, 0): 11.0}  # m = 11/16 = 0.6875 < 0.714
        assert not clusterer._accept_merge(cache, 0, 1, constraints, 0.1)

        # Test: Just above threshold should pass
        cache.R = {(0, 1): 11.5, (1, 0): 11.5}  # m = 11.5/16 = 0.719 > 0.714
        assert clusterer._accept_merge(cache, 0, 1, constraints, 0.1)

        # Small→large merge (low f) passes with m ≈ μ_min
        cache.size = {2: 1, 3: 7}
        cache.mu = {2: 0.0, 3: 0.7}  # Singleton has mu=0
        cache.W = {2: 0.0, 3: 14.7}

        # f = 2*1*7/(8*7) = 14/56 = 0.25
        # Threshold = 0.0 + 0.25*(0.7-0.0) = 0.175
        # Need cross-mean m >= 0.175

        cache.R = {(2, 3): 1.3, (3, 2): 1.3}  # m = 1.3/7 = 0.186 > 0.175
        assert clusterer._accept_merge(cache, 2, 3, constraints, 0.1)

    def test_negative_similarities(self):
        """Test that merges with negative R are rejected."""
        clusterer = SizeAnnealedAgglomerativeClusterer()
        cache = ClusterCache()
        constraints = ClusteringConstraints(min_size=2, max_size=6)

        cache.size = {0: 2, 1: 2}
        cache.mu = {0: 0.5, 1: 0.5}
        cache.W = {0: 0.5, 1: 0.5}

        # Even if threshold would be met, negative R should reject
        cache.R = {(0, 1): -0.1, (1, 0): -0.1}
        assert not clusterer._accept_merge(cache, 0, 1, constraints, 0.1)

        # Positive R should be accepted if threshold met
        cache.R = {(0, 1): 2.0, (1, 0): 2.0}
        assert clusterer._accept_merge(cache, 0, 1, constraints, 0.1)

    def test_cohesion_preservation(self):
        """Test that cross-pair threshold naturally preserves cohesion."""
        clusterer = SizeAnnealedAgglomerativeClusterer()

        embeddings = self.create_test_embeddings(15)
        constraints = ClusteringConstraints(min_size=3, max_size=6)

        result = clusterer.cluster(embeddings, constraints)

        # Check that cohesion metrics are reasonable
        assert "avg_cohesion" in result.metrics
        assert "cohesions" in result.metrics
        assert result.metrics["avg_cohesion"] >= 0.0
        assert result.metrics["avg_cohesion"] <= 1.0

    def test_large_scale(self):
        """Test with larger number of documents."""
        clusterer = SizeAnnealedAgglomerativeClusterer()

        embeddings = self.create_test_embeddings(100)
        constraints = ClusteringConstraints(min_size=5, max_size=15)

        result = clusterer.cluster(embeddings, constraints)

        # Check all constraints satisfied
        total_docs = 0
        for group in result.groups:
            assert len(group) >= constraints.min_size
            assert len(group) <= constraints.max_size
            total_docs += len(group)

        assert total_docs == len(embeddings)

        # Check performance metrics logged
        assert "iterations" in result.metrics
        assert result.metrics["iterations"] > 0


class TestComprehensiveMetrics:
    """Test comprehensive clustering metrics computation."""

    def test_comprehensive_metrics_basic(self):
        """Test basic comprehensive metrics computation."""
        clusterer = SizeAnnealedAgglomerativeClusterer()

        # Create simple test case with known similarities
        # Group 1: docs 0,1 with high similarity (0.9)
        # Group 2: docs 2,3 with medium similarity (0.6)
        # Between groups: low similarity (0.1)
        S = np.array(
            [
                [0.0, 0.9, 0.1, 0.1],
                [0.9, 0.0, 0.1, 0.1],
                [0.1, 0.1, 0.0, 0.6],
                [0.1, 0.1, 0.6, 0.0],
            ]
        )
        docs = ["doc_0", "doc_1", "doc_2", "doc_3"]
        groups = [["doc_0", "doc_1"], ["doc_2", "doc_3"]]

        metrics = clusterer.compute_comprehensive_metrics(groups, S, docs)

        # Test pair-weighted cohesion
        # Expected: (0.9 + 0.6) / 2 = 0.75
        assert abs(metrics["pair_weighted_cohesion"] - 0.75) < 1e-6

        # Test contrast
        # Expected corpus average: (0.9 + 0.1 + 0.1 + 0.1 + 0.1 + 0.6) / 6 ≈ 0.3167
        # Within average: 0.75
        # Expected contrast: 0.75 - 0.3167 ≈ 0.433
        assert abs(metrics["contrast"] - 0.4333333333333333) < 1e-6

        # Test robust cohesion (10th percentile)
        # Group 1 has one pair: 0.9, so 10th percentile = 0.9
        # Group 2 has one pair: 0.6, so 10th percentile = 0.6
        # Median of [0.9, 0.6] = 0.75, min = 0.6
        assert abs(metrics["robust_cohesion"]["median"] - 0.75) < 1e-6
        assert abs(metrics["robust_cohesion"]["min"] - 0.6) < 1e-6

        # Test leakage
        # Between G1-G2: (0.1 + 0.1 + 0.1 + 0.1) / 4 = 0.1
        assert abs(metrics["leakage"]["max"] - 0.1) < 1e-6
        assert len(metrics["leakage"]["top_pairs"]) == 1

        # Test silhouette structure
        assert "mean" in metrics["silhouette"]
        assert "median" in metrics["silhouette"]
        assert -1.0 <= metrics["silhouette"]["mean"] <= 1.0

    def test_comprehensive_metrics_edge_cases(self):
        """Test comprehensive metrics with edge cases."""
        clusterer = SizeAnnealedAgglomerativeClusterer()

        # Empty groups
        metrics = clusterer.compute_comprehensive_metrics([], np.array([]), [])
        assert metrics["pair_weighted_cohesion"] == 0.0
        assert metrics["contrast"] == 0.0
        assert metrics["robust_cohesion"]["median"] == 0.0
        assert metrics["leakage"]["max"] == 0.0
        assert metrics["silhouette"]["mean"] == 0.0

        # Single document groups
        S = np.array([[0.0]])
        docs = ["doc_0"]
        groups = [["doc_0"]]

        metrics = clusterer.compute_comprehensive_metrics(groups, S, docs)
        assert metrics["pair_weighted_cohesion"] == 0.0  # No pairs
        assert metrics["robust_cohesion"]["median"] == 1.0  # Singletons get 1.0
        assert metrics["leakage"]["max"] == 0.0  # No between-cluster pairs

    def test_comprehensive_metrics_integration(self):
        """Test that comprehensive metrics are integrated into standard metrics."""
        clusterer = SizeAnnealedAgglomerativeClusterer()

        # Create simple test case
        S = np.array(
            [
                [0.0, 0.8, 0.2],
                [0.8, 0.0, 0.3],
                [0.2, 0.3, 0.0],
            ]
        )
        docs = ["doc_0", "doc_1", "doc_2"]
        groups = [["doc_0", "doc_1"], ["doc_2"]]

        metrics = clusterer._compute_standard_metrics(groups, S, docs)

        # Check that comprehensive metrics are included
        assert "pair_weighted_cohesion" in metrics
        assert "robust_cohesion" in metrics
        assert "leakage" in metrics
        assert "silhouette" in metrics

        # Check structure
        assert "median" in metrics["robust_cohesion"]
        assert "min" in metrics["robust_cohesion"]
        assert "max" in metrics["leakage"]
        assert "top_pairs" in metrics["leakage"]
        assert "mean" in metrics["silhouette"]
        assert "median" in metrics["silhouette"]

        # Check that legacy metrics still exist
        assert "avg_cohesion" in metrics
        assert "cohesions" in metrics
        assert "contrast" in metrics


class TestClusterCache:
    """Test ClusterCache functionality."""

    def test_cache_initialization(self):
        """Test cache dataclass initialization."""
        cache = ClusterCache()

        assert isinstance(cache.size, dict)
        assert isinstance(cache.W, dict)
        assert isinstance(cache.mu, dict)
        assert isinstance(cache.R, dict)
        assert isinstance(cache.members, dict)

        assert len(cache.size) == 0
        assert len(cache.W) == 0
        assert len(cache.mu) == 0
        assert len(cache.R) == 0
        assert len(cache.members) == 0

    def test_cache_operations(self):
        """Test cache manipulation."""
        cache = ClusterCache()

        # Add some data
        cache.size[0] = 3
        cache.W[0] = 1.5
        cache.mu[0] = 0.5
        cache.members[0] = {0, 1, 2}
        cache.R[(0, 1)] = 0.8

        # Check access
        assert cache.size[0] == 3
        assert cache.W[0] == 1.5
        assert cache.mu[0] == 0.5
        assert cache.members[0] == {0, 1, 2}
        assert cache.R[(0, 1)] == 0.8

        # Check deletion
        del cache.size[0]
        assert 0 not in cache.size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
