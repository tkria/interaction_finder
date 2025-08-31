"""
Document clustering algorithms with size constraints.

This module provides multiple clustering strategies (agglomerative, spectral, hybrid)
that respect document group size constraints while optimizing cluster cohesion.
"""

from abc import ABC, abstractmethod
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Callable,
    Literal,
    TYPE_CHECKING,
    Any,
    Set,
)
import itertools
import heapq
import time
import concurrent.futures
from functools import partial

if TYPE_CHECKING:
    from .progress_display import StatusProtocol
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ClusteringConstraints:
    """Size constraints and parameters for clustering."""

    min_size: int
    max_size: int
    constraint_type: Literal["count", "words"] = "count"


@dataclass
class ClusteringResult:
    """Result of clustering with groups and comprehensive metrics."""

    groups: List[List[str]]
    metrics: Dict[str, any] = field(default_factory=dict)
    operation_log: List[Dict] = field(default_factory=list)


class DocumentClusterer(ABC):
    """Base class for all document clustering algorithms."""

    @abstractmethod
    def cluster(
        self,
        embeddings: Dict[str, np.ndarray],
        constraints: ClusteringConstraints,
        doc_metrics: Optional[Dict[str, int]] = None,
        status: Optional["StatusProtocol"] = None,
    ) -> ClusteringResult:
        """
        Cluster documents given embeddings and constraints.

        Args:
            embeddings: Dict mapping document ID to unit-norm embedding vector
            constraints: Size constraints for clustering
            doc_metrics: Optional dict mapping document ID to size metric
            status: Optional status object for progress reporting

        Returns:
            ClusteringResult with groups, metrics, and operation log
        """
        pass

    def _build_similarity_matrix(
        self, embeddings: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build document-document cosine similarity matrix.

        Following specification: S = V @ V.T with diagonal set to 0.

        Returns:
            Tuple of (similarity_matrix, document_list)
        """
        docs = list(embeddings.keys())
        if not docs:
            return np.array([]), []

        # Stack embeddings into matrix V (one row per document)
        V = np.stack([embeddings[doc] for doc in docs], axis=0)

        # Compute cosine similarity: S = V @ V.T
        S = V @ V.T

        # Set diagonal to 0 as specified
        np.fill_diagonal(S, 0.0)

        return S, docs

    def _compute_size(
        self, group: List[str], doc_metrics: Optional[Dict[str, int]] = None
    ) -> int:
        """Compute group size based on metrics (word count or document count)."""
        return (
            sum(doc_metrics.get(doc, 1) for doc in group) if doc_metrics else len(group)
        )

    def _apply_merge(
        self, clusters: List[set], i: int, j: int, merged_cluster: set
    ) -> None:
        """Apply merge by removing clusters i,j and adding merged cluster in-place."""
        clusters[:] = [
            cluster for k, cluster in enumerate(clusters) if k not in (i, j)
        ] + [merged_cluster]

    def _is_valid_size(
        self,
        group: List[str],
        constraints: ClusteringConstraints,
        doc_metrics: Optional[Dict[str, int]] = None,
    ) -> bool:
        """Check if group size meets constraints."""
        size = self._compute_size(group, doc_metrics)
        return constraints.min_size <= size <= constraints.max_size

    def _compute_cluster_similarity(
        self,
        cluster_a: set,
        cluster_b: set,
        S: np.ndarray,
        docs: List[str],
        method: str = "average",
    ) -> float:
        """Compute similarity between two clusters using specified linkage method."""
        doc_to_idx = {doc: i for i, doc in enumerate(docs)}

        # Convert clusters to index arrays for vectorized access
        indices_a = np.array(
            [doc_to_idx[doc] for doc in cluster_a if doc in doc_to_idx]
        )
        indices_b = np.array(
            [doc_to_idx[doc] for doc in cluster_b if doc in doc_to_idx]
        )

        if len(indices_a) == 0 or len(indices_b) == 0:
            return 0.0

        # Vectorized similarity computation using advanced indexing
        similarities = S[np.ix_(indices_a, indices_b)]

        if method == "average":
            return np.mean(similarities)
        elif method == "complete":
            return np.min(similarities)  # Complete linkage uses minimum similarity
        elif method == "single":
            return np.max(similarities)  # Single linkage uses maximum similarity
        else:
            return np.mean(similarities)  # Default to average

    def _compute_cluster_similarity_vectorized(
        self,
        indices_a: np.ndarray,
        indices_b: np.ndarray,
        S: np.ndarray,
        method: str = "average",
    ) -> float:
        """Vectorized cluster similarity computation using precomputed indices."""
        if len(indices_a) == 0 or len(indices_b) == 0:
            return 0.0

        # Vectorized similarity computation using advanced indexing
        similarities = S[np.ix_(indices_a, indices_b)]

        if method == "average":
            return np.mean(similarities)
        elif method == "complete":
            return np.min(similarities)  # Complete linkage uses minimum similarity
        elif method == "single":
            return np.max(similarities)  # Single linkage uses maximum similarity
        else:
            return np.mean(similarities)  # Default to average

    def _precompute_cluster_indices(
        self, clusters: List[set], docs: List[str]
    ) -> List[np.ndarray]:
        """Pre-compute cluster indices as numpy arrays for vectorized operations."""
        doc_to_idx = {doc: i for i, doc in enumerate(docs)}
        return [
            np.array([doc_to_idx[doc] for doc in cluster if doc in doc_to_idx])
            for cluster in clusters
        ]

    def _evaluate_all_candidates_vectorized(
        self,
        clusters: List[set],
        cluster_indices: List[np.ndarray],
        S: np.ndarray,
        constraints: ClusteringConstraints,
        current_threshold: float,
        doc_metrics: Optional[Dict[str, int]] = None,
    ) -> Tuple[List[Dict], Optional[Tuple[int, int]], float]:
        """Vectorized evaluation of all cluster pairs for candidate merging."""
        n_clusters = len(clusters)
        all_candidates = []
        best_pair = None
        best_similarity = -np.inf

        # Pre-compute cluster sizes for vectorized size checking
        cluster_sizes = np.array([len(cluster) for cluster in clusters])

        # Vectorized size constraint check for all pairs
        size_matrix = cluster_sizes[:, None] + cluster_sizes[None, :]
        valid_size_mask = (size_matrix <= constraints.max_size) & np.triu(
            np.ones((n_clusters, n_clusters), dtype=bool), k=1
        )

        # Get valid pair indices
        i_indices, j_indices = np.where(valid_size_mask)

        # Vectorized similarity computation for valid pairs
        similarities = []
        for idx, (i, j) in enumerate(zip(i_indices, j_indices)):
            similarity = self._compute_cluster_similarity_vectorized(
                cluster_indices[i], cluster_indices[j], S, self.linkage
            )
            similarities.append(similarity)

            # Check if above threshold
            if similarity >= current_threshold:
                # Check actual size constraint with doc_metrics if needed
                merged_size = self._compute_size(
                    list(clusters[i] | clusters[j]), doc_metrics
                )
                if merged_size <= constraints.max_size:
                    helps_undersized = (
                        len(clusters[i]) < constraints.min_size
                        or len(clusters[j]) < constraints.min_size
                    )

                    all_candidates.append(
                        {
                            "clusters": (i, j),
                            "sizes": (len(clusters[i]), len(clusters[j])),
                            "merged_size": merged_size,
                            "similarity": similarity,
                            "helps_undersized": helps_undersized,
                            "threshold_met": True,
                        }
                    )

                    # Update best pair with priority for undersized clusters
                    if helps_undersized and similarity > best_similarity:
                        best_similarity = similarity
                        best_pair = (i, j)
                    elif not best_pair and similarity > best_similarity:
                        best_similarity = similarity
                        best_pair = (i, j)

        return all_candidates, best_pair, best_similarity

    def _evaluate_candidates_parallel_batch(
        self,
        cluster_pairs: List[Tuple[int, int]],
        cluster_indices: List[np.ndarray],
        S: np.ndarray,
        clusters: List[set],
        constraints: ClusteringConstraints,
        current_threshold: float,
        doc_metrics: Optional[Dict[str, int]] = None,
    ) -> List[Dict]:
        """Evaluate a batch of cluster pairs in parallel."""
        batch_candidates = []

        for i, j in cluster_pairs:
            # Vectorized similarity computation
            similarity = self._compute_cluster_similarity_vectorized(
                cluster_indices[i], cluster_indices[j], S, self.linkage
            )

            # Check if above threshold
            if similarity >= current_threshold:
                # Check actual size constraint with doc_metrics if needed
                merged_size = self._compute_size(
                    list(clusters[i] | clusters[j]), doc_metrics
                )
                if merged_size <= constraints.max_size:
                    helps_undersized = (
                        len(clusters[i]) < constraints.min_size
                        or len(clusters[j]) < constraints.min_size
                    )

                    batch_candidates.append(
                        {
                            "clusters": (i, j),
                            "sizes": (len(clusters[i]), len(clusters[j])),
                            "merged_size": merged_size,
                            "similarity": similarity,
                            "helps_undersized": helps_undersized,
                            "threshold_met": True,
                        }
                    )

        return batch_candidates

    def _evaluate_all_candidates_parallel(
        self,
        clusters: List[set],
        cluster_indices: List[np.ndarray],
        S: np.ndarray,
        constraints: ClusteringConstraints,
        current_threshold: float,
        doc_metrics: Optional[Dict[str, int]] = None,
        max_workers: int = None,
    ) -> Tuple[List[Dict], Optional[Tuple[int, int]], float]:
        """Parallel evaluation of all cluster pairs using ThreadPoolExecutor."""
        n_clusters = len(clusters)
        all_candidates = []
        best_pair = None
        best_similarity = -np.inf

        # Pre-compute cluster sizes for vectorized size checking
        cluster_sizes = np.array([len(cluster) for cluster in clusters])

        # Vectorized size constraint check for all pairs
        size_matrix = cluster_sizes[:, None] + cluster_sizes[None, :]
        valid_size_mask = (size_matrix <= constraints.max_size) & np.triu(
            np.ones((n_clusters, n_clusters), dtype=bool), k=1
        )

        # Get valid pair indices
        i_indices, j_indices = np.where(valid_size_mask)
        valid_pairs = list(zip(i_indices, j_indices))

        # Only use parallel processing if we have enough pairs to justify overhead
        if len(valid_pairs) < 20:
            # Fall back to sequential processing for small problems
            return self._evaluate_all_candidates_vectorized(
                clusters,
                cluster_indices,
                S,
                constraints,
                current_threshold,
                doc_metrics,
            )

        # Split pairs into batches for parallel processing
        batch_size = max(1, len(valid_pairs) // (max_workers or 4))
        batches = [
            valid_pairs[i : i + batch_size]
            for i in range(0, len(valid_pairs), batch_size)
        ]

        # Evaluate batches in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_func = partial(
                self._evaluate_candidates_parallel_batch,
                cluster_indices=cluster_indices,
                S=S,
                clusters=clusters,
                constraints=constraints,
                current_threshold=current_threshold,
                doc_metrics=doc_metrics,
            )

            # Submit all batches
            future_to_batch = {
                executor.submit(batch_func, batch): batch for batch in batches
            }

            # Collect results
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_candidates = future.result()
                all_candidates.extend(batch_candidates)

        # Find best pair from all candidates
        for candidate in all_candidates:
            similarity = candidate["similarity"]
            helps_undersized = candidate["helps_undersized"]

            if helps_undersized and similarity > best_similarity:
                best_similarity = similarity
                best_pair = candidate["clusters"]
            elif not best_pair and similarity > best_similarity:
                best_similarity = similarity
                best_pair = candidate["clusters"]

        return all_candidates, best_pair, best_similarity

    def _compute_similarity_stats(
        self, S: np.ndarray, docs: List[str]
    ) -> Dict[str, float]:
        """Compute statistics for pairwise similarities in matrix."""
        if len(docs) <= 1:
            return {}

        similarities = [S[i, j] for i, j in itertools.combinations(range(len(docs)), 2)]

        similarities = np.array(similarities)
        return {
            "min": float(np.min(similarities)),
            "max": float(np.max(similarities)),
            "median": float(np.median(similarities)),
            "q25": float(np.percentile(similarities, 25)),
            "q75": float(np.percentile(similarities, 75)),
            "std": float(np.std(similarities)),
        }

    def _log_base(self, log_type: str, **kwargs) -> Dict[str, Any]:
        """Create base logging dictionary with timestamp and type."""
        return {"type": log_type, "timestamp": time.time(), **kwargs}

    def _log_merge(
        self,
        i: int,
        j: int,
        merged_group: List[str],
        doc_metrics: Optional[Dict[str, int]] = None,
        operation_type: str = "merge",
        **extra,
    ) -> Dict[str, Any]:
        """Log cluster merge operation."""
        return self._log_base(
            operation_type,
            clusters=[i, j],
            documents=merged_group,
            size=self._compute_size(merged_group, doc_metrics),
            **extra,
        )

    def _log_iteration(
        self, iteration_type: str, iteration: int, **metrics
    ) -> Dict[str, Any]:
        """Log iteration progress."""
        return self._log_base(
            f"{iteration_type}_iteration", iteration=iteration, **metrics
        )

    def _log_split(
        self,
        source_idx: int,
        source_size: int,
        result_groups: List[List[str]],
        operation_type: str = "split_oversized",
        **extra,
    ) -> Dict[str, Any]:
        """Log group splitting operation."""
        result_dict = self._log_base(operation_type, **extra)
        if operation_type == "oversized_split":
            result_dict.update(
                {"original_size": source_size, "split_into": len(result_groups)}
            )
        else:
            result_dict.update(
                {
                    "source_group": source_idx,
                    "source_size": source_size,
                    "result_sizes": [len(g) for g in result_groups],
                }
            )
        return result_dict

    def _log_initialization(
        self,
        docs: List[str],
        total_similarity: float,
        sim_stats: Dict[str, float],
        adaptive_thresholds: List[float],
        constraints: ClusteringConstraints,
    ) -> Dict[str, Any]:
        """Log clustering initialization with similarity statistics."""
        return self._log_base(
            "initialization",
            total_docs=len(docs),
            total_similarity=float(total_similarity),
            avg_similarity=float(
                total_similarity / (len(docs) * (len(docs) - 1)) if len(docs) > 1 else 0
            ),
            similarity_stats=sim_stats,
            adaptive_thresholds=adaptive_thresholds if len(docs) > 1 else [],
            constraints={
                "min_size": constraints.min_size,
                "max_size": constraints.max_size,
            },
        )

    def _log_ejection(
        self,
        document: str,
        source_group: int,
        target_group: int,
        delta_F: float,
        forced: bool = False,
    ) -> Dict[str, Any]:
        """Log document ejection operation."""
        ejection_type = "ejection_forced" if forced else "ejection"
        log_dict = self._log_base(
            ejection_type,
            document=document,
            source_group=source_group,
            target_group=target_group,
            delta_F=delta_F,
        )
        if forced:
            log_dict["feasibility_override"] = True
        return log_dict

    def _log_document_move(
        self,
        document: str,
        source_group: int,
        target_group: int,
        delta_F: float,
        move_type: str = "border_move",
    ) -> Dict[str, Any]:
        """Log document movement between groups."""
        return self._log_base(
            move_type,
            document=document,
            source_group=source_group,
            target_group=target_group,
            delta_F=delta_F,
        )

    def _log_document_swap(
        self, doc_a: str, doc_b: str, group_a: int, group_b: int, delta_F: float
    ) -> Dict[str, Any]:
        """Log document swap between groups."""
        return self._log_base(
            "document_swap",
            documents=[doc_a, doc_b],
            groups=[group_a, group_b],
            delta_F=delta_F,
        )

    def _compute_group_score(
        self, group: List[str], S: np.ndarray, docs: List[str]
    ) -> float:
        """
        Compute W(G) - sum of within-group pairwise cosines.

        Args:
            group: List of document identifiers
            S: Similarity matrix
            docs: List mapping indices to document identifiers

        Returns:
            Sum of pairwise cosines within the group
        """
        if len(group) < 2:
            return 0.0

        doc_to_idx = {doc: i for i, doc in enumerate(docs)}
        group_indices = [doc_to_idx[doc] for doc in group if doc in doc_to_idx]

        if len(group_indices) < 2:
            return 0.0

        total_score = 0.0
        for i, idx_a in enumerate(group_indices):
            for idx_b in group_indices[i + 1 :]:
                total_score += S[idx_a, idx_b]

        return total_score

    def _compute_global_objective(
        self, groups: List[List[str]], S: np.ndarray, docs: List[str]
    ) -> float:
        """
        Compute global objective F = sum of W(G) for all groups.

        Args:
            groups: List of document groups
            S: Similarity matrix
            docs: Document list for index mapping

        Returns:
            Global objective value F
        """
        return sum(self._compute_group_score(group, S, docs) for group in groups)

    def compute_comprehensive_metrics(
        self, groups: List[List[str]], S: np.ndarray, docs: List[str]
    ) -> Dict[str, any]:
        """
        Compute comprehensive clustering metrics from similarity matrix and group assignments.

        Args:
            groups: List of document groups (cluster assignments)
            S: Similarity matrix (cosine similarities with diagonal = 0)
            docs: List mapping indices to document identifiers

        Returns:
            Dict with comprehensive metrics:
            - pair_weighted_cohesion: Global tightness weighted by pairs
            - contrast: Within-cluster vs corpus average similarity
            - robust_cohesion: 10th percentile cohesion (uniformity check)
            - leakage: Maximum between-cluster similarity with top pairs
            - silhouette: Mean separation using cosine distance
        """
        if not groups or S.size == 0:
            return {
                "pair_weighted_cohesion": 0.0,
                "contrast": 0.0,
                "robust_cohesion": {"median": 0.0, "min": 0.0},
                "leakage": {"max": 0.0, "top_pairs": []},
                "silhouette": {"mean": 0.0, "median": 0.0},
            }

        doc_to_idx = {doc: i for i, doc in enumerate(docs)}

        # 1. Pair-weighted cohesion (global)
        total_within_sum = 0.0
        total_pairs = 0

        for group in groups:
            if len(group) < 2:
                continue
            group_indices = [doc_to_idx[doc] for doc in group if doc in doc_to_idx]
            if len(group_indices) < 2:
                continue

            # Sum all within-group pairs
            for i in range(len(group_indices)):
                for j in range(i + 1, len(group_indices)):
                    total_within_sum += S[group_indices[i], group_indices[j]]

            total_pairs += len(group_indices) * (len(group_indices) - 1) // 2

        pair_weighted_cohesion = (
            total_within_sum / total_pairs if total_pairs > 0 else 0.0
        )

        # 2. Contrast (tightness vs background)
        corpus_sum = 0.0
        corpus_pairs = 0
        n = len(docs)

        for i in range(n):
            for j in range(i + 1, n):
                if i != j:  # Skip diagonal
                    corpus_sum += S[i, j]
                    corpus_pairs += 1

        corpus_avg = corpus_sum / corpus_pairs if corpus_pairs > 0 else 0.0
        contrast = pair_weighted_cohesion - corpus_avg

        # 3. Robust cohesion (10th percentile check)
        cluster_10th_percentiles = []

        for group in groups:
            if len(group) < 2:
                cluster_10th_percentiles.append(
                    1.0
                )  # Single docs have perfect cohesion
                continue

            group_indices = [doc_to_idx[doc] for doc in group if doc in doc_to_idx]
            if len(group_indices) < 2:
                continue

            # Collect all within-group similarities
            group_sims = []
            for i in range(len(group_indices)):
                for j in range(i + 1, len(group_indices)):
                    group_sims.append(S[group_indices[i], group_indices[j]])

            if group_sims:
                percentile_10 = np.percentile(group_sims, 10)
                cluster_10th_percentiles.append(percentile_10)

        robust_cohesion = {
            "median": float(np.median(cluster_10th_percentiles))
            if cluster_10th_percentiles
            else 0.0,
            "min": float(np.min(cluster_10th_percentiles))
            if cluster_10th_percentiles
            else 0.0,
        }

        # 4. Leakage (between-cluster similarity)
        between_cluster_means = []

        for i, group_a in enumerate(groups):
            for j, group_b in enumerate(groups):
                if i >= j:
                    continue

                indices_a = [doc_to_idx[doc] for doc in group_a if doc in doc_to_idx]
                indices_b = [doc_to_idx[doc] for doc in group_b if doc in doc_to_idx]

                if not indices_a or not indices_b:
                    continue

                # Compute mean similarity between clusters
                cross_sims = []
                for idx_a in indices_a:
                    for idx_b in indices_b:
                        cross_sims.append(S[idx_a, idx_b])

                if cross_sims:
                    mean_cross = np.mean(cross_sims)
                    between_cluster_means.append((mean_cross, f"G{i + 1}-G{j + 1}"))

        # Sort and get top leaking pairs
        between_cluster_means.sort(reverse=True, key=lambda x: x[0])
        max_leakage = between_cluster_means[0][0] if between_cluster_means else 0.0
        top_3_pairs = (
            between_cluster_means[:3]
            if len(between_cluster_means) >= 3
            else between_cluster_means
        )

        leakage = {
            "max": float(max_leakage),
            "top_pairs": [(float(sim), pair) for sim, pair in top_3_pairs],
        }

        # 5. Silhouette (on cosine distance)
        silhouette_scores = []

        # Create cluster assignments for each document
        doc_to_cluster = {}
        for cluster_id, group in enumerate(groups):
            for doc in group:
                doc_to_cluster[doc] = cluster_id

        for doc in docs:
            if doc not in doc_to_cluster:
                continue

            doc_idx = doc_to_idx[doc]
            doc_cluster = doc_to_cluster[doc]

            # Compute a_i: mean distance to own cluster
            own_cluster_docs = groups[doc_cluster]
            if len(own_cluster_docs) == 1:
                silhouette_scores.append(0.0)  # Singleton has silhouette 0
                continue

            own_cluster_dists = []
            for other_doc in own_cluster_docs:
                if other_doc != doc and other_doc in doc_to_idx:
                    other_idx = doc_to_idx[other_doc]
                    dist = 1.0 - S[doc_idx, other_idx]  # Cosine distance
                    own_cluster_dists.append(dist)

            a_i = np.mean(own_cluster_dists) if own_cluster_dists else 0.0

            # Compute b_i: smallest mean distance to any other cluster
            other_cluster_means = []
            for other_cluster_id, other_group in enumerate(groups):
                if other_cluster_id == doc_cluster:
                    continue

                other_cluster_dists = []
                for other_doc in other_group:
                    if other_doc in doc_to_idx:
                        other_idx = doc_to_idx[other_doc]
                        dist = 1.0 - S[doc_idx, other_idx]  # Cosine distance
                        other_cluster_dists.append(dist)

                if other_cluster_dists:
                    other_cluster_means.append(np.mean(other_cluster_dists))

            b_i = min(other_cluster_means) if other_cluster_means else 0.0

            # Silhouette score
            if max(a_i, b_i) > 0:
                s_i = (b_i - a_i) / max(a_i, b_i)
            else:
                s_i = 0.0

            silhouette_scores.append(s_i)

        silhouette = {
            "mean": float(np.mean(silhouette_scores)) if silhouette_scores else 0.0,
            "median": float(np.median(silhouette_scores)) if silhouette_scores else 0.0,
        }

        return {
            "pair_weighted_cohesion": pair_weighted_cohesion,
            "contrast": contrast,
            "robust_cohesion": robust_cohesion,
            "leakage": leakage,
            "silhouette": silhouette,
        }

    def _compute_standard_metrics(
        self, groups: List[List[str]], S: np.ndarray, docs: List[str]
    ) -> Dict[str, any]:
        """Compute standard clustering metrics for all algorithms."""
        if not groups or S.size == 0:
            return {
                "cohesions": [],
                "avg_cohesion": 0.0,
                "contrast": 0.0,
                "num_groups": 0,
                "global_objective": 0.0,
                "pair_weighted_cohesion": 0.0,
                "robust_cohesion": {"median": 0.0, "min": 0.0},
                "leakage": {"max": 0.0, "top_pairs": []},
                "silhouette": {"mean": 0.0, "median": 0.0},
            }

        # Get comprehensive metrics using new helper function
        comprehensive = self.compute_comprehensive_metrics(groups, S, docs)

        # Legacy cohesions for backward compatibility
        cohesions = []
        for group in groups:
            if len(group) < 2:
                cohesions.append(1.0)
            else:
                W_g = self._compute_group_score(group, S, docs)
                num_pairs = len(group) * (len(group) - 1) / 2
                avg_cohesion = W_g / num_pairs if num_pairs > 0 else 0.0
                cohesions.append(avg_cohesion)

        # Legacy statistics
        within_avg = np.mean(cohesions) if cohesions else 0.0
        overall_avg = np.mean(S[S > 0]) if S.size > 0 else 0.0
        global_F = self._compute_global_objective(groups, S, docs)

        # Group size histogram
        size_histogram = {}
        for group in groups:
            size = len(group)
            size_histogram[size] = size_histogram.get(size, 0) + 1

        # Combine legacy and comprehensive metrics
        return {
            # Legacy metrics for backward compatibility
            "cohesions": cohesions,
            "avg_cohesion": within_avg,
            "cohesion_std": np.std(cohesions) if cohesions else 0.0,
            "contrast": comprehensive["contrast"],  # Use improved contrast calculation
            "overall_avg_similarity": overall_avg,
            "within_group_avg_similarity": within_avg,
            "num_groups": len(groups),
            "global_objective": global_F,
            "group_size_histogram": size_histogram,
            # New comprehensive metrics
            "pair_weighted_cohesion": comprehensive["pair_weighted_cohesion"],
            "robust_cohesion": comprehensive["robust_cohesion"],
            "leakage": comprehensive["leakage"],
            "silhouette": comprehensive["silhouette"],
        }

        print(
            f"DEBUG - Final _compute_standard_metrics result keys: {list(result.keys())}"
        )
        print(
            f"DEBUG - pair_weighted_cohesion in result: {result.get('pair_weighted_cohesion')}"
        )

        return result

    def _perform_agglomerative_merging(
        self,
        groups: List[List[str]],
        S: np.ndarray,
        docs: List[str],
        constraints: ClusteringConstraints,
        doc_metrics: Optional[Dict[str, int]] = None,
        status: Optional["StatusProtocol"] = None,
    ) -> Tuple[List[List[str]], List[Dict]]:
        """
        Shared agglomerative merging logic for both pure and hybrid clustering.

        Performs iterative merging of groups based on linkage similarity while
        respecting size constraints.

        Args:
            groups: Initial groups to merge
            S: Document similarity matrix
            docs: Document list for indexing
            constraints: Size constraints
            doc_metrics: Optional document metrics for size computation
            status: Optional status object for progress reporting

        Returns:
            Tuple of (final_groups, operation_log)
        """
        if doc_metrics is None:
            doc_metrics = {doc: 1 for doc in docs}

        operation_log = []
        groups = [group.copy() for group in groups if group]  # Copy and remove empty

        # Split oversized groups first
        groups = self._split_oversized_groups(
            groups, constraints, doc_metrics, operation_log
        )

        doc_to_idx = {doc: i for i, doc in enumerate(docs)}

        while True:
            # Update progress status
            if status:
                active_groups = [g for g in groups if g]
                in_range = sum(
                    1
                    for g in active_groups
                    if constraints.min_size
                    <= self._compute_size(g, doc_metrics)
                    <= constraints.max_size
                )
                status.update(
                    f"Clustering agglomeratively ([green]{in_range}[/green]/[blue]{len(active_groups)}[/blue] acceptably sized)"
                )

            # Check if all groups are in valid size range
            all_valid = all(
                constraints.min_size
                <= self._compute_size(group, doc_metrics)
                <= constraints.max_size
                for group in groups
            )

            if all_valid or len(groups) <= 1:
                break

            # Find best merge
            best_merge = self._find_best_merge_pair(
                groups, S, docs, doc_to_idx, constraints, doc_metrics
            )

            if best_merge is None:
                break  # No valid merges found

            # Perform merge
            i, j = best_merge
            if i > j:
                i, j = j, i  # Ensure i < j for consistent indexing

            merged_group = groups[i] + groups[j]
            operation_log.append(
                self._log_merge(
                    i,
                    j,
                    merged_group,
                    doc_metrics,
                    operation_type="agglomerative_merge",
                    similarity=self._compute_cluster_similarity(
                        set(groups[i]), set(groups[j]), S, docs, "average"
                    ),
                )
            )

            # Update groups list
            groups[i] = merged_group
            del groups[j]

        return groups, operation_log

    def _find_best_merge_pair(
        self,
        groups: List[List[str]],
        S: np.ndarray,
        docs: List[str],
        doc_to_idx: Dict[str, int],
        constraints: ClusteringConstraints,
        doc_metrics: Dict[str, int],
    ) -> Optional[Tuple[int, int]]:
        """Find the best pair of groups to merge based on linkage similarity."""
        best_merge = None
        best_similarity = -np.inf

        for (i, group_i), (j, group_j) in itertools.combinations(enumerate(groups), 2):
            # Check if merge would violate size constraints
            merged_size = self._compute_size(group_i + group_j, doc_metrics)
            if merged_size > constraints.max_size:
                continue

            # Compute linkage similarity
            similarity = self._compute_cluster_similarity(
                set(group_i), set(group_j), S, docs, "average"
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_merge = (i, j)

        return best_merge

    def _split_oversized_groups(
        self,
        groups: List[List[str]],
        constraints: ClusteringConstraints,
        doc_metrics: Dict[str, int],
        operation_log: List[Dict],
    ) -> List[List[str]]:
        """Split groups that are too large using random partition."""
        result_groups = []

        for group in groups:
            group_size = self._compute_size(group, doc_metrics)

            if group_size <= constraints.max_size:
                result_groups.append(group)
            else:
                # Split oversized group
                if constraints.constraint_type == "words":
                    # Split by cumulative word count
                    cumulative_words = 0
                    split_groups = []
                    current_group = []

                    for doc in group:
                        doc_words = doc_metrics.get(doc, 0)
                        if (
                            cumulative_words + doc_words > constraints.max_size
                            and current_group
                        ):
                            split_groups.append(current_group)
                            current_group = [doc]
                            cumulative_words = doc_words
                        else:
                            current_group.append(doc)
                            cumulative_words += doc_words

                    if current_group:
                        split_groups.append(current_group)

                else:  # count
                    # Split by document count
                    split_groups = []
                    for i in range(0, len(group), constraints.max_size):
                        split_groups.append(group[i : i + constraints.max_size])

                result_groups.extend(split_groups)
                operation_log.append(
                    self._log_split(
                        0, group_size, split_groups, operation_type="oversized_split"
                    )
                )

        return result_groups


class AgglomerativeClusterer(DocumentClusterer):
    """Constrained agglomerative clustering with configurable linkage."""

    def __init__(self, linkage: str = "average"):
        """
        Initialize agglomerative clusterer.

        Args:
            linkage: Linkage criterion ("average", "complete", "single")
        """
        self.linkage = linkage

    def cluster(
        self,
        embeddings: Dict[str, np.ndarray],
        constraints: ClusteringConstraints,
        doc_metrics: Optional[Dict[str, int]] = None,
        status: Optional["StatusProtocol"] = None,
    ) -> ClusteringResult:
        """Perform constrained agglomerative clustering."""
        if status:
            status.update("Building similarity matrix...")

        S, docs = self._build_similarity_matrix(embeddings)

        if not docs:
            return ClusteringResult(groups=[], metrics={})

        if len(docs) == 1:
            return ClusteringResult(
                groups=[docs], metrics=self._compute_standard_metrics([docs], S, docs)
            )

        if status:
            status.update("Performing agglomerative clustering...")

        # Initialize clusters (one document per cluster)
        clusters = [{doc} for doc in docs]
        operation_log = []

        # Precompute modularity constants
        total_similarity = np.sum(S)
        doc_to_idx = {doc: i for i, doc in enumerate(docs)}
        cluster_degrees = {i: np.sum(S[i, :]) for i in range(len(docs))}

        # Compute similarity statistics and adaptive thresholds
        sim_stats = self._compute_similarity_stats(S, docs)
        if sim_stats:
            # Data-driven thresholds for natural clustering
            adaptive_thresholds = [
                sim_stats["q75"],  # Start conservative (75th percentile)
                sim_stats["median"],  # Moderate (50th percentile)
                sim_stats["q25"],  # Liberal (25th percentile)
            ]
        else:
            adaptive_thresholds = [0.5]

        # Add detailed logging with similarity distribution
        operation_log.append(
            self._log_initialization(
                docs, total_similarity, sim_stats, adaptive_thresholds, constraints
            )
        )

        # Phase 1: Natural cluster formation using adaptive similarity thresholds
        if status:
            status.update("Phase 1: Building natural clusters...")

        phase1_merges = 0
        current_threshold_idx = 0

        while any(
            len(cluster) < constraints.min_size for cluster in clusters
        ) and current_threshold_idx < len(adaptive_thresholds):
            # Update status with current progress
            if status:
                active_groups = [c for c in clusters if c]
                in_range = sum(
                    1
                    for c in active_groups
                    if constraints.min_size <= len(c) <= constraints.max_size
                )
                undersized = sum(
                    1 for c in active_groups if len(c) < constraints.min_size
                )
                status.update(
                    f"Clustering agglomeratively ([green]{in_range}[/green]/[blue]{len(active_groups)}[/blue] acceptably sized)"
                )
            current_threshold = adaptive_thresholds[current_threshold_idx]

            # Pre-compute cluster indices for vectorized operations
            cluster_indices = self._precompute_cluster_indices(clusters, docs)

            # Use vectorized candidate evaluation
            all_candidates, best_pair, best_similarity = (
                self._evaluate_all_candidates_vectorized(
                    clusters,
                    cluster_indices,
                    S,
                    constraints,
                    current_threshold,
                    doc_metrics,
                )
            )

            # Log phase 1 iteration with similarity-based decisions
            operation_log.append(
                self._log_iteration(
                    "phase1",
                    phase1_merges,
                    clusters_below_min=sum(
                        1 for c in clusters if len(c) < constraints.min_size
                    ),
                    total_clusters=len(clusters),
                    candidates_evaluated=len(all_candidates),
                    current_threshold=current_threshold,
                    threshold_level=["Q3", "median", "Q1"][current_threshold_idx]
                    if current_threshold_idx < 3
                    else "emergency",
                    candidates_above_threshold=len(all_candidates),
                    best_similarity=float(best_similarity) if best_pair else None,
                    best_pair_sizes=(
                        len(clusters[best_pair[0]]),
                        len(clusters[best_pair[1]]),
                    )
                    if best_pair
                    else None,
                    similarity_distribution=[c["similarity"] for c in all_candidates]
                    if all_candidates
                    else [],
                )
            )

            # If no candidates found at current threshold, try next threshold
            if best_pair is None:
                operation_log.append(
                    self._log_base(
                        "threshold_lowered",
                        from_threshold=current_threshold,
                        from_level=["Q3", "median", "Q1"][current_threshold_idx]
                        if current_threshold_idx < 3
                        else "emergency",
                        remaining_undersized=sum(
                            1 for c in clusters if len(c) < constraints.min_size
                        ),
                        total_candidates_evaluated=len(all_candidates),
                    )
                )
                current_threshold_idx += 1
                continue

            # Perform merge
            i, j = best_pair
            merged_cluster = clusters[i] | clusters[j]

            # Find merge details for this pair
            helps_undersized = None
            for candidate in all_candidates:
                if candidate["clusters"] == (i, j):
                    helps_undersized = candidate["helps_undersized"]
                    break

            # Log the merge with similarity score for Phase 1
            operation_log.append(
                self._log_merge(
                    i,
                    j,
                    list(merged_cluster),
                    operation_type="phase1_merge",
                    sizes=[len(clusters[i]), len(clusters[j])],
                    similarity_score=best_similarity,
                    threshold_used=current_threshold,
                    threshold_level=["Q3", "median", "Q1"][current_threshold_idx]
                    if current_threshold_idx < 3
                    else "emergency",
                    helps_undersized=helps_undersized,
                )
            )

            # Update clusters list (remove i, j and add merged)
            self._apply_merge(clusters, i, j, merged_cluster)
            phase1_merges += 1

        # Emergency fallback: if clusters still below min_size after all thresholds
        if any(len(cluster) < constraints.min_size for cluster in clusters):
            operation_log.append(
                self._log_base(
                    "phase1_emergency_fallback",
                    remaining_undersized=[
                        len(c) for c in clusters if len(c) < constraints.min_size
                    ],
                    reason="all_thresholds_exhausted",
                )
            )
            # Try one final round with any similarity > 0
            while any(len(cluster) < constraints.min_size for cluster in clusters):
                best_pair = None
                best_similarity = -np.inf

                for (i, cluster_a), (j, cluster_b) in itertools.combinations(
                    enumerate(clusters), 2
                ):
                    merged_size = self._compute_size(
                        list(cluster_a | cluster_b), doc_metrics
                    )
                    if merged_size > constraints.max_size:
                        continue

                    if (
                        len(cluster_a) < constraints.min_size
                        or len(cluster_b) < constraints.min_size
                    ):
                        similarity = self._compute_cluster_similarity(
                            cluster_a, cluster_b, S, docs, self.linkage
                        )
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_pair = (i, j)

                if best_pair is None:
                    break

                # Perform emergency merge
                i, j = best_pair
                merged_cluster = clusters[i] | clusters[j]
                operation_log.append(
                    self._log_merge(
                        i,
                        j,
                        list(merged_cluster),
                        operation_type="phase1_emergency_merge",
                        sizes=[len(clusters[i]), len(clusters[j])],
                        similarity_score=best_similarity,
                    )
                )

                self._apply_merge(clusters, i, j, merged_cluster)
                phase1_merges += 1

        # Log Phase 1 completion
        operation_log.append(
            self._log_base(
                "phase1_complete",
                merges_performed=phase1_merges,
                clusters_at_min_size=sum(
                    1 for c in clusters if len(c) >= constraints.min_size
                ),
                total_clusters=len(clusters),
                cluster_sizes=[len(c) for c in clusters],
            )
        )

        # Phase 2: Optimize clustering based on modularity
        if status:
            status.update("Phase 2: Optimizing clusters...")

        phase2_merges = 0
        while len(clusters) > 1:
            best_pair = None
            best_delta_q = 0  # Only merge if improves modularity
            all_candidates = []

            for (i, cluster_a), (j, cluster_b) in itertools.combinations(
                enumerate(clusters), 2
            ):
                # Check size constraint
                merged_size = self._compute_size(
                    list(cluster_a | cluster_b), doc_metrics
                )
                if merged_size > constraints.max_size:
                    all_candidates.append(
                        {
                            "clusters": (i, j),
                            "sizes": (len(cluster_a), len(cluster_b)),
                            "merged_size": merged_size,
                            "rejected": "size_constraint",
                            "delta_q": None,
                        }
                    )
                    continue

                # Calculate modularity gain
                delta_q = self._compute_modularity_gain(
                    i, j, clusters, S, doc_to_idx, cluster_degrees, total_similarity
                )
                all_candidates.append(
                    {
                        "clusters": (i, j),
                        "sizes": (len(cluster_a), len(cluster_b)),
                        "merged_size": merged_size,
                        "delta_q": delta_q,
                        "rejected": "negative_modularity" if delta_q <= 0 else None,
                    }
                )

                if delta_q > best_delta_q:
                    best_delta_q = delta_q
                    best_pair = (i, j)

            # Log detailed phase 2 iteration
            operation_log.append(
                self._log_iteration(
                    "phase2",
                    phase2_merges,
                    total_clusters=len(clusters),
                    candidates_evaluated=len(all_candidates),
                    candidates_rejected_size=sum(
                        1
                        for c in all_candidates
                        if c.get("rejected") == "size_constraint"
                    ),
                    candidates_rejected_modularity=sum(
                        1
                        for c in all_candidates
                        if c.get("rejected") == "negative_modularity"
                    ),
                    candidates_valid=sum(
                        1 for c in all_candidates if not c.get("rejected")
                    ),
                    best_delta_q=float(best_delta_q) if best_pair else 0,
                    best_pair_sizes=(
                        len(clusters[best_pair[0]]),
                        len(clusters[best_pair[1]]),
                    )
                    if best_pair
                    else None,
                    cluster_sizes=[len(c) for c in clusters],
                    modularity_values=[
                        {
                            "clusters": c["clusters"],
                            "sizes": c["sizes"],
                            "delta_q": c["delta_q"],
                            "rejected": c.get("rejected"),
                        }
                        for c in all_candidates
                        if c["delta_q"] is not None
                    ],
                    modularity_distribution=[
                        c["delta_q"] for c in all_candidates if c["delta_q"] is not None
                    ],
                )
            )

            # Stop if no merge improves modularity
            if best_pair is None or best_delta_q <= 0:
                operation_log.append(
                    self._log_base(
                        "phase2_stopping",
                        reason="no_positive_modularity"
                        if best_delta_q <= 0
                        else "no_valid_candidates",
                        final_clusters=len(clusters),
                        final_delta_q=float(best_delta_q),
                        final_cluster_sizes=[len(c) for c in clusters],
                    )
                )
                break

            # Perform merge
            i, j = best_pair
            merged_cluster = clusters[i] | clusters[j]
            self._perform_merge(
                i,
                j,
                merged_cluster,
                clusters,
                cluster_degrees,
                S,
                doc_to_idx,
                operation_log,
                best_delta_q,
            )
            phase2_merges += 1

            # Update progress
            if status:
                total_docs = len(docs)
                docs_in_groups = sum(
                    len(cluster)
                    for cluster in clusters
                    if len(cluster) >= constraints.min_size
                )
                coverage_pct = (
                    int(100 * docs_in_groups / total_docs) if total_docs > 0 else 0
                )
                status.update(
                    f"Phase 2: {len(clusters)} groups, {coverage_pct}% coverage, modularity gain: {best_delta_q:.3f}"
                )

        # Convert to final format, filtering by min_size
        groups = []
        for cluster in clusters:
            cluster_size = self._compute_size(list(cluster), doc_metrics)
            if cluster_size >= constraints.min_size:
                groups.append(list(cluster))

        # Final logging
        operation_log.append(
            self._log_base(
                "clustering_complete",
                total_phase1_merges=phase1_merges,
                total_phase2_merges=phase2_merges,
                final_groups=len(groups),
                final_group_sizes=[len(g) for g in groups],
                groups_at_max_size=sum(
                    1 for g in groups if len(g) == constraints.max_size
                ),
                groups_at_min_size=sum(
                    1 for g in groups if len(g) == constraints.min_size
                ),
            )
        )

        # Compute metrics
        metrics = self._compute_standard_metrics(groups, S, docs)
        metrics["algorithm"] = f"agglomerative_{self.linkage}_modularity"

        return ClusteringResult(
            groups=groups, metrics=metrics, operation_log=operation_log
        )

    def _compute_modularity_gain(
        self,
        i: int,
        j: int,
        clusters: List[set],
        S: np.ndarray,
        doc_to_idx: Dict[str, int],
        cluster_degrees: Dict[int, float],
        total_similarity: float,
    ) -> float:
        """Calculate modularity gain from merging clusters i and j."""
        cluster_a, cluster_b = clusters[i], clusters[j]

        # Edge weight between clusters
        e_ij = 0.0
        for doc_a in cluster_a:
            for doc_b in cluster_b:
                if doc_a in doc_to_idx and doc_b in doc_to_idx:
                    e_ij += S[doc_to_idx[doc_a], doc_to_idx[doc_b]]

        # Total degrees for each cluster
        d_i = sum(
            cluster_degrees[doc_to_idx[doc]] for doc in cluster_a if doc in doc_to_idx
        )
        d_j = sum(
            cluster_degrees[doc_to_idx[doc]] for doc in cluster_b if doc in doc_to_idx
        )

        # Modularity gain formula: 2 * (e_ij/m - (d_i * d_j)/(2m)²)
        if total_similarity > 0:
            delta_q = 2 * (
                e_ij / total_similarity - (d_i * d_j) / (total_similarity**2)
            )
        else:
            delta_q = 0.0

        return delta_q

    def _perform_merge(
        self,
        i: int,
        j: int,
        merged_cluster: set,
        clusters: List[set],
        cluster_degrees: Dict[int, float],
        S: np.ndarray,
        doc_to_idx: Dict[str, int],
        operation_log: List[Dict],
        delta_q: float,
    ) -> None:
        """Perform cluster merge and update data structures."""
        operation_log.append(
            self._log_merge(
                i,
                j,
                list(merged_cluster),
                operation_type="modularity_merge",
                sizes=[len(clusters[i]), len(clusters[j])],
                modularity_gain=delta_q,
            )
        )

        # Update clusters list (remove i, j and add merged)
        self._apply_merge(clusters, i, j, merged_cluster)


class SpectralClusterer(DocumentClusterer):
    """Spectral clustering with size-aware post-processing."""

    def __init__(self, random_state: int = 42):
        """
        Initialize spectral clusterer.

        Args:
            random_state: Random seed for reproducible results
        """
        self.random_state = random_state

    def cluster(
        self,
        embeddings: Dict[str, np.ndarray],
        constraints: ClusteringConstraints,
        doc_metrics: Optional[Dict[str, int]] = None,
        status: Optional["StatusProtocol"] = None,
    ) -> ClusteringResult:
        """Perform spectral clustering with size constraints."""
        if status:
            status.update("Building similarity matrix...")

        S, docs = self._build_similarity_matrix(embeddings)

        if not docs:
            return ClusteringResult(groups=[], metrics={})

        if len(docs) == 1:
            return ClusteringResult(
                groups=[docs], metrics=self._compute_standard_metrics([docs], S, docs)
            )

        if status:
            status.update("Performing spectral clustering...")

        # Estimate reasonable number of clusters
        n_docs = len(docs)
        avg_target_size = (constraints.min_size + constraints.max_size) / 2
        estimated_groups = max(1, int(n_docs / avg_target_size))

        try:
            from sklearn.cluster import SpectralClustering

            # Apply spectral clustering
            clusterer = SpectralClustering(
                n_clusters=estimated_groups,
                affinity="precomputed",
                random_state=self.random_state,
                assign_labels="discretize",
            )

            labels = clusterer.fit_predict(S)

            # Convert labels to groups
            groups_dict = {}
            for i, label in enumerate(labels):
                if label not in groups_dict:
                    groups_dict[label] = []
                groups_dict[label].append(docs[i])

            initial_groups = list(groups_dict.values())

        except Exception:
            # Fallback to random grouping if spectral fails
            import random

            random.seed(self.random_state)
            shuffled_docs = docs.copy()
            random.shuffle(shuffled_docs)

            initial_groups = []
            current_group = []
            current_size = 0

            for doc in shuffled_docs:
                doc_size = doc_metrics.get(doc, 1) if doc_metrics else 1
                if current_size + doc_size > constraints.max_size and current_group:
                    initial_groups.append(current_group)
                    current_group = [doc]
                    current_size = doc_size
                else:
                    current_group.append(doc)
                    current_size += doc_size

            if current_group:
                initial_groups.append(current_group)

        if status:
            status.update("Enforcing size constraints...")

        # Post-process for size constraints
        final_groups = self._enforce_constraints(
            initial_groups, S, docs, constraints, doc_metrics
        )

        # Compute metrics
        metrics = self._compute_standard_metrics(final_groups, S, docs)
        metrics["algorithm"] = "spectral"
        metrics["initial_groups"] = len(initial_groups)

        return ClusteringResult(groups=final_groups, metrics=metrics)

    def _enforce_constraints(
        self,
        groups: List[List[str]],
        S: np.ndarray,
        docs: List[str],
        constraints: ClusteringConstraints,
        doc_metrics: Optional[Dict[str, int]],
    ) -> List[List[str]]:
        """Post-process groups to enforce size constraints."""
        doc_to_idx = {doc: i for i, doc in enumerate(docs)}
        result_groups = []

        for group in groups:
            group_size = self._compute_size(group, doc_metrics)

            if (
                group_size >= constraints.min_size
                and group_size <= constraints.max_size
            ):
                result_groups.append(group)
            elif group_size < constraints.min_size:
                # Try to merge with another small group or add to existing group
                merged = False
                for existing_group in result_groups:
                    combined_size = self._compute_size(
                        existing_group + group, doc_metrics
                    )
                    if combined_size <= constraints.max_size:
                        existing_group.extend(group)
                        merged = True
                        break

                if not merged:
                    # Keep undersized group if no merge possible
                    result_groups.append(group)

            elif group_size > constraints.max_size:
                # Simple split: divide roughly in half
                mid = len(group) // 2
                group1 = group[:mid]
                group2 = group[mid:]

                # Recursively enforce constraints on split groups
                for subgroup in [group1, group2]:
                    subgroup_size = self._compute_size(subgroup, doc_metrics)
                    if subgroup_size >= constraints.min_size:
                        result_groups.append(subgroup)

        return result_groups


class HybridClusterer(DocumentClusterer):
    """
    Hybrid clustering: Spectral seeding + hierarchical refinement.

    Implements the exact specification for hybrid clustering with
    spectral seeding and objective-based refinement operations.
    """

    def __init__(
        self,
        n_components: int = 10,
        seed_multiplier: float = 1.5,
        seeding_method: str = "kmeans",
        refinement_method: str = "hierarchical",
        random_state: int = 42,
        max_iterations: int = 100,
        enable_swaps: bool = False,
        hysteresis_margin: float = 1e-6,
    ):
        """
        Initialize hybrid clusterer.

        Args:
            n_components: Number of spectral embedding dimensions
            seed_multiplier: Multiplier for over-segmentation (creates k_seed = ceil(expected * multiplier))
            seeding_method: Spectral seeding method ("kmeans" or "fiedler")
            refinement_method: Post-seeding refinement ("hierarchical" or "agglomerative")
            random_state: Random seed for deterministic results
            max_iterations: Maximum refinement iterations
            enable_swaps: Whether to perform pairwise document swaps
            hysteresis_margin: Minimum improvement required to prevent oscillations
        """
        self.n_components = n_components
        self.seed_multiplier = seed_multiplier
        self.seeding_method = seeding_method
        self.refinement_method = refinement_method
        self.random_state = random_state
        self.max_iterations = max_iterations
        self.enable_swaps = enable_swaps
        self.hysteresis_margin = hysteresis_margin

    def cluster(
        self,
        embeddings: Dict[str, np.ndarray],
        constraints: ClusteringConstraints,
        doc_metrics: Optional[Dict[str, int]] = None,
        status: Optional["StatusProtocol"] = None,
    ) -> ClusteringResult:
        """
        Perform hybrid clustering following the exact specification.

        Pipeline:
        1. Build similarity matrix S (immutable)
        2. Spectral seeding without explicit graph
        3. Initialize objective and caches
        4. Hierarchical refinement (merge undersized -> reduce oversized -> border moves -> swaps)
        5. Output groups and metrics
        """
        if status:
            status.update("Building similarity matrix...")

        # Step 1: Build similarity matrix (single source of truth)
        S, docs = self._build_similarity_matrix(embeddings)

        if not docs:
            return ClusteringResult(groups=[], metrics={})

        if len(docs) == 1:
            return ClusteringResult(
                groups=[docs], metrics=self._compute_standard_metrics([docs], S, docs)
            )

        # Step 2: Spectral seeding
        if status:
            status.update("Performing spectral seeding...")

        if self.seeding_method == "fiedler":
            micro_communities = self._spectral_bipartition_seeding(
                S, docs, constraints, doc_metrics
            )
        else:
            micro_communities = self._spectral_seeding(
                S, docs, constraints, doc_metrics
            )

        # Step 3: Initialize caches for efficient refinement
        if status:
            status.update("Initializing refinement caches...")

        groups = micro_communities
        caches = self._initialize_caches(groups, S, docs, doc_metrics)
        operation_log = []
        iteration = 0  # Initialize iteration counter for both refinement methods

        # Step 4: Refinement
        if status:
            status.update("Performing refinement...")

        if self.refinement_method == "agglomerative":
            # Use shared agglomerative merging for micro-communities
            groups, operation_log = self._perform_agglomerative_merging(
                groups, S, docs, constraints, doc_metrics, status
            )
            # Agglomerative refinement doesn't use iterations in the same way
            iteration = 1
        else:
            # Original hierarchical approach
            # 4.1: Merge undersized groups (first)
            groups, caches, ops = self._merge_undersized_groups(
                groups, S, docs, constraints, doc_metrics, caches
            )
            operation_log.extend(ops)

            # 4.2: Reduce oversized groups
            groups, caches, ops = self._reduce_oversized_groups(
                groups, S, docs, constraints, doc_metrics, caches
            )
            operation_log.extend(ops)

            # 4.3: Border move sweep (iterative refinement)
            for iteration in range(self.max_iterations):
                if status:
                    status.update(f"Border moves iteration {iteration + 1}")

                old_F = caches.get(
                    "global_F", self._compute_global_objective(groups, S, docs)
                )

                groups, caches, move_ops = self._border_move_sweep(
                    groups, S, docs, constraints, doc_metrics, caches
                )
                operation_log.extend(move_ops)

                # 4.4: Optional pairwise swaps
                if self.enable_swaps:
                    groups, caches, swap_ops = self._perform_swaps(
                        groups, S, docs, constraints, doc_metrics, caches
                    )
                    operation_log.extend(swap_ops)

                new_F = self._compute_global_objective(groups, S, docs)
                caches["global_F"] = new_F

                # Termination check
                if new_F - old_F <= self.hysteresis_margin:
                    break

                if not move_ops and (not self.enable_swaps or not swap_ops):
                    break  # No operations performed

        # Step 5: Compute final metrics
        if status:
            status.update("Computing final metrics...")

        metrics = self._compute_standard_metrics(groups, S, docs)
        metrics["algorithm"] = "hybrid"
        metrics["spectral_seeds"] = len(micro_communities)
        metrics["refinement_iterations"] = min(iteration + 1, self.max_iterations)

        return ClusteringResult(
            groups=groups, metrics=metrics, operation_log=operation_log
        )

    def _spectral_seeding(
        self,
        S: np.ndarray,
        docs: List[str],
        constraints: ClusteringConstraints,
        doc_metrics: Optional[Dict[str, int]],
    ) -> List[List[str]]:
        """
        Step 2: Spectral seeding without explicit graph.

        Following specification:
        - Use W = clip(S, 0, None) as affinity
        - Spectral embedding with row normalization
        - Over-segment into micro-communities using k-means
        """
        from sklearn.manifold import SpectralEmbedding
        from sklearn.cluster import KMeans

        n_docs = len(docs)

        # Build affinity matrix (non-negative, diagonal 0)
        W = np.clip(S, 0, None)

        # Determine embedding dimension
        r = min(self.n_components, n_docs - 1)
        if r <= 0:
            return [docs]  # Single group fallback

        try:
            # Spectral embedding
            embedding = SpectralEmbedding(
                n_components=r, affinity="precomputed", random_state=self.random_state
            )
            Y = embedding.fit_transform(W)

            # Row-normalize to unit length
            from sklearn.preprocessing import normalize

            Y = normalize(Y, norm="l2", axis=1)

        except Exception:
            # Fallback to random assignment if spectral fails
            np.random.seed(self.random_state)
            Y = np.random.randn(n_docs, r)
            Y = normalize(Y, norm="l2", axis=1)

        # Over-segmentation into micro-communities
        avg_target_size = (constraints.min_size + constraints.max_size) / 2
        expected_final_groups = max(1, int(n_docs / avg_target_size))
        k_seed = int(np.ceil(expected_final_groups * self.seed_multiplier))
        k_seed = min(k_seed, n_docs)  # Can't have more clusters than documents

        try:
            # K-means on spectral embedding
            kmeans = KMeans(
                n_clusters=k_seed,
                n_init="auto",
                random_state=self.random_state,
                algorithm="lloyd",
            )
            labels = kmeans.fit_predict(Y)

            # Convert to micro-communities
            micro_communities = [[] for _ in range(k_seed)]
            for i, label in enumerate(labels):
                micro_communities[label].append(docs[i])

            # Remove empty communities
            return [community for community in micro_communities if community]

        except Exception:
            # Fallback: each document as its own micro-community
            return [[doc] for doc in docs]

    def _spectral_bipartition_seeding(
        self,
        S: np.ndarray,
        docs: List[str],
        constraints: ClusteringConstraints,
        doc_metrics: Optional[Dict[str, int]],
    ) -> List[List[str]]:
        """
        Spectral seeding using recursive bipartition with Fiedler vector.

        Uses size-constrained recursive bipartition instead of k-means
        to create initial clusters that naturally respect size constraints.
        """
        if doc_metrics is None:
            doc_metrics = {doc: 1 for doc in docs}

        # Start recursive bipartition with all documents
        all_indices = list(range(len(docs)))
        final_groups = []

        self._recursive_bipartition(
            S, docs, all_indices, constraints, doc_metrics, final_groups
        )

        return final_groups

    def _recursive_bipartition(
        self,
        S: np.ndarray,
        docs: List[str],
        indices: List[int],
        constraints: ClusteringConstraints,
        doc_metrics: Dict[str, int],
        final_groups: List[List[str]],
    ):
        """
        Recursively bipartition a group using Fiedler vector if beneficial.

        Args:
            S: Full similarity matrix
            docs: All document names
            indices: Indices of documents in current group
            constraints: Size constraints
            doc_metrics: Document size metrics
            final_groups: Output list to append final clusters
        """
        if not indices:
            return

        # Get documents and metrics for this group
        group_docs = [docs[i] for i in indices]
        group_size = len(group_docs)

        if constraints.constraint_type == "words":
            group_metric = sum(doc_metrics.get(docs[i], 0) for i in indices)
        else:
            group_metric = group_size

        # Step 1: Stop if size is valid
        if constraints.min_size <= group_metric <= constraints.max_size:
            final_groups.append(group_docs)
            return

        # Don't split if too small
        if group_size < 2:
            final_groups.append(group_docs)
            return

        # Step 2: Compute Fiedler vector for this subgroup
        try:
            fiedler_values = self._compute_fiedler_vector(S, indices)
            if fiedler_values is None:
                final_groups.append(group_docs)
                return

            # Step 3: Find best feasible cut
            best_cut = self._find_best_feasible_cut(
                S, docs, indices, fiedler_values, constraints, doc_metrics
            )

            if best_cut is None:
                # No beneficial cut found
                final_groups.append(group_docs)
                return

            # Step 4: Split and recurse
            left_indices, right_indices = best_cut
            self._recursive_bipartition(
                S, docs, left_indices, constraints, doc_metrics, final_groups
            )
            self._recursive_bipartition(
                S, docs, right_indices, constraints, doc_metrics, final_groups
            )

        except Exception:
            # Fallback on any error
            final_groups.append(group_docs)

    def _compute_fiedler_vector(
        self, S: np.ndarray, indices: List[int]
    ) -> Optional[np.ndarray]:
        """
        Compute Fiedler vector (2nd eigenvector) for subgroup.

        Args:
            S: Full similarity matrix
            indices: Indices of documents in subgroup

        Returns:
            Fiedler values for ordering, or None if computation fails
        """
        if len(indices) < 2:
            return None

        # Extract submatrix for this group
        S_sub = S[np.ix_(indices, indices)]

        # Build affinity matrix (non-negative, zero diagonal)
        W = np.clip(S_sub, 0, None)
        np.fill_diagonal(W, 0)

        # Compute degree matrix
        degrees = np.array(W.sum(axis=1)).flatten()

        # Handle isolated nodes
        degrees = np.where(degrees > 1e-8, degrees, 1.0)

        try:
            # Normalized Laplacian: L_sym = I - D^(-1/2) W D^(-1/2)
            D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
            L_sym = np.eye(len(indices)) - D_inv_sqrt @ W @ D_inv_sqrt

            # Compute eigenvalues and eigenvectors
            eigenvals, eigenvecs = np.linalg.eigh(L_sym)

            # Second smallest eigenvalue gives Fiedler vector
            if len(eigenvals) >= 2:
                fiedler_idx = np.argsort(eigenvals)[1]
                return eigenvecs[:, fiedler_idx]
            else:
                return None

        except Exception:
            return None

    def _find_best_feasible_cut(
        self,
        S: np.ndarray,
        docs: List[str],
        indices: List[int],
        fiedler_values: np.ndarray,
        constraints: ClusteringConstraints,
        doc_metrics: Dict[str, int],
    ) -> Optional[Tuple[List[int], List[int]]]:
        """
        Find the best feasible cut along Fiedler ordering.

        Args:
            S: Full similarity matrix
            docs: All document names
            indices: Indices of documents in current group
            fiedler_values: Fiedler vector values for ordering
            constraints: Size constraints
            doc_metrics: Document metrics

        Returns:
            Tuple of (left_indices, right_indices) or None if no good cut
        """
        n = len(indices)
        if n < 2:
            return None

        # Sort indices by Fiedler values
        sorted_pairs = sorted(zip(fiedler_values, indices))
        ordered_indices = [idx for _, idx in sorted_pairs]

        # Precompute group metrics for efficiency
        if constraints.constraint_type == "words":
            individual_metrics = [
                doc_metrics.get(docs[idx], 0) for idx in ordered_indices
            ]
        else:
            individual_metrics = [1 for _ in ordered_indices]

        # Precompute within-group score for efficiency
        S_sub = S[np.ix_(indices, indices)]
        total_within = np.sum(S_sub) / 2  # Avoid double counting

        best_cut = None
        best_delta_F = -np.inf

        # Try each cut position
        for t in range(1, n):  # t is left size
            left_indices = ordered_indices[:t]
            right_indices = ordered_indices[t:]

            # Check feasibility of both sides
            if not self._is_side_feasible(left_indices, constraints, doc_metrics, docs):
                continue
            if not self._is_side_feasible(
                right_indices, constraints, doc_metrics, docs
            ):
                continue

            # Compute delta F for this cut
            delta_F = self._compute_split_delta_F(
                S, left_indices, right_indices, total_within
            )

            if delta_F > best_delta_F:
                best_delta_F = delta_F
                best_cut = (left_indices, right_indices)

        # Only return cut if it improves objective
        return best_cut if best_delta_F > 0 else None

    def _is_side_feasible(
        self,
        side_indices: List[int],
        constraints: ClusteringConstraints,
        doc_metrics: Dict[str, int],
        docs: List[str],
    ) -> bool:
        """Check if a side can form valid groups."""
        if not side_indices:
            return False

        if constraints.constraint_type == "words":
            side_metric = sum(doc_metrics.get(docs[idx], 0) for idx in side_indices)
        else:
            side_metric = len(side_indices)

        # Feasibility test: can this side be split into valid groups?
        k_min = int(np.ceil(side_metric / constraints.max_size))
        k_max = int(np.floor(side_metric / constraints.min_size))

        return k_min <= k_max and side_metric >= constraints.min_size

    def _compute_split_delta_F(
        self,
        S: np.ndarray,
        left_indices: List[int],
        right_indices: List[int],
        total_within: float,
    ) -> float:
        """
        Compute change in objective function F for a split.

        F = Σ W(G), so ΔF = W(G1) + W(G2) - W(G_original)
        """
        # Compute within-group scores for left and right
        if left_indices:
            S_left = S[np.ix_(left_indices, left_indices)]
            W_left = np.sum(S_left) / 2
        else:
            W_left = 0

        if right_indices:
            S_right = S[np.ix_(right_indices, right_indices)]
            W_right = np.sum(S_right) / 2
        else:
            W_right = 0

        # Delta F = sum of split groups minus original group
        return W_left + W_right - total_within

    def _find_best_merge(
        self,
        groups: List[List[str]],
        S: np.ndarray,
        docs: List[str],
        doc_to_idx: Dict[str, int],
        constraints: ClusteringConstraints,
        doc_metrics: Dict[str, int],
    ) -> Optional[Tuple[int, int]]:
        """
        Find the best pair of groups to merge based on inter-cluster similarity.

        Returns:
            Tuple of (group_i, group_j) indices or None if no valid merge
        """
        best_merge = None
        best_similarity = -np.inf

        for (i, group_i), (j, group_j) in itertools.combinations(enumerate(groups), 2):
            # Check if merge would violate size constraints
            merged_size = self._compute_size(group_i + group_j, doc_metrics)
            if merged_size > constraints.max_size:
                continue

            # Compute inter-cluster similarity (average linkage)
            inter_sim = self._compute_cluster_similarity(
                set(group_i), set(group_j), S, docs, "average"
            )

            if inter_sim > best_similarity:
                best_similarity = inter_sim
                best_merge = (i, j)

        return best_merge

    def _initialize_caches(
        self,
        groups: List[List[str]],
        S: np.ndarray,
        docs: List[str],
        doc_metrics: Optional[Dict[str, int]],
    ) -> Dict[str, any]:
        """
        Step 3: Initialize caches for efficient incremental updates.

        Maintains:
        - W(G): within-group scores
        - size[G]: group sizes
        - membership: doc -> group mapping
        - global_F: total objective value
        """
        doc_to_idx = {doc: i for i, doc in enumerate(docs)}
        caches = {
            "W": {},  # W(G) scores per group
            "size": {},  # Group sizes
            "membership": {},  # Document -> group index mapping
            "doc_to_idx": doc_to_idx,
        }

        total_F = 0.0

        for g_idx, group in enumerate(groups):
            # Compute W(G) - sum of within-group pairwise cosines
            W_g = self._compute_group_score(group, S, docs)
            caches["W"][g_idx] = W_g
            total_F += W_g

            # Store group size
            caches["size"][g_idx] = self._compute_size(group, doc_metrics)

            # Store membership mapping
            for doc in group:
                caches["membership"][doc] = g_idx

        caches["global_F"] = total_F
        return caches

    def _merge_undersized_groups(
        self,
        groups: List[List[str]],
        S: np.ndarray,
        docs: List[str],
        constraints: ClusteringConstraints,
        doc_metrics: Optional[Dict[str, int]],
        caches: Dict,
    ) -> Tuple[List[List[str]], Dict, List[Dict]]:
        """
        Step 4.1: Merge undersized groups with best neighbors.

        For each undersized group, find neighbor that maximizes R(G,H)
        subject to size constraints. Use feasibility override if needed.
        """
        operations = []
        groups = [group.copy() for group in groups]
        doc_to_idx = caches["doc_to_idx"]

        made_changes = True
        while made_changes:
            made_changes = False

            # Find all undersized groups
            undersized_indices = [
                g_idx
                for g_idx, group in enumerate(groups)
                if self._compute_size(group, doc_metrics) < constraints.min_size
            ]

            if not undersized_indices:
                break

            # Process first undersized group
            g_idx = undersized_indices[0]
            if g_idx >= len(groups):  # May have been merged already
                continue

            group = groups[g_idx]
            best_partner = None
            best_R = -np.inf
            feasibility_override = False

            # Find best merge partner
            for h_idx, other_group in enumerate(groups):
                if h_idx == g_idx:
                    continue

                # Compute R(G,H) - cross-sum between groups
                R_gh = 0.0
                for doc_g in group:
                    for doc_h in other_group:
                        if doc_g in doc_to_idx and doc_h in doc_to_idx:
                            R_gh += S[doc_to_idx[doc_g], doc_to_idx[doc_h]]

                # Check size constraint
                merged_size = self._compute_size(group + other_group, doc_metrics)

                if merged_size <= constraints.max_size:
                    # Valid merge - check if best
                    if R_gh > best_R:
                        best_R = R_gh
                        best_partner = h_idx
                        feasibility_override = False
                elif best_partner is None:
                    # Feasibility override - choose best even if oversized
                    if R_gh > best_R:
                        best_R = R_gh
                        best_partner = h_idx
                        feasibility_override = True

            # Perform merge if partner found
            if best_partner is not None:
                partner_group = groups[best_partner]
                merged_group = group + partner_group

                operations.append(
                    self._log_base(
                        "merge_undersized",
                        source_groups=[g_idx, best_partner],
                        source_sizes=[len(group), len(partner_group)],
                        delta_F=best_R,  # ΔF_merge = R(G,H)
                        feasibility_override=feasibility_override,
                    )
                )

                # Update groups list (remove old, add merged)
                groups = [
                    g for k, g in enumerate(groups) if k not in (g_idx, best_partner)
                ] + [merged_group]

                # Rebuild caches after structural change
                caches = self._initialize_caches(groups, S, docs, doc_metrics)
                made_changes = True

        return groups, caches, operations

    def _reduce_oversized_groups(
        self,
        groups: List[List[str]],
        S: np.ndarray,
        docs: List[str],
        constraints: ClusteringConstraints,
        doc_metrics: Optional[Dict[str, int]],
        caches: Dict,
    ) -> Tuple[List[List[str]], Dict, List[Dict]]:
        """
        Step 4.2: Reduce oversized groups through ejection or splitting.

        Default: ejection-based shrinking. Escalation: one internal 2-way split if needed.
        """
        operations = []
        groups = [group.copy() for group in groups]
        doc_to_idx = caches["doc_to_idx"]

        i = 0
        while i < len(groups):
            group = groups[i]
            group_size = self._compute_size(group, doc_metrics)

            if group_size <= constraints.max_size:
                i += 1
                continue

            # Try ejection-based shrinking first
            ejection_success = False

            while group_size > constraints.max_size and len(group) > 1:
                # Find lowest-affinity member
                lowest_affinity = np.inf
                worst_member = None

                for doc in group:
                    if doc in doc_to_idx:
                        # Compute internal affinity a_G(v)
                        affinity = 0.0
                        for other_doc in group:
                            if other_doc != doc and other_doc in doc_to_idx:
                                affinity += S[doc_to_idx[doc], doc_to_idx[other_doc]]

                        if affinity < lowest_affinity:
                            lowest_affinity = affinity
                            worst_member = doc

                if worst_member is None:
                    break

                # Find best destination for ejected member
                best_destination = None
                best_delta_F = -np.inf

                for h_idx, target_group in enumerate(groups):
                    if h_idx == i:
                        continue

                    # Check if target can accommodate
                    target_size = self._compute_size(target_group, doc_metrics)
                    member_size = doc_metrics.get(worst_member, 1) if doc_metrics else 1

                    if target_size + member_size > constraints.max_size:
                        continue

                    # Compute ΔF_move = -a_G(v) + Σ_{j∈H} S_{vj}
                    delta_F = -lowest_affinity  # Remove from current group

                    if worst_member in doc_to_idx:
                        for target_doc in target_group:
                            if target_doc in doc_to_idx:
                                delta_F += S[
                                    doc_to_idx[worst_member], doc_to_idx[target_doc]
                                ]

                    if delta_F > best_delta_F:
                        best_delta_F = delta_F
                        best_destination = h_idx

                # Perform ejection
                if best_destination is not None and best_delta_F > 0:
                    # Beneficial ejection
                    groups[i].remove(worst_member)
                    groups[best_destination].append(worst_member)

                    operations.append(
                        self._log_ejection(
                            worst_member, i, best_destination, best_delta_F
                        )
                    )

                    ejection_success = True

                elif best_destination is not None:
                    # Feasibility override - eject to best available destination
                    groups[i].remove(worst_member)
                    groups[best_destination].append(worst_member)

                    operations.append(
                        self._log_ejection(
                            worst_member, i, best_destination, best_delta_F, forced=True
                        )
                    )

                    ejection_success = True
                else:
                    # No valid ejection possible
                    break

                group_size = self._compute_size(groups[i], doc_metrics)

            # Escalation: try internal 2-way split if ejections insufficient
            if group_size > constraints.max_size and not ejection_success:
                split_success = self._attempt_group_split(
                    i, groups, S, docs, constraints, doc_metrics, operations
                )

                if split_success:
                    # Split created new group, increment counter to skip it
                    i += 1

            i += 1

        # Rebuild caches after all changes
        caches = self._initialize_caches(groups, S, docs, doc_metrics)

        return groups, caches, operations

    def _attempt_group_split(
        self,
        group_idx: int,
        groups: List[List[str]],
        S: np.ndarray,
        docs: List[str],
        constraints: ClusteringConstraints,
        doc_metrics: Optional[Dict[str, int]],
        operations: List[Dict],
    ) -> bool:
        """
        Attempt to split an oversized group using 2-way split.

        Returns True if split was successful and beneficial.
        """
        group = groups[group_idx]
        if len(group) < 2:
            return False

        doc_to_idx = {doc: i for i, doc in enumerate(docs)}

        # Get group indices in similarity matrix
        group_indices = [doc_to_idx[doc] for doc in group if doc in doc_to_idx]
        if len(group_indices) < 2:
            return False

        # Extract submatrix for this group
        group_S = S[np.ix_(group_indices, group_indices)]

        try:
            # Simple 2-way split: use spectral bisection on group submatrix
            from sklearn.cluster import KMeans

            # Spectral embedding for bisection
            from sklearn.manifold import SpectralEmbedding

            embedder = SpectralEmbedding(
                n_components=1, affinity="precomputed", random_state=self.random_state
            )
            Y = embedder.fit_transform(np.clip(group_S, 0, None))

            # K-means with k=2
            kmeans = KMeans(n_clusters=2, n_init="auto", random_state=self.random_state)
            split_labels = kmeans.fit_predict(Y)

            # Convert back to document groups
            group1 = [group[i] for i, label in enumerate(split_labels) if label == 0]
            group2 = [group[i] for i, label in enumerate(split_labels) if label == 1]

        except Exception:
            # Fallback: simple half split
            mid = len(group) // 2
            group1 = group[:mid]
            group2 = group[mid:]

        # Check if split is beneficial and feasible
        size1 = self._compute_size(group1, doc_metrics)
        size2 = self._compute_size(group2, doc_metrics)

        # Check size constraints
        if size1 < constraints.min_size or size2 < constraints.min_size:
            return False

        # Check if split improves objective: W(G1) + W(G2) > W(G)
        current_score = self._compute_group_score(group, S, docs)
        score1 = self._compute_group_score(group1, S, docs)
        score2 = self._compute_group_score(group2, S, docs)
        new_score = score1 + score2

        delta_F = new_score - current_score

        if delta_F > 0:
            # Beneficial split
            operations.append(
                self._log_split(
                    group_idx, len(group), [group1, group2], delta_F=delta_F
                )
            )

            # Replace original group with split results
            groups[group_idx] = group1
            groups.insert(group_idx + 1, group2)

            return True

        return False

    def _border_move_sweep(
        self,
        groups: List[List[str]],
        S: np.ndarray,
        docs: List[str],
        constraints: ClusteringConstraints,
        doc_metrics: Optional[Dict[str, int]],
        caches: Dict,
    ) -> Tuple[List[List[str]], Dict, List[Dict]]:
        """
        Step 4.3: Border-move sweep for local optimization.

        For each document, consider moving to its best alternative group
        if it improves the global objective F.
        """
        operations = []
        groups = [group.copy() for group in groups]
        doc_to_idx = caches["doc_to_idx"]

        made_changes = True
        sweep_count = 0

        while (
            made_changes and sweep_count < 10
        ):  # Limit sweeps to prevent infinite loops
            made_changes = False
            sweep_count += 1

            for source_idx, source_group in enumerate(groups):
                for (
                    doc
                ) in source_group.copy():  # Copy to avoid modification during iteration
                    best_target = None
                    best_delta_F = 0  # Only accept positive improvements

                    # Try moving to each other group
                    for target_idx, target_group in enumerate(groups):
                        if target_idx == source_idx:
                            continue

                        # Check size constraints
                        source_size = self._compute_size(source_group, doc_metrics)
                        target_size = self._compute_size(target_group, doc_metrics)
                        doc_size = doc_metrics.get(doc, 1) if doc_metrics else 1

                        if (
                            source_size - doc_size < constraints.min_size
                            or target_size + doc_size > constraints.max_size
                        ):
                            continue

                        # Compute ΔF_move = -a_G(v) + Σ_{j∈H} S_{vj}
                        delta_F = 0.0

                        if doc in doc_to_idx:
                            # Remove from source: -a_G(v)
                            for other_doc in source_group:
                                if other_doc != doc and other_doc in doc_to_idx:
                                    delta_F -= S[doc_to_idx[doc], doc_to_idx[other_doc]]

                            # Add to target: +Σ_{j∈H} S_{vj}
                            for target_doc in target_group:
                                if target_doc in doc_to_idx:
                                    delta_F += S[
                                        doc_to_idx[doc], doc_to_idx[target_doc]
                                    ]

                        if delta_F > best_delta_F:
                            best_delta_F = delta_F
                            best_target = target_idx

                    # Perform best move if found
                    if best_target is not None:
                        operations.append(
                            self._log_document_move(
                                doc, source_idx, best_target, best_delta_F
                            )
                        )

                        # Update groups
                        groups[source_idx].remove(doc)
                        groups[best_target].append(doc)

                        made_changes = True
                        break  # One move per iteration for stability

                if made_changes:
                    break  # Restart outer loop

        # Update caches after all moves
        caches = self._initialize_caches(groups, S, docs, doc_metrics)

        return groups, caches, operations

    def _perform_swaps(
        self,
        groups: List[List[str]],
        S: np.ndarray,
        docs: List[str],
        constraints: ClusteringConstraints,
        doc_metrics: Optional[Dict[str, int]],
        caches: Dict,
    ) -> Tuple[List[List[str]], Dict, List[Dict]]:
        """
        Step 4.4: Optional pairwise document swaps.

        For efficiency, only try swaps between adjacent groups.
        """
        if not self.enable_swaps:
            return groups, caches, []

        operations = []
        groups = [group.copy() for group in groups]
        doc_to_idx = caches["doc_to_idx"]

        # Try swaps between adjacent groups
        for i in range(len(groups) - 1):
            group_a = groups[i]
            group_b = groups[i + 1]

            best_swap = None
            best_delta_F = 0

            for doc_a in group_a:
                for doc_b in group_b:
                    # Check size constraints after swap
                    size_a = self._compute_size(group_a, doc_metrics)
                    size_b = self._compute_size(group_b, doc_metrics)
                    doc_a_size = doc_metrics.get(doc_a, 1) if doc_metrics else 1
                    doc_b_size = doc_metrics.get(doc_b, 1) if doc_metrics else 1

                    new_size_a = size_a - doc_a_size + doc_b_size
                    new_size_b = size_b - doc_b_size + doc_a_size

                    if (
                        new_size_a < constraints.min_size
                        or new_size_a > constraints.max_size
                        or new_size_b < constraints.min_size
                        or new_size_b > constraints.max_size
                    ):
                        continue

                    # Compute swap gain (efficient formula from spec)
                    delta_F = 0.0

                    if doc_a in doc_to_idx and doc_b in doc_to_idx:
                        # Change for group A: lose doc_a, gain doc_b
                        for other_doc in group_a:
                            if (
                                other_doc not in (doc_a, doc_b)
                                and other_doc in doc_to_idx
                            ):
                                delta_F -= S[
                                    doc_to_idx[doc_a], doc_to_idx[other_doc]
                                ]  # Lose a-other
                                delta_F += S[
                                    doc_to_idx[doc_b], doc_to_idx[other_doc]
                                ]  # Gain b-other

                        # Change for group B: lose doc_b, gain doc_a
                        for other_doc in group_b:
                            if (
                                other_doc not in (doc_a, doc_b)
                                and other_doc in doc_to_idx
                            ):
                                delta_F -= S[
                                    doc_to_idx[doc_b], doc_to_idx[other_doc]
                                ]  # Lose b-other
                                delta_F += S[
                                    doc_to_idx[doc_a], doc_to_idx[other_doc]
                                ]  # Gain a-other

                    if delta_F > best_delta_F:
                        best_delta_F = delta_F
                        best_swap = (doc_a, doc_b)

            # Perform best swap if found
            if best_swap is not None:
                doc_a, doc_b = best_swap

                operations.append(
                    self._log_document_swap(doc_a, doc_b, i, i + 1, best_delta_F)
                )

                # Perform swap
                groups[i].remove(doc_a)
                groups[i].append(doc_b)
                groups[i + 1].remove(doc_b)
                groups[i + 1].append(doc_a)

        # Update caches after swaps
        if operations:
            caches = self._initialize_caches(groups, S, docs, doc_metrics)

        return groups, caches, operations


class RandomClusterer(DocumentClusterer):
    """
    Random baseline clusterer that produces groups with varied sizes.

    Creates random groupings while respecting size constraints, with
    bias toward creating a mixture of different group sizes.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize random clusterer.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state

    def cluster(
        self,
        embeddings: Dict[str, np.ndarray],
        constraints: ClusteringConstraints,
        doc_metrics: Optional[Dict[str, float]] = None,
        status: Optional["StatusProtocol"] = None,
    ) -> ClusteringResult:
        """
        Create random groups with varied sizes within constraints.

        Args:
            embeddings: Dict mapping document ID to embedding vector
            constraints: Size and type constraints for groups
            doc_metrics: Dict mapping document ID to metric value (count or words)
            status: Optional status object for progress reporting

        Returns:
            ClusteringResult with random groups and basic metrics
        """
        if status:
            status.update("Creating random baseline groups...")

        docs = list(embeddings.keys())
        if not docs:
            return ClusteringResult(groups=[], metrics={})

        # Set up document metrics
        if doc_metrics is None:
            doc_metrics = {doc: 1 for doc in docs}

        # Initialize random state
        rng = np.random.RandomState(self.random_state)

        # Shuffle documents for randomness
        shuffled_docs = docs.copy()
        rng.shuffle(shuffled_docs)

        groups = []
        remaining_docs = shuffled_docs.copy()

        # Create groups with varied sizes
        while remaining_docs:
            if status:
                status.update(f"Random grouping: {len(remaining_docs)} docs remaining")

            # Choose target size with bias toward variety
            available_sizes = list(
                range(
                    constraints.min_size,
                    min(constraints.max_size, len(remaining_docs)) + 1,
                )
            )

            if not available_sizes:
                # Handle case where remaining docs < min_size
                break

            # Bias toward mid-range sizes for variety
            if len(available_sizes) > 1:
                weights = self._compute_size_bias_weights(available_sizes)
                target_size = rng.choice(available_sizes, p=weights)
            else:
                target_size = available_sizes[0]

            # Select documents for this group
            group_docs = []
            group_metric = 0

            for doc in remaining_docs[:]:
                if len(group_docs) >= target_size:
                    break

                doc_value = doc_metrics[doc]

                # Check if adding this doc would violate constraints
                if constraints.constraint_type == "words":
                    if group_metric + doc_value > constraints.max_size:
                        continue
                elif constraints.constraint_type == "count":
                    if len(group_docs) >= constraints.max_size:
                        break

                group_docs.append(doc)
                group_metric += doc_value
                remaining_docs.remove(doc)

            # Only add group if it meets minimum size
            if self._meets_min_constraints(group_docs, group_metric, constraints):
                groups.append(group_docs)
            else:
                # Return docs to remaining if group too small
                remaining_docs.extend(group_docs)
                break

        # Handle remaining documents by merging with existing groups
        if remaining_docs:
            groups = self._merge_remaining_docs(
                remaining_docs, groups, doc_metrics, constraints, rng
            )

        # Compute basic metrics
        metrics = self._compute_basic_metrics(
            groups, embeddings, doc_metrics, constraints
        )

        return ClusteringResult(groups=groups, metrics=metrics)

    def _compute_size_bias_weights(self, available_sizes: List[int]) -> np.ndarray:
        """
        Compute bias weights favoring mid-range sizes for variety.

        Args:
            available_sizes: List of possible group sizes

        Returns:
            Probability weights for each size
        """
        if len(available_sizes) == 1:
            return np.array([1.0])

        # Create inverted-U shape bias (prefer middle sizes)
        weights = []
        for size in available_sizes:
            # Distance from middle, normalized
            mid_point = (available_sizes[0] + available_sizes[-1]) / 2
            distance = abs(size - mid_point) / (
                available_sizes[-1] - available_sizes[0] + 1e-8
            )
            # Inverted distance (closer to middle = higher weight)
            weight = 1.0 - distance
            weights.append(weight)

        weights = np.array(weights)
        return weights / np.sum(weights)

    def _meets_min_constraints(
        self,
        group_docs: List[str],
        group_metric: float,
        constraints: ClusteringConstraints,
    ) -> bool:
        """Check if group meets minimum size constraints."""
        if constraints.constraint_type == "words":
            return group_metric >= constraints.min_size
        else:
            return len(group_docs) >= constraints.min_size

    def _merge_remaining_docs(
        self,
        remaining_docs: List[str],
        groups: List[List[str]],
        doc_metrics: Dict[str, float],
        constraints: ClusteringConstraints,
        rng: np.random.RandomState,
    ) -> List[List[str]]:
        """
        Merge remaining documents into existing groups without violating constraints.

        Args:
            remaining_docs: Documents that couldn't form their own group
            groups: Existing groups to potentially merge into
            doc_metrics: Document metric values
            constraints: Size constraints
            rng: Random number generator

        Returns:
            Updated groups with remaining documents merged
        """
        for doc in remaining_docs:
            doc_value = doc_metrics[doc]

            # Find groups that can accommodate this document
            eligible_groups = []
            for i, group in enumerate(groups):
                current_metric = sum(doc_metrics[d] for d in group)

                if constraints.constraint_type == "words":
                    if current_metric + doc_value <= constraints.max_size:
                        eligible_groups.append(i)
                else:  # count
                    if len(group) < constraints.max_size:
                        eligible_groups.append(i)

            # Randomly select from eligible groups
            if eligible_groups:
                chosen_group = rng.choice(eligible_groups)
                groups[chosen_group].append(doc)

        return groups

    def _compute_basic_metrics(
        self,
        groups: List[List[str]],
        embeddings: Dict[str, np.ndarray],
        doc_metrics: Dict[str, float],
        constraints: ClusteringConstraints,
    ) -> Dict[str, any]:
        """Compute basic metrics for random clustering result."""
        if not groups:
            return {}

        # Group sizes
        group_sizes = [len(group) for group in groups]

        # Group metric totals
        group_totals = []
        for group in groups:
            total = sum(doc_metrics.get(doc, 0) for doc in group)
            group_totals.append(total)

        # Basic statistics
        metrics = {
            "num_groups": len(groups),
            "group_sizes": group_sizes,
            "group_totals": group_totals,
            "avg_group_size": np.mean(group_sizes),
            "group_size_std": np.std(group_sizes),
            "group_size_histogram": {
                size: group_sizes.count(size) for size in set(group_sizes)
            },
            "constraint_type": constraints.constraint_type,
            "clustering_method": "random",
        }

        # Compute cohesion if embeddings available
        cohesions = []
        for group in groups:
            if len(group) < 2:
                cohesions.append(1.0)
                continue

            similarities = []
            for i, doc_a in enumerate(group):
                for doc_b in group[i + 1 :]:
                    if doc_a in embeddings and doc_b in embeddings:
                        sim = np.dot(embeddings[doc_a], embeddings[doc_b])
                        similarities.append(sim)

            if similarities:
                cohesions.append(float(np.mean(similarities)))
            else:
                cohesions.append(0.0)

        if cohesions:
            metrics.update(
                {
                    "cohesions": cohesions,
                    "avg_cohesion": np.mean(cohesions),
                    "cohesion_std": np.std(cohesions),
                }
            )

        return metrics


@dataclass
class ClusterCache:
    """Efficient cache for cluster statistics during agglomeration."""

    size: Dict[int, int] = field(default_factory=dict)
    W: Dict[int, float] = field(default_factory=dict)  # Within-cluster sum
    mu: Dict[int, float] = field(default_factory=dict)  # Mean within-cluster similarity
    R: Dict[Tuple[int, int], float] = field(default_factory=dict)  # Cross-sums
    members: Dict[int, Set[int]] = field(default_factory=dict)  # Document indices
    similarity_cache: Dict[Tuple[int, int], float] = field(
        default_factory=dict
    )  # Cached inter-cluster similarities


class SizeAnnealedAgglomerativeClusterer(DocumentClusterer):
    """
    Parameter-free size-annealed agglomerative clustering.

    Implements capacity-aware agglomeration with automatic annealing based on
    cohesion preservation. No manual thresholds required - merges are accepted
    based on maintaining cluster cohesion and positive objective gain.
    """

    def __init__(
        self,
        enable_border_moves: bool = False,
        enable_swaps: bool = False,
    ):
        """
        Initialize size-annealed agglomerative clusterer.

        Uses parameter-free cross-pair threshold for merge acceptance.

        Args:
            enable_border_moves: Whether to perform border move optimization
            enable_swaps: Whether to perform document swaps
        """
        self.enable_border_moves = enable_border_moves
        self.enable_swaps = enable_swaps

    def cluster(
        self,
        embeddings: Dict[str, np.ndarray],
        constraints: ClusteringConstraints,
        doc_metrics: Optional[Dict[str, int]] = None,
        status: Optional["StatusProtocol"] = None,
    ) -> ClusteringResult:
        """Perform size-annealed agglomerative clustering."""
        if status:
            status.update("Building similarity matrix...")

        # 1. Build similarity matrix
        S, docs = self._build_similarity_matrix(embeddings)
        n = len(docs)

        if not docs:
            return ClusteringResult(groups=[], metrics={})

        if n == 1:
            return ClusteringResult(
                groups=[docs], metrics=self._compute_standard_metrics([docs], S, docs)
            )

        # 2. Check packability
        if not self._check_packability(n, constraints):
            raise ValueError(
                f"Constraints are not satisfiable: ⌈{n}/{constraints.max_size}⌉ > "
                f"⌊{n}/{constraints.min_size}⌋"
            )

        if status:
            status.update("Initializing cluster caches...")

        # 3. Initialize caches
        cache = self._initialize_caches(S, docs)
        operation_log = []

        # Compute global mean similarity for baseline floor
        total_similarity = np.sum(S)
        global_mean_similarity = float(total_similarity / (n * (n - 1)) if n > 1 else 0)

        # Log initialization with similarity statistics
        sim_stats = self._compute_similarity_stats(S, docs)
        operation_log.append(
            self._log_base(
                "initialization",
                total_docs=n,
                total_similarity=float(total_similarity),
                avg_similarity=global_mean_similarity,
                similarity_stats=sim_stats,
                constraints={
                    "min_size": constraints.min_size,
                    "max_size": constraints.max_size,
                    "type": constraints.constraint_type,
                },
            )
        )

        if status:
            status.update("Building merge heap...")

        # 4. Build initial heap
        heap = []
        for i in range(n):
            for j in range(i + 1, n):
                # Only check size constraint for heap building
                # All other checks (R≥0, cross-pair threshold) happen at merge time
                if cache.size[i] + cache.size[j] <= constraints.max_size:
                    priority = self._compute_merge_priority(cache, i, j)
                    heapq.heappush(heap, priority + (i, j))

        if status:
            status.update("Performing agglomerative clustering...")

        # 5. Main clustering loop - continues until heap empty and no stalls can be resolved
        iteration = 0
        merges_performed = 0
        baseline_violations = 0  # Count merges with m < S̄ (should be 0 after fix)

        while True:
            # Normal agglomeration phase
            while heap:
                iteration += 1

                # Update status more frequently
                if status and (iteration <= 5 or iteration % 5 == 0):
                    active_clusters = len(cache.size)
                    acceptably_sized = sum(
                        1 for c in cache.size if cache.size[c] >= constraints.min_size
                    )
                    status.update(
                        f"Clustering agglomeratively ([green]{acceptably_sized}[/green]/[blue]{active_clusters}[/blue] acceptably sized)"
                    )

                # Try normal merge
                merge_performed = False
                while heap and not merge_performed:
                    entry = heapq.heappop(heap)
                    A, B = entry[-2], entry[-1]

                    # Check if clusters still exist (lazy deletion)
                    if A not in cache.size or B not in cache.size:
                        continue

                    # Check acceptance
                    if self._accept_merge(
                        cache, A, B, constraints, global_mean_similarity
                    ):
                        # Compute detailed metrics for logging
                        a, b = cache.size[A], cache.size[B]
                        s = a + b
                        R_AB = cache.R.get((A, B), 0)
                        m = R_AB / (a * b) if a * b > 0 else 0.0

                        mu_A = cache.mu[A] if cache.size[A] >= 2 else 0.0
                        mu_B = cache.mu[B] if cache.size[B] >= 2 else 0.0
                        mu_min = min(mu_A, mu_B)
                        mu_max = max(mu_A, mu_B)

                        f = (2 * a * b) / (s * (s - 1)) if s >= 2 else 0.0
                        threshold = mu_min + f * (mu_max - mu_min)
                        delta = m - threshold

                        # Track baseline violations (should be zero after fix)
                        merges_performed += 1
                        if m < global_mean_similarity:
                            baseline_violations += 1

                        # Log merge with detailed metrics
                        operation_log.append(
                            self._log_base(
                                "merge",
                                iteration=iteration,
                                clusters=[A, B],
                                sizes=[a, b],
                                merged_size=s,
                                mu_A=mu_A,
                                mu_B=mu_B,
                                cross_mean=m,
                                f_value=f,
                                threshold=threshold,
                                margin=delta,
                                R=R_AB,
                                baseline=global_mean_similarity,
                                below_baseline=m < global_mean_similarity,
                            )
                        )

                        # Perform merge (merge B into A)
                        self._update_caches_after_merge(cache, A, B, S)

                        # Incremental heap update for merged cluster
                        self._incremental_heap_update(heap, cache, A, constraints)

                        merge_performed = True

            # Heap is empty - check for stalls
            if any(cache.size[c] < constraints.min_size for c in cache.size):
                if status:
                    status.update("Resolving stall...")

                if self._handle_stall(cache, constraints, S, operation_log, iteration):
                    # Rebuild heap after stall resolution - only check size constraint
                    for A in cache.size:
                        for B in cache.size:
                            if (
                                A < B
                                and cache.size[A] + cache.size[B]
                                <= constraints.max_size
                            ):
                                priority = self._compute_merge_priority(cache, A, B)
                                heapq.heappush(heap, priority + (A, B))
                    iteration += 1
                    # Continue to next iteration of main loop
                    continue
                else:
                    # No stall resolution possible
                    operation_log.append(
                        self._log_base(
                            "termination",
                            iteration=iteration,
                            reason="no_valid_merges_for_undersized",
                            active_clusters=len(cache.size),
                            undersized_clusters=sum(
                                1
                                for c in cache.size
                                if cache.size[c] < constraints.min_size
                            ),
                        )
                    )
                    break
            else:
                # All clusters acceptably sized
                break

        if status:
            status.update("Converting to final groups...")

        # 6. Convert cache to groups
        groups = []
        undersized_docs = []
        for cluster_id, members in cache.members.items():
            group = [docs[i] for i in sorted(members)]
            if cache.size[cluster_id] >= constraints.min_size:
                groups.append(group)
            else:
                # Collect undersized docs to merge with smallest acceptable group
                operation_log.append(
                    self._log_base(
                        "undersized_final",
                        cluster_id=cluster_id,
                        size=cache.size[cluster_id],
                        members=group,
                    )
                )
                undersized_docs.extend(group)

        # Merge undersized docs into smallest group that can accommodate them
        if undersized_docs and groups:
            # Find the smallest group that won't exceed max_size after merge
            best_group_idx = None
            min_size_after_merge = float("inf")

            for i, group in enumerate(groups):
                size_after_merge = len(group) + len(undersized_docs)
                if (
                    size_after_merge <= constraints.max_size
                    and size_after_merge < min_size_after_merge
                ):
                    best_group_idx = i
                    min_size_after_merge = size_after_merge

            if best_group_idx is not None:
                groups[best_group_idx].extend(undersized_docs)
                operation_log.append(
                    self._log_base(
                        "force_merge_undersized",
                        undersized_docs=undersized_docs,
                        target_group=best_group_idx,
                        final_size=len(groups[best_group_idx]),
                    )
                )
            else:
                # If no group can accommodate undersized docs, create a new group
                # This violates min_size but is better than violating max_size
                groups.append(undersized_docs)
                operation_log.append(
                    self._log_base(
                        "create_undersized_group",
                        undersized_docs=undersized_docs,
                        final_size=len(undersized_docs),
                        reason="no_group_can_accommodate_without_exceeding_max_size",
                    )
                )

        # 7. Optional border moves
        if self.enable_border_moves and len(groups) > 1:
            if status:
                status.update("Performing border move optimization...")
            groups = self._perform_border_moves(
                groups, S, docs, constraints, doc_metrics, operation_log
            )

        # 8. Optional swaps
        if self.enable_swaps and len(groups) > 1:
            if status:
                status.update("Performing document swaps...")
            groups = self._perform_document_swaps(
                groups, S, docs, constraints, doc_metrics, operation_log
            )

        # 9. Compute final metrics
        if status:
            status.update("Computing final metrics...")

        metrics = self._compute_standard_metrics(groups, S, docs)
        metrics["algorithm"] = "size_annealed_agglomerative"
        metrics["iterations"] = iteration
        metrics["final_groups"] = len(groups)
        metrics["merges_performed"] = merges_performed
        metrics["baseline_violations"] = baseline_violations

        # Add diagnostic summary to operation log
        operation_log.append(
            self._log_base(
                "diagnostic_summary",
                merges_performed=merges_performed,
                baseline_violations=baseline_violations,
                baseline_violation_rate=baseline_violations / max(1, merges_performed),
                global_baseline=global_mean_similarity,
            )
        )

        return ClusteringResult(
            groups=groups, metrics=metrics, operation_log=operation_log
        )

    def _check_packability(self, n: int, constraints: ClusteringConstraints) -> bool:
        """Check if constraints are satisfiable."""
        min_groups = np.ceil(n / constraints.max_size)
        max_groups = np.floor(n / constraints.min_size)
        return min_groups <= max_groups

    def _initialize_caches(self, S: np.ndarray, docs: List[str]) -> ClusterCache:
        """Initialize all caches from similarity matrix."""
        n = len(docs)
        cache = ClusterCache()

        # Initialize singletons
        for i in range(n):
            cache.size[i] = 1
            cache.W[i] = 0.0  # No within-cluster sum for singleton
            cache.mu[i] = 0.0  # Undefined for singleton
            cache.members[i] = {i}

        # Initialize all cross-sums R
        for i in range(n):
            for j in range(i + 1, n):
                cache.R[(i, j)] = S[i, j]
                cache.R[(j, i)] = S[i, j]  # Symmetric

        return cache

    def _update_caches_after_merge(
        self, cache: ClusterCache, A: int, B: int, S: np.ndarray
    ) -> None:
        """Update caches after merging B into A using vectorized operations."""
        # Update size
        cache.size[A] += cache.size[B]

        # Update within-cluster sum
        R_AB = cache.R.get((A, B), 0)
        cache.W[A] += cache.W[B] + R_AB

        # Update mean
        size_A = cache.size[A]
        if size_A > 1:
            cache.mu[A] = 2 * cache.W[A] / (size_A * (size_A - 1))
        else:
            cache.mu[A] = 0.0

        # Update members
        cache.members[A] |= cache.members[B]

        # Vectorized update of cross-sums with all other clusters
        other_clusters = [C for C in cache.size.keys() if C != A and C != B]
        if other_clusters:
            # Get member indices for vectorized computation
            members_A = np.array(list(cache.members[A]))

            # Update R values for all other clusters at once
            for C in other_clusters:
                members_C = np.array(list(cache.members[C]))
                if len(members_C) > 0:
                    # Vectorized computation of cross-cluster sum
                    R_AC = np.sum(S[np.ix_(members_A, members_C)])
                    cache.R[(A, C)] = R_AC
                    cache.R[(C, A)] = R_AC  # Maintain symmetry

        # Remove B from all caches
        del cache.size[B]
        del cache.W[B]
        del cache.mu[B]
        del cache.members[B]

        # Remove R entries involving B
        keys_to_remove = [k for k in cache.R if B in k]
        for k in keys_to_remove:
            del cache.R[k]

    def _incremental_heap_update(
        self,
        heap: List,
        cache: ClusterCache,
        merged_cluster: int,
        constraints: ClusteringConstraints,
    ) -> None:
        """Incrementally update heap with new pairs involving merged cluster."""
        for C in cache.size:
            if (
                C != merged_cluster
                and cache.size[merged_cluster] + cache.size[C] <= constraints.max_size
            ):
                # Check if we have cached similarity
                cache_key = tuple(sorted([merged_cluster, C]))
                if cache_key not in cache.similarity_cache:
                    # Compute and cache the similarity
                    priority = self._compute_merge_priority(cache, merged_cluster, C)
                    cache.similarity_cache[cache_key] = priority[1]  # Store -m value
                else:
                    # Use cached similarity to rebuild priority
                    priority = self._compute_merge_priority(cache, merged_cluster, C)

                heapq.heappush(heap, priority + (merged_cluster, C))

    def _accept_merge(
        self,
        cache: ClusterCache,
        A: int,
        B: int,
        constraints: ClusteringConstraints,
        global_baseline: float,
    ) -> bool:
        """
        Check if merge should be accepted using cross-pair threshold with global baseline.

        Acceptance criteria:
        1. Size cap: s <= U
        2. Objective safety: R >= 0
        3. Cross-pair threshold with baseline: m >= max(μ_min + f(a,b)*(μ_max - μ_min), S̄)
        """
        a, b = cache.size[A], cache.size[B]
        s = a + b

        # 1. Size cap
        if s > constraints.max_size:
            return False

        # 2. Objective safety: R >= 0 (guards global objective F)
        R_AB = cache.R.get((A, B), 0)
        if R_AB < 0:
            return False

        # Cross-mean m = R/(a*b)
        m = R_AB / (a * b) if a * b > 0 else 0.0

        # 3. Baseline floor: m >= S̄ (prevent below-average merges)
        if m < global_baseline:
            return False

        # 4. Cross-pair threshold with global baseline floor
        # Handle singletons: μ({i}) = 0
        mu_A = cache.mu[A] if cache.size[A] >= 2 else 0.0
        mu_B = cache.mu[B] if cache.size[B] >= 2 else 0.0

        mu_min = min(mu_A, mu_B)
        mu_max = max(mu_A, mu_B)

        # f(a,b) = 2ab/s(s-1) - fraction of new pairs introduced
        f = (2 * a * b) / (s * (s - 1)) if s >= 2 else 0.0

        # Dynamic cross-pair threshold: m >= μ_min + f*(μ_max - μ_min)
        dynamic_threshold = mu_min + f * (mu_max - mu_min)

        return m >= dynamic_threshold

    def _compute_merge_priority(
        self, cache: ClusterCache, A: int, B: int
    ) -> Tuple[float, float, float, float, int]:
        """
        Compute heap priority using post-merge cohesion.

        Returns tuple for min-heap: (-μ_merge, -m, -s, -R, min(idA,idB))
        where μ_merge is the mean within-similarity of the merged cluster.
        """
        a, b = cache.size[A], cache.size[B]
        s = a + b
        R_AB = cache.R.get((A, B), 0)

        # Cross-mean m = R/(a*b)
        m = R_AB / (a * b) if a * b > 0 else 0.0

        # Compute post-merge cohesion μ_merge = μ(A∪B)
        W_merged = cache.W[A] + cache.W[B] + R_AB
        mu_merge = 2 * W_merged / (s * (s - 1)) if s > 1 else 0.0

        # Heap key: (-μ_merge, -m, -s, -R, min(idA,idB))
        # Negatives for max-heap behavior in min-heap
        return (-mu_merge, -m, -float(s), -R_AB, min(A, B))

    def _handle_stall(
        self,
        cache: ClusterCache,
        constraints: ClusteringConstraints,
        S: np.ndarray,
        operation_log: List[Dict],
        iteration: int,
    ) -> bool:
        """Handle stall condition: find smallest undersized cluster and best partner."""
        # Find smallest undersized cluster
        undersized = [
            (cache.size[c], c)
            for c in cache.size
            if cache.size[c] < constraints.min_size
        ]

        if not undersized:
            return False

        undersized.sort()  # Sort by size
        _, G = undersized[0]

        # Find H* that maximizes μ(G∪H) with specified tie-breaking
        best_H = None
        best_mu_merged = -float("inf")
        best_R = -float("inf")
        best_size_distance = float("inf")
        target_size = (constraints.min_size + constraints.max_size) / 2

        for H in cache.size:
            if H == G:
                continue

            merged_size = cache.size[G] + cache.size[H]
            if merged_size > constraints.max_size:
                continue

            R_GH = cache.R.get((G, H), 0)
            if R_GH < 0:
                continue

            # Compute μ(G∪H)
            W_merged = cache.W[G] + cache.W[H] + R_GH
            mu_merged = (
                2 * W_merged / (merged_size * (merged_size - 1))
                if merged_size > 1
                else 0
            )

            size_distance = abs(merged_size - target_size)

            # Tie-breaking: max μ(G∪H), then larger R, then size closer to (L+U)/2, then IDs
            if (
                mu_merged > best_mu_merged
                or (mu_merged == best_mu_merged and R_GH > best_R)
                or (
                    mu_merged == best_mu_merged
                    and R_GH == best_R
                    and size_distance < best_size_distance
                )
                or (
                    mu_merged == best_mu_merged
                    and R_GH == best_R
                    and size_distance == best_size_distance
                    and H < best_H
                )
            ):
                best_mu_merged = mu_merged
                best_R = R_GH
                best_size_distance = size_distance
                best_H = H

        if best_H is None:
            return False

        # Log and perform merge
        operation_log.append(
            self._log_base(
                "stall_resolution",
                iteration=iteration,
                undersized_cluster=G,
                partner=best_H,
                mu_merged=best_mu_merged,
                R=best_R,
            )
        )

        self._update_caches_after_merge(cache, G, best_H, S)
        return True

    def _perform_border_moves(
        self,
        groups: List[List[str]],
        S: np.ndarray,
        docs: List[str],
        constraints: ClusteringConstraints,
        doc_metrics: Optional[Dict[str, int]],
        operation_log: List[Dict],
    ) -> List[List[str]]:
        """
        Perform border move optimization.
        Move documents between groups if it improves objective.
        """
        doc_to_idx = {doc: i for i, doc in enumerate(docs)}
        improved = True
        sweep = 0

        while improved and sweep < 10:
            improved = False
            sweep += 1

            for g_idx, source_group in enumerate(groups):
                for doc in source_group[
                    :
                ]:  # Copy to avoid modification during iteration
                    if len(source_group) <= constraints.min_size:
                        continue

                    doc_idx = doc_to_idx[doc]
                    best_target = None
                    best_delta_F = 0.0  # Only accept positive improvements

                    for h_idx, target_group in enumerate(groups):
                        if g_idx == h_idx:
                            continue

                        target_size = self._compute_size(target_group, doc_metrics)
                        if target_size >= constraints.max_size:
                            continue

                        # Compute delta F
                        delta_F = 0.0

                        # Loss from source
                        for other_doc in source_group:
                            if other_doc != doc:
                                other_idx = doc_to_idx[other_doc]
                                delta_F -= S[doc_idx, other_idx]

                        # Gain to target
                        for target_doc in target_group:
                            target_doc_idx = doc_to_idx[target_doc]
                            delta_F += S[doc_idx, target_doc_idx]

                        if delta_F > best_delta_F:
                            best_delta_F = delta_F
                            best_target = h_idx

                    if best_target is not None:
                        # Perform move
                        source_group.remove(doc)
                        groups[best_target].append(doc)

                        operation_log.append(
                            self._log_base(
                                "border_move",
                                sweep=sweep,
                                document=doc,
                                source_group=g_idx,
                                target_group=best_target,
                                delta_F=best_delta_F,
                            )
                        )

                        improved = True
                        break

                if improved:
                    break

        return groups

    def _perform_document_swaps(
        self,
        groups: List[List[str]],
        S: np.ndarray,
        docs: List[str],
        constraints: ClusteringConstraints,
        doc_metrics: Optional[Dict[str, int]],
        operation_log: List[Dict],
    ) -> List[List[str]]:
        """
        Perform document swaps between groups.
        Swap pairs of documents if it improves objective.
        """
        doc_to_idx = {doc: i for i, doc in enumerate(docs)}
        improved = True
        sweep = 0

        while improved and sweep < 5:
            improved = False
            sweep += 1

            for g_idx in range(len(groups) - 1):
                group_a = groups[g_idx]
                group_b = groups[g_idx + 1]

                best_swap = None
                best_delta_F = 0.0

                for doc_a in group_a:
                    for doc_b in group_b:
                        # Compute delta F for swap
                        doc_a_idx = doc_to_idx[doc_a]
                        doc_b_idx = doc_to_idx[doc_b]
                        delta_F = 0.0

                        # Changes in group A: lose doc_a, gain doc_b
                        for other_doc in group_a:
                            if other_doc != doc_a:
                                other_idx = doc_to_idx[other_doc]
                                delta_F -= S[doc_a_idx, other_idx]
                                delta_F += S[doc_b_idx, other_idx]

                        # Changes in group B: lose doc_b, gain doc_a
                        for other_doc in group_b:
                            if other_doc != doc_b:
                                other_idx = doc_to_idx[other_doc]
                                delta_F -= S[doc_b_idx, other_idx]
                                delta_F += S[doc_a_idx, other_idx]

                        if delta_F > best_delta_F:
                            best_delta_F = delta_F
                            best_swap = (doc_a, doc_b)

                if best_swap is not None:
                    doc_a, doc_b = best_swap

                    # Perform swap
                    group_a.remove(doc_a)
                    group_a.append(doc_b)
                    group_b.remove(doc_b)
                    group_b.append(doc_a)

                    operation_log.append(
                        self._log_base(
                            "document_swap",
                            sweep=sweep,
                            documents=[doc_a, doc_b],
                            groups=[g_idx, g_idx + 1],
                            delta_F=best_delta_F,
                        )
                    )

                    improved = True
                    break

        return groups
