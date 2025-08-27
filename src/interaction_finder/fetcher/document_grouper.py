"""
Document grouping using constrained agglomerative clustering based on chunk embeddings.

This module implements semantic document grouping with size constraints, using
chunk-level embeddings to compute document similarities and constrained
agglomerative clustering to form groups.
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.distance import cosine
import re


@dataclass
class Cluster:
    """Represents a cluster of documents during agglomerative clustering."""

    id: int
    documents: Set[str]
    centroid: Optional[np.ndarray] = None

    def __post_init__(self):
        """Initialize centroid if not provided."""
        if self.centroid is None:
            # Will be computed from document embeddings
            self.centroid = np.zeros(256)  # Default chonkie embedding dimension

    def merge_with(
        self, other: "Cluster", doc_embeddings: Dict[str, np.ndarray]
    ) -> "Cluster":
        """Merge two clusters into a new one."""
        new_docs = self.documents.union(other.documents)

        # Recompute centroid as average of all document centroids
        all_centroids = []
        for doc in new_docs:
            if doc in doc_embeddings:
                all_centroids.append(doc_embeddings[doc])

        if all_centroids:
            new_centroid = np.mean(all_centroids, axis=0)
        else:
            new_centroid = np.zeros_like(self.centroid)

        return Cluster(
            id=max(self.id, other.id) + 1, documents=new_docs, centroid=new_centroid
        )


class DocumentGrouper:
    """Groups documents using constrained agglomerative clustering."""

    def __init__(self, linkage_method: str = "average"):
        """
        Initialize the document grouper.

        Args:
            linkage_method: Linkage method for clustering ("average", "complete", "single")
        """
        self.linkage_method = linkage_method

    def group_documents(
        self,
        documents: List[str],
        chunk_data: Dict[str, List[Dict]],
        constraint_type: str = "count",
        min_size: int = 3,
        max_size: int = 8,
        similarity_threshold: float = 0.3,
    ) -> List[List[str]]:
        """
        Group documents using constrained agglomerative clustering.

        Args:
            documents: List of document URLs/identifiers
            chunk_data: Dict mapping document ID to list of chunk objects
            constraint_type: Either "count" (document count) or "words" (word count)
            min_size: Minimum group size (documents or words)
            max_size: Maximum group size (documents or words)
            similarity_threshold: Minimum similarity for merging clusters

        Returns:
            List of document groups, where each group is a list of document IDs
        """
        if not documents:
            return []

        if len(documents) == 1:
            return [documents]

        # Compute document embeddings (average of chunk embeddings)
        doc_embeddings = self._compute_document_embeddings(documents, chunk_data)

        # Compute document metrics for constraints
        if constraint_type == "words":
            doc_metrics = self._compute_word_counts(documents, chunk_data)
        else:
            doc_metrics = {doc: 1 for doc in documents}  # Each doc counts as 1

        # Run constrained agglomerative clustering
        groups = self._constrained_agglomerative_clustering(
            documents=documents,
            doc_embeddings=doc_embeddings,
            doc_metrics=doc_metrics,
            min_size=min_size,
            max_size=max_size,
            similarity_threshold=similarity_threshold,
        )

        # Handle orphaned documents
        groups = self._handle_orphans(
            groups=groups,
            all_documents=set(documents),
            doc_embeddings=doc_embeddings,
            doc_metrics=doc_metrics,
            max_size=max_size,
            similarity_threshold=similarity_threshold
            * 0.7,  # Lower threshold for orphans
        )

        return groups

    def _compute_document_embeddings(
        self, documents: List[str], chunk_data: Dict[str, List[Dict]]
    ) -> Dict[str, np.ndarray]:
        """Compute document-level embeddings as averages of chunk embeddings."""
        doc_embeddings = {}

        for doc in documents:
            chunks = chunk_data.get(doc, [])
            chunk_embeddings = []

            for chunk in chunks:
                if chunk.get("embedding"):
                    chunk_embeddings.append(np.array(chunk["embedding"]))

            if chunk_embeddings:
                # Average all chunk embeddings to get document embedding
                doc_embeddings[doc] = np.mean(chunk_embeddings, axis=0)
            else:
                # Fallback to zero vector if no embeddings
                doc_embeddings[doc] = np.zeros(256)

        return doc_embeddings

    def _compute_word_counts(
        self, documents: List[str], chunk_data: Dict[str, List[Dict]]
    ) -> Dict[str, int]:
        """Compute total word count for each document."""
        word_counts = {}

        for doc in documents:
            chunks = chunk_data.get(doc, [])
            total_words = sum(chunk.get("wordcount", 0) for chunk in chunks)
            word_counts[doc] = total_words

        return word_counts

    def _cluster_distance(
        self,
        cluster_a: Cluster,
        cluster_b: Cluster,
        doc_embeddings: Dict[str, np.ndarray],
    ) -> float:
        """Compute distance between two clusters based on linkage method."""
        if self.linkage_method == "average":
            return self._average_linkage_distance(cluster_a, cluster_b, doc_embeddings)
        elif self.linkage_method == "complete":
            return self._complete_linkage_distance(cluster_a, cluster_b, doc_embeddings)
        elif self.linkage_method == "single":
            return self._single_linkage_distance(cluster_a, cluster_b, doc_embeddings)
        else:
            # Default to average linkage
            return self._average_linkage_distance(cluster_a, cluster_b, doc_embeddings)

    def _average_linkage_distance(
        self,
        cluster_a: Cluster,
        cluster_b: Cluster,
        doc_embeddings: Dict[str, np.ndarray],
    ) -> float:
        """Average linkage: mean of all pairwise distances."""
        distances = []
        for doc_a in cluster_a.documents:
            for doc_b in cluster_b.documents:
                if doc_a in doc_embeddings and doc_b in doc_embeddings:
                    dist = cosine(doc_embeddings[doc_a], doc_embeddings[doc_b])
                    distances.append(dist)

        return np.mean(distances) if distances else 1.0

    def _complete_linkage_distance(
        self,
        cluster_a: Cluster,
        cluster_b: Cluster,
        doc_embeddings: Dict[str, np.ndarray],
    ) -> float:
        """Complete linkage: maximum of all pairwise distances."""
        distances = []
        for doc_a in cluster_a.documents:
            for doc_b in cluster_b.documents:
                if doc_a in doc_embeddings and doc_b in doc_embeddings:
                    dist = cosine(doc_embeddings[doc_a], doc_embeddings[doc_b])
                    distances.append(dist)

        return max(distances) if distances else 1.0

    def _single_linkage_distance(
        self,
        cluster_a: Cluster,
        cluster_b: Cluster,
        doc_embeddings: Dict[str, np.ndarray],
    ) -> float:
        """Single linkage: minimum of all pairwise distances."""
        distances = []
        for doc_a in cluster_a.documents:
            for doc_b in cluster_b.documents:
                if doc_a in doc_embeddings and doc_b in doc_embeddings:
                    dist = cosine(doc_embeddings[doc_a], doc_embeddings[doc_b])
                    distances.append(dist)

        return min(distances) if distances else 1.0

    def _constrained_agglomerative_clustering(
        self,
        documents: List[str],
        doc_embeddings: Dict[str, np.ndarray],
        doc_metrics: Dict[str, int],
        min_size: int,
        max_size: int,
        similarity_threshold: float,
    ) -> List[List[str]]:
        """Run constrained agglomerative clustering."""
        # Initialize each document as its own cluster
        clusters = {
            i: Cluster(i, {doc}, doc_embeddings.get(doc, np.zeros(256)))
            for i, doc in enumerate(documents)
        }

        # Main clustering loop
        while len(clusters) > 1:
            # Find best valid merge
            best_merge = None
            best_distance = float("inf")

            cluster_ids = list(clusters.keys())
            for i, id_a in enumerate(cluster_ids):
                for id_b in cluster_ids[i + 1 :]:
                    c_a, c_b = clusters[id_a], clusters[id_b]

                    # Check size constraint
                    merged_metric = sum(
                        doc_metrics[doc] for doc in c_a.documents.union(c_b.documents)
                    )
                    if merged_metric > max_size:
                        continue

                    # Calculate distance
                    dist = self._cluster_distance(c_a, c_b, doc_embeddings)

                    # Check similarity threshold (convert distance to similarity)
                    similarity = 1.0 - dist
                    if similarity < similarity_threshold:
                        continue

                    if dist < best_distance:
                        best_distance = dist
                        best_merge = (id_a, id_b)

            # If no valid merges, stop
            if best_merge is None:
                break

            # Perform merge
            id_a, id_b = best_merge
            new_cluster = clusters[id_a].merge_with(clusters[id_b], doc_embeddings)

            # Update clusters dict
            del clusters[id_a]
            del clusters[id_b]
            clusters[new_cluster.id] = new_cluster

        # Convert to final format, filtering by min_size
        result = []
        for cluster in clusters.values():
            cluster_metric = sum(doc_metrics[doc] for doc in cluster.documents)
            if cluster_metric >= min_size:
                result.append(list(cluster.documents))

        return result

    def _handle_orphans(
        self,
        groups: List[List[str]],
        all_documents: Set[str],
        doc_embeddings: Dict[str, np.ndarray],
        doc_metrics: Dict[str, int],
        max_size: int,
        similarity_threshold: float,
    ) -> List[List[str]]:
        """Handle orphaned documents by trying to add them to existing groups."""
        # Find orphaned documents
        grouped_docs = set()
        for group in groups:
            grouped_docs.update(group)

        orphans = list(all_documents - grouped_docs)

        if not orphans:
            return groups

        # Try to add each orphan to the best-fitting group
        for orphan in orphans:
            best_group_idx = None
            best_similarity = -1

            for i, group in enumerate(groups):
                # Check if adding orphan would exceed max_size
                group_metric = sum(doc_metrics[doc] for doc in group)
                if group_metric + doc_metrics[orphan] > max_size:
                    continue

                # Compute similarity to group
                if orphan in doc_embeddings:
                    similarities = []
                    for doc in group:
                        if doc in doc_embeddings:
                            sim = 1.0 - cosine(
                                doc_embeddings[orphan], doc_embeddings[doc]
                            )
                            similarities.append(sim)

                    if similarities:
                        avg_similarity = np.mean(similarities)
                        if (
                            avg_similarity > best_similarity
                            and avg_similarity >= similarity_threshold
                        ):
                            best_similarity = avg_similarity
                            best_group_idx = i

            # Add to best group if found
            if best_group_idx is not None:
                groups[best_group_idx].append(orphan)

        return groups

    @staticmethod
    def parse_constraint(constraint: str) -> Tuple[str, int, int]:
        """
        Parse constraint string like 'count:3-8' or 'words:5000-15000'.

        Returns:
            Tuple of (constraint_type, min_val, max_val)
        """
        match = re.match(r"(\w+):(\d+)-(\d+)", constraint)
        if not match:
            raise ValueError(
                f"Invalid constraint format: {constraint}. Expected 'type:min-max'"
            )

        constraint_type, min_val, max_val = match.groups()
        return constraint_type, int(min_val), int(max_val)


def compute_group_cohesion(
    group: List[str], doc_embeddings: Dict[str, np.ndarray]
) -> float:
    """
    Compute cohesion score for a group of documents.

    Args:
        group: List of document IDs
        doc_embeddings: Dict mapping document ID to embedding

    Returns:
        Average pairwise similarity within the group (0-1)
    """
    if len(group) < 2:
        return 1.0

    similarities = []
    for i, doc_a in enumerate(group):
        for doc_b in group[i + 1 :]:
            if doc_a in doc_embeddings and doc_b in doc_embeddings:
                sim = 1.0 - cosine(doc_embeddings[doc_a], doc_embeddings[doc_b])
                similarities.append(sim)

    return np.mean(similarities) if similarities else 0.0
