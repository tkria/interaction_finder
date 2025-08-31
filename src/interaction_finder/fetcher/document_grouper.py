"""
Document grouping using multiple embedding and clustering strategies.

This module provides a unified interface for document clustering with support for
multiple embedding methods (mean averaging, IDF-like) and clustering algorithms
(agglomerative, spectral, hybrid).

This is the new implementation that replaces the monolithic document_grouper.py
"""

from typing import List, Dict, Optional, Callable, Literal, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .progress_display import StatusProtocol
import warnings
from dataclasses import dataclass, field
import numpy as np

# Import the new modular components
from .document_embedding import (
    ChunkData,
    DocumentEmbedder,
    SimpleAverageEmbedder,
    IDFEmbedder,
    convert_legacy_chunk_data,
)
from .document_clustering import (
    ClusteringConstraints,
    ClusteringResult,
    DocumentClusterer,
    AgglomerativeClusterer,
    SpectralClusterer,
    HybridClusterer,
    RandomClusterer,
    SizeAnnealedAgglomerativeClusterer,
)


@dataclass
class GroupingResult:
    """Result of document grouping with metadata (backward compatibility)."""

    groups: List[List[str]]
    metadata: Dict[str, any] = field(default_factory=dict)
    operation_log: List[Dict] = field(default_factory=list)


class DocumentGrouper:
    """
    Unified document grouper with support for multiple embedding and clustering methods.

    This class provides backward compatibility with the original API while using
    the new modular implementation underneath.
    """

    def __init__(
        self,
        linkage_method: str = "average",
        clustering_method: str = "agglomerative",
        embedding_weights: str = "uniform",
        embedding_method: str = None,  # Legacy parameter
        use_specificity_weighting: bool = False,  # Legacy parameter
        seeding_method: str = "kmeans",
        refinement_method: str = "hierarchical",
        **kwargs,
    ):
        """
        Initialize document grouper.

        Args:
            linkage_method: Linkage method for agglomerative clustering
            clustering_method: Clustering algorithm ("agglomerative", "spectral", "hybrid")
            embedding_weights: Weighting strategy for chunk averaging ("uniform", "idf")
            embedding_method: Legacy parameter, maps to embedding_weights
            use_specificity_weighting: Legacy parameter, maps to "idf"
            **kwargs: Additional parameters passed to underlying algorithms
        """
        # Handle legacy parameter mapping
        if use_specificity_weighting:
            embedding_weights = "idf"
            warnings.warn(
                "use_specificity_weighting is deprecated, use embedding_weights='idf'",
                DeprecationWarning,
                stacklevel=2,
            )
        elif embedding_method is not None:
            # Map legacy embedding_method to new embedding_weights
            if embedding_method == "idf_weighted":
                embedding_weights = "idf"
            elif embedding_method == "mean":
                embedding_weights = "uniform"
            warnings.warn(
                f"embedding_method is deprecated, use embedding_weights='{embedding_weights}'",
                DeprecationWarning,
                stacklevel=2,
            )

        # Initialize embedder
        if embedding_weights == "idf":
            self.embedder = IDFEmbedder(
                similarity_threshold=kwargs.get("similarity_threshold", 0.3),
                sharpening_power=kwargs.get("sharpening_power", 2.0),
                blend_ratio=kwargs.get("blend_ratio", 0.05),
            )
        else:
            self.embedder = SimpleAverageEmbedder()

        # Initialize clustering algorithm
        if clustering_method == "agglomerative":
            self.clusterer = AgglomerativeClusterer(linkage=linkage_method)
        elif clustering_method == "spectral":
            self.clusterer = SpectralClusterer(
                random_state=kwargs.get("random_state", 42)
            )
        elif clustering_method == "hybrid":
            self.clusterer = HybridClusterer(
                n_components=kwargs.get("n_components", 10),
                seed_multiplier=kwargs.get("seed_multiplier", 1.5),
                seeding_method=seeding_method,
                refinement_method=refinement_method,
                random_state=kwargs.get("random_state", 42),
                max_iterations=kwargs.get("max_iterations", 100),
                enable_swaps=kwargs.get("enable_swaps", False),
            )
        elif clustering_method == "random":
            self.clusterer = RandomClusterer(
                random_state=kwargs.get("random_state", 42)
            )
        elif clustering_method == "size_annealed_agglomerative":
            self.clusterer = SizeAnnealedAgglomerativeClusterer(
                enable_border_moves=kwargs.get("enable_border_moves", False),
                enable_swaps=kwargs.get("enable_swaps", False),
            )
        else:
            raise ValueError(f"Unknown clustering method: {clustering_method}")

        self.embedding_weights = embedding_weights
        self.clustering_method = clustering_method

    def group_documents(
        self,
        documents: List[str],
        chunk_data: Dict[str, List[Dict]],
        constraint_type: str = "count",
        min_size: int = 3,
        max_size: int = 8,
    ) -> List[List[str]]:
        """
        Group documents using configured embedding and clustering methods.

        Args:
            documents: List of document URLs/identifiers
            chunk_data: Dict mapping document ID to list of chunk objects (legacy format)
            constraint_type: Either "count" (document count) or "words" (word count)
            min_size: Minimum group size
            max_size: Maximum group size

        Returns:
            List of document groups
        """
        # Convert legacy chunk format
        typed_chunk_data = convert_legacy_chunk_data(chunk_data)

        # Compute embeddings
        embeddings = self.embedder.compute_embeddings(documents, typed_chunk_data)

        # Set up constraints
        constraints = ClusteringConstraints(
            min_size=min_size, max_size=max_size, constraint_type=constraint_type
        )

        # Compute document metrics for constraints
        if constraint_type == "words":
            doc_metrics = self._compute_word_counts(documents, chunk_data)
        else:
            doc_metrics = {doc: 1 for doc in documents}

        # Perform clustering
        result = self.clusterer.cluster(
            embeddings=embeddings, constraints=constraints, doc_metrics=doc_metrics
        )

        return result.groups

    def group_documents_with_details(
        self,
        documents: List[str],
        chunk_data: Dict[str, List[Dict]],
        constraint_type: str = "count",
        min_size: int = 3,
        max_size: int = 8,
        status: Optional["StatusProtocol"] = None,
        original_document_count: Optional[int] = None,
        clustering_method: str = "agglomerative",
        embedding_weights: str = "uniform",
        embedding_type: str = None,  # Legacy parameter
        dual_evaluation: bool = False,
        seeding_method: str = "kmeans",
        refinement_method: str = "hierarchical",
        **kwargs,
    ) -> GroupingResult:
        """
        Group documents with detailed metadata and progress reporting.

        Args:
            documents: List of document URLs/identifiers
            chunk_data: Dict mapping document ID to list of chunk objects
            constraint_type: Either "count" (document count) or "words" (word count)
            min_size: Minimum group size
            max_size: Maximum group size
            status: Optional status object for progress updates
            original_document_count: Original document count before filtering
            clustering_method: Clustering algorithm to use
            embedding_weights: Weighting strategy for chunk averaging ("uniform", "idf")
            embedding_type: Legacy parameter, maps to embedding_weights
            dual_evaluation: If True, compute metrics using both embedding methods

        Returns:
            GroupingResult with groups and comprehensive metadata
        """
        # Convert legacy parameters to new format
        if clustering_method == "spectral_advanced":
            clustering_method = "hybrid"

        # Handle legacy embedding_type parameter
        if embedding_type is not None:
            if embedding_type == "idf_weighted":
                embedding_weights = "idf"
            elif embedding_type == "mean":
                embedding_weights = "uniform"

        # Create temporary grouper with specified methods
        temp_grouper = DocumentGrouper(
            clustering_method=clustering_method,
            embedding_weights=embedding_weights,
            seeding_method=seeding_method,
            refinement_method=refinement_method,
            **kwargs,
        )

        # Convert legacy chunk format
        typed_chunk_data = convert_legacy_chunk_data(chunk_data)

        # Compute primary embeddings
        embeddings = temp_grouper.embedder.compute_embeddings(
            documents, typed_chunk_data, status
        )

        # Set up constraints and metrics
        constraints = ClusteringConstraints(
            min_size=min_size, max_size=max_size, constraint_type=constraint_type
        )

        if constraint_type == "words":
            doc_metrics = self._compute_word_counts(documents, chunk_data)
        else:
            doc_metrics = {doc: 1 for doc in documents}

        # Perform clustering
        result = temp_grouper.clusterer.cluster(
            embeddings=embeddings,
            constraints=constraints,
            doc_metrics=doc_metrics,
            status=status,
        )

        # Dual evaluation if requested
        if dual_evaluation:
            # Compute alternative embeddings
            alt_embedding_weights = "uniform" if embedding_weights == "idf" else "idf"
            alt_embedder = (
                IDFEmbedder()
                if alt_embedding_weights == "idf"
                else SimpleAverageEmbedder()
            )
            alt_embeddings = alt_embedder.compute_embeddings(
                documents, typed_chunk_data
            )

            # Compute alternative metrics
            alt_clusterer = temp_grouper.clusterer  # Use same clustering algorithm
            alt_S, alt_docs = alt_clusterer._build_similarity_matrix(alt_embeddings)
            alt_metrics = alt_clusterer._compute_standard_metrics(
                result.groups, alt_S, alt_docs
            )

            # Add dual metrics to result
            result.metrics.update(
                {
                    "alternative_embedding_weights": alt_embedding_weights,
                    "alternative_cohesions": alt_metrics.get("cohesions", []),
                    "alternative_avg_cohesion": alt_metrics.get("avg_cohesion", 0.0),
                    "alternative_cohesion_std": alt_metrics.get("cohesion_std", 0.0),
                }
            )

        # Enhance metadata for backward compatibility
        enhanced_metadata = {
            "algorithm": f"{clustering_method}_clustering",
            "constraint_type": constraint_type,
            "constraints": {"min_size": min_size, "max_size": max_size},
            "documents_clustered": len([d for d in documents if d in embeddings]),
            "total_documents": original_document_count or len(documents),
            "documents_with_embeddings": len(embeddings),
            "embedding_weights": embedding_weights,
            "dual_evaluation": dual_evaluation,
            # Copy over computed metrics
            **result.metrics,
        }

        # Legacy format compatibility
        if "cohesions" in result.metrics:
            enhanced_metadata["group_cohesions"] = result.metrics["cohesions"]

        if "group_size_histogram" not in enhanced_metadata:
            # Compute group size histogram for legacy compatibility
            size_histogram = {}
            for group in result.groups:
                size = len(group)
                size_histogram[size] = size_histogram.get(size, 0) + 1
            enhanced_metadata["group_size_histogram"] = size_histogram

        # Word counts per group for legacy compatibility
        total_word_counts = []
        for group in result.groups:
            group_words = 0
            for doc in group:
                chunks = chunk_data.get(doc, [])
                group_words += sum(chunk.get("wordcount", 0) for chunk in chunks)
            total_word_counts.append(group_words)
        enhanced_metadata["total_word_counts"] = total_word_counts

        return GroupingResult(
            groups=result.groups,
            metadata=enhanced_metadata,
            operation_log=result.operation_log,
        )

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

    @staticmethod
    def parse_constraint(constraint: str) -> Tuple[str, int, int]:
        """
        Parse constraint string like 'count:3-8' or 'words:5000-15000'.

        Returns:
            Tuple of (constraint_type, min_val, max_val)
        """
        import re

        match = re.match(r"(\w+):(\d+)-(\d+)", constraint)
        if not match:
            raise ValueError(
                f"Invalid constraint format: {constraint}. Expected 'type:min-max'"
            )

        constraint_type, min_val, max_val = match.groups()
        return constraint_type, int(min_val), int(max_val)


# Factory functions for easy algorithm selection
def create_embedder(weights: str, **kwargs) -> DocumentEmbedder:
    """
    Create document embedder by weighting strategy.

    Args:
        weights: "uniform" or "idf"
        **kwargs: Parameters passed to embedder constructor

    Returns:
        DocumentEmbedder instance
    """
    if weights == "uniform":
        return SimpleAverageEmbedder()
    elif weights == "idf":
        return IDFEmbedder(**kwargs)
    else:
        raise ValueError(f"Unknown embedding weights: {weights}")


def create_clusterer(method: str, **kwargs) -> DocumentClusterer:
    """
    Create document clusterer by method name.

    Args:
        method: "agglomerative", "spectral", "hybrid", "random", or "size_annealed_agglomerative"
        **kwargs: Parameters passed to clusterer constructor

    Returns:
        DocumentClusterer instance
    """
    if method == "agglomerative":
        return AgglomerativeClusterer(**kwargs)
    elif method == "spectral":
        return SpectralClusterer(**kwargs)
    elif method == "hybrid":
        return HybridClusterer(**kwargs)
    elif method == "random":
        return RandomClusterer(**kwargs)
    elif method == "size_annealed_agglomerative":
        return SizeAnnealedAgglomerativeClusterer(**kwargs)
    else:
        raise ValueError(f"Unknown clustering method: {method}")


# Legacy function for backward compatibility
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
    warnings.warn(
        "compute_group_cohesion is deprecated, use ClusteringResult.metrics instead",
        DeprecationWarning,
        stacklevel=2,
    )

    if len(group) < 2:
        return 1.0

    from scipy.spatial.distance import cosine

    similarities = []
    for i, doc_a in enumerate(group):
        for doc_b in group[i + 1 :]:
            if doc_a in doc_embeddings and doc_b in doc_embeddings:
                sim = 1.0 - cosine(doc_embeddings[doc_a], doc_embeddings[doc_b])
                similarities.append(sim)

    return float(np.mean(similarities)) if similarities else 0.0
