"""
Fetcher package for web content retrieval and processing.

Provides the same public API as the original fetcher.py but with improved internal structure.
"""

# Import the main PageFetcher class
from .page_fetcher import PageFetcher

# Import standalone utility functions for backward compatibility
from .batch_operations import (
    fetch_urls_with_progress,
    fetch_urls_concurrent_with_progress,
    PreviousFailure,
)
from .content_processor import (
    refine_article_content,
    extract_headings,
    classify_heading_relevance,
    ChunkData,
    convert_legacy_chunk_data,
)
from .utils import url_to_hash, normalize_url, url_to_hash_base36

# Import the URLCache class for direct use if needed
from .cache import URLCache

# Import document grouping functionality
from .document_grouper import (
    DocumentGrouper,
    compute_group_cohesion,
    create_embedder,
    create_clusterer,
)

# Import embedding strategies (to be removed in Task 03)
from .document_embedding import (
    DocumentEmbedder,
    SimpleAverageEmbedder,
    IDFEmbedder,
)
from .document_clustering import (
    DocumentClusterer,
    AgglomerativeClusterer,
    SpectralClusterer,
    HybridClusterer,
    RandomClusterer,
    SizeAnnealedAgglomerativeClusterer,
    ClusterCache,
    ClusteringConstraints,
    ClusteringResult,
)


# Export the public API - maintains exact compatibility with original fetcher.py
__all__ = [
    "PageFetcher",
    "URLCache",
    "DocumentGrouper",
    "compute_group_cohesion",
    "create_embedder",
    "create_clusterer",
    # New modular components
    "DocumentEmbedder",
    "SimpleAverageEmbedder",
    "IDFEmbedder",
    "ChunkData",
    "convert_legacy_chunk_data",
    "DocumentClusterer",
    "AgglomerativeClusterer",
    "SpectralClusterer",
    "HybridClusterer",
    "RandomClusterer",
    "SizeAnnealedAgglomerativeClusterer",
    "ClusterCache",
    "ClusteringConstraints",
    "ClusteringResult",
    # Utilities
    "PreviousFailure",
    "fetch_urls_with_progress",
    "fetch_urls_concurrent_with_progress",
    "refine_article_content",
    "extract_headings",
    "classify_heading_relevance",
    "url_to_hash",
    "normalize_url",
    "url_to_hash_base36",
]
