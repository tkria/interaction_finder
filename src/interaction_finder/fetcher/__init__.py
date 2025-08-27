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
)
from .content_processor import (
    refine_article_content,
    extract_headings,
    classify_heading_relevance,
)
from .utils import url_to_hash, normalize_url, url_to_hash_base36

# Import the URLCache class for direct use if needed
from .cache import URLCache

# Import document grouping functionality
from .document_grouper import DocumentGrouper, compute_group_cohesion


# Import the exception class
class PreviousFailure(Exception):
    """Raised when a previous fetch failure sentinel is present for a URL."""

    def __init__(self, url: str, message: str | None = None):
        super().__init__(message or f"Previous failure recorded for URL: {url}")
        self.url = url


# Export the public API - maintains exact compatibility with original fetcher.py
__all__ = [
    "PageFetcher",
    "URLCache",
    "DocumentGrouper",
    "compute_group_cohesion",
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
