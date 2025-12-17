"""
Fetcher package for web content retrieval and processing.

Provides the same public API as the original fetcher.py but with improved internal structure.
"""

# Import the main PageFetcher class and FetchedDocument dataclass
from .page_fetcher import PageFetcher, FetchedDocument

from .web_client import ensure_playwright_installed, PlaywrightNotInstalledError

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

# Import OpenAlex metadata fetchers
from .doi_metadata import fetch_doi_metadata, fetch_work_metadata

# Export the public API
__all__ = [
    "PageFetcher",
    "FetchedDocument",
    "URLCache",
    "ChunkData",
    "convert_legacy_chunk_data",
    "ensure_playwright_installed",
    "PlaywrightNotInstalledError",
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
    # OpenAlex metadata
    "fetch_doi_metadata",
    "fetch_work_metadata",
]
