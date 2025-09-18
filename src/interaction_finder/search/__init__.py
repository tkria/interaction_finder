"""
Document search functionality for biomedical literature discovery.

This module provides pluggable search backends for finding relevant scientific
papers and documents based on research queries, with support for query expansion
and result evaluation.
"""

from .base import SearchBackend, SearchResult, SearchQuery
from .config import SearchConfig

__all__ = [
    "SearchBackend",
    "SearchResult",
    "SearchQuery",
    "SearchConfig",
]
