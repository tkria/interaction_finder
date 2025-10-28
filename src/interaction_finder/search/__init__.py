"""
Search backends for retrieving academic documents.

This module provides abstract interfaces and concrete implementations for searching
scientific literature databases like PubMed.
"""

from .models import SearchBackend, SearchQuery, SearchResult

__all__ = ["SearchBackend", "SearchQuery", "SearchResult"]
