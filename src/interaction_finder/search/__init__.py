"""
Document search functionality for biomedical literature discovery.

This module provides pluggable search backends for finding relevant scientific
papers and documents based on research queries, with support for query expansion,
result caching, and evaluation.
"""

from .base import SearchBackend, SearchResult, SearchQuery, SearchResults, SearchError
from .config import SearchConfig
from .backends.pubmed import PubMedBackend
from .backends.perplexica import PerplexicaBackend
from .backends.openai_search import OpenAISearchBackend
from .expansion.llm import LLMQueryExpander, create_llm_expander
from .expansion.advanced import AdvancedQueryExpander, create_advanced_expander
from .cache import SearchCache

__all__ = [
    # Base classes
    "SearchBackend",
    "SearchResult",
    "SearchQuery",
    "SearchResults",
    "SearchError",
    # Configuration
    "SearchConfig",
    # Backends
    "PubMedBackend",
    "PerplexicaBackend",
    "OpenAISearchBackend",
    # Query expansion
    "LLMQueryExpander",
    "create_llm_expander",
    "AdvancedQueryExpander",
    "create_advanced_expander",
    # Caching
    "SearchCache",
]
