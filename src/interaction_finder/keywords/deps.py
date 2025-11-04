"""External service dependencies for keyword research pipeline."""

from dataclasses import dataclass
from typing import Any

import httpx

from interaction_finder.fetcher import PageFetcher
from interaction_finder.keywords.extractors import KeywordExtractor
from interaction_finder.keywords.reranker import Reranker
from interaction_finder.search.models import SearchBackend


@dataclass
class Deps:
    """External services and long-lived clients for keyword research.

    Instantiated once per pipeline run. Passed to all agents and tools
    via context.
    """

    http_client: httpx.AsyncClient  # HTTP client for API calls
    fetcher: PageFetcher  # Web content fetcher with caching
    search_backend: SearchBackend  # Search backend (PubMed, Perplexica, etc.)
    reranker: Reranker  # Semantic reranker for search results
    extractors: dict[str, KeywordExtractor]  # name -> extractor instance
    config: dict[str, Any]  # Configuration dictionary
