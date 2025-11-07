"""External service dependencies for widesearch pipeline.

Dependencies encapsulate long-lived external services and clients.
Instantiated once per pipeline run and passed to all nodes/agents.
"""

from dataclasses import dataclass
from typing import Any

import httpx

from interaction_finder.resources import ResourcePool
from interaction_finder.search.models import SearchBackend
from interaction_finder.widesearch.reranker import Reranker


@dataclass
class Deps:
    """External services and long-lived clients for widesearch.

    Instantiated once per pipeline run. Passed to all agents and tools
    via GraphRunContext.
    """

    http_client: httpx.AsyncClient  # HTTP client for API calls
    search_backend: SearchBackend  # Search backend (PubMed, Perplexica, OpenAI)
    reranker: Reranker | None  # Semantic reranker for search results (None if disabled)
    resource_pool: ResourcePool  # Document pool with URL normalization
    config: dict[str, Any]  # Configuration dictionary (widesearch settings)
