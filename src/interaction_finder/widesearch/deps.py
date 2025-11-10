"""External service dependencies for widesearch pipeline.

Dependencies encapsulate long-lived external services and clients.
Instantiated once per pipeline run and passed to all nodes/agents.
"""

from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from interaction_finder.resources import ResourcePool
from interaction_finder.search.models import SearchBackend
from interaction_finder.settings import IfetcherConfig
from interaction_finder.widesearch.reranker import Reranker


class ProgressProtocol(Protocol):
    """Protocol for progress counters."""

    def increment_searches(self, count: int = 1) -> None: ...
    def add_results(self, count: int) -> None: ...
    def add_selected(self, count: int) -> None: ...
    def set_round(self, current: int, max_rounds: int) -> None: ...


@dataclass
class Deps:
    """External services and long-lived clients for widesearch.

    Instantiated once per pipeline run. Passed to all agents and tools
    via GraphRunContext.
    """

    http_client: httpx.AsyncClient  # HTTP client for API calls
    search_backend: SearchBackend  # Search backend (PubMed, Perplexica, OpenAI)
    reranker: Reranker | None  # Semantic reranker (None if reranking disabled)
    resource_pool: ResourcePool  # Document pool with URL normalization
    config: IfetcherConfig  # Full configuration object
    progress: ProgressProtocol | None = None  # Optional progress counter
