"""
Base classes and interfaces for search functionality.

This module defines the core abstractions that all search backends must implement,
along with standardized data models for queries, results, and expansion.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any, AsyncIterator
from pydantic import BaseModel, Field
from urllib.parse import urlparse


class SearchQuery(BaseModel):
    """Represents a search query with optional expansion and filters."""

    query: str = Field(description="The search query string")
    expanded_terms: List[str] = Field(
        default_factory=list, description="Additional terms from query expansion"
    )
    filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Search filters like date range, publication type, etc.",
    )
    max_results: int = Field(
        default=100, description="Maximum number of results to return", ge=1, le=1000
    )

    @property
    def full_query(self) -> str:
        """Get the complete query including expanded terms."""
        if not self.expanded_terms:
            return self.query

        # Combine original query with expanded terms using OR logic
        expanded = " OR ".join(f'"{term}"' for term in self.expanded_terms)
        return f"({self.query}) OR ({expanded})"


class SearchResult(BaseModel):
    """Represents a single search result from any backend."""

    title: str = Field(description="Paper title")
    url: str = Field(description="Direct URL to the paper")
    abstract: Optional[str] = Field(None, description="Paper abstract if available")
    authors: List[str] = Field(default_factory=list, description="List of author names")
    publication_date: Optional[datetime] = Field(
        None, description="Publication or online date"
    )
    journal: Optional[str] = Field(None, description="Journal or venue name")
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    pmid: Optional[str] = Field(None, description="PubMed ID if available")
    relevance_score: Optional[float] = Field(
        None,
        description="Relevance score from search backend (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    citation_count: Optional[int] = Field(
        None, description="Number of citations if available", ge=0
    )
    backend: str = Field(description="Which search backend found this result")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional backend-specific metadata"
    )

    @property
    def domain(self) -> str:
        """Get the domain from the URL for grouping/filtering."""
        return urlparse(self.url).netloc

    @property
    def short_title(self) -> str:
        """Get a shortened version of the title for display."""
        if len(self.title) <= 80:
            return self.title
        return self.title[:77] + "..."


class SearchResults(BaseModel):
    """Collection of search results with metadata."""

    query: SearchQuery = Field(description="The original search query")
    results: List[SearchResult] = Field(description="List of search results")
    total_found: Optional[int] = Field(
        None, description="Total results available (may be more than returned)"
    )
    search_time: Optional[float] = Field(
        None, description="Time taken for search in seconds"
    )
    backend: str = Field(description="Search backend used")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When the search was performed"
    )

    @property
    def result_count(self) -> int:
        """Number of results actually returned."""
        return len(self.results)

    @property
    def unique_domains(self) -> List[str]:
        """Get unique domains from all results."""
        return list(set(result.domain for result in self.results))

    def filter_by_domain(self, domains: List[str]) -> "SearchResults":
        """Filter results to only include specified domains."""
        filtered = [r for r in self.results if r.domain in domains]
        return self.model_copy(update={"results": filtered})

    def sort_by_relevance(self, reverse: bool = True) -> "SearchResults":
        """Sort results by relevance score."""
        sorted_results = sorted(
            self.results, key=lambda r: r.relevance_score or 0.0, reverse=reverse
        )
        return self.model_copy(update={"results": sorted_results})


class SearchBackend(ABC):
    """Abstract base class for all search backend implementations."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the search backend with configuration."""
        self.config = config

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Name identifier for this backend."""
        pass

    @abstractmethod
    async def search(self, query: SearchQuery) -> SearchResults:
        """
        Perform a search with the given query.

        Args:
            query: SearchQuery object with search parameters

        Returns:
            SearchResults object with found papers

        Raises:
            SearchError: If search fails or backend is unavailable
        """
        pass

    async def search_stream(self, query: SearchQuery) -> AsyncIterator[SearchResult]:
        """
        Stream search results as they become available.

        Default implementation just yields all results from search(),
        but backends can override for true streaming.
        """
        results = await self.search(query)
        for result in results.results:
            yield result

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the search backend is available and working.

        Returns:
            True if backend is healthy, False otherwise
        """
        pass

    def supports_filters(self) -> List[str]:
        """
        Return list of supported filter types for this backend.

        Common filters: 'date_range', 'publication_type', 'language',
                       'has_fulltext', 'journal', 'author'
        """
        return []

    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Return rate limiting information for this backend."""
        return {
            "requests_per_second": None,
            "requests_per_minute": None,
            "requests_per_hour": None,
            "burst_limit": None,
        }


@dataclass
class ExpansionTerm:
    """Represents an expanded query term with metadata."""

    term: str
    confidence: float = 1.0  # 0.0 to 1.0 confidence score
    source: str = "unknown"  # Source of expansion (dictionary, llm, mesh, etc.)
    category: Optional[str] = None  # gene, disease, protein, etc.

    def __str__(self) -> str:
        return self.term


@dataclass
class ExpandedQuery:
    """Result of query expansion with original and expanded terms."""

    original_query: str
    expanded_terms: List[ExpansionTerm]
    expansion_method: str
    total_confidence: float = 1.0

    @property
    def all_terms(self) -> List[str]:
        """Get all terms including original query."""
        return [self.original_query] + [term.term for term in self.expanded_terms]

    @property
    def high_confidence_terms(self, threshold: float = 0.7) -> List[str]:
        """Get only high-confidence expanded terms."""
        return [
            term.term for term in self.expanded_terms if term.confidence >= threshold
        ]


class QueryExpander(ABC):
    """Abstract base class for query expansion strategies."""

    @abstractmethod
    async def expand_query(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> ExpandedQuery:
        """
        Expand a search query with additional relevant terms.

        Args:
            query: Original search query
            context: Optional context like entity types, domain, etc.

        Returns:
            ExpandedQuery with original query and expansion results
        """
        pass

    @property
    @abstractmethod
    def expansion_method(self) -> str:
        """Name of the expansion method."""
        pass

    def supports_context(self) -> List[str]:
        """Return list of supported context keys for this expander."""
        return []


class SearchError(Exception):
    """Base exception for search-related errors."""

    def __init__(
        self, message: str, backend: Optional[str] = None, query: Optional[str] = None
    ):
        self.backend = backend
        self.query = query
        super().__init__(message)


class SearchTimeoutError(SearchError):
    """Raised when search operation times out."""

    pass


class SearchRateLimitError(SearchError):
    """Raised when search backend rate limit is exceeded."""

    pass


class SearchUnavailableError(SearchError):
    """Raised when search backend is unavailable."""

    pass
