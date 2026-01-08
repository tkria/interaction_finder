from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field


class QueryStyle(Enum):
    """Query formatting styles for search backends."""

    CONVERSATIONAL = "conversational"
    KEYWORD = "keyword"
    PUBMED = "pubmed"


# Style → directive mapping
STYLE_DIRECTIVES: Dict[QueryStyle, str] = {
    QueryStyle.CONVERSATIONAL: (
        "Use natural language queries. No boolean operators or special syntax."
    ),
    QueryStyle.KEYWORD: (
        "Use keyword search syntax: AND is implicit between terms, "
        'OR for alternatives, -term to exclude, "quoted phrases" for exact matches.'
    ),
    QueryStyle.PUBMED: (
        "Use PubMed syntax: boolean operators, quoted phrases, "
        "field tags like [tiab], [mesh], and filters like review[pt]."
    ),
}

# Backend → style mapping
BACKEND_STYLES: Dict[str, QueryStyle] = {
    "perplexica": QueryStyle.CONVERSATIONAL,
    "openai": QueryStyle.CONVERSATIONAL,
    "pubmed": QueryStyle.PUBMED,
    "searxng": QueryStyle.KEYWORD,
}


def get_query_directive(backend_name: str) -> str:
    """Get query formatting directive for a backend."""
    style = BACKEND_STYLES.get(backend_name.lower(), QueryStyle.CONVERSATIONAL)
    return STYLE_DIRECTIVES[style]


class SearchQuery(BaseModel):
    """A search query for retrieving documents."""

    query: str = Field(description="The search query string")
    max_results: int = Field(
        default=100, ge=1, description="Maximum number of search results to return"
    )


class SearchResult(BaseModel):
    """A single search result document."""

    title: str = Field(description="Title of the document")
    url: str = Field(description="URL of the document")
    snippet: Optional[str] = Field(description="A brief snippet from the document")
    relevance: Optional[float] = Field(
        None, description="Relevance score of the document", ge=0.0, le=1.0
    )


class SearchBackend(ABC):
    """Abstract base class for search backends."""

    def __init__(self, config: Dict[str, Any] = {}):
        """Initialize the search backend with optional configuration."""
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the search backend."""
        pass

    @abstractmethod
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform a search and return a list of SearchResult objects."""
        pass

    @abstractmethod
    def healthy(self) -> bool:
        """Check if the search backend is healthy."""
        pass
