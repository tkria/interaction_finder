from abc import ABC, abstractmethod

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

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
    relevance: Optional[float] = Field(None, description="Relevance score of the document", ge=0.0, le=1.0)

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
