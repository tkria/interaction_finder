"""Abstract base interface for keyword extraction backends."""

from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel, Field



class ScoredKeyword(BaseModel):
    """A keyword with an associated relevance score."""

    keyword: str = Field(description="The extracted keyword or keyphrase")
    score: float = Field(
        ge=0.0, le=1.0, description="Relevance score (0-1, higher is better)"
    )
    context: str | None = Field(
        None, description="Optional context where keyword appears"
    )


class KeywordExtractor(ABC):
    """Abstract base class for keyword extraction backends.

    Extractors take text and produce scored keywords. All scores should be
    normalized to [0, 1] range for consistency across methods.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return extractor identifier (e.g., 'rake', 'yake', 'tfidf', 'keybert')."""
        pass

    @abstractmethod
    def extract(self, text: str, max_keywords: int = 20) -> List[ScoredKeyword]:
        """Extract keywords from text.

        Parameters:
            text: str — input text to extract keywords from
            max_keywords: int — maximum number of keywords to return (default: 20)

        Returns:
            List[ScoredKeyword] — scored keywords, sorted by score (descending)

        Raises:
            ValueError — if text is empty or max_keywords < 1
        """
        pass

    def healthy(self) -> bool:
        """Check if extractor is ready to use (optional health check)."""
        return True
