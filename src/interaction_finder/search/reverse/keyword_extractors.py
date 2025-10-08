"""
Domain-agnostic keyword extraction algorithms for query generation.

This module provides a unified interface for multiple keyword extraction
approaches (YAKE, RAKE, TF-IDF) suitable for processing scientific literature
of any domain.

Order: ABC interface → implementations → factory
"""

from abc import ABC, abstractmethod
from typing import List

# ==============================================================================
# Abstract Interface
# ==============================================================================


class KeywordExtractor(ABC):
    """Abstract interface for keyword extraction algorithms."""

    @abstractmethod
    def extract(self, text: str, top_n: int) -> List[str]:
        """
        Extract top N keywords from text.

        Parameters:
            text: str - Input text to extract keywords from
            top_n: int - Maximum number of keywords to return

        Returns:
            List[str] - Top N keywords, may be fewer if text is short

        Raises:
            ValueError: If top_n <= 0
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return extractor name for logging/debugging."""
        pass


# ==============================================================================
# Implementations
# ==============================================================================


class YAKEExtractor(KeywordExtractor):
    """
    YAKE (Yet Another Keyword Extractor) keyword extraction.

    Unsupervised, domain-independent algorithm that uses statistical features
    to identify important keywords and keyphrases.

    Parameters:
        max_ngram_size: int - Maximum n-gram size (1=unigrams, 2=bigrams, 3=trigrams)
        dedup_threshold: float - Deduplication threshold for similar keywords (0.0-1.0)
    """

    def __init__(self, max_ngram_size: int = 3, dedup_threshold: float = 0.9):
        self.max_ngram_size = max_ngram_size
        self.dedup_threshold = dedup_threshold

    @property
    def name(self) -> str:
        return "yake"

    def extract(self, text: str, top_n: int) -> List[str]:
        if top_n <= 0:
            raise ValueError("top_n must be > 0")
        if not text or not text.strip():
            return []

        import yake

        # Create extractor per-call to handle different top_n values
        kw_extractor = yake.KeywordExtractor(
            lan="en",
            n=self.max_ngram_size,
            dedupLim=self.dedup_threshold,
            top=top_n,
        )

        keywords = kw_extractor.extract_keywords(text)
        # Return just the keyword strings, not scores
        return [kw for kw, score in keywords]


class RAKEExtractor(KeywordExtractor):
    """
    RAKE (Rapid Automatic Keyword Extraction) keyword extraction.

    Fast, simple algorithm that identifies keywords by analyzing word frequency
    and co-occurrence patterns. Particularly good at extracting multi-word phrases.
    """

    def __init__(self):
        from rake_nltk import Rake

        self.rake = Rake()

    @property
    def name(self) -> str:
        return "rake"

    def extract(self, text: str, top_n: int) -> List[str]:
        if top_n <= 0:
            raise ValueError("top_n must be > 0")
        if not text or not text.strip():
            return []

        # Extract keywords from text
        self.rake.extract_keywords_from_text(text)
        phrases = self.rake.get_ranked_phrases()
        # Return top N ranked phrases
        return phrases[:top_n]


class TFIDFExtractor(KeywordExtractor):
    """
    TF-IDF-based keyword extraction using scikit-learn.

    Extracts keywords based on term frequency and inverse document frequency.
    For single-document use, this degenerates to term frequency with stop word
    filtering, which is still useful for identifying important terms.

    Parameters:
        ngram_range: tuple - Range of n-gram sizes to consider (min, max)
    """

    def __init__(self, ngram_range: tuple = (1, 3)):
        self.ngram_range = ngram_range

    @property
    def name(self) -> str:
        return "tfidf"

    def extract(self, text: str, top_n: int) -> List[str]:
        if top_n <= 0:
            raise ValueError("top_n must be > 0")
        if not text or not text.strip():
            return []

        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np

        # Configure vectorizer for single-document keyword extraction
        vectorizer = TfidfVectorizer(
            max_features=top_n,
            ngram_range=self.ngram_range,
            stop_words="english",
        )

        try:
            # Fit and transform on single document
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()

            # Sort features by TF-IDF score (descending)
            scores = tfidf_matrix.toarray()[0]
            sorted_indices = np.argsort(scores)[::-1]

            # Return only features with non-zero scores
            return [feature_names[i] for i in sorted_indices if scores[i] > 0]
        except ValueError:
            # Handle case where text has no valid features after stop word removal
            return []


# ==============================================================================
# Factory Function
# ==============================================================================


def create_extractor(name: str) -> KeywordExtractor:
    """
    Create keyword extractor by name.

    Parameters:
        name: str - Extractor name ("yake", "rake", "tfidf"), case-insensitive

    Returns:
        KeywordExtractor instance

    Raises:
        ValueError: If name is not recognized

    Example:
        >>> extractor = create_extractor("yake")
        >>> keywords = extractor.extract("Machine learning and AI", top_n=3)
    """
    name_lower = name.lower()
    if name_lower == "yake":
        return YAKEExtractor()
    elif name_lower == "rake":
        return RAKEExtractor()
    elif name_lower == "tfidf":
        return TFIDFExtractor()
    else:
        raise ValueError(
            f"Unknown extractor name: {name}. Supported: yake, rake, tfidf"
        )
