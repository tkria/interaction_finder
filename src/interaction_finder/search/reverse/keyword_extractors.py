"""
Domain-agnostic keyword extraction algorithms for query generation.

This module provides a unified interface for multiple keyword extraction
approaches (YAKE, RAKE, TF-IDF, None) suitable for processing scientific literature
of any domain.

The None extractor is a null-object pattern that returns empty keywords,
signaling that query construction should use the full resource content
(typically with an LLM-based constructor) rather than extracted keywords.

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
        import nltk

        # Ensure required NLTK data is available
        required_data = ["corpora/stopwords", "tokenizers/punkt_tab"]
        for resource in required_data:
            try:
                nltk.data.find(resource)
            except LookupError:
                # Download missing data
                resource_name = resource.split("/")[-1]
                nltk.download(resource_name, quiet=True)

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


class NoneExtractor(KeywordExtractor):
    """
    Null extractor that returns empty keyword list.

    This is a null-object pattern extractor that signals "no keyword extraction".
    When paired with an LLM-based query constructor, this enables full-content
    LLM query generation without any keyword extraction step.

    Use this extractor when:
    - You want the query constructor to have full access to resource content
    - You're using an LLM constructor that can read the full text
    - You want to avoid the keyword extraction → assembly pipeline

    Returns:
        Always returns an empty list, regardless of input text.
    """

    @property
    def name(self) -> str:
        return "none"

    def extract(self, text: str, top_n: int) -> List[str]:
        """
        Return empty keyword list for any input.

        Parameters:
            text: str - Input text (ignored)
            top_n: int - Maximum keywords to return (ignored)

        Returns:
            List[str] - Always empty list

        Example:
            >>> extractor = NoneExtractor()
            >>> extractor.extract("Some long paper content...", top_n=5)
            []
        """
        return []


# ==============================================================================
# Factory Function
# ==============================================================================


def create_extractor(name: str, **kwargs) -> KeywordExtractor:
    """
    Create keyword extractor by name.

    Parameters:
        name: str - Extractor name ("yake", "rake", "tfidf", "none"), case-insensitive
        **kwargs: Additional parameters for extractor initialization (currently unused)

    Returns:
        KeywordExtractor instance

    Raises:
        ValueError: If name is not recognized

    Example:
        >>> extractor = create_extractor("yake")
        >>> keywords = extractor.extract("Machine learning and AI", top_n=3)
        >>> none_extractor = create_extractor("none")
        >>> keywords = none_extractor.extract("Any text", top_n=5)  # Returns []
    """
    name_lower = name.lower()
    if name_lower == "yake":
        return YAKEExtractor()
    elif name_lower == "rake":
        return RAKEExtractor()
    elif name_lower == "tfidf":
        return TFIDFExtractor()
    elif name_lower == "none":
        return NoneExtractor()
    else:
        raise ValueError(
            f"Unknown extractor name: {name}. Supported: yake, rake, tfidf, none"
        )
