"""RAKE (Rapid Automatic Keyword Extraction) backend."""

from typing import List

import nltk
from rake_nltk import Rake

from interaction_finder.keywords.extractors.base import KeywordExtractor, ScoredKeyword

# Ensure NLTK data is downloaded
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


class RAKEExtractor(KeywordExtractor):
    """RAKE keyword extractor using statistical co-occurrence.

    RAKE identifies keywords by analyzing word co-occurrence patterns and
    word frequency. Good for domain-agnostic extraction.

    Parameters:
        min_length: int — minimum words in a keyphrase (default: 1)
        max_length: int — maximum words in a keyphrase (default: 4)
    """

    def __init__(self, min_length: int = 1, max_length: int = 4):
        """Initialize RAKE extractor with phrase length constraints.

        Parameters:
            min_length: int — minimum words per phrase
            max_length: int — maximum words per phrase
        """
        if min_length < 1:
            raise ValueError(f"min_length must be >= 1, got {min_length}")
        if max_length < min_length:
            raise ValueError(
                f"max_length ({max_length}) must be >= min_length ({min_length})"
            )
        self.min_length = min_length
        self.max_length = max_length
        self._rake = Rake(min_length=min_length, max_length=max_length)

    @property
    def name(self) -> str:
        """Return extractor identifier."""
        return "rake"

    def extract(self, text: str, max_keywords: int = 20) -> List[ScoredKeyword]:
        """Extract keywords using RAKE algorithm.

        Parameters:
            text: str — input text
            max_keywords: int — maximum keywords to return

        Returns:
            List[ScoredKeyword] — scored keywords, sorted by score descending

        Raises:
            ValueError — if text is empty or max_keywords < 1
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        if max_keywords < 1:
            raise ValueError(f"max_keywords must be >= 1, got {max_keywords}")
        # Extract keywords and get scores
        self._rake.extract_keywords_from_text(text)
        ranked = self._rake.get_ranked_phrases_with_scores()
        # Handle empty results
        if not ranked:
            return []
        # Normalize scores to [0, 1] range
        # RAKE scores are unbounded, so we normalize by dividing by max score
        max_score = max(score for score, _ in ranked) if ranked else 1.0
        if max_score == 0:
            max_score = 1.0  # Avoid division by zero
        results = []
        for score, phrase in ranked[:max_keywords]:
            normalized_score = min(score / max_score, 1.0)
            results.append(ScoredKeyword(keyword=phrase, score=normalized_score))
        return results
