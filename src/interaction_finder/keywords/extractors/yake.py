"""YAKE (Yet Another Keyword Extractor) backend."""

from typing import List

import yake as yake_lib

from interaction_finder.logging import logfire
from interaction_finder.keywords.extractors.base import KeywordExtractor, ScoredKeyword


class YAKEExtractor(KeywordExtractor):
    """YAKE keyword extractor using statistical features.

    YAKE uses text statistical features (casing, position, frequency, context)
    to extract keywords. Works well without training data.

    Parameters:
        n_grams: int — maximum n-gram size (default: 3)
        deduplication_threshold: float — similarity threshold for deduplication (default: 0.9)
        window_size: int — context window size (default: 1)
    """

    def __init__(
        self,
        n_grams: int = 3,
        deduplication_threshold: float = 0.9,
        window_size: int = 1,
    ):
        """Initialize YAKE extractor.

        Parameters:
            n_grams: int — maximum n-gram size (1-5)
            deduplication_threshold: float — deduplication threshold (0-1)
            window_size: int — context window size
        """
        if not 1 <= n_grams <= 5:
            raise ValueError(f"n_grams must be in [1, 5], got {n_grams}")
        if not 0 <= deduplication_threshold <= 1:
            raise ValueError(
                f"deduplication_threshold must be in [0, 1], got {deduplication_threshold}"
            )
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self.n_grams = n_grams
        self.deduplication_threshold = deduplication_threshold
        self.window_size = window_size

    @property
    def name(self) -> str:
        """Return extractor identifier."""
        return "yake"

    def extract(self, text: str, max_keywords: int = 20) -> List[ScoredKeyword]:
        """Extract keywords using YAKE algorithm.

        Parameters:
            text: str — input text
            max_keywords: int — maximum keywords to return

        Returns:
            List[ScoredKeyword] — scored keywords, sorted by score descending

        Raises:
            ValueError — if text is empty or max_keywords < 1
        """
        with logfire.span(
            "YAKEExtractor.extract",
            text_length=len(text),
            max_keywords=max_keywords,
            n_grams=self.n_grams,
        ):
            if not text or not text.strip():
                raise ValueError("Text cannot be empty")
            if max_keywords < 1:
                raise ValueError(f"max_keywords must be >= 1, got {max_keywords}")
            # Create YAKE extractor instance
            # YAKE scores are lower=better, so we'll need to invert them
            kw_extractor = yake_lib.KeywordExtractor(
                lan="en",
                n=self.n_grams,
                dedupLim=self.deduplication_threshold,
                dedupFunc="seqm",
                windowsSize=self.window_size,
                top=max_keywords,
            )
            # Extract keywords
            keywords = kw_extractor.extract_keywords(text)
            # Handle empty results
            if not keywords:
                logfire.info("YAKE extracted 0 keywords")
                return []
            # YAKE scores are inverted (lower is better), so we convert to higher=better
            # and normalize to [0, 1]
            # We use 1 / (1 + score) to convert, which maps [0, inf) -> (0, 1]
            results = []
            for phrase, score in keywords:
                normalized_score = 1.0 / (1.0 + score)
                results.append(ScoredKeyword(keyword=phrase, score=normalized_score))
            # Sort by score descending
            results.sort(key=lambda x: x.score, reverse=True)
            logfire.info(f"YAKE extracted {len(results)} keywords")
            return results
