"""TF-IDF (Term Frequency-Inverse Document Frequency) backend."""

from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from interaction_finder.keywords.extractors.base import KeywordExtractor, ScoredKeyword
from interaction_finder.logging import logfire


class TFIDFExtractor(KeywordExtractor):
    """TF-IDF keyword extractor using statistical term weighting.

    Extracts keywords by computing TF-IDF scores. Works best with multiple
    documents but can operate on single documents by treating sentences as
    separate documents.

    Parameters:
        max_features: int — maximum vocabulary size (default: 50)
        ngram_range: tuple — n-gram range (min, max) (default: (1, 3))
        min_df: int — minimum document frequency (default: 1)
    """

    def __init__(
        self,
        max_features: int = 50,
        ngram_range: tuple[int, int] = (1, 3),
        min_df: int = 1,
    ):
        """Initialize TF-IDF extractor.

        Parameters:
            max_features: int — maximum vocabulary size
            ngram_range: tuple — (min_n, max_n) for n-grams
            min_df: int — minimum document frequency
        """
        if max_features < 1:
            raise ValueError(f"max_features must be >= 1, got {max_features}")
        if len(ngram_range) != 2:
            raise ValueError(f"ngram_range must be (min, max), got {ngram_range}")
        if ngram_range[0] < 1 or ngram_range[1] < ngram_range[0]:
            raise ValueError(f"Invalid ngram_range: {ngram_range}")
        if min_df < 1:
            raise ValueError(f"min_df must be >= 1, got {min_df}")
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df

    @property
    def name(self) -> str:
        """Return extractor identifier."""
        return "tfidf"

    def extract(self, text: str, max_keywords: int = 20) -> List[ScoredKeyword]:
        """Extract keywords using TF-IDF.

        Splits text into sentences and treats each as a document for TF-IDF
        computation.

        Parameters:
            text: str — input text
            max_keywords: int — maximum keywords to return

        Returns:
            List[ScoredKeyword] — scored keywords, sorted by score descending

        Raises:
            ValueError — if text is empty or max_keywords < 1
        """
        with logfire.span(
            "TFIDFExtractor.extract", text_length=len(text), max_keywords=max_keywords
        ):
            if not text or not text.strip():
                raise ValueError("Text cannot be empty")
            if max_keywords < 1:
                raise ValueError(f"max_keywords must be >= 1, got {max_keywords}")
            # Split text into sentences to create pseudo-documents
            # Simple sentence splitting by period, question mark, exclamation
            sentences = [
                s.strip()
                for s in text.replace("!", ".").replace("?", ".").split(".")
                if s.strip()
            ]
            # Need at least one sentence
            if not sentences:
                logfire.info("TF-IDF extracted 0 keywords")
                return []
            # Handle case where we have only one sentence
            if len(sentences) == 1:
                # Split by comma or semicolon to create pseudo-documents
                sentences = [
                    s.strip()
                    for s in sentences[0].replace(";", ",").split(",")
                    if s.strip()
                ]
            # If still only one document, we can't compute IDF meaningfully
            # In this case, we'll just use term frequency
            if len(sentences) == 1:
                sentences = [text]  # Use original text
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=min(self.min_df, len(sentences)),  # Adjust for small doc count
                stop_words="english",
            )
            # Fit and transform
            try:
                tfidf_matrix = vectorizer.fit_transform(sentences)
            except ValueError:
                # No valid terms found (e.g., all stop words)
                logfire.info("TF-IDF extracted 0 keywords (no valid terms)")
                return []
            # Get feature names (keywords)
            feature_names = vectorizer.get_feature_names_out()
            # Sum TF-IDF scores across all documents for each term
            scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
            # Create keyword-score pairs
            keyword_scores = list(zip(feature_names, scores))
            # Sort by score descending
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            # Normalize scores to [0, 1]
            if keyword_scores:
                max_score = keyword_scores[0][1] if keyword_scores[0][1] > 0 else 1.0
                results = []
                for keyword, score in keyword_scores[:max_keywords]:
                    normalized_score = min(score / max_score, 1.0)
                    results.append(
                        ScoredKeyword(keyword=keyword, score=normalized_score)
                    )
                logfire.info(f"TF-IDF extracted {len(results)} keywords")
                return results
            logfire.info("TF-IDF extracted 0 keywords")
            return []
