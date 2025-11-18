"""KeyBERT (BERT-based keyword extraction) backend."""

import logging
import threading
from typing import List

from keybert import KeyBERT as KeyBERTModel

from interaction_finder.keywords.extractors.base import KeywordExtractor, ScoredKeyword
from interaction_finder.logging import logfire

logger = logging.getLogger(__name__)


class KeyBERTExtractor(KeywordExtractor):
    """KeyBERT extractor using BERT embeddings for semantic keyword extraction.

    Uses contextualized embeddings to identify keywords that are semantically
    similar to the document. Requires sentence-transformers model.

    Parameters:
        model_name: str — sentence-transformers model name (default: "all-MiniLM-L6-v2")
        diversity: float — diversity parameter for MMR (0-1, default: 0.5)
        top_n: int — number of keywords to consider (default: 20)
        device: str | None — device to run model on ('cpu', 'cuda', or None for auto)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        diversity: float = 0.5,
        top_n: int = 20,
        device: str | None = None,
    ):
        """Initialize KeyBERT extractor.

        Parameters:
            model_name: str — sentence-transformers model name
            diversity: float — diversity for Maximal Marginal Relevance (0-1)
            top_n: int — number of candidates to extract
            device: str | None — device to run model on ('cpu', 'cuda', or None for auto-select)
        """
        if not 0 <= diversity <= 1:
            raise ValueError(f"diversity must be in [0, 1], got {diversity}")
        if top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {top_n}")
        self.model_name = model_name
        self.diversity = diversity
        self.top_n = top_n
        self.device = device
        # Lazy initialization of model with thread-safe lock
        self._model = None
        self._model_lock = threading.Lock()

    @property
    def name(self) -> str:
        """Return extractor identifier."""
        return "keybert"

    def _get_model(self) -> KeyBERTModel:
        """Get or initialize KeyBERT model (lazy loading, thread-safe)."""
        if self._model is None:
            with self._model_lock:
                # Double-check pattern: another thread may have initialized while waiting
                if self._model is None:
                    from sentence_transformers import SentenceTransformer
                    import torch

                    # Set default device to avoid GPU allocation when CPU is requested
                    # This prevents OOM errors on small GPUs when device='cpu' is specified
                    if self.device == "cpu":
                        with torch.device("cpu"):
                            st_model = SentenceTransformer(
                                self.model_name, device=self.device
                            )
                            self._model = KeyBERTModel(model=st_model)
                    else:
                        # For GPU or auto-detect, let SentenceTransformer handle device
                        st_model = SentenceTransformer(
                            self.model_name, device=self.device
                        )
                        self._model = KeyBERTModel(model=st_model)
        return self._model

    def extract(self, text: str, max_keywords: int = 20) -> List[ScoredKeyword]:
        """Extract keywords using KeyBERT.

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
        # Get model
        model = self._get_model()
        # Extract keywords with MMR for diversity
        # keyphrase_ngram_range controls phrase length
        keywords = model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=min(self.top_n, max_keywords),
            use_mmr=True,
            diversity=self.diversity,
        )
        # Handle empty results or if keywords is empty
        if not keywords:
            logger.info(
                "KeyBERT: no keywords extracted from text",
                extra={
                    "text_length": len(text),
                },
            )
            return []
        # Keywords are returned as (keyword, score) tuples
        # Scores are cosine similarities in [0, 1], with 1 being most similar
        results = []
        for keyword, score in keywords:
            # Scores are already in [0, 1] range
            results.append(ScoredKeyword(keyword=keyword, score=float(score)))
        # Sort by score descending (KeyBERT already returns sorted, but be explicit)
        results.sort(key=lambda x: x.score, reverse=True)
        final_results = results[:max_keywords]
        logger.info(
            f"KeyBERT: extracted {len(final_results)} keywords",
            extra={
                "top_3": [kw.keyword for kw in final_results[:3]],
            },
        )
        return final_results

    def healthy(self) -> bool:
        """Check if extractor is ready (model can be loaded)."""
        try:
            self._get_model()
            return True
        except Exception as exc:
            logger.warning(
                "KeyBERT extractor failed health check",
                extra={
                    "error": str(exc),
                    "model": self.model_name,
                },
            )
            return False
