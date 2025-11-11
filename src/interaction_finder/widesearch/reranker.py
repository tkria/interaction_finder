"""Semantic reranking of search results using cross-encoder models.

Reranker uses sentence-transformers CrossEncoder (zerank-1-small) to compute
semantic similarity between queries and search results, improving result
ordering for better relevance.
"""

import math
from typing import List

from sentence_transformers import CrossEncoder

from interaction_finder.logging import logfire
from interaction_finder.search.models import SearchResult


class Reranker:
    """Semantic reranker using cross-encoder models.

    Reranks search results by computing semantic similarity between the query
    and each result. Uses sentence-transformers CrossEncoder for efficient
    pairwise scoring with zerank-1-small model.

    Parameters:
        model_name: str — cross-encoder model name (default: zeroentropy/zerank-1-small)
        batch_size: int — batch size for encoding (default: 1)
        device: str | None — device to run model on ('cpu', 'cuda', or None for auto)
    """

    def __init__(
        self,
        model_name: str = "zeroentropy/zerank-1-small",
        batch_size: int = 1,
        device: str | None = None,
    ):
        """Initialize reranker with cross-encoder model.

        Parameters:
            model_name: str — cross-encoder model for reranking
            batch_size: int — batch size for processing (default: 1 to avoid padding issues)
            device: str | None — device to run model on ('cpu', 'cuda', or None for auto-select)

        Raises:
            ValueError — if batch_size < 1
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        # Lazy initialization - model loaded on first use
        self._model = None

    def _get_model(self) -> CrossEncoder:
        """Get or initialize CrossEncoder model (lazy loading).

        Returns:
            Initialized CrossEncoder instance

        Raises:
            Exception — if model cannot be loaded
        """
        if self._model is None:
            import torch

            # Set default device to avoid GPU allocation when CPU is requested
            # This prevents OOM errors on small GPUs when device='cpu' is specified
            if self.device == "cpu":
                with torch.device("cpu"):
                    self._model = CrossEncoder(
                        self.model_name, max_length=512, device=self.device
                    )
            else:
                self._model = CrossEncoder(
                    self.model_name, max_length=512, device=self.device
                )
        return self._model

    def rerank(
        self, query: str, results: List[SearchResult], top_k: int | None = None
    ) -> List[SearchResult]:
        """Rerank search results by semantic similarity to query.

        Computes semantic similarity scores between query and each result
        (using title + snippet), then sorts by score descending.

        Parameters:
            query: str — search query
            results: List[SearchResult] — results to rerank
            top_k: int | None — return only top k results (None = all)

        Returns:
            List[SearchResult] — reranked results with updated relevance scores

        Raises:
            ValueError — if query is empty, results is empty, or top_k < 1
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if not results:
            raise ValueError("Results cannot be empty")
        if top_k is not None and top_k < 1:
            raise ValueError(f"top_k must be >= 1 or None, got {top_k}")

        with logfire.span(
            "Reranker.rerank",
            num_results=len(results),
            top_k=top_k,
            model=self.model_name,
        ):
            # Get model
            model = self._get_model()

            # Create query-document pairs
            # Use title + snippet for each result
            pairs = []
            for result in results:
                doc_text = result.title
                if result.snippet:
                    doc_text = f"{result.title}. {result.snippet}"
                pairs.append([query, doc_text])

            # Compute scores (use batch_size=1 to avoid padding token issues)
            scores = model.predict(pairs, batch_size=self.batch_size)

            # Create new results with updated relevance scores
            # Scores are logits, normalize to [0, 1] using sigmoid
            reranked = []
            for result, score in zip(results, scores):
                # Apply sigmoid to convert logit to probability
                normalized_score = 1.0 / (1.0 + math.exp(-float(score)))
                # Create new result with updated relevance
                reranked_result = SearchResult(
                    title=result.title,
                    url=result.url,
                    snippet=result.snippet,
                    relevance=normalized_score,
                )
                reranked.append((normalized_score, reranked_result))

            # Sort by score descending
            reranked.sort(key=lambda x: x[0], reverse=True)

            # Return top_k or all
            if top_k is not None:
                reranked = reranked[:top_k]

            final_results = [result for _, result in reranked]

            # Compute score range for logging
            score_range = None
            if final_results:
                min_score = final_results[-1].relevance
                max_score = final_results[0].relevance
                score_range = {"min": min_score, "max": max_score}

            logfire.info(
                f"Reranked {len(results)} results → {len(final_results)} results"
                + (
                    f" (scores: {final_results[0].relevance:.2f}-{final_results[-1].relevance:.2f})"
                    if final_results
                    else ""
                ),
                input_count=len(results),
                output_count=len(final_results),
                score_range=score_range,
                results=final_results,
            )

            return final_results

    def healthy(self) -> bool:
        """Check if reranker is ready (model can be loaded).

        Returns:
            True if model loads successfully, False otherwise
        """
        try:
            self._get_model()
            return True
        except Exception:
            return False
