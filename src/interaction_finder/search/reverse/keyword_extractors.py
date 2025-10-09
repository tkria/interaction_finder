"""
Domain-agnostic keyword extraction algorithms for query generation.

This module provides a unified interface for multiple keyword extraction
approaches (YAKE, RAKE, TF-IDF, LLM) suitable for processing scientific literature
of any domain.

Order: ABC interface → implementations → factory
"""

import asyncio
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from rich.console import Console

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


class LLMExtractor(KeywordExtractor):
    """
    LLM-based query generation using Pydantic AI.

    Uses a language model to generate precision-focused search queries from
    paper content. Automatically falls back to YAKE on any LLM failures.

    Parameters:
        model: str - Pydantic AI model string (e.g., "openai:gpt-4o-mini")
        temperature: float - LLM temperature for generation (0.0-1.0)
        backend_specific: bool - Whether to generate backend-specific syntax (default True)
        console: Optional[Console] - Rich console for logging (default None)
    """

    def __init__(
        self,
        model: str = "openai:gpt-4o-mini",
        temperature: float = 0.3,
        backend_specific: bool = True,
        console: Optional[Console] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.backend_specific = backend_specific
        self.console = console
        self._agent = None

    @property
    def name(self) -> str:
        return "llm"

    def _create_agent(self):
        """
        Create Pydantic AI agent for query generation.

        Lazy initialization to avoid import overhead if LLM extractor not used.
        """
        from pydantic_ai import Agent
        from .llm_models import LLMQueryResponse
        from .prompts import (
            QUERY_GENERATION_PROMPT_PUBMED,
            QUERY_GENERATION_PROMPT_NATURAL,
        )

        # Select appropriate system prompt based on backend_specific setting
        system_prompt = (
            QUERY_GENERATION_PROMPT_PUBMED
            if self.backend_specific
            else QUERY_GENERATION_PROMPT_NATURAL
        )

        # Create agent with structured output
        agent = Agent(
            model=self.model,
            output_type=LLMQueryResponse,
            system_prompt=system_prompt,
        )

        return agent

    def extract(
        self, text: str, top_n: int, hint_fields: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Extract queries synchronously (wrapper for async implementation).

        Parameters:
            text: str - Input text to generate queries from
            top_n: int - Maximum number of queries to return (1-3)
            hint_fields: Optional[Dict[str, Any]] - Hint fields for LLM guidance

        Returns:
            List[str] - Generated queries (or YAKE fallback on error)
        """
        # Run async version in event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop in current thread, create new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.extract_async(text, top_n, hint_fields))

    async def extract_async(
        self, text: str, top_n: int, hint_fields: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Extract queries asynchronously using LLM.

        Parameters:
            text: str - Input text to generate queries from
            top_n: int - Maximum number of queries to return (1-3)
            hint_fields: Optional[Dict[str, Any]] - Hint fields for LLM guidance

        Returns:
            List[str] - Generated queries (or YAKE fallback on error)
        """
        if top_n <= 0:
            raise ValueError("top_n must be > 0")
        if not text or not text.strip():
            return []

        # Lazy agent initialization
        if self._agent is None:
            self._agent = self._create_agent()

        # Build user prompt with text and hint fields
        from .prompts import build_query_generation_prompt

        user_prompt = build_query_generation_prompt(
            text=text,
            hint_fields=hint_fields,
            backend_specific=self.backend_specific,
        )

        # Attempt LLM query generation with fallback to YAKE on any error
        try:
            response = await self._agent.run(user_prompt)
            queries = response.output.queries[:top_n]

            if self.console:
                self.console.print(
                    f"[green]LLM generated {len(queries)} queries[/green]"
                )

            return queries

        except Exception as e:
            # Log error and fallback to YAKE
            if self.console:
                self.console.print(
                    f"[yellow]LLM query generation failed ({type(e).__name__}: {str(e)}). "
                    f"Falling back to YAKE.[/yellow]"
                )

            # Fallback to YAKE extractor
            fallback = YAKEExtractor()
            return fallback.extract(text, top_n)


# ==============================================================================
# Factory Function
# ==============================================================================


def create_extractor(name: str, **kwargs) -> KeywordExtractor:
    """
    Create keyword extractor by name.

    Parameters:
        name: str - Extractor name ("yake", "rake", "tfidf", "llm"), case-insensitive
        **kwargs: Additional parameters for extractor initialization
            For LLM: model, temperature, backend_specific, console

    Returns:
        KeywordExtractor instance

    Raises:
        ValueError: If name is not recognized

    Example:
        >>> extractor = create_extractor("yake")
        >>> keywords = extractor.extract("Machine learning and AI", top_n=3)
        >>> llm_extractor = create_extractor("llm", model="openai:gpt-4o-mini")
        >>> queries = llm_extractor.extract("BRCA1 breast cancer", top_n=3)
    """
    name_lower = name.lower()
    if name_lower == "yake":
        return YAKEExtractor()
    elif name_lower == "rake":
        return RAKEExtractor()
    elif name_lower == "tfidf":
        return TFIDFExtractor()
    elif name_lower == "llm":
        return LLMExtractor(**kwargs)
    else:
        raise ValueError(
            f"Unknown extractor name: {name}. Supported: yake, rake, tfidf, llm"
        )
