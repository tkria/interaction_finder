"""
Query construction abstraction for Stage 2 of the two-stage pipeline.

This module defines the QueryConstructor interface that separates query
construction from keyword extraction. Different constructors can use the
same extracted keywords to build queries in different ways (direct keyword
assembly, LLM-based generation with full context, etc.).

The two-stage pipeline:
    Stage 1: Keyword Extraction - Extract keywords from resource content
    Stage 2: Query Construction - Build search queries from extracted keywords

This separation enables:
- Mixing extractors with constructors (YAKE+Direct, None+LLM, etc.)
- Clear investigation logging of each stage
- Independent testing and optimization of each stage
- Flexibility for future constructor strategies

Order: ABC interface → concrete implementations → factory function
"""

from abc import ABC, abstractmethod
from typing import Optional

from pydantic_ai import Agent
from rich.console import Console

from interaction_finder.search.reverse.llm_models import LLMQueryConstructionResponse
from interaction_finder.search.reverse.models import (
    QueryConstructionContext,
    QueryGenerationError,
)
from interaction_finder.search.reverse.prompts import QUERY_CONSTRUCTION_PROMPT


# ==============================================================================
# Abstract Interface
# ==============================================================================


class QueryConstructor(ABC):
    """
    Abstract interface for query construction strategies.

    Query constructors take the output of keyword extraction (along with
    other context) and build search queries. Different constructors can
    implement different strategies:
    - Direct assembly from keywords
    - LLM-based generation using full content
    - Template-based with backend-specific syntax
    - Hybrid approaches

    All constructors must be async to support LLM-based implementations.
    """

    @abstractmethod
    async def construct(self, context: QueryConstructionContext) -> str:
        """
        Construct a search query from the provided context.

        Parameters:
            context: QueryConstructionContext - All available information for
                query construction including keywords, scores, hint terms,
                backend target, and optional full content.

        Returns:
            str - Constructed search query ready for backend execution.
                Must be non-empty.

        Raises:
            ValueError: If context is invalid or query cannot be constructed.

        Example:
            >>> context = QueryConstructionContext(
            ...     keywords=["BRCA1", "breast cancer"],
            ...     keyword_scores=[0.95, 0.87],
            ...     hint_terms=["TP53"],
            ...     backend="pubmed",
            ...     resource_content=None,
            ...     extractor_used="yake"
            ... )
            >>> query = await constructor.construct(context)
            >>> # Direct constructor might produce: "BRCA1 breast cancer TP53"
            >>> # LLM constructor might produce: "BRCA1[Title/Abstract] AND (breast cancer OR mammary carcinoma)"
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return constructor name for logging and debugging.

        Returns:
            str - Short, lowercase name (e.g., "direct", "llm", "template")

        Example:
            >>> constructor.name
            'direct'
        """
        pass


# ==============================================================================
# Concrete Implementations
# ==============================================================================


class DirectQueryConstructor(QueryConstructor):
    """
    Simple query constructor that directly concatenates keywords and hint terms.

    This is a minimal fallback constructor that provides basic query assembly
    without LLM enhancement. It's used as a fallback for LLMQueryConstructor
    when LLM calls fail.

    Configuration:
        max_keywords: int - Maximum number of keywords to include (default: 10)
        include_hints: bool - Whether to include hint terms (default: True)
    """

    def __init__(self, max_keywords: int = 10, include_hints: bool = True):
        """
        Initialize DirectQueryConstructor.

        Parameters:
            max_keywords: int - Maximum keywords to include (default: 10)
            include_hints: bool - Whether to include hint terms (default: True)
        """
        self._max_keywords = max_keywords
        self._include_hints = include_hints

    async def construct(self, context: QueryConstructionContext) -> str:
        """
        Construct query by concatenating keywords and hint terms with OR logic.

        Parameters:
            context: QueryConstructionContext - Construction context

        Returns:
            str - OR-joined query string (empty if no keywords/hints available)
        """
        # Collect query terms
        terms = []

        # Add keywords (up to max_keywords)
        if context.keywords:
            terms.extend(context.keywords[: self._max_keywords])

        # Add hint terms if configured
        if self._include_hints and context.hint_terms:
            terms.extend(context.hint_terms)

        # Return empty string if no terms
        if not terms:
            return ""

        # Check if we have a single complete query (contains field tags)
        # Complete queries are already formatted and should be used as-is
        if len(terms) == 1 and self._is_complete_query(terms[0]):
            return terms[0]

        # Construct boolean OR query with quoted keywords for exact matching
        quoted_keywords = [f'"{kw}"' for kw in terms if kw]
        return " OR ".join(quoted_keywords)

    def _is_complete_query(self, term: str) -> bool:
        """
        Check if a term is a complete query with field tags.

        Complete queries contain PubMed field tags like [Title], [Abstract], etc.
        and should be used as-is without additional quoting or processing.

        Parameters:
            term: str - Term to check

        Returns:
            bool - True if term is a complete query with field tags
        """
        # Check for PubMed field tags (e.g., [Title], [Abstract], [Author], etc.)
        return "[" in term and "]" in term

    @property
    def name(self) -> str:
        """Return constructor name."""
        return "direct"


class LLMQueryConstructor(QueryConstructor):
    """
    LLM-based query constructor with maximum-context prompting.

    This constructor uses a Pydantic AI agent to build sophisticated queries
    from extracted keywords, keyword scores, hint terms, and optionally full
    resource content. It provides intelligent query optimization while optionally
    falling back to DirectQueryConstructor on LLM failures.

    The "maximum context" approach means the LLM receives BOTH extracted keywords
    (structural guidance) AND full content (semantic context) to enable hybrid
    strategies like "YAKE extraction + LLM construction".

    Configuration:
        model: str - LLM model identifier (e.g., "openai:gpt-4o-mini")
        temperature: float - Sampling temperature (default: 0.3)
        backend_specific: bool - Whether to use backend-specific syntax (default: True)
        enable_fallback: bool - Whether to fall back to DirectQueryConstructor on errors (default: True)
        include_scores: bool - Whether to include keyword scores in prompt (default: True)
        max_keywords: int - Maximum keywords to include in prompt (default: 15)
        console: Optional[Console] - Rich console for logging (optional)

    Example:
        >>> constructor = LLMQueryConstructor(
        ...     model="openai:gpt-4o-mini",
        ...     temperature=0.3,
        ...     enable_fallback=True
        ... )
        >>> context = QueryConstructionContext(
        ...     keywords=["BRCA1", "breast cancer"],
        ...     keyword_scores=[0.05, 0.12],
        ...     hint_terms=["mammary epithelial cell"],
        ...     backend="pubmed",
        ...     resource_content="BRCA1 mutations confer...",
        ...     extractor_used="yake"
        ... )
        >>> query = await constructor.construct(context)
        >>> # Produces sophisticated query like:
        >>> # 'BRCA1[Title/Abstract] AND ("breast cancer" OR "mammary carcinoma")'
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.3,
        backend_specific: bool = True,
        enable_fallback: bool = True,
        include_scores: bool = True,
        max_keywords: int = 15,
        console: Optional[Console] = None,
    ):
        """
        Initialize LLMQueryConstructor.

        Parameters:
            model: str - LLM model identifier
            temperature: float - Sampling temperature (default: 0.3)
            backend_specific: bool - Use backend-specific syntax (default: True)
            enable_fallback: bool - Fall back to DirectQueryConstructor on errors (default: True)
            include_scores: bool - Include keyword scores in prompt (default: True)
            max_keywords: int - Maximum keywords to include (default: 15)
            console: Optional[Console] - Rich console for logging (optional)
        """
        self._model = model
        self._temperature = temperature
        self._backend_specific = backend_specific
        self._enable_fallback = enable_fallback
        self._include_scores = include_scores
        self._max_keywords = max_keywords
        self._console = console

        # Lazy agent initialization (created on first use)
        self._agent: Optional[Agent] = None

        # Create fallback constructor if enabled
        self._fallback: Optional[DirectQueryConstructor] = None
        if self._enable_fallback:
            self._fallback = DirectQueryConstructor(
                max_keywords=max_keywords, include_hints=True
            )

    def _create_agent(self) -> Agent:
        """
        Create Pydantic AI agent for query construction.

        Returns:
            Agent configured for LLMQueryConstructionResponse output
        """
        agent = Agent(
            model=self._model,
            output_type=LLMQueryConstructionResponse,
            system_prompt=QUERY_CONSTRUCTION_PROMPT,
        )
        return agent

    def _build_prompt(self, context: QueryConstructionContext) -> str:
        """
        Build user prompt with maximum context.

        Includes all available information: keywords, scores, hint terms,
        full content, extractor name, and backend-specific instructions.

        Parameters:
            context: QueryConstructionContext - Construction context

        Returns:
            str - Formatted user prompt
        """
        sections = []

        # Section 1: Extracted keywords with optional scores
        keywords_section = "EXTRACTED KEYWORDS:\n"
        if context.keywords:
            # Truncate to max_keywords if needed
            keywords_to_show = context.keywords[: self._max_keywords]
            scores_to_show = (
                context.keyword_scores[: self._max_keywords]
                if context.keyword_scores
                else None
            )

            if self._include_scores and scores_to_show:
                # Show keywords with scores
                for kw, score in zip(keywords_to_show, scores_to_show):
                    keywords_section += f"  • {kw} (score: {score:.3f})\n"
            else:
                # Show keywords without scores
                for kw in keywords_to_show:
                    keywords_section += f"  • {kw}\n"

            # Add extractor context for score interpretation
            keywords_section += (
                f"\nExtractor: {context.extractor_used} "
                f"(keywords extracted using {context.extractor_used.upper()} algorithm)\n"
            )

            sections.append(keywords_section)
        else:
            sections.append("EXTRACTED KEYWORDS: (none provided)\n")

        # Section 2: Hint terms
        if context.hint_terms:
            hints_section = "\nHINT TERMS (domain-specific metadata):\n"
            for hint in context.hint_terms:
                hints_section += f"  • {hint}\n"
            sections.append(hints_section)

        # Section 3: Full resource content (if available)
        if context.resource_content:
            # Truncate content to first 2000 characters for efficiency
            content = context.resource_content[:2000]
            truncated_note = (
                " [truncated to first 2000 chars]"
                if len(context.resource_content) > 2000
                else ""
            )
            content_section = f"\nFULL RESOURCE CONTENT{truncated_note}:\n{content}\n"
            sections.append(content_section)

        # Section 4: Backend-specific instructions
        if self._backend_specific:
            backend_section = f"\nTARGET BACKEND: {context.backend}\n"
            if context.backend.lower() == "pubmed":
                backend_section += "Use PubMed field tags: [Title], [Abstract], [MeSH], [Author], etc.\n"
            sections.append(backend_section)
        else:
            sections.append(
                f"\nTARGET BACKEND: {context.backend} (use natural language syntax)\n"
            )

        # Section 5: Final instruction
        final_instruction = (
            "\nConstruct ONE optimized search query that:\n"
            "1. Uses extracted keywords as ANCHORS\n"
            "2. Enhances with content understanding (if available)\n"
            "3. Integrates relevant hint terms\n"
            "4. Applies appropriate backend syntax\n"
        )
        sections.append(final_instruction)

        return "".join(sections)

    async def construct(self, context: QueryConstructionContext) -> str:
        """
        Construct query using LLM with fallback support.

        Parameters:
            context: QueryConstructionContext - Construction context

        Returns:
            str - Constructed search query

        Raises:
            QueryGenerationError: If LLM fails and fallback disabled
        """
        # Lazy agent initialization
        if self._agent is None:
            self._agent = self._create_agent()

        # Build prompt with maximum context
        user_prompt = self._build_prompt(context)

        try:
            # Call LLM agent
            result = await self._agent.run(user_prompt)
            response = result.output

            # Extract and return query
            return response.query.strip()

        except Exception as e:
            # Log error
            error_msg = f"LLM query construction failed: {type(e).__name__}: {e}"

            if self._enable_fallback:
                # Attempt fallback to DirectQueryConstructor
                if self._console:
                    self._console.print(
                        f"[yellow]⚠ {error_msg}[/yellow]", style="yellow"
                    )
                    self._console.print(
                        "[yellow]→ Falling back to DirectQueryConstructor[/yellow]"
                    )

                # Use fallback
                return await self._fallback.construct(context)
            else:
                # No fallback - raise error
                raise QueryGenerationError(
                    f"LLM query construction failed (fallback disabled): {error_msg}",
                    context={
                        "model": self._model,
                        "extractor": context.extractor_used,
                        "keywords_count": len(context.keywords),
                        "has_content": context.resource_content is not None,
                        "error": str(e),
                    },
                ) from e

    @property
    def name(self) -> str:
        """Return constructor name."""
        return "llm"


# ==============================================================================
# Factory Function
# ==============================================================================


def create_constructor(name: str, **kwargs) -> QueryConstructor:
    """
    Create query constructor by name.

    Factory function for creating constructor instances. Supports dependency
    injection through kwargs for constructor-specific configuration.

    Parameters:
        name: str - Constructor name (case-insensitive)
        **kwargs: Additional parameters for constructor initialization
            Specific to each constructor type

    Returns:
        QueryConstructor instance

    Raises:
        ValueError: If name is not recognized

    Example:
        >>> # Direct constructor (simple keyword concatenation)
        >>> direct = create_constructor("direct", max_keywords=10)
        >>> query = await direct.construct(context)
        >>>
        >>> # LLM constructor with fallback
        >>> llm = create_constructor(
        ...     "llm",
        ...     model="openai:gpt-4o-mini",
        ...     temperature=0.3,
        ...     enable_fallback=True
        ... )
        >>> query = await llm.construct(context)
    """
    name_lower = name.lower()

    if name_lower == "direct":
        return DirectQueryConstructor(**kwargs)
    elif name_lower == "llm":
        return LLMQueryConstructor(**kwargs)
    else:
        raise ValueError(
            f"Unknown constructor name: {name}. Available constructors: 'direct', 'llm'"
        )
