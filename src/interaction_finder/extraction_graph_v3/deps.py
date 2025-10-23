"""
Dependency injection for extraction graph V3.

Extends V2 deps with:
- Fine-grained parallelism control (extraction, assessment, pair evaluation)
- Co-occurrence strategy configuration (tiered, same-kind pairs)
- Quote validation configuration (fuzzy matching, LLM fallback)
- Semantic caching configuration
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Union, Optional, List, Literal, TYPE_CHECKING
from pathlib import Path

from pydantic_ai.models import Model

from ..settings import IfetcherConfig
from ..fetcher import PageFetcher

if TYPE_CHECKING:
    from ..resources import Resource
    from ..extraction_graph_v2.models import QuoteErrorRecord


@dataclass
class ExtractionDepsV3:
    """
    External dependencies for V3 extraction pipeline.

    Extends V2 deps with configurable parallelism, co-occurrence strategies,
    quote validation, and semantic caching.
    """

    # LLM for agents
    model: Union[Model, str]

    # Configuration
    config: IfetcherConfig

    # Document fetching service
    page_fetcher: PageFetcher

    # Target term for extraction context
    target_term: Optional[str] = None

    # Parallelism control (0 = unlimited, >0 = max concurrent operations)
    extraction_parallelism: int = 0
    assessment_parallelism: int = 0
    pair_evaluation_parallelism: int = 0

    # Co-occurrence strategy configuration
    cooccurrence_strategy: Literal["tiered", "all"] = "tiered"
    enable_same_chunk: bool = True
    enable_adjacent_chunks: bool = True
    enable_document_level: bool = False  # Only if needed
    include_same_kind_pairs: bool = False  # Filter out same-kind pairs by default

    # Quote validation configuration
    fuzzy_matching_enabled: bool = True
    fuzzy_auto_correct_threshold: float = 0.9  # Auto-accept above this
    fuzzy_suggest_threshold: float = 0.75  # Suggest corrections above this
    enable_llm_fallback: bool = True  # Use LLM to fix quotes if fuzzy fails
    quote_similarity_threshold: float = 0.8  # Minimum similarity when matching quotes

    # Semantic caching configuration
    semantic_cache_enabled: bool = False  # Disabled by default until task 02

    # Quote error tracking
    quote_error_log: List["QuoteErrorRecord"] = field(default_factory=list)

    # Track which errors have been saved to avoid duplication
    saved_error_count: int = 0

    # Incremental output directory (optional)
    output_dir: Optional[Path] = None

    # Checkpoint callback for incremental saves (optional)
    checkpoint_callback: Optional[Callable] = None  # type: ignore[type-arg]

    @classmethod
    def from_config(
        cls,
        config: IfetcherConfig,
        page_fetcher: PageFetcher,
        model: Union[Model, str, None] = None,
        target_term: Optional[str] = None,
        # Parallelism
        extraction_parallelism: int = 0,
        assessment_parallelism: int = 0,
        pair_evaluation_parallelism: int = 0,
        # Co-occurrence
        cooccurrence_strategy: Literal["tiered", "all"] = "tiered",
        enable_same_chunk: bool = True,
        enable_adjacent_chunks: bool = True,
        enable_document_level: bool = False,
        include_same_kind_pairs: bool = False,
        # Quote validation
        fuzzy_matching_enabled: bool = True,
        fuzzy_auto_correct_threshold: float = 0.9,
        fuzzy_suggest_threshold: float = 0.75,
        enable_llm_fallback: bool = True,
        quote_similarity_threshold: float = 0.8,
        # Caching
        semantic_cache_enabled: bool = False,
        # Other
        output_dir: Optional[Path] = None,
        checkpoint_callback: Optional[Callable] = None,  # type: ignore[type-arg]
    ) -> "ExtractionDepsV3":
        """
        Create deps from configuration with sensible defaults.

        Args:
            config: Main configuration
            page_fetcher: PageFetcher instance
            model: Optional model override
            target_term: Optional target term for extraction context
            extraction_parallelism: Max concurrent extraction calls (0 = unlimited)
            assessment_parallelism: Max concurrent assessment calls (0 = unlimited)
            pair_evaluation_parallelism: Max concurrent pair evaluation calls (0 = unlimited)
            cooccurrence_strategy: Strategy for candidate generation ('tiered' or 'all')
            enable_same_chunk: Generate candidates from same-chunk co-occurrence
            enable_adjacent_chunks: Generate candidates from adjacent-chunk co-occurrence
            enable_document_level: Generate candidates from document-level co-occurrence
            include_same_kind_pairs: Whether to include pairs of entities with same kind
            fuzzy_matching_enabled: Enable fuzzy quote matching
            fuzzy_auto_correct_threshold: Auto-accept threshold for fuzzy matching
            fuzzy_suggest_threshold: Suggestion threshold for fuzzy matching
            enable_llm_fallback: Use LLM to fix quotes if fuzzy fails
            semantic_cache_enabled: Enable semantic caching (task 02)
            current_resources: Optional list of resources for validation
            output_dir: Optional output directory for incremental saves
            checkpoint_callback: Optional callback for incremental checkpoints

        Returns:
            ExtractionDepsV3 instance
        """
        # Determine model from config hierarchy
        if model is None:
            default_agent = config.agents.get("_", config.AgentSpec())  # type: ignore[call-arg]
            model = default_agent.llm or "openai:gpt-4o"

        return cls(
            model=model,
            config=config,
            page_fetcher=page_fetcher,
            target_term=target_term,
            extraction_parallelism=extraction_parallelism,
            assessment_parallelism=assessment_parallelism,
            pair_evaluation_parallelism=pair_evaluation_parallelism,
            cooccurrence_strategy=cooccurrence_strategy,
            enable_same_chunk=enable_same_chunk,
            enable_adjacent_chunks=enable_adjacent_chunks,
            enable_document_level=enable_document_level,
            include_same_kind_pairs=include_same_kind_pairs,
            fuzzy_matching_enabled=fuzzy_matching_enabled,
            fuzzy_auto_correct_threshold=fuzzy_auto_correct_threshold,
            fuzzy_suggest_threshold=fuzzy_suggest_threshold,
            enable_llm_fallback=enable_llm_fallback,
            quote_similarity_threshold=quote_similarity_threshold,
            semantic_cache_enabled=semantic_cache_enabled,
            output_dir=output_dir,
            checkpoint_callback=checkpoint_callback,
        )

    def get_entity_kinds(self) -> List[str]:
        """Get entity kinds from task configuration."""
        return self.config.task.get_kind_names()

    def get_relation_type(self) -> str:
        """Get relation type from task configuration."""
        return self.config.task.relation

    def get_task_context(self) -> str:
        """Get task context from configuration."""
        return self.config.task.context
