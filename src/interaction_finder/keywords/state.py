"""Per-run mutable state for keyword research pipeline."""

from dataclasses import dataclass, field

from interaction_finder.keywords.models import DocumentSummaryOut
from interaction_finder.keywords.extractors import ScoredKeyword
from interaction_finder.search.models import SearchResult


@dataclass
class State:
    """Mutable state shared across all nodes in a single keyword research run.

    Required fields must be provided at initialization (usually from the
    starting node). Other fields are set during pipeline execution.
    """

    # Required initialization (from starting node)
    topic: str  # Research topic to find bridging terms for
    max_rounds: int = 5  # Maximum search rounds before stopping

    # Set during execution
    current_round: int = 0  # Current search round (incremented in ExpandQuery)
    search_queries: list[str] = field(
        default_factory=list
    )  # Queries generated this round
    all_search_results: list[SearchResult] = field(
        default_factory=list
    )  # All results from current round
    selected_results: list[SearchResult] = field(
        default_factory=list
    )  # Results selected for fetching
    fetched_content: dict[str, str] = field(
        default_factory=dict
    )  # URL -> markdown content
    document_summaries: list[DocumentSummaryOut] = field(
        default_factory=list
    )  # All document summaries (cumulative across rounds)
    extracted_keywords: dict[str, list[ScoredKeyword]] = field(
        default_factory=dict
    )  # URL -> keywords
    final_bridging_terms: list[str] = field(
        default_factory=list
    )  # Deduplicated final output
