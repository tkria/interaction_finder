"""Per-run mutable state for widesearch pipeline.

State encapsulates all mutable data shared across nodes during a single
search session. Nodes read and write state fields to coordinate the
multi-round query expansion workflow.
"""

from dataclasses import dataclass, field

from interaction_finder.search.models import SearchResult


@dataclass
class State:
    """Mutable state shared across all nodes in a single widesearch run.

    Required fields must be provided at initialization. Other fields are
    set during pipeline execution as the workflow progresses.
    """

    # Required initialization (from entry point)
    topic: str  # Research topic to search for
    keyphrases: list[str]  # Selected keyphrases to incorporate in queries
    max_rounds: int = 5  # Maximum search rounds before forced termination

    # Set during execution (nodes update these)
    current_round: int = 0  # Current search round (incremented in GenerateQueries)
    subject_goals: list[str] = field(
        default_factory=list
    )  # Subject areas to cover (set in PlanGoals)
    satisfied_goals: list[str] = field(
        default_factory=list
    )  # Goals marked as satisfied (updated in Reflect)
    search_summaries: list[str] = field(
        default_factory=list
    )  # Per-round coverage summaries (from SelectResults)
    all_queries: list[str] = field(
        default_factory=list
    )  # All queries executed (cumulative)
    current_queries: list[str] = field(
        default_factory=list
    )  # Queries for current round (from GenerateQueries)
    current_results: list[SearchResult] = field(
        default_factory=list
    )  # Results from current round (from Search)
    selected_results: dict[str, list[str]] = field(
        default_factory=dict
    )  # query -> list[url] mapping
    should_continue: bool = True  # Early stopping flag (set by Reflect)
