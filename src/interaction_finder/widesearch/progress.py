"""Live progress display for widesearch operations.

Provides a factory function for creating a StatusTable configured for
widesearch with searches, results, and selection counters.
"""

from interaction_finder.progress import Counter, DummyProgress, StatusTable


def create_widesearch_progress() -> StatusTable:
    """Create a StatusTable configured for widesearch.

    Counters:
    - Round: current/max round indicator (category: Search)
    - Searches run: 3-part counter for search progress (category: Search)
    - Results found: 1-part counter (category: Search)
    - Results selected: 1-part counter (category: Selection)
    """
    return StatusTable(
        Counter("Round", category="Search"),
        Counter("Searches run", track_in_progress=True, category="Search"),
        Counter("Results found", category="Search"),
        Counter("Results selected", category="Selection"),
    )


__all__ = ["create_widesearch_progress", "DummyProgress"]
