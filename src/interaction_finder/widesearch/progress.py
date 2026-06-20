"""Live progress display for widesearch operations.

Provides a factory function for creating a LiveStatusTable configured for
widesearch with searches, results, and selection counters.
"""

from interaction_finder.progress import Counter, DummyProgress, StatusTable


def create_widesearch_progress(table: StatusTable) -> StatusTable:
    """Add the widesearch-stage counters to ``table`` and return it.

    Counters:
    - Round: current/max round indicator (category: Search)
    - Searches run: 3-part counter for search progress (category: Search)
    - Results found: 1-part counter (category: Search)
    - Results selected: 1-part counter (category: Selection)
    """
    table.add_counters(
        Counter("Round", category="Search"),
        Counter("Searches run", track_in_progress=True, category="Search"),
        Counter("Results found", category="Search"),
        Counter("Results selected", category="Selection"),
    )
    return table


__all__ = ["create_widesearch_progress", "DummyProgress"]
