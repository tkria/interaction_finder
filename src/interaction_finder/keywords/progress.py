"""Live progress display for keyword extraction operations.

Provides a factory function for creating a LiveStatusTable configured for
keyword extraction with searches, documents, and keywords counters.
"""

from interaction_finder.progress import Counter, DummyProgress, StatusTable


def create_keywords_progress(table: StatusTable) -> StatusTable:
    """Add the keyword-stage counters to ``table`` and return it.

    Counters:
    - Round: current/max round indicator (category: Search)
    - Searches run: 1-part counter (category: Search)
    - Results found: 1-part counter (category: Search)
    - Fetched: documents downloaded from URLs (category: Documents)
    - Processed: documents with keywords extracted and evaluated (category: Documents)
    - Keywords: bridging terms identified (category: Keywords)
    """
    table.add_counters(
        Counter("Round", category="Search"),
        Counter("Searches run", category="Search"),
        Counter("Results found", category="Search"),
        Counter("Fetched", track_in_progress=True, category="Documents"),
        Counter("Processed", track_in_progress=True, category="Documents"),
        Counter("Keywords", track_in_progress=True, category="Keywords"),
    )
    return table


__all__ = ["create_keywords_progress", "DummyProgress"]
