"""Live progress display for keyword extraction operations.

Provides a factory function for creating a LiveStatusTable configured for
keyword extraction with searches, documents, and keywords counters.
"""

from interaction_finder.progress import Counter, DummyProgress, LiveStatusTable


def create_keywords_progress() -> LiveStatusTable:
    """Create a LiveStatusTable configured for keyword extraction.

    Counters:
    - Round: current/max round indicator (category: Search)
    - Searches run: 1-part counter (category: Search)
    - Results found: 1-part counter (category: Search)
    - Documents: 3-part counter for document processing (category: Documents)
    - Keywords: 3-part counter for keyword evaluation (category: Keywords)
    """
    return LiveStatusTable(
        Counter("Round", category="Search"),
        Counter("Searches run", category="Search"),
        Counter("Results found", category="Search"),
        Counter("Documents", track_in_progress=True, category="Documents"),
        Counter("Keywords", track_in_progress=True, category="Keywords"),
    )


__all__ = ["create_keywords_progress", "DummyProgress"]
