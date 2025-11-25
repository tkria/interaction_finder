"""Live progress display for extraction operations.

Provides a factory function for creating a StatusTable configured for
entity extraction with documents, sweep, and judgment counters.
"""

from interaction_finder.progress import Counter, DummyProgress, StatusTable


def create_extraction_progress() -> StatusTable:
    """Create a StatusTable configured for entity extraction.

    Counters:
    - Processed: 3-part counter for document processing (category: Documents)
    - Entities: 1-part counter (category: Documents)
    - Quotes: 1-part counter with note for invalid count (category: Documents)
    - Pairs assessed: 3-part counter (category: Documents)
    - Found: 1-part counter with note for breakdown (category: Co-mention Sweep)
    - Regions: 3-part counter (category: Co-mention Sweep)
    - Pairs added: 2-part counter (category: Co-mention Sweep)
    - Unique pairs: 3-part counter (category: Combined Judgment)
    - Accepted: 1-part counter (category: Combined Judgment)
    - Rejected: 1-part counter (category: Combined Judgment)
    """
    return StatusTable(
        Counter("Processed", track_in_progress=True, category="Documents"),
        Counter("Entities", category="Documents"),
        Counter("Quotes", category="Documents"),
        Counter("Pairs assessed", track_in_progress=True, category="Documents"),
        Counter("Found", category="Co-mention Sweep"),
        Counter("Regions", track_in_progress=True, category="Co-mention Sweep"),
        Counter("Pairs added", category="Co-mention Sweep"),
        Counter("Unique pairs", track_in_progress=True, category="Combined Judgment"),
        Counter("Accepted", category="Combined Judgment"),
        Counter("Rejected", category="Combined Judgment"),
    )


__all__ = ["create_extraction_progress", "DummyProgress"]
