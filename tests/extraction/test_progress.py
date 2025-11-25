"""Tests for extraction progress display."""

from interaction_finder.extraction.progress import (
    DummyProgress,
    create_extraction_progress,
)


def test_create_extraction_progress():
    """create_extraction_progress returns configured StatusTable."""
    progress = create_extraction_progress()
    # Check expected counters exist
    assert progress["Processed"] is not None
    assert progress["Entities"] is not None
    assert progress["Quotes"] is not None
    assert progress["Pairs assessed"] is not None
    assert progress["Found"] is not None
    assert progress["Regions"] is not None
    assert progress["Pairs added"] is not None
    assert progress["Unique pairs"] is not None
    assert progress["Accepted"] is not None
    assert progress["Rejected"] is not None


def test_extraction_progress_counters():
    """Extraction counters have correct configuration."""
    progress = create_extraction_progress()
    # Documents section: Processed is 3-part
    assert progress["Processed"].in_progress == 0
    # Entities, Quotes are 1-part
    assert progress["Entities"].in_progress is None
    assert progress["Quotes"].in_progress is None
    # Pairs assessed is 3-part
    assert progress["Pairs assessed"].in_progress == 0
    # Sweep section: Found is 1-part, Regions is 3-part, Pairs added is 1-part
    assert progress["Found"].in_progress is None
    assert progress["Regions"].in_progress == 0
    assert progress["Pairs added"].in_progress is None
    # Judgment section: Unique pairs is 3-part, Accepted/Rejected are 1-part
    assert progress["Unique pairs"].in_progress == 0
    assert progress["Accepted"].in_progress is None
    assert progress["Rejected"].in_progress is None


def test_extraction_progress_categories():
    """Extraction counters have correct categories."""
    progress = create_extraction_progress()
    # Documents section
    assert progress["Processed"].category == "Documents"
    assert progress["Entities"].category == "Documents"
    assert progress["Quotes"].category == "Documents"
    assert progress["Pairs assessed"].category == "Documents"
    # Co-mention Sweep section
    assert progress["Found"].category == "Co-mention Sweep"
    assert progress["Regions"].category == "Co-mention Sweep"
    assert progress["Pairs added"].category == "Co-mention Sweep"
    # Combined Judgment section
    assert progress["Unique pairs"].category == "Combined Judgment"
    assert progress["Accepted"].category == "Combined Judgment"
    assert progress["Rejected"].category == "Combined Judgment"


def test_extraction_progress_context_manager():
    """Extraction progress works as context manager."""
    progress = create_extraction_progress()
    with progress:
        progress["Processed"].total = 10
        progress["Processed"].activate()
        progress["Processed"].work()
        progress["Processed"].done()
    assert progress["Processed"].completed == 1


def test_extraction_progress_typical_usage():
    """Simulate typical extraction progress updates."""
    progress = create_extraction_progress()
    # Document processing
    progress["Processed"].total = 10
    progress["Processed"].activate()
    progress.set_status("Extracting entities")
    # Process document
    progress["Processed"].work()
    progress["Entities"].add(25)
    progress["Quotes"].completed = 15
    progress["Quotes"].note = "(2 invalid)"
    progress["Processed"].done()
    assert progress["Processed"].completed == 1
    assert progress["Entities"].completed == 25
    # Pairs
    progress["Pairs assessed"].total = 50
    progress["Pairs assessed"].in_progress = 10
    progress["Pairs assessed"].activate()
    progress["Pairs assessed"].completed = 40
    progress["Pairs assessed"].in_progress = 0
    # Sweep
    progress["Found"].completed = 15
    progress["Found"].note = "(10 new, 5 uncovered)"
    progress["Regions"].total = 5
    progress["Regions"].activate()
    progress["Regions"].completed = 5
    progress["Pairs added"].completed = 3
    progress["Pairs added"].total = 8
    # Judgment
    progress["Unique pairs"].total = 20
    progress["Unique pairs"].activate()
    progress["Unique pairs"].work()
    progress["Accepted"].add()
    progress["Unique pairs"].done()
    assert progress["Accepted"].completed == 1


def test_extraction_progress_notes():
    """Progress counters can have dynamic notes."""
    progress = create_extraction_progress()
    progress["Quotes"].completed = 100
    progress["Quotes"].note = "(5 invalid)"
    assert progress["Quotes"].note == "(5 invalid)"
    progress["Found"].completed = 15
    progress["Found"].note = "(10 new, 5 uncovered)"
    assert progress["Found"].note == "(10 new, 5 uncovered)"


def test_dummy_progress_no_op():
    """DummyProgress accepts all operations without error."""
    progress = DummyProgress()
    progress.start()
    progress.stop()
    progress.update()
    progress.set_status("test")
    progress.succeed()
    progress.fail()
    progress.nonexistent_method()
    with progress:
        progress.start()
