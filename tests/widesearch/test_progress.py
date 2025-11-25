"""Tests for widesearch progress display."""

from interaction_finder.widesearch.progress import (
    DummyProgress,
    create_widesearch_progress,
)


def test_create_widesearch_progress():
    """create_widesearch_progress returns configured StatusTable."""
    progress = create_widesearch_progress()
    # Check expected counters exist
    assert progress["Round"] is not None
    assert progress["Searches run"] is not None
    assert progress["Results found"] is not None
    assert progress["Results selected"] is not None


def test_widesearch_progress_counters():
    """Widesearch counters have correct configuration."""
    progress = create_widesearch_progress()
    # Round is a 2-part counter (no in_progress tracking)
    assert progress["Round"].in_progress is None
    # Searches run is a 3-part counter
    assert progress["Searches run"].in_progress == 0
    # Results found is 1-part
    assert progress["Results found"].in_progress is None
    # Results selected is 1-part
    assert progress["Results selected"].in_progress is None


def test_widesearch_progress_categories():
    """Widesearch counters have correct categories."""
    progress = create_widesearch_progress()
    assert progress["Round"].category == "Search"
    assert progress["Searches run"].category == "Search"
    assert progress["Results found"].category == "Search"
    assert progress["Results selected"].category == "Selection"


def test_widesearch_progress_context_manager():
    """Widesearch progress works as context manager."""
    progress = create_widesearch_progress()
    with progress:
        progress["Searches run"].total = 10
        progress["Searches run"].activate()
        progress["Searches run"].done()
    assert progress["Searches run"].completed == 1


def test_widesearch_progress_typical_usage():
    """Simulate typical widesearch progress updates."""
    progress = create_widesearch_progress()
    # Start round
    progress["Round"].total = 3
    progress["Round"].completed = 1
    progress["Round"].activate()
    # Start searching
    progress["Searches run"].total = 5
    progress["Searches run"].in_progress = 5
    progress["Searches run"].activate()
    progress.set_status("Searching PubMed")
    # Search completes
    progress["Searches run"].done()
    progress["Results found"].add(10)
    assert progress["Searches run"].completed == 1
    assert progress["Results found"].completed == 10
    # Selection
    progress["Results selected"].activate()
    progress["Results selected"].add(3)
    assert progress["Results selected"].completed == 3


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
