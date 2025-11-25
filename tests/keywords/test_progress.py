"""Tests for keywords progress display."""

from interaction_finder.keywords.progress import DummyProgress, create_keywords_progress


def test_create_keywords_progress():
    """create_keywords_progress returns configured StatusTable."""
    progress = create_keywords_progress()
    # Check expected counters exist
    assert progress["Round"] is not None
    assert progress["Searches run"] is not None
    assert progress["Results found"] is not None
    assert progress["Documents"] is not None
    assert progress["Keywords"] is not None


def test_keywords_progress_counters():
    """Keywords counters have correct configuration."""
    progress = create_keywords_progress()
    # Round is a 2-part counter (no in_progress tracking)
    assert progress["Round"].in_progress is None
    # Searches run is 1-part
    assert progress["Searches run"].in_progress is None
    # Results found is 1-part
    assert progress["Results found"].in_progress is None
    # Documents is a 3-part counter
    assert progress["Documents"].in_progress == 0
    # Keywords is a 3-part counter
    assert progress["Keywords"].in_progress == 0


def test_keywords_progress_categories():
    """Keywords counters have correct categories."""
    progress = create_keywords_progress()
    assert progress["Round"].category == "Search"
    assert progress["Searches run"].category == "Search"
    assert progress["Results found"].category == "Search"
    assert progress["Documents"].category == "Documents"
    assert progress["Keywords"].category == "Keywords"


def test_keywords_progress_context_manager():
    """Keywords progress works as context manager."""
    progress = create_keywords_progress()
    with progress:
        progress["Documents"].total = 10
        progress["Documents"].activate()
        progress["Documents"].work()
        progress["Documents"].done()
    assert progress["Documents"].completed == 1


def test_keywords_progress_typical_usage():
    """Simulate typical keywords progress updates."""
    progress = create_keywords_progress()
    # Start round
    progress["Round"].total = 3
    progress["Round"].completed = 1
    progress["Round"].activate()
    # Searching
    progress["Searches run"].activate()
    progress["Results found"].activate()
    progress.set_status("Searching PubMed")
    progress["Searches run"].add()
    progress["Results found"].add(10)
    assert progress["Searches run"].completed == 1
    assert progress["Results found"].completed == 10
    # Document processing
    progress["Documents"].total = 5
    progress["Documents"].activate()
    progress.set_status("Fetching documents")
    progress["Documents"].work()
    progress["Documents"].done()
    assert progress["Documents"].completed == 1
    # Keyword extraction and evaluation
    progress["Keywords"].total = 50
    progress["Keywords"].activate()
    progress.set_status("Evaluating keywords")
    progress["Keywords"].completed = 15
    progress["Keywords"].complete()
    assert progress["Keywords"].completed == 15


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
