"""Tests for widesearch progress display."""

from interaction_finder.widesearch.progress import DummyProgress, WidesearchProgress


def test_widesearch_progress_initialization():
    """Progress counter initializes with zero values."""
    progress = WidesearchProgress()
    assert progress.searches_run == 0
    assert progress.results_found == 0
    assert progress.results_selected == 0
    assert progress.current_round == 0
    assert progress.max_rounds == 0


def test_widesearch_progress_increment_searches():
    """increment_searches increases counter correctly."""
    progress = WidesearchProgress()
    progress.increment_searches()
    assert progress.searches_run == 1
    progress.increment_searches(5)
    assert progress.searches_run == 6


def test_widesearch_progress_add_results():
    """add_results increases results counter."""
    progress = WidesearchProgress()
    progress.add_results(10)
    assert progress.results_found == 10
    progress.add_results(5)
    assert progress.results_found == 15


def test_widesearch_progress_add_selected():
    """add_selected increases selected counter."""
    progress = WidesearchProgress()
    progress.add_selected(3)
    assert progress.results_selected == 3
    progress.add_selected(2)
    assert progress.results_selected == 5


def test_widesearch_progress_set_round():
    """set_round updates round information."""
    progress = WidesearchProgress()
    progress.set_round(2, 5)
    assert progress.current_round == 2
    assert progress.max_rounds == 5


def test_widesearch_progress_context_manager():
    """Progress counter works as context manager."""
    progress = WidesearchProgress()
    # Should not raise
    with progress:
        progress.increment_searches()
        assert progress.searches_run == 1


def test_dummy_progress_no_op():
    """DummyProgress accepts all operations without error."""
    progress = DummyProgress()
    # All operations should be no-ops and not raise
    progress.start()
    progress.increment_searches()
    progress.increment_searches(10)
    progress.add_results(50)
    progress.add_selected(5)
    progress.set_round(3, 5)
    progress.stop()
    # Context manager should work
    with progress:
        progress.increment_searches()
