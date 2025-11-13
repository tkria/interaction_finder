"""Tests for widesearch progress display."""

from interaction_finder.widesearch.progress import DummyProgress, WidesearchProgress


def test_widesearch_progress_initialization():
    """Progress counter initializes with zero values."""
    progress = WidesearchProgress()
    assert progress.searches_run == 0
    assert progress.searches_run_this_round == 0
    assert progress.searches_in_progress == 0
    assert progress.searches_total_this_round == 0
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


def test_widesearch_progress_phase_transitions():
    """Phase transitions update correctly."""
    progress = WidesearchProgress()
    # Should initialize idle
    assert progress._status_msg == ""
    assert progress._highlight == ""
    # Set to searching without backend
    progress.set_phase_searching()
    assert progress._status_msg == "Searching"
    assert progress._highlight == "search"
    # Set to searching with backend
    progress.set_phase_searching(backend="PubMed")
    assert progress._status_msg == "Searching PubMed"
    assert progress._highlight == "search"
    # Set to reranking
    progress.set_phase_reranking()
    assert progress._status_msg == "Reranking results"
    assert progress._highlight == "search"
    # Set to selecting
    progress.set_phase_selecting()
    assert progress._status_msg == "Selecting results"
    assert progress._highlight == "select"
    # Set to completed
    progress.set_completed()
    assert progress._status_msg.startswith("✓ Completed in")
    # Set back to idle
    progress.set_phase_idle()
    assert progress._status_msg == ""
    assert progress._highlight == ""


def test_per_round_tracking():
    """Per-round counters track independently of cumulative."""
    progress = WidesearchProgress()
    # First round
    progress.searches_run = 5
    progress.searches_run_this_round = 5
    progress.searches_in_progress = 0
    progress.searches_total_this_round = 5
    assert progress.searches_run == 5
    assert progress.searches_run_this_round == 5
    # Simulate reset for new round
    progress.searches_run_this_round = 0
    progress.searches_in_progress = 0
    progress.searches_total_this_round = 0
    # Second round
    progress.searches_run = 10  # Cumulative
    progress.searches_run_this_round = 5  # Just this round
    progress.searches_total_this_round = 5
    assert progress.searches_run == 10  # Total across rounds
    assert progress.searches_run_this_round == 5  # Just this round


def test_in_progress_tracking():
    """In-progress counter tracks active searches."""
    progress = WidesearchProgress()
    progress.searches_total_this_round = 10
    progress.searches_in_progress = 10
    progress.searches_run_this_round = 0
    # Simulate searches completing
    progress.searches_in_progress = 7
    progress.searches_run_this_round = 3
    assert progress.searches_in_progress == 7
    assert progress.searches_run_this_round == 3
    assert progress.searches_total_this_round == 10


def test_three_part_display_format():
    """_render uses three-part format for searches this round."""
    progress = WidesearchProgress()
    progress.searches_run_this_round = 5
    progress.searches_in_progress = 3
    progress.searches_total_this_round = 8
    progress.results_found = 87
    progress.results_selected = 12
    # Call render (should not raise)
    result = progress._render()
    assert result is not None


def test_render_with_round_indicator():
    """_render includes round indicator when max_rounds > 0."""
    progress = WidesearchProgress()
    progress.current_round = 2
    progress.max_rounds = 5
    result = progress._render()
    assert result is not None


def test_render_without_round_indicator():
    """_render excludes round indicator when max_rounds = 0."""
    progress = WidesearchProgress()
    progress.max_rounds = 0
    result = progress._render()
    assert result is not None


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
    progress.set_phase_searching()
    progress.set_phase_searching(backend="PubMed")
    progress.set_phase_reranking()
    progress.set_phase_selecting()
    progress.set_phase_idle()
    progress.set_completed()
    progress.stop()
    # New fields
    progress.searches_run_this_round = 10
    progress.searches_in_progress = 5
    progress.searches_total_this_round = 10
    progress.update()
    # Context manager should work
    with progress:
        progress.increment_searches()
