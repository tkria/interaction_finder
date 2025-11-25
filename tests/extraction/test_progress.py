"""Tests for extraction progress display."""

from interaction_finder.extraction.progress import DummyProgress, ExtractionProgress


def test_extraction_progress_initialization():
    """Progress counter initializes with zero values."""
    progress = ExtractionProgress()
    assert progress.documents_processed == 0
    assert progress.documents_in_progress == 0
    assert progress.documents_total == 0
    assert progress.entities_found == 0
    assert progress.pairs_found == 0
    assert progress.pairs_assessed == 0
    assert progress.pairs_in_progress == 0
    assert progress.pairs_total == 0
    assert progress.quotes_validated == 0
    assert progress.quotes_failed == 0
    assert progress.unique_pairs == 0
    assert progress.judgments_in_progress == 0
    assert progress.accepted == 0
    assert progress.rejected == 0


def test_extraction_progress_document_tracking():
    """Document counters track correctly."""
    progress = ExtractionProgress()
    progress.documents_total = 10
    progress.documents_in_progress = 5
    progress.documents_processed = 5
    assert progress.documents_total == 10
    assert progress.documents_in_progress == 5
    assert progress.documents_processed == 5


def test_extraction_progress_pairs_tracking():
    """Pairs counters track correctly."""
    progress = ExtractionProgress()
    progress.pairs_found = 100
    progress.pairs_in_progress = 20
    progress.pairs_assessed = 80
    assert progress.pairs_found == 100
    assert progress.pairs_in_progress == 20
    assert progress.pairs_assessed == 80


def test_extraction_progress_judgments_tracking():
    """Judgment counters track correctly."""
    progress = ExtractionProgress()
    progress.unique_pairs = 50
    progress.judgments_in_progress = 10
    progress.accepted = 30
    progress.rejected = 10
    assert progress.unique_pairs == 50
    assert progress.judgments_in_progress == 10
    assert progress.accepted == 30
    assert progress.rejected == 10


def test_extraction_progress_entities_tracking():
    """Entity counter tracks correctly."""
    progress = ExtractionProgress()
    progress.entities_found = 250
    assert progress.entities_found == 250


def test_extraction_progress_quotes_tracking():
    """Quote validation counters track correctly."""
    progress = ExtractionProgress()
    progress.quotes_validated = 100
    progress.quotes_failed = 5
    assert progress.quotes_validated == 100
    assert progress.quotes_failed == 5


def test_extraction_progress_sweep_tracking():
    """Sweep counters track correctly."""
    progress = ExtractionProgress()
    progress.sweep_co_mentions_found = 15
    progress.sweep_no_existing = 10
    progress.sweep_uncovered = 5
    progress.sweep_assessed = 12
    progress.sweep_relationships = 4
    assert progress.sweep_co_mentions_found == 15
    assert progress.sweep_no_existing == 10
    assert progress.sweep_uncovered == 5
    assert progress.sweep_assessed == 12
    assert progress.sweep_relationships == 4


def test_context_manager():
    """Progress counter works as context manager."""
    progress = ExtractionProgress()
    # Should not raise
    with progress:
        progress.documents_processed = 1
        assert progress.documents_processed == 1


def test_phase_extracting():
    """set_phase_extracting sets correct status."""
    progress = ExtractionProgress()
    progress.set_phase_extracting()
    assert progress._status_msg == "Extracting entities"
    assert progress._highlight == "docs"


def test_phase_validating():
    """set_phase_validating sets correct status."""
    progress = ExtractionProgress()
    progress.set_phase_validating()
    assert progress._status_msg == "Validating entities"
    assert progress._highlight == "docs"


def test_phase_proximal():
    """set_phase_proximal sets correct status."""
    progress = ExtractionProgress()
    progress.set_phase_proximal()
    assert progress._status_msg == "Finding proximal pairs"
    assert progress._highlight == "pairs"


def test_phase_extracting_pairs():
    """set_phase_extracting_pairs sets correct status."""
    progress = ExtractionProgress()
    progress.set_phase_extracting_pairs()
    assert progress._status_msg == "Extracting relationships"
    assert progress._highlight == "pairs"


def test_phase_assessing():
    """set_phase_assessing sets correct status."""
    progress = ExtractionProgress()
    progress.set_phase_assessing()
    assert progress._status_msg == "Assessing pairs"
    assert progress._highlight == "assessment"


def test_phase_sweeping():
    """set_phase_sweeping sets correct status."""
    progress = ExtractionProgress()
    progress.set_phase_sweeping()
    assert progress._status_msg == "Sweeping for missed co-mentions"
    assert progress._highlight == "sweep"


def test_phase_judging():
    """set_phase_judging sets correct status."""
    progress = ExtractionProgress()
    progress.set_phase_judging()
    assert progress._status_msg == "Cross-document validation"
    assert progress._highlight == "assessment"


def test_phase_finalizing():
    """set_phase_finalizing sets correct status."""
    progress = ExtractionProgress()
    progress.set_phase_finalizing()
    assert progress._status_msg == "Finalizing results"
    assert progress._highlight == ""


def test_phase_idle():
    """set_phase_idle clears status."""
    progress = ExtractionProgress()
    progress.set_phase_extracting()
    progress.set_phase_idle()
    assert progress._status_msg == ""
    assert progress._highlight == ""


def test_phase_transitions():
    """Phase transitions update correctly."""
    progress = ExtractionProgress()
    # Initialize idle
    assert progress._status_msg == ""
    assert progress._highlight == ""
    # Cycle through phases
    progress.set_phase_extracting()
    assert progress._highlight == "docs"
    progress.set_phase_validating()
    assert progress._highlight == "docs"
    progress.set_phase_proximal()
    assert progress._highlight == "pairs"
    progress.set_phase_extracting_pairs()
    assert progress._highlight == "pairs"
    progress.set_phase_assessing()
    assert progress._highlight == "assessment"
    progress.set_phase_judging()
    assert progress._highlight == "assessment"
    progress.set_phase_finalizing()
    assert progress._highlight == ""
    # Set to completed
    progress.set_completed()
    assert progress._status_msg.startswith("✓ Completed in")
    # Back to idle
    progress.set_phase_idle()
    assert progress._status_msg == ""
    assert progress._highlight == ""


def test_render_with_values():
    """_render produces output with counter values."""
    progress = ExtractionProgress()
    progress.documents_processed = 5
    progress.documents_in_progress = 2
    progress.documents_total = 10
    progress.entities_found = 127
    progress.pairs_assessed = 42
    progress.pairs_in_progress = 8
    progress.pairs_found = 156
    progress.quotes_validated = 89
    progress.quotes_failed = 3
    progress.unique_pairs = 28
    progress.judgments_in_progress = 5
    progress.accepted = 12
    progress.rejected = 11
    # Call render (should not raise)
    result = progress._render()
    assert result is not None


def test_render_documents_three_part():
    """_render uses three-part format for documents."""
    progress = ExtractionProgress()
    progress.documents_processed = 5
    progress.documents_in_progress = 3
    progress.documents_total = 10
    result = progress._render()
    # Result should be a Table that can be rendered
    assert result is not None


def test_render_pairs_three_part_updating():
    """_render uses three-part format for pairs with ? when total updating."""
    progress = ExtractionProgress()
    progress.documents_in_progress = 3  # Still processing documents
    progress.documents_total = 10
    progress.pairs_assessed = 42
    progress.pairs_in_progress = 8
    progress.pairs_found = 156
    result = progress._render()
    assert result is not None


def test_render_pairs_three_part_final():
    """_render uses three-part format for pairs with final total."""
    progress = ExtractionProgress()
    progress.documents_processed = 10
    progress.documents_in_progress = 0  # All docs processed
    progress.documents_total = 10
    progress.pairs_assessed = 156
    progress.pairs_in_progress = 0
    progress.pairs_found = 156
    result = progress._render()
    assert result is not None


def test_render_judgments_three_part():
    """_render uses three-part format for judgments."""
    progress = ExtractionProgress()
    progress.unique_pairs = 28
    progress.judgments_in_progress = 5
    progress.accepted = 12
    progress.rejected = 11
    result = progress._render()
    assert result is not None


def test_dummy_progress_no_op():
    """DummyProgress accepts all operations without error."""
    progress = DummyProgress()
    # All operations should be no-ops and not raise
    progress.start()
    progress.documents_processed = 10
    progress.documents_in_progress = 5
    progress.entities_found = 100
    progress.set_phase_extracting()
    progress.set_phase_validating()
    progress.set_phase_proximal()
    progress.set_phase_extracting_pairs()
    progress.set_phase_assessing()
    progress.set_phase_judging()
    progress.set_phase_finalizing()
    progress.set_phase_idle()
    progress.set_completed()
    progress.stop()
    progress.update()
    # Context manager should work
    with progress:
        progress.documents_processed = 42
