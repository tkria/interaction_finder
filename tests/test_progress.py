"""Tests for shared progress display infrastructure."""

from dataclasses import dataclass

from rich.console import RenderableType
from rich.table import Table

from interaction_finder.progress import DummyProgress, LiveProgressCounter


@dataclass
class TestProgressCounter(LiveProgressCounter):
    """Concrete test implementation of LiveProgressCounter."""

    test_counter: int = 0

    def _render(self) -> RenderableType:
        """Simple test rendering."""
        table = Table.grid()
        table.add_column()
        table.add_row(f"Counter: {self.test_counter}")
        return self._render_with_header(table)


def test_live_progress_counter_initialization():
    """LiveProgressCounter initializes with correct defaults."""
    progress = TestProgressCounter()
    assert progress._status_msg == ""
    assert progress._highlight == ""
    assert progress._live is None
    assert progress._console is not None
    # _enabled depends on TTY, just check it exists
    assert isinstance(progress._enabled, bool)


def test_set_status():
    """set_status updates status message and highlight."""
    progress = TestProgressCounter()
    progress.set_status("Processing", highlight="test")
    assert progress._status_msg == "Processing"
    assert progress._highlight == "test"


def test_set_completed():
    """set_completed formats completion message with elapsed time."""
    progress = TestProgressCounter()
    progress.set_completed()
    assert progress._status_msg.startswith("✓ Completed in")
    assert "s" in progress._status_msg  # Has time unit


def test_set_completed_failed():
    """set_completed with failed=True shows failure message."""
    progress = TestProgressCounter()
    progress.set_completed(failed=True)
    assert progress._status_msg.startswith("✗ Failed after")
    assert "s" in progress._status_msg  # Has time unit


def test_set_phase_idle():
    """set_phase_idle clears status and highlight."""
    progress = TestProgressCounter()
    progress.set_status("Active", highlight="test")
    progress.set_phase_idle()
    assert progress._status_msg == ""
    assert progress._highlight == ""


def test_format_three_part_all_values():
    """_format_three_part with work in progress shows all three parts."""
    progress = TestProgressCounter()
    result = progress._format_three_part(
        complete=5, in_progress=3, total=10, is_highlighted=False, total_is_final=False
    )
    assert "5" in result
    assert "3" in result
    assert "10" in result
    assert "bold green" in result  # complete style
    assert "dim" in result  # in-progress style (not highlighted)
    assert "bold yellow" in result  # total style


def test_format_three_part_highlighted():
    """_format_three_part with highlighting brightens in-progress."""
    progress = TestProgressCounter()
    result = progress._format_three_part(
        complete=5, in_progress=3, total=10, is_highlighted=True, total_is_final=False
    )
    assert "bold bright_yellow" in result  # highlighted in-progress style


def test_format_three_part_complete_hides_in_progress():
    """_format_three_part hides in-progress when final and in_progress=0."""
    progress = TestProgressCounter()
    result = progress._format_three_part(
        complete=10, in_progress=0, total=10, is_highlighted=False, total_is_final=True
    )
    # Should only show complete/total (two parts, one separator)
    # Format: [bold green]10[/]/[bold yellow]10[/]
    assert result == "[bold green]10[/]/[bold yellow]10[/]"


def test_format_three_part_not_final_shows_in_progress():
    """_format_three_part shows in-progress even when 0 if not final."""
    progress = TestProgressCounter()
    result = progress._format_three_part(
        complete=5, in_progress=0, total=10, is_highlighted=False, total_is_final=False
    )
    # Should show all three parts (waiting for more work)
    # Format: [bold green]5[/]/[dim]0[/]/[bold yellow]10[/]
    assert result == "[bold green]5[/]/[dim]0[/]/[bold yellow]10[/]"


def test_format_three_part_total_none():
    """_format_three_part with None total shows single zero."""
    progress = TestProgressCounter()
    result = progress._format_three_part(
        complete=5,
        in_progress=3,
        total=None,
        is_highlighted=False,
        total_is_final=False,
    )
    assert result == "[bold yellow]0[/]"


def test_format_three_part_zero_total():
    """_format_three_part with zero total shows single zero."""
    progress = TestProgressCounter()
    result = progress._format_three_part(
        complete=0, in_progress=0, total=0, is_highlighted=False, total_is_final=True
    )
    assert result == "[bold yellow]0[/]"


def test_context_manager():
    """LiveProgressCounter works as context manager."""
    progress = TestProgressCounter()
    # Should not raise
    with progress:
        progress.test_counter = 42
        assert progress.test_counter == 42


def test_context_manager_with_exception():
    """LiveProgressCounter shows failure message when exception occurs."""
    import pytest

    progress = TestProgressCounter()
    # Force enabled to test the stop behavior (normally disabled without TTY)
    progress._enabled = True
    progress.start()
    with pytest.raises(ValueError):
        with progress:
            raise ValueError("test error")
    assert progress._status_msg.startswith("✗ Failed after")


def test_dummy_progress_no_op():
    """DummyProgress accepts all operations without error."""
    progress = DummyProgress()
    # All operations should be no-ops and not raise
    progress.start()
    progress.stop()
    progress.update()
    progress.set_status("test", highlight="test")
    progress.set_completed()
    progress.set_phase_idle()
    # Arbitrary method calls should also work
    progress.nonexistent_method()
    progress.another_method(1, 2, 3, foo="bar")
    # Context manager should work
    with progress:
        progress.start()


def test_render_with_header_no_status():
    """_render_with_header returns bare table when no status set."""
    progress = TestProgressCounter()
    table = Table.grid()
    table.add_row("test")
    result = progress._render_with_header(table)
    # Should return the table itself when no status
    assert result == table


def test_render_with_header_with_status():
    """_render_with_header adds header when status is set."""
    progress = TestProgressCounter()
    progress._status_msg = "Processing"
    table = Table.grid()
    table.add_row("test")
    result = progress._render_with_header(table)
    # Should return a Group, not the bare table
    assert result != table
    # Group contains header, separator, and table
    from rich.console import Group

    assert isinstance(result, Group)


def test_render_with_header_completion_status():
    """_render_with_header renders completion with green check."""
    progress = TestProgressCounter()
    progress._status_msg = "✓ Completed in 5s"
    table = Table.grid()
    table.add_row("test")
    result = progress._render_with_header(table)
    # Should create a Group with completion header
    from rich.console import Group

    assert isinstance(result, Group)


def test_render_with_header_failure_status():
    """_render_with_header renders failure with red X."""
    progress = TestProgressCounter()
    progress._status_msg = "✗ Failed after 5s"
    table = Table.grid()
    table.add_row("test")
    result = progress._render_with_header(table)
    # Should create a Group with failure header
    from rich.console import Group

    assert isinstance(result, Group)
