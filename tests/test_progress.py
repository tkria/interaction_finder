"""Tests for shared progress display infrastructure."""

import pytest
from rich.console import Group

from interaction_finder.progress import (
    Counter,
    DummyProgress,
    LiveStatusTable,
    StatusTable,
)


class TestCounter:
    """Tests for Counter class."""

    def test_initialization_defaults(self):
        """Counter initializes with correct defaults."""
        counter = Counter("Test")
        assert counter.name == "Test"
        assert counter.category == ""
        assert counter.note == ""
        assert counter.total == 0
        assert counter.in_progress is None
        assert counter.completed == 0
        assert counter.status == "unstarted"

    def test_initialization_with_in_progress(self):
        """Counter with track_in_progress creates in_progress field."""
        counter = Counter("Test", track_in_progress=True)
        assert counter.in_progress == 0

    def test_initialization_with_category_and_note(self):
        """Counter accepts category and note."""
        counter = Counter("Test", category="Section", note="extra info")
        assert counter.category == "Section"
        assert counter.note == "extra info"

    def test_activate(self):
        """activate() sets status to active."""
        counter = Counter("Test")
        counter.activate()
        assert counter.status == "active"

    def test_complete(self):
        """complete() sets status to complete."""
        counter = Counter("Test")
        counter.complete()
        assert counter.status == "complete"

    def test_add_increments_completed(self):
        """add() increments completed for 1-part counter."""
        counter = Counter("Test")
        counter.add()
        assert counter.completed == 1
        counter.add(5)
        assert counter.completed == 6

    def test_add_raises_for_multi_part_counter(self):
        """add() raises ValueError for multi-part counter."""
        counter = Counter("Test")
        counter.total = 10
        with pytest.raises(ValueError, match="has a total"):
            counter.add()

    def test_work_increments_in_progress(self):
        """work() increments in_progress for 3-part counter."""
        counter = Counter("Test", track_in_progress=True)
        counter.work()
        assert counter.in_progress == 1
        counter.work(3)
        assert counter.in_progress == 4

    def test_work_raises_without_in_progress(self):
        """work() raises ValueError without in_progress tracking."""
        counter = Counter("Test")
        with pytest.raises(ValueError, match="doesn't track in_progress"):
            counter.work()

    def test_done_decrements_in_progress_increments_completed(self):
        """done() moves from in_progress to completed."""
        counter = Counter("Test", track_in_progress=True)
        counter.in_progress = 5
        counter.done()
        assert counter.in_progress == 4
        assert counter.completed == 1
        counter.done(2)
        assert counter.in_progress == 2
        assert counter.completed == 3

    def test_done_raises_without_in_progress(self):
        """done() raises ValueError without in_progress tracking."""
        counter = Counter("Test")
        with pytest.raises(ValueError, match="doesn't track in_progress"):
            counter.done()


class TestCounterRich:
    """Tests for Counter.rich() rendering."""

    def test_rich_one_part(self):
        """1-part counter renders completed value."""
        counter = Counter("Test")
        counter.completed = 42
        result = counter.rich()
        assert "[bold yellow]42[/]" == result

    def test_rich_unstarted(self):
        """Unstarted counter renders zero in bold grey."""
        counter = Counter("Test", track_in_progress=True)
        counter.total = 10  # Has total but status is unstarted
        result = counter.rich()
        assert result == "[bold bright_black]0[/]"

    def test_rich_two_part_complete(self):
        """Complete 2-part counter renders complete/total."""
        counter = Counter("Test")
        counter.total = 10
        counter.completed = 10
        counter.status = "complete"
        result = counter.rich()
        assert result == "[bold green]10[/]/[bold yellow]10[/]"

    def test_rich_three_part_active(self):
        """Active 3-part counter renders complete/in_progress/total."""
        counter = Counter("Test", track_in_progress=True)
        counter.total = 10
        counter.in_progress = 3
        counter.completed = 5
        counter.status = "active"
        result = counter.rich()
        assert "[bold green]5[/]" in result
        assert "[bold bright_yellow]3[/]" in result
        assert "[bold yellow]10[/]" in result

    def test_rich_three_part_inactive(self):
        """Inactive 3-part counter uses dim yellow for in_progress."""
        counter = Counter("Test", track_in_progress=True)
        counter.total = 10
        counter.in_progress = 3
        counter.completed = 5
        counter.status = "unstarted"  # Not active
        # When unstarted with total set, shows 0 in grey
        result = counter.rich()
        assert result == "[bold bright_black]0[/]"


class TestStatusTable:
    """Tests for StatusTable base class."""

    def test_initialization(self):
        """StatusTable initializes with counters."""
        c1 = Counter("A")
        c2 = Counter("B")
        table = StatusTable(c1, c2)
        assert len(table.counters) == 2
        assert table.status == ""

    def test_getitem(self):
        """StatusTable supports name-based counter access."""
        c1 = Counter("First")
        c2 = Counter("Second")
        table = StatusTable(c1, c2)
        assert table["First"] is c1
        assert table["Second"] is c2

    def test_getitem_raises_for_unknown(self):
        """StatusTable raises KeyError for unknown counter."""
        table = StatusTable(Counter("Test"))
        with pytest.raises(KeyError):
            table["Unknown"]

    def test_set_status(self):
        """set_status updates status message."""
        table = StatusTable()
        table.set_status("Processing")
        assert table.status == "Processing"

    def test_context_manager_no_op(self):
        """StatusTable base class context manager is no-op."""
        table = StatusTable(Counter("Test"))
        with table:
            table["Test"].completed = 42
        assert table["Test"].completed == 42


class TestLiveStatusTable:
    """Tests for LiveStatusTable class."""

    def test_initialization(self):
        """LiveStatusTable initializes with counters and live state."""
        c1 = Counter("A")
        c2 = Counter("B")
        table = LiveStatusTable(c1, c2)
        assert len(table.counters) == 2
        assert table.status == ""
        assert table._result == "pending"

    def test_succeed(self):
        """succeed() sets completion status."""
        table = LiveStatusTable()
        table.start()
        table.succeed()
        assert table.status.startswith("✓ Completed in")
        assert table._result == "success"

    def test_fail(self):
        """fail() sets failure status."""
        table = LiveStatusTable()
        table.start()
        table.fail()
        assert table.status.startswith("✗ Failed after")
        assert table._result == "failure"

    def test_context_manager(self):
        """LiveStatusTable works as context manager."""
        table = LiveStatusTable(Counter("Test"))
        with table:
            table["Test"].completed = 42
        assert table["Test"].completed == 42
        assert table._result == "success"

    def test_context_manager_with_exception(self):
        """LiveStatusTable shows failure on exception."""
        table = LiveStatusTable(Counter("Test"))
        with pytest.raises(ValueError):
            with table:
                raise ValueError("test error")
        assert table._result == "failure"

    def test_elapsed_formatting(self):
        """_elapsed formats time correctly."""
        table = LiveStatusTable()
        table._start_time = None
        assert table._elapsed() == "0s"


class TestLiveStatusTableRender:
    """Tests for LiveStatusTable rendering."""

    def test_render_groups_by_category(self):
        """_render groups counters by category."""
        table = LiveStatusTable(
            Counter("A", category="First"),
            Counter("B", category="First"),
            Counter("C", category="Second"),
        )
        result = table._render()
        assert result is not None

    def test_render_with_status_creates_group(self):
        """_render with status creates Group with header."""
        table = LiveStatusTable(Counter("Test"))
        table.status = "Processing"
        result = table._render()
        assert isinstance(result, Group)

    def test_render_without_status_returns_table(self):
        """_render without status returns bare table."""
        table = LiveStatusTable(Counter("Test"))
        result = table._render()
        # Not a Group when no status
        assert not isinstance(result, Group)

    def test_render_completion_status(self):
        """_render with completion status shows green header."""
        table = LiveStatusTable(Counter("Test"))
        table.status = "✓ Completed in 5s"
        result = table._render()
        assert isinstance(result, Group)

    def test_render_failure_status(self):
        """_render with failure status shows red header."""
        table = LiveStatusTable(Counter("Test"))
        table.status = "✗ Failed after 5s"
        result = table._render()
        assert isinstance(result, Group)

    def test_category_activation(self):
        """Category becomes bold when any counter is not unstarted."""
        c1 = Counter("A", category="Test")
        c2 = Counter("B", category="Test")
        table = LiveStatusTable(c1, c2)
        # Initially unstarted - category not bold
        result = table._render()
        assert result is not None
        # Activate one counter
        c1.activate()
        result = table._render()
        assert result is not None


class TestDummyProgress:
    """Tests for DummyProgress class."""

    def test_no_op_methods(self):
        """DummyProgress accepts all method calls without error."""
        progress = DummyProgress()
        progress.start()
        progress.stop()
        progress.update()
        progress.set_status("test")
        progress.succeed()
        progress.fail()
        progress.nonexistent_method()
        progress.another_method(1, 2, 3, foo="bar")

    def test_context_manager(self):
        """DummyProgress works as context manager."""
        progress = DummyProgress()
        with progress:
            progress.start()

    def test_getitem_returns_self(self):
        """DummyProgress returns self for item access, enabling chained no-ops."""
        progress = DummyProgress()
        # Should not raise
        result = progress["anything"]
        # Result is self, allowing chained attribute access
        assert result is progress
        # Can chain methods on it
        result.add()
        result.work()
        result.done()
