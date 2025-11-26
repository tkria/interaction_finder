"""Shared progress display infrastructure for pipeline operations.

Provides Counter and StatusTable for tracking progress, with LiveStatusTable
for live-updating Rich displays with status messages and category grouping.
"""

from dataclasses import dataclass
from time import time
from typing import Literal

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text


@dataclass(init=False)
class Counter:
    """Progress counter with 1, 2, or 3 parts.

    Parts determined by state:
    - total == 0 → 1-part (just completed)
    - total > 0, in_progress is None → 2-part (completed/total)
    - total > 0, in_progress is int → 3-part (completed/in_progress/total)
    """

    name: str
    category: str
    note: str
    total: int
    in_progress: int | None
    completed: int
    status: Literal["unstarted", "active", "complete"]

    def __init__(
        self,
        name: str,
        *,
        track_in_progress: bool = False,
        category: str = "",
        note: str = "",
    ):
        self.name = name
        self.category = category
        self.note = note
        self.total = 0
        self.in_progress = 0 if track_in_progress else None
        self.completed = 0
        self.status = "unstarted"

    def activate(self) -> None:
        """Mark counter as active."""
        self.status = "active"

    def complete(self) -> None:
        """Mark counter as complete."""
        self.status = "complete"

    def add(self, n: int = 1) -> None:
        """Increment completed (1-part counters only)."""
        if self.total != 0:
            raise ValueError(f"Counter {self.name!r} has a total; use work/done")
        self.completed += n

    def work(self, n: int = 1) -> None:
        """Move n items into in_progress (3-part counters only)."""
        if self.in_progress is None:
            raise ValueError(f"Counter {self.name!r} doesn't track in_progress")
        self.in_progress += n

    def done(self, n: int = 1) -> None:
        """Move n items from in_progress to completed (3-part counters only)."""
        if self.in_progress is None:
            raise ValueError(f"Counter {self.name!r} doesn't track in_progress")
        self.in_progress -= n
        self.completed += n

    def rich(self) -> str:
        """Render counter value with Rich markup."""
        if self.total == 0:
            return f"[bold yellow]{self.completed}[/]"
        if self.status == "unstarted":
            return "[bold yellow]0[/]"
        if self.status == "complete" or self.in_progress is None:
            return f"[bold green]{self.completed}[/]/[bold yellow]{self.total}[/]"
        in_prog_style = (
            "bold bright_yellow" if self.status == "active" else "dim yellow"
        )
        return (
            f"[bold green]{self.completed}[/]/"
            f"[{in_prog_style}]{self.in_progress}[/]/"
            f"[bold yellow]{self.total}[/]"
        )


class StatusTable:
    """Progress tracking with named counters and status message.

    Pure data container for progress state. Use LiveStatusTable for
    terminal display with Rich Live updates.
    """

    def __init__(self, *counters: Counter):
        self.status: str = ""
        self.counters: list[Counter] = list(counters)

    def __getitem__(self, name: str) -> Counter:
        """Access counter by name."""
        for counter in self.counters:
            if counter.name == name:
                return counter
        raise KeyError(name)

    def set_status(self, message: str) -> None:
        """Set status message."""
        self.status = message

    # No-op methods for compatibility when used without live display
    def start(self) -> None:
        """No-op for base class."""

    def stop(self) -> None:
        """No-op for base class."""

    def update(self) -> None:
        """Trigger immediate display refresh. Usually not needed due to auto-refresh."""

    def succeed(self) -> None:
        """No-op for base class."""

    def fail(self) -> None:
        """No-op for base class."""

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            self.fail()
        else:
            self.succeed()


class LiveStatusTable(StatusTable):
    """Live-updating Rich display for progress tracking.

    Renders counters as a table with category grouping, status header
    with spinner, and automatic TTY detection.
    """

    def __init__(self, *counters: Counter):
        super().__init__(*counters)
        self._start_time: float | None = None
        self._result: Literal["pending", "running", "success", "failure"] = "pending"
        self._console: Console = Console()
        self._live: Live | None = None
        self._enabled: bool = self._console.is_terminal
        # Configure logging integration
        from interaction_finder.logging import configure_logging

        configure_logging(console=self._console, verbose=False)

    def start(self) -> None:
        """Start the operation timer and live display."""
        self._start_time = time()
        self._result = "running"
        if not self._enabled:
            return
        self._live = Live(self._render(), console=self._console, refresh_per_second=4)
        self._live.start()

    def stop(self) -> None:
        """Stop the live display."""
        if self._live:
            self._live.stop()
            self._live = None

    def update(self) -> None:
        """Trigger immediate display refresh.

        Usually not needed - the display auto-refreshes at 4fps. Only call this
        if you need sub-250ms feedback for a specific update.
        """
        if self._live is not None:
            self._live.update(self._render())

    def set_status(self, message: str) -> None:
        """Set status message and refresh display."""
        self.status = message
        self.update()

    def succeed(self) -> None:
        """Mark operation as successful and stop display."""
        self._result = "success"
        self.status = f"✓ Completed in {self._elapsed()}"
        self.update()
        self.stop()

    def fail(self) -> None:
        """Mark operation as failed and stop display."""
        self._result = "failure"
        self.status = f"✗ Failed after {self._elapsed()}"
        self.update()
        self.stop()

    def _elapsed(self) -> str:
        """Format elapsed time since start."""
        if self._start_time is None:
            return "0s"
        elapsed = int(time() - self._start_time)
        mins, secs = divmod(elapsed, 60)
        return f"{mins}m {secs}s" if mins else f"{secs}s"

    def _render(self) -> RenderableType:
        """Render status header and counter table."""
        # Group counters by category, preserving order
        categories: dict[str, list[Counter]] = {}
        for counter in self.counters:
            categories.setdefault(counter.category, []).append(counter)
        # Build rows: (label, value, annotation)
        rows: list[tuple[str, str, str]] = []
        for category, cat_counters in categories.items():
            # Category activation: any counter not unstarted
            cat_active = any(c.status != "unstarted" for c in cat_counters)
            label_style = "bold cyan" if cat_active else "cyan"
            # Category header (if named)
            if category:
                rows.append((f"[{label_style}]{category}[/]", "", ""))
            # Counter rows
            for counter in cat_counters:
                indent = "  " if category else ""
                label = f"[{label_style}]{indent}{counter.name}[/]"
                rows.append((label, counter.rich(), counter.note))
        # Build two-column table for width measurement (excludes annotations)
        width_table = Table.grid(padding=(0, 2))
        width_table.add_column()
        width_table.add_column(justify="right")
        for label, value, _ in rows:
            width_table.add_row(label, value)
        content_width = self._console.measure(width_table).maximum
        # Build three-column table for display
        table = Table.grid(padding=(0, 2))
        table.add_column()
        table.add_column(justify="right")
        table.add_column(style="dim", justify="left")
        for label, value, annotation in rows:
            table.add_row(label, value, annotation)
        return self._render_with_header(table, content_width)

    def _render_with_header(self, table, content_width: int) -> RenderableType:
        """Add status header with spinner/checkmark above table."""
        if not self.status:
            return table
        # Create header
        if self.status.startswith("✓"):
            header = Text(self.status, style="bold green")
        elif self.status.startswith("✗"):
            header = Text(self.status, style="bold red")
        else:
            header = Spinner("dots", text=self.status, style="cyan")
        # Separator
        header_width = self._console.measure(header).maximum
        separator_width = max(header_width, content_width)
        separator = Text("─" * separator_width, style="bold cyan")
        return Group(header, separator, table)


class DummyProgress:
    """No-op progress counter for when display is disabled."""

    def __getattr__(self, name):
        """Return no-op function for any method call."""
        return lambda *args, **kwargs: None

    def __getitem__(self, key):
        """Return self for item access, allowing chained no-op calls."""
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
