"""Shared progress display infrastructure for pipeline operations.

Provides base classes and protocols for live-updating Rich displays with
status messages, phase highlighting, and TTY detection.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import time
from typing import Optional

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text


@dataclass
class LiveProgressCounter(ABC):
    """Base class for live progress displays with status and phase management.

    Handles Rich display lifecycle (start/stop/update), elapsed time tracking,
    status messages with spinners, and TTY detection. Subclasses implement
    _render() to define table structure and counter layout.
    """

    _status_msg: str = field(default="", init=False)
    _highlight: str = field(default="", init=False)
    _start_time: float = field(default_factory=time, init=False)
    _live: Optional[Live] = field(default=None, init=False, repr=False)
    _console: Console = field(default_factory=Console, init=False, repr=False)
    _enabled: bool = field(default=True, init=False)

    def __post_init__(self):
        """Check if display should be enabled based on TTY status."""
        if not self._console.is_terminal:
            self._enabled = False
        # Configure logging to integrate with this console
        from interaction_finder.logging import configure_logging

        configure_logging(console=self._console, verbose=False)

    def start(self) -> None:
        """Start the live display."""
        if not self._enabled:
            return
        self._live = Live(self._render(), console=self._console, refresh_per_second=4)
        self._live.start()

    def stop(self, failed: bool = False) -> None:
        """Stop the live display, showing completion status.

        Parameters:
            failed: bool — if True, show failure message instead of success
        """
        if self._live:
            self.set_completed(failed=failed)
            self._live.stop()
            self._live = None

    def update(self) -> None:
        """Update the live display with current values."""
        if self._live is not None:
            self._live.update(self._render())

    def set_status(self, message: str, highlight: str = "") -> None:
        """Set status message and which metrics to highlight.

        Parameters:
            message: str — status text (empty for idle, "✓ ..." for completed)
            highlight: str — domain-specific highlight key
        """
        self._status_msg = message
        self._highlight = highlight
        self.update()

    def set_completed(self, failed: bool = False) -> None:
        """Show completion message with elapsed time.

        Parameters:
            failed: bool — if True, show failure message instead of success
        """
        elapsed = int(time() - self._start_time)
        mins, secs = divmod(elapsed, 60)
        time_str = f"{mins}m {secs}s" if mins else f"{secs}s"
        if failed:
            self.set_status(f"✗ Failed after {time_str}")
        else:
            self.set_status(f"✓ Completed in {time_str}")

    def set_phase_idle(self) -> None:
        """Clear status and highlighting."""
        self.set_status("")

    def _format_three_part(
        self,
        complete: int,
        in_progress: int,
        total: int | None,
        is_highlighted: bool = False,
        total_is_final: bool = False,
    ) -> str:
        """Format counter as {complete}/{in_progress}/{total} with colored parts.

        Parameters:
            complete: int — count of finished items
            in_progress: int — count of currently processing items
            total: int | None — total count (None if unknown)
            is_highlighted: bool — whether to brighten the in-progress count
            total_is_final: bool — whether total is static (True) or still updating (False)

        Returns:
            str — Rich markup string like "[bold green]5[/]/[bright_yellow]3[/]/[dim]10[/]"
        """
        complete_style = "bold green"
        in_progress_style = "bold bright_yellow" if is_highlighted else "bold yellow"
        total_style = "bold bright_blue" if total_is_final else "dim"
        total_str = str(total) if total is not None else "?"
        # Append ? to total when not final and total exists
        if not total_is_final and total is not None:
            total_str = f"{total}?"
        return (
            f"[{complete_style}]{complete}[/]/"
            f"[{in_progress_style}]{in_progress}[/]/"
            f"[{total_style}]{total_str}[/]"
        )

    def _render_with_header(self, table) -> RenderableType:
        """Render table with optional status header.

        Common rendering logic: adds spinner/completion header if status is set,
        calculates separator width, and returns Group or bare table.

        Parameters:
            table: Rich Table object to render

        Returns:
            RenderableType — Group(header, separator, table) or bare table
        """
        if not self._status_msg:
            return table
        # Create header (spinner, success, or failure)
        if self._status_msg.startswith("✓"):
            header = Text(self._status_msg, style="bold green")
        elif self._status_msg.startswith("✗"):
            header = Text(self._status_msg, style="bold red")
        else:
            header = Spinner("dots", text=self._status_msg, style="cyan")
        # Calculate separator width
        header_width = self._console.measure(header).maximum
        table_width = self._console.measure(table).maximum
        separator_width = max(header_width, table_width)
        separator = Text("─" * separator_width, style="bold cyan")
        return Group(header, separator, table)

    @abstractmethod
    def _render(self) -> RenderableType:
        """Render the progress table.

        Subclasses implement this to define table structure, columns,
        and highlighting logic based on self._highlight.
        """
        ...

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit, showing failure status if an exception occurred."""
        self.stop(failed=exc_type is not None)


class DummyProgress:
    """No-op progress counter for when display is disabled."""

    def __getattr__(self, name):
        """Return no-op function for any method call."""
        return lambda *args, **kwargs: None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
