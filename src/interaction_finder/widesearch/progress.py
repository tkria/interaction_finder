"""Live progress display for widesearch operations.

Provides an in-place updating counter showing searches run, results found,
and results selected during widesearch execution.
"""

from dataclasses import dataclass, field
from time import time
from typing import Optional

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text


@dataclass
class WidesearchProgress:
    """Live progress counter for widesearch operations.

    Displays real-time statistics with spinner showing current operation
    and bold highlighting of active metrics.
    """

    searches_run: int = 0
    results_found: int = 0
    results_selected: int = 0
    current_round: int = 0
    max_rounds: int = 0
    _status_msg: str = field(default="", init=False)  # Empty = idle, text = active
    _highlight: str = field(default="", init=False)  # "search" or "select"
    _start_time: float = field(default_factory=time, init=False)
    _live: Optional[Live] = field(default=None, init=False, repr=False)
    _console: Console = field(default_factory=Console, init=False, repr=False)
    _enabled: bool = field(default=True, init=False)

    def __post_init__(self):
        """Check if display should be enabled based on TTY status."""
        if not self._console.is_terminal:
            self._enabled = False

    def start(self) -> None:
        """Start the live display."""
        if not self._enabled:
            return
        self._live = Live(self._render(), console=self._console, refresh_per_second=4)
        self._live.start()

    def set_status(self, message: str, highlight: str = "") -> None:
        """Set status message and which metrics to highlight.

        Parameters:
            message: str — status text (empty for idle, "✓ ..." for completed)
            highlight: str — "search" or "select" to highlight those metrics
        """
        self._status_msg = message
        self._highlight = highlight
        self.update()

    def set_completed(self) -> None:
        """Show completion message with elapsed time."""
        elapsed = int(time() - self._start_time)
        mins, secs = divmod(elapsed, 60)
        time_str = f"{mins}m {secs}s" if mins else f"{secs}s"
        self.set_status(f"✓ Completed in {time_str}")

    def stop(self) -> None:
        """Stop the live display, showing completion status."""
        if self._live:
            self.set_completed()
            self._live.stop()
            self._live = None

    def _render(self) -> RenderableType:
        """Render progress table with optional status header."""
        # Build metrics table
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold cyan")
        table.add_column(style="bold yellow", justify="right")

        if self.max_rounds > 0:
            table.add_row("Round", f"{self.current_round}/{self.max_rounds}")

        # Apply highlighting based on active operation
        bright = "bold bright_yellow" if self._highlight == "search" else "bold yellow"
        table.add_row("Searches run", f"[{bright}]{self.searches_run}[/]")
        table.add_row("Results found", f"[{bright}]{self.results_found}[/]")

        bright = "bold bright_yellow" if self._highlight == "select" else "bold yellow"
        table.add_row("Results selected", f"[{bright}]{self.results_selected}[/]")

        # Add status header if active or completed
        if self._status_msg:
            rule = Rule(style="dim", characters="─")
            if self._status_msg.startswith("✓"):
                header = Text(self._status_msg, style="bold green")
            else:
                header = Spinner("dots", text=self._status_msg, style="cyan")
            return Group(header, rule, table)
        return table

    def update(self) -> None:
        """Update the live display with current values."""
        if self._live is not None:
            self._live.update(self._render())

    def set_phase_searching(self, backend: str = "") -> None:
        """Show searching status with search metrics highlighted."""
        msg = f"Searching {backend}" if backend else "Searching"
        self.set_status(msg, highlight="search")

    def set_phase_reranking(self) -> None:
        """Show reranking status with search metrics highlighted."""
        self.set_status("Reranking results", highlight="search")

    def set_phase_selecting(self) -> None:
        """Show selecting status with selection metrics highlighted."""
        self.set_status("Selecting results", highlight="select")

    def set_phase_idle(self) -> None:
        """Clear status and highlighting."""
        self.set_status("")

    def increment_searches(self, count: int = 1) -> None:
        """Increment the searches run counter."""
        self.searches_run += count
        self.update()

    def add_results(self, count: int) -> None:
        """Add to the results found counter."""
        self.results_found += count
        self.update()

    def add_selected(self, count: int) -> None:
        """Add to the results selected counter."""
        self.results_selected += count
        self.update()

    def set_round(self, current: int, max_rounds: int) -> None:
        """Update the current round indicator."""
        self.current_round = current
        self.max_rounds = max_rounds
        self.update()

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        self.stop()


class DummyProgress:
    """No-op progress counter for when display is disabled."""

    def __getattr__(self, name):
        """Return no-op function for any method call."""
        return lambda *args, **kwargs: None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
