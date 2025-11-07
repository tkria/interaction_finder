"""Live progress display for widesearch operations.

Provides an in-place updating counter showing searches run, results found,
and results selected during widesearch execution.
"""

from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console, RenderableType
from rich.live import Live
from rich.table import Table


@dataclass
class WidesearchProgress:
    """Live progress counter for widesearch operations.

    Displays real-time statistics:
    - Searches run: Total number of search queries executed
    - Results found: Total number of search results returned
    - Results selected: Number of results chosen by the agent

    The display updates in-place without creating new lines, providing
    a clean interface for monitoring long-running search sessions.
    """

    searches_run: int = 0
    results_found: int = 0
    results_selected: int = 0
    current_round: int = 0
    max_rounds: int = 0
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

    def stop(self) -> None:
        """Stop the live display."""
        if self._live is not None:
            self._live.stop()
            self._live = None

    def _render(self) -> RenderableType:
        """Render the current progress state as a Rich table."""
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold cyan")
        table.add_column(style="bold yellow", justify="right")

        # Round indicator if available
        if self.max_rounds > 0:
            table.add_row("Round", f"{self.current_round}/{self.max_rounds}")

        table.add_row("Searches run", str(self.searches_run))
        table.add_row("Results found", str(self.results_found))
        table.add_row("Results selected", str(self.results_selected))

        return table

    def update(self) -> None:
        """Update the live display with current values."""
        if self._live is not None:
            self._live.update(self._render())

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

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def increment_searches(self, count: int = 1) -> None:
        pass

    def add_results(self, count: int) -> None:
        pass

    def add_selected(self, count: int) -> None:
        pass

    def set_round(self, current: int, max_rounds: int) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
