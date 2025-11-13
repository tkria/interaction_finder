"""Live progress display for widesearch operations.

Provides an in-place updating counter showing searches run, results found,
and results selected during widesearch execution.
"""

from dataclasses import dataclass

from rich.console import RenderableType
from rich.table import Table

from interaction_finder.progress import LiveProgressCounter


@dataclass
class WidesearchProgress(LiveProgressCounter):
    """Live progress counter for widesearch operations.

    Displays real-time statistics with spinner showing current operation
    and bold highlighting of active metrics.
    """

    searches_run: int = 0
    searches_run_this_round: int = 0
    searches_in_progress: int = 0
    searches_total_this_round: int = 0
    results_found: int = 0
    results_selected: int = 0
    current_round: int = 0
    max_rounds: int = 0

    def _render(self) -> RenderableType:
        """Render progress table with widesearch-specific layout."""
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold cyan")
        table.add_column(style="bold yellow", justify="right")
        # Round indicator (conditional)
        if self.max_rounds > 0:
            table.add_row("Round", f"{self.current_round}/{self.max_rounds}")
        # Search metrics (three-part format for this round)
        searches_display = self._format_three_part(
            complete=self.searches_run_this_round,
            in_progress=self.searches_in_progress,
            total=self.searches_total_this_round,
            is_highlighted=(self._highlight == "search"),
            total_is_final=True,  # Total for round known when queries generated
        )
        table.add_row("Searches run", searches_display)
        # Results found (simple counter - cumulative)
        bright = "bold bright_yellow" if self._highlight == "search" else "bold yellow"
        table.add_row("Results found", f"[{bright}]{self.results_found}[/]")
        # Selection metrics (simple counter)
        bright = "bold bright_yellow" if self._highlight == "select" else "bold yellow"
        table.add_row("Results selected", f"[{bright}]{self.results_selected}[/]")
        return self._render_with_header(table)

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


class DummyProgress:
    """No-op progress counter for when display is disabled."""

    def __getattr__(self, name):
        """Return no-op function for any method call."""
        return lambda *args, **kwargs: None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
