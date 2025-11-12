"""Live progress display for extraction operations.

Provides an in-place updating counter showing extraction progress through
documents, entities, pairs, and quote validation.
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
class ExtractionProgress:
    """Live progress counter for extraction operations.

    Displays real-time statistics with spinner showing current operation
    and bold highlighting of active metrics.
    """

    documents_processed: int = 0
    documents_total: int = 0
    entities_found: int = 0
    pairs_found: int = 0
    pairs_assessed: int = 0
    pairs_total: int = 0
    quotes_validated: int = 0
    quotes_failed: int = 0
    accepted: int = 0
    rejected: int = 0
    _status_msg: str = field(default="", init=False)
    _highlight: str = field(default="", init=False)  # "docs", "pairs", "assessment"
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
            highlight: str — "docs", "pairs", or "assessment" to highlight those metrics
        """
        self._status_msg = message
        self._highlight = highlight
        self.update()

    def set_completed(self) -> None:
        """Show completion message with elapsed time."""
        elapsed = int(time() - self._start_time)
        mins, secs = divmod(elapsed, 60)
        time_str = f"{mins}m {secs}s" if mins else f"{secs}s"
        # Calculate acceptance rate
        total = self.accepted + self.rejected
        if total > 0:
            rate = (self.accepted / total) * 100
            self.set_status(
                f"✓ Completed in {time_str} ({self.accepted}/{total} pairs accepted, {rate:.1f}%)"
            )
        else:
            self.set_status(f"✓ Completed in {time_str}")

    def stop(self) -> None:
        """Stop the live display, showing completion status."""
        if self._live:
            self.set_completed()
            self._live.stop()
            self._live = None

    def _render(self) -> RenderableType:
        """Render progress table with optional status header."""
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold cyan")
        table.add_column(style="bold yellow", justify="right")

        # Documents progress
        if self.documents_total > 0:
            bright = (
                "bold bright_yellow" if self._highlight == "docs" else "bold yellow"
            )
            table.add_row(
                "Documents",
                f"[{bright}]{self.documents_processed}/{self.documents_total}[/]",
            )

        # Entity metrics
        bright = "bold bright_yellow" if self._highlight == "docs" else "bold yellow"
        table.add_row("Entities", f"[{bright}]{self.entities_found}[/]")

        # Pair metrics
        bright = "bold bright_yellow" if self._highlight == "pairs" else "bold yellow"
        table.add_row("Pairs found", f"[{bright}]{self.pairs_found}[/]")

        # Assessment progress
        if self.pairs_total > 0:
            bright = (
                "bold bright_yellow"
                if self._highlight == "assessment"
                else "bold yellow"
            )
            table.add_row(
                "Pairs assessed",
                f"[{bright}]{self.pairs_assessed}/{self.pairs_total}[/]",
            )

        # Quote validation
        quotes_total = self.quotes_validated + self.quotes_failed
        quote_style = "bold yellow"
        if self.quotes_failed > 0:
            quote_text = f"{self.quotes_validated}/{quotes_total} [dim](⚠ {self.quotes_failed} failed)[/]"
        else:
            quote_text = f"{self.quotes_validated}/{quotes_total}"
        table.add_row("Quotes valid", f"[{quote_style}]{quote_text}[/]")

        # Final counts
        table.add_row("Accepted", f"[bold green]{self.accepted}[/]")
        table.add_row("Rejected", f"[bold yellow]{self.rejected}[/]")

        # Add status header if active or completed
        if self._status_msg:
            rule = Rule(style="dim", characters="─", width=35)
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

    def set_phase_extracting(self) -> None:
        """Show entity extraction status."""
        self.set_status("Extracting entities", highlight="docs")

    def set_phase_validating(self) -> None:
        """Show entity validation status."""
        self.set_status("Validating entities", highlight="docs")

    def set_phase_proximal(self) -> None:
        """Show proximal set identification status."""
        self.set_status("Finding proximal pairs", highlight="pairs")

    def set_phase_extracting_pairs(self) -> None:
        """Show pair extraction status."""
        self.set_status("Extracting relationships", highlight="pairs")

    def set_phase_assessing(self) -> None:
        """Show pair assessment status."""
        self.set_status("Assessing pairs", highlight="assessment")

    def set_phase_judging(self) -> None:
        """Show cross-document judgment status."""
        self.set_status("Cross-document validation", highlight="assessment")

    def set_phase_finalizing(self) -> None:
        """Show finalization status."""
        self.set_status("Finalizing results")

    def set_phase_idle(self) -> None:
        """Clear status and highlighting."""
        self.set_status("")

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
