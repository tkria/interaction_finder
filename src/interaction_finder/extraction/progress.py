"""Live progress display for extraction operations.

Provides an in-place updating counter showing extraction progress through
documents, entities, pairs, and quote validation.
"""

from dataclasses import dataclass

from rich.console import RenderableType
from rich.table import Table

from interaction_finder.progress import LiveProgressCounter


@dataclass
class ExtractionProgress(LiveProgressCounter):
    """Live progress counter for extraction operations.

    Displays real-time statistics with spinner showing current operation
    and bold highlighting of active metrics.
    """

    documents_processed: int = 0
    documents_in_progress: int = 0
    documents_total: int = 0
    entities_found: int = 0
    pairs_found: int = 0
    pairs_assessed: int = 0
    pairs_in_progress: int = 0
    pairs_total: int = 0
    quotes_validated: int = 0
    quotes_failed: int = 0
    # Co-mention sweep counters
    sweep_co_mentions_found: int = 0
    sweep_no_existing: int = 0
    sweep_uncovered: int = 0
    sweep_assessed: int = 0
    sweep_relationships: int = 0
    # Judgment counters
    unique_pairs: int = 0
    judgments_in_progress: int = 0
    accepted: int = 0
    rejected: int = 0

    def _render(self) -> RenderableType:
        """Render progress table with extraction-specific layout."""
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold cyan")
        table.add_column(style="bold yellow", justify="right")
        table.add_column(style="dim", justify="left")
        # Documents section header
        table.add_row("[bold cyan]Documents[/]", "", "")
        # Documents processed (three-part format: complete/in-progress/total)
        docs_display = self._format_three_part(
            complete=self.documents_processed,
            in_progress=self.documents_in_progress,
            total=self.documents_total,
            is_highlighted=(self._highlight == "docs"),
            total_is_final=True,  # Total known upfront
        )
        table.add_row("  Processed", docs_display, "")
        # Document-level metrics (simple counters)
        bright = "bold bright_yellow" if self._highlight == "docs" else "bold yellow"
        table.add_row("  Entities", f"[{bright}]{self.entities_found}[/]", "")
        # Quote validation (show invalid count in third column)
        quote_annotation = (
            f"({self.quotes_failed} invalid)" if self.quotes_failed > 0 else ""
        )
        table.add_row(
            "  Quotes", f"[bold yellow]{self.quotes_validated}[/]", quote_annotation
        )
        # Pairs assessed (three-part format: assessed/in-progress/found?)
        pairs_total_is_final = (
            self.documents_in_progress == 0
            and self.documents_processed == self.documents_total
        )
        pairs_display = self._format_three_part(
            complete=self.pairs_assessed,
            in_progress=self.pairs_in_progress,
            total=self.pairs_found,
            is_highlighted=(self._highlight in ("pairs", "assessment")),
            total_is_final=pairs_total_is_final,
        )
        table.add_row("  Pairs assessed", pairs_display, "")
        # Co-mention sweep section (only show if sweep has been run)
        if self.sweep_co_mentions_found > 0 or self._highlight == "sweep":
            table.add_row("", "", "")
            table.add_row("[bold cyan]Co-mention Sweep[/]", "", "")
            sweep_bright = (
                "bold bright_yellow" if self._highlight == "sweep" else "bold yellow"
            )
            sweep_annotation = (
                f"({self.sweep_no_existing} new, {self.sweep_uncovered} uncovered)"
                if self.sweep_co_mentions_found > 0
                else ""
            )
            table.add_row(
                "  Found",
                f"[{sweep_bright}]{self.sweep_co_mentions_found}[/]",
                sweep_annotation,
            )
            table.add_row(
                "  Assessed",
                f"[{sweep_bright}]{self.sweep_assessed}[/]",
                f"({self.sweep_relationships} with relationships)"
                if self.sweep_assessed > 0
                else "",
            )
        # Combined judgment section
        table.add_row("", "", "")
        table.add_row("[bold cyan]Combined Judgment[/]", "", "")
        # Unique pairs judgment (three-part format: judged/in-progress/total)
        judged_count = self.accepted + self.rejected
        judgments_total_is_final = (
            self.judgments_in_progress == 0 and self.unique_pairs > 0
        )
        judgment_display = self._format_three_part(
            complete=judged_count,
            in_progress=self.judgments_in_progress,
            total=self.unique_pairs,
            is_highlighted=(self._highlight == "assessment"),
            total_is_final=judgments_total_is_final,
        )
        table.add_row("  Unique pairs", judgment_display, "")
        table.add_row("  Accepted", f"[bold green]{self.accepted}[/]", "")
        table.add_row("  Rejected", f"[bold yellow]{self.rejected}[/]", "")
        return self._render_with_header(table)

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

    def set_phase_sweeping(self) -> None:
        """Show co-mention sweep status."""
        self.set_status("Sweeping for missed co-mentions", highlight="sweep")

    def set_phase_judging(self) -> None:
        """Show cross-document judgment status."""
        self.set_status("Cross-document validation", highlight="assessment")

    def set_phase_finalizing(self) -> None:
        """Show finalization status."""
        self.set_status("Finalizing results")


class DummyProgress:
    """No-op progress counter for when display is disabled."""

    def __getattr__(self, name):
        """Return no-op function for any method call."""
        return lambda *args, **kwargs: None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
