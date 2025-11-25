"""Live progress display for keyword extraction operations.

Provides an in-place updating counter showing searches, documents processed,
keywords extracted, and keywords evaluated during keyword research execution.
"""

from dataclasses import dataclass

from rich.console import RenderableType
from rich.table import Table

from interaction_finder.progress import LiveProgressCounter


@dataclass
class KeywordsProgress(LiveProgressCounter):
    """Live progress counter for keyword extraction operations.

    Displays real-time statistics with spinner showing current operation
    and bold highlighting of active metrics.
    """

    searches_run: int = 0
    results_found: int = 0
    # Document processing (three-part: processed/in_progress/total)
    documents_processed: int = 0
    documents_in_progress: int = 0
    documents_total: int = 0
    # Keyword evaluation (three-part: accepted/evaluating/total_extracted)
    keywords_extracted: int = 0
    keywords_evaluating: int = 0
    keywords_accepted: int = 0
    current_round: int = 0
    max_rounds: int = 0

    def start(self) -> None:
        """Start the live display with initial warming up status."""
        self._status_msg = "Warming up"
        super().start()

    def _render(self) -> RenderableType:
        """Render progress table with keywords-specific layout."""
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold cyan")
        table.add_column(style="bold yellow", justify="right")
        # Round indicator (conditional)
        if self.max_rounds > 0:
            table.add_row("Round", f"{self.current_round}/{self.max_rounds}")
        # Search metrics
        bright = "bold bright_yellow" if self._highlight == "search" else "bold yellow"
        table.add_row("Searches run", f"[{bright}]{self.searches_run}[/]")
        table.add_row("Results found", f"[{bright}]{self.results_found}[/]")
        # Document processing (three-part: processed/in_progress/total)
        docs_total_is_final = self.documents_total > 0
        docs_display = self._format_three_part(
            complete=self.documents_processed,
            in_progress=self.documents_in_progress,
            total=self.documents_total if self.documents_total > 0 else None,
            is_highlighted=(self._highlight == "fetch"),
            total_is_final=docs_total_is_final,
        )
        table.add_row("Documents", docs_display)
        # Keyword evaluation (three-part: accepted/evaluating/extracted)
        keywords_total_is_final = (
            self.documents_in_progress == 0
            and self.documents_processed == self.documents_total
            and self.documents_total > 0
        )
        keywords_display = self._format_three_part(
            complete=self.keywords_accepted,
            in_progress=self.keywords_evaluating,
            total=self.keywords_extracted if self.keywords_extracted > 0 else None,
            is_highlighted=(self._highlight == "evaluate"),
            total_is_final=keywords_total_is_final,
        )
        table.add_row("Keywords", keywords_display)
        return self._render_with_header(table)

    def set_phase_searching(self, backend: str = "") -> None:
        """Show searching status with search metrics highlighted."""
        msg = f"Searching {backend}" if backend else "Searching"
        self.set_status(msg, highlight="search")

    def set_phase_reranking(self) -> None:
        """Show reranking status with search metrics highlighted."""
        self.set_status("Reranking results", highlight="search")

    def set_phase_selecting(self) -> None:
        """Show selecting status with search metrics highlighted."""
        self.set_status("Selecting results", highlight="search")

    def set_phase_fetching(self) -> None:
        """Show fetching status with document metrics highlighted."""
        self.set_status("Fetching documents", highlight="fetch")

    def set_phase_extracting(self) -> None:
        """Show extracting status with keyword metrics highlighted."""
        self.set_status("Extracting keywords", highlight="extract")

    def set_phase_evaluating(self) -> None:
        """Show evaluating status with evaluation metrics highlighted."""
        self.set_status("Evaluating keywords", highlight="evaluate")

    def set_phase_reflecting(self) -> None:
        """Show reflection status."""
        self.set_status("Reflecting on coverage", highlight="")

    def increment_searches(self, count: int = 1) -> None:
        """Increment the searches run counter."""
        self.searches_run += count
        self.update()

    def add_results(self, count: int) -> None:
        """Add to the results found counter."""
        self.results_found += count
        self.update()

    def set_documents_total(self, count: int) -> None:
        """Set the total number of documents to process."""
        self.documents_total = count
        self.update()

    def start_document(self) -> None:
        """Mark a document as starting processing."""
        self.documents_in_progress += 1
        self.update()

    def finish_document(self) -> None:
        """Mark a document as finished processing."""
        self.documents_in_progress -= 1
        self.documents_processed += 1
        self.update()

    def add_keywords_extracted(self, count: int) -> None:
        """Add to the keywords extracted counter."""
        self.keywords_extracted += count
        self.update()

    def start_keyword_evaluation(self, count: int = 1) -> None:
        """Mark keywords as starting evaluation."""
        self.keywords_evaluating += count
        self.update()

    def finish_keyword_evaluation(self, accepted: int, rejected: int = 0) -> None:
        """Mark keywords as finished evaluation with accept/reject counts."""
        self.keywords_evaluating -= accepted + rejected
        self.keywords_accepted += accepted
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
