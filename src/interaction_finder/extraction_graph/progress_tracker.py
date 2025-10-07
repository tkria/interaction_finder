"""
Progress tracking and display for verbose mode in the extraction graph.

This module provides real-time progress reporting using Rich components
to show detailed information about the extraction pipeline progress.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import time

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.layout import Layout
from rich.text import Text
from rich import box


@dataclass
class ProgressState:
    """Current state of extraction progress."""

    # Group processing
    total_groups: int = 0
    current_group: int = 0
    current_group_urls: List[str] = field(default_factory=list)
    current_group_chunks: int = 0

    # Node execution
    current_node: Optional[str] = None
    node_path: List[str] = field(default_factory=list)

    # Entity tracking
    entities_found: Dict[str, int] = field(default_factory=dict)
    entities_processed: int = 0

    # Processing stages
    current_stage: Optional[str] = None
    batch_progress: Optional[str] = None

    # Token usage
    total_tokens_used: int = 0

    # Timing
    start_time: float = field(default_factory=time.time)
    stage_start_time: float = field(default_factory=time.time)

    # Errors
    errors_count: int = 0


class VerboseProgressTracker:
    """
    Real-time progress tracking and display for verbose extraction mode.

    This class manages Rich Live display components to show detailed
    progress information during the extraction pipeline execution.
    """

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize the progress tracker.

        Args:
            console: Rich Console instance (creates new if None)
        """
        self.console = console or Console()
        self.state = ProgressState()
        self.live: Optional[Live] = None
        self._is_active = False

    def start(self, total_groups: int) -> None:
        """
        Start progress tracking.

        Args:
            total_groups: Total number of document groups to process
        """
        self.state.total_groups = total_groups
        self.state.start_time = time.time()
        self._is_active = True

        # Create live display
        layout = self._create_layout()
        self.live = Live(layout, console=self.console, refresh_per_second=2)
        self.live.start()

    def stop(self) -> None:
        """Stop progress tracking and clean up display."""
        if self.live and self._is_active:
            self.live.stop()
            self._is_active = False

    def update_group(self, group_index: int, urls: List[str], chunks: int) -> None:
        """
        Update current group information.

        Args:
            group_index: Current group index (0-based)
            urls: URLs in the current group
            chunks: Total chunks in the group
        """
        self.state.current_group = group_index + 1
        self.state.current_group_urls = urls
        self.state.current_group_chunks = chunks
        self._update_display()

    def update_node(self, node_name: str) -> None:
        """
        Update current node being executed.

        Args:
            node_name: Name of the current node
        """
        self.state.current_node = node_name
        if node_name not in self.state.node_path:
            self.state.node_path.append(node_name)
        self.state.stage_start_time = time.time()
        self._update_display()

    def update_stage(self, stage_description: str) -> None:
        """
        Update current processing stage.

        Args:
            stage_description: Description of current stage
        """
        self.state.current_stage = stage_description
        self.state.stage_start_time = time.time()
        self._update_display()

    def update_batch_progress(self, progress_description: str) -> None:
        """
        Update batch processing progress.

        Args:
            progress_description: Description of batch progress (e.g., "Processing entities 1-3 of 15")
        """
        self.state.batch_progress = progress_description
        self._update_display()

    def update_entities(self, entities_by_kind: Dict[str, int]) -> None:
        """
        Update entity counts.

        Args:
            entities_by_kind: Dictionary mapping entity kinds to counts
        """
        self.state.entities_found.update(entities_by_kind)
        self._update_display()

    def update_tokens(self, tokens_used: int) -> None:
        """
        Update token usage.

        Args:
            tokens_used: Additional tokens used
        """
        self.state.total_tokens_used += tokens_used
        self._update_display()

    def update_processed(self, entities_processed: int) -> None:
        """
        Update number of entities processed.

        Args:
            entities_processed: Number of entities processed so far
        """
        self.state.entities_processed = entities_processed
        self._update_display()

    def increment_errors(self, error_details: str = None) -> None:
        """Increment error count with optional details.

        Args:
            error_details: Optional details about the error for verbose reporting
        """
        self.state.errors_count += 1
        if error_details:
            # Store error details for potential verbose reporting
            if not hasattr(self.state, "error_details"):
                self.state.error_details = []
            self.state.error_details.append(error_details)
        self._update_display()

    def report_error(self, error_message: str, context: str = None) -> None:
        """Report an error with detailed context.

        Args:
            error_message: The error message
            context: Additional context about when/where the error occurred
        """
        self.increment_errors()
        error_detail = f"{error_message}"
        if context:
            error_detail = f"{context}: {error_message}"

        # Update the current stage to show error state
        self.update_stage(
            f"⚠️  Error: {error_detail[:50]}{'...' if len(error_detail) > 50 else ''}"
        )

    def _create_layout(self) -> Panel:
        """Create the main layout for the progress display."""
        return self._create_progress_panel()

    def _create_progress_panel(self) -> Panel:
        """Create the main progress panel."""
        table = Table.grid(padding=(0, 2))
        table.add_column("Label", style="cyan", min_width=12)
        table.add_column("Value", style="bright_white")

        # Group progress
        group_text = f"{self.state.current_group}/{self.state.total_groups}"
        if self.state.current_group_urls:
            group_text += f" │ URLs: {len(self.state.current_group_urls)} │ Chunks: {self.state.current_group_chunks}"
        table.add_row("Group:", group_text)

        # Current node path
        if self.state.node_path:
            path_text = " → ".join(self.state.node_path)
            if self.state.current_node:
                # Highlight current node
                path_parts = path_text.split(" → ")
                if self.state.current_node in path_parts:
                    idx = path_parts.index(self.state.current_node)
                    path_parts[idx] = (
                        f"[bold yellow]{self.state.current_node}[/bold yellow]"
                    )
                path_text = " → ".join(path_parts)
            table.add_row("Path:", path_text)

        # Current stage
        if self.state.current_stage:
            stage_text = self.state.current_stage
            # Add spinner for active processing
            stage_text = f"⠸ {stage_text}"
            table.add_row("Stage:", stage_text)

        # Batch progress
        if self.state.batch_progress:
            table.add_row("Progress:", self.state.batch_progress)

        # Entities found
        if self.state.entities_found:
            entities_text = []
            for kind, count in self.state.entities_found.items():
                entities_text.append(f"{kind}: {count}")
            table.add_row("Entities:", " │ ".join(entities_text))

        # Processing stats
        if self.state.entities_processed > 0:
            table.add_row("Processed:", str(self.state.entities_processed))

        # Token usage
        if self.state.total_tokens_used > 0:
            tokens_text = f"{self.state.total_tokens_used:,}"
            table.add_row("Tokens:", tokens_text)

        # Timing
        elapsed = time.time() - self.state.start_time
        stage_elapsed = time.time() - self.state.stage_start_time

        elapsed_text = f"{elapsed:.0f}s"
        if stage_elapsed < elapsed:
            elapsed_text += f" │ Stage: {stage_elapsed:.1f}s"

        table.add_row("Elapsed:", elapsed_text)

        # Errors
        if self.state.errors_count > 0:
            table.add_row("Errors:", f"[red]{self.state.errors_count}[/red]")

        return Panel(
            table,
            title="[bold green]Extraction Progress[/bold green]",
            border_style="green",
            box=box.ROUNDED,
        )

    def _update_display(self) -> None:
        """Update the live display if active."""
        if self.live and self._is_active:
            self.live.update(self._create_progress_panel())

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop display."""
        self.stop()


def create_progress_tracker(
    console: Optional[Console] = None,
) -> Optional[VerboseProgressTracker]:
    """
    Create a progress tracker if console output is supported.

    Args:
        console: Optional console instance

    Returns:
        VerboseProgressTracker if console is interactive, None otherwise
    """
    if console is None:
        console = Console()

    # Only create tracker for interactive terminals
    if console.is_terminal:
        return VerboseProgressTracker(console)
    return None
