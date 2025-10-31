"""Progress display and status management for fetcher operations."""

from typing import Optional, Protocol
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
)


class StatusProtocol(Protocol):
    """Protocol for status objects that can be updated with text."""

    def update(self, text: str) -> None:
        """Update the status display with new text."""
        ...


class StatusDisplay:
    """Manages Rich status indicators for fetch operations."""

    def __init__(self, show_status: bool = True):
        self.show_status = show_status
        self.console = Console() if show_status else None
        # Avoid noisy "heartbeat" updates when not attached to an interactive TTY
        if self.console and not self.console.is_terminal:
            self.show_status = False

    def create_status(self, message: str, progress_info=None):
        """
        Create appropriate status indicator based on context.

        Args:
            message: Initial status message to display
            progress_info: Optional tuple of (progress_instance, task_id) for progress bars

        Returns:
            Rich status object or DummyStatus for no-op
        """
        if progress_info is not None:
            progress_instance, task_id = progress_info
            progress_instance.update(task_id, description=message)
            return DummyStatus()
        elif self.show_status and self.console:
            return self.console.status(message, spinner="dots")
        else:
            return DummyStatus()

    def batch_progress(self, total: int, description: str = "Processing"):
        """Create a progress bar for batch operations."""
        if not self.show_status:
            return DummyProgress(total)

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(complete_style="blue", finished_style="green"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
        )
        return BatchProgressContext(progress, total, description)


class DummyStatus:
    """No-op status indicator for when status display is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def update(self, text: str):
        pass


class DummyProgress:
    """No-op progress indicator for when progress display is disabled."""

    def __init__(self, total: int):
        self.total = total
        self.completed = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def update(self, advance: int = 1, description: Optional[str] = None):
        self.completed += advance


class BatchProgressContext:
    """Context manager for batch progress operations with Rich progress bars."""

    def __init__(self, progress: Progress, total: int, description: str):
        self.progress = progress
        self.total = total
        self.description = description
        self.task_id = None

    def __enter__(self):
        self.progress.start()
        self.task_id = self.progress.add_task(self.description, total=self.total)
        return self

    def __exit__(self, *args):
        self.progress.stop()

    def update(self, advance: int = 1, description: Optional[str] = None):
        """Update progress bar."""
        if self.task_id is not None:
            self.progress.update(
                self.task_id,
                advance=advance,
                description=description or self.description,
            )
