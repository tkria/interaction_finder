"""Web-facing progress table: streams progress to SSE subscribers.

``WebStatusTable`` is the non-terminal sibling of ``LiveStatusTable``. Instead
of redrawing a Rich Live display, its observer hooks publish JSON messages to
any number of subscriber queues (one per connected SSE client).

Two message kinds travel the same stream:

- ``event`` -- append-only; the client tracks the highest index it has seen and
  resumes with ``events[since:]`` from the inherited log (the queues are
  ephemeral, the log is the source of truth).
- ``counters`` / ``status`` / ``done`` -- latest-wins snapshots; the client
  simply replaces its current view, mirroring how the TUI re-renders the whole
  table on every change.

A stage boundary (``clear_all``) surfaces as a ``meta.cleared`` event: the
current stage, whatever it was, has ended; the next stage is identified by the
scope of the following event.
"""

import asyncio
from dataclasses import asdict
from time import time
from typing import Any, Literal

from interaction_finder.progress import Counter, Event, StatusTable

# Per-subscriber queue depth. The event log is the source of truth for
# catch-up, so a full queue drops its oldest message rather than blocking the
# pipeline; a lagging client recovers via the log on its next poll/reconnect.
_QUEUE_MAXSIZE = 1000


class WebStatusTable(StatusTable):
    """Progress table that publishes JSON messages to SSE subscriber queues."""

    def __init__(self, *counters: Counter):
        super().__init__(*counters)
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._start_time: float | None = None
        self._result: Literal["pending", "running", "success", "failure"] = "pending"

    # === Subscription ===
    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """Register a new SSE client; returns its message queue."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Drop a disconnected client's queue."""
        self._subscribers.discard(queue)

    def events_since(self, index: int) -> list[dict[str, Any]]:
        """Catch-up payload: events at or after ``index`` as wire messages."""
        return [
            self._event_message(event, index + offset)
            for offset, event in enumerate(self.events[index:])
        ]

    def counter_snapshot(self) -> dict[str, Any]:
        """Latest counter-table snapshot as a wire message."""
        return {"type": "counters", "counters": self._counters_payload()}

    def status_snapshot(self) -> dict[str, Any]:
        """Current status line + elapsed as a wire message."""
        return {"type": "status", "status": self.status, "elapsed": self._elapsed()}

    # === Lifecycle (mirrors LiveStatusTable semantics) ===
    def start(self) -> None:
        """Begin the run timer; idempotent across re-entered stage contexts."""
        if self._start_time is None:
            self._start_time = time()
        self._result = "running"

    def stop(self) -> None:
        """No persistent display to tear down."""

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Stages run inside `with progress:`; the base __exit__ would publish a
        # terminal `done` per stage. The run's terminal state spans the whole
        # ensure_extraction chain and is published by RunManager._drive, so a
        # per-stage exit must stay quiet.
        self.stop()

    def succeed(self, has_report: bool = True) -> None:
        """Mark the run successful and publish a terminal message.

        ``has_report`` is False when extraction completed but found nothing to
        report, so the UI suppresses the View-report action.
        """
        self._result = "success"
        self.status = f"✓ Completed in {self._elapsed()}"
        self._publish(
            {
                "type": "done",
                "result": "success",
                "status": self.status,
                "has_report": has_report,
            }
        )

    def fail(self) -> None:
        """Mark the run failed and publish a terminal message."""
        self._result = "failure"
        self.status = f"✗ Failed after {self._elapsed()}"
        self._publish({"type": "done", "result": "failure", "status": self.status})

    def cancel(self) -> None:
        """Mark the run cancelled and publish a terminal message."""
        self._result = "cancelled"
        self.status = f"⊘ Cancelled after {self._elapsed()}"
        self._publish({"type": "done", "result": "cancelled", "status": self.status})

    # === Observer hooks ===
    def set_status(self, message: str) -> None:
        self.status = message
        self._publish(self.status_snapshot())

    def _on_counter_changed(self) -> None:
        self._publish(self.counter_snapshot())

    def _update(self) -> None:
        # add_counters/clear_all call this; reflect the new counter set.
        self._publish(self.counter_snapshot())

    def _on_event(self, event: Event) -> None:
        self._publish(self._event_message(event, len(self.events) - 1))

    def _on_clear(self) -> None:
        # A stage boundary is itself an event so the append-only log (and thus
        # back-fill and reconnection) records it like everything else.
        self.emit("meta.cleared", "")

    # === Internal ===
    def _publish(self, message: dict[str, Any]) -> None:
        """Fan a message out to every subscriber, dropping oldest on overflow."""
        for queue in self._subscribers:
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                # Source of truth is the event log; shed the oldest queued
                # message so a slow client never back-pressures the pipeline.
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                queue.put_nowait(message)

    @staticmethod
    def _event_message(event: Event, index: int) -> dict[str, Any]:
        return {"type": "event", "index": index, **asdict(event)}

    def _counters_payload(self) -> list[dict[str, Any]]:
        """JSON projection of the counter table (data behind _render, not markup)."""
        return [
            {
                "name": c.name,
                "category": c.category,
                "note": c.note,
                "total": c.total,
                "in_progress": c.in_progress,
                "completed": c.completed,
                "status": c.status,
            }
            for c in self.counters
        ]

    def _elapsed(self) -> str:
        if self._start_time is None:
            return "0s"
        elapsed = int(time() - self._start_time)
        mins, secs = divmod(elapsed, 60)
        return f"{mins}m {secs}s" if mins else f"{secs}s"
