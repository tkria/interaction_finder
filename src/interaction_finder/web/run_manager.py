"""In-memory registry and lifecycle for pipeline runs driven from the web UI.

A ``RunManager`` owns the active runs for a single-user local server. Each run
is an asyncio task wrapping the same ``ensure_extraction`` chain the CLI uses,
fed by a ``WebStatusTable`` so the browser can stream progress over SSE.

Starting a run from an existing checkpoint path *is* resumption: the
``ensure_*`` chain is idempotent and picks up at the first incomplete stage.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from interaction_finder.checkpoint import PipelineCheckpoint
from interaction_finder.settings import IfetcherConfig
from interaction_finder.progress import Counter
from interaction_finder.web.table import WebStatusTable

RunStatus = Literal["running", "success", "failure", "cancelled"]


@dataclass(frozen=True)
class RunSpec:
    """Inputs for a run, as supplied by the new-run form.

    ``checkpoint_or_topic`` is either an existing checkpoint file path (resume)
    or a topic string (fresh run). ``entity_kinds`` is the raw list with repeats
    encoding self-pair permission, exactly as the CLI ``-e`` flag works.
    """

    checkpoint_or_topic: str
    entity_kinds: list[str]
    backend: str | None = None
    output_path: str | None = None
    config_overrides: dict[str, str] = field(default_factory=dict)
    force: bool = False


@dataclass
class RunRecord:
    """A live or finished run and everything the UI needs to observe it."""

    id: str
    topic: str
    spec: RunSpec
    table: WebStatusTable
    checkpoint_path: str | None
    status: RunStatus = "running"
    error: str | None = None
    task: asyncio.Task | None = None
    checkpoint: PipelineCheckpoint | None = None
    # Drive args for a prepared-but-not-started resume (set by prepare_resume,
    # consumed by spawn); None once the task is running.
    pending_drive: tuple | None = None
    # Progress counter for an in-flight report generation (advanced per
    # document); read by the report-progress SSE route.
    report_counter: Counter | None = None


class RunManager:
    """Tracks runs by id; starts and observes their lifecycle."""

    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}

    def get(self, run_id: str) -> RunRecord | None:
        """Return the run with this id, or None."""
        return self._runs.get(run_id)

    def load_checkpoint(self, path: str) -> RunRecord:
        """Register a finished run reconstructed from a saved checkpoint file.

        The record's table is pre-seeded with the back-filled event log, so the
        UI renders a loaded file through the same path as a live run. No task is
        spawned -- the run is already complete.

        Raises:
            FileNotFoundError: path does not exist.
            ValueError: file is not a readable checkpoint.
        """
        from interaction_finder.web.backfill import checkpoint_to_events

        checkpoint = _read_checkpoint(path)
        events = checkpoint_to_events(checkpoint)
        if not events:
            raise ValueError(
                f"{path!r} has no recognisable pipeline stages to display."
            )
        table = WebStatusTable()
        for event in events:
            table.events.append(event)
        record = RunRecord(
            id=uuid.uuid4().hex[:12],
            topic=checkpoint.topic,
            spec=RunSpec(checkpoint_or_topic=path, entity_kinds=[]),
            table=table,
            checkpoint_path=path,
            status="success",
            checkpoint=checkpoint,
        )
        self._runs[record.id] = record
        self._record_recent(record)
        return record

    def _record_recent(self, record: RunRecord) -> None:
        """Remember this checkpoint in the recent-history registry."""
        from datetime import datetime, timezone

        from interaction_finder.web.recent import record_recent
        from interaction_finder.web.report_cache import report_cache_key

        if not record.checkpoint_path or record.checkpoint is None:
            return
        extraction = record.checkpoint.extraction
        pair_count = len(extraction.judgments) if extraction is not None else 0
        record_recent(
            path=record.checkpoint_path,
            topic=record.topic,
            pair_count=pair_count,
            opened_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            report_key=report_cache_key(record.checkpoint),
            complete=checkpoint_status(record.checkpoint)["complete"],
        )

    def _record_partial(self, record: RunRecord) -> None:
        """Record a cancelled/failed run's on-disk checkpoint as incomplete.

        The run never returned a checkpoint object, but completed stages have
        written one to ``checkpoint_path``; re-read it so the partial result is
        relistable (and flagged incomplete). Silent no-op if nothing was
        written yet -- there is simply nothing to remember.
        """
        if not record.checkpoint_path or not Path(record.checkpoint_path).is_file():
            return
        try:
            record.checkpoint = _read_checkpoint(record.checkpoint_path)
        except (FileNotFoundError, ValueError):
            return
        self._record_recent(record)

    def report_path(self, record: RunRecord) -> Path | None:
        """Cache path for this run's report (None if it has no checkpoint)."""
        from interaction_finder.web.report_cache import report_cache_path

        if record.checkpoint is None:
            return None
        return report_cache_path(record.checkpoint)

    def report_cached(self, record: RunRecord) -> bool:
        """True if a cached report already exists for this run's checkpoint."""
        path = self.report_path(record)
        return path is not None and path.exists()

    def begin_report_counter(self, record: RunRecord) -> Counter:
        """Create the progress counter the report-generation stream reads from.

        Returned to the SSE route, which polls its completed/total while
        generate_report runs in a worker thread and advances it per document.
        """
        counter = Counter("Report documents", track_in_progress=False)
        record.report_counter = counter
        return counter

    def generate_report(self, record: RunRecord) -> Path:
        """Render the report into the cross-platform cache and return its path.

        A cache hit returns immediately. On a miss, document rendering advances
        ``record.report_counter`` (created by begin_report_counter) so an SSE
        route can report progress. Heavy/blocking; call off the event loop.
        """
        from interaction_finder.report.generator import generate_report

        if record.checkpoint is None:
            raise ValueError("This run has no checkpoint to report on.")
        path = self.report_path(record)
        if path.exists():
            return path
        counter = record.report_counter

        def on_progress(done: int, total: int) -> None:
            if counter is not None:
                counter.total = total
                counter.completed = done

        generate_report(
            record.checkpoint, path, format="html", progress_callback=on_progress
        )
        # Generating a report counts as interacting with the checkpoint; refresh
        # its recent-history entry to the front.
        self._record_recent(record)
        return path

    def start(self, spec: RunSpec) -> RunRecord:
        """Build and spawn a run; return its record immediately.

        Resolves config, backend, and the starting checkpoint up front (so input
        errors surface synchronously), then spawns the pipeline as a tracked
        task. The record's table is already subscribable before the task runs.
        """
        from interaction_finder.cli import (
            create_search_backend,
            default_output_path,
            load_checkpoint_or_create,
            load_config,
        )

        checkpoint, topic = load_checkpoint_or_create(spec.checkpoint_or_topic)
        config = load_config(
            None,
            list(_as_override_list(spec.config_overrides)),
            None,
            fallback_config=_baseline_config(checkpoint),
        )
        input_is_file = Path(spec.checkpoint_or_topic).is_file()
        output = spec.output_path
        if not output and not input_is_file:
            output = str(default_output_path(config, topic))
        checkpoint_path = output or (
            spec.checkpoint_or_topic if input_is_file else None
        )
        backend_name = spec.backend or config.stage.search.search_backend
        search_backend = create_search_backend(backend_name, config)

        record = RunRecord(
            id=uuid.uuid4().hex[:12],
            topic=topic,
            spec=spec,
            table=WebStatusTable(),
            checkpoint_path=checkpoint_path,
        )
        self._runs[record.id] = record
        record.task = asyncio.ensure_future(
            self._drive(record, checkpoint, search_backend, config)
        )
        return record

    def prepare_resume(
        self, path: str, entity_kinds: list[str], backend: str | None
    ) -> RunRecord:
        """Build (but do not start) a resumed run from a checkpoint file.

        Does the heavy, synchronous work -- parsing the checkpoint and seeding
        the table with its back-filled events -- so it is safe to call off the
        event loop. The drive coroutine's arguments are stashed on the record;
        call ``spawn`` (on the loop) to start it. Entity kinds may be empty when
        the checkpoint already records them (a resumed incomplete extraction).
        """
        from interaction_finder.cli import create_search_backend, load_config
        from interaction_finder.web.backfill import checkpoint_to_events

        checkpoint = _read_checkpoint(path)
        # The checkpoint's embedded config is the implicit --config baseline
        # when the user has not supplied a config file (falling back to the
        # saved user-default, then bare spec defaults).
        config = load_config(
            None, None, None, fallback_config=_baseline_config(checkpoint)
        )
        kinds = entity_kinds or (
            list(checkpoint.extraction.target_entity_types)
            if checkpoint.extraction is not None
            else []
        )
        backend_name = backend or config.stage.search.search_backend
        search_backend = create_search_backend(backend_name, config)
        table = WebStatusTable()
        for event in checkpoint_to_events(checkpoint):
            table.events.append(event)
        record = RunRecord(
            id=uuid.uuid4().hex[:12],
            topic=checkpoint.topic,
            spec=RunSpec(checkpoint_or_topic=path, entity_kinds=kinds),
            table=table,
            checkpoint_path=path,
        )
        record.pending_drive = (checkpoint, search_backend, config)
        self._runs[record.id] = record
        return record

    def spawn(self, record: RunRecord) -> None:
        """Start a prepared run's pipeline task (call from the event loop)."""
        checkpoint, search_backend, config = record.pending_drive
        record.pending_drive = None
        record.task = asyncio.ensure_future(
            self._drive(record, checkpoint, search_backend, config)
        )

    def cancel(self, run_id: str) -> bool:
        """Request cancellation of a running run. Returns True if it was running.

        Cancels the pipeline task; ``_drive`` catches the resulting
        CancelledError, marks the run cancelled, and emits the terminal event.
        Any checkpoint already written by completed stages is left on disk, so
        the run can be resumed from where it stopped.
        """
        record = self._runs.get(run_id)
        if record is None or record.status != "running" or record.task is None:
            return False
        record.task.cancel()
        return True

    async def _drive(
        self,
        record: RunRecord,
        checkpoint: PipelineCheckpoint,
        search_backend,
        config: IfetcherConfig,
    ) -> None:
        """Run ensure_extraction to completion, updating the record + table."""
        from interaction_finder.upgrade import ensure_extraction

        record.table.start()
        try:
            result = await ensure_extraction(
                checkpoint,
                record.spec.entity_kinds,
                search_backend,
                config,
                record.table,
                console=None,
                checkpoint_path=record.checkpoint_path,
                force=record.spec.force,
            )
            record.checkpoint = result
            record.status = "success"
            record.table.succeed()
            self._record_recent(record)
        except asyncio.CancelledError:
            # User-requested cancel: report it as a terminal state rather than
            # letting it propagate as an error. Partial checkpoints survive, so
            # record what reached disk (flagged as incomplete in recents).
            record.status = "cancelled"
            record.table.cancel()
            self._record_partial(record)
        except Exception as exc:  # surfaced to the UI, never silently swallowed
            record.status = "failure"
            record.error = f"{type(exc).__name__}: {exc}"
            record.table.fail()
            self._record_partial(record)


def _read_checkpoint(path: str) -> PipelineCheckpoint:
    """Parse a checkpoint file directly (no .bak, no version-warning prints).

    Migration/rehydration is handled by model_validate_json itself; the CLI
    loader's backup-and-print side effects are unwanted for a read-only UI load.

    Raises:
        FileNotFoundError: path is missing or not a file.
        ValueError: file is not a valid checkpoint.
    """
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"No checkpoint file at {path!r}.")
    try:
        return PipelineCheckpoint.model_validate_json(file.read_text())
    except Exception as exc:
        raise ValueError(f"{path!r} is not a valid checkpoint: {exc}") from exc


def _baseline_config(checkpoint: PipelineCheckpoint) -> dict | None:
    """Baseline config for a run when the user supplies no config file.

    Precedence: the checkpoint's embedded config (the implicit ``--config``),
    then the persisted user-default config, then None (bare spec defaults).
    """
    if checkpoint.config:
        return checkpoint.config
    from interaction_finder.web.config_store import load_default_config

    return load_default_config()


def checkpoint_status(checkpoint: PipelineCheckpoint) -> dict:
    """Summarise a loaded checkpoint for the UI's continue/view decision.

    Returns:
        stage: highest completed stage ("none"|"keywords"|"search"|"extraction").
        complete: True when extraction has run to completion (view-only).
        resumable: True when the pipeline can be continued from here.
        needs_entity_kinds: True when continuing requires entity kinds the
            checkpoint does not yet carry (i.e. extraction has not run).
        entity_kinds: kinds already recorded (for an incomplete extraction).
    """
    from interaction_finder.upgrade import checkpoint_stage

    stage = checkpoint_stage(checkpoint)
    extraction = checkpoint.extraction
    complete = extraction is not None and extraction.metadata.is_complete
    has_extraction = extraction is not None
    return {
        "stage": stage,
        "complete": complete,
        "resumable": not complete,
        "needs_entity_kinds": not has_extraction,
        "entity_kinds": list(extraction.target_entity_types) if has_extraction else [],
    }


def _as_override_list(overrides: dict[str, str]):
    """Render an overrides dict as the ``key=value`` strings load_config expects."""
    for key, value in overrides.items():
        yield f"{key}={value}"
