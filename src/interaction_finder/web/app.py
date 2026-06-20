"""Starlette application: serves the runner UI and streams run progress.

Routes:
- ``GET  /``                  the single-page runner UI
- ``POST /runs``              start (or resume) a run; returns ``{run_id}``
- ``GET  /runs/{id}/events``  SSE stream of progress (catch-up + live)
- ``GET  /report/{id}``       the generated HTML report for a finished run
- ``GET  /static/*``          vendored Pico CSS and assets

A run started from an existing checkpoint path resumes it (ensure_* is
idempotent). The SSE stream first replays ``events_since(since)`` and the
current counter/status snapshots, then forwards live messages until ``done``.
"""

import asyncio
import json
import re
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from starlette.routing import Mount, Route
from starlette.concurrency import run_in_threadpool
from starlette.staticfiles import StaticFiles

from interaction_finder.web.run_manager import (
    RunManager,
    RunRecord,
    RunSpec,
    checkpoint_status,
)

_WEB_DIR = Path(__file__).parent
_STATIC_DIR = _WEB_DIR / "static"
_TEMPLATE = _WEB_DIR / "templates" / "ui.html"

# SSE heartbeat so proxies/clients keep the idle connection open between events.
_HEARTBEAT_SECONDS = 15.0
# How often the report-generation stream emits a progress tick.
_REPORT_POLL_SECONDS = 0.1


def create_app() -> Starlette:
    """Build the Starlette app with a fresh in-process RunManager."""
    manager = RunManager()

    async def index(request: Request) -> HTMLResponse:
        return HTMLResponse(
            _TEMPLATE.read_text().replace("{{ config_form }}", _config_form_html())
        )

    async def theme_css(request: Request) -> Response:
        # Shared colour/spacing tokens, reused from the report so the UI and the
        # report it produces look like one product.
        from interaction_finder.report.assets import THEME_CSS

        return Response(THEME_CSS, media_type="text/css")

    async def start_run(request: Request) -> JSONResponse:
        body = await request.json()
        spec = _spec_from_body(body)
        if not spec.checkpoint_or_topic.strip():
            return JSONResponse({"error": "A topic or checkpoint is required."}, 400)
        if not spec.entity_kinds:
            return JSONResponse({"error": "At least one entity kind is required."}, 400)
        try:
            record = manager.start(spec)
        except (FileNotFoundError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, 400)
        return JSONResponse({"run_id": record.id, "topic": record.topic})

    async def cancel_run(request: Request) -> JSONResponse:
        ok = manager.cancel(request.path_params["run_id"])
        if not ok:
            return JSONResponse({"error": "No running run with that id."}, 404)
        return JSONResponse({"ok": True})

    async def run_events(request: Request) -> Response:
        record = manager.get(request.path_params["run_id"])
        if record is None:
            return JSONResponse({"error": "Unknown run id."}, 404)
        since = int(request.query_params.get("since", 0))
        return _sse_response(record, since)

    async def load_checkpoint(request: Request) -> JSONResponse:
        path = (await request.json()).get("path", "").strip()
        if not path:
            return JSONResponse({"error": "A checkpoint path is required."}, 400)
        try:
            # Parsing/rehydrating a large checkpoint is slow and synchronous;
            # run it off the event loop so the server stays responsive.
            record = await run_in_threadpool(manager.load_checkpoint, path)
        except (FileNotFoundError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, 400)
        return JSONResponse(
            {
                "run_id": record.id,
                "topic": record.topic,
                "report_cached": manager.report_cached(record),
                **checkpoint_status(record.checkpoint),
            }
        )

    async def resume_run(request: Request) -> JSONResponse:
        body = await request.json()
        path = str(body.get("path", "")).strip()
        if not path:
            return JSONResponse({"error": "A checkpoint path is required."}, 400)
        kinds = [k for k in body.get("entity_kinds", []) if k]
        try:
            # Heavy parse off the loop; spawn the task back on the loop.
            record = await run_in_threadpool(
                manager.prepare_resume, path, kinds, body.get("backend") or None
            )
        except (FileNotFoundError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, 400)
        manager.spawn(record)
        return JSONResponse({"run_id": record.id, "topic": record.topic})

    async def browse_dir(request: Request) -> JSONResponse:
        # Read-only directory listing for the in-page file picker. Lists
        # subdirectories and *.json files only; never reads file contents.
        raw = request.query_params.get("dir", "")
        try:
            return JSONResponse(_list_dir(raw))
        except (FileNotFoundError, NotADirectoryError, PermissionError) as exc:
            return JSONResponse({"error": str(exc)}, 400)

    async def recent_list(request: Request) -> JSONResponse:
        from dataclasses import asdict

        from interaction_finder.web.recent import load_recent
        from interaction_finder.web.report_cache import report_path_for_key

        entries = []
        for e in load_recent():
            d = asdict(e)
            # Cheap existence check by stored key -- no checkpoint reload.
            d["report_cached"] = bool(
                e.report_key and report_path_for_key(e.report_key).exists()
            )
            entries.append(d)
        return JSONResponse({"entries": entries})

    async def recent_report(request: Request) -> Response:
        from interaction_finder.web.recent import find_recent
        from interaction_finder.web.report_cache import report_path_for_key

        entry = find_recent(request.path_params["entry_id"])
        if entry is None or not entry.report_key:
            return JSONResponse({"error": "No report for this entry."}, 404)
        path = report_path_for_key(entry.report_key)
        if not path.exists():
            return JSONResponse({"error": "Report not generated."}, 404)
        return FileResponse(str(path), media_type="text/html")

    async def recent_delete(request: Request) -> JSONResponse:
        from interaction_finder.web.recent import remove_recent

        remove_recent(request.path_params["entry_id"])
        return JSONResponse({"ok": True})

    async def recent_clear(request: Request) -> JSONResponse:
        from interaction_finder.web.recent import clear_recent

        clear_recent()
        return JSONResponse({"ok": True})

    async def report_generate(request: Request) -> Response:
        record = manager.get(request.path_params["run_id"])
        if record is None or record.checkpoint is None:
            return JSONResponse({"error": "No report available."}, 404)
        if manager.report_cached(record):
            return JSONResponse({"ok": True})  # nothing to render

        # Stream document-rendering progress while generating in a worker thread.
        async def stream():
            counter = manager.begin_report_counter(record)
            task = asyncio.ensure_future(
                run_in_threadpool(manager.generate_report, record)
            )
            while not task.done():
                yield _sse(
                    {
                        "type": "report_progress",
                        "completed": counter.completed,
                        "total": counter.total,
                    }
                )
                try:
                    await asyncio.wait_for(
                        asyncio.shield(task), timeout=_REPORT_POLL_SECONDS
                    )
                except asyncio.TimeoutError:
                    continue
            exc = task.exception()
            if exc is not None:
                yield _sse({"type": "report_error", "error": str(exc)})
            else:
                yield _sse({"type": "report_done"})

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def keys_status(request: Request) -> JSONResponse:
        from interaction_finder.web.api_keys import key_status

        return JSONResponse({"providers": key_status()})

    async def keys_set(request: Request) -> JSONResponse:
        # Set provider keys on the server process env for this session only;
        # values are never persisted or echoed back.
        from interaction_finder.web.api_keys import key_status, set_keys

        values = dict((await request.json()).get("keys", {}))
        set_keys(values)
        return JSONResponse({"providers": key_status()})

    async def config_default(request: Request) -> JSONResponse:
        # Persist the edited config as the user's default. The body carries only
        # the fields changed from spec defaults (dotted-key overrides); we apply
        # them onto a fresh default config and save the full result.
        from interaction_finder.settings import IfetcherConfig
        from interaction_finder.web.config_store import save_default_config

        overrides = dict((await request.json()).get("overrides", {}))
        base = IfetcherConfig().model_dump()
        try:
            merged = IfetcherConfig.apply_overrides(base, overrides)
            IfetcherConfig.model_validate(merged)  # reject invalid edits up front
        except (ValueError, TypeError) as exc:
            return JSONResponse({"error": f"Invalid configuration: {exc}"}, 400)
        save_default_config(merged)
        return JSONResponse({"ok": True})

    async def report(request: Request) -> Response:
        record = manager.get(request.path_params["run_id"])
        if record is None or record.checkpoint is None:
            return JSONResponse({"error": "No report available."}, 404)
        path = await run_in_threadpool(manager.generate_report, record)
        # ?download=1 -> attachment with a topic-derived filename; otherwise
        # serve inline so a new browser tab renders it.
        if request.query_params.get("download"):
            return FileResponse(
                path, media_type="text/html", filename=_report_filename(record.topic)
            )
        return FileResponse(path, media_type="text/html")

    routes = [
        Route("/", index),
        Route("/theme.css", theme_css),
        Route("/runs", start_run, methods=["POST"]),
        Route("/load", load_checkpoint, methods=["POST"]),
        Route("/resume", resume_run, methods=["POST"]),
        Route("/browse", browse_dir),
        Route("/config/default", config_default, methods=["POST"]),
        Route("/keys", keys_status),
        Route("/keys", keys_set, methods=["POST"]),
        Route("/recent", recent_list),
        Route("/recent", recent_clear, methods=["DELETE"]),
        Route("/recent/{entry_id}/report", recent_report),
        Route("/recent/{entry_id}", recent_delete, methods=["DELETE"]),
        Route("/runs/{run_id}/cancel", cancel_run, methods=["POST"]),
        Route("/runs/{run_id}/events", run_events),
        Route("/report/{run_id}/events", report_generate),
        Route("/report/{run_id}", report),
        Mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static"),
    ]
    return Starlette(routes=routes)


def _spec_from_body(body: dict) -> RunSpec:
    return RunSpec(
        checkpoint_or_topic=str(body.get("checkpoint_or_topic", "")).strip(),
        entity_kinds=[k for k in body.get("entity_kinds", []) if k],
        backend=body.get("backend") or None,
        output_path=body.get("output_path") or None,
        config_overrides=dict(body.get("config_overrides", {})),
        force=bool(body.get("force", False)),
    )


def _sse_response(record: RunRecord, since: int) -> Response:
    """An SSE stream: catch-up from the log, then live messages until done."""
    table = record.table

    async def stream():
        # 1. Catch up: every event the client hasn't seen, plus latest snapshots.
        for message in table.events_since(since):
            yield _sse(message)
        yield _sse(table.counter_snapshot())
        yield _sse(table.status_snapshot())
        # If the run already finished before we attached, emit the terminal msg.
        if record.status != "running":
            yield _sse({"type": "done", "result": record.status, "error": record.error})
            return
        # 2. Live: forward messages as they are published.
        queue = table.subscribe()
        try:
            while True:
                try:
                    message = await asyncio.wait_for(
                        queue.get(), timeout=_HEARTBEAT_SECONDS
                    )
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield _sse(message)
                if message.get("type") == "done":
                    return
        finally:
            table.unsubscribe(queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _config_form_html() -> str:
    """Build the config editor form from the spec, pre-filled with the effective
    default config (spec defaults overlaid with any saved user-default)."""
    from interaction_finder.settings import IfetcherConfig
    from interaction_finder.web.config_form import build_config_form_html
    from interaction_finder.web.config_store import load_default_config

    effective = IfetcherConfig().model_dump()
    saved = load_default_config()
    if saved is not None:
        effective = IfetcherConfig.apply_overrides(effective, _flatten_config(saved))
    return build_config_form_html(effective)


def _flatten_config(data: dict, prefix: str = "") -> dict:
    """Flatten a nested config dict to dotted-key leaves for apply_overrides."""
    flat = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_config(value, path))
        else:
            flat[path] = value
    return flat


def _sse(message: dict) -> str:
    return f"data: {json.dumps(message)}\n\n"


def _report_filename(topic: str) -> str:
    """A safe download filename derived from the run's topic.

    Keeps alphanumerics and spaces/hyphens, collapses runs of other
    characters to a single hyphen, and falls back to "report" when nothing
    usable remains.
    """
    slug = re.sub(r"[^\w\s-]", "", topic).strip().lower()
    slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
    return f"{slug or 'report'}.html"


def _list_dir(raw: str) -> dict:
    """List a directory's subdirectories and .json files for the file picker.

    Parameters:
        raw: Requested directory path; empty string falls back to the cwd.

    Returns a dict with the resolved ``dir``, its ``parent`` (None at the root),
    and ``entries`` (directories first, then .json files, each alphabetical).

    Raises FileNotFoundError / NotADirectoryError / PermissionError on a path
    that cannot be listed.
    """
    base = Path(raw).expanduser() if raw else Path.cwd()
    base = base.resolve()
    if not base.exists():
        raise FileNotFoundError(f"No such directory: {base}")
    if not base.is_dir():
        raise NotADirectoryError(f"Not a directory: {base}")
    dirs, files = [], []
    for entry in base.iterdir():
        if entry.name.startswith("."):
            continue  # hide dotfiles/dirs from the picker
        if entry.is_dir():
            dirs.append(entry)
        elif entry.suffix == ".json":
            files.append(entry)
    sort_key = lambda p: p.name.lower()
    entries = [
        {"name": p.name, "path": str(p), "is_dir": True}
        for p in sorted(dirs, key=sort_key)
    ] + [
        {"name": p.name, "path": str(p), "is_dir": False}
        for p in sorted(files, key=sort_key)
    ]
    parent = None if base.parent == base else str(base.parent)
    return {
        "dir": str(base),
        "parent": parent,
        "home": str(Path.home()),
        "entries": entries,
    }
