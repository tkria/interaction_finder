"""Route + SSE tests for the web app, with the pipeline stubbed out.

These exercise the HTTP/SSE wiring (start -> stream -> done, catch-up, errors)
without running the real pipeline: the RunManager's pipeline driver is replaced
by a coroutine that drives a WebStatusTable through a few events and succeeds.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from starlette.testclient import TestClient

from interaction_finder.web import app as app_module
from interaction_finder.web.run_manager import RunRecord, RunSpec
from interaction_finder.web.table import WebStatusTable


@pytest.fixture
def stub_driver(monkeypatch):
    """Replace RunManager._drive so start() spawns a fast, network-free run."""

    async def fake_drive(self, record, *_args):
        record.table.start()
        record.table.emit("search.queries", "2 queries", round=1, queries=["a", "b"])
        record.table.clear_all()
        record.table.emit(
            "extraction.pair_judged", "TP53–cancer", entity1="TP53", entity2="cancer"
        )
        record.status = "success"
        record.table.succeed()

    def fake_start(self, spec: RunSpec) -> RunRecord:
        record = RunRecord(
            id="testrun01",
            topic=spec.checkpoint_or_topic,
            spec=spec,
            table=WebStatusTable(),
            checkpoint_path=None,
        )
        self._runs[record.id] = record
        record.task = asyncio.ensure_future(fake_drive(self, record))
        return record

    monkeypatch.setattr(
        "interaction_finder.web.run_manager.RunManager.start", fake_start
    )


@pytest.fixture
def client(stub_driver):
    return TestClient(app_module.create_app())


def parse_sse(text: str) -> list[dict]:
    """Extract JSON payloads from an SSE response body (skip keep-alives)."""
    out = []
    for line in text.splitlines():
        if line.startswith("data: "):
            out.append(json.loads(line[len("data: ") :]))
    return out


def test_index_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Interaction Finder" in resp.text


def test_start_run_requires_topic(client):
    resp = client.post("/runs", json={"entity_kinds": ["gene"]})
    assert resp.status_code == 400
    assert "required" in resp.json()["error"]


def test_start_run_requires_entity_kinds(client):
    resp = client.post("/runs", json={"checkpoint_or_topic": "cancer"})
    assert resp.status_code == 400


def test_start_run_returns_run_id(client):
    resp = client.post(
        "/runs", json={"checkpoint_or_topic": "cancer", "entity_kinds": ["gene"]}
    )
    assert resp.status_code == 200
    assert resp.json()["run_id"] == "testrun01"


def test_events_unknown_run(client):
    assert client.get("/runs/nope/events").status_code == 404


def test_events_stream_replays_full_run(client):
    run_id = client.post(
        "/runs", json={"checkpoint_or_topic": "cancer", "entity_kinds": ["gene"]}
    ).json()["run_id"]
    # The stubbed run finishes synchronously on the loop; the SSE catch-up path
    # replays the whole event log plus the terminal done message.
    body = client.get(f"/runs/{run_id}/events?since=0").text
    messages = parse_sse(body)
    scopes = [m["scope"] for m in messages if m["type"] == "event"]
    assert "search.queries" in scopes
    assert "meta.cleared" in scopes
    assert "extraction.pair_judged" in scopes
    assert any(m["type"] == "done" and m["result"] == "success" for m in messages)


def test_events_since_skips_seen(client):
    run_id = client.post(
        "/runs", json={"checkpoint_or_topic": "cancer", "entity_kinds": ["gene"]}
    ).json()["run_id"]
    body = client.get(f"/runs/{run_id}/events?since=2").text
    events = [m for m in parse_sse(body) if m["type"] == "event"]
    # First two events (indices 0,1) are skipped; only index 2 onward replays.
    assert all(m["index"] >= 2 for m in events)


# Cancellation is tested at the RunManager level on a live event loop, because
# Starlette's TestClient tears down pending tasks between requests (so a
# blocking run never survives to a second HTTP call). The route is a thin
# wrapper over RunManager.cancel, which these cover directly.
def test_cancel_marks_run_cancelled_and_emits_terminal_event():
    from interaction_finder.web.run_manager import RunManager

    async def scenario():
        mgr = RunManager()
        record = RunRecord(
            id="r1",
            topic="x",
            spec=RunSpec(checkpoint_or_topic="x", entity_kinds=["gene"]),
            table=WebStatusTable(),
            checkpoint_path=None,
        )

        # Drive the real _drive against a hanging ensure_extraction.
        async def hang(*_a, **_k):
            await asyncio.Event().wait()

        import interaction_finder.upgrade as upgrade

        upgrade.ensure_extraction = hang
        mgr._runs[record.id] = record
        record.task = asyncio.ensure_future(mgr._drive(record, None, None, None))
        await asyncio.sleep(0)  # let _drive reach the await point
        assert mgr.cancel("r1") is True
        await record.task  # _drive swallows the CancelledError
        return record

    record = asyncio.run(scenario())
    assert record.status == "cancelled"
    # table.cancel() records the terminal cancelled result.
    assert record.table._result == "cancelled"


def test_cancel_returns_false_for_unknown_or_finished_run():
    from interaction_finder.web.run_manager import RunManager

    mgr = RunManager()
    assert mgr.cancel("missing") is False
    record = RunRecord(
        id="done1",
        topic="x",
        spec=RunSpec(checkpoint_or_topic="x", entity_kinds=["gene"]),
        table=WebStatusTable(),
        checkpoint_path=None,
        status="success",
    )
    mgr._runs[record.id] = record
    assert mgr.cancel("done1") is False  # not running


def test_cancel_route_404_for_unknown_run(client):
    assert client.post("/runs/nope/cancel").status_code == 404


def test_load_checkpoint_streams_backfilled_events(tmp_path):
    """POST /load registers a finished run whose SSE stream replays the file."""
    from interaction_finder.checkpoint import (
        KeywordsStageData,
        PipelineCheckpoint,
    )
    from interaction_finder.resources import ResourcePool

    checkpoint = PipelineCheckpoint(
        topic="loaded topic",
        resources=ResourcePool(),
        keywords=KeywordsStageData(
            terms=["BMPR2"],
            scores=[0.9],
            total_documents_processed=1,
            rounds_completed=1,
            coverage_assessment="Sufficient coverage of the relevant research literature obtained",
            resource_urls=["http://example.com/1"],
        ),
    )
    path = tmp_path / "saved.json"
    path.write_text(checkpoint.model_dump_json())

    # Real RunManager (no stub) -- back-fill is exercised end to end.
    real_client = TestClient(app_module.create_app())
    resp = real_client.post("/load", json={"path": str(path)})
    assert resp.status_code == 200
    assert resp.json()["topic"] == "loaded topic"
    run_id = resp.json()["run_id"]

    body = real_client.get(f"/runs/{run_id}/events?since=0").text
    scopes = [m["scope"] for m in parse_sse(body) if m["type"] == "event"]
    assert "keywords.scored" in scopes
    # The load status drives the UI's continue/view decision.
    status = resp.json()
    assert status["stage"] == "keywords"
    assert status["complete"] is False
    assert status["resumable"] is True
    assert status["needs_entity_kinds"] is True


def test_load_does_not_write_bak(tmp_path):
    """A read-only load must not create a .bak (unlike the CLI loader)."""
    from interaction_finder.checkpoint import KeywordsStageData, PipelineCheckpoint
    from interaction_finder.resources import ResourcePool

    checkpoint = PipelineCheckpoint(
        topic="t",
        resources=ResourcePool(),
        keywords=KeywordsStageData(
            terms=["x"],
            scores=[0.5],
            total_documents_processed=1,
            rounds_completed=1,
            coverage_assessment="Sufficient coverage of the relevant research literature obtained",
            resource_urls=["http://example.com/1"],
        ),
    )
    path = tmp_path / "saved.json"
    path.write_text(checkpoint.model_dump_json())
    client = TestClient(app_module.create_app())
    assert client.post("/load", json={"path": str(path)}).status_code == 200
    assert not (tmp_path / "saved.json.bak").exists()


def test_load_missing_file_is_400():
    real_client = TestClient(app_module.create_app())
    resp = real_client.post("/load", json={"path": "/nonexistent/x.json"})
    assert resp.status_code == 400


def test_browse_lists_dirs_and_json(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "b.txt").write_text("ignored")
    (tmp_path / ".hidden.json").write_text("{}")
    client = TestClient(app_module.create_app())
    data = client.get(f"/browse?dir={tmp_path}").json()
    assert data["dir"] == str(tmp_path)
    assert data["parent"] == str(tmp_path.parent)
    names = [e["name"] for e in data["entries"]]
    # Dirs first, then .json files; non-json and dotfiles excluded.
    assert names == ["sub", "a.json"]
    assert data["entries"][0]["is_dir"] is True


def test_browse_bad_dir_is_400():
    client = TestClient(app_module.create_app())
    assert client.get("/browse?dir=/no/such/dir").status_code == 400


def test_report_unavailable_before_completion(client):
    # Stub never sets record.checkpoint, so the report is reported unavailable.
    run_id = client.post(
        "/runs", json={"checkpoint_or_topic": "cancer", "entity_kinds": ["gene"]}
    ).json()["run_id"]
    assert client.get(f"/report/{run_id}").status_code == 404


def _empty_extraction_checkpoint():
    """A checkpoint whose extraction completed but found no associations."""
    from interaction_finder.checkpoint import (
        ExtractionStageData,
        PipelineCheckpoint,
    )
    from interaction_finder.extraction.models import ExtractionMetadata
    from interaction_finder.resources import ResourcePool

    return PipelineCheckpoint(
        topic="empty topic",
        resources=ResourcePool(),
        extraction=ExtractionStageData(
            target_entity_types=["gene"],
            permitted_pairs={"gene": ["gene"]},
            judgments=[],
            metadata=ExtractionMetadata(
                topic="empty topic",
                resource_count=2,
                total_entities_found=0,
                entities_after_validation=0,
                entities_merged=0,
                merge_cache_hits=0,
                merge_cache_misses=0,
                proximal_sets_found=0,
                total_pairs_found=0,
                pairs_accepted=0,
                pairs_rejected=0,
                quotes_validated=0,
                quotes_failed=0,
            ),
        ),
    )


def test_has_report_false_for_complete_but_empty_extraction():
    from interaction_finder.web.run_manager import has_report

    assert has_report(_empty_extraction_checkpoint()) is False


def test_load_reports_no_report_for_empty_extraction(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(_empty_extraction_checkpoint().model_dump_json())
    client = TestClient(app_module.create_app())
    status = client.post("/load", json={"path": str(path)}).json()
    assert status["complete"] is True
    assert status["has_report"] is False


def test_active_runs_filters_by_status_and_orders_by_start():
    from interaction_finder.web.run_manager import RunManager

    manager = RunManager()
    for run_id, status, started in [
        ("r1", "running", 100.0),
        ("r2", "success", 50.0),
        ("r3", "running", 75.0),
    ]:
        rec = RunRecord(
            id=run_id,
            topic=run_id,
            spec=RunSpec(checkpoint_or_topic="t", entity_kinds=["gene"]),
            table=WebStatusTable(),
            checkpoint_path=None,
            status=status,
            started_at=started,
        )
        manager._runs[run_id] = rec
    # Only running runs, oldest-started first.
    assert [r.id for r in manager.active_runs()] == ["r3", "r1"]


def test_active_runs_route_shape():
    from interaction_finder.web.run_manager import RunManager

    app = app_module.create_app()
    # Reach the manager the routes close over via a registered running record.
    client = TestClient(app)
    assert client.get("/runs/active").json() == {"runs": []}


def test_report_route_422_not_500_for_empty_extraction(tmp_path):
    # A completed-but-empty run must not crash report generation with a 500.
    path = tmp_path / "empty.json"
    path.write_text(_empty_extraction_checkpoint().model_dump_json())
    client = TestClient(app_module.create_app())
    run_id = client.post("/load", json={"path": str(path)}).json()["run_id"]
    assert client.get(f"/report/{run_id}").status_code == 422
    assert client.get(f"/report/{run_id}/events").status_code == 422
