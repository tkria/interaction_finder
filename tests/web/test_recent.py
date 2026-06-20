"""Tests for the recent-checkpoint registry."""

from __future__ import annotations

import interaction_finder.web.recent as recent


def _redirect_registry(tmp_path, monkeypatch):
    """Point the registry file at a temp dir for the duration of a test."""
    target = tmp_path / "recent.json"
    monkeypatch.setattr(recent, "_registry_path", lambda: target)
    return target


def test_empty_when_no_file(tmp_path, monkeypatch):
    _redirect_registry(tmp_path, monkeypatch)
    assert recent.load_recent() == []


def test_record_then_load_newest_first(tmp_path, monkeypatch):
    _redirect_registry(tmp_path, monkeypatch)
    recent.record_recent("/a.json", "Topic A", 5, "2026-01-01T00:00:00")
    recent.record_recent("/b.json", "Topic B", 0, "2026-01-02T00:00:00")
    entries = recent.load_recent()
    assert [e.topic for e in entries] == ["Topic B", "Topic A"]
    assert entries[1].pair_count == 5


def test_complete_defaults_true(tmp_path, monkeypatch):
    _redirect_registry(tmp_path, monkeypatch)
    recent.record_recent("/a.json", "A", 5, "t1")
    assert recent.load_recent()[0].complete is True


def test_partial_entry_round_trips(tmp_path, monkeypatch):
    _redirect_registry(tmp_path, monkeypatch)
    recent.record_recent("/a.json", "A", 2, "t1", complete=False)
    assert recent.load_recent()[0].complete is False


def test_legacy_entry_without_complete_loads_as_complete(tmp_path, monkeypatch):
    target = _redirect_registry(tmp_path, monkeypatch)
    import json

    target.write_text(
        json.dumps(
            [
                {
                    "id": "x",
                    "path": "/a.json",
                    "topic": "A",
                    "pair_count": 1,
                    "opened_at": "t1",
                }
            ]
        )
    )
    assert recent.load_recent()[0].complete is True


def test_record_same_path_dedupes_and_moves_front(tmp_path, monkeypatch):
    _redirect_registry(tmp_path, monkeypatch)
    recent.record_recent("/a.json", "Topic A", 5, "2026-01-01T00:00:00")
    recent.record_recent("/b.json", "Topic B", 1, "2026-01-02T00:00:00")
    # Re-opening A refreshes it to the front without duplicating.
    recent.record_recent("/a.json", "Topic A", 7, "2026-01-03T00:00:00")
    entries = recent.load_recent()
    assert [e.path for e in entries] == ["/a.json", "/b.json"]
    assert entries[0].pair_count == 7  # updated


def test_remove_one(tmp_path, monkeypatch):
    _redirect_registry(tmp_path, monkeypatch)
    recent.record_recent("/a.json", "A", 0, "t1")
    recent.record_recent("/b.json", "B", 0, "t2")
    b_id = recent.load_recent()[0].id
    recent.remove_recent(b_id)
    assert [e.topic for e in recent.load_recent()] == ["A"]


def test_clear_all(tmp_path, monkeypatch):
    _redirect_registry(tmp_path, monkeypatch)
    recent.record_recent("/a.json", "A", 0, "t1")
    recent.clear_recent()
    assert recent.load_recent() == []


def test_corrupt_file_is_treated_as_empty(tmp_path, monkeypatch):
    target = _redirect_registry(tmp_path, monkeypatch)
    target.write_text("not json{")
    assert recent.load_recent() == []
