"""Tests for WebStatusTable (the SSE-facing progress table)."""

from __future__ import annotations

import asyncio

import pytest

from interaction_finder.progress import Counter, StatusTable
from interaction_finder.web.table import WebStatusTable


def drain(queue: asyncio.Queue) -> list[dict]:
    out = []
    while not queue.empty():
        out.append(queue.get_nowait())
    return out


class TestClearAll:
    """clear_all drops counters, keeps the event log, and marks the boundary."""

    def test_base_clear_all_keeps_event_log(self):
        table = StatusTable(Counter("A"))
        table.emit("x.happened", "did x")
        table.clear_all()
        assert table.counters == []
        # Append-only: the event survives the boundary; index is its sequence.
        assert [e.scope for e in table.events] == ["x.happened"]

    def test_web_clear_all_emits_meta_cleared(self):
        table = WebStatusTable()
        table.emit("search.results", "3 results")
        table.clear_all()
        assert [e.scope for e in table.events] == ["search.results", "meta.cleared"]


class TestPublishing:
    """Events and snapshots fan out to every subscriber."""

    def test_event_published_to_subscribers(self):
        table = WebStatusTable()
        q1, q2 = table.subscribe(), table.subscribe()
        table.emit("search.queries", "5 queries", round=1, queries=["a", "b"])
        for q in (q1, q2):
            msgs = drain(q)
            assert len(msgs) == 1
            assert msgs[0]["type"] == "event"
            assert msgs[0]["scope"] == "search.queries"
            assert msgs[0]["data"] == {"round": 1, "queries": ["a", "b"]}
            assert msgs[0]["index"] == 0

    def test_counter_change_publishes_snapshot(self):
        table = WebStatusTable()
        q = table.subscribe()
        table.add_counters(Counter("Processed", track_in_progress=True))
        table["Processed"].total = 4
        msgs = drain(q)
        assert msgs[-1]["type"] == "counters"
        snap = {c["name"]: c for c in msgs[-1]["counters"]}
        assert snap["Processed"]["total"] == 4

    def test_unsubscribe_stops_delivery(self):
        table = WebStatusTable()
        q = table.subscribe()
        table.unsubscribe(q)
        table.emit("x", "y")
        assert drain(q) == []


class TestCatchUp:
    """A late subscriber rebuilds state from the event log, not the queue."""

    def test_events_since_returns_wire_messages(self):
        table = WebStatusTable()
        table.emit("a", "1")
        table.emit("b", "2")
        table.emit("c", "3")
        # A client that has seen index 0 resumes from 1.
        catch_up = table.events_since(1)
        assert [m["scope"] for m in catch_up] == ["b", "c"]
        assert [m["index"] for m in catch_up] == [1, 2]

    def test_late_subscriber_misses_live_but_catches_up_from_log(self):
        table = WebStatusTable()
        table.emit("early", "before subscribing")
        q = table.subscribe()  # subscribed after the first event
        table.emit("late", "after subscribing")
        # Queue only has the post-subscription event...
        live = [m for m in drain(q) if m["type"] == "event"]
        assert [m["scope"] for m in live] == ["late"]
        # ...but the full history is recoverable from the log.
        assert [m["scope"] for m in table.events_since(0)] == ["early", "late"]


class TestLifecycle:
    def test_succeed_publishes_done(self):
        table = WebStatusTable()
        q = table.subscribe()
        table.start()
        table.succeed()
        done = [m for m in drain(q) if m["type"] == "done"]
        assert done and done[0]["result"] == "success"

    def test_fail_publishes_done(self):
        table = WebStatusTable()
        q = table.subscribe()
        table.start()
        table.fail()
        done = [m for m in drain(q) if m["type"] == "done"]
        assert done and done[0]["result"] == "failure"

    def test_full_queue_drops_oldest_not_newest(self):
        # Overflow sheds the oldest queued message; the log stays complete.
        from interaction_finder.web import table as table_mod

        table = WebStatusTable()
        q = table.subscribe()
        for i in range(table_mod._QUEUE_MAXSIZE + 5):
            table.emit("tick", str(i))
        msgs = [m for m in drain(q) if m["type"] == "event"]
        # Newest survived and the queue never exceeded its bound.
        assert msgs[-1]["description"] == str(table_mod._QUEUE_MAXSIZE + 4)
        assert len(msgs) <= table_mod._QUEUE_MAXSIZE
        # The event log keeps everything regardless of queue shedding.
        assert len(table.events) == table_mod._QUEUE_MAXSIZE + 5
