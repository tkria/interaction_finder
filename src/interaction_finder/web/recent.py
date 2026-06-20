"""Cross-platform registry of recently opened checkpoints / generated reports.

A small JSON file under ``platformdirs.user_data_dir`` (not the cache dir --
this is user history that should outlive a cache purge) records the checkpoints
a user has loaded or produced a report for, so the UI can offer a "recent" list.

Entries are keyed by checkpoint path: re-opening the same file refreshes its
timestamp rather than adding a duplicate. The newest entry is listed first.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from platformdirs import user_data_dir

_REGISTRY_FILE = "recent.json"
_MAX_ENTRIES = 50


@dataclass
class RecentEntry:
    """One remembered checkpoint: enough to relist and re-open it."""

    id: str  # stable hash of the path
    path: str  # checkpoint file path (what /load re-opens)
    topic: str
    pair_count: int  # accepted+rejected judgments, 0 if pre-extraction
    opened_at: str  # ISO timestamp, supplied by the caller
    report_key: str = ""  # report cache key; lets us check for a cached report
    # without reloading the checkpoint
    complete: bool = True  # False for a partial/incomplete checkpoint (extraction
    # not finished); the UI flags these. Defaults True so legacy entries load.


def _registry_path() -> Path:
    data_dir = Path(user_data_dir("interaction-finder"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / _REGISTRY_FILE


def _entry_id(path: str) -> str:
    import hashlib

    return hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]


def load_recent() -> list[RecentEntry]:
    """Return remembered entries, newest first; empty when none/unreadable."""
    file = _registry_path()
    if not file.exists():
        return []
    try:
        raw = json.loads(file.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    return [RecentEntry(**e) for e in raw if isinstance(e, dict)]


def _write(entries: list[RecentEntry]) -> None:
    _registry_path().write_text(
        json.dumps([asdict(e) for e in entries], indent=2, ensure_ascii=False)
    )


def record_recent(
    path: str,
    topic: str,
    pair_count: int,
    opened_at: str,
    report_key: str = "",
    complete: bool = True,
) -> None:
    """Add or refresh the entry for ``path``, moving it to the front.

    Parameters:
        path: Checkpoint file path; the de-duplication key.
        topic: Research topic, for display.
        pair_count: Number of judged pairs, for display (0 if none yet).
        opened_at: ISO timestamp of this open (the caller stamps it, so the
            registry stays deterministic/testable).
        report_key: Report cache key, for the open-report affordance.
        complete: False when the checkpoint's extraction has not finished; the
            UI flags such entries as partial.
    """
    entries = [e for e in load_recent() if e.path != path]
    entries.insert(
        0,
        RecentEntry(
            id=_entry_id(path),
            path=path,
            topic=topic,
            pair_count=pair_count,
            opened_at=opened_at,
            report_key=report_key,
            complete=complete,
        ),
    )
    _write(entries[:_MAX_ENTRIES])


def find_recent(entry_id: str) -> "RecentEntry | None":
    """Return the entry with this id, or None."""
    for entry in load_recent():
        if entry.id == entry_id:
            return entry
    return None


def remove_recent(entry_id: str) -> None:
    """Drop the entry with this id, if present."""
    _write([e for e in load_recent() if e.id != entry_id])


def clear_recent() -> None:
    """Forget all remembered entries."""
    _write([])
