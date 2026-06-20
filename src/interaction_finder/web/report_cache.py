"""Cross-platform cache for generated HTML reports.

A report is fully determined by the topic, the entity kinds, and the final
associations, so the cache is keyed by a hash of those -- not by file path or
mtime, which would stale-hit when a checkpoint is rewritten in place (e.g. on
resume) and miss spuriously when an identical result is saved elsewhere.

Cached files live under ``platformdirs.user_cache_dir`` so they never clutter
the user's data directories and survive across sessions.
"""

import hashlib
import json
from pathlib import Path

from platformdirs import user_cache_dir

from interaction_finder.checkpoint import PipelineCheckpoint

_CACHE_SUBDIR = "reports"


def report_cache_key(checkpoint: PipelineCheckpoint) -> str:
    """Stable hash of the content that determines a report's HTML.

    Combines the topic, the target entity kinds, and a per-pair signature of
    the final associations (entities, relationship, accept/reject). Two
    checkpoints with the same answer set hash identically regardless of where
    they are saved or how they were produced.
    """
    extraction = checkpoint.extraction
    pairs = []
    if extraction is not None:
        for j in extraction.judgments:
            pairs.append(
                [j.entity1.name, j.entity2.name, j.relationship, bool(j.accepted)]
            )
        pairs.sort()
    signature = {
        "topic": checkpoint.topic,
        "kinds": sorted(extraction.target_entity_types) if extraction else [],
        "pairs": pairs,
    }
    payload = json.dumps(signature, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def report_path_for_key(key: str) -> Path:
    """Absolute cache path for a report cache key (file may not exist)."""
    cache_dir = Path(user_cache_dir("interaction-finder")) / _CACHE_SUBDIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{key}.html"


def report_cache_path(checkpoint: PipelineCheckpoint) -> Path:
    """Absolute path the report for this checkpoint caches to (may not exist)."""
    return report_path_for_key(report_cache_key(checkpoint))
