#!/usr/bin/env -S uv run --with httpx python
"""Produce a shareable copy of the deposited supplementary data.

The deposit embeds the full text of every article the pipeline retrieved. Text
is ours to redistribute only where its licence says so, so this applies the same
policy the tool's ``interaction-finder redact`` command applies to a single
checkpoint, across every file in the deposit:

- Articles under an open licence (CC-BY, CC-BY-SA, CC0, public domain) or a
  conditional one (CC-BY-NC, -ND, -SA) keep their text, with the licence
  recorded on the resource so attribution and any non-commercial or
  no-derivatives terms travel with it.
- Records holding only an abstract, rather than an article body, are kept as
  they are whatever the article's licence: abstracts are distributed openly by
  publishers and indexing services for exactly this kind of reuse.
- Every other article --- closed access, free-to-read with no licence granted
  ("bronze"), repository manuscripts, and anything whose licence cannot be
  resolved --- is reduced to a *quote skeleton*: the passages the results quote,
  each under its section heading, with omitted stretches marked by size.
  Coverage is capped so a skeleton cannot substitute for the article.

Provenance is unaffected either way: URLs, DOIs, titles, dates, chunk offsets,
per-quote spans and match metadata are all retained, plus a SHA-256 and length
of any text removed, so a re-fetched source stays checkable against the record.

The HTML reports are regenerated from the redacted checkpoints rather than
copied, so they display exactly the text the checkpoints hold --- quote
skeletons where the article text was withheld, full text where its licence
permits. Regenerating uses ``interaction-finder report``, which makes no API
calls.

Usage:
    redact_deposit.py SOURCE_DIR DEST_DIR [--everything] [--dry-run] [--licence-cache FILE]

SOURCE_DIR is never modified. Licence verdicts are cached (by default in
DEST_DIR's parent), so re-runs need no further OpenAlex requests; the archive
keeps its cache at data/licence-cache.json. Set OPENALEX_EMAIL to use
OpenAlex's faster polite pool.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from interaction_finder.licences import Verdict, resolve_licences  # noqa: E402
from interaction_finder.redact_checkpoint import (  # noqa: E402
    RedactionPolicy,
    redact_checkpoint_data,
)

# Files excluded from the shareable deposit entirely (globs under SOURCE_DIR).
# HTML reports are NOT excluded: a report built from a redacted checkpoint shows
# the skeletons, so it carries the same text the checkpoint does. They must be
# regenerated from the redacted checkpoints rather than copied, since the
# originals in SOURCE_DIR render the unredacted text.
EXCLUDE_PATTERNS = (
    "**/*.html",
    "**/cache/**/*",
)


def excluded_paths(source: Path) -> set[Path]:
    """Absolute paths under `source` matched by any EXCLUDE_PATTERNS entry."""
    excluded: set[Path] = set()
    for pattern in EXCLUDE_PATTERNS:
        excluded.update(p.resolve() for p in source.glob(pattern) if p.is_file())
    return excluded


def article_records(data: object) -> dict[str, dict]:
    """Every article record in a parsed file, keyed by URL.

    Covers both shapes the deposit uses: ``resources[]`` lists in checkpoints,
    and URL-keyed dicts in the contamination pools.
    """
    records: dict[str, dict] = {}

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "resources" and isinstance(value, list):
                    for entry in value:
                        if isinstance(entry, dict) and entry.get("url"):
                            records[entry["url"]] = entry
                elif key == "resources" and isinstance(value, dict):
                    for url, entry in value.items():
                        if isinstance(entry, dict):
                            records[url] = entry
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(data)
    # The pools are a bare URL-keyed mapping at the top level.
    if isinstance(data, dict) and not records:
        for url, entry in data.items():
            if isinstance(entry, dict) and "text" in entry and url.startswith("http"):
                records[url] = entry
    return records


def as_pool_checkpoint(data: dict) -> dict | None:
    """Wrap a bare URL-keyed document pool so it can be redacted uniformly.

    Returns None when `data` is not a pool. The wrapper shares the same dict
    objects, so redacting it mutates the original records in place.
    """
    entries = [
        (url, entry)
        for url, entry in data.items()
        if isinstance(entry, dict) and "text" in entry and str(url).startswith("http")
    ]
    if not entries or len(entries) != len(data):
        return None
    for url, entry in entries:
        entry.setdefault("url", url)
    return {"resources": [entry for _, entry in entries]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("dest", type=Path)
    parser.add_argument(
        "--everything",
        action="store_true",
        help="redact all article text, ignoring licences",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--licence-cache", type=Path,
        help="OpenAlex verdict cache (default: licence-cache.json beside DEST_DIR)",
    )
    args = parser.parse_args()

    source, dest = args.source.resolve(), args.dest.resolve()
    if not source.is_dir():
        parser.error(f"source is not a directory: {source}")
    if dest.exists() and not args.dry_run:
        parser.error(f"destination exists, refusing to overwrite: {dest}")
    if source == dest:
        parser.error("source and destination must differ")

    skip = excluded_paths(source)
    files = sorted(p for p in source.rglob("*") if p.is_file())
    json_files = [p for p in files if p.suffix == ".json" and p.resolve() not in skip]

    # One licence lookup per distinct article across the whole deposit.
    print("Collecting articles...", flush=True)
    parsed: dict[Path, object] = {}
    articles: dict[str, dict] = {}
    for path in json_files:
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        records = article_records(data)
        if not records:
            continue
        parsed[path] = data
        articles.update(records)
    print(
        f"  {len(parsed):,} files carry article records; "
        f"{len(articles):,} distinct articles",
        flush=True,
    )

    print("Resolving licences via OpenAlex (cached)...", flush=True)
    licences = resolve_licences(
        articles,
        email=os.environ.get("OPENALEX_EMAIL"),
        cache_path=args.licence_cache or dest.parent / "licence-cache.json",
    )
    verdicts = collections.Counter(l.verdict.value for l in licences.values())
    for verdict in Verdict:
        print(f"  {verdict.value:16} {verdicts[verdict.value]:6,}")

    policy = RedactionPolicy(
        keep_redistributable=not args.everything,
        keep_share_alike=not args.everything,
    )
    totals = collections.Counter()
    kept_by_licence: collections.Counter[str] = collections.Counter()
    for path, data in parsed.items():
        target = as_pool_checkpoint(data) if isinstance(data, dict) else None
        _, summary = redact_checkpoint_data(target or data, licences, policy)
        totals["kept"] += summary.kept
        totals["abstracts"] += summary.abstracts_kept
        totals["redacted"] += summary.redacted
        totals["chars_removed"] += summary.chars_removed
        kept_by_licence.update(summary.kept_by_licence)

    if not args.dry_run:
        for path in files:
            if path.resolve() in skip:
                continue
            relative = path.relative_to(source)
            output = dest / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            if path in parsed:
                output.write_text(json.dumps(parsed[path]))
            else:
                shutil.copy2(path, output)

    excluded = sum(1 for p in files if p.resolve() in skip)
    print(f"\n{'Would write' if args.dry_run else 'Wrote'} {dest}")
    print(f"  articles keeping their text  : {totals['kept']:,}")
    print(f"  abstracts kept as they are   : {totals['abstracts']:,}")
    print(f"  articles reduced to skeletons: {totals['redacted']:,}")
    print(f"  article text removed         : {totals['chars_removed'] / 1e6:,.0f} MB")
    print(f"  files excluded (HTML, cache) : {excluded:,}")
    if kept_by_licence:
        print(
            "  text kept under              : "
            + ", ".join(
                f"{count:,}× {name}" for name, count in sorted(kept_by_licence.items())
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
