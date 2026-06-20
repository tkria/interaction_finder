"""Reconstruct a progress event stream from a saved checkpoint.

Replays a ``PipelineCheckpoint`` as the same ``Event`` sequence a live run would
have emitted, so a loaded file renders through the identical UI path as a run in
progress. Only what the checkpoint faithfully preserves is reconstructed: the
keyword scores, the per-query result selections, and every judged pair. Stage
boundaries are emitted as ``meta.cleared`` events, mirroring ``clear_all``.

Per-round search structure (``search.queries`` / ``search.results``) is not
reconstructed -- the checkpoint stores cumulative queries and final results, not
per-round batches -- so back-fill carries the durable facts (selections), which
is what the search accordion shows.
"""

from interaction_finder.checkpoint import PipelineCheckpoint
from interaction_finder.progress import Event


def checkpoint_to_events(checkpoint: PipelineCheckpoint) -> list[Event]:
    """Synthesize the event log implied by a checkpoint's completed stages.

    Returns events in pipeline order (keywords -> search -> extraction), with a
    ``meta.cleared`` boundary before each stage that ran, so the result is
    indistinguishable from a live stream to the frontend renderer.
    """
    events: list[Event] = []
    if checkpoint.keywords is not None:
        events.append(_boundary())
        events.append(_keywords_event(checkpoint.keywords))
    if checkpoint.search is not None:
        events.append(_boundary())
        events.extend(_search_events(checkpoint.search))
    if checkpoint.extraction is not None:
        events.append(_boundary())
        events.extend(_extraction_events(checkpoint.extraction))
    return events


def _boundary() -> Event:
    return Event("meta.cleared", "")


def _keywords_event(keywords) -> Event:
    terms = [
        {"term": term, "score": score}
        for term, score in zip(keywords.terms, keywords.scores)
    ]
    return Event(
        "keywords.scored",
        f"Identified {len(terms)} bridging terms",
        {"terms": terms},
    )


def _search_events(search) -> list[Event]:
    """One search.selected per query, resolving selected URLs to result records."""
    by_url = {r.url: r for r in search.results}
    events: list[Event] = []
    for query, urls in search.query_results.items():
        picked = [
            {
                "title": r.title,
                "url": r.url,
                "snippet": r.snippet,
                "relevance": r.relevance,
            }
            for url in urls
            if (r := by_url.get(url)) is not None
        ]
        events.append(
            Event(
                "search.selected",
                f"Selected {len(picked)} results for {query!r}",
                {"query": query, "picked": picked, "rejected": []},
            )
        )
    return events


def _extraction_events(extraction) -> list[Event]:
    """One extraction.pair_judged per judgment, matching the live emit shape."""
    events: list[Event] = []
    for judgment in extraction.judgments:
        verdict = "accepted" if judgment.accepted else "rejected"
        events.append(
            Event(
                "extraction.pair_judged",
                f"{judgment.entity1.name} – {judgment.entity2.name}: {verdict}",
                {
                    "entity1": judgment.entity1.name,
                    "entity2": judgment.entity2.name,
                    "relationship": judgment.relationship,
                    "accepted": judgment.accepted,
                    "evidence": judgment.evidence.overall,
                },
            )
        )
    return events
