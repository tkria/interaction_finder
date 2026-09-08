"""Resolve whether a retrieved article may be redistributed, via OpenAlex.

Verdicts are coarse because only one distinction changes what a shareable file
may contain. Note that OpenAlex's "bronze" status means free to read at the
publisher's discretion under no open licence, which is not permission to
redistribute.

    REDISTRIBUTABLE  an open licence permits sharing the text
    SHARE_ALIKE      permits sharing under conditions (NC/ND/SA)
    RESTRICTED       no licence permits sharing
    UNKNOWN          could not be resolved

Public API:
    Verdict, ArticleLicence, resolve_licences, summarise
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Mapping

import httpx

__all__ = [
    "ArticleLicence",
    "Verdict",
    "resolve_licences",
]

OPENALEX_WORKS = "https://api.openalex.org/works"
# Works requested per call. OpenAlex accepts long OR-filters; 100 keeps the URL
# near 3 kB and turns a 10,000-article corpus into ~100 requests.
_BATCH_SIZE = 100
# Pause when the remaining request allowance falls to this, so a long run leaves
# headroom rather than failing at the limit.
_RATE_LIMIT_FLOOR = 20
_MAX_RATE_WAIT_SECONDS = 300
_PERMISSIVE = frozenset({"cc-by", "cc-by-sa", "cc0", "public-domain", "pd"})
_CONDITIONAL = frozenset(
    {
        "cc-by-nc",
        "cc-by-nd",
        "cc-by-nc-nd",
        "cc-by-nc-sa",
    }
)
_PMID_IN_URL = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)")


class Verdict(str, Enum):
    """Whether an article's text may be included in a file shared onward."""

    REDISTRIBUTABLE = "redistributable"
    SHARE_ALIKE = "share-alike"
    RESTRICTED = "restricted"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ArticleLicence:
    """One article's redistribution verdict and the evidence for it.

    verdict: Verdict -- what the licence permits.
    licence: str | None -- the licence recorded, if any.
    oa_status: str | None -- OpenAlex open-access status (gold/green/bronze/...).
    reason: str -- short human-readable justification, for reporting.
    """

    verdict: Verdict
    licence: str | None
    oa_status: str | None
    reason: str


def resolve_licences(
    articles: Mapping[str, Mapping[str, object]],
    *,
    email: str | None = None,
    cache_path: Path | None = None,
    client: httpx.Client | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, ArticleLicence]:
    """Resolve a redistribution verdict for each article.

    articles: mapping of article key (its URL) to a record carrying at least a
        ``doi``, or a PubMed ``url`` from which a PMID can be read.
    email: optional contact address. OpenAlex serves anonymous callers; supplying
        one opts into its faster "polite pool".
    cache_path: optional JSON file to read and write resolved verdicts, so
        repeated runs over the same corpus need no further requests.
    client: optional httpx client, for tests or connection reuse.
    progress: optional callback receiving (requested, returned) per batch, for
        reporting progress on a long run.

    Returns: dict mapping the same keys to ArticleLicence. Articles that cannot
        be identified or resolved get Verdict.UNKNOWN rather than being omitted,
        so callers always get a verdict for every input.
    """
    cache = _load_cache(cache_path)
    owned_client = client is None
    agent = "interaction-finder licence check"
    client = client or httpx.Client(
        timeout=30,
        headers={"User-Agent": f"{agent} ({email})" if email else agent},
    )
    resolved: dict[str, ArticleLicence] = {}
    identifiers: dict[str, str] = {}
    try:
        for key, record in articles.items():
            identifier = _identifier(key, record)
            if identifier is None:
                resolved[key] = ArticleLicence(
                    Verdict.UNKNOWN,
                    None,
                    None,
                    "no DOI or PMID recorded, so the licence cannot be looked up",
                )
            elif identifier in cache and not cache[identifier].get("error"):
                resolved[key] = _from_payload(cache[identifier])
            else:
                identifiers[key] = identifier
        outstanding = sorted(set(identifiers.values()))
        for start in range(0, len(outstanding), _BATCH_SIZE):
            batch = outstanding[start : start + _BATCH_SIZE]
            cache.update(_fetch_batch(client, batch, email, progress))
            if cache_path is not None:
                _save_cache(cache_path, cache)
        # Retry anything a batch did not cover as a single-entity GET. Those are
        # free of credits (a list request is billed, a single fetch is not), so
        # the few stragglers are worth the extra requests.
        missed = [
            identifier
            for identifier in outstanding
            if cache.get(identifier, {}).get("error") == "not-returned"
        ]
        for identifier in missed:
            cache[identifier] = _fetch_one(client, identifier, email)
        if missed and cache_path is not None:
            _save_cache(cache_path, cache)
        for key, identifier in identifiers.items():
            resolved[key] = _from_payload(
                cache.get(identifier, {"error": "not-returned"})
            )
    finally:
        if owned_client:
            client.close()
        _save_cache(cache_path, cache)
    return resolved


def _identifier(key: str, record: Mapping[str, object]) -> str | None:
    """OpenAlex-resolvable identifier for an article: DOI, else PMID."""
    doi = record.get("doi")
    if isinstance(doi, str) and doi.strip():
        return f"doi:{doi.strip()}"
    for candidate in (record.get("url"), key):
        if isinstance(candidate, str):
            match = _PMID_IN_URL.search(candidate)
            if match:
                return f"pmid:{match.group(1)}"
    return None


def _classify_fields(work: Mapping[str, object]) -> dict:
    """Reduce an OpenAlex work to the fields a verdict is derived from."""
    locations = work.get("locations") or []
    return {
        "oa_status": (work.get("open_access") or {}).get("oa_status"),
        "licences": sorted(
            {
                location.get("license")
                for location in locations
                if location.get("license")
            }
        ),
    }


def _respect_rate_limit(response: httpx.Response) -> None:
    """Sleep if the server says its request allowance is nearly spent.

    OpenAlex reports ``X-RateLimit-Remaining`` and ``X-RateLimit-Reset`` (seconds
    until the allowance refills); a 429 carries ``Retry-After``. Waiting on the
    server's own numbers is what keeps a corpus-sized run inside them.
    """
    headers = response.headers
    wait = 0.0
    if response.status_code == 429:
        wait = _header_number(headers, "retry-after", default=60.0)
    else:
        remaining = _header_number(headers, "x-ratelimit-remaining")
        if remaining is not None and remaining <= _RATE_LIMIT_FLOOR:
            wait = _header_number(headers, "x-ratelimit-reset", default=60.0) or 60.0
    if wait and wait <= _MAX_RATE_WAIT_SECONDS:
        time.sleep(wait)


def _header_number(
    headers: Mapping[str, str], name: str, default: float | None = None
) -> float | None:
    """Read a numeric response header, falling back to `default` if unusable."""
    try:
        return float(headers[name])
    except (KeyError, TypeError, ValueError):
        return default


def _fetch_batch(
    client: httpx.Client,
    identifiers: list[str],
    email: str | None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, dict]:
    """Fetch many works in one request, keyed by the identifier asked for.

    identifiers: OpenAlex identifiers of one kind or mixed ("doi:...", "pmid:...").
    Returns: identifier -> classification payload, with an ``error`` entry for any
        identifier the response did not cover.
    """
    payloads: dict[str, dict] = {}
    for kind, group in _group_by_kind(identifiers).items():
        values = "|".join(group)
        try:
            response = client.get(
                OPENALEX_WORKS,
                params={
                    "filter": f"{kind}:{values}",
                    "select": "doi,ids,open_access,locations",
                    "per-page": len(group),
                    **({"mailto": email} if email else {}),
                },
            )
            _respect_rate_limit(response)
            if response.status_code == 429:
                response = client.get(
                    OPENALEX_WORKS,
                    params={
                        "filter": f"{kind}:{values}",
                        "select": "doi,ids,open_access,locations",
                        "per-page": len(group),
                        **({"mailto": email} if email else {}),
                    },
                )
            if response.status_code != 200:
                error = {"error": f"http-{response.status_code}"}
                payloads.update({f"{kind}:{value}": error for value in group})
                continue
            results = response.json().get("results") or []
        except (httpx.HTTPError, json.JSONDecodeError) as error:
            failure = {"error": type(error).__name__}
            payloads.update({f"{kind}:{value}": failure for value in group})
            continue
        # Match results back by whichever identifier we asked on: OpenAlex
        # normalises DOIs to URLs and nests PMIDs under `ids`.
        wanted = {value.lower(): f"{kind}:{value}" for value in group}
        for work in results:
            for value in _identifying_values(work, kind):
                key = wanted.get(value.lower())
                if key:
                    payloads[key] = _classify_fields(work)
        for value in group:
            payloads.setdefault(f"{kind}:{value}", {"error": "not-returned"})
        if progress:
            progress(len(group), len(results))
    return payloads


def _fetch_one(client: httpx.Client, identifier: str, email: str | None) -> dict:
    """Fetch a single work by identifier.

    Used for stragglers a batch did not return. Single-entity requests are not
    billed against the credit allowance, unlike list requests.
    """
    try:
        response = client.get(
            f"{OPENALEX_WORKS}/{identifier}",
            params={"mailto": email} if email else None,
        )
        _respect_rate_limit(response)
        if response.status_code != 200:
            return {"error": f"http-{response.status_code}"}
        return _classify_fields(response.json())
    except (httpx.HTTPError, json.JSONDecodeError) as error:
        return {"error": type(error).__name__}


def _group_by_kind(identifiers: list[str]) -> dict[str, list[str]]:
    """Split "kind:value" identifiers into {kind: [value, ...]}."""
    grouped: dict[str, list[str]] = {}
    for identifier in identifiers:
        kind, _, value = identifier.partition(":")
        if value:
            grouped.setdefault(kind, []).append(value)
    return grouped


def _identifying_values(work: Mapping[str, object], kind: str) -> list[str]:
    """Values of `kind` by which a returned work can be matched to a request."""
    if kind == "doi":
        doi = work.get("doi") or ""
        bare = str(doi).replace("https://doi.org/", "")
        return [value for value in (bare, str(doi)) if value]
    identifier = (work.get("ids") or {}).get(kind) or ""
    bare = str(identifier).rsplit("/", 1)[-1]
    return [value for value in (bare, str(identifier)) if value]


def _from_payload(payload: Mapping[str, object]) -> ArticleLicence:
    """Classify a cached OpenAlex payload into a verdict."""
    if payload.get("error"):
        return ArticleLicence(
            Verdict.UNKNOWN, None, None, f"licence lookup failed ({payload['error']})"
        )
    oa_status = payload.get("oa_status")
    licences = [str(licence) for licence in (payload.get("licences") or [])]
    permissive = [licence for licence in licences if licence in _PERMISSIVE]
    if permissive:
        return ArticleLicence(
            Verdict.REDISTRIBUTABLE,
            permissive[0],
            oa_status,
            f"{permissive[0]} permits redistribution with attribution",
        )
    conditional = [licence for licence in licences if licence in _CONDITIONAL]
    if conditional:
        return ArticleLicence(
            Verdict.SHARE_ALIKE,
            conditional[0],
            oa_status,
            f"{conditional[0]} permits redistribution only under its conditions",
        )
    if licences:
        return ArticleLicence(
            Verdict.RESTRICTED,
            licences[0],
            oa_status,
            f"{licences[0]} is not a recognised redistribution grant",
        )
    if oa_status == "bronze":
        return ArticleLicence(
            Verdict.RESTRICTED,
            None,
            oa_status,
            "free to read on the publisher's site but under no open licence",
        )
    if oa_status == "green":
        return ArticleLicence(
            Verdict.RESTRICTED,
            None,
            oa_status,
            "only a repository manuscript copy is free; no redistribution grant",
        )
    if oa_status == "closed":
        return ArticleLicence(Verdict.RESTRICTED, None, oa_status, "not open access")
    return ArticleLicence(
        Verdict.RESTRICTED,
        None,
        oa_status,
        f"no licence recorded (open-access status: {oa_status or 'unknown'})",
    )


def _load_cache(path: Path | None) -> dict[str, dict]:
    """Read the verdict cache, treating an unreadable file as empty."""
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(path: Path | None, cache: Mapping[str, dict]) -> None:
    """Write the verdict cache, if one was requested.

    Failed lookups are not persisted: a transient error would otherwise pin an
    article to UNKNOWN --- and so to redaction --- on every future run.
    """
    if path is None:
        return
    cache = {k: v for k, v in cache.items() if not v.get("error")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=1, sort_keys=True))


def summarise(verdicts: Iterable[ArticleLicence]) -> dict[str, int]:
    """Count verdicts by kind, for reporting to a user."""
    counts = {verdict.value: 0 for verdict in Verdict}
    for licence in verdicts:
        counts[licence.verdict.value] += 1
    return counts
