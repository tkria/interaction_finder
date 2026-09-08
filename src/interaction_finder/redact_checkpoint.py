"""Apply a per-article redaction policy to a whole checkpoint.

Text under a permitting licence is kept, abstract-only records are kept, and
every other article's text becomes a quote skeleton (see `redact`). Quote spans
are rewritten to index the skeleton, with the original offsets kept as
`spans_in_source` so a re-fetched source remains checkable.

Public API:
    RedactionPolicy, RedactionSummary, redact_checkpoint_data
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping

from interaction_finder.licences import ArticleLicence, Verdict
from interaction_finder.redact import (
    SkeletonLimits,
    SourceKind,
    build_skeleton,
    classify_source,
    render_skeleton,
)

__all__ = [
    "RedactionError",
    "RedactionPolicy",
    "RedactionSummary",
    "redact_checkpoint_data",
]


class RedactionError(ValueError):
    """A checkpoint could not be redacted safely.

    Raised rather than returning a partly-redacted file: a caller that cannot
    tell redaction failed would share text believing it had been withheld.
    """

_ASSESSMENT_CATEGORIES = ("positive", "negative", "neutral", "irrelevant")


@dataclass(frozen=True)
class RedactionPolicy:
    """Which articles keep their full text when a checkpoint is redacted.

    keep_redistributable: bool -- keep text under an open licence (CC-BY, CC0).
    keep_share_alike: bool -- keep text under conditional licences (NC/ND/SA).
        These do permit redistribution, so the default is to keep it; the
        licence travels with the text so a recipient can meet its terms.
    keep_abstracts: bool -- keep records holding only an abstract rather than an
        article body. Abstracts are distributed openly, so there is nothing to
        withhold; set False to skeletonise them regardless.
    limits: SkeletonLimits -- bounds on how much of a redacted article the
        skeleton may reproduce. Deliberately not exposed as a CLI flag: the
        coverage ceiling is what keeps a skeleton from substituting for the
        article, so raising it is not a user-facing choice.
    """

    keep_redistributable: bool = True
    keep_share_alike: bool = True
    keep_abstracts: bool = True
    limits: SkeletonLimits = field(default_factory=SkeletonLimits)

    def keeps(self, verdict: Verdict) -> bool:
        """Whether an article with this verdict keeps its full text."""
        if verdict is Verdict.REDISTRIBUTABLE:
            return self.keep_redistributable
        if verdict is Verdict.SHARE_ALIKE:
            return self.keep_share_alike
        return False


@dataclass
class RedactionSummary:
    """What a redaction pass did, for reporting back to the user.

    kept: int -- articles whose full text was retained under their licence.
    abstracts_kept: int -- records holding only an abstract, kept as they are.
    redacted: int -- articles reduced to a skeleton.
    already_redacted: int -- articles that arrived redacted and were left alone.
    chars_removed: int -- article characters no longer present.
    by_verdict: dict[str, int] -- article count per licence verdict.
    kept_by_licence: dict[str, int] -- article count per licence whose text was
        kept, so a sharer can see which attributions they owe.
    """

    kept: int = 0
    abstracts_kept: int = 0
    redacted: int = 0
    already_redacted: int = 0
    chars_removed: int = 0
    by_verdict: dict[str, int] = field(default_factory=dict)
    kept_by_licence: dict[str, int] = field(default_factory=dict)


def redact_checkpoint_data(
    data: dict[str, Any],
    licences: Mapping[str, ArticleLicence],
    policy: RedactionPolicy | None = None,
) -> tuple[dict[str, Any], RedactionSummary]:
    """Redact article text in a parsed checkpoint according to `licences`.

    data: parsed checkpoint JSON; mutated and returned.
    licences: article URL -> licence verdict, as `resolve_licences` returns.
    policy: which verdicts keep their text; defaults to keeping only openly
        licensed articles.

    Returns: (checkpoint, summary). Resources gain `text_redacted`,
        `text_sha256`, and `text_length` when reduced, and `text_licence` when
        kept on the strength of a licence.
    """
    policy = policy or RedactionPolicy()
    summary = RedactionSummary()
    spans_by_url = _spans_by_resource(data)
    resources = data.get("resources")
    if isinstance(resources, dict):
        # Legacy wrapped form: a URL-keyed mapping rather than a list. Iterating
        # it would walk keys and silently redact nothing.
        resources = list(resources.values())
    elif resources is None:
        resources = []
    elif not isinstance(resources, list):
        raise RedactionError(
            f"cannot redact: 'resources' is a {type(resources).__name__}, "
            "expected a list or a URL-keyed mapping"
        )
    for resource in resources:
        if not isinstance(resource, dict):
            raise RedactionError(
                f"cannot redact: found a {type(resource).__name__} among "
                "resources, so its text cannot be inspected"
            )
        url = resource.get("url") or ""
        licence = licences.get(url)
        verdict = licence.verdict if licence else Verdict.UNKNOWN
        summary.by_verdict[verdict.value] = summary.by_verdict.get(verdict.value, 0) + 1
        if licence and licence.licence:
            resource["licence"] = licence.licence
        text = resource.get("text")
        if text is None:
            continue
        if not isinstance(text, str):
            raise RedactionError(
                f"cannot redact {url or 'a resource'}: 'text' is a "
                f"{type(text).__name__}, expected a string"
            )
        if resource.get("text_redacted"):
            # Trust the marker only if the text is plausibly a skeleton: a record
            # flagged redacted while still holding its full body would otherwise
            # be passed straight through.
            if resource.get("text_length") and len(text) >= resource["text_length"]:
                raise RedactionError(
                    f"cannot redact {url or 'a resource'}: marked text_redacted "
                    "but its text is not shorter than the length it records"
                )
            summary.already_redacted += 1
            continue
        if policy.keeps(verdict):
            summary.kept += 1
            name = resource.get("licence") or "unstated"
            summary.kept_by_licence[name] = summary.kept_by_licence.get(name, 0) + 1
            continue
        if policy.keep_abstracts and classify_source(text) is SourceKind.ABSTRACT:
            summary.abstracts_kept += 1
            continue
        skeleton = build_skeleton(text, spans_by_url.get(url, []), policy.limits)
        replacement = render_skeleton(skeleton) if skeleton.snippets else ""
        resource["text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        resource["text_length"] = len(text)
        resource["text_redacted"] = True
        resource["text"] = replacement
        summary.redacted += 1
        summary.chars_removed += max(0, len(text) - len(replacement))
        _rewrite_spans(data, url, text, replacement)
    return data, summary


def _spans_by_resource(data: Mapping[str, Any]) -> dict[str, list[list[int]]]:
    """Collect every quote span in the checkpoint, keyed by resource URL."""
    spans: dict[str, list[list[int]]] = {}
    for quote, url in _iter_quotes(data):
        for span in quote.get("spans") or []:
            if isinstance(span, (list, tuple)) and len(span) >= 2:
                spans.setdefault(url, []).append([int(span[0]), int(span[1])])
    return spans


def _rewrite_spans(
    data: Mapping[str, Any], url: str, source: str, replacement: str
) -> None:
    """Point each quote's spans at its text within `replacement`.

    Original offsets are kept as `spans_in_source`, so a re-fetched article can
    still be searched at the recorded position.
    """
    consumed: dict[str, int] = {}
    for quote, quote_url in _iter_quotes(data):
        if quote_url != url:
            continue
        original = [list(span) for span in (quote.get("spans") or [])]
        if "spans_in_source" not in quote and original:
            quote["spans_in_source"] = original
        # Locate the quote by the source text it matched, not by `query_text`:
        # for a fuzzy match those differ, and it is the source wording that the
        # skeleton reproduces.
        rewritten = []
        for start, end in original:
            passage = source[start:end]
            if not passage:
                continue
            # Advance the search so repeated passages map to distinct
            # occurrences rather than all collapsing onto the first.
            position = replacement.find(passage, consumed.get(passage, 0))
            if position == -1:
                position = replacement.find(passage)
            if position != -1:
                rewritten.append([position, position + len(passage)])
                consumed[passage] = position + len(passage)
        quote["spans"] = rewritten
        # A quote whose passage the coverage limits left out has no offsets into
        # what ships. Say so, rather than leaving an empty list that reads as
        # "no provenance"; `spans_in_source` still locates it in the article.
        if original and not rewritten:
            quote["omitted_from_skeleton"] = True
        else:
            quote.pop("omitted_from_skeleton", None)


def _iter_quotes(data: Mapping[str, Any]):
    """Yield (quote, resource_url) for every quote in the extraction stage.

    Covers all three places a checkpoint carries quotes: per-assessment quotes
    under `spread` and under the legacy `assessments` key, and the entity-mention
    quotes under `entities`. A quote missed here keeps offsets into text that no
    longer ships, so its highlight would land on unrelated words.
    """
    extraction = data.get("extraction") or {}
    for judgment in extraction.get("judgments") or []:
        spread = judgment.get("spread") or {}
        groups = [spread.get(category) for category in _ASSESSMENT_CATEGORIES]
        groups.append(judgment.get("assessments"))
        for assessments in groups:
            for assessment in assessments or []:
                if not isinstance(assessment, dict):
                    continue
                fallback = ((assessment.get("resource_id") or {}).get("url")) or ""
                for quote in assessment.get("quotes") or []:
                    if isinstance(quote, dict):
                        yield quote, (quote.get("resource_url") or fallback)
    entities = extraction.get("entities")
    for entity in (entities or {}).values() if isinstance(entities, dict) else []:
        if not isinstance(entity, dict):
            continue
        for mention in entity.get("mentions") or []:
            if not isinstance(mention, dict):
                continue
            fallback = ((mention.get("resource_id") or {}).get("url")) or ""
            for quote in mention.get("quotes") or []:
                if isinstance(quote, dict):
                    yield quote, (quote.get("resource_url") or fallback)
