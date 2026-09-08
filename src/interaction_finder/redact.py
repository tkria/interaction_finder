"""Reduce article text to a quote skeleton: the passages a result quotes, each
extended to sentence bounds under its section heading, with omitted gaps marked
by word count. Coverage is capped so a skeleton cannot substitute for the source.

Abstract-only records need no skeleton (`classify_source` identifies them), since
abstracts are distributed openly.

Public API:
    SourceKind, classify_source   article body, or only its abstract?
    SkeletonLimits               bounds on what a skeleton may reproduce
    Snippet, Skeleton            one passage; the passages for one document
    build_skeleton               text + quote spans -> Skeleton
    render_skeleton              Skeleton -> Markdown
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "Skeleton",
    "SkeletonLimits",
    "Snippet",
    "SourceKind",
    "build_skeleton",
    "classify_source",
    "render_skeleton",
]

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)
_PARAGRAPH_BREAK = "\n\n"
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")

_ABSTRACT_HEADING = re.compile(r"^#+[ \t]*abstract\b", re.IGNORECASE | re.MULTILINE)
_ANY_HEADING = re.compile(r"^#+[ \t]+(.+?)[ \t]*$", re.MULTILINE)
# Headings that accompany a bare abstract record, so their presence does not
# make a record an article body.
_FRONT_BACK_MATTER = re.compile(
    r"^(?:abstract|keywords?|author\s+(?:information|contributions)|"
    r"data\s+availability(?:\s+statement)?|conflicts?\s+of\s+interest"
    r"(?:\s+statement)?|competing\s+interests?|funding|acknowledge?ments?|"
    r"references|footnotes|associated\s+data|similar\s+articles|citing\s+articles|"
    r"publication\s+types|mesh\s+terms|related\s+information|"
    r"grants?\s+and\s+funding|actions|permalink|resources|figures?|tables?|"
    r"supplementary\s+materials?|ethics\s+statement|licence|license|copyright|"
    # Publisher page furniture: a fetched abstract page carries paywall and
    # navigation headings, which say nothing about whether a body is present.
    r"buy\s+now|access\s+(?:this\s+article|options|through\s+your\s+institution)|"
    r"read\s+this\s+article|rent\s+or\s+buy|subscribe|institutional\s+access|"
    r"affiliations?|authors?(?:\s+and\s+affiliations?)?|subjects?|"
    r"explore\s+related\s+subjects|linkout|update\s+(?:of|in)|"
    r"supporting\s+information|financial\s+&?\s*competing\s+interests"
    r"(?:\s+disclosure)?|cite\s+this\s+article|share\s+this\s+article|"
    r"similar\s+content|about\s+this\s+article|rights\s+and\s+permissions|"
    r"search|navigation|main\s+content)\b",
    re.IGNORECASE,
)
# An abstract-only record is short; a body that happens to carry no recognised
# heading is not treated as an abstract above this length.
_ABSTRACT_MAX_CHARS = 3_500


class SourceKind(str, Enum):
    """What a retrieved resource's text represents.

    ABSTRACT records need no skeleton: an abstract is distributed openly by
    publishers and indexing services, so it is reproduced as it stands.
    """

    EMPTY = "empty"
    ABSTRACT = "abstract"
    FULL_TEXT = "full-text"


def classify_source(text: str) -> SourceKind:
    """Judge whether `text` is an article body or only its abstract.

    Deliberately conservative: a record counts as an abstract only when it is
    short *and* carries no heading beyond front and back matter. Anything else is
    treated as a body, because misclassifying a body as an abstract would
    reproduce it whole.
    """
    if not text or not text.strip():
        return SourceKind.EMPTY
    if len(text) > _ABSTRACT_MAX_CHARS:
        return SourceKind.FULL_TEXT
    # Require positive evidence of an abstract: unmarked prose could be a body
    # whose headings the fetcher did not preserve, and reproducing that whole is
    # the failure this classification exists to prevent.
    if not _ABSTRACT_HEADING.search(text):
        return SourceKind.FULL_TEXT
    for heading in [h.strip() for h in _ANY_HEADING.findall(text)][1:]:
        if not _FRONT_BACK_MATTER.match(heading):
            return SourceKind.FULL_TEXT
    return SourceKind.ABSTRACT


@dataclass(frozen=True)
class SkeletonLimits:
    """Bounds on reproduction, enforced together (the tightest one wins).

    max_snippet_chars: int -- longest single reproduced passage; a quote whose
        sentence context would exceed this keeps only the matched span.
    max_fraction: float -- largest share of the document reproduced in total.
    max_total_chars: int -- absolute ceiling on total reproduced characters.
    min_total_chars: int -- floor on the budget, so a percentage of a short
        document does not shrink to a fragment too small to carry a quote.
    merge_gap_chars: int -- snippets closer than this are merged, so adjacent
        quotes read as one passage instead of repeating overlapping text.
    """

    max_snippet_chars: int = 600
    max_fraction: float = 0.05
    max_total_chars: int = 5_000
    min_total_chars: int = 300
    merge_gap_chars: int = 60


@dataclass(frozen=True)
class Snippet:
    """One reproduced passage of a document.

    start, end: int -- character offsets into the original text.
    text: str -- the reproduced passage.
    heading: str | None -- nearest preceding section heading, None before any.
    gap_words_before: int -- words omitted since the previous snippet (0 for
        the first, which omits any leading matter without counting it).
    """

    start: int
    end: int
    text: str
    heading: str | None
    gap_words_before: int


@dataclass(frozen=True)
class Skeleton:
    """The shareable form of one document.

    snippets: list[Snippet] -- reproduced passages, in document order.
    source_chars: int -- length of the original text.
    reproduced_chars: int -- total characters reproduced.
    truncated: bool -- whether a limit dropped snippets that quotes supported.
    """

    snippets: list[Snippet] = field(default_factory=list)
    source_chars: int = 0
    reproduced_chars: int = 0
    truncated: bool = False

    @property
    def reproduced_fraction(self) -> float:
        """Share of the source document reproduced, 0.0 for an empty source."""
        return self.reproduced_chars / self.source_chars if self.source_chars else 0.0


def build_skeleton(
    text: str, spans: list[list[int]], limits: SkeletonLimits | None = None
) -> Skeleton:
    """Reduce `text` to the passages covering `spans`, with structure.

    text: str -- the full document text.
    spans: list[list[int]] -- [start, end] character offsets of quote matches;
        order does not matter, and spans outside the text are ignored.
    limits: SkeletonLimits | None -- reproduction bounds; defaults apply if None.

    Returns: Skeleton -- snippets in document order, with coverage accounting.
    """
    limits = limits or SkeletonLimits()
    if not text or not spans:
        return Skeleton(source_chars=len(text))
    headings = [(m.start(), m.group(2).strip()) for m in _HEADING.finditer(text)]
    stops = sorted(m.start() for m in _HEADING.finditer(text))
    valid = _valid_spans(text, spans)
    extended = [_extend(text, start, end, stops, limits) for start, end in valid]
    clipped = any(flag for _, _, flag in extended)
    merged = _merge([(a, b) for a, b, _ in extended], limits.merge_gap_chars)
    return _assemble(text, merged, headings, limits, clipped)


def render_skeleton(skeleton: Skeleton) -> str:
    """Render a skeleton as Markdown, with headings and gap markers."""
    lines: list[str] = []
    current_heading: str | None = ""  # sentinel: not yet emitted anything
    for snippet in skeleton.snippets:
        if snippet.heading != current_heading:
            if snippet.heading:
                lines.append(f"\n## {snippet.heading}\n")
            current_heading = snippet.heading
        if snippet.gap_words_before:
            lines.append(f"…{snippet.gap_words_before} words…")
        lines.append(f"> {snippet.text}")
    if skeleton.truncated:
        lines.append(
            "\n*(further quoted passages truncated to limit how much "
            "of the article is reproduced)*"
        )
    return "\n".join(lines).strip()


def _valid_spans(text: str, spans: list[list[int]]) -> list[tuple[int, int]]:
    """Drop unusable spans and clip those running past the end, then sort."""
    usable = []
    for span in spans:
        if len(span) < 2:
            continue
        start, end = int(span[0]), int(span[1])
        if start < 0 or start >= len(text) or end <= start:
            continue
        usable.append((start, min(end, len(text))))
    return sorted(usable)


def _extend(
    text: str, start: int, end: int, stops: list[int], limits: SkeletonLimits
) -> tuple[int, int, bool]:
    """Grow a span to sentence bounds, without crossing structural boundaries.

    Returns: (begin, end, clipped) -- `clipped` is True when the span had to be
        shortened to stay within max_snippet_chars, which a badly-parsed document
        with runaway sentences can force.
    """
    window = limits.max_snippet_chars
    lower = _section_floor(stops, start)
    upper = _section_ceiling(stops, end, len(text))
    begin = _sentence_start(text, start, max(lower, start - window))
    finish = _sentence_end(text, end, min(upper, end + window), floor=start)
    if finish - begin > limits.max_snippet_chars:
        capped = min(end, start + limits.max_snippet_chars)
        return start, capped, capped < end
    return begin, finish, False


def _section_floor(stops: list[int], position: int) -> int:
    """Offset just past the heading line preceding `position`."""
    prior = [stop for stop in stops if stop < position]
    return prior[-1] if prior else 0


def _section_ceiling(stops: list[int], position: int, length: int) -> int:
    """Offset of the next heading at or after `position`, else end of text."""
    following = [stop for stop in stops if stop >= position]
    return following[0] if following else length


def _sentence_start(text: str, position: int, floor: int) -> int:
    """Start of the sentence (or paragraph) containing `position`."""
    para = text.rfind(_PARAGRAPH_BREAK, floor, position)
    lower = para + len(_PARAGRAPH_BREAK) if para != -1 else floor
    best = lower
    for match in _SENTENCE_END.finditer(text, lower, position):
        best = match.end()
    # Skip whitespace and any heading hashes left at the boundary.
    return min(
        position, best + len(text[best:position]) - len(text[best:position].lstrip())
    )


def _sentence_end(text: str, position: int, ceiling: int, floor: int = 0) -> int:
    """End of the sentence containing `position`, bounded by `ceiling`.

    Searches from just before `position` so a span that already ends on its own
    full stop is not extended into the following sentence.
    """
    para = text.find(_PARAGRAPH_BREAK, position, ceiling)
    upper = para if para != -1 else ceiling
    match = _SENTENCE_END.search(text, max(floor, position - 1), upper)
    return match.end() if match else upper


def _merge(spans: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    """Coalesce spans separated by less than `gap` characters."""
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1] + gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _assemble(
    text: str,
    spans: list[tuple[int, int]],
    headings: list[tuple[int, str]],
    limits: SkeletonLimits,
    clipped: bool = False,
) -> Skeleton:
    """Build snippets from merged spans, clipping when a coverage limit binds.

    The budget binds strictly, including on the first snippet: a passage that
    would overrun it is clipped to what remains rather than admitted whole, so
    the reproduced fraction can never exceed `max_fraction`. Without this a
    single long quote from a short article could reproduce most of it.
    """
    budget = min(limits.max_total_chars, int(limits.max_fraction * len(text)))
    budget = max(budget, min(limits.min_total_chars, len(text)))
    snippets: list[Snippet] = []
    reproduced = 0
    truncated = clipped
    previous_end = 0
    for start, end in spans:
        remaining = budget - reproduced
        if remaining <= 0:
            truncated = True
            break
        passage = text[start:end].strip()
        if not passage:
            continue
        if len(passage) > remaining:
            clipped = passage[:remaining].rsplit(" ", 1)[0]
            if not clipped:
                truncated = True
                break
            passage = clipped + "…"
            truncated = True
        gap = len(text[previous_end:start].split()) if snippets else 0
        snippets.append(
            Snippet(
                start=start,
                end=start + len(passage),
                text=passage,
                heading=_heading_for(headings, start),
                gap_words_before=gap,
            )
        )
        reproduced += len(passage)
        previous_end = end
    return Skeleton(
        snippets=snippets,
        source_chars=len(text),
        reproduced_chars=reproduced,
        truncated=truncated,
    )


def _heading_for(headings: list[tuple[int, str]], position: int) -> str | None:
    """Nearest heading preceding `position`, or None if it precedes them all."""
    prior = [title for offset, title in headings if offset < position]
    return prior[-1] if prior else None
