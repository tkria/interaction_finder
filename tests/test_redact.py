"""Tests for redacting documents to a quote skeleton.

A skeleton reproduces the quoted passages of a document plus the structure
around them (section headings, and the size of each omitted gap) in place of the
full text, so a reader can judge quoted evidence without the article being
redistributed.
"""

import pytest

from interaction_finder.redact import (
    SkeletonLimits,
    build_skeleton,
    render_skeleton,
)

# A document with real Markdown structure, two sections, and quotable sentences.
DOC = """\
# A Study of Things

## Abstract

Things are known to interact. We found that ACME1 binds WIDGET2 in vitro. \
This was unexpected.

## Introduction

Prior work established little. {filler} The field has moved slowly.

## Results

We observed strong binding. ACME1 depletion abolished the WIDGET2 signal \
entirely. Controls behaved as expected.
""".replace("{filler}", "Padding sentence. " * 60)


def span_of(text: str, needle: str) -> list[int]:
    """Character span of `needle` in `text`, as [start, end]."""
    start = text.index(needle)
    return [start, start + len(needle)]


class TestSnippetSelection:
    """What text a skeleton reproduces, and what it leaves out."""

    def test_quote_extends_to_sentence_boundaries(self):
        """A quote grows to whole sentences so it reads in context."""
        span = span_of(DOC, "binds WIDGET2")
        skeleton = build_skeleton(DOC, [span])
        (snippet,) = skeleton.snippets
        assert snippet.text.startswith("We found that ACME1")
        assert snippet.text.endswith("in vitro.")

    def test_extension_stops_at_heading(self):
        """Extension must not run across a heading into the next section."""
        span = span_of(DOC, "Controls behaved as expected")
        skeleton = build_skeleton(DOC, [span])
        (snippet,) = skeleton.snippets
        assert "##" not in snippet.text
        assert "Results" not in snippet.text

    def test_extension_stops_at_paragraph_break(self):
        """A blank line bounds a snippet even without a heading."""
        span = span_of(DOC, "The field has moved slowly")
        skeleton = build_skeleton(DOC, [span])
        (snippet,) = skeleton.snippets
        assert "Results" not in snippet.text
        assert "\n\n" not in snippet.text.strip()

    def test_long_sentence_falls_back_to_the_match(self):
        """A pathological sentence must not drag in unbounded text."""
        runaway = "word " * 500 + "TARGET here."
        span = span_of(runaway, "TARGET")
        skeleton = build_skeleton(runaway, [span], SkeletonLimits(max_snippet_chars=80))
        (snippet,) = skeleton.snippets
        assert "TARGET" in snippet.text
        assert len(snippet.text) <= 80

    def test_adjacent_spans_merge_into_one_snippet(self):
        """Overlapping or near-touching quotes should not be repeated."""
        first = span_of(DOC, "We observed strong binding")
        second = span_of(DOC, "ACME1 depletion abolished")
        skeleton = build_skeleton(DOC, [first, second])
        assert len(skeleton.snippets) == 1

    def test_distant_spans_stay_separate(self):
        """Quotes far apart remain distinct snippets, in document order."""
        first = span_of(DOC, "binds WIDGET2")
        second = span_of(DOC, "ACME1 depletion abolished")
        skeleton = build_skeleton(DOC, [second, first])  # deliberately unordered
        assert len(skeleton.snippets) == 2
        assert skeleton.snippets[0].start < skeleton.snippets[1].start


class TestStructure:
    """The headings and gaps that make a skeleton navigable."""

    def test_snippet_carries_its_section_heading(self):
        span = span_of(DOC, "ACME1 depletion abolished")
        skeleton = build_skeleton(DOC, [span])
        assert skeleton.snippets[0].heading == "Results"

    def test_heading_is_the_nearest_preceding_one(self):
        span = span_of(DOC, "binds WIDGET2")
        skeleton = build_skeleton(DOC, [span])
        assert skeleton.snippets[0].heading == "Abstract"

    def test_text_before_any_heading_has_no_section(self):
        doc = "Loose opening sentence about ACME1.\n\n## Later\n\nMore."
        skeleton = build_skeleton(doc, [span_of(doc, "Loose opening")])
        assert skeleton.snippets[0].heading is None

    def test_gap_records_omitted_word_count(self):
        first = span_of(DOC, "binds WIDGET2")
        second = span_of(DOC, "ACME1 depletion abolished")
        skeleton = build_skeleton(DOC, [first, second])
        gap = skeleton.snippets[1].gap_words_before
        assert gap > 100  # the padded Introduction sits between them

    def test_first_snippet_has_no_preceding_gap(self):
        skeleton = build_skeleton(DOC, [span_of(DOC, "binds WIDGET2")])
        assert skeleton.snippets[0].gap_words_before == 0


class TestCoverageCeiling:
    """The archive must never approach a substitute for the article."""

    def test_reproduced_fraction_is_capped(self):
        """Many quotes cannot cumulatively reproduce the document."""
        spans = [
            span_of(DOC, needle)
            for needle in (
                "Things are known",
                "Prior work established",
                "We observed strong binding",
                "Controls behaved",
            )
        ]
        limits = SkeletonLimits(max_fraction=0.02, max_total_chars=10_000,
                                min_total_chars=0)
        skeleton = build_skeleton(DOC, spans, limits)
        assert skeleton.reproduced_chars <= 0.02 * len(DOC) + limits.max_snippet_chars
        assert skeleton.truncated is True

    def test_absolute_character_cap_applies(self):
        spans = [span_of(DOC, "We observed strong binding")]
        limits = SkeletonLimits(max_total_chars=20, min_total_chars=0)
        skeleton = build_skeleton(DOC, spans, limits)
        assert skeleton.reproduced_chars <= 20 + limits.max_snippet_chars

    def test_untruncated_skeleton_reports_so(self):
        skeleton = build_skeleton(DOC, [span_of(DOC, "binds WIDGET2")])
        assert skeleton.truncated is False


class TestDegenerateInput:
    """Inputs that must not raise."""

    def test_no_spans_yields_empty_skeleton(self):
        skeleton = build_skeleton(DOC, [])
        assert skeleton.snippets == []
        assert skeleton.reproduced_chars == 0

    def test_empty_document_yields_empty_skeleton(self):
        assert build_skeleton("", [[0, 5]]).snippets == []

    @pytest.mark.parametrize("span", [[-5, 3], [10, 5], [900_000, 900_010]])
    def test_out_of_range_spans_are_dropped(self, span):
        """Offsets that cannot index the text are skipped, not clamped blindly."""
        assert build_skeleton(DOC, [span]).snippets == []

    def test_span_partially_past_end_is_clipped(self):
        doc = "Short document about ACME1."
        skeleton = build_skeleton(doc, [[21, 999]])
        assert skeleton.snippets[0].text.endswith("ACME1.")


class TestRendering:
    """The human-readable form that goes into a report."""

    def test_render_shows_headings_snippets_and_gaps(self):
        first = span_of(DOC, "binds WIDGET2")
        second = span_of(DOC, "ACME1 depletion abolished")
        rendered = render_skeleton(build_skeleton(DOC, [first, second]))
        assert "## Abstract" in rendered
        assert "## Results" in rendered
        assert "words" in rendered  # gap marker
        assert "ACME1 depletion abolished" in rendered

    def test_render_notes_truncation(self):
        """When a limit drops quotes the document supported, say so."""
        spans = [
            span_of(DOC, needle)
            for needle in (
                "Things are known",
                "Prior work established",
                "We observed strong binding",
                "Controls behaved",
            )
        ]
        limits = SkeletonLimits(max_total_chars=20, min_total_chars=0)
        skeleton = build_skeleton(DOC, spans, limits)
        assert skeleton.truncated is True
        assert "truncated" in render_skeleton(skeleton).lower()


class TestCoverageCapBindsStrictly:
    """The fraction cap must hold, including against a single long quote."""

    def test_single_long_quote_cannot_exceed_the_fraction(self):
        doc = "## Results\n\n" + ("A" * 40 + ". ") * 20
        skeleton = build_skeleton(
            doc, [[12, 412]], SkeletonLimits(max_fraction=0.05, max_total_chars=5000, min_total_chars=0)
        )
        # The per-snippet floor is the one concession; beyond it the cap binds.
        assert skeleton.reproduced_chars <= max(
            0.05 * len(doc), SkeletonLimits().max_snippet_chars
        )

    def test_overrunning_snippet_is_clipped_not_dropped(self):
        doc = "## R\n\n" + "word " * 400
        skeleton = build_skeleton(
            doc, [[6, 2000]], SkeletonLimits(max_snippet_chars=120, max_fraction=0.05, min_total_chars=0)
        )
        assert skeleton.snippets, "a quote should still be represented"
        assert skeleton.truncated is True
        assert skeleton.snippets[0].text.endswith("…")

    def test_clipping_is_reported_as_truncation(self):
        doc = "## R\n\n" + "sentence here. " * 200
        skeleton = build_skeleton(
            doc, [[6, 1500]], SkeletonLimits(max_fraction=0.02, max_snippet_chars=100, min_total_chars=0)
        )
        assert skeleton.truncated is True

    def test_snippet_end_matches_the_text_actually_kept(self):
        """A clipped snippet's offsets must not claim more than it shows."""
        doc = "## R\n\n" + "word " * 400
        skeleton = build_skeleton(
            doc, [[6, 2000]], SkeletonLimits(max_snippet_chars=120, max_fraction=0.05, min_total_chars=0)
        )
        snippet = skeleton.snippets[0]
        assert snippet.end - snippet.start == len(snippet.text)
