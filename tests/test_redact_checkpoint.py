"""Tests for redacting a whole checkpoint for sharing.

These cover the checkpoint-level policy: which documents keep their text, which
are reduced to a quote skeleton, and that the provenance a reader needs to
verify a quote survives either way. Licence lookups are stubbed.
"""

import json

import pytest

from interaction_finder.licences import ArticleLicence, Verdict
from interaction_finder.redact import SourceKind, classify_source
from interaction_finder.redact_checkpoint import (
    RedactionPolicy,
    redact_checkpoint_data,
)

ARTICLE = """\
# Title

## Results

Background sentence. ACME1 binds WIDGET2 tightly.

{pad}

Depletion of ACME1 abolished the signal. Closing remark.
""".replace("{pad}", "Filler prose. " * 80)

ABSTRACT_ONLY = (
    "# A Study of Things\n\n## Abstract\n\n"
    "Background sentence. ACME1 binds WIDGET2 tightly. Closing remark.\n"
)

QUOTE = "ACME1 binds WIDGET2 tightly."
SECOND_QUOTE = "Depletion of ACME1 abolished the signal."


def quote_record(text: str, needle: str, url: str) -> dict:
    """One quote entry pointing at `needle` within `text`."""
    start = text.index(needle)
    return {
        "query_text": needle,
        "resource_url": url,
        "spans": [[start, start + len(needle)]],
        "fuzzy_similarity": 1.0,
    }


def checkpoint(text=ARTICLE, url="https://pubmed.ncbi.nlm.nih.gov/1/"):
    """Minimal checkpoint with one resource and two well-separated quotes."""
    return {
        "topic": "t",
        "resources": [
            {
                "url": url,
                "doi": "10.1/x",
                "title": "A Study",
                "publication_date": "2024-01-01",
                "text": text,
                "chunks": [[0, 50], [50, len(text)]],
            }
        ],
        "extraction": {
            "judgments": [
                {
                    "accepted": True,
                    "entity1": {"name": "ACME1", "kind": "gene"},
                    "entity2": {"name": "WIDGET2", "kind": "gene"},
                    "spread": {
                        "positive": [
                            {
                                "resource_id": {"url": url},
                                "quotes": [
                                    quote_record(text, QUOTE, url),
                                    *(
                                        [quote_record(text, SECOND_QUOTE, url)]
                                        if SECOND_QUOTE in text
                                        else []
                                    ),
                                ],
                            }
                        ]
                    },
                }
            ]
        },
    }


_STUB_LICENCE = {
    Verdict.REDISTRIBUTABLE: "cc-by",
    Verdict.SHARE_ALIKE: "cc-by-nc-nd",
    Verdict.RESTRICTED: None,
    Verdict.UNKNOWN: None,
}


def licences(verdict):
    """Stub resolver output marking every article with one verdict."""
    return {
        "https://pubmed.ncbi.nlm.nih.gov/1/": ArticleLicence(
            verdict, _STUB_LICENCE[verdict], "gold", "stub"
        )
    }


class TestRestrictedArticles:
    """Text that may not be redistributed becomes a skeleton."""

    def test_full_text_is_not_retained(self):
        data, _ = redact_checkpoint_data(checkpoint(), licences(Verdict.RESTRICTED))
        text = data["resources"][0]["text"]
        assert "Filler prose." not in text
        assert len(text) < len(ARTICLE) / 4

    def test_quote_survives_in_the_skeleton(self):
        data, _ = redact_checkpoint_data(checkpoint(), licences(Verdict.RESTRICTED))
        assert QUOTE in data["resources"][0]["text"]

    def test_skeleton_keeps_section_heading_and_gap_marker(self):
        data, _ = redact_checkpoint_data(checkpoint(), licences(Verdict.RESTRICTED))
        text = data["resources"][0]["text"]
        assert "Results" in text
        assert "words" in text  # the omitted-gap marker

    def test_provenance_fields_are_kept(self):
        data, _ = redact_checkpoint_data(checkpoint(), licences(Verdict.RESTRICTED))
        resource = data["resources"][0]
        for field in ("url", "doi", "title", "publication_date", "chunks"):
            assert resource[field], f"{field} should survive redaction"

    def test_original_text_is_described_not_discarded(self):
        """A reader must be able to tell what was removed and check a re-fetch."""
        data, _ = redact_checkpoint_data(checkpoint(), licences(Verdict.RESTRICTED))
        resource = data["resources"][0]
        assert resource["text_redacted"] is True
        assert resource["text_length"] == len(ARTICLE)
        assert len(resource["text_sha256"]) == 64

    def test_quote_dropped_by_coverage_limits_is_flagged(self):
        """A quote the skeleton could not include must say so, not go silent."""
        from interaction_finder.redact import SkeletonLimits
        from interaction_finder.redact_checkpoint import RedactionPolicy

        policy = RedactionPolicy(
            limits=SkeletonLimits(max_total_chars=20, min_total_chars=0)
        )
        data, _ = redact_checkpoint_data(
            checkpoint(), licences(Verdict.RESTRICTED), policy
        )
        quotes = data["extraction"]["judgments"][0]["spread"]["positive"][0]["quotes"]
        dropped = [q for q in quotes if q.get("omitted_from_skeleton")]
        assert dropped, "a quote past the coverage cap should be flagged"
        # Its location in the article is still recorded, so it stays checkable.
        assert dropped[0]["spans_in_source"]

    def test_fuzzy_quote_is_relocated_by_source_text(self):
        """Relocation uses the matched source passage, not the LLM's wording."""
        data = checkpoint()
        quote = data["extraction"]["judgments"][0]["spread"]["positive"][0]["quotes"][0]
        quote["query_text"] = "ACME1 binds WIDGET-2 tightly."  # paraphrased
        quote["fuzzy_similarity"] = 0.94
        redacted, _ = redact_checkpoint_data(data, licences(Verdict.RESTRICTED))
        text = redacted["resources"][0]["text"]
        rewritten = redacted["extraction"]["judgments"][0]["spread"]["positive"][0][
            "quotes"
        ][0]
        start, end = rewritten["spans"][0]
        assert text[start:end] == QUOTE

    def test_quote_spans_are_rewritten_to_the_skeleton(self):
        """Spans must index the text that ships, or highlighting breaks."""
        data, _ = redact_checkpoint_data(checkpoint(), licences(Verdict.RESTRICTED))
        text = data["resources"][0]["text"]
        quote = data["extraction"]["judgments"][0]["spread"]["positive"][0]["quotes"][0]
        start, end = quote["spans"][0]
        assert text[start:end] == QUOTE

    def test_original_spans_are_preserved_separately(self):
        data, _ = redact_checkpoint_data(checkpoint(), licences(Verdict.RESTRICTED))
        quote = data["extraction"]["judgments"][0]["spread"]["positive"][0]["quotes"][0]
        assert quote["spans_in_source"] == [
            [ARTICLE.index(QUOTE), ARTICLE.index(QUOTE) + len(QUOTE)]
        ]


class TestRedistributableArticles:
    """Openly licensed text may be kept, when the policy allows it."""

    def test_permissive_text_is_kept_by_default(self):
        data, _ = redact_checkpoint_data(
            checkpoint(), licences(Verdict.REDISTRIBUTABLE)
        )
        assert data["resources"][0]["text"] == ARTICLE
        assert not data["resources"][0].get("text_redacted")

    def test_everything_policy_redacts_regardless_of_licence(self):
        """`--everything` ignores licences, for a single defensible basis."""
        data, _ = redact_checkpoint_data(
            checkpoint(),
            licences(Verdict.REDISTRIBUTABLE),
            RedactionPolicy(keep_redistributable=False, keep_share_alike=False),
        )
        assert data["resources"][0]["text_redacted"] is True


class TestLicenceRecording:
    """The licence is a fact about the article, not about the redaction."""

    def test_licence_is_recorded_when_text_is_kept(self):
        data, _ = redact_checkpoint_data(
            checkpoint(), licences(Verdict.REDISTRIBUTABLE)
        )
        assert data["resources"][0]["licence"] == "cc-by"

    def test_licence_is_recorded_even_when_text_is_redacted(self):
        """A recipient needs the licence whether or not the text survived."""
        data, _ = redact_checkpoint_data(checkpoint(), licences(Verdict.SHARE_ALIKE))
        assert data["resources"][0]["licence"] == "cc-by-nc-nd"

    def test_summary_groups_kept_articles_by_licence(self):
        _, summary = redact_checkpoint_data(
            checkpoint(), licences(Verdict.REDISTRIBUTABLE)
        )
        assert summary.kept_by_licence == {"cc-by": 1}


class TestConditionalAndUnknown:
    """Cases where sharing is neither clearly permitted nor clearly barred."""

    def test_share_alike_text_is_kept_by_default(self):
        """NC/ND permits redistribution, so its text is kept with the licence."""
        data, _ = redact_checkpoint_data(checkpoint(), licences(Verdict.SHARE_ALIKE))
        assert data["resources"][0]["text"] == ARTICLE
        assert data["resources"][0]["licence"] == "cc-by-nc-nd"

    def test_share_alike_can_be_declined(self):
        """`--no-share-alike` redacts it for those who want only free terms."""
        data, _ = redact_checkpoint_data(
            checkpoint(),
            licences(Verdict.SHARE_ALIKE),
            RedactionPolicy(keep_share_alike=False),
        )
        assert data["resources"][0]["text_redacted"] is True

    def test_unknown_licence_is_redacted(self):
        """An unresolvable licence is not treated as permission."""
        data, _ = redact_checkpoint_data(checkpoint(), licences(Verdict.UNKNOWN))
        assert data["resources"][0]["text_redacted"] is True


class TestSummary:
    """What the command reports back to the user."""

    def test_summary_counts_actions_and_savings(self):
        _, summary = redact_checkpoint_data(checkpoint(), licences(Verdict.RESTRICTED))
        assert summary.redacted == 1
        assert summary.kept == 0
        assert summary.chars_removed > 0
        assert summary.by_verdict["restricted"] == 1

    def test_summary_reports_kept_documents(self):
        _, summary = redact_checkpoint_data(
            checkpoint(), licences(Verdict.REDISTRIBUTABLE)
        )
        assert summary.kept == 1
        assert summary.redacted == 0


class TestIdempotenceAndEdgeCases:
    def test_redacting_twice_is_stable(self):
        once, _ = redact_checkpoint_data(checkpoint(), licences(Verdict.RESTRICTED))
        twice, summary = redact_checkpoint_data(
            json.loads(json.dumps(once)), licences(Verdict.RESTRICTED)
        )
        assert twice["resources"][0]["text"] == once["resources"][0]["text"]
        assert summary.already_redacted == 1

    def test_resource_without_quotes_keeps_no_text(self):
        """An article supporting no quote has nothing to justify reproducing."""
        data = checkpoint()
        data["extraction"]["judgments"] = []
        redacted, _ = redact_checkpoint_data(data, licences(Verdict.RESTRICTED))
        assert redacted["resources"][0]["text"] == ""

    def test_missing_extraction_stage_is_handled(self):
        data = checkpoint()
        del data["extraction"]
        redacted, summary = redact_checkpoint_data(data, licences(Verdict.RESTRICTED))
        assert redacted["resources"][0]["text_redacted"] is True
        assert summary.redacted == 1


class TestLicenceSurvivesCheckpointLoad:
    """The licence must reach the report, which reads the parsed model."""

    def test_flat_resource_entry_keeps_its_licence(self):
        """ResourcePool coercion enumerates fields, so new ones must be added."""
        from interaction_finder.resources import ResourcePool

        pool = ResourcePool.model_validate(
            [
                {
                    "url": "https://pubmed.ncbi.nlm.nih.gov/1/",
                    "title": "A Study",
                    "text": "Body.",
                    "licence": "cc-by",
                }
            ]
        )
        resource = next(iter(pool.resource_map.values()))
        assert resource.licence == "cc-by"

    def test_redacted_resources_reach_the_model_with_their_licence(self):
        """What redaction writes must survive the load path reports read."""
        from interaction_finder.resources import ResourcePool

        data, _ = redact_checkpoint_data(
            checkpoint(), licences(Verdict.REDISTRIBUTABLE)
        )
        pool = ResourcePool.model_validate(data["resources"])
        resources = [r for r in pool.resource_map.values() if r]
        assert [r.licence for r in resources] == ["cc-by"]


class TestResourceKind:
    """Abstract-only records are a different kind of source, not a short article."""

    def test_abstract_only_record_is_recognised(self):
        text = (
            "# A Study of Things\n\n## Abstract\n\n"
            "Things interact. We found ACME1 binds WIDGET2.\n"
        )
        assert classify_source(text) is SourceKind.ABSTRACT

    def test_numbered_body_headings_mean_full_text(self):
        text = (
            "# A Study\n\n## Abstract\n\nSummary.\n\n"
            "## 1. Introduction\n\nPrior work.\n\n## 2. Results\n\nFindings.\n"
        )
        assert classify_source(text) is SourceKind.FULL_TEXT

    def test_plain_body_headings_mean_full_text(self):
        text = "## Abstract\n\nS.\n\n## Methods\n\nM.\n\n## Discussion\n\nD.\n"
        assert classify_source(text) is SourceKind.FULL_TEXT

    def test_front_and_back_matter_do_not_make_it_full_text(self):
        """Data-availability and conflict statements accompany bare abstracts."""
        text = (
            "# A Study\n\n## Abstract\n\nSummary.\n\n"
            "## Data availability statement\n\nOn request.\n\n"
            "## Conflict of interest statement\n\nNone.\n\n## Footnotes\n\nx\n"
        )
        assert classify_source(text) is SourceKind.ABSTRACT

    def test_body_text_without_an_abstract_heading_is_full_text(self):
        text = "## Introduction\n\nPrior work.\n\n## Results\n\nFindings.\n"
        assert classify_source(text) is SourceKind.FULL_TEXT

    def test_unmarked_prose_is_treated_as_a_body(self):
        """Without an abstract heading there is no evidence it is an abstract, and
        misclassifying a body would reproduce it whole."""
        assert classify_source("A short unstructured summary.") is SourceKind.FULL_TEXT
        assert classify_source("word " * 2000) is SourceKind.FULL_TEXT

    def test_plain_text_body_is_not_mistaken_for_an_abstract(self):
        body = "Body sentence about ACME1 binding. " * 90
        assert classify_source(body) is SourceKind.FULL_TEXT

    def test_uppercase_section_headings_mean_full_text(self):
        body = "INTRODUCTION\n\nText.\n\nRESULTS\n\nMore about ACME1. " * 20
        assert classify_source(body) is SourceKind.FULL_TEXT

    def test_case_report_sections_mean_full_text(self):
        text = "# T\n\n## Abstract\n\nS.\n\n## Case Presentation\n\nA patient.\n"
        assert classify_source(text) is SourceKind.FULL_TEXT

    def test_empty_text_is_neither(self):
        assert classify_source("") is SourceKind.EMPTY


class TestAbstractsAreKeptWhole:
    """An abstract needs no skeleton: it is reproduced as it stands."""

    def test_abstract_record_keeps_its_text(self):
        data = checkpoint(text=ABSTRACT_ONLY)
        redacted, summary = redact_checkpoint_data(
            data, licences(Verdict.RESTRICTED)
        )
        resource = redacted["resources"][0]
        assert resource["text"] == ABSTRACT_ONLY
        assert not resource.get("text_redacted")
        assert summary.abstracts_kept == 1

    def test_abstract_quote_spans_are_untouched(self):
        data = checkpoint(text=ABSTRACT_ONLY)
        original = data["extraction"]["judgments"][0]["spread"]["positive"][0][
            "quotes"
        ][0]["spans"][0][:]
        redacted, _ = redact_checkpoint_data(data, licences(Verdict.RESTRICTED))
        quote = redacted["extraction"]["judgments"][0]["spread"]["positive"][0][
            "quotes"
        ][0]
        assert quote["spans"][0] == original
        assert "spans_in_source" not in quote

    def test_full_text_record_is_still_skeletonised(self):
        """The change must not spare full-text articles."""
        redacted, summary = redact_checkpoint_data(
            checkpoint(), licences(Verdict.RESTRICTED)
        )
        assert redacted["resources"][0]["text_redacted"] is True
        assert summary.abstracts_kept == 0


class TestNoTextSurvivesForRestrictedArticles:
    """The invariant, tested over the whole serialised file rather than a field."""

    def test_no_long_run_of_article_text_survives(self):
        """No 200-char stretch of a restricted article may appear anywhere."""
        body = (
            "## Introduction\n\n"
            + "".join(f"Distinctive sentence {i} concerning ACME1 and WIDGET2. "
                      for i in range(200))
        )
        quote = "Distinctive sentence 7 concerning ACME1 and WIDGET2."
        start = body.index(quote)
        data = {
            "resources": [{"url": "u", "title": "t", "text": body, "chunks": []}],
            "extraction": {"judgments": [{
                "accepted": True,
                "entity1": {"name": "ACME1", "kind": "gene"},
                "entity2": {"name": "WIDGET2", "kind": "gene"},
                "spread": {"positive": [{
                    "resource_id": {"url": "u"},
                    "quotes": [{"query_text": quote, "resource_url": "u",
                                "spans": [[start, start + len(quote)]]}],
                }]},
            }]},
        }
        redacted, _ = redact_checkpoint_data(data, licences(Verdict.RESTRICTED))
        serialised = json.dumps(redacted)
        leaked = [
            body[i:i + 200]
            for i in range(0, len(body) - 200, 50)
            if body[i:i + 200] in serialised
        ]
        assert not leaked, f"{len(leaked)} stretches of the article survived"

    def test_densely_quoted_article_stays_within_the_cap(self):
        """Adjacent quotes must not chain into a near-complete reproduction."""
        sentences = [f"Sentence {i} about ACME1 binding WIDGET2 tightly. "
                     for i in range(200)]
        body = "## Results\n\n" + "".join(sentences)
        quotes, position = [], len("## Results\n\n")
        for sentence in sentences:
            quotes.append({
                "query_text": sentence.strip(), "resource_url": "u",
                "spans": [[position, position + len(sentence) - 1]],
            })
            position += len(sentence)
        data = {
            "resources": [{"url": "u", "title": "t", "text": body, "chunks": []}],
            "extraction": {"judgments": [{
                "accepted": True,
                "entity1": {"name": "ACME1", "kind": "gene"},
                "entity2": {"name": "WIDGET2", "kind": "gene"},
                "spread": {"positive": [
                    {"resource_id": {"url": "u"}, "quotes": quotes}
                ]},
            }]},
        }
        redacted, _ = redact_checkpoint_data(data, licences(Verdict.RESTRICTED))
        kept = len(redacted["resources"][0]["text"])
        assert kept < 0.2 * len(body), f"reproduced {kept / len(body):.0%} of the article"


class TestUnsafeInputIsRefused:
    """Shapes that cannot be redacted must raise, not pass through."""

    def test_non_dict_resource_raises(self):
        from interaction_finder.redact_checkpoint import RedactionError

        with pytest.raises(RedactionError):
            redact_checkpoint_data(
                {"resources": ["not a resource"]}, licences(Verdict.RESTRICTED)
            )

    def test_non_string_text_raises(self):
        from interaction_finder.redact_checkpoint import RedactionError

        with pytest.raises(RedactionError):
            redact_checkpoint_data(
                {"resources": [{"url": "u", "text": ["body"]}]},
                licences(Verdict.RESTRICTED),
            )

    def test_wrapped_dict_resources_are_redacted_not_skipped(self):
        data = {"resources": {"https://pubmed.ncbi.nlm.nih.gov/1/": {
            "url": "https://pubmed.ncbi.nlm.nih.gov/1/", "title": "t", "text": ARTICLE,
        }}}
        redacted, summary = redact_checkpoint_data(data, licences(Verdict.RESTRICTED))
        assert summary.redacted == 1
        assert ARTICLE not in json.dumps(redacted)

    def test_redacted_marker_with_full_text_raises(self):
        """A record claiming to be redacted while holding its body is refused."""
        from interaction_finder.redact_checkpoint import RedactionError

        with pytest.raises(RedactionError):
            redact_checkpoint_data(
                {"resources": [{"url": "u", "text": ARTICLE, "text_redacted": True,
                                "text_length": len(ARTICLE)}]},
                licences(Verdict.RESTRICTED),
            )


class TestRedactionMarkersSurviveSerialisation:
    def test_dump_and_reload_keeps_the_markers(self):
        from interaction_finder.resources import ResourcePool

        pool = ResourcePool.model_validate([{
            "url": "u", "title": "t", "text": "skeleton", "licence": "cc-by",
            "text_redacted": True, "text_length": 9999, "text_sha256": "a" * 64,
        }])
        reloaded = ResourcePool.model_validate(json.loads(pool.model_dump_json()))
        resource = next(iter(reloaded.resource_map.values()))
        assert resource.text_redacted is True
        assert resource.licence == "cc-by"
        assert resource.text_length == 9999
        assert resource.text_sha256 == "a" * 64
