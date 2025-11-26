"""Tests for document citation extraction and validation."""

from interaction_finder.extraction.utils import (
    extract_document_citations,
    validate_document_citations,
)


class TestExtractDocumentCitations:
    """Tests for extract_document_citations function."""

    def test_extracts_single_citation(self):
        text = "Evidence from [1_abc12345] supports this."
        assert extract_document_citations(text) == ["1_abc12345"]

    def test_extracts_multiple_citations(self):
        text = "Found in [1_abc12345] and confirmed by [2_def67890]."
        assert extract_document_citations(text) == ["1_abc12345", "2_def67890"]

    def test_deduplicates_repeated_citations(self):
        text = "Cited [1_abc12345] here and [1_abc12345] again."
        assert extract_document_citations(text) == ["1_abc12345"]

    def test_preserves_first_appearance_order(self):
        text = (
            "[3_cccccccc] then [1_aaaaaaaa] then [2_bbbbbbbb] then [1_aaaaaaaa] again."
        )
        assert extract_document_citations(text) == [
            "3_cccccccc",
            "1_aaaaaaaa",
            "2_bbbbbbbb",
        ]

    def test_returns_empty_for_no_citations(self):
        text = "This text has no document citations."
        assert extract_document_citations(text) == []

    def test_returns_empty_for_empty_string(self):
        assert extract_document_citations("") == []

    def test_ignores_malformed_citations(self):
        # All malformed: wrong length, uppercase, missing parts
        text = "[abc] [1_] [_abc] [1abc] [1_ABC12345] [1_abc123] [1_abc1234567] valid: [1_abc12345]"
        # Only [1_abc12345] matches (exactly 8 lowercase alphanumeric after _)
        assert extract_document_citations(text) == ["1_abc12345"]

    def test_handles_citations_at_boundaries(self):
        text = "[1_start000] middle [2_end00000]"
        assert extract_document_citations(text) == ["1_start000", "2_end00000"]

    def test_handles_adjacent_citations(self):
        text = "[1_aaaaaaaa][2_bbbbbbbb][3_cccccccc]"
        assert extract_document_citations(text) == [
            "1_aaaaaaaa",
            "2_bbbbbbbb",
            "3_cccccccc",
        ]

    def test_handles_multiline_text(self):
        text = "Line one [1_abcdefgh]\nLine two [2_ijklmnop]\nLine three"
        assert extract_document_citations(text) == ["1_abcdefgh", "2_ijklmnop"]

    def test_handles_realistic_reasoning(self):
        text = """The association between BRCA1 and breast cancer is well-supported.
        Document [1_a7f3b2c1] provides strong evidence from a meta-analysis,
        while [2_d4e5f6g7] offers mechanistic insights. However, [3_h8i9j0k1]
        presents some conflicting data. Overall, the evidence from [1_a7f3b2c1]
        and [2_d4e5f6g7] outweighs the concerns raised in [3_h8i9j0k1]."""
        assert extract_document_citations(text) == [
            "1_a7f3b2c1",
            "2_d4e5f6g7",
            "3_h8i9j0k1",
        ]

    def test_rejects_wrong_length_hash(self):
        # 7 chars (too short)
        assert extract_document_citations("[1_abc1234]") == []
        # 9 chars (too long)
        assert extract_document_citations("[1_abc123456]") == []
        # Exactly 8 chars (valid)
        assert extract_document_citations("[1_abc12345]") == ["1_abc12345"]


class TestValidateDocumentCitations:
    """Tests for validate_document_citations function."""

    def test_all_valid(self):
        cited = ["1_aaaaaaaa", "2_bbbbbbbb"]
        valid_ids = {"1_aaaaaaaa", "2_bbbbbbbb", "3_cccccccc"}
        valid, invalid = validate_document_citations(cited, valid_ids)
        assert valid == ["1_aaaaaaaa", "2_bbbbbbbb"]
        assert invalid == []

    def test_all_invalid(self):
        cited = ["4_xxxxxxxx", "5_yyyyyyyy"]
        valid_ids = {"1_aaaaaaaa", "2_bbbbbbbb"}
        valid, invalid = validate_document_citations(cited, valid_ids)
        assert valid == []
        assert invalid == ["4_xxxxxxxx", "5_yyyyyyyy"]

    def test_mixed_valid_invalid(self):
        cited = ["1_aaaaaaaa", "4_xxxxxxxx", "2_bbbbbbbb", "5_yyyyyyyy"]
        valid_ids = {"1_aaaaaaaa", "2_bbbbbbbb", "3_cccccccc"}
        valid, invalid = validate_document_citations(cited, valid_ids)
        assert valid == ["1_aaaaaaaa", "2_bbbbbbbb"]
        assert invalid == ["4_xxxxxxxx", "5_yyyyyyyy"]

    def test_preserves_order(self):
        cited = ["3_cccccccc", "1_aaaaaaaa", "2_bbbbbbbb"]
        valid_ids = {"1_aaaaaaaa", "2_bbbbbbbb", "3_cccccccc"}
        valid, invalid = validate_document_citations(cited, valid_ids)
        assert valid == ["3_cccccccc", "1_aaaaaaaa", "2_bbbbbbbb"]

    def test_empty_cited(self):
        valid, invalid = validate_document_citations([], {"1_aaaaaaaa"})
        assert valid == []
        assert invalid == []

    def test_empty_valid_ids(self):
        cited = ["1_aaaaaaaa", "2_bbbbbbbb"]
        valid, invalid = validate_document_citations(cited, set())
        assert valid == []
        assert invalid == ["1_aaaaaaaa", "2_bbbbbbbb"]
