"""Tests for report data preparation functions."""

from interaction_finder.report.data_prep import _group_assessments_by_document


class TestGroupAssessmentsByDocument:
    """Tests for _group_assessments_by_document function."""

    def test_single_assessment_per_document(self):
        """Single assessment per document should be grouped correctly."""
        assessments = [
            {
                "doc_idx": 0,
                "title": "Study 1",
                "relationship": "causes",
                "quotes": [
                    {"text": "Quote 1", "spans": [[0, 7]], "fuzzy_corrected": False}
                ],
            },
            {
                "doc_idx": 1,
                "title": "Study 2",
                "relationship": "prevents",
                "quotes": [
                    {"text": "Quote 2", "spans": [[10, 17]], "fuzzy_corrected": False}
                ],
            },
        ]

        groups = _group_assessments_by_document(assessments, {}, [])

        assert len(groups) == 2
        assert groups[0]["doc_idx"] == 0
        assert groups[0]["total_quotes"] == 1
        assert len(groups[0]["assessments"]) == 1
        assert groups[1]["doc_idx"] == 1
        assert groups[1]["total_quotes"] == 1
        assert len(groups[1]["assessments"]) == 1

    def test_multiple_assessments_same_document(self):
        """Multiple assessments from same document should be grouped."""
        assessments = [
            {
                "doc_idx": 0,
                "title": "Study 1",
                "relationship": "causes",
                "quotes": [
                    {"text": "Quote 1", "spans": [[0, 7]], "fuzzy_corrected": False}
                ],
            },
            {
                "doc_idx": 0,
                "title": "Study 1",
                "relationship": "prevents",
                "quotes": [
                    {"text": "Quote 2", "spans": [[10, 17]], "fuzzy_corrected": False}
                ],
            },
        ]

        groups = _group_assessments_by_document(assessments, {}, [])

        assert len(groups) == 1
        assert groups[0]["doc_idx"] == 0
        assert groups[0]["total_quotes"] == 2
        assert len(groups[0]["assessments"]) == 2
        assert groups[0]["relationships"] == ["causes", "prevents"]

    def test_quote_deduplication_across_assessments(self):
        """Duplicate quotes across assessments should be deduplicated."""
        # Both assessments use the same quote
        shared_quote = {"text": "Shared", "spans": [[0, 6]], "fuzzy_corrected": False}
        assessments = [
            {
                "doc_idx": 0,
                "title": "Study 1",
                "relationship": "causes",
                "quotes": [
                    shared_quote,
                    {"text": "Unique1", "spans": [[10, 17]], "fuzzy_corrected": False},
                ],
            },
            {
                "doc_idx": 0,
                "title": "Study 1",
                "relationship": "prevents",
                "quotes": [
                    shared_quote,
                    {"text": "Unique2", "spans": [[20, 27]], "fuzzy_corrected": False},
                ],
            },
        ]

        groups = _group_assessments_by_document(assessments, {}, [])

        assert len(groups) == 1
        # Should be 3 unique quotes, not 4 (shared quote counted once)
        assert groups[0]["total_quotes"] == 3
        assert len(groups[0]["assessments"]) == 2

    def test_quote_deduplication_considers_spans(self):
        """Quotes with same text but different spans are different."""
        assessments = [
            {
                "doc_idx": 0,
                "title": "Study 1",
                "relationship": "causes",
                "quotes": [
                    {"text": "Same", "spans": [[0, 4]], "fuzzy_corrected": False},
                ],
            },
            {
                "doc_idx": 0,
                "title": "Study 1",
                "relationship": "prevents",
                "quotes": [
                    {"text": "Same", "spans": [[10, 14]], "fuzzy_corrected": False},
                ],
            },
        ]

        groups = _group_assessments_by_document(assessments, {}, [])

        assert len(groups) == 1
        # Different spans = different quotes
        assert groups[0]["total_quotes"] == 2

    def test_quote_deduplication_considers_fuzzy_flag(self):
        """Quotes differing only in fuzzy_corrected flag are different."""
        assessments = [
            {
                "doc_idx": 0,
                "title": "Study 1",
                "relationship": "causes",
                "quotes": [
                    {"text": "Same", "spans": [[0, 4]], "fuzzy_corrected": False},
                ],
            },
            {
                "doc_idx": 0,
                "title": "Study 1",
                "relationship": "prevents",
                "quotes": [
                    {"text": "Same", "spans": [[0, 4]], "fuzzy_corrected": True},
                ],
            },
        ]

        groups = _group_assessments_by_document(assessments, {}, [])

        assert len(groups) == 1
        # Different fuzzy_corrected flag = different quotes
        assert groups[0]["total_quotes"] == 2

    def test_relationships_deduplicated_preserve_order(self):
        """Duplicate relationships should be deduplicated while preserving order."""
        assessments = [
            {"doc_idx": 0, "relationship": "causes", "quotes": []},
            {"doc_idx": 0, "relationship": "prevents", "quotes": []},
            {"doc_idx": 0, "relationship": "causes", "quotes": []},
        ]

        groups = _group_assessments_by_document(assessments, {}, [])

        assert len(groups) == 1
        # Should preserve first occurrence order
        assert groups[0]["relationships"] == ["causes", "prevents"]

    def test_preserves_original_assessment_order(self):
        """Document groups should appear in order of first assessment occurrence."""
        assessments = [
            {"doc_idx": 2, "relationship": "r1", "quotes": []},
            {"doc_idx": 0, "relationship": "r2", "quotes": []},
            {"doc_idx": 1, "relationship": "r3", "quotes": []},
            {"doc_idx": 0, "relationship": "r4", "quotes": []},
        ]

        groups = _group_assessments_by_document(assessments, {}, [])

        assert len(groups) == 3
        # Order should be 2, 0, 1 (order of first occurrence)
        assert [g["doc_idx"] for g in groups] == [2, 0, 1]
        # Doc 0 should have both assessments
        assert len(groups[1]["assessments"]) == 2

    def test_empty_assessments_list(self):
        """Empty assessments list should return empty groups."""
        groups = _group_assessments_by_document([], {}, [])
        assert groups == []

    def test_missing_quotes_field_handled(self):
        """Assessments without quotes field should be handled gracefully."""
        assessments = [
            {"doc_idx": 0, "relationship": "causes"},  # No quotes field
        ]

        groups = _group_assessments_by_document(assessments, {}, [])

        assert len(groups) == 1
        assert groups[0]["total_quotes"] == 0
