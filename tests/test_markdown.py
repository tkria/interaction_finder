"""Tests for markdown utilities."""

import pytest

from interaction_finder.markdown import adjust_heading_levels


class TestAdjustHeadingLevels:
    """Tests for adjust_heading_levels function."""

    def test_shifts_level_1_to_level_2(self):
        text = "# Title\n## Section\nSome text"
        result = adjust_heading_levels(text, target_min_level=2)
        assert result == "## Title\n### Section\nSome text"

    def test_shifts_level_2_to_level_1(self):
        text = "## Title\n### Section"
        result = adjust_heading_levels(text, target_min_level=1)
        assert result == "# Title\n## Section"

    def test_no_change_when_already_at_target(self):
        text = "## Title\n### Section"
        result = adjust_heading_levels(text, target_min_level=2)
        assert result == text

    def test_returns_unchanged_when_no_headings(self):
        text = "Just some plain text\nwith multiple lines"
        result = adjust_heading_levels(text, target_min_level=3)
        assert result == text

    def test_clamps_to_level_6_maximum(self):
        text = "##### Level 5\n###### Level 6"
        result = adjust_heading_levels(text, target_min_level=6)
        # Level 5 becomes 6, level 6 stays 6 (clamped)
        assert result == "###### Level 5\n###### Level 6"

    def test_handles_single_heading(self):
        text = "### Only heading"
        result = adjust_heading_levels(text, target_min_level=2)
        assert result == "## Only heading"

    def test_preserves_heading_text(self):
        text = "# Complex heading with **bold** and `code`"
        result = adjust_heading_levels(text, target_min_level=2)
        assert result == "## Complex heading with **bold** and `code`"

    def test_preserves_non_heading_hashes(self):
        text = "# Title\nThis has a #hashtag and code `#comment`"
        result = adjust_heading_levels(text, target_min_level=2)
        assert result == "## Title\nThis has a #hashtag and code `#comment`"

    def test_handles_multiple_headings_same_level(self):
        text = "## First\nText\n## Second\nMore text"
        result = adjust_heading_levels(text, target_min_level=3)
        assert result == "### First\nText\n### Second\nMore text"

    def test_preserves_content_between_headings(self):
        text = "# Title\n\nParagraph with content.\n\n## Section\n\n- List item"
        result = adjust_heading_levels(text, target_min_level=2)
        assert (
            result
            == "## Title\n\nParagraph with content.\n\n### Section\n\n- List item"
        )

    def test_handles_deep_hierarchy(self):
        text = "# H1\n## H2\n### H3\n#### H4"
        result = adjust_heading_levels(text, target_min_level=2)
        assert result == "## H1\n### H2\n#### H3\n##### H4"

    def test_shift_down_from_level_3(self):
        text = "### Deep\n#### Deeper"
        result = adjust_heading_levels(text, target_min_level=1)
        assert result == "# Deep\n## Deeper"

    def test_invalid_target_level_zero(self):
        with pytest.raises(ValueError, match="target_min_level must be 1-6"):
            adjust_heading_levels("# Title", target_min_level=0)

    def test_invalid_target_level_seven(self):
        with pytest.raises(ValueError, match="target_min_level must be 1-6"):
            adjust_heading_levels("# Title", target_min_level=7)

    def test_empty_string(self):
        result = adjust_heading_levels("", target_min_level=2)
        assert result == ""

    def test_heading_at_end_of_text(self):
        text = "Some intro\n# Heading"
        result = adjust_heading_levels(text, target_min_level=2)
        assert result == "Some intro\n## Heading"

    def test_does_not_match_hash_without_space(self):
        text = "#NotAHeading\n# Real Heading"
        result = adjust_heading_levels(text, target_min_level=2)
        assert result == "#NotAHeading\n## Real Heading"
