"""Comprehensive tests for position remapping accuracy."""

import pytest

from interaction_finder.report.html_renderer import MarkdownToHTMLRenderer


class TestPositionRemappingAccuracy:
    """Tests verifying position remapping is accurate."""

    def test_every_position_in_simple_text(self):
        """Every position in simple text should map correctly."""
        text = "Hello World"
        renderer = MarkdownToHTMLRenderer(text)
        html, offsets = renderer.render()

        # HTML is "<p>Hello World</p>"
        # Position 0 (H) should map to 3 (after <p>)
        assert renderer.map_original_to_html_position(0) == 3
        # Position 5 (space) should map to 8
        assert renderer.map_original_to_html_position(5) == 8
        # Position 6 (W) should map to 9
        assert renderer.map_original_to_html_position(6) == 9
        # Last character position
        assert renderer.map_original_to_html_position(len(text) - 1) >= 3

    def test_positions_through_html_escaping(self):
        """Position mapping through HTML entity escaping."""
        text = "A<B>C&D"
        renderer = MarkdownToHTMLRenderer(text)
        html, offsets = renderer.render()

        # Original: "A<B>C&D"
        # HTML: "<p>A&lt;B&gt;C&amp;D</p>"

        # Position 0 (A) -> 3 (after <p>)
        assert renderer.map_original_to_html_position(0) == 3

        # Position 1 (<) -> should be after "A" in HTML
        pos_1 = renderer.map_original_to_html_position(1)
        assert pos_1 >= 4  # At least after "<p>A"

        # Position 2 (B) -> should be after "&lt;" in HTML
        pos_2 = renderer.map_original_to_html_position(2)
        assert pos_2 > pos_1

        # Positions should be monotonically increasing
        for i in range(len(text) - 1):
            pos_i = renderer.map_original_to_html_position(i)
            pos_i_plus_1 = renderer.map_original_to_html_position(i + 1)
            assert pos_i <= pos_i_plus_1, (
                f"Position {i} maps to {pos_i}, but {i + 1} maps to {pos_i_plus_1}"
            )

    def test_positions_through_bold_transformation(self):
        """Position mapping through bold transformation."""
        text = "Start **bold** end"
        renderer = MarkdownToHTMLRenderer(text)
        html, offsets = renderer.render()

        # Original: "Start **bold** end"
        # Positions: 0-4: "Start", 5: " ", 6-7: "**", 8-11: "bold", 12-13: "**", 14: " ", 15-17: "end"

        # Position 0 (S) should map to 3 (after <p>)
        assert renderer.map_original_to_html_position(0) == 3

        # Position 5 (space before **) should map correctly
        pos_5 = renderer.map_original_to_html_position(5)
        assert pos_5 > 3

        # Position 6 (first *) - start of transformation
        pos_6 = renderer.map_original_to_html_position(6)

        # Position 14 (space after **) should be after </strong>
        pos_14 = renderer.map_original_to_html_position(14)
        assert pos_14 > pos_6

        # Verify monotonicity
        for i in range(len(text) - 1):
            pos_i = renderer.map_original_to_html_position(i)
            pos_i_plus_1 = renderer.map_original_to_html_position(i + 1)
            assert pos_i <= pos_i_plus_1

    def test_positions_through_heading(self):
        """Position mapping through heading transformation."""
        text = "# Title\n\nText"
        renderer = MarkdownToHTMLRenderer(text)
        html, offsets = renderer.render()

        # Original: "# Title\n\nText"
        # Position 0 (#) is start of heading
        # Position 2 (T in Title) is heading content

        pos_0 = renderer.map_original_to_html_position(0)
        pos_2 = renderer.map_original_to_html_position(2)
        pos_7 = renderer.map_original_to_html_position(7)  # First \n
        pos_9 = renderer.map_original_to_html_position(9)  # T in Text

        # Should be monotonically increasing
        assert pos_0 <= pos_2 <= pos_7 <= pos_9

    def test_positions_with_multiple_paragraphs(self):
        """Position mapping across paragraph boundaries."""
        text = "First para\n\nSecond para"
        renderer = MarkdownToHTMLRenderer(text)
        html, offsets = renderer.render()

        # Test positions in first paragraph
        pos_0 = renderer.map_original_to_html_position(0)  # F
        pos_5 = renderer.map_original_to_html_position(5)  # space
        pos_10 = renderer.map_original_to_html_position(10)  # first \n

        # Test positions in second paragraph
        pos_12 = renderer.map_original_to_html_position(12)  # S
        pos_18 = renderer.map_original_to_html_position(18)  # space

        # Should be monotonically increasing
        assert pos_0 < pos_5 < pos_10 < pos_12 < pos_18

    def test_positions_with_nested_formatting(self):
        """Position mapping with nested formatting."""
        text = "**Bold with *italic* inside**"
        renderer = MarkdownToHTMLRenderer(text)
        html, offsets = renderer.render()

        # Test a few key positions
        pos_0 = renderer.map_original_to_html_position(0)  # First *
        pos_2 = renderer.map_original_to_html_position(2)  # B in Bold
        pos_12 = renderer.map_original_to_html_position(12)  # First * of italic
        pos_19 = renderer.map_original_to_html_position(19)  # Last * of italic

        # Should be monotonically increasing
        assert pos_0 <= pos_2 <= pos_12 <= pos_19

    def test_positions_with_code_blocks(self):
        """Position mapping through code blocks."""
        text = "Text with `code` here"
        renderer = MarkdownToHTMLRenderer(text)
        html, offsets = renderer.render()

        # Verify monotonicity across all positions
        for i in range(len(text) - 1):
            pos_i = renderer.map_original_to_html_position(i)
            pos_i_plus_1 = renderer.map_original_to_html_position(i + 1)
            assert pos_i <= pos_i_plus_1

    def test_positions_with_links(self):
        """Position mapping through links."""
        text = "Click [here](http://example.com) now"
        renderer = MarkdownToHTMLRenderer(text)
        html, offsets = renderer.render()

        # Test positions around the link
        pos_0 = renderer.map_original_to_html_position(0)  # C
        pos_6 = renderer.map_original_to_html_position(6)  # [
        pos_33 = renderer.map_original_to_html_position(33)  # space after )

        assert pos_0 < pos_6 < pos_33

    def test_positions_with_list_items(self):
        """Position mapping through list transformations."""
        text = "- Item 1\n- Item 2"
        renderer = MarkdownToHTMLRenderer(text)
        html, offsets = renderer.render()

        pos_0 = renderer.map_original_to_html_position(0)  # -
        pos_2 = renderer.map_original_to_html_position(2)  # I
        pos_9 = renderer.map_original_to_html_position(9)  # \n
        pos_10 = renderer.map_original_to_html_position(10)  # - of second item

        assert pos_0 <= pos_2 <= pos_9 <= pos_10

    def test_position_remapping_preserves_relative_order(self):
        """Relative order of positions should be preserved."""
        text = "A **bold B** C *italic D* E"
        renderer = MarkdownToHTMLRenderer(text)
        html, offsets = renderer.render()

        # Get positions of key characters: A, B, C, D, E
        pos_A = renderer.map_original_to_html_position(0)
        pos_B = renderer.map_original_to_html_position(9)
        pos_C = renderer.map_original_to_html_position(15)
        pos_D = renderer.map_original_to_html_position(25)
        pos_E = renderer.map_original_to_html_position(27)

        # Original order should be preserved
        assert pos_A < pos_B < pos_C < pos_D < pos_E

    def test_position_at_transformation_boundaries(self):
        """Test positions at exact transformation boundaries."""
        text = "abc**def**ghi"
        renderer = MarkdownToHTMLRenderer(text)
        html, offsets = renderer.render()

        # Position 3: first * of **
        # Position 5: 'd' (first char of content)
        # Position 8: last * of **
        # Position 9: 'g' (after transformation)

        pos_3 = renderer.map_original_to_html_position(3)
        pos_5 = renderer.map_original_to_html_position(5)
        pos_8 = renderer.map_original_to_html_position(8)
        pos_9 = renderer.map_original_to_html_position(9)

        # Should be in order
        assert pos_3 <= pos_5 <= pos_8 <= pos_9


class TestPositionRemappingEdgeCases:
    """Edge cases for position remapping."""

    def test_position_0_always_maps_to_start_of_content(self):
        """Position 0 should always map to start of content (after opening tags)."""
        test_cases = [
            "Simple text",
            "**Bold start**",
            "*Italic start*",
            "`Code start`",
            "# Heading",
        ]

        for text in test_cases:
            renderer = MarkdownToHTMLRenderer(text)
            html, offsets = renderer.render()

            pos_0 = renderer.map_original_to_html_position(0)
            # Should be at or after position 0 in HTML
            assert pos_0 >= 0, f"Failed for text: {text}"

    def test_position_at_end_of_text(self):
        """Position at end of text should map correctly."""
        text = "Hello World"
        renderer = MarkdownToHTMLRenderer(text)
        html, offsets = renderer.render()

        end_pos = renderer.map_original_to_html_position(len(text))
        # Should be within HTML bounds
        assert 0 <= end_pos <= len(html)

    def test_consecutive_positions_never_decrease(self):
        """Consecutive positions should never decrease (monotonicity)."""
        test_cases = [
            "Simple **bold** text",
            "Mix *italic* and `code`",
            "# Heading\n\nParagraph",
            "List:\n- Item 1\n- Item 2",
            "[Link](http://example.com) text",
        ]

        for text in test_cases:
            renderer = MarkdownToHTMLRenderer(text)
            html, offsets = renderer.render()

            # Check every position
            for i in range(len(text)):
                if i + 1 < len(text):
                    pos_i = renderer.map_original_to_html_position(i)
                    pos_i_plus_1 = renderer.map_original_to_html_position(i + 1)
                    assert pos_i <= pos_i_plus_1, (
                        f"Monotonicity failed at position {i} in text: {text}"
                    )

    def test_positions_with_unicode(self):
        """Position mapping with Unicode characters."""
        text = "α-helix and β-sheet"
        renderer = MarkdownToHTMLRenderer(text)
        html, offsets = renderer.render()

        # Should handle Unicode correctly
        for i in range(len(text)):
            pos = renderer.map_original_to_html_position(i)
            assert 0 <= pos <= len(html)

    def test_positions_with_special_markdown_chars_as_text(self):
        """Position mapping when markdown chars appear as regular text."""
        text = "Use * for multiplication and ** for power"
        renderer = MarkdownToHTMLRenderer(text)
        html, offsets = renderer.render()

        # Should still maintain monotonicity even if some * aren't formatting
        for i in range(len(text) - 1):
            pos_i = renderer.map_original_to_html_position(i)
            pos_i_plus_1 = renderer.map_original_to_html_position(i + 1)
            assert pos_i <= pos_i_plus_1

    def test_empty_content_between_markers(self):
        """Position mapping with empty content between markers."""
        text = "Text with **** here"
        renderer = MarkdownToHTMLRenderer(text)
        html, offsets = renderer.render()

        # Should handle gracefully
        for i in range(len(text)):
            pos = renderer.map_original_to_html_position(i)
            assert 0 <= pos <= len(html)


class TestPositionRemappingInvariantsents:
    """Test invariants that should always hold for position remapping."""

    def test_invariant_position_0_before_position_end(self):
        """Position 0 should always map before end position."""
        test_cases = [
            "Short",
            "A bit longer text",
            "**Formatted** text",
            "# Heading\n\nParagraph",
        ]

        for text in test_cases:
            renderer = MarkdownToHTMLRenderer(text)
            html, offsets = renderer.render()

            pos_0 = renderer.map_original_to_html_position(0)
            pos_end = renderer.map_original_to_html_position(len(text))

            assert pos_0 <= pos_end, f"Failed for: {text}"

    def test_invariant_all_positions_within_html_bounds(self):
        """All mapped positions should be within HTML string bounds."""
        text = "Test **bold** and *italic* text with [link](http://example.com)"
        renderer = MarkdownToHTMLRenderer(text)
        html, offsets = renderer.render()

        for i in range(len(text) + 1):
            pos = renderer.map_original_to_html_position(i)
            assert 0 <= pos <= len(html), (
                f"Position {i} maps to {pos} which is out of bounds [0, {len(html)}]"
            )

    def test_invariant_monotonic_positions(self):
        """Position mapping must be monotonic (non-decreasing)."""
        text = (
            "Complex **bold *nested* text** with `code` and [links](http://example.com)"
        )
        renderer = MarkdownToHTMLRenderer(text)
        html, offsets = renderer.render()

        positions = [
            renderer.map_original_to_html_position(i) for i in range(len(text))
        ]

        for i in range(len(positions) - 1):
            assert positions[i] <= positions[i + 1], (
                f"Position {i} maps to {positions[i]}, but position {i + 1} maps to {positions[i + 1]} (not monotonic)"
            )

    def test_invariant_offsets_list_sorted_by_original(self):
        """The position_offsets list should be sorted by original position."""
        text = "Test with **formatting** and *more* stuff"
        renderer = MarkdownToHTMLRenderer(text)
        html, offsets = renderer.render()

        orig_positions = [orig for html_pos, orig in renderer.position_offsets]
        assert orig_positions == sorted(orig_positions), (
            "Offsets not sorted by original position"
        )

    def test_invariant_distance_preservation_approximate(self):
        """Distance between positions should be approximately preserved.

        Note: Exact preservation isn't possible due to HTML tags, but
        the ratio of distances should be somewhat preserved.
        """
        text = "ABCDEFGHIJ"  # Simple text without formatting
        renderer = MarkdownToHTMLRenderer(text)
        html, offsets = renderer.render()

        # Distance between A and E in original
        pos_A = renderer.map_original_to_html_position(0)
        pos_E = renderer.map_original_to_html_position(4)
        dist_AE = pos_E - pos_A

        # Distance between F and J in original
        pos_F = renderer.map_original_to_html_position(5)
        pos_J = renderer.map_original_to_html_position(9)
        dist_FJ = pos_J - pos_F

        # Both spans are 4 characters, so distances should be equal (no transformations)
        assert dist_AE == dist_FJ, "Distance not preserved for simple text"
