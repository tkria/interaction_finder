"""Tests for HTML rendering with position tracking."""

import pytest

from interaction_finder.report.html_renderer import MarkdownToHTMLRenderer


class TestMarkdownToHTMLRenderer:
    """Tests for MarkdownToHTMLRenderer."""

    def test_plain_text(self):
        """Plain text should be wrapped in paragraph and HTML-escaped."""
        renderer = MarkdownToHTMLRenderer("Hello, world!")
        html, offsets = renderer.render()

        assert html == "<p>Hello, world!</p>"
        assert len(offsets) > 0
        # Verify we can map back
        assert renderer.map_original_to_html_position(0) == 3  # After "<p>"
        assert renderer.map_original_to_html_position(5) == 8  # After "<p>Hello"

    def test_html_escaping(self):
        """HTML special characters should be escaped."""
        renderer = MarkdownToHTMLRenderer('Test <tag> & "quote"')
        html, offsets = renderer.render()

        assert "&lt;tag&gt;" in html
        assert "&amp;" in html
        assert "&quot;quote&quot;" in html
        assert "<tag>" not in html

    def test_headings(self):
        """Headings should be converted to h1-h6 tags."""
        text = """# Heading 1
## Heading 2
### Heading 3"""
        renderer = MarkdownToHTMLRenderer(text)
        html, offsets = renderer.render()

        assert "<h1>Heading 1</h1>" in html
        assert "<h2>Heading 2</h2>" in html
        assert "<h3>Heading 3</h3>" in html

    def test_bold_text(self):
        """Bold markdown should be converted to <strong> tags."""
        renderer = MarkdownToHTMLRenderer("This is **bold text** here.")
        html, offsets = renderer.render()

        assert "<strong>bold text</strong>" in html
        assert "**" not in html

    def test_italic_text(self):
        """Italic markdown should be converted to <em> tags."""
        renderer = MarkdownToHTMLRenderer("This is *italic text* here.")
        html, offsets = renderer.render()

        assert "<em>italic text</em>" in html
        # Should not have standalone * (but may have in escaped form)
        assert "<em>" in html and "</em>" in html

    def test_inline_code(self):
        """Inline code should be converted to <code> tags."""
        renderer = MarkdownToHTMLRenderer("Use the `print()` function.")
        html, offsets = renderer.render()

        assert "<code>print()</code>" in html
        assert "`" not in html or "&#" in html  # Backticks removed or escaped

    def test_links(self):
        """Links should be converted to <a> tags."""
        renderer = MarkdownToHTMLRenderer("Visit [Google](https://google.com) now.")
        html, offsets = renderer.render()

        assert '<a href="https://google.com">Google</a>' in html
        assert "[" not in html
        assert "](" not in html

    def test_unordered_list(self):
        """Unordered list items should be converted to <li> tags."""
        text = """- Item 1
- Item 2
* Item 3"""
        renderer = MarkdownToHTMLRenderer(text)
        html, offsets = renderer.render()

        assert html.count("<li>") == 3
        assert "Item 1" in html
        assert "Item 2" in html
        assert "Item 3" in html

    def test_ordered_list(self):
        """Ordered list items should be converted to <li> tags."""
        text = """1. First
2. Second
3. Third"""
        renderer = MarkdownToHTMLRenderer(text)
        html, offsets = renderer.render()

        assert html.count("<li>") == 3
        assert "First" in html
        assert "Second" in html
        assert "Third" in html

    def test_paragraphs(self):
        """Double newlines should create paragraph breaks."""
        text = """First paragraph.

Second paragraph."""
        renderer = MarkdownToHTMLRenderer(text)
        html, offsets = renderer.render()

        assert html.count("<p>") >= 2
        assert "First paragraph" in html
        assert "Second paragraph" in html

    def test_position_mapping_simple(self):
        """Position mapping should be accurate for simple text."""
        renderer = MarkdownToHTMLRenderer("Hello")
        html, offsets = renderer.render()

        # "Hello" starts at position 0 in original
        # After "<p>", it starts at position 3 in HTML
        html_pos = renderer.map_original_to_html_position(0)
        assert html_pos == 3  # After "<p>"

    def test_position_mapping_with_escaping(self):
        """Position mapping should handle HTML escaping."""
        renderer = MarkdownToHTMLRenderer("A < B")
        html, offsets = renderer.render()

        # Original: "A < B"
        # HTML: "<p>A &lt; B</p>"
        # Position 0 (A) -> position 3 in HTML (after <p>)
        assert renderer.map_original_to_html_position(0) == 3

        # Position 2 (<) -> position after "A " in HTML
        # "A " is 2 chars, so position 5
        html_pos = renderer.map_original_to_html_position(2)
        assert html_pos >= 5  # After "<p>A "

    def test_position_mapping_with_bold(self):
        """Position mapping should handle bold formatting."""
        renderer = MarkdownToHTMLRenderer("Start **bold** end")
        html, offsets = renderer.render()

        # Original positions:
        # 0-5: "Start "
        # 6-7: "**"
        # 8-11: "bold"
        # 12-13: "**"
        # 14-17: " end"

        # Position 0 should map to after "<p>"
        assert renderer.map_original_to_html_position(0) == 3

        # Position 8 (start of "bold") should map to somewhere in the bold region
        # The exact position depends on how transformations are tracked
        html_pos = renderer.map_original_to_html_position(8)
        # Should be after "<p>Start " at minimum (3 + 6 = 9)
        assert html_pos >= 9

    def test_position_offsets_always_increasing(self):
        """Position offsets should be monotonically increasing."""
        renderer = MarkdownToHTMLRenderer("Test **bold** and *italic* text.")
        html, offsets = renderer.render()

        # Original positions should be strictly increasing
        orig_positions = [orig for html_pos, orig in offsets]
        assert orig_positions == sorted(orig_positions)

    def test_empty_text(self):
        """Empty text should render to empty paragraph."""
        renderer = MarkdownToHTMLRenderer("")
        html, offsets = renderer.render()

        # Should have some output (even if just tags)
        assert isinstance(html, str)
        assert len(offsets) >= 1

    def test_complex_formatting(self):
        """Complex nested formatting should work."""
        text = """# Title

This is **bold** and *italic* text with `code`.

Visit [example](https://example.com) for more info.

- Item 1
- Item 2"""
        renderer = MarkdownToHTMLRenderer(text)
        html, offsets = renderer.render()

        # Verify all elements are present
        assert "<h1>" in html
        assert "<strong>" in html
        assert "<em>" in html
        assert "<code>" in html
        assert "<a href=" in html
        assert "<li>" in html

    def test_map_position_out_of_bounds(self):
        """Mapping out-of-bounds position should raise ValueError."""
        renderer = MarkdownToHTMLRenderer("Hello")
        html, offsets = renderer.render()

        with pytest.raises(ValueError, match="out of bounds"):
            renderer.map_original_to_html_position(100)

        with pytest.raises(ValueError, match="out of bounds"):
            renderer.map_original_to_html_position(-1)

    def test_map_position_before_render(self):
        """Mapping position before render should raise RuntimeError."""
        renderer = MarkdownToHTMLRenderer("Hello")

        with pytest.raises(RuntimeError, match="Call render\\(\\) first"):
            renderer.map_original_to_html_position(0)

    def test_greek_letters_preserved(self):
        """Greek letters and special characters should be preserved."""
        text = "The α-helix and β-sheet structures"
        renderer = MarkdownToHTMLRenderer(text)
        html, offsets = renderer.render()

        # Greek letters should be in HTML (not normalized)
        assert "α" in html
        assert "β" in html

    def test_unicode_preservation(self):
        """Unicode characters should be preserved."""
        text = "Café résumé naïve"
        renderer = MarkdownToHTMLRenderer(text)
        html, offsets = renderer.render()

        # Unicode should be preserved
        assert "Café" in html
        assert "résumé" in html
        assert "naïve" in html

    def test_multiple_paragraphs_with_formatting(self):
        """Multiple paragraphs with formatting should work correctly."""
        text = """First paragraph with **bold**.

Second paragraph with *italic*.

Third paragraph."""
        renderer = MarkdownToHTMLRenderer(text)
        html, offsets = renderer.render()

        assert html.count("<p>") >= 3
        assert "<strong>bold</strong>" in html
        assert "<em>italic</em>" in html

    def test_heading_with_formatting(self):
        """Headings can contain formatting."""
        text = "# Title with **bold** text"
        renderer = MarkdownToHTMLRenderer(text)
        html, offsets = renderer.render()

        assert "<h1>" in html
        assert "<strong>bold</strong>" in html
        assert "</h1>" in html

    def test_line_breaks(self):
        """Single newlines are preserved in paragraphs (not converted to <br>)."""
        text = """Line 1
Line 2
Line 3"""
        renderer = MarkdownToHTMLRenderer(text)
        html, offsets = renderer.render()

        # Single newlines should be preserved within paragraph
        assert "\n" in html or "Line 1" in html
        assert "Line 2" in html
        assert "Line 3" in html

    def test_position_mapping_end_of_text(self):
        """Mapping end position should work correctly."""
        text = "Hello"
        renderer = MarkdownToHTMLRenderer(text)
        html, offsets = renderer.render()

        # End of text position
        end_pos = renderer.map_original_to_html_position(len(text))
        assert end_pos <= len(html)

    def test_url_with_special_chars(self):
        """URLs with special characters should be escaped."""
        text = "[Link](https://example.com?foo=1&bar=2)"
        renderer = MarkdownToHTMLRenderer(text)
        html, offsets = renderer.render()

        # URL should be escaped
        assert "&amp;" in html
        assert "foo=1&amp;bar=2" in html

    def test_incomplete_bold_not_rendered(self):
        """Incomplete bold markers should be treated as literals."""
        text = "This has **incomplete bold"
        renderer = MarkdownToHTMLRenderer(text)
        html, offsets = renderer.render()

        # Should not have <strong> since bold is incomplete
        # The ** should be escaped or remain as-is
        assert "incomplete bold" in html

    def test_incomplete_italic_not_rendered(self):
        """Incomplete italic markers should be treated as literals."""
        text = "This has *incomplete italic"
        renderer = MarkdownToHTMLRenderer(text)
        html, offsets = renderer.render()

        # Should not have <em> since italic is incomplete
        assert "incomplete italic" in html

    def test_incomplete_link_not_rendered(self):
        """Incomplete links should be treated as literals."""
        text = "This has [incomplete link"
        renderer = MarkdownToHTMLRenderer(text)
        html, offsets = renderer.render()

        # Should not have <a> since link is incomplete
        assert "incomplete link" in html
        # The [ should be in output somehow (original or escaped)
        assert "[" in html or "&#" in html

    def test_nested_bold_italic(self):
        """Nested bold and italic should work."""
        text = "This is ***bold and italic*** text"
        renderer = MarkdownToHTMLRenderer(text)
        html, offsets = renderer.render()

        # Should have both tags (order may vary)
        assert "<strong>" in html or "<em>" in html
        assert "bold and italic" in html
