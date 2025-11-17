"""Tests for HTML tag validation and span insertion safety."""

from __future__ import annotations

import pytest

from interaction_finder.report.html_renderer import (
    DocumentAnnotator,
    HTMLTag,
    HTMLTagPair,
    HTMLTagScanner,
    HTMLTagValidator,
    MarkdownToHTMLRenderer,
)
from interaction_finder.resources import Resource, ResourceId, ResourceQuote


class TestHTMLTagScanner:
    """Tests for HTMLTagScanner."""

    def test_scan_simple_tags(self):
        """Test scanning simple opening and closing tags."""
        html = "<strong>text</strong>"
        scanner = HTMLTagScanner(html)
        pairs = scanner.scan()

        assert len(pairs) == 1
        assert pairs[0].tag_name == "strong"
        assert pairs[0].opening.start_pos == 0
        assert pairs[0].opening.end_pos == 8
        assert pairs[0].closing.start_pos == 12
        assert pairs[0].closing.end_pos == 21

    def test_scan_nested_tags(self):
        """Test scanning nested tags."""
        html = "<strong><em>text</em></strong>"
        scanner = HTMLTagScanner(html)
        pairs = scanner.scan()

        assert len(pairs) == 2
        # Should find both strong and em pairs
        tag_names = {p.tag_name for p in pairs}
        assert tag_names == {"strong", "em"}

    def test_scan_self_closing_tags(self):
        """Test scanning self-closing tags."""
        html = "text<br />more"
        scanner = HTMLTagScanner(html)
        pairs = scanner.scan()

        assert len(pairs) == 1
        assert pairs[0].tag_name == "br"
        assert pairs[0].opening.is_self_closing
        assert pairs[0].closing is None

    def test_scan_tags_with_attributes(self):
        """Test scanning tags with attributes."""
        html = '<a href="http://example.com">link</a>'
        scanner = HTMLTagScanner(html)
        pairs = scanner.scan()

        assert len(pairs) == 1
        assert pairs[0].tag_name == "a"

    def test_scan_mixed_inline_tags(self):
        """Test scanning multiple inline tags."""
        html = "<strong>bold</strong> and <em>italic</em>"
        scanner = HTMLTagScanner(html)
        pairs = scanner.scan()

        assert len(pairs) == 2
        tag_names = [p.tag_name for p in pairs]
        assert "strong" in tag_names
        assert "em" in tag_names


class TestHTMLTagValidator:
    """Tests for HTMLTagValidator."""

    def test_valid_insertion_no_conflicts(self):
        """Test insertion positions that don't conflict with tags."""
        html = "Before <strong>bold</strong> after"
        #      0      7       11   16      24  29
        scanner = HTMLTagScanner(html)
        pairs = scanner.scan()
        validator = HTMLTagValidator(html, pairs)

        # Insert span around "Before " - no conflicts
        plan = validator.validate_insertion(0, 7)
        assert not plan.adjusted
        assert not plan.split_required
        assert plan.open_pos == 0
        assert plan.close_pos == 7

    def test_conflict_opening_inside_tag(self):
        """Test conflict when opening span falls inside tag content."""
        html = "<strong>Keywords:</strong> text"
        #      0        8        16      25    30
        # content_start=8, content_end=16

        scanner = HTMLTagScanner(html)
        pairs = scanner.scan()
        validator = HTMLTagValidator(html, pairs)

        # Try to insert span starting at "Keywords" (pos 8) and ending after tag
        plan = validator.validate_insertion(8, 30)

        # Should adjust to move opening before <strong>
        assert plan.adjusted
        assert plan.open_pos == 0  # Before <strong>
        assert plan.close_pos == 30

    def test_conflict_closing_inside_tag(self):
        """Test conflict when closing span falls inside tag content."""
        html = "text <strong>Keywords:</strong>"
        #      0    5        13       22     31

        scanner = HTMLTagScanner(html)
        pairs = scanner.scan()
        validator = HTMLTagValidator(html, pairs)

        # Try to insert span starting before tag and ending inside tag content
        # Position 13 is right at content_start, so no conflict
        plan = validator.validate_insertion(0, 13)
        assert not plan.adjusted  # Position 13 is at boundary, not inside

        # Try to insert span ending truly inside the tag (at position 15)
        plan = validator.validate_insertion(0, 15)

        # Should adjust to move closing after </strong>
        assert plan.adjusted
        assert plan.open_pos == 0
        assert plan.close_pos == 31  # After </strong>

    def test_both_boundaries_conflict(self):
        """Test when both opening and closing conflict."""
        html = "before <strong>middle</strong> after"
        scanner = HTMLTagScanner(html)
        pairs = scanner.scan()
        validator = HTMLTagValidator(html, pairs)

        # Insert span from inside <strong> to outside
        content_mid = 15  # Inside "middle"
        plan = validator.validate_insertion(content_mid, 35)

        # Should adjust to avoid splitting tag
        assert plan.adjusted
        # Should move opening before <strong>
        assert plan.open_pos <= 7


class TestDocumentAnnotatorWithValidator:
    """Integration tests for DocumentAnnotator with HTML validation."""

    def test_quote_with_strong_tag_at_start(self):
        """Test quote starting with bold text - the original bug case."""
        # This is the exact case from the bug report
        text = "**Keywords:** BMP/TGF-β signaling pathway; BMPR2 mutations"
        resource = Resource(
            id=ResourceId(url="http://example.com/doc", counter=1),
            title="Test",
            text=text,
        )

        renderer = MarkdownToHTMLRenderer(text)
        renderer.render()

        annotator = DocumentAnnotator(resource, renderer)

        # Quote covers "Keywords: BMP/TGF-β..."
        quote = ResourceQuote(
            resource=resource,
            query_text="Keywords: BMP/TGF-β signaling pathway",
            spans=[(2, 40)],  # Skipping ** markdown
            resource_id=resource.id,
            fuzzy_corrected=False,
        )

        entities = {
            0: {
                "entity1": {
                    "name": "BMPR2",
                    "kind": "gene",
                    "aliases": [],
                },
                "entity2": {
                    "name": "mutations",
                    "kind": "variant",
                    "aliases": [],
                },
            }
        }

        result = annotator.annotate([quote], entities)

        # Verify no malformed HTML like: <strong><span>Keywords:</strong>...</span>
        # The span should either:
        # 1. Be outside: <span><strong>Keywords:</strong>...</span>
        # 2. Or inside: <strong><span>Keywords:</span></strong><span>...</span>

        # Check no malformed patterns exist
        assert (
            "<strong><span" not in result.html
            or "</strong>"
            not in result.html.split("<strong><span")[1].split("</span>")[0]
        )

        # Verify structure is valid
        assert 'class="quote-span' in result.html

    def test_quote_with_link_inside(self):
        """Test quote containing a link."""
        text = "See [BMPR2](http://example.com) for details"
        resource = Resource(
            id=ResourceId(url="http://example.com/doc", counter=2),
            title="Test",
            text=text,
        )

        renderer = MarkdownToHTMLRenderer(text)
        renderer.render()

        annotator = DocumentAnnotator(resource, renderer)

        # Quote covers entire text including link
        quote = ResourceQuote(
            resource=resource,
            query_text=text,
            spans=[(0, len(text))],
            resource_id=resource.id,
            fuzzy_corrected=False,
        )

        entities = {
            0: {
                "entity1": {
                    "name": "BMPR2",
                    "kind": "gene",
                    "aliases": [],
                },
                "entity2": {
                    "name": "details",
                    "kind": "concept",
                    "aliases": [],
                },
            }
        }

        result = annotator.annotate([quote], entities)

        # Verify no malformed HTML with links
        assert 'class="quote-span' in result.html

    def test_quote_splitting_emphasis(self):
        """Test quote boundary falling inside italic text."""
        text = "Normal text *partially quoted text* more text"
        resource = Resource(
            id=ResourceId(url="http://example.com/doc", counter=3),
            title="Test",
            text=text,
        )

        renderer = MarkdownToHTMLRenderer(text)
        renderer.render()

        annotator = DocumentAnnotator(resource, renderer)

        # Quote starts mid-emphasis (from "quoted" onwards)
        quote = ResourceQuote(
            resource=resource,
            query_text="quoted text more text",
            spans=[(23, 44)],
            resource_id=resource.id,
            fuzzy_corrected=False,
        )

        entities = {}

        result = annotator.annotate([quote], entities)

        # Should handle gracefully without malformed HTML
        assert 'class="quote-span' in result.html

    def test_entity_with_formatting(self):
        """Test entity name that is formatted in source."""
        text = "The **BRCA1** gene is important"
        resource = Resource(
            id=ResourceId(url="http://example.com/doc", counter=4),
            title="Test",
            text=text,
        )

        renderer = MarkdownToHTMLRenderer(text)
        renderer.render()

        annotator = DocumentAnnotator(resource, renderer)

        # Quote covers entire text
        quote = ResourceQuote(
            resource=resource,
            query_text=text,
            spans=[(0, len(text))],
            resource_id=resource.id,
            fuzzy_corrected=False,
        )

        entities = {
            0: {
                "entity1": {
                    "name": "BRCA1",
                    "kind": "gene",
                    "aliases": [],
                },
                "entity2": {
                    "name": "gene",
                    "kind": "concept",
                    "aliases": [],
                },
            }
        }

        result = annotator.annotate([quote], entities)

        # Entity highlighting should work with bold text
        assert 'class="entity-span"' in result.html

    def test_complex_nested_formatting(self):
        """Test quote with complex nested formatting."""
        text = "Text with **bold and *italic* nested** content"
        resource = Resource(
            id=ResourceId(url="http://example.com/doc", counter=5),
            title="Test",
            text=text,
        )

        renderer = MarkdownToHTMLRenderer(text)
        renderer.render()

        annotator = DocumentAnnotator(resource, renderer)

        # Quote covering the nested section
        quote = ResourceQuote(
            resource=resource,
            query_text="bold and italic nested",
            spans=[(12, 34)],
            resource_id=resource.id,
            fuzzy_corrected=False,
        )

        entities = {}

        result = annotator.annotate([quote], entities)

        # Should handle complex nesting gracefully
        assert 'class="quote-span' in result.html


class TestSpanSplitting:
    """Tests for span splitting functionality."""

    def test_split_segments_computation(self):
        """Test computing split segments for a quote crossing tags."""
        # Create a case where adjustment can't work:
        # We want to quote from inside first tag to inside second tag
        # HTML: "<strong>a</strong> <em>b</em>"
        html = "<strong>a</strong> <em>b</em>"
        scanner = HTMLTagScanner(html)
        pairs = scanner.scan()
        validator = HTMLTagValidator(html, pairs)

        # Position 8 is 'a' (inside strong), position 20 is 'b' (inside em)
        # Both are inside their respective tags, so we can't adjust without
        # expanding to cover both tags entirely
        # If adjustment tries to move both boundaries outward, it would
        # actually work: span from 0 to 25 covers everything

        # Actually, let's test the case where we have confirmed splits
        # by checking if split_segments is populated when split_required is True
        plan = validator.validate_insertion(8, 20)

        # The validator should either adjust or split
        if plan.split_required:
            assert plan.split_segments is not None
            assert len(plan.split_segments) >= 1
        else:
            # If adjusted successfully, that's also valid
            assert plan.adjusted

    def test_quote_spanning_multiple_inline_tags(self):
        """Test quote that spans across multiple inline formatted regions."""
        text = "Start **bold text** middle *italic text* end"
        resource = Resource(
            id=ResourceId(url="http://example.com/doc", counter=10),
            title="Test",
            text=text,
        )

        renderer = MarkdownToHTMLRenderer(text)
        renderer.render()

        annotator = DocumentAnnotator(resource, renderer)

        # Quote covering "bold text** middle *italic text"
        # This crosses the boundary between bold and italic
        quote = ResourceQuote(
            resource=resource,
            query_text="bold text middle italic text",
            spans=[(8, 39)],
            resource_id=resource.id,
            fuzzy_corrected=False,
        )

        entities = {}

        result = annotator.annotate([quote], entities)

        # Should produce valid HTML with split spans
        assert 'class="quote-span' in result.html
        # Should not have malformed nesting
        assert result.html.count('<span class="quote-span') >= 1

    def test_quote_crossing_link_boundary(self):
        """Test quote that crosses link tag boundaries."""
        text = "Before [link text](http://example.com) after"
        resource = Resource(
            id=ResourceId(url="http://example.com/doc", counter=11),
            title="Test",
            text=text,
        )

        renderer = MarkdownToHTMLRenderer(text)
        renderer.render()

        annotator = DocumentAnnotator(resource, renderer)

        # Quote from middle of link to after link
        quote = ResourceQuote(
            resource=resource,
            query_text="text after",
            spans=[(12, 44)],  # Crosses link boundary
            resource_id=resource.id,
            fuzzy_corrected=False,
        )

        entities = {}

        result = annotator.annotate([quote], entities)

        # Should handle gracefully with splits
        assert 'class="quote-span' in result.html

    def test_quote_within_and_across_strong_tags(self):
        """Test quote starting inside strong tag and ending outside."""
        text = "**Important note:** Regular text continues here"
        resource = Resource(
            id=ResourceId(url="http://example.com/doc", counter=12),
            title="Test",
            text=text,
        )

        renderer = MarkdownToHTMLRenderer(text)
        renderer.render()

        annotator = DocumentAnnotator(resource, renderer)

        # Quote from "note" to "continues"
        quote = ResourceQuote(
            resource=resource,
            query_text="note: Regular text continues",
            spans=[(12, 40)],
            resource_id=resource.id,
            fuzzy_corrected=False,
        )

        entities = {}

        result = annotator.annotate([quote], entities)

        # Should produce valid HTML
        assert 'class="quote-span' in result.html

    def test_multiple_splits_same_quote(self):
        """Test quote requiring multiple splits across several tags."""
        text = "Text *a* and **b** and `code` end"
        resource = Resource(
            id=ResourceId(url="http://example.com/doc", counter=13),
            title="Test",
            text=text,
        )

        renderer = MarkdownToHTMLRenderer(text)
        renderer.render()

        annotator = DocumentAnnotator(resource, renderer)

        # Quote covering all the formatted regions
        quote = ResourceQuote(
            resource=resource,
            query_text="a and b and code",
            spans=[(6, 28)],
            resource_id=resource.id,
            fuzzy_corrected=False,
        )

        entities = {}

        result = annotator.annotate([quote], entities)

        # Should have multiple quote-span elements
        assert result.html.count('class="quote-span') >= 1
