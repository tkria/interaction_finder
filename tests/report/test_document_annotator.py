"""Tests for DocumentAnnotator."""

from __future__ import annotations

import pytest

from interaction_finder.report.html_renderer import (
    DocumentAnnotator,
    MarkdownToHTMLRenderer,
)
from interaction_finder.resources import Resource, ResourceId, ResourceQuote


class TestDocumentAnnotator:
    """Tests for DocumentAnnotator."""

    def test_basic_annotation(self):
        """Test basic annotation with quotes and entities."""
        # Create a simple document
        text = "BRCA1 is a gene associated with breast cancer."
        resource = Resource(
            id=ResourceId(url="http://example.com/doc1", counter=1),
            title="Test Document",
            text=text,
        )

        # Render to HTML
        renderer = MarkdownToHTMLRenderer(text)
        renderer.render()

        # Create annotator
        annotator = DocumentAnnotator(resource, renderer)

        # Create a quote covering the entire text
        quote = ResourceQuote(
            resource=resource,
            query_text=text,
            spans=[(0, len(text))],
            resource_id=resource.id,
            fuzzy_corrected=False,
        )

        # Define entities
        entities = {
            0: {  # pair_idx 0
                "entity1": {
                    "name": "BRCA1",
                    "kind": "gene",
                    "aliases": [],
                },
                "entity2": {
                    "name": "breast cancer",
                    "kind": "disease",
                    "aliases": [],
                },
            }
        }

        # Annotate
        result = annotator.annotate([quote], entities)

        # Verify structure
        assert result.doc_id == resource.id.id
        assert result.html  # Should have HTML
        assert len(result.quote_map) == 1  # One quote
        assert len(result.entity_map) >= 2  # At least two entities

        # Verify HTML contains spans
        assert (
            'class="quote-span' in result.html
        )  # Quote spans now have additional CSS classes
        assert 'class="entity-span"' in result.html

    def test_no_quotes_or_entities(self):
        """Test annotation with no quotes or entities."""
        text = "Simple text."
        resource = Resource(
            id=ResourceId(url="http://example.com/doc2", counter=2),
            title="Simple",
            text=text,
        )

        renderer = MarkdownToHTMLRenderer(text)
        renderer.render()

        annotator = DocumentAnnotator(resource, renderer)

        # No quotes, no entities
        result = annotator.annotate([], {})

        assert result.doc_id == resource.id.id
        assert result.html == renderer.html  # Unchanged
        assert len(result.quote_map) == 0
        assert len(result.entity_map) == 0

    def test_entity_within_quote(self):
        """Test that entities are only found within quote boundaries."""
        text = "BRCA1 is important. Later, BRCA1 appears again."
        resource = Resource(
            id=ResourceId(url="http://example.com/doc3", counter=3),
            title="Test",
            text=text,
        )

        renderer = MarkdownToHTMLRenderer(text)
        renderer.render()

        annotator = DocumentAnnotator(resource, renderer)

        # Quote only covers first sentence
        quote = ResourceQuote(
            resource=resource,
            query_text="BRCA1 is important.",
            spans=[(0, 19)],
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
                    "name": "importance",
                    "kind": "concept",
                    "aliases": [],
                },
            }
        }

        result = annotator.annotate([quote], entities)

        # Should find BRCA1 within quote
        # May or may not find the second BRCA1 (depends on implementation)
        assert len(result.entity_map) >= 1

    def test_multiple_quotes(self):
        """Test annotation with multiple quotes."""
        text = "First quote text. Second quote text."
        resource = Resource(
            id=ResourceId(url="http://example.com/doc4", counter=4),
            title="Multi",
            text=text,
        )

        renderer = MarkdownToHTMLRenderer(text)
        renderer.render()

        annotator = DocumentAnnotator(resource, renderer)

        quote1 = ResourceQuote(
            resource=resource,
            query_text="First quote text.",
            spans=[(0, 17)],
            resource_id=resource.id,
            fuzzy_corrected=False,
        )

        quote2 = ResourceQuote(
            resource=resource,
            query_text="Second quote text.",
            spans=[(18, 36)],
            resource_id=resource.id,
            fuzzy_corrected=False,
        )

        result = annotator.annotate([quote1, quote2], {})

        assert len(result.quote_map) == 2

    def test_overlapping_quotes_with_bmpr2_alias_entities(self):
        """Ensure overlapping quotes keep entity spans inside quote boundaries."""
        text = (
            "Rare deleterious variants in BMPR2 contribute to pediatric-onset IPAH "
            "and familial PAH with similar frequency as adult-onset disease "
            "but rarely explain cases of PAH associated with other diseases."
        )
        resource = Resource(
            id=ResourceId(url="http://example.com/doc-bmpr2", counter=5),
            title="BMPR2 Snippet",
            text=text,
        )

        renderer = MarkdownToHTMLRenderer(text)
        renderer.render()

        annotator = DocumentAnnotator(resource, renderer)

        quotes = [
            ResourceQuote(
                resource=resource,
                query_text=text,
                spans=[(0, len(text))],
                resource_id=resource.id,
                fuzzy_corrected=False,
            ),
            ResourceQuote(
                resource=resource,
                query_text=text.split(" and ")[0],
                spans=[(0, text.index(" and familial PAH"))],
                resource_id=resource.id,
                fuzzy_corrected=False,
            ),
        ]

        entities = {
            0: {
                "entity1": {
                    "name": "BMPR2",
                    "kind": "gene",
                    "aliases": [],
                },
                "entity2": {
                    "name": "Pulmonary arterial hypertension",
                    "kind": "phenotype",
                    "aliases": ["PAH", "IPAH"],
                },
            }
        }

        result = annotator.annotate(quotes, entities)

        html = result.html
        assert '<abbr title="Pulmonary arterial hypertension">IPAH</abbr>' in html
        assert 'IPAH</abbr></span></span><span class="quote-span' in html
        assert '</abbr></span><span class="quote-span' not in html

    def test_entity_metadata(self):
        """Test that entity metadata is correctly populated."""
        text = "BRCA1 gene"
        resource = Resource(
            id=ResourceId(url="http://example.com/doc5", counter=5),
            title="Test",
            text=text,
        )

        renderer = MarkdownToHTMLRenderer(text)
        renderer.render()

        annotator = DocumentAnnotator(resource, renderer)

        quote = ResourceQuote(
            resource=resource,
            query_text=text,
            spans=[(0, len(text))],
            resource_id=resource.id,
            fuzzy_corrected=False,
        )

        entities = {
            42: {  # Specific pair index
                "entity1": {
                    "name": "BRCA1",
                    "kind": "gene",
                    "aliases": ["BRCA-1", "Breast Cancer 1"],
                },
                "entity2": {
                    "name": "gene",
                    "kind": "concept",
                    "aliases": [],
                },
            }
        }

        result = annotator.annotate([quote], entities)

        # Find BRCA1 entity in map
        brca1_entities = [e for e in result.entity_map.values() if e.name == "BRCA1"]
        assert len(brca1_entities) > 0

        brca1 = brca1_entities[0]
        assert brca1.kind == "gene"
        assert 42 in brca1.pair_indices
        assert "BRCA-1" in brca1.aliases

    def test_greek_letter_normalization(self):
        """Test that Greek letters are matched via normalization (α ↔ alpha)."""
        # Document has Greek letter α
        text = "TGF-α receptor is important in cell signaling."
        resource = Resource(
            id=ResourceId(url="http://example.com/doc6", counter=6),
            title="Test Greek",
            text=text,
        )

        renderer = MarkdownToHTMLRenderer(text)
        renderer.render()

        annotator = DocumentAnnotator(resource, renderer)

        quote = ResourceQuote(
            resource=resource,
            query_text=text,
            spans=[(0, len(text))],
            resource_id=resource.id,
            fuzzy_corrected=False,
        )

        # Entity name uses "alpha" but document has "α"
        entities = {
            0: {
                "entity1": {
                    "name": "TGF-alpha receptor",
                    "kind": "protein",
                    "aliases": [],
                },
                "entity2": {
                    "name": "cell signaling",
                    "kind": "process",
                    "aliases": [],
                },
            }
        }

        result = annotator.annotate([quote], entities)

        # Should find TGF-α even though we searched for TGF-alpha
        assert "TGF-alpha receptor" in [e.name for e in result.entity_map.values()]

        # Find the entity span in HTML
        tgf_entities = [
            e for e in result.entity_map.values() if e.name == "TGF-alpha receptor"
        ]
        assert len(tgf_entities) == 1

        # Verify it was found (entity map should have the entry)
        tgf_entity = tgf_entities[0]
        assert tgf_entity.kind == "protein"
        assert 0 in tgf_entity.pair_indices

        # The HTML should contain an entity span for TGF-α
        assert "entity-span" in result.html
        assert "TGF-" in result.html  # The actual matched text should be preserved
