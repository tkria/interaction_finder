"""Tests for reasoning sidebar template rendering."""

import pytest
from interaction_finder.report.reasoning_renderer import (
    EntityHighlighter,
    ReasoningTemplateRenderer,
    render_all_reasoning_templates,
)


class TestEntityHighlighter:
    """Tests for EntityHighlighter class."""

    def test_basic_highlighting(self):
        """Test basic entity highlighting."""
        highlighter = EntityHighlighter(
            entity1_terms=["BMPR2"],
            entity2_terms=["PAH"],
            entity1_name="BMPR2",
            entity2_name="pulmonary arterial hypertension",
        )

        text = "Mutations in BMPR2 cause PAH in many patients."
        result = highlighter.highlight(text)

        # Check that entities are wrapped in spans
        assert '<span class="entity-highlight entity1"' in result
        assert '<span class="entity-highlight entity2"' in result
        assert "BMPR2" in result
        assert "PAH" in result

    def test_html_escaping(self):
        """Test that HTML is properly escaped."""
        highlighter = EntityHighlighter(
            entity1_terms=["Gene<X>"],
            entity2_terms=["Disease&Y"],
            entity1_name="Gene<X>",
            entity2_name="Disease&Y",
        )

        text = "Gene<X> causes Disease&Y"
        result = highlighter.highlight(text)

        # Check that special characters are escaped
        assert "&lt;" in result  # <
        assert "&gt;" in result  # >
        assert "&amp;" in result  # &
        # But entity markup should still be present
        assert '<span class="entity-highlight' in result

    def test_longest_match_first(self):
        """Test that longest matches are prioritized."""
        highlighter = EntityHighlighter(
            entity1_terms=["BRCA", "BRCA1"],
            entity2_terms=["cancer"],
            entity1_name="BRCA1",
            entity2_name="cancer",
        )

        text = "BRCA1 mutations cause cancer"
        result = highlighter.highlight(text)

        # BRCA1 should be matched as a whole, not BRCA + 1
        assert "BRCA1" in result
        # Should have two entity highlights (BRCA1 and cancer)
        assert result.count('<span class="entity-highlight entity1"') == 1
        assert result.count('<span class="entity-highlight entity2"') == 1

    def test_case_insensitive_matching(self):
        """Test that matching is case-insensitive."""
        highlighter = EntityHighlighter(
            entity1_terms=["bmpr2"],
            entity2_terms=["pah"],
            entity1_name="BMPR2",
            entity2_name="PAH",
        )

        text = "BMPR2 and Bmpr2 and bmpr2 cause PAH and pah."
        result = highlighter.highlight(text)

        # All variations should be highlighted
        assert result.count('<span class="entity-highlight entity1"') == 3
        assert result.count('<span class="entity-highlight entity2"') == 2

    def test_title_attribute(self):
        """Test that canonical names appear in title attributes."""
        highlighter = EntityHighlighter(
            entity1_terms=["BMPR-II", "BMPR2"],
            entity2_terms=["PAH"],
            entity1_name="BMPR2",
            entity2_name="pulmonary arterial hypertension",
        )

        text = "BMPR-II causes PAH"
        result = highlighter.highlight(text)

        # Check title attributes
        assert 'title="BMPR2"' in result
        assert 'title="pulmonary arterial hypertension"' in result


class TestReasoningTemplateRenderer:
    """Tests for ReasoningTemplateRenderer class."""

    def test_render_overall_template(self):
        """Test rendering overall pair reasoning template."""
        pair = {
            "entity1": {
                "name": "BMPR2",
                "kind": "gene",
                "aliases": ["BMPR-II"],
            },
            "entity2": {
                "name": "pulmonary arterial hypertension",
                "kind": "disease",
                "aliases": ["PAH"],
            },
            "relationship": "causative mutation in",
            "confidence": "high",
            "reasoning": "BMPR2 mutations cause PAH in multiple studies.",
            "assessments": [],
        }

        renderer = ReasoningTemplateRenderer(pair, pair_idx=0)
        result = renderer.render_overall_template()

        # Check structure
        assert '<div class="reasoning-panel">' in result
        assert '<div class="reasoning-title">Overall Assessment</div>' in result
        assert "BMPR2" in result
        assert "pulmonary arterial hypertension" in result
        assert "causative mutation in" in result
        assert "confidence-high" in result
        # Check entity highlighting
        assert '<span class="entity-highlight' in result

    def test_render_assessment_template(self):
        """Test rendering document-specific assessment template."""
        pair = {
            "entity1": {
                "name": "BMPR2",
                "kind": "gene",
                "aliases": [],
            },
            "entity2": {
                "name": "PAH",
                "kind": "disease",
                "aliases": [],
            },
            "relationship": "causative mutation in",
            "confidence": "high",
            "reasoning": "Overall reasoning",
            "assessments": [],
        }

        assess = {
            "resource_id": "doc-123",
            "title": "BMPR2 and PAH Study",
            "reasoning": "This document shows BMPR2 causes PAH.",
            "relationship": "causative mutation in",
            "confidence": "high",
            "quotes": [{"text": "BMPR2 mutations lead to PAH", "spans": [[0, 28]]}],
        }

        renderer = ReasoningTemplateRenderer(pair, pair_idx=0)
        result = renderer.render_assessment_template(
            assess, assess_idx=0, all_pairs=[pair]
        )

        # Check structure
        assert '<div class="reasoning-panel">' in result
        assert '<div class="reasoning-title">Document Assessment</div>' in result
        assert "BMPR2 and PAH Study" in result
        assert "causative mutation in" in result
        # Check quote navigation
        assert '<div class="quote-navigation">' in result
        assert "Jump to Quotes (1)" in result
        assert "BMPR2 mutations lead to PAH" in result

    def test_render_other_pairs_navigation(self):
        """Test rendering other pairs navigation."""
        pair1 = {
            "entity1": {"name": "BMPR2", "kind": "gene", "aliases": []},
            "entity2": {"name": "PAH", "kind": "disease", "aliases": []},
            "relationship": "causative mutation in",
            "confidence": "high",
            "reasoning": "",
            "assessments": [{"resource_id": "doc-123", "quotes": []}],
        }

        pair2 = {
            "entity1": {"name": "SMAD9", "kind": "gene", "aliases": []},
            "entity2": {"name": "PAH", "kind": "disease", "aliases": []},
            "relationship": "associated with",
            "confidence": "medium",
            "reasoning": "",
            "assessments": [{"resource_id": "doc-123", "quotes": []}],  # Same doc
        }

        assess = {
            "resource_id": "doc-123",
            "title": "Multi-gene PAH study",
            "reasoning": "Test",
            "relationship": "causative mutation in",
            "confidence": "high",
            "quotes": [],
        }

        renderer = ReasoningTemplateRenderer(pair1, pair_idx=0)
        result = renderer.render_assessment_template(
            assess, assess_idx=0, all_pairs=[pair1, pair2]
        )

        # Should have other pairs navigation
        assert "Other Pairs (1)" in result
        assert "SMAD9 PAH" in result
        assert 'onclick="selectPairAndDocument(1,' in result


class TestRenderAllReasoningTemplates:
    """Tests for render_all_reasoning_templates function."""

    def test_render_all_templates(self):
        """Test rendering templates for all pairs."""
        pairs = [
            {
                "entity1": {"name": "BMPR2", "kind": "gene", "aliases": []},
                "entity2": {"name": "PAH", "kind": "disease", "aliases": []},
                "relationship": "causative mutation in",
                "confidence": "high",
                "reasoning": "BMPR2 causes PAH",
                "assessments": [
                    {
                        "resource_id": "doc-1",
                        "title": "Study 1",
                        "reasoning": "Document evidence",
                        "relationship": "causative mutation in",
                        "confidence": "high",
                        "quotes": [],
                    }
                ],
            }
        ]

        templates = render_all_reasoning_templates(pairs)

        # Check structure
        assert "0" in templates  # Pair index as string
        assert "overall" in templates["0"]
        assert "assess_0" in templates["0"]

        # Check content
        overall_html = templates["0"]["overall"]
        assert "BMPR2" in overall_html
        assert "PAH" in overall_html
        assert "Overall Assessment" in overall_html

        assess_html = templates["0"]["assess_0"]
        assert "Study 1" in assess_html
        assert "Document Assessment" in assess_html

    def test_multiple_pairs(self):
        """Test rendering templates for multiple pairs."""
        pairs = [
            {
                "entity1": {"name": "GeneA", "kind": "gene", "aliases": []},
                "entity2": {"name": "DiseaseX", "kind": "disease", "aliases": []},
                "relationship": "causes",
                "confidence": "high",
                "reasoning": "Evidence",
                "assessments": [],
            },
            {
                "entity1": {"name": "GeneB", "kind": "gene", "aliases": []},
                "entity2": {"name": "DiseaseY", "kind": "disease", "aliases": []},
                "relationship": "associated with",
                "confidence": "medium",
                "reasoning": "Evidence",
                "assessments": [],
            },
        ]

        templates = render_all_reasoning_templates(pairs)

        # Should have templates for both pairs
        assert "0" in templates
        assert "1" in templates
        assert "overall" in templates["0"]
        assert "overall" in templates["1"]

        # Check content is different
        assert "GeneA" in templates["0"]["overall"]
        assert "GeneB" in templates["1"]["overall"]
