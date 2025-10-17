"""
Integration tests for resource tracking in the extraction pipeline.

This module tests that resource tracking works end-to-end with the extraction
models and validation system.
"""

import pytest
from interaction_finder.extraction_graph.state import DocumentGroup
from interaction_finder.extraction_graph.models import (
    EntityInfo,
    EntityAssessmentOut,
    EntityPairOut,
)
from interaction_finder.resources import ResourcePool
# Remove validation imports - using simple ResourceQuote validation instead


class TestResourceTrackingIntegration:
    """Test integration between resource tracking and extraction models."""

    def test_document_group_resource_population(self):
        """Test that DocumentGroup can populate ResourcePool correctly."""
        doc_group = DocumentGroup(
            urls=["http://example.com/paper1", "http://example.com/paper2"],
            chunks={
                "http://example.com/paper1": [
                    {"text": "BRCA1 mutations cause breast cancer."},
                    {"text": "The BRCA1 gene is located on chromosome 17."},
                ],
                "http://example.com/paper2": [
                    {"text": "p53 is a tumor suppressor that interacts with BRCA1."},
                ],
            },
            metadata={
                "http://example.com/paper1": {"title": "BRCA1 and Cancer Risk"},
                "http://example.com/paper2": {"title": "p53-BRCA1 Interactions"},
            },
        )

        resource_pool = ResourcePool()
        resources = doc_group.populate_resource_pool(resource_pool)

        # Verify resources were created
        assert len(resources) == 2
        assert len(resource_pool.resource_map) == 2

        # Verify resource content
        paper1_resource = resources["http://example.com/paper1"]
        assert paper1_resource.title == "BRCA1 and Cancer Risk"
        assert "BRCA1 mutations cause breast cancer" in paper1_resource.text
        assert "chromosome 17" in paper1_resource.text

        paper2_resource = resources["http://example.com/paper2"]
        assert paper2_resource.title == "p53-BRCA1 Interactions"
        assert "p53 is a tumor suppressor" in paper2_resource.text

    def test_entity_info_with_resource_quotes(self):
        """Test EntityInfo model with ResourceQuotes integration."""
        # Set up resource pool
        resource_pool = ResourcePool()
        resource_id = resource_pool.register("http://example.com/paper")
        resource = resource_pool.add_content(
            resource_id,
            "Research Paper",
            "The BRCA1 gene is a tumor suppressor. BRCA1 mutations increase cancer risk. Studies show BRCA1 deficiency leads to genomic instability.",
        )

        # Create ResourceQuotes
        brca1_quote = resource.quote("BRCA1")
        assert brca1_quote is not None
        assert brca1_quote.count == 3  # Should find 3 occurrences

        # Create EntityInfo with ResourceQuotes
        entity = EntityInfo(
            name="BRCA1",
            kind="gene",
            source_url="http://example.com/paper",
            confidence=0.9,
            context="tumor suppressor gene",
            resource_quotes=[brca1_quote],
        )

        # Test helper methods
        all_urls = entity.get_all_source_urls()
        assert "http://example.com/paper" in all_urls
        assert len(all_urls) == 1  # No duplicates

        # Test basic ResourceQuote functionality
        assert brca1_quote.count > 0  # Quote was found in the resource
        assert "BRCA1" in brca1_quote.query_text  # Query text contains expected term

    def test_entity_assessment_with_evidence_quotes(self):
        """Test EntityAssessmentOut model with evidence quotes."""
        # Set up resources
        resource_pool = ResourcePool()
        resource_id = resource_pool.register("http://example.com/study")
        resource = resource_pool.add_content(
            resource_id,
            "Cancer Study",
            "BRCA1 mutations cause breast cancer. The relationship between BRCA1 and cancer is well established. Hereditary breast cancer often results from BRCA1 defects.",
        )

        # Create entities
        brca1_entity = EntityInfo(
            name="BRCA1",
            kind="gene",
            source_url="http://example.com/study",
            confidence=0.9,
        )
        cancer_entity = EntityInfo(
            name="breast cancer",
            kind="disease",
            source_url="http://example.com/study",
            confidence=0.85,
        )

        # Create evidence quote
        cause_quote = resource.quote("BRCA1 mutations cause breast cancer")
        assert cause_quote is not None

        # Create EntityAssessmentOut with evidence quotes
        assessment = EntityAssessmentOut(
            entity_a=brca1_entity,
            entity_b=cancer_entity,
            relationship_type="causes",
            confidence="high",
            evidence=["BRCA1 mutations cause breast cancer"],
            source_urls=["http://example.com/study"],
            reasoning="Clear causal relationship established in literature",
            evidence_quotes=[cause_quote],
        )

        # Test helper methods
        all_evidence = assessment.get_all_evidence_texts()
        assert "BRCA1 mutations cause breast cancer" in all_evidence

        all_urls = assessment.get_all_source_urls()
        assert "http://example.com/study" in all_urls

        # Test basic evidence quote validation
        assert cause_quote.count > 0  # Quote was found
        assert "cause" in cause_quote.query_text.lower()  # Contains key term

    def test_entity_pair_with_complete_provenance(self):
        """Test EntityPairOut with complete resource provenance."""
        # Set up multiple resources
        resource_pool = ResourcePool()

        # Resource 1
        resource_id1 = resource_pool.register("http://example.com/study1")
        resource1 = resource_pool.add_content(
            resource_id1,
            "Primary Study",
            "BRCA1 is a tumor suppressor gene. Loss of BRCA1 function leads to breast cancer development.",
        )

        # Resource 2
        resource_id2 = resource_pool.register("http://example.com/study2")
        resource2 = resource_pool.add_content(
            resource_id2,
            "Follow-up Study",
            "Clinical trials confirm that BRCA1 mutations cause hereditary breast cancer in patients.",
        )

        # Create entities with resource quotes
        brca1_quote1 = resource1.quote("BRCA1")
        brca1_quote2 = resource2.quote("BRCA1")

        brca1_entity = EntityInfo(
            name="BRCA1",
            kind="gene",
            source_url="http://example.com/study1",
            confidence=0.95,
            resource_quotes=[brca1_quote1, brca1_quote2],
        )

        cancer_entity = EntityInfo(
            name="breast cancer",
            kind="disease",
            source_url="http://example.com/study1",
            confidence=0.9,
        )

        # Create evidence quotes from both resources
        evidence_quote1 = resource1.quote("leads to breast cancer")
        evidence_quote2 = resource2.quote("cause hereditary breast cancer")

        # Create final EntityPairOut with complete provenance
        pair = EntityPairOut(
            entity_a=brca1_entity,
            entity_b=cancer_entity,
            relationship="causes",
            confidence=0.92,
            evidence=[
                "Loss of BRCA1 function leads to breast cancer development",
                "BRCA1 mutations cause hereditary breast cancer",
            ],
            source_documents=[
                "http://example.com/study1",
                "http://example.com/study2",
            ],
            evidence_quotes=[evidence_quote1, evidence_quote2],
        )

        # Test helper methods
        all_evidence = pair.get_all_evidence_texts()
        assert len(all_evidence) >= 2  # Original plus quote texts

        all_urls = pair.get_all_source_urls()
        assert "http://example.com/study1" in all_urls
        assert "http://example.com/study2" in all_urls

        # Test basic evidence validation
        assert evidence_quote1 is not None
        assert evidence_quote2 is not None
        assert evidence_quote1.count >= 1
        assert evidence_quote2.count >= 1

    def test_resource_functionality(self):
        """Test core resource functionality without complex validation."""
        # Set up comprehensive test data
        resource_pool = ResourcePool()
        resource_id = resource_pool.register("http://example.com/comprehensive")
        resource = resource_pool.add_content(
            resource_id,
            "Comprehensive Study",
            "BRCA1 is a critical tumor suppressor gene. When BRCA1 is mutated, it causes breast cancer and ovarian cancer. The BRCA1 protein normally repairs DNA damage, but defective BRCA1 leads to genomic instability and tumor formation.",
        )

        # Create various quotes
        brca1_quote = resource.quote("BRCA1")
        cause_quote = resource.quote("BRCA1 is mutated, it causes breast cancer")

        # Test basic functionality
        assert brca1_quote is not None
        assert brca1_quote.count == 4  # All BRCA1 mentions
        assert brca1_quote.count > 0  # Quote was found
        assert "BRCA1" in brca1_quote.query_text  # Contains expected term

        assert cause_quote is not None
        assert cause_quote.count == 1
        assert (
            "BRCA1 is mutated, it causes breast cancer" in cause_quote.get_quote_text(1)
        )

        # Test context retrieval
        context = brca1_quote.get_context(1, context_chars=50)
        assert "BRCA1" in context
        assert len(context) > 0

    def test_backward_compatibility(self):
        """Test that models work without ResourceQuotes (backward compatibility)."""
        # Create entities without ResourceQuotes
        entity = EntityInfo(
            name="BRCA1",
            kind="gene",
            source_url="http://example.com/paper",
            confidence=0.9,
            context="tumor suppressor gene",
            # No resource_quotes provided - should default to empty list
        )

        # Should work fine
        all_urls = entity.get_all_source_urls()
        assert "http://example.com/paper" in all_urls
        assert len(all_urls) == 1

        # Create assessment without evidence quotes
        cancer_entity = EntityInfo(
            name="breast cancer",
            kind="disease",
            source_url="http://example.com/paper",
            confidence=0.85,
        )

        assessment = EntityAssessmentOut(
            entity_a=entity,
            entity_b=cancer_entity,
            relationship_type="causes",
            confidence="high",
            evidence=["BRCA1 mutations cause breast cancer"],
            source_urls=["http://example.com/paper"],
            reasoning="Literature evidence",
            # No evidence_quotes provided - should default to empty list
        )

        # Should work with just string evidence
        all_evidence = assessment.get_all_evidence_texts()
        assert "BRCA1 mutations cause breast cancer" in all_evidence
        assert len(all_evidence) == 1  # Just the string evidence

        all_urls = assessment.get_all_source_urls()
        assert "http://example.com/paper" in all_urls
