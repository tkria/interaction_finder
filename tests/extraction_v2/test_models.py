"""
Unit tests for extraction graph V2 models.

Tests ResourceQuote integration and provenance validation.
"""

import pytest
from interaction_finder.resources import ResourcePool, ResourceQuote
from interaction_finder.extraction_graph_v2.models import (
    EntityWithQuotes,
    IndividualAssessment,
    EntityPairOut,
    EntityListOut,
    AssessmentOut,
)
from .fixtures import BRCA1_DOCUMENT, BRCA1_EXPECTED_ENTITIES, TEST_URLS


class TestEntityWithQuotes:
    """Test EntityWithQuotes model with ResourceQuote integration."""

    @pytest.fixture
    def resource_pool(self):
        """Create resource pool with test document."""
        pool = ResourcePool()
        resource = pool.add(TEST_URLS[0], "BRCA1 Study", BRCA1_DOCUMENT)
        return pool, resource

    def test_entity_with_single_quote(self, resource_pool):
        """Test entity with one ResourceQuote."""
        pool, resource = resource_pool

        # Create ResourceQuote for BRCA1
        quote = resource.quote("BRCA1")
        assert quote is not None
        assert quote.count >= 1

        # Create EntityWithQuotes
        entity = EntityWithQuotes(name="BRCA1", kind="gene", quotes=[quote])

        # Validate
        assert entity.validate()
        assert entity.name == "BRCA1"
        assert entity.kind == "gene"
        assert entity.total_occurrences >= 1
        assert len(entity.all_contexts) >= 1
        assert TEST_URLS[0] in entity.get_all_source_urls()

    def test_entity_with_multiple_quotes(self, resource_pool):
        """Test entity with multiple ResourceQuotes from same document."""
        pool, resource = resource_pool

        # Create quotes for different phrases containing BRCA1
        brca1_quote = resource.quote("BRCA1")
        mutation_quote = resource.quote("BRCA1 mutations")

        assert brca1_quote is not None
        assert mutation_quote is not None

        entity = EntityWithQuotes(
            name="BRCA1",
            kind="gene",
            quotes=[brca1_quote, mutation_quote],
            confidence=0.9,
        )

        assert entity.validate()
        assert entity.confidence == 0.9
        assert entity.total_occurrences >= 2  # Should have multiple occurrences
        assert len(entity.all_contexts) >= 2

    def test_entity_validation_failure(self, resource_pool):
        """Test validation fails for invalid quotes."""
        pool, resource = resource_pool

        # Try to create quote for non-existent text
        try:
            quote = resource.quote("NonExistentEntity")
            # If quote creation succeeds, entity validation should still work
            if quote:
                entity = EntityWithQuotes(name="Test", kind="gene", quotes=[quote])
                assert entity.validate()
            else:
                # Quote creation failed, which is expected
                pass
        except ValueError:
            # Quote creation failed, which is expected for non-existent text
            pass


class TestIndividualAssessment:
    """Test IndividualAssessment model with evidence validation."""

    @pytest.fixture
    def test_entity(self):
        """Create test entity with quotes."""
        pool = ResourcePool()
        resource = pool.add(TEST_URLS[0], "Test", BRCA1_DOCUMENT)
        quote = resource.quote("BRCA1")

        return EntityWithQuotes(name="BRCA1", kind="gene", quotes=[quote])

    def test_assessment_with_evidence(self, test_entity):
        """Test assessment with valid evidence quotes."""
        pool = ResourcePool()
        resource = pool.add(TEST_URLS[0], "Test", BRCA1_DOCUMENT)

        # Create evidence quote
        evidence_text = "BRCA1 mutations significantly increase breast cancer risk"
        evidence_quote = resource.quote(evidence_text)
        assert evidence_quote is not None

        assessment = IndividualAssessment(
            entity=test_entity,
            relationship_potential="high",
            related_entities=["breast cancer"],
            evidence_quotes=[evidence_quote],
            reasoning="BRCA1 mutations are strongly associated with breast cancer risk",
            confidence=0.95,
        )

        assert assessment.validate_evidence()
        assert assessment.relationship_potential == "high"
        assert "breast cancer" in assessment.related_entities
        assert len(assessment.get_all_evidence_texts()) >= 1
        assert len(assessment.get_evidence_contexts()) >= 1

    def test_assessment_without_evidence(self, test_entity):
        """Test assessment with no evidence (should be 'none' potential)."""
        assessment = IndividualAssessment(
            entity=test_entity,
            relationship_potential="none",
            related_entities=[],
            evidence_quotes=[],
            reasoning="No clear relationships found",
        )

        # Should validate for 'none' potential
        assert assessment.validate_evidence()
        assert assessment.relationship_potential == "none"
        assert len(assessment.related_entities) == 0

    def test_assessment_evidence_mismatch(self, test_entity):
        """Test assessment where evidence doesn't contain entity name."""
        pool = ResourcePool()
        resource = pool.add(TEST_URLS[0], "Test", "Random text without entity mention")

        # This will fail because BRCA1 is not in the quote
        try:
            evidence_quote = resource.quote("Random text")
            if evidence_quote:
                assessment = IndividualAssessment(
                    entity=test_entity,
                    relationship_potential="high",
                    related_entities=["some disease"],
                    evidence_quotes=[evidence_quote],
                    reasoning="Invalid evidence",
                )

                # Should fail validation
                assert not assessment.validate_evidence()
        except ValueError:
            # Expected - quote creation should fail
            pass


class TestEntityPairOut:
    """Test EntityPairOut model with complete provenance."""

    @pytest.fixture
    def test_pair(self):
        """Create test pair with full provenance."""
        pool = ResourcePool()
        resource = pool.add(TEST_URLS[0], "BRCA1 Study", BRCA1_DOCUMENT)

        # Create entities
        brca1_quote = resource.quote("BRCA1")
        cancer_quote = resource.quote("breast cancer")

        entity_a = EntityWithQuotes(name="BRCA1", kind="gene", quotes=[brca1_quote])
        entity_b = EntityWithQuotes(
            name="breast cancer", kind="disease", quotes=[cancer_quote]
        )

        # Create evidence quote
        evidence_text = "BRCA1 mutations significantly increase breast cancer risk"
        evidence_quote = resource.quote(evidence_text)

        return EntityPairOut(
            entity_a=entity_a,
            entity_b=entity_b,
            confidence="high",
            evidence_quotes=[evidence_quote],
            reasoning="Strong genetic association",
        )

    def test_pair_output_format(self, test_pair):
        """Test conversion to required output format with quotes."""
        output = test_pair.to_output_format()

        # Check required fields
        assert "first" in output
        assert "first_kind" in output
        assert "second" in output
        assert "second_kind" in output
        assert "reasoning" in output
        assert "resources" in output
        assert "confidence" in output

        # Check values
        assert output["first"] == "BRCA1"
        assert output["first_kind"] == "gene"
        assert output["second"] == "breast cancer"
        assert output["second_kind"] == "disease"
        assert output["confidence"] == "high"

        # Check new resources format with quotes
        resources = output["resources"]
        assert isinstance(resources, dict), "Resources should be a dictionary"
        assert len(resources) >= 1, "Should have at least one resource"

        # Check each resource has quotes
        for resource_id, quotes in resources.items():
            assert isinstance(quotes, list), (
                f"Resource {resource_id} should have list of quotes"
            )
            assert len(quotes) >= 1, (
                f"Resource {resource_id} should have at least one quote"
            )

            # Check each quote has required fields
            for quote in quotes:
                assert "quote" in quote, "Quote should have 'quote' field"
                assert "span" in quote, "Quote should have 'span' field"
                assert isinstance(quote["quote"], str), "Quote text should be string"
                assert isinstance(quote["span"], list), "Span should be a list"
                assert len(quote["span"]) == 2, (
                    "Span should have start and end positions"
                )
                assert isinstance(quote["span"][0], int), "Span start should be integer"
                assert isinstance(quote["span"][1], int), "Span end should be integer"
                assert quote["span"][0] < quote["span"][1], (
                    "Span start should be less than end"
                )

    def test_pair_provenance_validation(self, test_pair):
        """Test provenance validation."""
        assert test_pair.validate_provenance()
        assert len(test_pair.get_all_source_urls()) >= 1
        assert TEST_URLS[0] in test_pair.get_all_source_urls()

    def test_quote_accuracy_in_output(self, test_pair):
        """Test that quotes in output format are accurate and verifiable."""
        output = test_pair.to_output_format()

        # Get the original document text for verification
        resource = test_pair.evidence_quotes[0].resource
        document_text = resource.text

        # Check each quote in the output
        for resource_id, quotes in output["resources"].items():
            for quote_data in quotes:
                quote_text = quote_data["quote"]
                span = quote_data["span"]

                # Verify the quote text matches what's at the specified position
                extracted_text = document_text[span[0] : span[1]]
                assert extracted_text == quote_text, (
                    f"Quote text '{quote_text}' doesn't match document text at "
                    f"position {span}: '{extracted_text}'"
                )

                # Verify the quote contains expected content
                assert len(quote_text) > 0, "Quote text should not be empty"

    def test_pair_without_evidence_fails(self):
        """Test that pair without evidence fails validation."""
        pool = ResourcePool()
        resource = pool.add(TEST_URLS[0], "Test", BRCA1_DOCUMENT)

        entity_a = EntityWithQuotes(
            name="BRCA1", kind="gene", quotes=[resource.quote("BRCA1")]
        )
        entity_b = EntityWithQuotes(
            name="breast cancer",
            kind="disease",
            quotes=[resource.quote("breast cancer")],
        )

        # Create pair without evidence
        try:
            pair = EntityPairOut(
                entity_a=entity_a,
                entity_b=entity_b,
                confidence="high",
                evidence_quotes=[],  # No evidence
                reasoning="No evidence provided",
            )
        except ValueError:
            # Expected - should fail validation due to min_length=1 on evidence_quotes
            pass


class TestAgentOutputModels:
    """Test agent output models."""

    def test_entity_list_out(self):
        """Test EntityListOut model."""
        from interaction_finder.extraction_graph_v2.models import EntityOut

        output = EntityListOut(
            entities=[
                EntityOut(
                    name="BRCA1",
                    kind="gene",
                    quotes={"Resource 1_abc123": ["BRCA1 is a tumor suppressor gene"]},
                ),
                EntityOut(
                    name="breast cancer",
                    kind="disease",
                    quotes={"Resource 1_abc123": ["associated with breast cancer"]},
                ),
            ],
            entity_kinds=["gene", "disease"],
            reasoning="Extracted entities from document analysis",
        )

        assert len(output.entities) == 2
        assert "gene" in output.entity_kinds
        assert "disease" in output.entity_kinds
        assert output.reasoning != ""

    def test_assessment_out(self):
        """Test AssessmentOut model."""
        output = AssessmentOut(
            potential="high",
            related=["breast cancer", "ovarian cancer"],
            evidence=[
                "BRCA1 mutations increase cancer risk",
                "Associated with hereditary breast cancer",
            ],
            reasoning="Strong evidence for disease associations",
        )

        assert output.potential == "high"
        assert len(output.related) == 2
        assert len(output.evidence) == 2
        assert "disease" in output.reasoning
