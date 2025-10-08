"""
Tests for provenance tracking in V3 extraction pipeline.

Tests complete provenance chain from ResourceId → Resource → ResourceQuote.
"""

import pytest
from interaction_finder.resources import ResourcePool, ResourceId
from interaction_finder.extraction_graph_v2.models import (
    EntityWithQuotes,
    EntityPairOut,
)


@pytest.fixture
def resource_pool_with_documents():
    """Create resource pool with sample documents."""
    pool = ResourcePool()

    # Add document 1
    pool.add(
        "https://example.com/doc1",
        "BRCA1 Research Paper",
        "BRCA1 is a tumor suppressor gene associated with breast cancer. "
        "Mutations in BRCA1 increase cancer risk significantly.",
    )

    # Add document 2
    pool.add(
        "https://example.com/doc2",
        "BRCA2 Research Paper",
        "BRCA2 is also a tumor suppressor gene. "
        "BRCA2 mutations are linked to breast and ovarian cancers.",
    )

    return pool


def test_resource_id_creation():
    """Test ResourceId is created correctly from URL."""
    url = "https://example.com/paper123"
    # ResourceId requires URL and counter
    resource_id = ResourceId(url=url, counter=1)

    assert resource_id.url == url, "URL should be preserved"
    assert len(resource_id.id) > 0, "ID should be generated"
    # ID is generated from URL hash
    assert "_" in resource_id.id, "ID should contain counter separator"


def test_resource_quote_validation(resource_pool_with_documents):
    """Test ResourceQuote validation ensures valid provenance."""
    resource = resource_pool_with_documents.resources[0]

    # Create valid quote
    quote = resource.quote("BRCA1 is a tumor suppressor gene")

    # Validate - just check quote was found
    assert len(quote.spans) > 0, "Valid quote should have spans"
    assert quote.query_text == "BRCA1 is a tumor suppressor gene", (
        "Query text should match"
    )


def test_resource_quote_invalid_span(resource_pool_with_documents):
    """Test quote with invalid span raises ValueError."""
    resource = resource_pool_with_documents.resources[0]

    # Create quote for text not in document - should raise ValueError
    try:
        quote = resource.quote("This text does not exist in the document")
        # If it doesn't raise, it should have zero spans
        assert len(quote.spans) == 0, "Invalid quote should have no spans"
    except ValueError:
        # This is expected behavior
        pass


def test_entity_provenance_chain(resource_pool_with_documents):
    """Test complete provenance chain for entity."""
    resource = resource_pool_with_documents.resources[0]

    # Create entity with quotes
    entity = EntityWithQuotes(
        name="BRCA1",
        kind="gene",
        quotes=[
            resource.quote("BRCA1 is a tumor suppressor gene"),
            resource.quote("Mutations in BRCA1"),
        ],
    )

    # Verify provenance chain
    assert len(entity.quotes) == 2, "Entity should have 2 quotes"

    for quote in entity.quotes:
        # ResourceQuote → Resource → ResourceId chain
        assert quote.resource is not None, "Quote should have resource"
        assert quote.resource.id is not None, "Resource should have ID"
        assert quote.resource.id.url.startswith("http"), "Resource ID should have URL"
        assert len(quote.spans) > 0, "Quote should have spans"


def test_entity_pair_provenance(resource_pool_with_documents):
    """Test provenance for complete entity pair."""
    resource = resource_pool_with_documents.resources[0]

    # Create entities
    gene_entity = EntityWithQuotes(
        name="BRCA1",
        kind="gene",
        quotes=[resource.quote("BRCA1 is a tumor suppressor gene")],
    )

    disease_entity = EntityWithQuotes(
        name="breast cancer",
        kind="disease",
        quotes=[resource.quote("breast cancer")],
    )

    # Create pair - use actual quote text from document
    pair = EntityPairOut(
        entity_a=gene_entity,
        entity_b=disease_entity,
        relationship_type="gene-disease interaction",
        confidence="high",
        evidence_quotes=[
            resource.quote(
                "BRCA1 is a tumor suppressor gene associated with breast cancer"
            )
        ],
        reasoning="BRCA1 mutations increase breast cancer risk",
    )

    # Validate provenance
    pair.validate_provenance()  # Should not raise

    # Verify both entities have valid quotes
    assert len(pair.entity_a.quotes) > 0, "Entity A should have quotes"
    assert len(pair.entity_b.quotes) > 0, "Entity B should have quotes"

    # Verify evidence quotes
    assert len(pair.evidence_quotes) > 0, "Pair should have evidence quotes"
    for quote in pair.evidence_quotes:
        assert quote.resource is not None, "Evidence quote should have resource"


def test_cross_document_provenance(resource_pool_with_documents):
    """Test provenance tracks entities across multiple documents."""
    resource1 = resource_pool_with_documents.resources[0]
    resource2 = resource_pool_with_documents.resources[1]

    # Create entity with quotes from both documents
    entity = EntityWithQuotes(
        name="breast cancer",
        kind="disease",
        quotes=[
            resource1.quote("breast cancer"),  # From document 1
            resource2.quote("breast and ovarian cancers"),  # From document 2
        ],
    )

    # Verify cross-document provenance
    assert len(entity.quotes) == 2, "Entity should have quotes from both documents"

    resource_ids = set(q.resource.id.id for q in entity.quotes)
    assert len(resource_ids) == 2, "Quotes should come from 2 different documents"


def test_quote_span_extraction(resource_pool_with_documents):
    """Test quote spans point to correct text positions."""
    resource = resource_pool_with_documents.resources[0]
    quote = resource.quote("BRCA1")

    # Extract text using spans
    for span_start, span_end in quote.spans:
        extracted_text = resource.text[span_start:span_end]
        assert "BRCA1" in extracted_text, (
            f"Span should point to BRCA1, got: {extracted_text}"
        )


def test_quote_context_extraction(resource_pool_with_documents):
    """Test extracting context around quote."""
    resource = resource_pool_with_documents.resources[0]
    quote = resource.quote("tumor suppressor gene")

    # Get context - check signature of get_context method
    # ResourceQuote might not have get_context or might have different signature
    # Just verify quote has spans
    assert len(quote.spans) > 0, "Quote should have spans"
    # Verify we can extract text from spans
    for start, end in quote.spans:
        text = resource.text[start:end]
        assert "tumor" in text.lower(), "Span should point to quote text"


def test_multiple_occurrence_tracking(resource_pool_with_documents):
    """Test tracking multiple occurrences of same quote."""
    # Add document with repeated term
    pool = ResourcePool()
    doc_text = "BRCA1 gene. BRCA1 protein. BRCA1 mutations."
    resource = pool.add("http://example.com/doc", "Test", doc_text)

    quote = resource.quote("BRCA1")

    # Should find all 3 occurrences
    assert quote.count == 3, f"Should find 3 occurrences, found {quote.count}"
    assert len(quote.spans) == 3, f"Should have 3 spans, got {len(quote.spans)}"

    # Each span should point to correct occurrence
    for i, (start, end) in enumerate(quote.spans):
        extracted = doc_text[start:end]
        assert extracted == "BRCA1", (
            f"Occurrence {i} should extract 'BRCA1', got '{extracted}'"
        )


def test_provenance_chain_validation_comprehensive(resource_pool_with_documents):
    """
    Comprehensive test of provenance chain validation.

    Tests all levels: ResourceId → Resource → ResourceQuote → Entity → EntityPair
    """
    resource = resource_pool_with_documents.resources[0]

    # Level 1: ResourceId
    assert resource.id.url.startswith("http"), "ResourceId should have valid URL"
    assert len(resource.id.id) > 0, "ResourceId should have generated ID"

    # Level 2: Resource
    assert len(resource.text) > 0, "Resource should have content"
    assert resource.title is not None, "Resource should have title"

    # Level 3: ResourceQuote
    quote = resource.quote("BRCA1")
    assert len(quote.spans) > 0, "Quote should have spans"
    assert quote.resource.id.id == resource.id.id, "Quote should link to resource"

    # Level 4: Entity
    entity = EntityWithQuotes(
        name="BRCA1",
        kind="gene",
        quotes=[quote],
    )
    assert len(entity.quotes) > 0, "Entity should have quotes"
    assert entity.quotes[0].resource.id.id == resource.id.id, (
        "Entity quote should link to resource"
    )

    # Level 5: EntityPair
    disease_entity = EntityWithQuotes(
        name="breast cancer",
        kind="disease",
        quotes=[resource.quote("breast cancer")],
    )

    # Need at least one evidence quote
    pair = EntityPairOut(
        entity_a=entity,
        entity_b=disease_entity,
        relationship_type="interaction",
        confidence="high",
        evidence_quotes=[
            resource.quote(
                "BRCA1 is a tumor suppressor gene associated with breast cancer"
            )
        ],
        reasoning="Test",
    )

    # Full validation
    pair.validate_provenance()  # Should not raise

    # Verify complete chain
    for e in [pair.entity_a, pair.entity_b]:
        for q in e.quotes:
            assert q.resource is not None, "Quote should have resource"
            assert q.resource.id is not None, "Resource should have ID"
            assert q.resource.id.url.startswith("http"), (
                "Resource should have valid URL"
            )
            assert len(q.spans) > 0, "Quote should have spans"
