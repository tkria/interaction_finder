"""Tests for extraction sub-stage resumption."""

import pytest

from interaction_finder.extraction.models import (
    EntityMention,
    EntityPairKey,
    EntityRef,
    ExtractionMetadata,
    PairAssessment,
    PairJudgment,
    ProximalEntitySet,
)
from interaction_finder.extraction.state import State
from interaction_finder.resources import (
    Resource,
    ResourceId,
    ResourcePool,
    ResourceQuote,
)


def test_extraction_metadata_resume_fields():
    """Test that ExtractionMetadata has resume fields with correct defaults."""
    metadata = ExtractionMetadata(
        topic="test",
        resource_count=0,
        total_entities_found=0,
        entities_after_validation=0,
        entities_merged=0,
        merge_cache_hits=0,
        merge_cache_misses=0,
        proximal_sets_found=0,
        total_pairs_found=0,
        pairs_accepted=0,
        pairs_rejected=0,
        quotes_validated=0,
        quotes_failed=0,
    )

    # Default: no resume fields (complete)
    assert metadata.resume_from is None
    assert metadata.resume_state is None
    assert metadata.is_complete is True
    assert metadata.is_resumable is False


def test_extraction_metadata_with_resume_data():
    """Test ExtractionMetadata with resume data (incomplete extraction)."""
    metadata = ExtractionMetadata(
        topic="test",
        resource_count=5,
        total_entities_found=10,
        entities_after_validation=8,
        entities_merged=0,
        merge_cache_hits=0,
        merge_cache_misses=0,
        proximal_sets_found=3,
        total_pairs_found=0,
        pairs_accepted=0,
        pairs_rejected=0,
        quotes_validated=10,
        quotes_failed=0,
        resume_from="process_documents",
        resume_state={"some": "state"},
    )

    assert metadata.resume_from == "process_documents"
    assert metadata.resume_state == {"some": "state"}
    assert metadata.is_complete is False
    assert metadata.is_resumable is True


def test_state_serialization_roundtrip():
    """Test that State can be serialized and deserialized correctly."""
    from interaction_finder.extraction.models import RelationshipConsolidation

    # Create a minimal state
    pool = ResourcePool()
    state = State(
        topic="test topic",
        target_entity_types=["gene", "disease"],
        permitted_pairs={"gene": {"disease"}, "disease": {"gene"}},
    )

    # Add some simple data
    state.entities_merged = 5
    state.quotes_validated = 10
    state.quotes_failed = 2
    state.relationship_polarities = {"activates": "positive"}
    state.consolidated.relationships.append(
        RelationshipConsolidation(
            original="activates",
            consolidated="activates",
            polarity="positive",
            opposites=["inhibits"],
            reasoning="Test consolidation reasoning with sufficient length",
        )
    )

    # Serialize
    state_dict = state.to_dict()

    # Check serialized structure
    assert "entities_merged" in state_dict
    assert state_dict["entities_merged"] == 5
    assert state_dict["quotes_validated"] == 10
    assert state_dict["relationship_polarities"] == {"activates": "positive"}
    assert "consolidated" in state_dict
    assert len(state_dict["consolidated"]["relationships"]) == 1

    # Deserialize
    restored_state = State.from_dict(
        state_dict,
        topic="test topic",
        target_entity_types=["gene", "disease"],
        permitted_pairs={"gene": {"disease"}, "disease": {"gene"}},
        resource_pool=pool,
    )

    # Verify restoration
    assert restored_state.entities_merged == 5
    assert restored_state.quotes_validated == 10
    assert restored_state.relationship_polarities == {"activates": "positive"}
    assert len(restored_state.consolidated.relationships) == 1
    assert restored_state.consolidated.relationships[0].opposites == ["inhibits"]


def test_state_serialization_preserves_structure():
    """Test that State serialization preserves basic structure."""
    pool = ResourcePool()

    # Note: In actual extraction, entities_by_resource and validated_entities_by_resource
    # contain complex nested objects with ResourceQuotes. For the checkpoint resumption
    # use case, these are serialized to JSON and then reconstructed. The rehydration
    # of quotes happens during deserialization via Pydantic's validation.
    #
    # This test focuses on ensuring the registry-based serialization works correctly
    # for the different field types we have.

    state = State(
        topic="test",
        target_entity_types=["gene"],
        permitted_pairs={"gene": {"gene"}},
    )

    # Test serialization of empty collections
    state_dict = state.to_dict()

    # Verify all registry fields are present
    assert "entities_by_resource" in state_dict
    assert "validated_entities_by_resource" in state_dict
    assert "proximal_sets_by_resource" in state_dict
    assert "pair_assessments_by_resource" in state_dict
    assert "pair_judgments" in state_dict
    assert "agent_merge_cache" in state_dict
    assert "consolidated" in state_dict

    # Deserialize
    restored_state = State.from_dict(
        state_dict,
        topic="test",
        target_entity_types=["gene"],
        permitted_pairs={"gene": {"gene"}},
        resource_pool=pool,
    )

    # Verify structure restored
    assert len(restored_state.entities_by_resource) == 0
    assert len(restored_state.validated_entities_by_resource) == 0
    assert len(restored_state.pair_judgments) == 0


def test_state_serialization_with_tuple_keys():
    """Test State serialization handles tuple keys correctly."""
    pool = ResourcePool()
    state = State(
        topic="test",
        target_entity_types=["gene"],
        permitted_pairs={"gene": {"gene"}},
    )

    # Add agent merge cache (tuple keys)
    state.agent_merge_cache = {
        ("BRCA1", "BRCA-1", "gene"): ("BRCA1", "merge into canonical"),
        ("TP53", "p53", "gene"): ("TP53", "standard nomenclature"),
    }

    # Serialize
    state_dict = state.to_dict()

    # Check tuple keys converted to strings
    assert "BRCA1|BRCA-1|gene" in state_dict["agent_merge_cache"]
    assert "TP53|p53|gene" in state_dict["agent_merge_cache"]

    # Deserialize
    restored_state = State.from_dict(
        state_dict,
        topic="test",
        target_entity_types=["gene"],
        permitted_pairs={"gene": {"gene"}},
        resource_pool=pool,
    )

    # Verify restoration
    assert ("BRCA1", "BRCA-1", "gene") in restored_state.agent_merge_cache
    assert restored_state.agent_merge_cache[("BRCA1", "BRCA-1", "gene")] == (
        "BRCA1",
        "merge into canonical",
    )


def test_state_serialization_empty_collections():
    """Test State serialization handles empty collections correctly."""
    pool = ResourcePool()
    state = State(
        topic="test",
        target_entity_types=["gene"],
        permitted_pairs={"gene": {"gene"}},
    )

    # State with all empty collections
    state_dict = state.to_dict()

    # Verify all fields present
    assert "entities_by_resource" in state_dict
    assert "validated_entities_by_resource" in state_dict
    assert "proximal_sets_by_resource" in state_dict
    assert "pair_assessments_by_resource" in state_dict
    assert "pair_judgments" in state_dict

    # Deserialize
    restored_state = State.from_dict(
        state_dict,
        topic="test",
        target_entity_types=["gene"],
        permitted_pairs={"gene": {"gene"}},
        resource_pool=pool,
    )

    # Verify empty collections restored
    assert len(restored_state.entities_by_resource) == 0
    assert len(restored_state.validated_entities_by_resource) == 0
    assert len(restored_state.pair_judgments) == 0


def test_checkpoint_backward_compatibility():
    """Test that checkpoints without resume fields are handled correctly."""
    # Old-style metadata (no resume fields)
    metadata = ExtractionMetadata(
        topic="test",
        resource_count=5,
        total_entities_found=10,
        entities_after_validation=8,
        entities_merged=2,
        merge_cache_hits=5,
        merge_cache_misses=1,
        proximal_sets_found=3,
        total_pairs_found=5,
        pairs_accepted=3,
        pairs_rejected=2,
        quotes_validated=20,
        quotes_failed=1,
    )

    # Should be considered complete (backward compatibility)
    assert metadata.is_complete is True
    assert metadata.is_resumable is False


def test_validated_entities_preserve_mentions_and_aliases():
    """Test that validated_entities_by_resource preserves full EntityRef data.

    EntityRef contains mentions with original names, aliases, and quotes.
    This data must survive serialization roundtrip for checkpoint resumption.
    """
    pool = ResourcePool()
    resource = pool.add(
        url="http://example.com/doc1",
        title="Test Document",
        document_text="Gene BRCA1 (breast cancer 1) is important.",
    )
    rid = resource.id

    state = State(
        topic="test",
        target_entity_types=["gene"],
        permitted_pairs={"gene": {"gene"}},
    )

    # Create EntityRef with mentions that have aliases and quotes
    mention = EntityMention(
        kind="gene",
        name="BRCA1",
        aliases=["breast cancer 1", "BRCA-1"],
        quotes=[
            ResourceQuote(
                resource=resource,
                text="Gene BRCA1 (breast cancer 1) is important.",
            )
        ],
        reasoning="Important cancer gene",
    )
    entity_ref = EntityRef(canonical="BRCA1", mentions=[mention])

    state.validated_entities_by_resource[rid] = {"BRCA1": entity_ref}

    # Serialize
    state_dict = state.to_dict()

    # Verify serialized structure preserves full data
    serialized_entities = state_dict["validated_entities_by_resource"]
    assert rid.url in serialized_entities
    entity_data = serialized_entities[rid.url]["BRCA1"]
    assert entity_data["canonical"] == "BRCA1"
    assert len(entity_data["mentions"]) == 1
    assert entity_data["mentions"][0]["name"] == "BRCA1"
    assert entity_data["mentions"][0]["aliases"] == ["breast cancer 1", "BRCA-1"]
    assert len(entity_data["mentions"][0]["quotes"]) == 1

    # Deserialize
    restored_state = State.from_dict(
        state_dict,
        topic="test",
        target_entity_types=["gene"],
        permitted_pairs={"gene": {"gene"}},
        resource_pool=pool,
    )

    # Verify full EntityRef data restored
    restored_entities = restored_state.validated_entities_by_resource[rid]
    assert "BRCA1" in restored_entities
    restored_ref = restored_entities["BRCA1"]
    assert restored_ref.canonical == "BRCA1"
    assert len(restored_ref.mentions) == 1
    assert restored_ref.mentions[0].name == "BRCA1"
    assert restored_ref.mentions[0].aliases == ["breast cancer 1", "BRCA-1"]
    assert len(restored_ref.mentions[0].quotes) == 1
    # Verify aliases() method works correctly
    assert set(restored_ref.aliases()) == {"breast cancer 1", "BRCA-1"}


def test_validated_entities_preserve_merged_mentions():
    """Test that merged EntityRefs preserve all original mentions.

    When entities A and B merge into A, the resulting EntityRef should have
    mentions from both original entities, and B's original name should appear
    in aliases.
    """
    pool = ResourcePool()
    resource = pool.add(
        url="http://example.com/doc1",
        title="Test Document",
        document_text="TP53 and p53 are the same gene.",
    )
    rid = resource.id

    state = State(
        topic="test",
        target_entity_types=["gene"],
        permitted_pairs={"gene": {"gene"}},
    )

    # Simulate merged entity: TP53 absorbed p53
    mention1 = EntityMention(
        kind="gene",
        name="TP53",
        aliases=["tumor protein p53"],
        quotes=[],
        reasoning="Standard name",
    )
    mention2 = EntityMention(
        kind="gene",
        name="p53",  # Original name differs from canonical
        aliases=["tumor suppressor p53"],
        quotes=[],
        reasoning="Common alias",
    )
    # Merged EntityRef has both mentions
    merged_ref = EntityRef(canonical="TP53", mentions=[mention1, mention2])

    state.validated_entities_by_resource[rid] = {"TP53": merged_ref}

    # Serialize and deserialize
    state_dict = state.to_dict()
    restored_state = State.from_dict(
        state_dict,
        topic="test",
        target_entity_types=["gene"],
        permitted_pairs={"gene": {"gene"}},
        resource_pool=pool,
    )

    # Verify both mentions preserved
    restored_ref = restored_state.validated_entities_by_resource[rid]["TP53"]
    assert len(restored_ref.mentions) == 2

    # Verify aliases() aggregates correctly:
    # - "p53" from mention2.name (differs from canonical)
    # - "tumor protein p53" from mention1.aliases
    # - "tumor suppressor p53" from mention2.aliases
    aliases = restored_ref.aliases()
    assert "p53" in aliases  # Original name of merged entity
    assert "tumor protein p53" in aliases
    assert "tumor suppressor p53" in aliases


def test_global_entities_serialization_roundtrip():
    """Test that global_entities survives serialization roundtrip.

    The global entity index aggregates mentions across all resources and is
    used for PairJudgment alias lookup.
    """
    pool = ResourcePool()
    resource1 = pool.add(
        url="http://example.com/doc1",
        title="Document 1",
        document_text="BRCA1 is important.",
    )
    resource2 = pool.add(
        url="http://example.com/doc2",
        title="Document 2",
        document_text="BRCA-1 is studied.",
    )

    state = State(
        topic="test",
        target_entity_types=["gene"],
        permitted_pairs={"gene": {"gene"}},
    )

    # Simulate global_entities built by ConsolidateEntitiesNode
    # Two mentions from different documents, aggregated into one EntityRef
    mention1 = EntityMention(
        kind="gene",
        name="BRCA1",
        aliases=[],
        quotes=[],
        reasoning="From doc1",
    )
    mention2 = EntityMention(
        kind="gene",
        name="BRCA-1",  # Different name in doc2
        aliases=["breast cancer 1"],
        quotes=[],
        reasoning="From doc2",
    )
    state.global_entities = {
        "BRCA1": EntityRef(canonical="BRCA1", mentions=[mention1, mention2])
    }

    # Serialize
    state_dict = state.to_dict()

    # Verify serialized structure
    assert "global_entities" in state_dict
    assert "BRCA1" in state_dict["global_entities"]
    entity_data = state_dict["global_entities"]["BRCA1"]
    assert entity_data["canonical"] == "BRCA1"
    assert len(entity_data["mentions"]) == 2

    # Deserialize
    restored_state = State.from_dict(
        state_dict,
        topic="test",
        target_entity_types=["gene"],
        permitted_pairs={"gene": {"gene"}},
        resource_pool=pool,
    )

    # Verify global_entities restored
    assert "BRCA1" in restored_state.global_entities
    restored_ref = restored_state.global_entities["BRCA1"]
    assert restored_ref.canonical == "BRCA1"
    assert len(restored_ref.mentions) == 2

    # Verify aliases() aggregates from both mentions
    aliases = restored_ref.aliases()
    assert "BRCA-1" in aliases  # From mention2.name (differs from canonical)
    assert "breast cancer 1" in aliases  # From mention2.aliases


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
