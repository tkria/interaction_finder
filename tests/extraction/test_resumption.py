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
    state.relationship_oppositions = {"activates": {"inhibits"}}

    # Serialize
    state_dict = state.to_dict()

    # Check serialized structure
    assert "entities_merged" in state_dict
    assert state_dict["entities_merged"] == 5
    assert state_dict["quotes_validated"] == 10
    assert state_dict["relationship_polarities"] == {"activates": "positive"}
    assert state_dict["relationship_oppositions"] == {"activates": ["inhibits"]}

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
    assert restored_state.relationship_oppositions == {"activates": {"inhibits"}}


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
    assert "consolidation_rules" in state_dict
    assert "agent_merge_cache" in state_dict
    assert "relationship_oppositions" in state_dict

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

    # Add consolidation rules (tuple keys)
    state.consolidation_rules = {
        ("brca1", "gene"): ("BRCA1", "auto:0:original:exact"),
        ("brca-1", "gene"): ("BRCA1", "auto:1:hyphen:fuzzy"),
    }

    # Add agent merge cache (tuple keys)
    state.agent_merge_cache = {
        ("BRCA1", "BRCA-1", "gene"): ("BRCA1", "merge into canonical"),
    }

    # Serialize
    state_dict = state.to_dict()

    # Check tuple keys converted to strings
    assert "brca1|gene" in state_dict["consolidation_rules"]
    assert "BRCA1|BRCA-1|gene" in state_dict["agent_merge_cache"]

    # Deserialize
    restored_state = State.from_dict(
        state_dict,
        topic="test",
        target_entity_types=["gene"],
        permitted_pairs={"gene": {"gene"}},
        resource_pool=pool,
    )

    # Verify restoration
    assert ("brca1", "gene") in restored_state.consolidation_rules
    assert restored_state.consolidation_rules[("brca1", "gene")] == (
        "BRCA1",
        "auto:0:original:exact",
    )
    assert ("BRCA1", "BRCA-1", "gene") in restored_state.agent_merge_cache


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
