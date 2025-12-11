"""Tests for updating entity references in PairAssessments after merging.

After entity consolidation, PairAssessment objects must have their entity
references updated to use the new canonical names.
"""

import pytest

from interaction_finder.extraction.stages.consolidate_entities import (
    _update_pair_entity_references,
)
from interaction_finder.extraction.models import (
    EntityMention,
    EntityRef,
    EvidenceQuality,
    PairAssessment,
)
from interaction_finder.extraction.state import State
from interaction_finder.resources import ResourceId


def make_evidence(level: int = 5) -> EvidenceQuality:
    return EvidenceQuality(
        directness="explicit",
        source_type="primary",
        specificity="mechanistic",
        language="definitive",
        overall=level,
    )


def build_permitted_pairs(kinds: list[str]) -> dict[str, set[str]]:
    return {kind: {kind} for kind in kinds}


class TestUpdatePairEntityReferences:
    """Test that pair assessments are updated after entity merges."""

    def test_updates_entity1_canonical(self):
        """Entity1 canonical name is updated when merged."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        resource = ResourceId(url="https://example.com", counter=1)
        entity1 = EntityMention(
            kind="gene", name="brca1", aliases=[], quotes=[], reasoning="test"
        )
        entity2 = EntityMention(
            kind="disease", name="Cancer", aliases=[], quotes=[], reasoning="test"
        )
        assessment = PairAssessment(
            resource_id=resource,
            entity1=EntityRef(canonical="brca1", mentions=[entity1]),
            entity2=EntityRef(canonical="Cancer", mentions=[entity2]),
            relationship="associated_with",
            quotes=[],
            evidence=make_evidence(8),
            reasoning="Test",
        )
        state.pair_assessments_by_resource[resource] = [assessment]
        state.validated_entities_by_resource[resource] = {
            "BRCA1": EntityRef(canonical="BRCA1", mentions=[entity1])
        }
        rules = {("brca1", "gene"): ("BRCA1", "llm:merge")}
        _update_pair_entity_references(rules, state)
        updated = state.pair_assessments_by_resource[resource][0]
        assert updated.entity1.canonical == "BRCA1"

    def test_updates_entity2_canonical(self):
        """Entity2 canonical name is updated when merged."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        resource = ResourceId(url="https://example.com", counter=1)
        entity1 = EntityMention(
            kind="gene", name="BRCA1", aliases=[], quotes=[], reasoning="test"
        )
        entity2 = EntityMention(
            kind="disease", name="cancer", aliases=[], quotes=[], reasoning="test"
        )
        assessment = PairAssessment(
            resource_id=resource,
            entity1=EntityRef(canonical="BRCA1", mentions=[entity1]),
            entity2=EntityRef(canonical="cancer", mentions=[entity2]),
            relationship="associated_with",
            quotes=[],
            evidence=make_evidence(8),
            reasoning="Test",
        )
        state.pair_assessments_by_resource[resource] = [assessment]
        state.validated_entities_by_resource[resource] = {
            "Cancer": EntityRef(canonical="Cancer", mentions=[entity2])
        }
        rules = {("cancer", "disease"): ("Cancer", "llm:merge")}
        _update_pair_entity_references(rules, state)
        updated = state.pair_assessments_by_resource[resource][0]
        assert updated.entity2.canonical == "Cancer"

    def test_updates_both_entities(self):
        """Both entities updated when both have merge rules."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        resource = ResourceId(url="https://example.com", counter=1)
        entity1 = EntityMention(
            kind="gene", name="brca1", aliases=[], quotes=[], reasoning="test"
        )
        entity2 = EntityMention(
            kind="disease", name="cancer", aliases=[], quotes=[], reasoning="test"
        )
        assessment = PairAssessment(
            resource_id=resource,
            entity1=EntityRef(canonical="brca1", mentions=[entity1]),
            entity2=EntityRef(canonical="cancer", mentions=[entity2]),
            relationship="associated_with",
            quotes=[],
            evidence=make_evidence(8),
            reasoning="Test",
        )
        state.pair_assessments_by_resource[resource] = [assessment]
        state.validated_entities_by_resource[resource] = {}
        rules = {
            ("brca1", "gene"): ("BRCA1", "llm:merge"),
            ("cancer", "disease"): ("Cancer", "llm:merge"),
        }
        _update_pair_entity_references(rules, state)
        updated = state.pair_assessments_by_resource[resource][0]
        assert updated.entity1.canonical == "BRCA1"
        assert updated.entity2.canonical == "Cancer"

    def test_preserves_mentions(self):
        """Mentions are preserved when canonical is updated."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        resource = ResourceId(url="https://example.com", counter=1)
        entity1 = EntityMention(
            kind="gene", name="brca1", aliases=["BRCA-1"], quotes=[], reasoning="test"
        )
        entity2 = EntityMention(
            kind="gene", name="TP53", aliases=[], quotes=[], reasoning="test"
        )
        assessment = PairAssessment(
            resource_id=resource,
            entity1=EntityRef(canonical="brca1", mentions=[entity1]),
            entity2=EntityRef(canonical="TP53", mentions=[entity2]),
            relationship="interacts_with",
            quotes=[],
            evidence=make_evidence(7),
            reasoning="Test",
        )
        state.pair_assessments_by_resource[resource] = [assessment]
        state.validated_entities_by_resource[resource] = {}
        rules = {("brca1", "gene"): ("BRCA1", "llm:merge")}
        _update_pair_entity_references(rules, state)
        updated = state.pair_assessments_by_resource[resource][0]
        # Canonical updated but mention preserved
        assert updated.entity1.canonical == "BRCA1"
        assert updated.entity1.mentions[0].name == "brca1"

    def test_no_change_when_no_matching_rules(self):
        """Entities unchanged when no rules match."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        resource = ResourceId(url="https://example.com", counter=1)
        entity1 = EntityMention(
            kind="gene", name="BRCA1", aliases=[], quotes=[], reasoning="test"
        )
        entity2 = EntityMention(
            kind="gene", name="TP53", aliases=[], quotes=[], reasoning="test"
        )
        assessment = PairAssessment(
            resource_id=resource,
            entity1=EntityRef(canonical="BRCA1", mentions=[entity1]),
            entity2=EntityRef(canonical="TP53", mentions=[entity2]),
            relationship="interacts_with",
            quotes=[],
            evidence=make_evidence(7),
            reasoning="Test",
        )
        state.pair_assessments_by_resource[resource] = [assessment]
        state.validated_entities_by_resource[resource] = {}
        rules = {("egfr", "gene"): ("EGFR", "llm:merge")}  # Unrelated rule
        _update_pair_entity_references(rules, state)
        updated = state.pair_assessments_by_resource[resource][0]
        assert updated.entity1.canonical == "BRCA1"
        assert updated.entity2.canonical == "TP53"

    def test_empty_rules_no_change(self):
        """Empty rules leave assessments unchanged."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        resource = ResourceId(url="https://example.com", counter=1)
        entity1 = EntityMention(
            kind="gene", name="BRCA1", aliases=[], quotes=[], reasoning="test"
        )
        entity2 = EntityMention(
            kind="gene", name="TP53", aliases=[], quotes=[], reasoning="test"
        )
        assessment = PairAssessment(
            resource_id=resource,
            entity1=EntityRef(canonical="BRCA1", mentions=[entity1]),
            entity2=EntityRef(canonical="TP53", mentions=[entity2]),
            relationship="interacts_with",
            quotes=[],
            evidence=make_evidence(7),
            reasoning="Test",
        )
        state.pair_assessments_by_resource[resource] = [assessment]
        _update_pair_entity_references({}, state)
        updated = state.pair_assessments_by_resource[resource][0]
        assert updated.entity1.canonical == "BRCA1"
        assert updated.entity2.canonical == "TP53"

    def test_multiple_assessments_updated(self):
        """Multiple assessments in same resource all updated."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        resource = ResourceId(url="https://example.com", counter=1)
        entity_brca = EntityMention(
            kind="gene", name="brca1", aliases=[], quotes=[], reasoning="test"
        )
        entity_tp53 = EntityMention(
            kind="gene", name="TP53", aliases=[], quotes=[], reasoning="test"
        )
        entity_egfr = EntityMention(
            kind="gene", name="EGFR", aliases=[], quotes=[], reasoning="test"
        )
        assessment1 = PairAssessment(
            resource_id=resource,
            entity1=EntityRef(canonical="brca1", mentions=[entity_brca]),
            entity2=EntityRef(canonical="TP53", mentions=[entity_tp53]),
            relationship="interacts_with",
            quotes=[],
            evidence=make_evidence(7),
            reasoning="Test1",
        )
        assessment2 = PairAssessment(
            resource_id=resource,
            entity1=EntityRef(canonical="brca1", mentions=[entity_brca]),
            entity2=EntityRef(canonical="EGFR", mentions=[entity_egfr]),
            relationship="regulates",
            quotes=[],
            evidence=make_evidence(6),
            reasoning="Test2",
        )
        state.pair_assessments_by_resource[resource] = [assessment1, assessment2]
        state.validated_entities_by_resource[resource] = {}
        rules = {("brca1", "gene"): ("BRCA1", "llm:merge")}
        _update_pair_entity_references(rules, state)
        # Both assessments should have entity1 updated
        assert (
            state.pair_assessments_by_resource[resource][0].entity1.canonical == "BRCA1"
        )
        assert (
            state.pair_assessments_by_resource[resource][1].entity1.canonical == "BRCA1"
        )

    def test_applies_across_multiple_resources(self):
        """Merge rules apply consistently across all documents."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        # Three different resources with same entity in different cases
        resource1 = ResourceId(url="https://doc1.com", counter=1)
        resource2 = ResourceId(url="https://doc2.com", counter=2)
        resource3 = ResourceId(url="https://doc3.com", counter=3)
        # Create assessments for each resource with lowercase 'brca1'
        for resource in [resource1, resource2, resource3]:
            entity1 = EntityMention(
                kind="gene", name="brca1", aliases=[], quotes=[], reasoning="test"
            )
            entity2 = EntityMention(
                kind="disease", name="Cancer", aliases=[], quotes=[], reasoning="test"
            )
            assessment = PairAssessment(
                resource_id=resource,
                entity1=EntityRef(canonical="brca1", mentions=[entity1]),
                entity2=EntityRef(canonical="Cancer", mentions=[entity2]),
                relationship="associated_with",
                quotes=[],
                evidence=make_evidence(7),
                reasoning="Test",
            )
            state.pair_assessments_by_resource[resource] = [assessment]
            state.validated_entities_by_resource[resource] = {}
        # Single rule should apply to all three resources
        rules = {("brca1", "gene"): ("BRCA1", "llm:merge")}
        _update_pair_entity_references(rules, state)
        # Verify all three resources have updated canonical
        for resource in [resource1, resource2, resource3]:
            updated = state.pair_assessments_by_resource[resource][0]
            assert updated.entity1.canonical == "BRCA1", f"Failed for {resource.url}"
            # entity2 should be unchanged
            assert updated.entity2.canonical == "Cancer"
