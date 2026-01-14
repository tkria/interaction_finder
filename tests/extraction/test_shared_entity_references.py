"""Tests for shared EntityRef references during entity consolidation.

This verifies that:
1. PairAssessments see entity renames after consolidation
2. Entity data (mentions, aliases) is properly aggregated after merges
3. The mutation contract for EntityMention objects is maintained
"""

import pytest

from interaction_finder.extraction.models import (
    EntityMention,
    EntityRef,
    EvidenceQuality,
    PairAssessment,
)
from interaction_finder.extraction.stages.consolidate_entities import (
    _apply_merge_rules_globally,
    _update_pair_entity_references,
)
from interaction_finder.extraction.state import State
from interaction_finder.resources import ResourceId


def build_permitted_pairs(kinds: list[str]) -> dict[str, set[str]]:
    """Build permitted pairs dict for testing."""
    return {kind: {kind} for kind in kinds}


def make_evidence(level: int = 5) -> EvidenceQuality:
    """Create a default EvidenceQuality for testing."""
    return EvidenceQuality(
        directness="explicit",
        source_type="primary",
        specificity="mechanistic",
        language="definitive",
        overall=level,
    )


class TestPairAssessmentReferenceUpdates:
    """Test that PairAssessments are updated when entities are renamed/merged."""

    def test_rename_visible_in_assessments(self):
        """After entity rename, PairAssessment.entity1.canonical is updated."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        resource = ResourceId(url="https://doc1.com", counter=0)
        # Create entities: BRCA (parent), brca (child)
        brca_parent = EntityMention(
            kind="gene",
            name="BRCA",
            aliases=[],
            quotes=[],
            reasoning="Parent",
        )
        brca_child = EntityMention(
            kind="gene",
            name="brca",
            aliases=[],
            quotes=[],
            reasoning="Child",
        )
        state.validated_entities_by_resource = {
            resource: {
                "BRCA": EntityRef(canonical="BRCA", mentions=[brca_parent]),
                "brca": EntityRef(canonical="brca", mentions=[brca_child]),
            }
        }
        # Create assessment referencing the child entity
        assessment = PairAssessment(
            topic_relevance=3,
            resource_id=resource,
            entity1=EntityRef(canonical="brca", mentions=[brca_child]),
            entity2=EntityRef(canonical="BRCA", mentions=[brca_parent]),
            relationship="interacts_with",
            quotes=[],
            evidence=make_evidence(8),
            reasoning="Test",
        )
        state.pair_assessments_by_resource = {resource: [assessment]}
        # Merge child → parent
        merge_rules = {("brca", "gene"): ("BRCA", "0:merge:exact", None)}
        _apply_merge_rules_globally(merge_rules, state)
        _update_pair_entity_references(merge_rules, state)
        # Assessment should now reference BRCA for entity1
        assert assessment.entity1.canonical == "BRCA"

    def test_entity2_update_independent(self):
        """Entity2 can be updated independently of entity1."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        resource = ResourceId(url="https://doc1.com", counter=0)
        gene_entity = EntityMention(
            kind="gene",
            name="TP53",
            aliases=[],
            quotes=[],
            reasoning="gene",
        )
        disease_child = EntityMention(
            kind="disease",
            name="cancer",
            aliases=[],
            quotes=[],
            reasoning="disease child",
        )
        disease_parent = EntityMention(
            kind="disease",
            name="Cancer",
            aliases=[],
            quotes=[],
            reasoning="disease parent",
        )
        state.validated_entities_by_resource = {
            resource: {
                "TP53": EntityRef(canonical="TP53", mentions=[gene_entity]),
                "cancer": EntityRef(canonical="cancer", mentions=[disease_child]),
                "Cancer": EntityRef(canonical="Cancer", mentions=[disease_parent]),
            }
        }
        # Assessment: TP53 (gene) associated with cancer (disease)
        assessment = PairAssessment(
            topic_relevance=3,
            resource_id=resource,
            entity1=EntityRef(canonical="TP53", mentions=[gene_entity]),
            entity2=EntityRef(canonical="cancer", mentions=[disease_child]),
            relationship="associated_with",
            quotes=[],
            evidence=make_evidence(7),
            reasoning="Test",
        )
        state.pair_assessments_by_resource = {resource: [assessment]}
        # Only merge the disease entity
        merge_rules = {("cancer", "disease"): ("Cancer", "0:merge:exact", None)}
        _apply_merge_rules_globally(merge_rules, state)
        _update_pair_entity_references(merge_rules, state)
        # Entity1 unchanged (gene)
        assert assessment.entity1.canonical == "TP53"
        # Entity2 updated (disease)
        assert assessment.entity2.canonical == "Cancer"


class TestPairAssessmentMergeUpdates:
    """Test that merges properly aggregate entity data."""

    def test_same_doc_merge_combines_mentions(self):
        """When entities in same document merge, mentions are combined."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        resource = ResourceId(url="https://doc1.com", counter=0)
        # Parent and child have different mentions
        parent_mention = EntityMention(
            kind="gene",
            name="BRCA",
            aliases=["Breast Cancer Gene"],
            quotes=[],
            reasoning="Parent mention",
        )
        child_mention = EntityMention(
            kind="gene",
            name="brca",
            aliases=["brca1"],
            quotes=[],
            reasoning="Child mention",
        )
        state.validated_entities_by_resource = {
            resource: {
                "BRCA": EntityRef(canonical="BRCA", mentions=[parent_mention]),
                "brca": EntityRef(canonical="brca", mentions=[child_mention]),
            }
        }
        # Merge child → parent
        merge_rules = {("brca", "gene"): ("BRCA", "0:merge:exact", None)}
        _apply_merge_rules_globally(merge_rules, state)
        # After merge, BRCA should have both mentions
        merged_entity = state.validated_entities_by_resource[resource]["BRCA"]
        assert len(merged_entity.mentions) == 2
        # Child entity should be removed
        assert "brca" not in state.validated_entities_by_resource[resource]

    def test_cross_doc_entities_merged_separately(self):
        """Entities in different documents are merged to same canonical."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        resource1 = ResourceId(url="https://doc1.com", counter=0)
        resource2 = ResourceId(url="https://doc2.com", counter=1)
        # Doc1 has BRCA (parent)
        parent_mention = EntityMention(
            kind="gene",
            name="BRCA",
            aliases=[],
            quotes=[],
            reasoning="Doc1",
        )
        # Doc2 has brca (child variant)
        child_mention = EntityMention(
            kind="gene",
            name="brca",
            aliases=[],
            quotes=[],
            reasoning="Doc2",
        )
        state.validated_entities_by_resource = {
            resource1: {"BRCA": EntityRef(canonical="BRCA", mentions=[parent_mention])},
            resource2: {"brca": EntityRef(canonical="brca", mentions=[child_mention])},
        }
        # Merge rule applies to both docs
        merge_rules = {("brca", "gene"): ("BRCA", "0:merge:exact", None)}
        _apply_merge_rules_globally(merge_rules, state)
        # Doc1 still has BRCA
        assert "BRCA" in state.validated_entities_by_resource[resource1]
        # Doc2 now has BRCA (renamed from brca)
        assert "BRCA" in state.validated_entities_by_resource[resource2]
        assert "brca" not in state.validated_entities_by_resource[resource2]


class TestEntityMutationContract:
    """Document the mutation contract: EntityMention objects can be shared."""

    def test_shared_object_mutation_visible(self):
        """Mutations to shared EntityMention objects are visible everywhere."""
        entity = EntityMention(
            kind="gene",
            name="BRCA1",
            aliases=["Breast Cancer 1"],
            quotes=[],
            reasoning="Test",
        )
        # Store same entity in two references
        ref1 = EntityRef(canonical="BRCA1", mentions=[entity])
        ref2 = EntityRef(canonical="BRCA1", mentions=[entity])
        # Mutate via one reference
        ref1.mentions[0].aliases.append("New alias")
        # Both see the change
        assert "New alias" in ref1.mentions[0].aliases
        assert "New alias" in ref2.mentions[0].aliases

    def test_dict_key_update_required(self):
        """Dict keys must be explicitly updated when canonical changes."""
        entity = EntityMention(
            kind="gene",
            name="BRCA1",
            aliases=[],
            quotes=[],
            reasoning="Test",
        )
        ref = EntityRef(canonical="BRCA1", mentions=[entity])
        entities = {"BRCA1": ref}
        # Update canonical on the EntityRef
        ref.canonical = "BRCA-renamed"
        # Dict key is NOT automatically updated
        assert "BRCA1" in entities
        assert "BRCA-renamed" not in entities
        # Must explicitly move the entry
        entities["BRCA-renamed"] = entities.pop("BRCA1")
        assert "BRCA-renamed" in entities
        assert "BRCA1" not in entities
