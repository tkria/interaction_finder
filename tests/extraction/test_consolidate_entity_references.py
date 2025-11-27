"""Tests for entity reference updates in PairAssessments after merging.

The ConsolidateEntitiesNode now updates entity references in PairAssessments after
merging entities globally. These tests verify that:
1. Entity references in assessments are updated to use merged canonical names
2. Original names are preserved in aliases
3. Reasoning tracks the merge operation
4. Lookups handle normalized name matching correctly
"""

from unittest.mock import MagicMock

import pytest
from pydantic_graph import GraphRunContext

from interaction_finder.extraction.models import (
    EntityMention,
    EntityRef,
    PairAssessment,
)
from interaction_finder.extraction.nodes import ConsolidateEntitiesNode
from interaction_finder.extraction.state import State
from interaction_finder.extraction.utils import build_permitted_pairs
from interaction_finder.resources import ResourceId, ResourcePool


def ref_map(data: dict[str, EntityMention]) -> dict[str, EntityRef]:
    return {
        name: EntityRef(canonical=name, mentions=[mention])
        for name, mention in data.items()
    }


@pytest.fixture
def mock_deps():
    """Create mock dependencies."""
    deps = MagicMock()
    deps.config = MagicMock()
    deps.logger = MagicMock()
    deps.resource_pool = ResourcePool()
    return deps


# TestFindCanonicalName class removed - _find_canonical_name no longer exists
# The new implementation uses canonical names directly in rules, eliminating
# the need for normalized-to-canonical lookups.


# NOTE: TestFindCanonicalName tests removed
# _find_canonical_name() no longer exists in the new implementation.
# Rules now use canonical names directly, eliminating normalized-to-canonical lookups.


class TestUpdateEntityInAssessment:
    """EntityRef-focused update tests using _update_pair_entity_references."""

    def _make_assessment(self) -> PairAssessment:
        entity1 = EntityMention(
            kind="gene", name="BRCA", aliases=["BRCA-1"], quotes=[], reasoning="Original"
        )
        entity2 = EntityMention(
            kind="disease",
            name="Cancer",
            aliases=["CA"],
            quotes=[],
            reasoning="Original",
        )
        return PairAssessment(
            resource_id=ResourceId(url="https://example.com", counter=1),
            entity1=EntityRef(canonical=entity1.name, mentions=[entity1]),
            entity2=EntityRef(canonical=entity2.name, mentions=[entity2]),
            relationship="associated_with",
            quotes=[],
            confidence="high",
            reasoning="Test",
        )

    def test_updates_entity1_canonical_preserves_mentions(self, mock_deps):
        """Entity1 canonical name is updated; mentions stay unchanged."""
        node = ConsolidateEntitiesNode()
        assessment = self._make_assessment()
        resource_id = assessment.resource_id
        state = State(
            topic="test", target_entity_types=["gene", "disease"], permitted_pairs={}
        )
        state.validated_entities_by_resource[resource_id] = {}
        state.pair_assessments_by_resource[resource_id] = [assessment]
        ctx = GraphRunContext(state=state, deps=mock_deps)

        node._update_pair_entity_references({("brca", "gene"): ("BRCA1", "test")}, ctx)

        updated = state.pair_assessments_by_resource[resource_id][0]
        assert updated.entity1.canonical == "BRCA1"
        # Mentions are preserved (no alias or reasoning mutation)
        assert updated.entity1.mentions[0].name == "BRCA"
        assert updated.entity1.mentions[0].aliases == ["BRCA-1"]
        assert updated.entity1.mentions[0].reasoning == "Original"
        # Entity2 untouched
        assert updated.entity2.canonical == "Cancer"

    def test_updates_entity2_canonical(self, mock_deps):
        """Entity2 canonical updated independently."""
        node = ConsolidateEntitiesNode()
        assessment = self._make_assessment()
        resource_id = assessment.resource_id
        state = State(
            topic="test", target_entity_types=["gene", "disease"], permitted_pairs={}
        )
        state.validated_entities_by_resource[resource_id] = {}
        state.pair_assessments_by_resource[resource_id] = [assessment]
        ctx = GraphRunContext(state=state, deps=mock_deps)

        node._update_pair_entity_references(
            {("cancer", "disease"): ("Neoplasm", "test")}, ctx
        )

        updated = state.pair_assessments_by_resource[resource_id][0]
        assert updated.entity2.canonical == "Neoplasm"
        assert updated.entity2.mentions[0].name == "Cancer"
        assert updated.entity2.mentions[0].aliases == ["CA"]

class TestUpdatePairEntityReferences:
    """Test _update_pair_entity_references method."""

    def test_updates_both_entities_in_assessment(self, mock_deps):
        """Should update both entity references when both are merged."""
        node = ConsolidateEntitiesNode()
        resource_id = ResourceId(url="https://example.com", counter=1)

        # Create state with assessments
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )

        # Add entities (post-merge)
        state.validated_entities_by_resource[resource_id] = ref_map(
            {
                "BRCA1": EntityMention(
                    kind="gene", name="BRCA1", aliases=[], quotes=[], reasoning="Merged"
                ),
                "Cancer": EntityMention(
                    kind="disease",
                    name="Cancer",
                    aliases=[],
                    quotes=[],
                    reasoning="Merged",
                ),
            }
        )

        # Add assessment with old names
        state.pair_assessments_by_resource[resource_id] = [
            PairAssessment(
                resource_id=resource_id,
                entity1=EntityRef(
                    canonical="BRCA",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="BRCA",
                            aliases=[],
                            quotes=[],
                            reasoning="Original",
                        )
                    ],
                ),
                entity2=EntityRef(
                    canonical="cancer",
                    mentions=[
                        EntityMention(
                            kind="disease",
                            name="cancer",
                            aliases=[],
                            quotes=[],
                            reasoning="Original",
                        )
                    ],
                ),
                relationship="associated_with",
                quotes=[],
                confidence="high",
                reasoning="Test",
            )
        ]

        ctx = GraphRunContext(state=state, deps=mock_deps)

        # Merge rules keyed by normalized form, target is canonical
        merge_rules = {
            ("brca", "gene"): ("BRCA1", "test"),
            ("cancer", "disease"): ("Cancer", "test"),
        }

        node._update_pair_entity_references(merge_rules, ctx)

        assessment = state.pair_assessments_by_resource[resource_id][0]
        assert assessment.entity1.canonical == "BRCA1"
        assert assessment.entity2.canonical == "Cancer"

    def test_handles_no_merge_needed(self, mock_deps):
        """Should leave assessment unchanged if no merges apply."""
        node = ConsolidateEntitiesNode()
        resource_id = ResourceId(url="https://example.com", counter=1)

        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )

        state.validated_entities_by_resource[resource_id] = ref_map(
            {
                "BRCA1": EntityMention(
                    kind="gene", name="BRCA1", aliases=[], quotes=[], reasoning="Original"
                )
            }
        )

        entity1 = EntityMention(
            kind="gene",
            name="BRCA1",
            aliases=["BRCA-1"],
            quotes=[],
            reasoning="Original",
        )
        entity2 = EntityMention(
            kind="gene",
            name="TP53",
            aliases=[],
            quotes=[],
            reasoning="Original",
        )
        original_assessment = PairAssessment(
            resource_id=resource_id,
            entity1=EntityRef(canonical=entity1.name, mentions=[entity1]),
            entity2=EntityRef(canonical=entity2.name, mentions=[entity2]),
            relationship="interacts_with",
            quotes=[],
            confidence="medium",
            reasoning="Test",
        )

        state.pair_assessments_by_resource[resource_id] = [original_assessment]

        ctx = GraphRunContext(state=state, deps=mock_deps)

        # Empty merge rules - no changes needed
        merge_rules = {}

        node._update_pair_entity_references(merge_rules, ctx)

        assessment = state.pair_assessments_by_resource[resource_id][0]
        assert assessment.entity1.canonical == "BRCA1"
        assert assessment.entity2.canonical == "TP53"
        # Aliases and reasoning should be unchanged
        assert assessment.entity1.mentions[0].aliases == ["BRCA-1"]
        assert "MERGED_FROM" not in assessment.entity1.mentions[0].reasoning

    def test_handles_only_entity1_merged(self, mock_deps):
        """Should update only entity1 if entity2 is not merged."""
        node = ConsolidateEntitiesNode()
        resource_id = ResourceId(url="https://example.com", counter=1)

        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )

        state.validated_entities_by_resource[resource_id] = ref_map(
            {
                "BRCA1": EntityMention(
                    kind="gene", name="BRCA1", aliases=[], quotes=[], reasoning="Merged"
                ),
                "TP53": EntityMention(
                    kind="gene", name="TP53", aliases=[], quotes=[], reasoning="Original"
                ),
            }
        )

        state.pair_assessments_by_resource[resource_id] = [
            PairAssessment(
                resource_id=resource_id,
                entity1=EntityRef(
                    canonical="BRCA",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="BRCA",
                            aliases=[],
                            quotes=[],
                            reasoning="Original",
                        )
                    ],
                ),
                entity2=EntityRef(
                    canonical="TP53",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="TP53",
                            aliases=[],
                            quotes=[],
                            reasoning="Original",
                        )
                    ],
                ),
                relationship="interacts_with",
                quotes=[],
                confidence="high",
                reasoning="Test",
            )
        ]

        ctx = GraphRunContext(state=state, deps=mock_deps)

        # New format: canonical names
        merge_rules = {
            ("brca", "gene"): ("BRCA1", "test"),
        }

        node._update_pair_entity_references(merge_rules, ctx)

        assessment = state.pair_assessments_by_resource[resource_id][0]
        assert assessment.entity1.canonical == "BRCA1"
        assert assessment.entity2.canonical == "TP53"  # Unchanged
        assert "MERGED_FROM" not in assessment.entity2.mentions[0].reasoning

    def test_handles_multiple_assessments(self, mock_deps):
        """Should update all assessments in a resource."""
        node = ConsolidateEntitiesNode()
        resource_id = ResourceId(url="https://example.com", counter=1)

        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )

        state.validated_entities_by_resource[resource_id] = ref_map(
            {
                "BRCA1": EntityMention(
                    kind="gene", name="BRCA1", aliases=[], quotes=[], reasoning="Merged"
                ),
                "Cancer": EntityMention(
                    kind="disease",
                    name="Cancer",
                    aliases=[],
                    quotes=[],
                    reasoning="Original",
                ),
                "TP53": EntityMention(
                    kind="gene", name="TP53", aliases=[], quotes=[], reasoning="Original"
                ),
            }
        )

        state.pair_assessments_by_resource[resource_id] = [
            PairAssessment(
                resource_id=resource_id,
                entity1=EntityRef(
                    canonical="BRCA",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="BRCA",
                            aliases=[],
                            quotes=[],
                            reasoning="Original",
                        )
                    ],
                ),
                entity2=EntityRef(
                    canonical="Cancer",
                    mentions=[
                        EntityMention(
                            kind="disease",
                            name="Cancer",
                            aliases=[],
                            quotes=[],
                            reasoning="Original",
                        )
                    ],
                ),
                relationship="associated_with",
                quotes=[],
                confidence="high",
                reasoning="Test 1",
            ),
            PairAssessment(
                resource_id=resource_id,
                entity1=EntityRef(
                    canonical="BRCA",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="BRCA",
                            aliases=[],
                            quotes=[],
                            reasoning="Original",
                        )
                    ],
                ),
                entity2=EntityRef(
                    canonical="TP53",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="TP53",
                            aliases=[],
                            quotes=[],
                            reasoning="Original",
                        )
                    ],
                ),
                relationship="interacts_with",
                quotes=[],
                confidence="medium",
                reasoning="Test 2",
            ),
        ]

        ctx = GraphRunContext(state=state, deps=mock_deps)

        # New format: canonical names
        merge_rules = {
            ("brca", "gene"): ("BRCA1", "test"),
        }

        node._update_pair_entity_references(merge_rules, ctx)

        # Both assessments should have entity1 updated
        for assessment in state.pair_assessments_by_resource[resource_id]:
            assert assessment.entity1.canonical == "BRCA1"

    def test_handles_multiple_resources(self, mock_deps):
        """Should update assessments across multiple resources."""
        node = ConsolidateEntitiesNode()
        resource_id1 = ResourceId(url="https://example.com/doc1", counter=1)
        resource_id2 = ResourceId(url="https://example.com/doc2", counter=2)

        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )

        # Both resources have merged entity
        for resource_id in [resource_id1, resource_id2]:
            state.validated_entities_by_resource[resource_id] = ref_map(
                {
                    "BRCA1": EntityMention(
                        kind="gene",
                        name="BRCA1",
                        aliases=[],
                        quotes=[],
                        reasoning="Merged",
                    )
                }
            )

            state.pair_assessments_by_resource[resource_id] = [
                PairAssessment(
                    resource_id=resource_id,
                    entity1=EntityRef(
                        canonical="BRCA",
                        mentions=[
                            EntityMention(
                                kind="gene",
                                name="BRCA",
                                aliases=[],
                                quotes=[],
                                reasoning="Original",
                            )
                        ],
                    ),
                    entity2=EntityRef(
                        canonical="BRCA",
                        mentions=[
                            EntityMention(
                                kind="gene",
                                name="BRCA",
                                aliases=[],
                                quotes=[],
                                reasoning="Original",
                            )
                        ],
                    ),
                    relationship="self_reference",
                    quotes=[],
                    confidence="low",
                    reasoning="Test",
                )
            ]

        ctx = GraphRunContext(state=state, deps=mock_deps)

        # New format: canonical names
        merge_rules = {
            ("brca", "gene"): ("BRCA1", "test"),
        }

        node._update_pair_entity_references(merge_rules, ctx)

        # Both resources should be updated
        for resource_id in [resource_id1, resource_id2]:
            assessment = state.pair_assessments_by_resource[resource_id][0]
            assert assessment.entity1.canonical == "BRCA1"
            assert assessment.entity2.canonical == "BRCA1"
