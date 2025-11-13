"""Tests for entity reference updates in PairAssessments after merging.

The MergeEntitiesNode now updates entity references in PairAssessments after
merging entities globally. These tests verify that:
1. Entity references in assessments are updated to use merged canonical names
2. Original names are preserved in aliases
3. Reasoning tracks the merge operation
4. Lookups handle normalized name matching correctly
"""

from unittest.mock import MagicMock

import pytest
from pydantic_graph import GraphRunContext

from interaction_finder.extraction.models import EntityMention, PairAssessment
from interaction_finder.extraction.nodes import MergeEntitiesNode
from interaction_finder.extraction.state import State
from interaction_finder.extraction.utils import build_permitted_pairs
from interaction_finder.resources import ResourceId, ResourcePool


@pytest.fixture
def mock_deps():
    """Create mock dependencies."""
    deps = MagicMock()
    deps.config = MagicMock()
    deps.logger = MagicMock()
    deps.resource_pool = ResourcePool()
    return deps


class TestFindCanonicalName:
    """Test _find_canonical_name helper method."""

    def test_finds_exact_match(self):
        """Should find entity by normalized name match."""
        node = MergeEntitiesNode()
        entities = {
            "BRCA1": EntityMention(
                kind="gene",
                name="BRCA1",
                aliases=[],
                quotes=[],
                reasoning="test",
            ),
            "TP53": EntityMention(
                kind="gene", name="TP53", aliases=[], quotes=[], reasoning="test"
            ),
        }

        result = node._find_canonical_name("brca1", entities, "gene")
        assert result == "BRCA1"

    def test_returns_none_for_no_match(self):
        """Should return None if no matching entity found."""
        node = MergeEntitiesNode()
        entities = {
            "BRCA1": EntityMention(
                kind="gene",
                name="BRCA1",
                aliases=[],
                quotes=[],
                reasoning="test",
            )
        }

        result = node._find_canonical_name("tp53", entities, "gene")
        assert result is None

    def test_respects_kind_filter(self):
        """Should not match if kind differs."""
        node = MergeEntitiesNode()
        entities = {
            "BRCA1": EntityMention(
                kind="gene",
                name="BRCA1",
                aliases=[],
                quotes=[],
                reasoning="test",
            )
        }

        result = node._find_canonical_name("brca1", entities, "disease")
        assert result is None

    def test_handles_whitespace_normalization(self):
        """Should match despite whitespace differences in normalized input."""
        from interaction_finder.extraction.utils import normalize_for_comparison

        node = MergeEntitiesNode()
        entities = {
            "Alzheimer's disease": EntityMention(
                kind="disease",
                name="Alzheimer's disease",
                aliases=[],
                quotes=[],
                reasoning="test",
            )
        }

        # _find_canonical_name expects already-normalized input
        normalized_search = normalize_for_comparison("alzheimer's  disease")
        result = node._find_canonical_name(normalized_search, entities, "disease")
        assert result == "Alzheimer's disease"


class TestUpdateEntityInAssessment:
    """Test _update_entity_in_assessment helper method."""

    def test_updates_entity_name(self, mock_deps):
        """Should update entity name to merged canonical name."""
        node = MergeEntitiesNode()
        assessment = PairAssessment(
            resource_id=ResourceId(url="https://example.com", counter=1),
            entity1=EntityMention(
                kind="gene", name="BRCA", aliases=[], quotes=[], reasoning="Original"
            ),
            entity2=EntityMention(
                kind="disease",
                name="Cancer",
                aliases=[],
                quotes=[],
                reasoning="Original",
            ),
            relationship="associated_with",
            quotes=[],
            confidence="high",
            reasoning="Test",
        )

        node._update_entity_in_assessment(assessment, "entity1", "BRCA1")

        assert assessment.entity1.name == "BRCA1"
        assert assessment.entity2.name == "Cancer"  # Unchanged

    def test_adds_old_name_to_aliases(self):
        """Should add original name to aliases list."""
        node = MergeEntitiesNode()
        assessment = PairAssessment(
            resource_id=ResourceId(url="https://example.com", counter=1),
            entity1=EntityMention(
                kind="gene",
                name="BRCA",
                aliases=["BRCA-1"],
                quotes=[],
                reasoning="Original",
            ),
            entity2=EntityMention(
                kind="disease",
                name="Cancer",
                aliases=[],
                quotes=[],
                reasoning="Original",
            ),
            relationship="associated_with",
            quotes=[],
            confidence="high",
            reasoning="Test",
        )

        node._update_entity_in_assessment(assessment, "entity1", "BRCA1")

        assert "BRCA" in assessment.entity1.aliases
        assert "BRCA-1" in assessment.entity1.aliases
        assert assessment.entity1.name == "BRCA1"

    def test_updates_reasoning_with_merge_info(self):
        """Should append merge information to reasoning."""
        node = MergeEntitiesNode()
        assessment = PairAssessment(
            resource_id=ResourceId(url="https://example.com", counter=1),
            entity1=EntityMention(
                kind="gene", name="BRCA", aliases=[], quotes=[], reasoning="Original"
            ),
            entity2=EntityMention(
                kind="disease",
                name="Cancer",
                aliases=[],
                quotes=[],
                reasoning="Original",
            ),
            relationship="associated_with",
            quotes=[],
            confidence="high",
            reasoning="Test",
        )

        node._update_entity_in_assessment(assessment, "entity1", "BRCA1")

        assert "MERGED_FROM(BRCA)" in assessment.entity1.reasoning
        assert "Original" in assessment.entity1.reasoning

    def test_handles_entity2_update(self):
        """Should update entity2 when specified."""
        node = MergeEntitiesNode()
        assessment = PairAssessment(
            resource_id=ResourceId(url="https://example.com", counter=1),
            entity1=EntityMention(
                kind="gene", name="BRCA1", aliases=[], quotes=[], reasoning="Original"
            ),
            entity2=EntityMention(
                kind="disease",
                name="breast cancer",
                aliases=[],
                quotes=[],
                reasoning="Original",
            ),
            relationship="associated_with",
            quotes=[],
            confidence="high",
            reasoning="Test",
        )

        node._update_entity_in_assessment(assessment, "entity2", "Breast Cancer")

        assert assessment.entity1.name == "BRCA1"  # Unchanged
        assert assessment.entity2.name == "Breast Cancer"
        assert "breast cancer" in assessment.entity2.aliases


class TestUpdatePairEntityReferences:
    """Test _update_pair_entity_references method."""

    def test_updates_both_entities_in_assessment(self, mock_deps):
        """Should update both entity references when both are merged."""
        node = MergeEntitiesNode()
        resource_id = ResourceId(url="https://example.com", counter=1)

        # Create state with assessments
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )

        # Add entities (post-merge)
        state.validated_entities_by_resource[resource_id] = {
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

        # Add assessment with old names
        state.pair_assessments_by_resource[resource_id] = [
            PairAssessment(
                resource_id=resource_id,
                entity1=EntityMention(
                    kind="gene",
                    name="BRCA",
                    aliases=[],
                    quotes=[],
                    reasoning="Original",
                ),
                entity2=EntityMention(
                    kind="disease",
                    name="cancer",
                    aliases=[],
                    quotes=[],
                    reasoning="Original",
                ),
                relationship="associated_with",
                quotes=[],
                confidence="high",
                reasoning="Test",
            )
        ]

        ctx = GraphRunContext(state=state, deps=mock_deps)

        # Merge rules: BRCA→BRCA1, cancer→Cancer
        merge_rules = {
            ("brca", "gene"): "brca1",
            ("cancer", "disease"): "cancer",
        }

        node._update_pair_entity_references(merge_rules, ctx)

        assessment = state.pair_assessments_by_resource[resource_id][0]
        assert assessment.entity1.name == "BRCA1"
        assert assessment.entity2.name == "Cancer"
        assert "BRCA" in assessment.entity1.aliases
        assert "cancer" in assessment.entity2.aliases

    def test_handles_no_merge_needed(self, mock_deps):
        """Should leave assessment unchanged if no merges apply."""
        node = MergeEntitiesNode()
        resource_id = ResourceId(url="https://example.com", counter=1)

        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )

        state.validated_entities_by_resource[resource_id] = {
            "BRCA1": EntityMention(
                kind="gene", name="BRCA1", aliases=[], quotes=[], reasoning="Original"
            )
        }

        original_assessment = PairAssessment(
            resource_id=resource_id,
            entity1=EntityMention(
                kind="gene",
                name="BRCA1",
                aliases=["BRCA-1"],
                quotes=[],
                reasoning="Original",
            ),
            entity2=EntityMention(
                kind="gene",
                name="TP53",
                aliases=[],
                quotes=[],
                reasoning="Original",
            ),
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
        assert assessment.entity1.name == "BRCA1"
        assert assessment.entity2.name == "TP53"
        # Aliases and reasoning should be unchanged
        assert assessment.entity1.aliases == ["BRCA-1"]
        assert "MERGED_FROM" not in assessment.entity1.reasoning

    def test_handles_only_entity1_merged(self, mock_deps):
        """Should update only entity1 if entity2 is not merged."""
        node = MergeEntitiesNode()
        resource_id = ResourceId(url="https://example.com", counter=1)

        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )

        state.validated_entities_by_resource[resource_id] = {
            "BRCA1": EntityMention(
                kind="gene", name="BRCA1", aliases=[], quotes=[], reasoning="Merged"
            ),
            "TP53": EntityMention(
                kind="gene", name="TP53", aliases=[], quotes=[], reasoning="Original"
            ),
        }

        state.pair_assessments_by_resource[resource_id] = [
            PairAssessment(
                resource_id=resource_id,
                entity1=EntityMention(
                    kind="gene",
                    name="BRCA",
                    aliases=[],
                    quotes=[],
                    reasoning="Original",
                ),
                entity2=EntityMention(
                    kind="gene",
                    name="TP53",
                    aliases=[],
                    quotes=[],
                    reasoning="Original",
                ),
                relationship="interacts_with",
                quotes=[],
                confidence="high",
                reasoning="Test",
            )
        ]

        ctx = GraphRunContext(state=state, deps=mock_deps)

        merge_rules = {
            ("brca", "gene"): "brca1",
        }

        node._update_pair_entity_references(merge_rules, ctx)

        assessment = state.pair_assessments_by_resource[resource_id][0]
        assert assessment.entity1.name == "BRCA1"
        assert "BRCA" in assessment.entity1.aliases
        assert assessment.entity2.name == "TP53"  # Unchanged
        assert "MERGED_FROM" not in assessment.entity2.reasoning

    def test_handles_multiple_assessments(self, mock_deps):
        """Should update all assessments in a resource."""
        node = MergeEntitiesNode()
        resource_id = ResourceId(url="https://example.com", counter=1)

        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )

        state.validated_entities_by_resource[resource_id] = {
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

        state.pair_assessments_by_resource[resource_id] = [
            PairAssessment(
                resource_id=resource_id,
                entity1=EntityMention(
                    kind="gene",
                    name="BRCA",
                    aliases=[],
                    quotes=[],
                    reasoning="Original",
                ),
                entity2=EntityMention(
                    kind="disease",
                    name="Cancer",
                    aliases=[],
                    quotes=[],
                    reasoning="Original",
                ),
                relationship="associated_with",
                quotes=[],
                confidence="high",
                reasoning="Test 1",
            ),
            PairAssessment(
                resource_id=resource_id,
                entity1=EntityMention(
                    kind="gene",
                    name="BRCA",
                    aliases=[],
                    quotes=[],
                    reasoning="Original",
                ),
                entity2=EntityMention(
                    kind="gene",
                    name="TP53",
                    aliases=[],
                    quotes=[],
                    reasoning="Original",
                ),
                relationship="interacts_with",
                quotes=[],
                confidence="medium",
                reasoning="Test 2",
            ),
        ]

        ctx = GraphRunContext(state=state, deps=mock_deps)

        merge_rules = {
            ("brca", "gene"): "brca1",
        }

        node._update_pair_entity_references(merge_rules, ctx)

        # Both assessments should have entity1 updated
        for assessment in state.pair_assessments_by_resource[resource_id]:
            assert assessment.entity1.name == "BRCA1"
            assert "BRCA" in assessment.entity1.aliases
            assert "MERGED_FROM(BRCA)" in assessment.entity1.reasoning

    def test_handles_multiple_resources(self, mock_deps):
        """Should update assessments across multiple resources."""
        node = MergeEntitiesNode()
        resource_id1 = ResourceId(url="https://example.com/doc1", counter=1)
        resource_id2 = ResourceId(url="https://example.com/doc2", counter=2)

        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )

        # Both resources have merged entity
        for resource_id in [resource_id1, resource_id2]:
            state.validated_entities_by_resource[resource_id] = {
                "BRCA1": EntityMention(
                    kind="gene",
                    name="BRCA1",
                    aliases=[],
                    quotes=[],
                    reasoning="Merged",
                )
            }

            state.pair_assessments_by_resource[resource_id] = [
                PairAssessment(
                    resource_id=resource_id,
                    entity1=EntityMention(
                        kind="gene",
                        name="BRCA",
                        aliases=[],
                        quotes=[],
                        reasoning="Original",
                    ),
                    entity2=EntityMention(
                        kind="gene",
                        name="BRCA",
                        aliases=[],
                        quotes=[],
                        reasoning="Original",
                    ),
                    relationship="self_reference",
                    quotes=[],
                    confidence="low",
                    reasoning="Test",
                )
            ]

        ctx = GraphRunContext(state=state, deps=mock_deps)

        merge_rules = {
            ("brca", "gene"): "brca1",
        }

        node._update_pair_entity_references(merge_rules, ctx)

        # Both resources should be updated
        for resource_id in [resource_id1, resource_id2]:
            assessment = state.pair_assessments_by_resource[resource_id][0]
            assert assessment.entity1.name == "BRCA1"
            assert assessment.entity2.name == "BRCA1"
