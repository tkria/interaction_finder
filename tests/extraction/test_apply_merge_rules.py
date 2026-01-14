"""Test applying merge rules globally across documents."""

import pytest

from interaction_finder.extraction.stages.consolidate_entities import (
    _apply_merge_rules_globally,
    _resolve_transitive_merges,
)
from interaction_finder.extraction.models import EntityMention, EntityRef
from interaction_finder.extraction.state import State
from interaction_finder.resources import ResourceId


def build_permitted_pairs(kinds: list[str]) -> dict[str, set[str]]:
    """Build permitted pairs dict for testing."""
    return {kind: {kind} for kind in kinds}


class TestApplyMergeRulesGlobally:
    """Test that merge rules are correctly applied to validated entities."""

    def test_single_document_single_merge(self):
        """Single entity merged to new canonical in one document."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        resource = ResourceId(url="https://doc1.com", counter=0)
        state.validated_entities_by_resource = {
            resource: {
                "brca1": EntityRef(
                    canonical="brca1",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="brca1",
                            aliases=[],
                            quotes=[],
                            reasoning="test",
                        )
                    ],
                )
            }
        }
        rules = {("brca1", "gene"): ("BRCA1", "substring", "merged")}
        _apply_merge_rules_globally(rules, state)
        assert "BRCA1" in state.validated_entities_by_resource[resource]
        assert "brca1" not in state.validated_entities_by_resource[resource]

    def test_transitive_chain_applied(self):
        """Transitive chain A→B→C results in all pointing to C."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        resource1 = ResourceId(url="https://doc1.com", counter=0)
        resource2 = ResourceId(url="https://doc2.com", counter=1)
        resource3 = ResourceId(url="https://doc3.com", counter=2)
        state.validated_entities_by_resource = {
            resource1: {
                "Gene A": EntityRef(
                    canonical="Gene A",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="Gene A",
                            aliases=[],
                            quotes=[],
                            reasoning="doc1",
                        )
                    ],
                )
            },
            resource2: {
                "Gene B": EntityRef(
                    canonical="Gene B",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="Gene B",
                            aliases=[],
                            quotes=[],
                            reasoning="doc2",
                        )
                    ],
                )
            },
            resource3: {
                "Gene C": EntityRef(
                    canonical="Gene C",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="Gene C",
                            aliases=[],
                            quotes=[],
                            reasoning="doc3",
                        )
                    ],
                )
            },
        }
        # Chain: A → B → C (unresolved)
        unresolved_rules = {
            ("gene a", "gene"): ("Gene B", "1:original:exact", None),
            ("gene b", "gene"): ("Gene C", "substring", "merged"),
        }
        rules = _resolve_transitive_merges(unresolved_rules)
        _apply_merge_rules_globally(rules, state)
        # All three documents should have "Gene C"
        assert "Gene C" in state.validated_entities_by_resource[resource1]
        assert "Gene C" in state.validated_entities_by_resource[resource2]
        assert "Gene C" in state.validated_entities_by_resource[resource3]

    def test_mentions_aggregated_on_merge(self):
        """When entities merge, their mentions should be combined."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        resource = ResourceId(url="https://doc1.com", counter=0)
        state.validated_entities_by_resource = {
            resource: {
                "brca1": EntityRef(
                    canonical="brca1",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="brca1",
                            aliases=[],
                            quotes=[],
                            reasoning="mention1",
                        )
                    ],
                ),
                "BRCA1": EntityRef(
                    canonical="BRCA1",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="BRCA1",
                            aliases=[],
                            quotes=[],
                            reasoning="mention2",
                        )
                    ],
                ),
            }
        }
        rules = {("brca1", "gene"): ("BRCA1", "substring", "merged")}
        _apply_merge_rules_globally(rules, state)
        # Should have one entity with two mentions
        assert len(state.validated_entities_by_resource[resource]) == 1
        assert "BRCA1" in state.validated_entities_by_resource[resource]
        merged = state.validated_entities_by_resource[resource]["BRCA1"]
        assert len(merged.mentions) == 2

    def test_empty_rules_no_change(self):
        """Empty rules should not modify state."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        resource = ResourceId(url="https://doc1.com", counter=0)
        state.validated_entities_by_resource = {
            resource: {
                "BRCA1": EntityRef(
                    canonical="BRCA1",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="BRCA1",
                            aliases=[],
                            quotes=[],
                            reasoning="test",
                        )
                    ],
                )
            }
        }
        _apply_merge_rules_globally({}, state)
        assert "BRCA1" in state.validated_entities_by_resource[resource]

    def test_merge_count_tracked(self):
        """entities_merged counter should be updated correctly."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        resource = ResourceId(url="https://doc1.com", counter=0)
        state.validated_entities_by_resource = {
            resource: {
                "brca1": EntityRef(
                    canonical="brca1",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="brca1",
                            aliases=[],
                            quotes=[],
                            reasoning="m1",
                        )
                    ],
                ),
                "BRCA1": EntityRef(
                    canonical="BRCA1",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="BRCA1",
                            aliases=[],
                            quotes=[],
                            reasoning="m2",
                        )
                    ],
                ),
            }
        }
        assert state.entities_merged == 0
        rules = {("brca1", "gene"): ("BRCA1", "substring", "merged")}
        _apply_merge_rules_globally(rules, state)
        # Went from 2 entities to 1, so 1 merge
        assert state.entities_merged == 1

    def test_different_kinds_independent(self):
        """Merge rules for different kinds should not interfere."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        resource = ResourceId(url="https://doc1.com", counter=0)
        state.validated_entities_by_resource = {
            resource: {
                "brca1": EntityRef(
                    canonical="brca1",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="brca1",
                            aliases=[],
                            quotes=[],
                            reasoning="gene",
                        )
                    ],
                ),
                "cancer": EntityRef(
                    canonical="cancer",
                    mentions=[
                        EntityMention(
                            kind="disease",
                            name="cancer",
                            aliases=[],
                            quotes=[],
                            reasoning="disease",
                        )
                    ],
                ),
            }
        }
        # Only merge the gene
        rules = {("brca1", "gene"): ("BRCA1", "substring", "merged")}
        _apply_merge_rules_globally(rules, state)
        assert "BRCA1" in state.validated_entities_by_resource[resource]
        assert "cancer" in state.validated_entities_by_resource[resource]
