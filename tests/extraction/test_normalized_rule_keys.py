"""Tests for normalized rule key behavior in entity consolidation.

Rules are keyed by normalized form (not canonical name) to ensure that:
1. All spelling/case variants of an entity match the same rule
2. Rules apply consistently across documents with different capitalizations
3. Parenthetical abbreviations (e.g., "Name (Abbrev)") match rules for base forms
"""

import pytest

from interaction_finder.extraction.models import EntityMention, EntityRef
from interaction_finder.extraction.stages.consolidate_entities import (
    _apply_merge_rules_globally,
    _resolve_transitive_merges,
)
from interaction_finder.extraction.state import State
from interaction_finder.extraction.utils import normalize_for_comparison
from interaction_finder.resources import ResourceId


def build_permitted_pairs(kinds: list[str]) -> dict[str, set[str]]:
    """Build permitted pairs dict for testing."""
    return {kind: {kind} for kind in kinds}


class TestNormalizedRuleKeys:
    """Test that rules are keyed by normalized form for cross-document consistency."""

    def test_single_rule_applies_to_case_variants(self):
        """A single normalized rule should apply to all case variants.

        Rule: ("brca1", "gene") → "BRCA"
        Should merge entities: "BRCA1", "brca1", "Brca1" → "BRCA"
        """
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        resource1 = ResourceId(url="https://doc1.com", counter=0)
        resource2 = ResourceId(url="https://doc2.com", counter=1)
        resource3 = ResourceId(url="https://doc3.com", counter=2)
        # Three docs with different capitalizations of child and parent
        state.validated_entities_by_resource = {
            resource1: {
                "BRCA": EntityRef(
                    canonical="BRCA",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="BRCA",
                            aliases=[],
                            quotes=[],
                            reasoning="doc1",
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
                            reasoning="doc1",
                        )
                    ],
                ),
            },
            resource2: {
                "brca": EntityRef(
                    canonical="brca",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="brca",
                            aliases=[],
                            quotes=[],
                            reasoning="doc2",
                        )
                    ],
                ),
                "brca1": EntityRef(
                    canonical="brca1",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="brca1",
                            aliases=[],
                            quotes=[],
                            reasoning="doc2",
                        )
                    ],
                ),
            },
            resource3: {
                "Brca": EntityRef(
                    canonical="Brca",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="Brca",
                            aliases=[],
                            quotes=[],
                            reasoning="doc3",
                        )
                    ],
                ),
                "Brca1": EntityRef(
                    canonical="Brca1",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="Brca1",
                            aliases=[],
                            quotes=[],
                            reasoning="doc3",
                        )
                    ],
                ),
            },
        }
        # Rules using normalized keys: (normalized_name, kind) -> (target, reasoning)
        # The target is the canonical name to merge into
        merge_rules = {
            ("brca1", "gene"): ("BRCA", "auto:substring"),
            ("brca", "gene"): ("BRCA", "auto:alias"),
        }
        _apply_merge_rules_globally(merge_rules, state)
        # All three documents should merge both entities into "BRCA"
        # But wait - doc1 already has "BRCA", so we need to check the target handling
        # The rules say merge both lowercase variants into BRCA
        for resource_id in [resource1, resource2, resource3]:
            entities = state.validated_entities_by_resource[resource_id]
            # Should have one entity after both merges
            assert len(entities) == 1, (
                f"Expected 1 entity in {resource_id}, got {len(entities)}: {list(entities.keys())}"
            )
            # Parent should be "BRCA"
            assert "BRCA" in entities

    def test_normalized_rule_matches_greek_variants(self):
        """Normalized rules should match Greek letter variants.

        "TGF-β" and "TGF-beta" both normalize to "tgf beta"
        """
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        resource1 = ResourceId(url="https://doc1.com", counter=0)
        resource2 = ResourceId(url="https://doc2.com", counter=1)
        state.validated_entities_by_resource = {
            resource1: {
                "TGF": EntityRef(
                    canonical="TGF",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="TGF",
                            aliases=[],
                            quotes=[],
                            reasoning="parent",
                        )
                    ],
                ),
                "TGF-β": EntityRef(
                    canonical="TGF-β",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="TGF-β",
                            aliases=[],
                            quotes=[],
                            reasoning="child",
                        )
                    ],
                ),
            },
            resource2: {
                "TGF": EntityRef(
                    canonical="TGF",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="TGF",
                            aliases=[],
                            quotes=[],
                            reasoning="parent",
                        )
                    ],
                ),
                "TGF-beta": EntityRef(
                    canonical="TGF-beta",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="TGF-beta",
                            aliases=[],
                            quotes=[],
                            reasoning="child",
                        )
                    ],
                ),
            },
        }
        # Rule for normalized "tgf beta" matches both TGF-β and TGF-beta
        # First verify that both normalize to the same key
        assert normalize_for_comparison("TGF-β") == normalize_for_comparison("TGF-beta")
        merge_rules = {("tgf beta", "gene"): ("TGF", "test")}
        _apply_merge_rules_globally(merge_rules, state)
        # Both docs should have merged the beta variant
        for resource_id in [resource1, resource2]:
            entities = state.validated_entities_by_resource[resource_id]
            assert len(entities) == 1
            assert "TGF" in entities

    def test_rule_applies_to_all_case_variants(self):
        """A rule keyed by lowercase should match any case variant of that entity."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        resource = ResourceId(url="https://doc1.com", counter=0)
        # Entity with uppercase canonical name
        state.validated_entities_by_resource = {
            resource: {
                "Parent": EntityRef(
                    canonical="Parent",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="Parent",
                            aliases=[],
                            quotes=[],
                            reasoning="test",
                        )
                    ],
                ),
                "CHILD": EntityRef(
                    canonical="CHILD",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="CHILD",
                            aliases=[],
                            quotes=[],
                            reasoning="test",
                        )
                    ],
                ),
            }
        }
        # Rule uses lowercase normalized key
        merge_rules = {("child", "gene"): ("Parent", "test")}
        _apply_merge_rules_globally(merge_rules, state)
        entities = state.validated_entities_by_resource[resource]
        assert len(entities) == 1
        assert "Parent" in entities
        # "CHILD" should be absorbed
        assert "CHILD" not in entities


class TestParentheticalMatching:
    """Test that parenthetical forms are handled correctly."""

    def test_parenthetical_entity_merged_via_normalized_key(self):
        """Entity with parenthetical form should merge via normalized key lookup.

        extract_all_forms() expands "PAH (Pulmonary arterial hypertension)"
        to include both the full form and the expanded inner content.
        """
        state = State(
            topic="test",
            target_entity_types=["disease"],
            permitted_pairs=build_permitted_pairs(["disease"]),
        )
        resource = ResourceId(url="https://doc1.com", counter=0)
        state.validated_entities_by_resource = {
            resource: {
                "PAH": EntityRef(
                    canonical="PAH",
                    mentions=[
                        EntityMention(
                            kind="disease",
                            name="PAH",
                            aliases=[],
                            quotes=[],
                            reasoning="abbreviation",
                        )
                    ],
                ),
                "Pulmonary arterial hypertension": EntityRef(
                    canonical="Pulmonary arterial hypertension",
                    mentions=[
                        EntityMention(
                            kind="disease",
                            name="Pulmonary arterial hypertension",
                            aliases=[],
                            quotes=[],
                            reasoning="full name",
                        )
                    ],
                ),
            }
        }
        # Rule uses normalized form of full name
        full_name = "Pulmonary arterial hypertension"
        normalized = normalize_for_comparison(full_name)
        merge_rules = {(normalized, "disease"): ("PAH", "auto")}
        _apply_merge_rules_globally(merge_rules, state)
        entities = state.validated_entities_by_resource[resource]
        assert len(entities) == 1
        assert "PAH" in entities


class TestTransitiveChainNormalization:
    """Test that transitive chains are resolved correctly with normalized keys."""

    def test_chain_follows_normalized_lookups(self):
        """Transitive chain A→B→C should resolve all to C."""
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
            ("gene a", "gene"): ("Gene B", "auto:1"),
            ("gene b", "gene"): ("Gene C", "llm:merge"),
        }
        rules = _resolve_transitive_merges(unresolved_rules)
        _apply_merge_rules_globally(rules, state)
        # All three documents should have "Gene C"
        assert "Gene C" in state.validated_entities_by_resource[resource1]
        assert "Gene C" in state.validated_entities_by_resource[resource2]
        assert "Gene C" in state.validated_entities_by_resource[resource3]

    def test_case_mismatched_targets_resolved(self):
        """Transitive chains should work even with case differences in targets."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        resource = ResourceId(url="https://doc1.com", counter=0)
        state.validated_entities_by_resource = {
            resource: {
                "brca2": EntityRef(
                    canonical="brca2",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="brca2",
                            aliases=[],
                            quotes=[],
                            reasoning="test",
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
                            reasoning="test",
                        )
                    ],
                ),
                "BRCA": EntityRef(
                    canonical="BRCA",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="BRCA",
                            aliases=[],
                            quotes=[],
                            reasoning="test",
                        )
                    ],
                ),
            }
        }
        # Chain: brca2 → BRCA1 → BRCA (different cases)
        unresolved_rules = {
            ("brca2", "gene"): ("BRCA1", "auto"),
            ("brca1", "gene"): ("BRCA", "auto"),
        }
        rules = _resolve_transitive_merges(unresolved_rules)
        _apply_merge_rules_globally(rules, state)
        entities = state.validated_entities_by_resource[resource]
        # All should merge into BRCA
        assert len(entities) == 1
        assert "BRCA" in entities

    def test_long_chain_resolution(self):
        """Long chains (4+ levels) should resolve correctly."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        resource = ResourceId(url="https://doc1.com", counter=0)
        # Use multi-character names that extract_all_forms recognizes
        state.validated_entities_by_resource = {
            resource: {
                "GeneA": EntityRef(
                    canonical="GeneA",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="GeneA",
                            aliases=[],
                            quotes=[],
                            reasoning="test",
                        )
                    ],
                ),
                "GeneB": EntityRef(
                    canonical="GeneB",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="GeneB",
                            aliases=[],
                            quotes=[],
                            reasoning="test",
                        )
                    ],
                ),
                "GeneC": EntityRef(
                    canonical="GeneC",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="GeneC",
                            aliases=[],
                            quotes=[],
                            reasoning="test",
                        )
                    ],
                ),
                "GeneD": EntityRef(
                    canonical="GeneD",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="GeneD",
                            aliases=[],
                            quotes=[],
                            reasoning="test",
                        )
                    ],
                ),
                "GeneE": EntityRef(
                    canonical="GeneE",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="GeneE",
                            aliases=[],
                            quotes=[],
                            reasoning="test",
                        )
                    ],
                ),
            }
        }
        # Chain: GeneA → GeneB → GeneC → GeneD → GeneE
        unresolved_rules = {
            ("genea", "gene"): ("GeneB", "auto"),
            ("geneb", "gene"): ("GeneC", "auto"),
            ("genec", "gene"): ("GeneD", "auto"),
            ("gened", "gene"): ("GeneE", "auto"),
        }
        rules = _resolve_transitive_merges(unresolved_rules)
        _apply_merge_rules_globally(rules, state)
        entities = state.validated_entities_by_resource[resource]
        assert len(entities) == 1
        assert "GeneE" in entities
