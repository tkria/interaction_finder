"""Tests for normalized rule key behavior in entity consolidation.

Rules are keyed by normalized form (not canonical name) to ensure that:
1. All spelling/case variants of an entity match the same rule
2. Rules apply consistently across documents with different capitalizations
3. Parenthetical abbreviations (e.g., "Name (Abbrev)") match rules for base forms
"""

import pytest
from unittest.mock import MagicMock

from interaction_finder.extraction.nodes import ConsolidateEntitiesNode
from interaction_finder.extraction.models import EntityMention
from interaction_finder.resources import ResourceId
from interaction_finder.extraction.state import State
from interaction_finder.extraction.utils import normalize_for_comparison
from pydantic_graph import GraphRunContext


@pytest.fixture
def mock_deps():
    """Create mock dependencies."""
    mock = MagicMock()
    mock.logger = MagicMock()
    mock.config.tools.extraction.merge_batch_size = 10
    return mock


def build_permitted_pairs(kinds: list[str]) -> dict[str, set[str]]:
    """Build permitted pairs dict for testing."""
    return {kind: {kind} for kind in kinds}


class TestNormalizedRuleKeys:
    """Test that rules are keyed by normalized form for cross-document consistency."""

    def test_single_rule_applies_to_case_variants(self, mock_deps):
        """A single normalized rule should apply to all case variants.

        Rule: ("brca1", "gene") → "BRCA"
        Should match entities: "BRCA1", "brca1", "Brca1"
        """
        node = ConsolidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        resource1 = ResourceId(url="https://doc1.com", counter=0)
        resource2 = ResourceId(url="https://doc2.com", counter=1)
        resource3 = ResourceId(url="https://doc3.com", counter=2)

        # Three docs with different capitalizations
        ctx.state.validated_entities_by_resource = {
            resource1: {
                "BRCA": EntityMention(
                    kind="gene", name="BRCA", aliases=[], quotes=[], reasoning="doc1"
                ),
                "BRCA1": EntityMention(
                    kind="gene", name="BRCA1", aliases=[], quotes=[], reasoning="doc1"
                ),
            },
            resource2: {
                "brca": EntityMention(
                    kind="gene", name="brca", aliases=[], quotes=[], reasoning="doc2"
                ),
                "brca1": EntityMention(
                    kind="gene", name="brca1", aliases=[], quotes=[], reasoning="doc2"
                ),
            },
            resource3: {
                "Brca": EntityMention(
                    kind="gene", name="Brca", aliases=[], quotes=[], reasoning="doc3"
                ),
                "Brca1": EntityMention(
                    kind="gene", name="Brca1", aliases=[], quotes=[], reasoning="doc3"
                ),
            },
        }

        # Single rule using normalized key, canonical target
        merge_rules = {("brca1", "gene"): "BRCA"}

        node._apply_merge_rules_globally(merge_rules, ctx)

        # All three documents should have merged BRCA1 variant into parent
        for resource_id in [resource1, resource2, resource3]:
            entities = ctx.state.validated_entities_by_resource[resource_id]
            # Should only have one entity (parent)
            assert len(entities) == 1
            # Parent should contain child in aliases
            parent_name = list(entities.keys())[0]
            parent = entities[parent_name]
            assert any(
                normalize_for_comparison(a) == "brca1" for a in parent.aliases
            ), f"BRCA1 variant not in aliases for {resource_id}"

    def test_normalized_rule_matches_greek_variants(self, mock_deps):
        """Normalized rules should match Greek letter variants.

        "TGF-β" and "TGF-beta" both normalize to "tgf beta"
        """
        node = ConsolidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        resource1 = ResourceId(url="https://doc1.com", counter=0)
        resource2 = ResourceId(url="https://doc2.com", counter=1)

        ctx.state.validated_entities_by_resource = {
            resource1: {
                "TGF": EntityMention(
                    kind="gene", name="TGF", aliases=[], quotes=[], reasoning="parent"
                ),
                "TGF-β": EntityMention(
                    kind="gene", name="TGF-β", aliases=[], quotes=[], reasoning="child"
                ),
            },
            resource2: {
                "TGF": EntityMention(
                    kind="gene", name="TGF", aliases=[], quotes=[], reasoning="parent"
                ),
                "TGF-beta": EntityMention(
                    kind="gene",
                    name="TGF-beta",
                    aliases=[],
                    quotes=[],
                    reasoning="child",
                ),
            },
        }

        # Rule for normalized "tgf beta" matches both TGF-β and TGF-beta
        merge_rules = {("tgf beta", "gene"): "TGF"}

        node._apply_merge_rules_globally(merge_rules, ctx)

        # Both docs should have merged the beta variant
        for resource_id in [resource1, resource2]:
            entities = ctx.state.validated_entities_by_resource[resource_id]
            assert "TGF" in entities
            assert len(entities) == 1
            assert ctx.state.entities_merged == 2


class TestParentheticalMatching:
    """Test that rules match entities with parenthetical abbreviations."""

    def test_rule_matches_parenthetical_entity(self, mock_deps):
        """Rule for base form should match "Name (Abbrev)" entity.

        Rule: ("pulmonary arterial hypertension", ...) → target
        Should match: "Pulmonary arterial hypertension (PAH)"
        """
        node = ConsolidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["disease"],
            permitted_pairs=build_permitted_pairs(["disease"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        resource1 = ResourceId(url="https://doc1.com", counter=0)

        ctx.state.validated_entities_by_resource = {
            resource1: {
                "PAH": EntityMention(
                    kind="disease",
                    name="PAH",
                    aliases=[],
                    quotes=[],
                    reasoning="parent",
                ),
                "Pulmonary arterial hypertension (PAH)": EntityMention(
                    kind="disease",
                    name="Pulmonary arterial hypertension (PAH)",
                    aliases=[],
                    quotes=[],
                    reasoning="child",
                ),
            },
        }

        # Rule uses base form (without parenthetical)
        merge_rules = {("pulmonary arterial hypertension", "disease"): "PAH"}

        node._apply_merge_rules_globally(merge_rules, ctx)

        entities = ctx.state.validated_entities_by_resource[resource1]
        assert "PAH" in entities
        assert len(entities) == 1
        assert "Pulmonary arterial hypertension (PAH)" in entities["PAH"].aliases

    def test_parenthetical_entity_finds_parenthetical_target(self, mock_deps):
        """Parenthetical entity should find parenthetical target by expanded form.

        Entity: "Idiopathic PAH (IPAH)"
        Target: "Pulmonary arterial hypertension (PAH)"
        Match via: "pulmonary arterial hypertension" expanded from target
        """
        node = ConsolidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["disease"],
            permitted_pairs=build_permitted_pairs(["disease"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        resource1 = ResourceId(url="https://doc1.com", counter=0)

        ctx.state.validated_entities_by_resource = {
            resource1: {
                "Pulmonary arterial hypertension (PAH)": EntityMention(
                    kind="disease",
                    name="Pulmonary arterial hypertension (PAH)",
                    aliases=[],
                    quotes=[],
                    reasoning="parent",
                ),
                "Idiopathic pulmonary arterial hypertension (IPAH)": EntityMention(
                    kind="disease",
                    name="Idiopathic pulmonary arterial hypertension (IPAH)",
                    aliases=[],
                    quotes=[],
                    reasoning="child",
                ),
            },
        }

        # Rule uses base forms - target matches via parenthetical expansion
        merge_rules = {
            (
                "idiopathic pulmonary arterial hypertension",
                "disease",
            ): "Pulmonary arterial hypertension"
        }

        node._apply_merge_rules_globally(merge_rules, ctx)

        entities = ctx.state.validated_entities_by_resource[resource1]
        # Should have merged - child entity absorbed into parent
        assert len(entities) == 1
        remaining = list(entities.values())[0]
        assert "Idiopathic pulmonary arterial hypertension (IPAH)" in remaining.aliases


class TestTransitiveChainNormalization:
    """Test that transitive chain resolution works with mixed canonical/normalized."""

    def test_chain_follows_normalized_lookups(self, mock_deps):
        """Transitive chains should follow normalized lookups.

        Rules:
        - ("brca2", "gene") → "BRCA1"
        - ("brca1", "gene") → "BRCA"

        Result: BRCA2 → BRCA1 → BRCA, so BRCA2 → BRCA directly
        """
        node = ConsolidateEntitiesNode()

        # Input rules with canonical targets but normalized keys
        merge_rules = {
            ("brca2", "gene"): "BRCA1",
            ("brca1", "gene"): "BRCA",
        }

        resolved = node._resolve_transitive_merges(merge_rules)

        # Both should point to final parent BRCA
        assert resolved[("brca2", "gene")] == "BRCA"
        assert resolved[("brca1", "gene")] == "BRCA"

    def test_chain_handles_case_mismatch_in_targets(self, mock_deps):
        """Chain should handle when target canonical doesn't match key case.

        Rules:
        - ("gene x", "gene") → "Gene X"  (target is title case)
        - ("gene x variant", "gene") → "gene x"  (target is lowercase)

        Chain: "gene x variant" → "gene x" → "Gene X"
        Resolved: "gene x variant" → "Gene X"
        """
        node = ConsolidateEntitiesNode()

        merge_rules = {
            ("gene x", "gene"): "Gene X",
            ("gene x variant", "gene"): "gene x",
        }

        resolved = node._resolve_transitive_merges(merge_rules)

        # Chain should resolve to final canonical
        assert resolved[("gene x variant", "gene")] == "Gene X"
        assert resolved[("gene x", "gene")] == "Gene X"
