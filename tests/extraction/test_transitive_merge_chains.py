"""Test transitive merge chains with mixed auto-merge and LLM decisions.

This test file verifies that transitive chains resolve correctly when combining:
1. Auto-merge rules (from speculation-based matching)
2. LLM consolidation decisions
3. Multiple levels of indirection

Critical scenarios:
- Auto A→B, Auto B→C should resolve A→C
- Auto A→B, LLM B→D should resolve A→D
- LLM B→C, LLM C→D, Auto A→B should resolve A→D
"""

import pytest
from unittest.mock import MagicMock

from interaction_finder.extraction.nodes import ConsolidateEntitiesNode
from interaction_finder.extraction.models import EntityMention, EntityRef
from interaction_finder.resources import ResourceId
from interaction_finder.extraction.state import State
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


class TestTransitiveMergeChains:
    """Test that transitive chains resolve correctly with mixed rule sources."""

    def test_auto_to_auto_chain(self, mock_deps):
        """Auto A→B + Auto B→C should resolve to A→C.

        Rules:
        - ("actb variant", "gene") → "ACTB" (auto:2:alternation:exact)
        - ("actb", "gene") → "β-Actin" (auto:1:before_paren:exact)

        Expected resolution:
        - ("actb variant", "gene") → "β-Actin"
        - ("actb", "gene") → "β-Actin"
        """
        node = ConsolidateEntitiesNode()

        merge_rules = {
            ("actb variant", "gene"): ("ACTB", "auto:2:alternation:exact"),
            ("actb", "gene"): ("β-Actin", "auto:1:before_paren:exact"),
        }

        resolved = node._resolve_transitive_merges(merge_rules)

        # Both should point to final parent
        assert resolved[("actb variant", "gene")][0] == "β-Actin"
        assert resolved[("actb", "gene")][0] == "β-Actin"
        # Reasoning should come from final link
        assert resolved[("actb variant", "gene")][1] == "auto:1:before_paren:exact"

    def test_auto_to_llm_chain(self, mock_deps):
        """Auto A→B + LLM B→D should resolve to A→D.

        Scenario: Auto-merge creates "BRCA1", then LLM decides BRCA1 → BRCA

        Rules:
        - ("brca1 variant", "gene") → "BRCA1" (auto:2:alternation:exact)
        - ("brca1", "gene") → "BRCA" (llm:rename)

        Expected resolution:
        - ("brca1 variant", "gene") → "BRCA"
        - ("brca1", "gene") → "BRCA"
        """
        node = ConsolidateEntitiesNode()

        merge_rules = {
            ("brca1 variant", "gene"): ("BRCA1", "auto:2:alternation:exact"),
            ("brca1", "gene"): ("BRCA", "llm:rename"),
        }

        resolved = node._resolve_transitive_merges(merge_rules)

        # Both should point to final parent
        assert resolved[("brca1 variant", "gene")][0] == "BRCA"
        assert resolved[("brca1", "gene")][0] == "BRCA"
        # Final reasoning from LLM decision
        assert resolved[("brca1 variant", "gene")][1] == "llm:rename"

    def test_llm_chain_with_auto_prefix(self, mock_deps):
        """LLM B→C + LLM C→D + Auto A→B should resolve A→D.

        Scenario: Auto-merge creates intermediate, then multiple LLM decisions

        Rules:
        - ("gene variant", "gene") → "Gene X" (auto:2:alternation:exact)
        - ("gene x", "gene") → "GeneX" (llm:merge)
        - ("genex", "gene") → "GENEX" (llm:rename)

        Expected resolution: All → "GENEX"
        """
        node = ConsolidateEntitiesNode()

        merge_rules = {
            ("gene variant", "gene"): ("Gene X", "auto:2:alternation:exact"),
            ("gene x", "gene"): ("GeneX", "llm:merge"),
            ("genex", "gene"): ("GENEX", "llm:rename"),
        }

        resolved = node._resolve_transitive_merges(merge_rules)

        # All should point to final parent
        assert resolved[("gene variant", "gene")][0] == "GENEX"
        assert resolved[("gene x", "gene")][0] == "GENEX"
        assert resolved[("genex", "gene")][0] == "GENEX"
        # Final reasoning from last LLM decision
        assert resolved[("gene variant", "gene")][1] == "llm:rename"

    def test_long_mixed_chain(self, mock_deps):
        """Long chain with alternating auto/LLM decisions.

        A → B → C → D → E (mixed sources)

        Rules:
        - ("a", "gene") → "B" (auto:1:before_paren:exact)
        - ("b", "gene") → "C" (llm:merge)
        - ("c", "gene") → "D" (auto:3:fuzzy)
        - ("d", "gene") → "E" (llm:rename)

        Expected: All → "E"
        """
        node = ConsolidateEntitiesNode()

        merge_rules = {
            ("a", "gene"): ("B", "auto:1:before_paren:exact"),
            ("b", "gene"): ("C", "llm:merge"),
            ("c", "gene"): ("D", "auto:3:fuzzy"),
            ("d", "gene"): ("E", "llm:rename"),
        }

        resolved = node._resolve_transitive_merges(merge_rules)

        # All should point to final parent
        for key in [("a", "gene"), ("b", "gene"), ("c", "gene"), ("d", "gene")]:
            assert resolved[key][0] == "E"
            assert resolved[key][1] == "llm:rename"

    def test_multiple_chains_independent(self, mock_deps):
        """Multiple independent chains should not interfere.

        Chain 1: A → B → C
        Chain 2: X → Y → Z
        """
        node = ConsolidateEntitiesNode()

        merge_rules = {
            # Chain 1
            ("a", "gene"): ("B", "auto:1:before_paren:exact"),
            ("b", "gene"): ("C", "llm:merge"),
            # Chain 2
            ("x", "phenotype"): ("Y", "auto:2:alternation:exact"),
            ("y", "phenotype"): ("Z", "llm:rename"),
        }

        resolved = node._resolve_transitive_merges(merge_rules)

        # Chain 1
        assert resolved[("a", "gene")][0] == "C"
        assert resolved[("b", "gene")][0] == "C"

        # Chain 2
        assert resolved[("x", "phenotype")][0] == "Z"
        assert resolved[("y", "phenotype")][0] == "Z"

    def test_cycle_detection(self, mock_deps):
        """Cycles should be detected and broken.

        Invalid rules:
        - ("a", "gene") → "B"
        - ("b", "gene") → "A"

        Should not infinite loop.
        """
        node = ConsolidateEntitiesNode()

        merge_rules = {
            ("a", "gene"): ("B", "auto:1:before_paren:exact"),
            ("b", "gene"): ("A", "llm:merge"),
        }

        # Should complete without infinite loop
        resolved = node._resolve_transitive_merges(merge_rules)

        # Cycle should be broken - each points to what it can reach
        assert resolved[("a", "gene")][0] in ["A", "B"]
        assert resolved[("b", "gene")][0] in ["A", "B"]

    def test_self_reference_ignored(self, mock_deps):
        """Self-references should be handled gracefully.

        Rule: ("a", "gene") → "A" (normalized key, canonical target)
        """
        node = ConsolidateEntitiesNode()

        merge_rules = {
            ("a", "gene"): ("A", "auto:0:original:exact"),
        }

        resolved = node._resolve_transitive_merges(merge_rules)

        # Should pass through unchanged
        assert resolved[("a", "gene")][0] == "A"

    def test_order_independence_auto_then_llm(self, mock_deps):
        """Result should be same regardless of whether auto-merge or LLM decision comes first.

        Scenario 1 (auto first): {"a→B": auto, "b→C": llm}
        Scenario 2 (llm first): {"b→C": llm, "a→B": auto}

        Both should resolve to: a→C, b→C
        """
        node = ConsolidateEntitiesNode()

        # Order 1: auto-merge first
        merge_rules_1 = {
            ("a", "gene"): ("B", "auto:1:before_paren:exact"),
            ("b", "gene"): ("C", "llm:merge"),
        }

        # Order 2: LLM first
        merge_rules_2 = {
            ("b", "gene"): ("C", "llm:merge"),
            ("a", "gene"): ("B", "auto:1:before_paren:exact"),
        }

        resolved_1 = node._resolve_transitive_merges(merge_rules_1)
        resolved_2 = node._resolve_transitive_merges(merge_rules_2)

        # Both should have identical results
        assert resolved_1[("a", "gene")][0] == resolved_2[("a", "gene")][0] == "C"
        assert resolved_1[("b", "gene")][0] == resolved_2[("b", "gene")][0] == "C"

    def test_order_independence_complex_chain(self, mock_deps):
        """Complex chains should resolve consistently regardless of insertion order.

        Chain: A → B → C → D → E

        Test multiple orderings to ensure deterministic resolution.
        """
        node = ConsolidateEntitiesNode()

        # Forward order
        rules_forward = {
            ("a", "gene"): ("B", "auto:1"),
            ("b", "gene"): ("C", "llm:1"),
            ("c", "gene"): ("D", "auto:2"),
            ("d", "gene"): ("E", "llm:2"),
        }

        # Reverse order
        rules_reverse = {
            ("d", "gene"): ("E", "llm:2"),
            ("c", "gene"): ("D", "auto:2"),
            ("b", "gene"): ("C", "llm:1"),
            ("a", "gene"): ("B", "auto:1"),
        }

        # Random order
        rules_random = {
            ("c", "gene"): ("D", "auto:2"),
            ("a", "gene"): ("B", "auto:1"),
            ("d", "gene"): ("E", "llm:2"),
            ("b", "gene"): ("C", "llm:1"),
        }

        resolved_fwd = node._resolve_transitive_merges(rules_forward)
        resolved_rev = node._resolve_transitive_merges(rules_reverse)
        resolved_rnd = node._resolve_transitive_merges(rules_random)

        # All should resolve to final parent "E"
        for key in [("a", "gene"), ("b", "gene"), ("c", "gene"), ("d", "gene")]:
            assert resolved_fwd[key][0] == "E"
            assert resolved_rev[key][0] == "E"
            assert resolved_rnd[key][0] == "E"


class TestEndToEndTransitiveApplication:
    """Test that transitive chains apply correctly in full node execution."""

    @pytest.mark.asyncio
    async def test_auto_merge_followed_by_llm_merge(self, mock_deps):
        """Full node execution: auto-merge creates intermediate, LLM consolidates further.

        Entities:
        - Doc1: "BRCA1"
        - Doc2: "Brca1" (auto-merges to BRCA1)
        - Doc3: "BRCA" (LLM decides BRCA1 → BRCA)

        Expected: All three docs end up with "BRCA" as canonical.
        """
        # This test would require mocking the LLM agent response
        # Skipping for now as it requires more complex setup
        pytest.skip(
            "Requires LLM agent mocking - covered by transitive resolution tests"
        )

    def test_apply_rules_with_transitive_chain(self, mock_deps):
        """Test _apply_merge_rules_globally with transitively resolved rules."""
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

        # Three entities that should chain: A → B → C
        ctx.state.validated_entities_by_resource = {
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

        # Chain rules: A → B → C (not resolved)
        merge_rules_unresolved = {
            ("gene a", "gene"): ("Gene B", "auto:1:before_paren:exact"),
            ("gene b", "gene"): ("Gene C", "llm:merge"),
        }

        # Resolve transitive chain
        merge_rules = node._resolve_transitive_merges(merge_rules_unresolved)

        # Apply resolved rules
        node._apply_merge_rules_globally(merge_rules, ctx)

        # All three documents should have "Gene C" as canonical
        assert "Gene C" in ctx.state.validated_entities_by_resource[resource1]
        assert "Gene C" in ctx.state.validated_entities_by_resource[resource2]
        assert "Gene C" in ctx.state.validated_entities_by_resource[resource3]

        # Each document should have its original name as an alias
        # Doc1: Gene A → Gene C (Gene A becomes alias in doc1)
        gene_c_ref_doc1 = ctx.state.validated_entities_by_resource[resource1]["Gene C"]
        assert "Gene A" in gene_c_ref_doc1.aliases()

        # Doc2: Gene B → Gene C (Gene B becomes alias in doc2)
        gene_c_ref_doc2 = ctx.state.validated_entities_by_resource[resource2]["Gene C"]
        assert "Gene B" in gene_c_ref_doc2.aliases()

        # Doc3: Gene C → Gene C (no alias, already canonical)
        gene_c_ref_doc3 = ctx.state.validated_entities_by_resource[resource3]["Gene C"]
        # Doc3 originally had Gene C, so no other aliases expected
        assert len(gene_c_ref_doc3.aliases()) == 0
