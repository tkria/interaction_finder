"""Test transitive merge chain resolution.

Verifies that merge chains resolve correctly when combining:
1. Auto-merge rules (from speculation-based matching)
2. LLM consolidation decisions
3. Multiple levels of indirection
"""

import pytest

from interaction_finder.extraction.stages.consolidate_entities import (
    _resolve_transitive_merges,
)


class TestResolveTransitiveMerges:
    """Test that transitive chains resolve correctly."""

    def test_auto_to_auto_chain(self):
        """Auto A→B + Auto B→C should resolve to A→C."""
        merge_rules = {
            ("actb variant", "gene"): ("ACTB", "2:alternation:exact", None),
            ("actb", "gene"): ("β-Actin", "1:before_paren:exact", None),
        }
        resolved = _resolve_transitive_merges(merge_rules)
        assert resolved[("actb variant", "gene")][0] == "β-Actin"
        assert resolved[("actb", "gene")][0] == "β-Actin"
        # Trigger from final link
        assert resolved[("actb variant", "gene")][1] == "1:before_paren:exact"
        # Auto-merge has None reasoning
        assert resolved[("actb variant", "gene")][2] is None

    def test_auto_to_llm_chain(self):
        """Auto A→B + LLM B→D should resolve to A→D."""
        merge_rules = {
            ("brca1 variant", "gene"): ("BRCA1", "2:alternation:exact", None),
            ("brca1", "gene"): ("BRCA", "substring", "rename"),
        }
        resolved = _resolve_transitive_merges(merge_rules)
        assert resolved[("brca1 variant", "gene")][0] == "BRCA"
        assert resolved[("brca1", "gene")][0] == "BRCA"
        assert resolved[("brca1 variant", "gene")][1] == "substring"
        assert resolved[("brca1 variant", "gene")][2] == "rename"

    def test_llm_chain_with_auto_prefix(self):
        """LLM B→C + LLM C→D + Auto A→B should resolve A→D."""
        merge_rules = {
            ("gene variant", "gene"): ("Gene Y", "2:alternation:exact", None),
            ("gene y", "gene"): ("GeneY", "substring", "merge"),
            ("geney", "gene"): ("GENEY", "substring", "rename"),
        }
        resolved = _resolve_transitive_merges(merge_rules)
        assert resolved[("gene variant", "gene")][0] == "GENEY"
        assert resolved[("gene y", "gene")][0] == "GENEY"
        assert resolved[("geney", "gene")][0] == "GENEY"
        assert resolved[("gene variant", "gene")][1] == "substring"
        assert resolved[("gene variant", "gene")][2] == "rename"

    def test_long_mixed_chain(self):
        """Long chain with alternating auto/LLM decisions."""
        # Note: avoid C/D/I/L/M/V/X as they're Roman numerals
        merge_rules = {
            ("a", "gene"): ("B", "1:before_paren:exact", None),
            ("b", "gene"): ("E", "substring", "merge"),
            ("e", "gene"): ("F", "3:original:fuzzy", None),
            ("f", "gene"): ("G", "substring", "rename"),
        }
        resolved = _resolve_transitive_merges(merge_rules)
        for key in [("a", "gene"), ("b", "gene"), ("e", "gene"), ("f", "gene")]:
            assert resolved[key][0] == "G"
            assert resolved[key][1] == "substring"
            assert resolved[key][2] == "rename"

    def test_multiple_chains_independent(self):
        """Multiple independent chains should not interfere."""
        merge_rules = {
            ("a", "gene"): ("B", "1:before_paren:exact", None),
            ("b", "gene"): ("C", "substring", "merge"),
            ("x", "phenotype"): ("Y", "2:alternation:exact", None),
            ("y", "phenotype"): ("Z", "substring", "rename"),
        }
        resolved = _resolve_transitive_merges(merge_rules)
        assert resolved[("a", "gene")][0] == "C"
        assert resolved[("b", "gene")][0] == "C"
        assert resolved[("x", "phenotype")][0] == "Z"
        assert resolved[("y", "phenotype")][0] == "Z"

    def test_cycle_detection(self):
        """Cycles should be detected and broken without infinite loop."""
        merge_rules = {
            ("a", "gene"): ("B", "1:before_paren:exact", None),
            ("b", "gene"): ("A", "substring", "merge"),
        }
        # Should complete without infinite loop
        resolved = _resolve_transitive_merges(merge_rules)
        assert resolved[("a", "gene")][0] in ["A", "B"]
        assert resolved[("b", "gene")][0] in ["A", "B"]

    def test_self_reference_ignored(self):
        """Self-references should be handled gracefully."""
        merge_rules = {
            ("a", "gene"): ("A", "0:original:exact", None),
        }
        resolved = _resolve_transitive_merges(merge_rules)
        assert resolved[("a", "gene")][0] == "A"

    def test_order_independence(self):
        """Result should be same regardless of rule insertion order."""
        merge_rules_1 = {
            ("a", "gene"): ("B", "1:before_paren:exact", None),
            ("b", "gene"): ("C", "substring", "merge"),
        }
        merge_rules_2 = {
            ("b", "gene"): ("C", "substring", "merge"),
            ("a", "gene"): ("B", "1:before_paren:exact", None),
        }
        resolved_1 = _resolve_transitive_merges(merge_rules_1)
        resolved_2 = _resolve_transitive_merges(merge_rules_2)
        assert resolved_1[("a", "gene")][0] == resolved_2[("a", "gene")][0] == "C"
        assert resolved_1[("b", "gene")][0] == resolved_2[("b", "gene")][0] == "C"

    def test_empty_rules(self):
        """Empty rules should return empty dict."""
        resolved = _resolve_transitive_merges({})
        assert resolved == {}

    def test_single_rule_no_chain(self):
        """Single rule with no chain should pass through unchanged."""
        merge_rules = {
            ("brca1", "gene"): ("BRCA1", "substring", "merge"),
        }
        resolved = _resolve_transitive_merges(merge_rules)
        assert resolved[("brca1", "gene")] == ("BRCA1", "substring", "merge")
