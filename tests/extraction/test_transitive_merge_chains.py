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
            ("actb variant", "gene"): ("ACTB", "auto:2:alternation:exact"),
            ("actb", "gene"): ("β-Actin", "auto:1:before_paren:exact"),
        }
        resolved = _resolve_transitive_merges(merge_rules)
        assert resolved[("actb variant", "gene")][0] == "β-Actin"
        assert resolved[("actb", "gene")][0] == "β-Actin"
        # Reasoning from final link
        assert resolved[("actb variant", "gene")][1] == "auto:1:before_paren:exact"

    def test_auto_to_llm_chain(self):
        """Auto A→B + LLM B→D should resolve to A→D."""
        merge_rules = {
            ("brca1 variant", "gene"): ("BRCA1", "auto:2:alternation:exact"),
            ("brca1", "gene"): ("BRCA", "llm:rename"),
        }
        resolved = _resolve_transitive_merges(merge_rules)
        assert resolved[("brca1 variant", "gene")][0] == "BRCA"
        assert resolved[("brca1", "gene")][0] == "BRCA"
        assert resolved[("brca1 variant", "gene")][1] == "llm:rename"

    def test_llm_chain_with_auto_prefix(self):
        """LLM B→C + LLM C→D + Auto A→B should resolve A→D."""
        merge_rules = {
            ("gene variant", "gene"): ("Gene X", "auto:2:alternation:exact"),
            ("gene x", "gene"): ("GeneX", "llm:merge"),
            ("genex", "gene"): ("GENEX", "llm:rename"),
        }
        resolved = _resolve_transitive_merges(merge_rules)
        assert resolved[("gene variant", "gene")][0] == "GENEX"
        assert resolved[("gene x", "gene")][0] == "GENEX"
        assert resolved[("genex", "gene")][0] == "GENEX"
        assert resolved[("gene variant", "gene")][1] == "llm:rename"

    def test_long_mixed_chain(self):
        """Long chain with alternating auto/LLM decisions."""
        merge_rules = {
            ("a", "gene"): ("B", "auto:1:before_paren:exact"),
            ("b", "gene"): ("C", "llm:merge"),
            ("c", "gene"): ("D", "auto:3:fuzzy"),
            ("d", "gene"): ("E", "llm:rename"),
        }
        resolved = _resolve_transitive_merges(merge_rules)
        for key in [("a", "gene"), ("b", "gene"), ("c", "gene"), ("d", "gene")]:
            assert resolved[key][0] == "E"
            assert resolved[key][1] == "llm:rename"

    def test_multiple_chains_independent(self):
        """Multiple independent chains should not interfere."""
        merge_rules = {
            ("a", "gene"): ("B", "auto:1:before_paren:exact"),
            ("b", "gene"): ("C", "llm:merge"),
            ("x", "phenotype"): ("Y", "auto:2:alternation:exact"),
            ("y", "phenotype"): ("Z", "llm:rename"),
        }
        resolved = _resolve_transitive_merges(merge_rules)
        assert resolved[("a", "gene")][0] == "C"
        assert resolved[("b", "gene")][0] == "C"
        assert resolved[("x", "phenotype")][0] == "Z"
        assert resolved[("y", "phenotype")][0] == "Z"

    def test_cycle_detection(self):
        """Cycles should be detected and broken without infinite loop."""
        merge_rules = {
            ("a", "gene"): ("B", "auto:1:before_paren:exact"),
            ("b", "gene"): ("A", "llm:merge"),
        }
        # Should complete without infinite loop
        resolved = _resolve_transitive_merges(merge_rules)
        assert resolved[("a", "gene")][0] in ["A", "B"]
        assert resolved[("b", "gene")][0] in ["A", "B"]

    def test_self_reference_ignored(self):
        """Self-references should be handled gracefully."""
        merge_rules = {
            ("a", "gene"): ("A", "auto:0:original:exact"),
        }
        resolved = _resolve_transitive_merges(merge_rules)
        assert resolved[("a", "gene")][0] == "A"

    def test_order_independence(self):
        """Result should be same regardless of rule insertion order."""
        merge_rules_1 = {
            ("a", "gene"): ("B", "auto:1:before_paren:exact"),
            ("b", "gene"): ("C", "llm:merge"),
        }
        merge_rules_2 = {
            ("b", "gene"): ("C", "llm:merge"),
            ("a", "gene"): ("B", "auto:1:before_paren:exact"),
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
            ("brca1", "gene"): ("BRCA1", "llm:merge"),
        }
        resolved = _resolve_transitive_merges(merge_rules)
        assert resolved[("brca1", "gene")] == ("BRCA1", "llm:merge")
