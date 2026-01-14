"""Tests for entity consolidation caching behavior.

The consolidation system caches LLM decisions to avoid redundant calls:
1. Per-kind caches store (child, parent) → (target, reasoning) mappings
2. Cache hits/misses are tracked for diagnostics
3. _build_rules_from_cache extracts rules from cached decisions
"""

import pytest

from interaction_finder.extraction.stages.consolidate_entities import (
    _add_new_names_to_kind,
    _build_rules_from_cache,
    _get_merge_cache,
)
from interaction_finder.extraction.state import MergeCacheForKind, State
from interaction_finder.extraction.entity_matching import SpeculatedVariant


def build_permitted_pairs(kinds: list[str]) -> dict[str, set[str]]:
    """Build permitted pairs dict for testing."""
    return {kind: {kind} for kind in kinds}


class TestGetMergeCache:
    """Tests for _get_merge_cache function."""

    def test_creates_cache_on_first_access(self):
        """First access to a kind creates a new MergeCacheForKind."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        assert "gene" not in state.merge_cache_by_kind
        cache = _get_merge_cache(state, "gene")
        assert "gene" in state.merge_cache_by_kind
        assert isinstance(cache, MergeCacheForKind)
        assert cache.hits == 0
        assert cache.misses == 0

    def test_returns_existing_cache(self):
        """Subsequent accesses return the same cache instance."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        cache1 = _get_merge_cache(state, "gene")
        cache1.hits = 5  # Modify it
        cache2 = _get_merge_cache(state, "gene")
        assert cache1 is cache2
        assert cache2.hits == 5

    def test_separate_caches_per_kind(self):
        """Different kinds have independent caches."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        gene_cache = _get_merge_cache(state, "gene")
        disease_cache = _get_merge_cache(state, "disease")
        assert gene_cache is not disease_cache
        gene_cache.hits = 10
        assert disease_cache.hits == 0


class TestMergeCacheForKind:
    """Tests for MergeCacheForKind data structure."""

    def test_cache_stores_decisions(self):
        """Cache stores (child, parent) → (target, reasoning) mappings."""
        cache = MergeCacheForKind()
        # Store a merge decision
        cache.cache[("brca1", "BRCA")] = ("BRCA", "llm:merge")
        # Store a skip decision (None target)
        cache.cache[("tp53", "P53")] = (None, "llm:skip")
        # Store a rename decision
        cache.cache[("tgf", "TGF")] = ("TGF-beta", "llm:rename")
        assert cache.cache[("brca1", "BRCA")] == ("BRCA", "llm:merge")
        assert cache.cache[("tp53", "P53")] == (None, "llm:skip")
        assert cache.cache[("tgf", "TGF")] == ("TGF-beta", "llm:rename")

    def test_cache_hit_miss_tracking(self):
        """Hits and misses are tracked correctly."""
        cache = MergeCacheForKind()
        assert cache.hits == 0
        assert cache.misses == 0
        # Simulate cache operations
        cache.hits += 1
        cache.misses += 2
        assert cache.hits == 1
        assert cache.misses == 2


class TestBuildRulesFromCache:
    """Tests for _build_rules_from_cache function."""

    def _make_state_with_cache(self, cache: MergeCacheForKind, kind: str) -> State:
        """Create a State with pre-populated cache."""
        state = State(
            topic="test",
            target_entity_types=[kind],
            permitted_pairs=build_permitted_pairs([kind]),
        )
        state.merge_cache_by_kind[kind] = cache
        return state

    def test_builds_rules_for_merge_decisions(self):
        """Merge decisions (target = parent) create rules."""
        cache = MergeCacheForKind()
        cache.cache[("brca1", "BRCA")] = ("BRCA", "llm:merge")
        state = self._make_state_with_cache(cache, "gene")
        entities = {"BRCA": [], "brca1": []}  # Both exist
        pairs = {("brca1", "BRCA")}
        rules, new_names = _build_rules_from_cache(pairs, "gene", entities, state)
        assert ("brca1", "gene") in rules
        # _build_rules_from_cache injects trigger="substring" for pairwise LLM
        assert rules[("brca1", "gene")] == ("BRCA", "substring", "llm:merge")
        assert len(new_names) == 0  # Target exists

    def test_builds_rules_for_rename_decisions(self):
        """Rename decisions (target != parent) create rules and mark new names."""
        cache = MergeCacheForKind()
        cache.cache[("tgf", "TGF")] = ("TGF-beta", "llm:rename")
        state = self._make_state_with_cache(cache, "gene")
        entities = {"TGF": [], "tgf": []}  # TGF-beta doesn't exist yet
        pairs = {("tgf", "TGF")}
        rules, new_names = _build_rules_from_cache(pairs, "gene", entities, state)
        assert ("tgf", "gene") in rules
        # _build_rules_from_cache injects trigger="substring" for pairwise LLM
        assert rules[("tgf", "gene")] == ("TGF-beta", "substring", "llm:rename")
        assert "TGF-beta" in new_names  # New canonical name

    def test_skip_decisions_not_in_rules(self):
        """Skip decisions (target = None) don't create rules."""
        cache = MergeCacheForKind()
        cache.cache[("tp53", "P53")] = (None, "llm:skip")
        state = self._make_state_with_cache(cache, "gene")
        entities = {"P53": [], "tp53": []}
        pairs = {("tp53", "P53")}
        rules, new_names = _build_rules_from_cache(pairs, "gene", entities, state)
        assert ("tp53", "gene") not in rules
        assert len(new_names) == 0

    def test_uncached_pairs_increments_misses(self):
        """Pairs not in cache increment miss counter."""
        cache = MergeCacheForKind()
        # Cache has one pair
        cache.cache[("brca1", "BRCA")] = ("BRCA", "llm:merge")
        state = self._make_state_with_cache(cache, "gene")
        entities = {"BRCA": [], "brca1": [], "TP53": [], "tp53": []}
        # Query for different pair
        pairs = {("tp53", "TP53")}
        rules, new_names = _build_rules_from_cache(pairs, "gene", entities, state)
        assert len(rules) == 0
        assert cache.misses == 1

    def test_existing_target_not_marked_as_new(self):
        """Target that already exists in entities is not marked as new."""
        cache = MergeCacheForKind()
        cache.cache[("brca1", "BRCA")] = ("BRCA", "llm:merge")
        state = self._make_state_with_cache(cache, "gene")
        entities = {"BRCA": [], "brca1": []}  # BRCA exists
        pairs = {("brca1", "BRCA")}
        rules, new_names = _build_rules_from_cache(pairs, "gene", entities, state)
        assert "BRCA" not in new_names

    def test_cache_hits_incremented(self):
        """Cache hits are incremented for found pairs."""
        cache = MergeCacheForKind()
        cache.cache[("brca1", "BRCA")] = ("BRCA", "llm:merge")
        cache.cache[("tp53", "TP53")] = ("TP53", "llm:merge")
        state = self._make_state_with_cache(cache, "gene")
        entities = {"BRCA": [], "brca1": [], "TP53": [], "tp53": []}
        pairs = {("brca1", "BRCA"), ("tp53", "TP53")}
        _build_rules_from_cache(pairs, "gene", entities, state)
        assert cache.hits == 2


class TestAddNewNamesToKind:
    """Tests for _add_new_names_to_kind function."""

    def test_adds_new_canonical_names(self):
        """New names are added to the entity dict."""
        entities = {
            "TGF": [SpeculatedVariant("TGF", 0, "original", False)],
        }
        new_names = {"TGF-beta"}
        _add_new_names_to_kind(new_names, entities)
        assert "TGF-beta" in entities
        assert len(entities["TGF-beta"]) == 1
        variant = entities["TGF-beta"][0]
        assert variant.form == "TGF-beta"
        assert variant.speculation == 0
        assert variant.source == "original"

    def test_skips_existing_names(self):
        """Names that already exist are not duplicated."""
        entities = {
            "TGF": [SpeculatedVariant("TGF", 0, "original", False)],
        }
        original_variant = entities["TGF"][0]
        new_names = {"TGF"}  # Already exists
        _add_new_names_to_kind(new_names, entities)
        # Should still have exactly one entry
        assert len(entities) == 1
        assert entities["TGF"][0] is original_variant

    def test_adds_multiple_new_names(self):
        """Multiple new names can be added at once."""
        entities = {
            "Gene1": [SpeculatedVariant("Gene1", 0, "original", False)],
        }
        new_names = {"Gene2", "Gene3", "Gene4"}
        _add_new_names_to_kind(new_names, entities)
        assert len(entities) == 4
        assert all(name in entities for name in ["Gene1", "Gene2", "Gene3", "Gene4"])

    def test_skips_case_variants_of_existing_names(self):
        """Names differing only in case from existing entries are not added."""
        entities = {
            "cataract": [SpeculatedVariant("cataract", 0, "original", False)],
        }
        new_names = {"Cataract", "CATARACT"}  # Case variants of existing
        _add_new_names_to_kind(new_names, entities)
        # Should still have exactly one entry - the original
        assert len(entities) == 1
        assert "cataract" in entities
        assert "Cataract" not in entities
        assert "CATARACT" not in entities

    def test_case_insensitive_check_allows_different_entities(self):
        """Genuinely different entities are still added."""
        entities = {
            "cataract": [SpeculatedVariant("cataract", 0, "original", False)],
        }
        new_names = {
            "Glaucoma",
            "Cataract",
        }  # Glaucoma is new, Cataract is case variant
        _add_new_names_to_kind(new_names, entities)
        assert len(entities) == 2
        assert "cataract" in entities
        assert "Glaucoma" in entities
        assert "Cataract" not in entities


class TestCacheStatisticsIntegration:
    """Integration tests for cache statistics tracking."""

    def test_cache_stats_aggregated_across_kinds(self):
        """Cache stats from multiple kinds are aggregated correctly."""
        from interaction_finder.extraction.shared import aggregate_cache_stats

        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        # Set up caches with different stats
        gene_cache = _get_merge_cache(state, "gene")
        gene_cache.hits = 10
        gene_cache.misses = 2
        disease_cache = _get_merge_cache(state, "disease")
        disease_cache.hits = 5
        disease_cache.misses = 3
        # Aggregate
        total_hits, total_misses = aggregate_cache_stats(state)
        assert total_hits == 15
        assert total_misses == 5

    def test_empty_cache_aggregation(self):
        """Empty caches return zeros."""
        from interaction_finder.extraction.shared import aggregate_cache_stats

        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        total_hits, total_misses = aggregate_cache_stats(state)
        assert total_hits == 0
        assert total_misses == 0
