"""Tests for per-kind entity consolidation behavior.

Verifies that:
1. Each entity kind has its own independent cache
2. Merge rules are keyed by (normalized_name, kind) - kinds don't cross-contaminate
3. Same normalized name in different kinds stays separate
"""

import pytest

from interaction_finder.extraction.models import EntityMention, EntityRef
from interaction_finder.extraction.stages.consolidate_entities import (
    _add_new_names_to_kind,
    _apply_merge_rules_globally,
    _get_merge_cache,
    _resolve_transitive_merges,
)
from interaction_finder.extraction.state import MergeCacheForKind, State
from interaction_finder.extraction.entity_matching import SpeculatedVariant
from interaction_finder.resources import ResourceId


def build_permitted_pairs(kinds: list[str]) -> dict[str, set[str]]:
    """Build permitted pairs dict for testing."""
    return {kind: {kind} for kind in kinds}


class TestPerKindCacheIndependence:
    """Test that each kind has its own independent cache."""

    def test_separate_caches_per_kind(self):
        """Different kinds have completely independent caches."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        gene_cache = _get_merge_cache(state, "gene")
        disease_cache = _get_merge_cache(state, "disease")
        # Populate gene cache
        gene_cache.cache[("brca1", "BRCA")] = ("BRCA", "llm:merge")
        gene_cache.hits = 10
        # Disease cache should be unaffected
        assert ("brca1", "BRCA") not in disease_cache.cache
        assert disease_cache.hits == 0

    def test_cache_operations_isolated(self):
        """Cache hits/misses are tracked per-kind."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease", "phenotype"],
            permitted_pairs=build_permitted_pairs(["gene", "disease", "phenotype"]),
        )
        # Simulate different cache patterns for each kind
        gene_cache = _get_merge_cache(state, "gene")
        gene_cache.hits = 5
        gene_cache.misses = 2
        disease_cache = _get_merge_cache(state, "disease")
        disease_cache.hits = 3
        disease_cache.misses = 7
        phenotype_cache = _get_merge_cache(state, "phenotype")
        phenotype_cache.hits = 0
        phenotype_cache.misses = 1
        # Verify each cache maintains its own stats
        assert state.merge_cache_by_kind["gene"].hits == 5
        assert state.merge_cache_by_kind["disease"].hits == 3
        assert state.merge_cache_by_kind["phenotype"].hits == 0


class TestSameNameDifferentKinds:
    """Test that same normalized name in different kinds stays separate."""

    def test_same_name_different_kinds_not_merged(self):
        """Entity 'PAH' as gene and 'PAH' as disease should stay separate."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        resource = ResourceId(url="https://doc1.com", counter=0)
        # Same canonical name, different kinds
        state.validated_entities_by_resource = {
            resource: {
                "PAH_gene": EntityRef(
                    canonical="PAH_gene",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="PAH",
                            aliases=[],
                            quotes=[],
                            reasoning="Gene entity",
                        )
                    ],
                ),
                "PAH_disease": EntityRef(
                    canonical="PAH_disease",
                    mentions=[
                        EntityMention(
                            kind="disease",
                            name="PAH",
                            aliases=["Pulmonary arterial hypertension"],
                            quotes=[],
                            reasoning="Disease entity",
                        )
                    ],
                ),
            }
        }
        # Rule only for gene kind
        merge_rules = {("pah_gene", "gene"): ("PAH", "0:merge:exact", None)}
        _apply_merge_rules_globally(merge_rules, state)
        entities = state.validated_entities_by_resource[resource]
        # Disease should be unaffected (rule is for gene kind)
        assert "PAH_disease" in entities

    def test_rules_keyed_by_kind(self):
        """Merge rules use (normalized_name, kind) tuple as key."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
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
                            reasoning="gene",
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
                            reasoning="gene variant",
                        )
                    ],
                ),
            }
        }
        # Rule specifies "gene" kind explicitly
        merge_rules = {("brca1", "gene"): ("BRCA1", "0:case:exact", None)}
        _apply_merge_rules_globally(merge_rules, state)
        entities = state.validated_entities_by_resource[resource]
        # Should have merged the gene variants
        assert len(entities) == 1
        assert "BRCA1" in entities


class TestAddNewNamesPerKind:
    """Test that new names are added to correct kind's entity dict."""

    def test_new_names_added_to_correct_dict(self):
        """_add_new_names_to_kind only affects the target entities dict."""
        gene_entities = {
            "BRCA1": [SpeculatedVariant("BRCA1", 0, "original", False)],
        }
        disease_entities = {
            "Cancer": [SpeculatedVariant("Cancer", 0, "original", False)],
        }
        # Add new name to gene entities
        new_names = {"BRCA2"}
        _add_new_names_to_kind(new_names, gene_entities)
        # Gene dict has new name
        assert "BRCA2" in gene_entities
        # Disease dict unaffected
        assert "BRCA2" not in disease_entities
        assert len(disease_entities) == 1


class TestTransitiveMergesPerKind:
    """Test that transitive merge resolution respects kinds."""

    def test_transitive_merges_use_kind_key(self):
        """Transitive chains are resolved using (normalized, kind) keys."""
        # Chain: geneA -> geneB -> geneC
        unresolved_rules = {
            ("genea", "gene"): ("GeneB", "0:1:exact", None),
            ("geneb", "gene"): ("GeneC", "0:2:exact", None),
        }
        resolved = _resolve_transitive_merges(unresolved_rules)
        # Both should resolve to GeneC
        assert resolved[("genea", "gene")][0] == "GeneC"
        assert resolved[("geneb", "gene")][0] == "GeneC"

    def test_chains_dont_cross_kinds(self):
        """A chain in one kind doesn't affect another kind."""
        # Gene chain: geneA -> geneB
        # Disease chain: diseaseX -> diseaseY (separate)
        unresolved_rules = {
            ("genea", "gene"): ("GeneB", "0:gene:exact", None),
            ("diseasex", "disease"): ("DiseaseY", "0:disease:exact", None),
        }
        resolved = _resolve_transitive_merges(unresolved_rules)
        # Each resolves independently
        assert resolved[("genea", "gene")][0] == "GeneB"
        assert resolved[("diseasex", "disease")][0] == "DiseaseY"


class TestMergeRulesApplicationPerKind:
    """Test that merge rules are correctly applied per kind."""

    def test_gene_rules_only_affect_genes(self):
        """Rules with kind='gene' only merge gene entities."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
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
                            reasoning="gene parent",
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
                            reasoning="gene child",
                        )
                    ],
                ),
                "Cancer": EntityRef(
                    canonical="Cancer",
                    mentions=[
                        EntityMention(
                            kind="disease",
                            name="Cancer",
                            aliases=[],
                            quotes=[],
                            reasoning="disease",
                        )
                    ],
                ),
            }
        }
        # Only gene rule
        merge_rules = {("brca1", "gene"): ("BRCA1", "0:case:exact", None)}
        _apply_merge_rules_globally(merge_rules, state)
        entities = state.validated_entities_by_resource[resource]
        # Genes merged, disease untouched
        assert len(entities) == 2
        assert "BRCA1" in entities
        assert "Cancer" in entities
        assert "brca1" not in entities

    def test_multiple_kind_rules_applied_correctly(self):
        """Rules for multiple kinds are applied to respective kinds."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
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
                            reasoning="gene parent",
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
                            reasoning="gene child",
                        )
                    ],
                ),
                "Cancer": EntityRef(
                    canonical="Cancer",
                    mentions=[
                        EntityMention(
                            kind="disease",
                            name="Cancer",
                            aliases=[],
                            quotes=[],
                            reasoning="disease parent",
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
                            reasoning="disease child",
                        )
                    ],
                ),
            }
        }
        # Rules for both kinds
        merge_rules = {
            ("brca1", "gene"): ("BRCA1", "0:gene:exact", None),
            ("cancer", "disease"): ("Cancer", "0:disease:exact", None),
        }
        _apply_merge_rules_globally(merge_rules, state)
        entities = state.validated_entities_by_resource[resource]
        # Both kinds merged
        assert len(entities) == 2
        assert "BRCA1" in entities
        assert "Cancer" in entities
