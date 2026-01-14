"""Tests for neighbour-based entity consolidation.

Tests the consolidate_by_neighbours stage which detects entity fragmentation
by analyzing neighbour sets from pair assessments.
"""

import pytest

from interaction_finder.extraction.stages.consolidate_by_neighbours import (
    _build_neighbour_sets,
    _collect_entities_for_clustering,
    _deduplicate_clusters,
    _find_neighbour_clusters,
    _resolve_transitive_merges,
    _apply_merge_rules,
    _update_pair_entity_references,
    NeighbourClusterCandidate,
)
from interaction_finder.extraction.clustering import Cluster
from interaction_finder.extraction.utils import normalize_for_comparison
from interaction_finder.extraction.models import (
    EntityMention,
    EntityRef,
    EvidenceQuality,
    PairAssessment,
)
from interaction_finder.extraction.state import State
from interaction_finder.resources import ResourceId


def make_evidence(level: int = 5) -> EvidenceQuality:
    return EvidenceQuality(
        directness="explicit",
        source_type="primary",
        specificity="mechanistic",
        language="definitive",
        overall=level,
    )


def build_permitted_pairs(kinds: list[str]) -> dict[str, set[str]]:
    return {kind: {kind} for kind in kinds}


def make_entity_ref(name: str, kind: str) -> EntityRef:
    """Create an EntityRef with a single mention."""
    mention = EntityMention(
        kind=kind, name=name, aliases=[], quotes=[], reasoning="test"
    )
    return EntityRef(canonical=name, mentions=[mention])


def make_assessment(
    resource: ResourceId,
    entity1_name: str,
    entity1_kind: str,
    entity2_name: str,
    entity2_kind: str,
    relationship: str = "associated_with",
) -> PairAssessment:
    """Create a minimal PairAssessment for testing."""
    return PairAssessment(
        topic_relevance=3,
        resource_id=resource,
        entity1=make_entity_ref(entity1_name, entity1_kind),
        entity2=make_entity_ref(entity2_name, entity2_kind),
        relationship=relationship,
        quotes=[],
        evidence=make_evidence(7),
        reasoning="Test",
    )


class TestBuildNeighbourSets:
    """Tests for _build_neighbour_sets function."""

    def test_builds_bidirectional_neighbours(self):
        """Neighbour sets include both directions of each pair."""
        state = State(
            topic="test",
            target_entity_types=["gene", "cell"],
            permitted_pairs=build_permitted_pairs(["gene", "cell"]),
        )
        resource = ResourceId(url="https://example.com", counter=1)
        state.pair_assessments_by_resource[resource] = [
            make_assessment(resource, "BMI1", "gene", "Stem cell", "cell"),
        ]
        neighbour_sets = _build_neighbour_sets(state)
        assert "BMI1" in neighbour_sets
        assert "Stem cell" in neighbour_sets
        assert "Stem cell" in neighbour_sets["BMI1"]
        assert "BMI1" in neighbour_sets["Stem cell"]

    def test_aggregates_across_resources(self):
        """Neighbour sets aggregate pairs from all resources."""
        state = State(
            topic="test",
            target_entity_types=["gene", "cell"],
            permitted_pairs=build_permitted_pairs(["gene", "cell"]),
        )
        resource1 = ResourceId(url="https://doc1.com", counter=1)
        resource2 = ResourceId(url="https://doc2.com", counter=2)
        state.pair_assessments_by_resource[resource1] = [
            make_assessment(resource1, "BMI1", "gene", "Cell A", "cell"),
        ]
        state.pair_assessments_by_resource[resource2] = [
            make_assessment(resource2, "BMI1", "gene", "Cell B", "cell"),
        ]
        neighbour_sets = _build_neighbour_sets(state)
        assert neighbour_sets["BMI1"] == {"Cell A", "Cell B"}

    def test_multiple_neighbours_for_anchor(self):
        """Anchor with multiple pairs collects all neighbours."""
        state = State(
            topic="test",
            target_entity_types=["gene", "cell"],
            permitted_pairs=build_permitted_pairs(["gene", "cell"]),
        )
        resource = ResourceId(url="https://example.com", counter=1)
        state.pair_assessments_by_resource[resource] = [
            make_assessment(resource, "BMI1", "gene", "+4 cell", "cell"),
            make_assessment(resource, "BMI1", "gene", "+4 stem cell", "cell"),
            make_assessment(resource, "BMI1", "gene", "Reserve stem cell", "cell"),
        ]
        neighbour_sets = _build_neighbour_sets(state)
        assert neighbour_sets["BMI1"] == {
            "+4 cell",
            "+4 stem cell",
            "Reserve stem cell",
        }


class TestFindNeighbourClusters:
    """Tests for _find_neighbour_clusters function."""

    def test_clusters_similar_neighbours(self):
        """Similar neighbour names are clustered together."""
        state = State(
            topic="test",
            target_entity_types=["gene", "cell"],
            permitted_pairs=build_permitted_pairs(["gene", "cell"]),
        )
        resource = ResourceId(url="https://example.com", counter=1)
        # Set up validated entities with kinds
        state.validated_entities_by_resource[resource] = {
            "BMI1": make_entity_ref("BMI1", "gene"),
            "+4 cell": make_entity_ref("+4 cell", "cell"),
            "+4 stem cell": make_entity_ref("+4 stem cell", "cell"),
            "Reserve stem cell": make_entity_ref("Reserve stem cell", "cell"),
        }
        neighbour_sets = {
            "BMI1": {"+4 cell", "+4 stem cell", "Reserve stem cell"},
        }
        # Extract variants for all entities
        from interaction_finder.extraction.entity_matching import (
            extract_entity_variants,
        )

        all_entities = {
            name: extract_entity_variants(name)
            for name in ["+4 cell", "+4 stem cell", "Reserve stem cell"]
        }
        candidates = _find_neighbour_clusters(
            neighbour_sets, all_entities, threshold=0.5, state=state
        )
        # Should find at least one cluster (+4 cell and +4 stem cell should cluster)
        assert len(candidates) >= 1
        # At least one cluster should contain "+4 cell" and "+4 stem cell"
        found_plus4_cluster = any(
            "+4 cell" in c.cluster and "+4 stem cell" in c.cluster for c in candidates
        )
        assert found_plus4_cluster

    def test_skips_small_neighbour_sets(self):
        """Anchors with fewer than 2 neighbours are skipped."""
        state = State(
            topic="test",
            target_entity_types=["gene", "cell"],
            permitted_pairs=build_permitted_pairs(["gene", "cell"]),
        )
        resource = ResourceId(url="https://example.com", counter=1)
        state.validated_entities_by_resource[resource] = {
            "BMI1": make_entity_ref("BMI1", "gene"),
            "Single cell": make_entity_ref("Single cell", "cell"),
        }
        neighbour_sets = {
            "BMI1": {"Single cell"},  # Only one neighbour
        }
        from interaction_finder.extraction.entity_matching import (
            extract_entity_variants,
        )

        all_entities = {"Single cell": extract_entity_variants("Single cell")}
        candidates = _find_neighbour_clusters(
            neighbour_sets, all_entities, threshold=0.5, state=state
        )
        assert len(candidates) == 0

    def test_distinct_neighbours_not_clustered(self):
        """Distinct neighbour names don't form clusters."""
        state = State(
            topic="test",
            target_entity_types=["gene", "cell"],
            permitted_pairs=build_permitted_pairs(["gene", "cell"]),
        )
        resource = ResourceId(url="https://example.com", counter=1)
        state.validated_entities_by_resource[resource] = {
            "ISC": make_entity_ref("ISC", "cell"),
            "LGR5": make_entity_ref("LGR5", "gene"),
            "OLFM4": make_entity_ref("OLFM4", "gene"),
            "BMI1": make_entity_ref("BMI1", "gene"),
        }
        neighbour_sets = {
            "ISC": {"LGR5", "OLFM4", "BMI1"},  # All distinct genes
        }
        from interaction_finder.extraction.entity_matching import (
            extract_entity_variants,
        )

        all_entities = {
            name: extract_entity_variants(name) for name in ["LGR5", "OLFM4", "BMI1"]
        }
        candidates = _find_neighbour_clusters(
            neighbour_sets, all_entities, threshold=0.5, state=state
        )
        # No clusters should form - these are distinct gene symbols
        assert len(candidates) == 0


class TestDeduplicateClusters:
    """Tests for _deduplicate_clusters function."""

    def test_removes_duplicate_clusters(self):
        """Same cluster from different anchors is deduplicated."""
        cluster = frozenset({"Cell A", "Cell B"})
        tree = Cluster(entities=cluster, similarity=0.8)
        candidates = [
            NeighbourClusterCandidate(
                anchor="Gene1",
                anchor_kind="gene",
                cluster=cluster,
                merge_tree=tree,
                similarity=0.8,
                cluster_kind="cell",
            ),
            NeighbourClusterCandidate(
                anchor="Gene2",
                anchor_kind="gene",
                cluster=cluster,
                merge_tree=tree,
                similarity=0.7,
                cluster_kind="cell",
            ),
        ]
        unique = _deduplicate_clusters(candidates)
        assert len(unique) == 1
        # Should keep the one with higher similarity
        assert unique[0].similarity == 0.8
        assert unique[0].anchor == "Gene1"

    def test_keeps_different_clusters(self):
        """Different clusters are preserved."""
        cluster1 = frozenset({"Cell A", "Cell B"})
        cluster2 = frozenset({"Cell X", "Cell Y"})
        tree1 = Cluster(entities=cluster1, similarity=0.8)
        tree2 = Cluster(entities=cluster2, similarity=0.7)
        candidates = [
            NeighbourClusterCandidate(
                anchor="Gene1",
                anchor_kind="gene",
                cluster=cluster1,
                merge_tree=tree1,
                similarity=0.8,
                cluster_kind="cell",
            ),
            NeighbourClusterCandidate(
                anchor="Gene2",
                anchor_kind="gene",
                cluster=cluster2,
                merge_tree=tree2,
                similarity=0.7,
                cluster_kind="cell",
            ),
        ]
        unique = _deduplicate_clusters(candidates)
        assert len(unique) == 2

    def test_same_cluster_different_kinds_not_deduplicated(self):
        """Same cluster with different kinds are kept separate."""
        cluster = frozenset({"Entity A", "Entity B"})
        tree = Cluster(entities=cluster, similarity=0.8)
        candidates = [
            NeighbourClusterCandidate(
                anchor="Gene1",
                anchor_kind="gene",
                cluster=cluster,
                merge_tree=tree,
                similarity=0.8,
                cluster_kind="celltype",
            ),
            NeighbourClusterCandidate(
                anchor="Gene2",
                anchor_kind="gene",
                cluster=cluster,
                merge_tree=tree,
                similarity=0.8,
                cluster_kind="disease",
            ),
        ]
        unique = _deduplicate_clusters(candidates)
        # Same cluster but different kinds should NOT be deduplicated
        assert len(unique) == 2


class TestResolveTransitiveMerges:
    """Tests for _resolve_transitive_merges function."""

    def test_resolves_chain(self):
        """A→B, B→C becomes A→C, B→C."""
        rules = {
            ("a", "cell"): ("B", "neighbour(X):cluster(1,0.5)", "reason1"),
            ("b", "cell"): ("C", "neighbour(X):cluster(1,0.5)", "reason2"),
        }
        resolved = _resolve_transitive_merges(rules)
        assert resolved[("a", "cell")][0] == "C"
        assert resolved[("b", "cell")][0] == "C"

    def test_preserves_direct_rules(self):
        """Non-chained rules are preserved unchanged."""
        rules = {
            ("a", "cell"): ("B", "neighbour(X):cluster(1,0.5)", "reason1"),
            ("x", "gene"): ("Y", "neighbour(X):cluster(1,0.5)", "reason2"),
        }
        resolved = _resolve_transitive_merges(rules)
        assert resolved[("a", "cell")] == (
            "B",
            "neighbour(X):cluster(1,0.5)",
            "reason1",
        )
        assert resolved[("x", "gene")] == (
            "Y",
            "neighbour(X):cluster(1,0.5)",
            "reason2",
        )


class TestApplyMergeRules:
    """Tests for _apply_merge_rules function."""

    def test_updates_validated_entities(self):
        """Merge rules update validated_entities_by_resource."""
        state = State(
            topic="test",
            target_entity_types=["cell"],
            permitted_pairs=build_permitted_pairs(["cell"]),
        )
        resource = ResourceId(url="https://example.com", counter=1)
        state.validated_entities_by_resource[resource] = {
            "+4 cell": make_entity_ref("+4 cell", "cell"),
            "+4 stem cell": make_entity_ref("+4 stem cell", "cell"),
        }
        # Rules use normalized keys - format: (target, trigger, reasoning)
        rules = {
            (normalize_for_comparison("+4 cell"), "cell"): (
                "+4 stem cell",
                "neighbour(BMI1):cluster(abc,0.65)",
                "merged",
            ),
        }
        _apply_merge_rules(rules, state)
        # +4 cell should be merged into +4 stem cell
        entities = state.validated_entities_by_resource[resource]
        assert "+4 cell" not in entities
        assert "+4 stem cell" in entities

    def test_tracks_in_consolidated(self):
        """Merge rules are tracked in state.consolidated."""
        state = State(
            topic="test",
            target_entity_types=["cell"],
            permitted_pairs=build_permitted_pairs(["cell"]),
        )
        resource = ResourceId(url="https://example.com", counter=1)
        state.validated_entities_by_resource[resource] = {
            "+4 cell": make_entity_ref("+4 cell", "cell"),
            "+4 stem cell": make_entity_ref("+4 stem cell", "cell"),
        }
        # Rules use normalized keys - format: (target, trigger, reasoning)
        rules = {
            (normalize_for_comparison("+4 cell"), "cell"): (
                "+4 stem cell",
                "neighbour(BMI1):cluster(abc,0.65)",
                "merged",
            ),
        }
        _apply_merge_rules(rules, state)
        # Should be tracked in consolidated
        assert "cell" in state.consolidated.entities.merges
        assert len(state.consolidated.entities.merges["cell"].llm_decided) == 1


class TestUpdatePairEntityReferences:
    """Tests for _update_pair_entity_references function."""

    def test_updates_entity_canonicals(self):
        """Pair assessment entity references are updated."""
        state = State(
            topic="test",
            target_entity_types=["gene", "cell"],
            permitted_pairs=build_permitted_pairs(["gene", "cell"]),
        )
        resource = ResourceId(url="https://example.com", counter=1)
        state.pair_assessments_by_resource[resource] = [
            make_assessment(resource, "BMI1", "gene", "+4 cell", "cell"),
        ]
        state.validated_entities_by_resource[resource] = {
            "+4 stem cell": make_entity_ref("+4 stem cell", "cell"),
        }
        # Rules use normalized keys
        rules = {
            (normalize_for_comparison("+4 cell"), "cell"): (
                "+4 stem cell",
                "llm:neighbour(BMI1):merged",
            ),
        }
        _update_pair_entity_references(rules, state)
        updated = state.pair_assessments_by_resource[resource][0]
        assert updated.entity2.canonical == "+4 stem cell"


class TestIntegration:
    """Integration tests for the full neighbour clustering workflow."""

    def test_bmi1_neighbours_cluster(self):
        """BMI1's neighbours cluster as expected (real-world example)."""
        from interaction_finder.extraction.entity_matching import (
            extract_entity_variants,
        )
        from interaction_finder.extraction.clustering import cluster_entities

        # Real example from the discussion
        neighbours = [
            "+4 cell",
            "+4 stem cell",
            "+4 position cell",
            "Position +4 stem cells",
            "Reserve stem cell",
            "Reserve intestinal stem cell",
            "Intestinal stem cell",
        ]
        entities = {n: extract_entity_variants(n) for n in neighbours}
        clusters, trees = cluster_entities(entities, threshold=0.5)
        # Should find multi-member clusters
        multi_member = [c for c in clusters if len(c) > 1]
        assert len(multi_member) >= 1
        # +4 variants should cluster together
        for cluster in multi_member:
            plus4_count = sum(1 for e in cluster if "+4" in e or "Plus 4" in e)
            if plus4_count > 0:
                assert plus4_count >= 2  # At least 2 +4 variants in a cluster

    def test_isc_neighbours_mostly_singletons(self):
        """ISC's gene neighbours remain mostly singletons (real-world example)."""
        from interaction_finder.extraction.entity_matching import (
            extract_entity_variants,
        )
        from interaction_finder.extraction.clustering import cluster_entities

        # Real gene markers - should not cluster
        neighbours = [
            "LGR5",
            "OLFM4",
            "BMI1",
            "ASCL2",
            "SOX9",
            "HOPX",
            "CD44",
            "CD133",
            "TERT",
            "MYC",
            "AXIN2",
            "MSI1",
        ]
        entities = {n: extract_entity_variants(n) for n in neighbours}
        clusters, _ = cluster_entities(entities, threshold=0.5)
        # Most should be singletons
        singletons = [c for c in clusters if len(c) == 1]
        multi_member = [c for c in clusters if len(c) > 1]
        # At most 1-2 small clusters from any coincidental similarity
        assert len(singletons) >= len(neighbours) - 4
        # No large clusters
        for cluster in multi_member:
            assert len(cluster) <= 3
