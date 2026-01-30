"""Tests for hierarchical entity clustering with IDF weighting."""

import pytest
from interaction_finder.extraction.clustering import (
    Cluster,
    tokenize,
    compute_token_specificity,
    agglomerative_cluster,
    cluster_entities,
    _weighted_similarity,
    _get_entity_tokens,
)
from interaction_finder.extraction.entity_matching import (
    SpeculatedVariant,
    SPEC_ORIGINAL,
    SPEC_PRIMARY_CONTENT,
    SPEC_SECONDARY_CONTENT,
)


def make_variant(form: str, spec: int = SPEC_ORIGINAL) -> SpeculatedVariant:
    """Create a SpeculatedVariant for testing."""
    return SpeculatedVariant(form=form, speculation=spec, source="test")


def make_entities(
    data: dict[str, list[str | tuple[str, int]]],
) -> dict[str, list[SpeculatedVariant]]:
    """Create entity dict from simplified spec.

    Args:
        data: Dict of canonical → list of variants. Each variant can be:
              - str: variant form with SPEC_ORIGINAL
              - (str, int): variant form with specific speculation level
    """
    result = {}
    for canonical, variants in data.items():
        var_list = []
        for v in variants:
            if isinstance(v, str):
                var_list.append(make_variant(v, SPEC_ORIGINAL))
            else:
                var_list.append(make_variant(v[0], v[1]))
        result[canonical] = var_list
    return result


class TestTokenize:
    """Test token extraction."""

    def test_basic_tokenization(self):
        """Extract normalized tokens from text."""
        tokens = tokenize("Pulmonary arterial hypertension")
        # Base tokens from splitting on whitespace
        assert {"pulmonary", "arterial", "hypertension"}.issubset(tokens)
        # Collapsed form also added for multi-word entities
        assert "pulmonaryarterialhypertension" in tokens

    def test_hyphenated(self):
        """Hyphens split tokens AND add dehyphenated form."""
        tokens = tokenize("IL-6")
        # Split tokens preserved
        assert "il" in tokens
        assert "6" in tokens
        # Dehyphenated form added for better clustering
        assert "il6" in tokens

    def test_stopwords_filtered(self):
        """Common stopwords are removed."""
        tokens = tokenize("The role of genetics in disease")
        assert "the" not in tokens
        assert "of" not in tokens
        assert "in" not in tokens
        assert "role" in tokens
        assert "genetics" in tokens
        assert "disease" in tokens

    def test_single_char_non_stopwords_preserved(self):
        """Single chars preserved unless stopwords (a/i/s/t are stopwords)."""
        tokens = tokenize("B K EFG")
        assert "b" in tokens
        assert "k" in tokens
        assert "efg" in tokens

    def test_empty_after_filtering(self):
        """Returns empty set if all tokens filtered."""
        tokens = tokenize("the of in")  # all stopwords
        assert tokens == frozenset()

    def test_slash_splits_tokens(self):
        r"""Slashes should split tokens (regression test for PAH clustering bug).

        Bug: tokenize() was splitting on [\s\-]+ but not /, causing
        "idiopathic/heritable PAH" to become single token "idiopathic heritable".
        This prevented clustering of entities with slash-separated qualifiers.
        """
        tokens = tokenize("idiopathic/heritable PAH")
        assert "idiopathic" in tokens
        assert "heritable" in tokens
        assert "pah" in tokens
        assert "idiopathic heritable" not in tokens  # Should NOT be single token

    def test_hyphenated_vs_unhyphenated_overlap(self):
        """IGF-1 and IGF1 should share tokens via dehyphenated form.

        The tokenizer adds dehyphenated forms so that entities with different
        hyphenation conventions can still cluster together.
        """
        tokens_hyphen = tokenize("IGF-1")
        tokens_no_hyphen = tokenize("IGF1")
        # Both should contain the collapsed form
        assert "igf1" in tokens_hyphen
        assert "igf1" in tokens_no_hyphen
        # They should have overlap
        overlap = tokens_hyphen & tokens_no_hyphen
        assert "igf1" in overlap

    def test_mmp_variants_overlap(self):
        """MMP-1 and MMP1 should share tokens."""
        tokens_hyphen = tokenize("MMP-1")
        tokens_no_hyphen = tokenize("MMP1")
        overlap = tokens_hyphen & tokens_no_hyphen
        assert "mmp1" in overlap

    def test_multiword_collapsed_form(self):
        """p16 INK4A and p16INK4A should share tokens via collapsed form."""
        tokens_spaced = tokenize("p16 INK4A")
        tokens_collapsed = tokenize("p16INK4A")
        # Spaced version should have collapsed form
        assert "p16ink4a" in tokens_spaced
        # Collapsed version normalizes to same
        assert "p16ink4a" in tokens_collapsed
        # They should overlap
        overlap = tokens_spaced & tokens_collapsed
        assert "p16ink4a" in overlap

    def test_unicode_hyphen_ascii_hyphen_overlap(self):
        """SSEA-4 (ASCII) and SSEA‑4 (Unicode U+2011) should share tokens.

        Unicode hyphens are normalized to ASCII in text_mapping, then
        the tokenizer produces the same dehyphenated form.
        """
        from interaction_finder.text_mapping import NormalizedTextMapper

        # Verify normalization produces same result
        ascii_norm = NormalizedTextMapper.normalize("SSEA-4")
        unicode_norm = NormalizedTextMapper.normalize("SSEA\u20114")  # U+2011
        assert ascii_norm == unicode_norm == "ssea4"
        # Both should tokenize to overlapping forms
        tokens_ascii = tokenize("SSEA-4")
        tokens_unicode = tokenize("SSEA\u20114")
        overlap = tokens_ascii & tokens_unicode
        assert "ssea4" in overlap

    def test_different_numbers_no_overlap(self):
        """CD105 and CD106 should NOT overlap (different numbers)."""
        tokens1 = tokenize("CD105")
        tokens2 = tokenize("CD106")
        # They have different numbers, so no meaningful overlap
        overlap = tokens1 & tokens2
        # The only possible overlap would be 'cd' if numbers were split,
        # but single tokens like 'cd105' and 'cd106' shouldn't overlap
        assert "cd105" in tokens1
        assert "cd106" in tokens2
        assert "cd105" not in overlap
        assert "cd106" not in overlap


class TestSpecificity:
    """Test IDF-like specificity scoring."""

    def test_common_token_low_specificity(self):
        """Tokens appearing in many entities have lower specificity."""
        entities = make_entities(
            {
                "A": ["alpha common"],
                "B": ["beta common"],
                "C": ["gamma common"],
                "D": ["delta rare"],
            }
        )
        spec = compute_token_specificity(entities)
        # "common" appears in 3 entities, "rare" in 1
        assert spec["common"] < spec["rare"]

    def test_mention_weighting(self):
        """Entities with more mentions weight tokens higher (lower specificity)."""
        # Use non-Greek letters since "alpha"→"a" and "beta"→"b" after normalization
        entities = make_entities(
            {
                "HighMention": ["zinc"],
                "LowMention": ["iron"],
            }
        )
        mention_counts = {"HighMention": 100, "LowMention": 1}
        spec = compute_token_specificity(entities, mention_counts)
        # "zinc" has weight 100, "iron" has weight 1
        # Lower weight → higher specificity
        assert spec["zinc"] < spec["iron"]

    def test_speculation_weighting(self):
        """Lower speculation weights tokens higher."""
        entities = make_entities(
            {
                "A": [
                    ("confident", SPEC_ORIGINAL),
                    ("speculative", SPEC_SECONDARY_CONTENT),
                ],
            }
        )
        mention_counts = {"A": 40}  # 40 mentions
        spec = compute_token_specificity(entities, mention_counts)
        # "confident" weight: 40 / (0+1) = 40
        # "speculative" weight: 40 / (3+1) = 10
        # Lower weight → higher specificity
        assert spec["speculative"] > spec["confident"]

    def test_max_weight_per_entity(self):
        """Each token takes max weight from variants within same entity."""
        entities = make_entities(
            {
                "A": [
                    ("shared term", SPEC_ORIGINAL),  # weight 10
                    ("shared other", SPEC_SECONDARY_CONTENT),  # weight 2.5
                ],
            }
        )
        mention_counts = {"A": 10}
        spec = compute_token_specificity(entities, mention_counts)
        # "shared" appears at SPEC=0 (weight 10) and SPEC=3 (weight 2.5)
        # Takes max: 10
        # Both "term" (SPEC=0) and "shared" should have weight 10
        assert spec["shared"] == spec["term"]

    def test_empty_entities(self):
        """Empty input returns empty dict."""
        spec = compute_token_specificity({})
        assert spec == {}

    def test_canonical_name_half_weighting(self):
        """Canonical name tokens are weighted 0.5x vs variant tokens."""
        entities = make_entities(
            {
                "Breast cancer": ["Breast cancer"],  # Canonical
                "Colorectal cancer": ["Colorectal cancer"],  # Canonical
            }
        )
        spec = compute_token_specificity(entities)
        # Both entities have "cancer" in canonical name (0.5x weight each)
        # Each entity has unique tokens (breast, colorectal) in canonical (0.5x weight)
        # "cancer" appears with weight 0.5 + 0.5 = 1.0
        # "breast" appears with weight 0.5
        # "colorectal" appears with weight 0.5
        # "cancer" is more common → lower specificity
        assert spec["cancer"] < spec["breast"]  # More common → lower specificity
        assert spec["cancer"] < spec["colorectal"]

    def test_canonical_weighting_enables_variant_clustering(self):
        """0.5x canonical weighting strengthens clustering on shared canonical terms."""
        # Simulate PAH variants that should cluster together
        # Both share "pulmonary arterial hypertension" in canonical names
        entities = make_entities(
            {
                "Familial pulmonary arterial hypertension": [
                    "Familial pulmonary arterial hypertension"
                ],
                "Heritable pulmonary arterial hypertension": [
                    "Heritable pulmonary arterial hypertension"
                ],
            }
        )
        spec = compute_token_specificity(entities)
        # With 0.5x weighting, shared canonical tokens have higher specificity
        # "pulmonary", "arterial", "hypertension" appear with weight 0.5 + 0.5 = 1.0
        # "familial", "heritable" appear with weight 0.5 each
        # Shared tokens should have lower specificity (more common)
        assert spec["pulmonary"] < spec["familial"]
        assert spec["arterial"] < spec["heritable"]
        # Now check similarity - should be high due to shared canonical terms
        from interaction_finder.extraction.clustering import _get_entity_tokens

        tokens1 = _get_entity_tokens(
            entities["Familial pulmonary arterial hypertension"]
        )
        tokens2 = _get_entity_tokens(
            entities["Heritable pulmonary arterial hypertension"]
        )
        sim, _ = _weighted_similarity(tokens1, tokens2, spec)
        # Similarity should be high (>0.5) due to shared canonical terms
        # dominating over unique modifiers
        assert sim > 0.5, f"Expected similarity >0.5, got {sim}"


class TestWeightedSimilarity:
    """Test similarity calculation."""

    def test_identical_tokens(self):
        """Identical token sets have similarity 1.0."""
        tokens = frozenset({"alpha", "beta"})
        spec = {"alpha": 1.0, "beta": 1.0}
        sim, _ = _weighted_similarity(tokens, tokens, spec)
        assert sim == 1.0

    def test_no_overlap(self):
        """Disjoint token sets have similarity 0.0."""
        t1 = frozenset({"alpha", "beta"})
        t2 = frozenset({"gamma", "delta"})
        spec = {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "delta": 1.0}
        sim, tokens = _weighted_similarity(t1, t2, spec)
        assert sim == 0.0
        assert tokens == ()

    def test_partial_overlap(self):
        """Partial overlap gives proportional similarity."""
        t1 = frozenset({"alpha", "beta"})  # total weight 2.0
        t2 = frozenset({"beta", "gamma"})  # total weight 2.0, shared "beta" = 1.0
        spec = {"alpha": 1.0, "beta": 1.0, "gamma": 1.0}
        sim, _ = _weighted_similarity(t1, t2, spec)
        assert sim == 0.5  # 1.0 / 2.0

    def test_high_specificity_tokens_matter_more(self):
        """Sharing high-specificity tokens gives higher similarity."""
        t1 = frozenset({"rare", "common"})
        t2 = frozenset({"rare", "other"})
        spec = {"rare": 5.0, "common": 0.5, "other": 0.5}  # rare is 10x more specific
        sim, tokens = _weighted_similarity(t1, t2, spec)
        # Shared: rare (5.0)
        # t1 total: 5.5, t2 total: 5.5, min = 5.5
        # sim = 5.0 / 5.5 ≈ 0.91
        assert sim > 0.9
        # Contributing tokens should list "rare" first
        assert tokens[0][0] == "rare"

    def test_contributing_tokens_sorted(self):
        """Contributing tokens are sorted by specificity descending."""
        t1 = frozenset({"a", "b", "c"})
        t2 = frozenset({"a", "b", "c"})
        spec = {"a": 1.0, "b": 3.0, "c": 2.0}
        _, tokens = _weighted_similarity(t1, t2, spec)
        assert tokens[0][0] == "b"  # highest specificity
        assert tokens[1][0] == "c"
        assert tokens[2][0] == "a"


class TestClusterDataStructure:
    """Test Cluster split operations."""

    def test_leaf_split(self):
        """Leaf cluster returns itself."""
        leaf = Cluster(frozenset({"A"}))
        result = leaf.split(0.5)
        assert result == [frozenset({"A"})]

    def test_split_at_threshold(self):
        """Split cuts links below threshold."""
        # Tree: [A,B,C] with [A,B] merged at 0.8, then [A,B,C] at 0.4
        leaf_a = Cluster(frozenset({"A"}))
        leaf_b = Cluster(frozenset({"B"}))
        leaf_c = Cluster(frozenset({"C"}))
        ab = Cluster(frozenset({"A", "B"}), left=leaf_a, right=leaf_b, similarity=0.8)
        abc = Cluster(frozenset({"A", "B", "C"}), left=ab, right=leaf_c, similarity=0.4)
        # Split at 0.5 → cuts the 0.4 link, keeps 0.8
        result = abc.split(0.5)
        assert len(result) == 2
        assert frozenset({"A", "B"}) in result
        assert frozenset({"C"}) in result

    def test_split_preserves_entities(self):
        """Splitting never loses entities."""
        leaf_a = Cluster(frozenset({"A"}))
        leaf_b = Cluster(frozenset({"B"}))
        leaf_c = Cluster(frozenset({"C"}))
        ab = Cluster(frozenset({"A", "B"}), left=leaf_a, right=leaf_b, similarity=0.8)
        abc = Cluster(frozenset({"A", "B", "C"}), left=ab, right=leaf_c, similarity=0.4)
        for threshold in [0.0, 0.3, 0.5, 0.9, 1.0]:
            result = abc.split(threshold)
            recovered = frozenset().union(*result)
            assert recovered == abc.entities

    def test_split_into_n(self):
        """split_into_n returns exactly n clusters."""
        # Build a tree with 4 leaves
        leaves = [Cluster(frozenset({chr(65 + i)})) for i in range(4)]  # A,B,C,D
        ab = Cluster(
            frozenset({"A", "B"}), left=leaves[0], right=leaves[1], similarity=0.9
        )
        cd = Cluster(
            frozenset({"C", "D"}), left=leaves[2], right=leaves[3], similarity=0.8
        )
        root = Cluster(
            frozenset({"A", "B", "C", "D"}), left=ab, right=cd, similarity=0.5
        )
        # Split into various n
        assert len(root.split_into_n(1)) == 1
        assert len(root.split_into_n(2)) == 2
        assert len(root.split_into_n(3)) == 3
        assert len(root.split_into_n(4)) == 4
        # All preserve entities
        for n in [1, 2, 3, 4]:
            result = root.split_into_n(n)
            recovered = frozenset().union(*result)
            assert recovered == root.entities


class TestAgglomerativeClustering:
    """Test full clustering algorithm."""

    def test_empty_input(self):
        """Empty input returns empty list."""
        clusters, trees = cluster_entities({})
        assert clusters == []
        assert trees == []

    def test_single_entity(self):
        """Single entity returns one cluster."""
        entities = make_entities({"A": ["alpha"]})
        clusters, trees = cluster_entities(entities)
        assert len(clusters) == 1
        assert len(trees) == 1
        assert frozenset({"A"}) in clusters

    def test_no_overlap_separate_clusters(self):
        """Entities with no token overlap stay separate."""
        entities = make_entities(
            {
                "Alpha": ["completely different"],
                "Beta": ["nothing shared"],
                "Gamma": ["unique tokens"],
            }
        )
        clusters, trees = cluster_entities(entities, threshold=0.5)
        # Each in own cluster
        assert len(clusters) == 3

    def test_high_overlap_merges(self):
        """Entities with high token overlap cluster together."""
        entities = make_entities(
            {
                "Pulmonary hypertension": ["Pulmonary hypertension"],
                "Pulmonary arterial hypertension": ["Pulmonary arterial hypertension"],
            }
        )
        clusters, trees = cluster_entities(entities, threshold=0.5)
        assert len(clusters) == 1
        assert "Pulmonary hypertension" in list(clusters[0])
        assert "Pulmonary arterial hypertension" in list(clusters[0])

    def test_average_linkage_prevents_chaining(self):
        """Average linkage prevents single-linkage chaining artifacts."""
        # A and B share "alpha", B and C share "beta", but A and C share nothing
        # Single-linkage would chain A→B→C
        # Average-linkage should not if threshold is high enough
        entities = make_entities(
            {
                "A": ["alpha unique_a"],  # 2 tokens
                "B": ["alpha beta"],  # 2 tokens, shares 1 with A, 1 with C
                "C": ["beta unique_c"],  # 2 tokens
            }
        )
        # With threshold 0.6, A-B share 0.5 (1/2), B-C share 0.5 (1/2)
        # These won't merge initially. But if they did, A-C average would be 0.
        clusters, _ = cluster_entities(entities, threshold=0.6)
        # Should stay separate
        assert len(clusters) == 3

    def test_tree_structure_valid(self):
        """Tree has valid structure (entities = left.entities ∪ right.entities)."""
        entities = make_entities(
            {
                "A": ["alpha beta"],
                "B": ["alpha gamma"],
                "C": ["alpha delta"],
            }
        )
        _, trees = cluster_entities(entities, threshold=0.3)

        def check_tree(node):
            if node.left is None:
                assert len(node.entities) == 1
            else:
                assert node.entities == node.left.entities | node.right.entities
                check_tree(node.left)
                check_tree(node.right)

        for tree in trees:
            check_tree(tree)

    def test_merge_tokens_recorded(self):
        """Merge tokens are recorded in tree nodes."""
        entities = make_entities(
            {
                "A": ["sharedterm alpha"],
                "B": ["sharedterm beta"],
            }
        )
        _, trees = cluster_entities(entities, threshold=0.3)
        # Should have one tree with merge
        assert len(trees) == 1
        root = trees[0]
        assert not root.is_leaf()
        # Merge tokens should include "sharedterm"
        merge_token_names = [t[0] for t in root.merge_tokens]
        assert "sharedterm" in merge_token_names

    def test_deterministic_ordering(self):
        """Clustering produces same results on repeated runs."""
        entities = make_entities(
            {
                "A": ["token1 token2"],
                "B": ["token1 token3"],
                "C": ["token2 token3"],
            }
        )
        results = [cluster_entities(entities, threshold=0.3) for _ in range(5)]
        # All should produce same clusters
        first_clusters = results[0][0]
        for clusters, _ in results[1:]:
            assert clusters == first_clusters


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_pah_variants(self):
        """PAH and Pulmonary arterial hypertension should cluster."""
        entities = make_entities(
            {
                "Pulmonary arterial hypertension": [
                    "Pulmonary arterial hypertension",
                    ("PAH", SPEC_PRIMARY_CONTENT),
                ],
                "PAH mutations": ["PAH mutations"],
            }
        )
        # Threshold lowered from 0.3 to 0.25 because the new tokenizer adds
        # collapsed forms which dilute per-token similarity slightly
        clusters, _ = cluster_entities(entities, threshold=0.25)
        # Should cluster via shared "pah" token
        assert len(clusters) == 1

    def test_common_terms_dont_dominate(self):
        """Common terms like 'signaling' shouldn't cause unrelated clustering."""
        # "signaling" will be common (low specificity)
        # IL-6 and TNF should NOT cluster just because both have "signaling"
        entities = make_entities(
            {
                "IL-6 signaling": ["IL-6 signaling"],
                "TNF signaling": ["TNF signaling"],
                "ER signaling": ["ER signaling"],
                "WNT signaling": ["WNT signaling"],
            }
        )
        # With 4 entities all sharing "signaling", it has low specificity
        clusters, _ = cluster_entities(entities, threshold=0.5)
        # With high threshold, they should stay separate
        # (sharing only low-specificity "signaling" isn't enough)
        assert len(clusters) >= 2  # At minimum, not all in one cluster

    def test_mention_count_impact(self):
        """Heavily-mentioned entities' tokens get lower specificity."""
        entities = make_entities(
            {
                "Common disease": ["hypertension"],
                "Rare variant": ["hypertension subtype"],
            }
        )
        mention_counts = {"Common disease": 200, "Rare variant": 5}
        spec = compute_token_specificity(entities, mention_counts)
        # "hypertension" appears in both, heavily weighted by Common disease
        # "subtype" only in Rare variant with 5 mentions
        assert spec["hypertension"] < spec["subtype"]

    def test_numbered_entities_have_distinct_tokens(self):
        """Numbers in entity names produce distinct tokens (not filtered)."""
        tokens1 = tokenize("Type 1")
        tokens2 = tokenize("Type 2")
        assert "1" in tokens1
        assert "2" in tokens2
        assert tokens1 != tokens2
