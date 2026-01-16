"""Hierarchical entity clustering with specificity-weighted token matching.

Implements agglomerative clustering that:
1. Weights tokens by mention frequency and speculation confidence (IDF-like)
2. Uses average-linkage to avoid single-linkage chaining
3. Builds merge trees enabling surgical splits without re-clustering
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from heapq import heappush, heappop
import itertools
import math
import re

from interaction_finder.extraction.entity_matching import (
    SpeculatedVariant,
    _STOPWORDS,
)
from interaction_finder.extraction.utils import normalize_for_comparison


@dataclass
class Cluster:
    """Node in hierarchical merge tree."""

    entities: frozenset[str]
    left: Cluster | None = None
    right: Cluster | None = None
    similarity: float = 1.0
    merge_tokens: tuple[tuple[str, float], ...] = ()

    def is_leaf(self) -> bool:
        return self.left is None

    def to_dict(self) -> dict:
        """Serialize cluster tree to dict for JSON output."""
        result = {
            "entities": sorted(list(self.entities)),
            "similarity": self.similarity,
            "is_leaf": self.is_leaf(),
        }
        if not self.is_leaf():
            result["merge_tokens"] = [
                {"token": token, "weight": weight}
                for token, weight in self.merge_tokens
            ]
            result["left"] = self.left.to_dict()
            result["right"] = self.right.to_dict()
        return result

    def split(self, threshold: float) -> list[frozenset[str]]:
        """Recursively split at all links below threshold."""
        if self.left is None:
            return [self.entities]
        if self.similarity < threshold:
            return self.left.split(threshold) + self.right.split(threshold)
        return [self.entities]

    def split_into_n(self, n: int) -> list[frozenset[str]]:
        """Split tree to create exactly n clusters by cutting n-1 weakest links."""
        if n <= 1:
            return [self.entities]
        if n >= len(self.entities):
            return [frozenset({e}) for e in self.entities]
        # Collect internal nodes
        internal: list[Cluster] = []

        def collect(node: Cluster) -> None:
            if node.left is not None:
                internal.append(node)
                collect(node.left)
                collect(node.right)

        collect(self)
        internal.sort(key=lambda c: c.similarity)
        # Mark n-1 weakest for cutting
        to_cut = {id(node) for node in internal[: n - 1]}

        def split_at_marked(node: Cluster) -> list[frozenset[str]]:
            if node.left is None:
                return [node.entities]
            if id(node) in to_cut:
                return split_at_marked(node.left) + split_at_marked(node.right)
            return [node.entities]

        return split_at_marked(self)

    def find_subtree(self, target: frozenset[str]) -> Cluster | None:
        """Find the subtree whose entities exactly match target."""
        if self.entities == target:
            return self
        if self.left is None:
            return None
        left = self.left.find_subtree(target)
        return left if left else self.right.find_subtree(target)


def tokenize(text: str) -> frozenset[str]:
    """Extract normalized tokens, excluding stopwords.

    Single characters are preserved since they carry meaning in biology:
    "X-linked", "T cells", "B cells", "Type 1 vs Type 2", etc.
    """
    return frozenset(
        t
        for word in re.split(r"[\s\-/]+", text)
        if (t := normalize_for_comparison(word)) and t not in _STOPWORDS
    )


def compute_token_specificity(
    entities: dict[str, list[SpeculatedVariant]],
    mention_counts: dict[str, int] | None = None,
) -> dict[str, float]:
    """Compute IDF-like specificity scores weighted by mentions and speculation.

    For each token, accumulates: mentions × 1/(speculation + 1) × multiplier
    where multiplier is 0.5 for canonical name tokens, 1.0 for variant tokens.
    Then converts to specificity: log(total_weight / token_weight)

    Higher score = more specific (rare), lower score = more common.

    Canonical name tokens receive 0.5x weight, giving them higher specificity
    and making them MORE influential in similarity calculations. This ensures
    entities cluster strongly on shared canonical terms (e.g., PAH variants
    cluster on "pulmonary arterial hypertension").

    Args:
        entities: Dict of canonical_name → list of SpeculatedVariant
        mention_counts: Optional dict of canonical_name → mention count.
                       If not provided, each entity counts as 1 mention.

    Returns:
        Dict mapping token → specificity score (log-scaled)
    """
    weights: dict[str, float] = defaultdict(float)
    for name, variants in entities.items():
        mentions = mention_counts.get(name, 1) if mention_counts else 1
        if mentions <= 0:
            continue
        canonical_tokens = tokenize(name)
        # Track max weight per token within this entity
        entity_tokens: dict[str, float] = {}
        for var in variants:
            base_weight = mentions / (var.speculation + 1)
            # 0.5x weight for canonical name tokens, 1x for variant tokens
            # This makes canonical tokens MORE influential (higher specificity)
            is_canonical = normalize_for_comparison(
                var.form
            ) == normalize_for_comparison(name)
            multiplier = 0.5 if is_canonical else 1.0
            weight = base_weight * multiplier
            for token in tokenize(var.form):
                entity_tokens[token] = max(entity_tokens.get(token, 0.0), weight)
        for token, w in entity_tokens.items():
            weights[token] += w
    # Convert to IDF-style specificity
    total = sum(weights.values()) + 1.0
    return {token: math.log(total / (w + 1.0)) for token, w in weights.items()}


def _get_entity_tokens(variants: list[SpeculatedVariant]) -> frozenset[str]:
    """Get all tokens from entity variant forms (includes canonical)."""
    tokens: set[str] = set()
    for var in variants:
        tokens.update(tokenize(var.form))
    return frozenset(tokens)


def _weighted_similarity(
    tokens1: frozenset[str],
    tokens2: frozenset[str],
    specificity: dict[str, float],
) -> tuple[float, tuple[tuple[str, float], ...]]:
    """Compute similarity and contributing tokens between two token sets."""
    shared = tokens1 & tokens2
    if not shared:
        return 0.0, ()
    shared_weight = sum(specificity.get(t, 1.0) for t in shared)
    weight1 = sum(specificity.get(t, 1.0) for t in tokens1)
    weight2 = sum(specificity.get(t, 1.0) for t in tokens2)
    min_weight = min(weight1, weight2)
    sim = shared_weight / min_weight if min_weight > 0 else 0.0
    # Top contributing tokens, sorted by specificity descending
    contributing = tuple(
        sorted(
            ((t, specificity.get(t, 1.0)) for t in shared),
            key=lambda x: x[1],
            reverse=True,
        )[:5]
    )
    return sim, contributing


def _compute_cluster_similarity(
    c1: Cluster,
    c2: Cluster,
    get_pair_similarity: callable,
) -> tuple[float, tuple[tuple[str, float], ...]]:
    """Compute average-linkage similarity between two clusters."""
    sims = []
    all_tokens: dict[str, float] = {}
    for e1 in c1.entities:
        for e2 in c2.entities:
            s, toks = get_pair_similarity(e1, e2)
            sims.append(s)
            for token, spec in toks:
                all_tokens[token] = max(all_tokens.get(token, 0.0), spec)
    avg_sim = sum(sims) / len(sims) if sims else 0.0
    agg_tokens = tuple(sorted(all_tokens.items(), key=lambda x: x[1], reverse=True)[:5])
    return avg_sim, agg_tokens


def agglomerative_cluster(
    entities: dict[str, list[SpeculatedVariant]],
    specificity: dict[str, float],
    threshold: float = 0.5,
) -> list[Cluster]:
    """Build hierarchical merge trees using Reliable Agglomerative Clustering (RAC).

    Uses mutual nearest-neighbour constraint: only merges clusters that are each
    other's best available partner. This improves stability by filtering out
    fragile merges that depend on slight ordering differences.

    Args:
        entities: Dict of canonical_name → list of SpeculatedVariant
        specificity: Token → specificity score from compute_token_specificity
        threshold: Minimum similarity to merge clusters

    Returns:
        List of Cluster trees (forest if threshold prevents full merge)
    """
    if not entities:
        return []
    # Pre-compute tokens for all entities
    entity_tokens = {
        name: _get_entity_tokens(variants) for name, variants in entities.items()
    }
    # Cache pairwise similarities
    sim_cache: dict[tuple[str, str], tuple[float, tuple[tuple[str, float], ...]]] = {}

    def get_pair_similarity(
        e1: str,
        e2: str,
    ) -> tuple[float, tuple[tuple[str, float], ...]]:
        key = (min(e1, e2), max(e1, e2))
        if key not in sim_cache:
            sim_cache[key] = _weighted_similarity(
                entity_tokens[e1], entity_tokens[e2], specificity
            )
        return sim_cache[key]

    # Initialize singleton clusters
    initial_clusters = [Cluster(frozenset({name})) for name in entities]
    clusters: dict[int, Cluster] = {id(c): c for c in initial_clusters}
    entity_to_cluster_id = {
        name: id(c) for c in initial_clusters for name in c.entities
    }
    # Compute all pairwise cluster similarities above threshold
    # similarity_map: cluster_id → {other_id → (similarity, tokens)}
    similarity_map: dict[int, dict[int, tuple[float, tuple]]] = defaultdict(dict)
    entity_list = list(entities.keys())
    for i, e1 in enumerate(entity_list):
        for e2 in entity_list[i + 1 :]:
            sim, tokens = get_pair_similarity(e1, e2)
            if sim >= threshold:
                id1, id2 = entity_to_cluster_id[e1], entity_to_cluster_id[e2]
                similarity_map[id1][id2] = (sim, tokens)
                similarity_map[id2][id1] = (sim, tokens)
    # RAC: iteratively merge mutual nearest neighbours
    made_progress = True
    while made_progress:
        made_progress = False
        # Find best neighbour for each cluster
        best_neighbour: dict[int, int] = {}
        for cid in clusters:
            if cid not in similarity_map or not similarity_map[cid]:
                continue
            # Find highest similarity neighbour
            candidates = [
                (sim, other_id)
                for other_id, (sim, _) in similarity_map[cid].items()
                if other_id in clusters
            ]
            if candidates:
                best_sim = max(c[0] for c in candidates)
                best_neighbour[cid] = max(
                    (oid for s, oid in candidates if s == best_sim)
                )
        # Find mutual pairs
        mutual_pairs: list[tuple[int, int]] = []
        seen: set[int] = set()
        for id1, id2 in best_neighbour.items():
            if id1 in seen or id2 in seen:
                continue
            if best_neighbour.get(id2) == id1:
                mutual_pairs.append((id1, id2))
                seen.add(id1)
                seen.add(id2)
        # Merge mutual pairs
        for id1, id2 in mutual_pairs:
            c1, c2 = clusters[id1], clusters[id2]
            sim, tokens = similarity_map[id1][id2]
            # Create merged cluster
            merged = Cluster(
                entities=c1.entities | c2.entities,
                left=c1,
                right=c2,
                similarity=sim,
                merge_tokens=tokens,
            )
            merged_id = id(merged)
            # Update clusters
            del clusters[id1]
            del clusters[id2]
            clusters[merged_id] = merged
            # Update similarity map
            del similarity_map[id1]
            del similarity_map[id2]
            for other_id in list(similarity_map.keys()):
                similarity_map[other_id].pop(id1, None)
                similarity_map[other_id].pop(id2, None)
            # Compute new similarities
            for other_id, other in clusters.items():
                if other_id == merged_id:
                    continue
                avg_sim, agg_tokens = _compute_cluster_similarity(
                    merged, other, get_pair_similarity
                )
                if avg_sim >= threshold:
                    similarity_map[merged_id][other_id] = (avg_sim, agg_tokens)
                    similarity_map[other_id][merged_id] = (avg_sim, agg_tokens)
            made_progress = True
    return list(clusters.values())


def cluster_entities(
    entities: dict[str, list[SpeculatedVariant]],
    threshold: float = 0.5,
    mention_counts: dict[str, int] | None = None,
) -> tuple[list[frozenset[str]], list[Cluster]]:
    """Cluster entities and return both flat clusters and merge trees.

    Args:
        entities: Dict of canonical_name → list of SpeculatedVariant
        threshold: Minimum similarity for clustering
        mention_counts: Optional dict of entity_name → mention count for IDF weighting

    Returns:
        (clusters, trees) where:
        - clusters: List of entity sets at the given threshold
        - trees: Merge trees enabling surgical splitting
    """
    if not entities:
        return [], []
    specificity = compute_token_specificity(entities, mention_counts)
    trees = agglomerative_cluster(entities, specificity, threshold)
    flat_clusters = [tree.entities for tree in trees]
    return flat_clusters, trees
