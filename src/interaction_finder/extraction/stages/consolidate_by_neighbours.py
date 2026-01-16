"""Consolidate entities by analyzing neighbour fragmentation patterns.

When an anchor entity connects to multiple similar-sounding neighbours,
those neighbours may represent the same concept with different names.
This stage clusters neighbours by textual similarity and presents
clusters to an LLM for potential merging.

Algorithm:
1. Build neighbour sets from pair_assessments_by_resource
2. Cluster each anchor's neighbours using token-based similarity
3. Present multi-member clusters to LLM with anchor context
4. Apply merge rules to update pair_assessments
"""

import logging
import re
import secrets
import string
from collections import defaultdict
from dataclasses import dataclass

from interaction_finder.agent_config import AGENT_CALL_ERRORS, agent_getter
from interaction_finder.usage import record_usage
from interaction_finder.extraction.clustering import Cluster, cluster_entities
from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.entity_matching import (
    extract_entity_variants,
    find_entity_match,
    SpeculatedVariant,
)
from interaction_finder.extraction.models import (
    ClusterDecision,
    ClusterDecisions,
    EntityKindMerges,
    EntityMention,
    EntityMergeRule,
    EntityRef,
)
from interaction_finder.extraction.shared import save_checkpoint
from interaction_finder.extraction.state import State
from interaction_finder.extraction.utils import normalize_for_comparison
from interaction_finder.logging import logfire

_TOKEN_CHARS = string.ascii_letters + string.digits


def _generate_token(length: int = 4) -> str:
    """Generate a random alphanumeric token for group verification."""
    return "".join(secrets.choice(_TOKEN_CHARS) for _ in range(length))


@dataclass
class NeighbourClusterCandidate:
    """A cluster of similar neighbours sharing a common anchor."""

    anchor: str
    anchor_kind: str
    cluster: frozenset[str]
    merge_tree: Cluster
    similarity: float
    cluster_kind: (
        str  # Kind of entities in this cluster (for kind-specific merge rules)
    )


async def consolidate_by_neighbours(state: State, deps: Deps) -> bool:
    """Consolidate entities that appear as fragmented neighbours of common anchors.

    For each entity with multiple neighbours in pair_assessments, clusters
    the neighbours by textual similarity. Multi-member clusters are presented
    to an LLM for potential merging.
    """
    with logfire.span("consolidate_by_neighbours"):
        if deps.progress:
            deps.progress.set_status("Analyzing neighbour patterns")
        # Count entities by kind before consolidation
        counts_before = _count_entities_by_kind(state)
        # Build neighbour sets from pair assessments
        neighbour_sets = _build_neighbour_sets(state)
        if not neighbour_sets:
            deps.logger.info("No neighbour sets to analyze")
            return True
        # Collect all entities for variant extraction
        all_entities = _collect_entities_for_clustering(state)
        # Find clusters in neighbour sets
        threshold = deps.config.stage.extraction.neighbour_cluster_similarity_threshold
        candidates = _find_neighbour_clusters(
            neighbour_sets, all_entities, threshold, state
        )
        if not candidates:
            deps.logger.info("No neighbour clusters found for consolidation")
            return True
        # Deduplicate clusters across anchors
        unique_candidates = _deduplicate_clusters(candidates)
        # Count unique entities by kind across all clusters
        entities_by_kind: dict[str, set[str]] = defaultdict(set)
        for c in unique_candidates:
            entities_by_kind[c.cluster_kind].update(c.cluster)
        entity_counts = {k: len(v) for k, v in entities_by_kind.items()}
        coverage = ", ".join(f"{n} {k}" for k, n in sorted(entity_counts.items()))
        deps.logger.info(
            f"Found {len(unique_candidates)} unique neighbour clusters "
            f"(from {len(candidates)} across {len(neighbour_sets)} anchors; "
            f"covering {coverage})",
            extra={"entities_by_kind": entity_counts},
        )
        # Present clusters to LLM for review
        all_rules = await _review_neighbour_clusters(
            unique_candidates, all_entities, state, deps
        )
        # Apply merge rules
        if all_rules:
            resolved_rules = _resolve_transitive_merges(all_rules)
            _apply_merge_rules(resolved_rules, state)
            _update_pair_entity_references(resolved_rules, state)
            _rebuild_global_entity_index(state)
            # Count merges by kind
            merges_by_kind: dict[str, int] = defaultdict(int)
            for (_norm, kind), _ in resolved_rules.items():
                merges_by_kind[kind] += 1
            deps.logger.info(
                f"Neighbour consolidation: {len(resolved_rules)} merge rules applied",
                extra={"merges_by_kind": dict(merges_by_kind)},
            )
        else:
            deps.logger.info("Neighbour consolidation: no merges needed")
        # Log entity count changes
        counts_after = _count_entities_by_kind(state)
        _log_entity_count_changes(counts_before, counts_after, deps)
        # Save checkpoint
        await save_checkpoint(state, deps, "consolidate_by_neighbours")
        return True


def _build_neighbour_sets(state: State) -> dict[str, set[str]]:
    """Build mapping: anchor_canonical -> {neighbour canonicals}.

    Collects all neighbours regardless of relationship type/polarity.
    """
    neighbour_sets: dict[str, set[str]] = defaultdict(set)
    for assessments in state.pair_assessments_by_resource.values():
        for assessment in assessments:
            e1 = assessment.entity1.canonical
            e2 = assessment.entity2.canonical
            # Add both directions
            neighbour_sets[e1].add(e2)
            neighbour_sets[e2].add(e1)
    return dict(neighbour_sets)


def _collect_entities_for_clustering(
    state: State,
) -> dict[str, list[SpeculatedVariant]]:
    """Collect all entities with their variants for clustering."""
    entities: dict[str, list[SpeculatedVariant]] = {}
    for resource_entities in state.validated_entities_by_resource.values():
        for name, ref in resource_entities.items():
            if name not in entities:
                entities[name] = extract_entity_variants(name, ref.aliases())
    return entities


def _find_neighbour_clusters(
    neighbour_sets: dict[str, set[str]],
    all_entities: dict[str, list[SpeculatedVariant]],
    threshold: float,
    state: State,
) -> list[NeighbourClusterCandidate]:
    """Find clusters of similar neighbours for each anchor.

    Clusters neighbours per-kind to ensure merge rules are kind-specific.
    """
    # Build kind lookup once for all entities
    entity_kinds = _build_entity_kind_lookup(state)
    candidates: list[NeighbourClusterCandidate] = []
    for anchor, neighbours in neighbour_sets.items():
        if len(neighbours) < 2:
            continue
        # Group neighbours by kind
        neighbours_by_kind: dict[str, set[str]] = defaultdict(set)
        for n in neighbours:
            if n in all_entities:
                kind = entity_kinds.get(n, "unknown")
                if kind != "unknown":
                    neighbours_by_kind[kind].add(n)
        # Look up anchor kind
        anchor_kind = entity_kinds.get(anchor, "unknown")
        # Cluster each kind separately
        for kind, kind_neighbours in neighbours_by_kind.items():
            if len(kind_neighbours) < 2:
                continue
            # Build variant dict for this kind's neighbours
            neighbour_entities = {n: all_entities[n] for n in kind_neighbours}
            # Cluster
            clusters, trees = cluster_entities(neighbour_entities, threshold=threshold)
            # Collect multi-member clusters
            for cluster_set, tree in zip(clusters, trees):
                if len(cluster_set) >= 2:
                    candidates.append(
                        NeighbourClusterCandidate(
                            anchor=anchor,
                            anchor_kind=anchor_kind,
                            cluster=cluster_set,
                            merge_tree=tree,
                            similarity=tree.similarity,
                            cluster_kind=kind,
                        )
                    )
    return candidates


def _build_entity_kind_lookup(state: State) -> dict[str, str]:
    """Build a lookup from entity name to kind for all validated entities."""
    kinds: dict[str, str] = {}
    for resource_entities in state.validated_entities_by_resource.values():
        for name, ref in resource_entities.items():
            if name not in kinds:
                kinds[name] = ref.kind
    return kinds


def _count_entities_by_kind(state: State) -> dict[str, int]:
    """Count unique entities by kind across all resources."""
    entities_by_kind: dict[str, set[str]] = defaultdict(set)
    for resource_entities in state.validated_entities_by_resource.values():
        for name, ref in resource_entities.items():
            entities_by_kind[ref.kind].add(name)
    return {kind: len(names) for kind, names in entities_by_kind.items()}


def _log_entity_count_changes(
    before: dict[str, int],
    after: dict[str, int],
    deps: Deps,
) -> None:
    """Log changes in entity counts by kind."""
    all_kinds = set(before.keys()) | set(after.keys())
    changes = []
    for kind in sorted(all_kinds):
        b, a = before.get(kind, 0), after.get(kind, 0)
        if b != a:
            changes.append(f"{kind}: {b} → {a}")
    if changes:
        deps.logger.info(
            f"Entity counts changed: {', '.join(changes)}",
            extra={"before": before, "after": after},
        )
    else:
        deps.logger.info("Entity counts unchanged")


def _deduplicate_clusters(
    candidates: list[NeighbourClusterCandidate],
) -> list[NeighbourClusterCandidate]:
    """Deduplicate clusters that appear for multiple anchors.

    Same cluster may appear when clustering neighbours of different anchors.
    Keep one representative (the one with highest similarity).
    Key includes kind to avoid conflating clusters across kinds.
    """
    seen_clusters: dict[tuple[frozenset[str], str], NeighbourClusterCandidate] = {}
    for candidate in candidates:
        key = (candidate.cluster, candidate.cluster_kind)
        existing = seen_clusters.get(key)
        if existing is None or candidate.similarity > existing.similarity:
            seen_clusters[key] = candidate
    return list(seen_clusters.values())


async def _review_neighbour_clusters(
    candidates: list[NeighbourClusterCandidate],
    all_entities: dict[str, list[SpeculatedVariant]],
    state: State,
    deps: Deps,
) -> dict[tuple[str, str], tuple[str, str, str | None]]:
    """Present neighbour clusters to LLM for consolidation decisions."""
    if not candidates:
        return {}
    get_agent = agent_getter(
        "extraction",
        "neighbour_consolidation",
        ClusterDecisions,
        Deps,
        """You are an expert at consolidating biomedical entity names for systematic literature analysis.

You will be given groups of entities that all connect to the same anchor entity.
For each group, decide if all members should unify to a single canonical name, based on the research topic.

**Decision framework:**
1. Evaluate each group in the context of the research topic
2. Ask: Are these the same entity, or subtypes that the research topic wouldn't distinguish between? If so, merge. If the distinction matters for this research, reject.
3. Apply these principles:
   - Different phrasings of the same concept → merge
   - Variant spellings or abbreviations → merge
   - Subtypes of a concept the topic targets → merge to that concept
   - Entities whose distinction matters for the research → reject
   - One member doesn't belong → exclude it

**Actions:**
- **merge**: Members should unify → specify target (member number, member name, or a new parent concept)
- **reject**: Distinction matters for this research → keep them all separate
- **exclude**: One specific member doesn't belong → specify which one to remove
- **split**: Cluster mixes unrelated entities → system splits at weakest link

You must return an explicit decision for every group.""",
    )
    agent = get_agent(deps.config)
    batch_size = deps.config.stage.extraction.merge_batch_size
    max_rounds = deps.config.stage.extraction.cluster_refinement_max_rounds
    all_rules: dict[tuple[str, str], tuple[str, str, str | None]] = {}
    # Key clusters by (frozenset, kind) to track kind through splits
    ClusterKey = tuple[frozenset[str], str]  # (entities, kind)
    # Build maps from cluster key → metadata
    cluster_to_tree: dict[ClusterKey, Cluster] = {
        (c.cluster, c.cluster_kind): c.merge_tree for c in candidates
    }
    cluster_to_anchor: dict[ClusterKey, tuple[str, str]] = {
        (c.cluster, c.cluster_kind): (c.anchor, c.anchor_kind) for c in candidates
    }
    current_groups: list[ClusterKey] = [(c.cluster, c.cluster_kind) for c in candidates]
    for round_num in range(1, max_rounds + 1):
        if not current_groups:
            break
        num_batches = -(-len(current_groups) // batch_size)
        batch_suffix = f" in {num_batches} batches" if num_batches > 1 else ""
        deps.logger.info(
            f"Neighbour consolidation round {round_num}/{max_rounds}: "
            f"{len(current_groups)} clusters{batch_suffix}"
        )
        groups_needing_review: list[ClusterKey] = []
        stats = {"merged": 0, "rejected": 0, "split": 0, "excluded": 0, "undecided": 0}
        for batch_start in range(0, len(current_groups), batch_size):
            batch = current_groups[batch_start : batch_start + batch_size]
            # Build prompt
            group_data = []
            for cluster_key in batch:
                cluster, cluster_kind = cluster_key
                anchor, anchor_kind = cluster_to_anchor.get(
                    cluster_key, ("unknown", "unknown")
                )
                tree = cluster_to_tree.get(cluster_key)
                group_data.append(
                    {
                        "id": _generate_token(),
                        "cluster_key": cluster_key,
                        "cluster": cluster,
                        "cluster_kind": cluster_kind,
                        "members": sorted(cluster, key=lambda e: (len(e), e)),
                        "anchor": anchor,
                        "anchor_kind": anchor_kind,
                        "similarity": tree.similarity if tree else 0.0,
                    }
                )
            groups_text = "\n\n".join(
                f"## Group {g['id']}\n"
                f"**Context:** These entities all connect to **{g['anchor']}** ({g['anchor_kind']})\n\n"
                f"Members:\n"
                + "\n".join(f"  {i + 1}. {m}" for i, m in enumerate(g["members"]))
                for g in group_data
            )
            prompt = f"""**Research topic:** {state.topic}

**Neighbour clusters to evaluate:**

{groups_text}
"""
            try:
                async with deps.agent_semaphore:
                    result = await agent.run(prompt, deps=deps)
                record_usage(deps.usage, "neighbour_consolidation", agent, result)
            except AGENT_CALL_ERRORS as e:
                deps.logger.error(
                    f"Neighbour consolidation failed for batch: {type(e).__name__}: {e}"
                )
                stats["kept_separate"] += len(batch)
                continue
            # Index decisions by group_id
            id_to_group = {g["id"]: g for g in group_data}
            decisions_by_id: dict[str, list[ClusterDecision]] = {}
            for d in result.output.decisions:
                gid = d.group_id
                if gid.lower().startswith("group "):
                    gid = gid[6:].lstrip()
                if gid not in id_to_group:
                    deps.logger.warning(f"Unknown group_id '{d.group_id}'")
                    continue
                decisions_by_id.setdefault(gid, []).append(d)
            # Process decisions
            groups_with_decisions: set[ClusterKey] = set()
            for gid, decisions in decisions_by_id.items():
                g = id_to_group[gid]
                groups_with_decisions.add(g["cluster_key"])
                members = g["members"]
                anchor = g["anchor"]
                similarity = g["similarity"]
                cluster_kind = g["cluster_kind"]
                # Categorize decisions
                excludes = [d for d in decisions if d.action == "exclude"]
                merges = [d for d in decisions if d.action == "merge"]
                splits = [d for d in decisions if d.action == "split"]
                rejects = [d for d in decisions if d.action == "reject"]
                if merges and splits:
                    deps.logger.warning(f"Group {gid}: merge+split conflict")
                    continue
                # Apply excludes first
                remaining = set(members)
                for exc in excludes:
                    if exc.target:
                        member = _resolve_group_target(
                            exc.target, members, all_entities, deps.logger
                        )
                        if member in remaining:
                            remaining.discard(member)
                            stats["excluded"] += 1
                if len(remaining) <= 1:
                    continue
                remaining_entities = frozenset(remaining)
                # Apply merge, reject, or split
                if rejects:
                    stats["rejected"] += 1
                elif merges:
                    merge = merges[0]
                    if not merge.target:
                        continue
                    target = _resolve_group_target(
                        merge.target, members, all_entities, deps.logger
                    )
                    if target in members and target not in remaining:
                        deps.logger.warning(
                            f"Group {gid}: merge target '{target}' was excluded"
                        )
                        continue
                    # Build trigger with provenance info
                    trigger = f"neighbour({anchor}):cluster({gid},{similarity:.2f})"
                    for member in remaining:
                        if member != target:
                            all_rules[
                                (normalize_for_comparison(member), cluster_kind)
                            ] = (
                                target,
                                trigger,
                                merge.reasoning,
                            )
                    stats["merged"] += 1
                elif splits:
                    cluster_key = g["cluster_key"]
                    tree = cluster_to_tree.get(cluster_key) or cluster_to_tree.get(
                        (remaining_entities, cluster_kind)
                    )
                    if tree and len(remaining) < len(g["cluster"]):
                        tree = tree.find_subtree(remaining_entities) or tree
                    if not tree:
                        deps.logger.warning(f"No tree for group {gid}")
                        continue
                    for sub in tree.split_into_n(2):
                        if len(sub) > 1:
                            sub_key = (sub, cluster_kind)
                            groups_needing_review.append(sub_key)
                            # Preserve anchor mapping for sub-clusters
                            cluster_to_anchor[sub_key] = (anchor, g["anchor_kind"])
                            if sub_key not in cluster_to_tree and (
                                st := tree.find_subtree(sub)
                            ):
                                cluster_to_tree[sub_key] = st
                    stats["split"] += 1
                elif excludes:
                    # Excludes without merge/reject/split - re-review the remainder
                    remaining_key = (remaining_entities, cluster_kind)
                    groups_needing_review.append(remaining_key)
                    cluster_to_anchor[remaining_key] = (anchor, g["anchor_kind"])
                    cluster_key = g["cluster_key"]
                    if (tree := cluster_to_tree.get(cluster_key)) and (
                        st := tree.find_subtree(remaining_entities)
                    ):
                        cluster_to_tree[remaining_key] = st
            # Groups without decisions need re-review
            undecided = set(batch) - groups_with_decisions
            for cluster_key in undecided:
                groups_needing_review.append(cluster_key)
            stats["undecided"] += len(undecided)
        deps.logger.info(
            f"  Results: {stats['merged']} merged, {stats['rejected']} rejected, "
            f"{stats['split']} split, {stats['excluded']} excluded, "
            f"{stats['undecided']} undecided"
        )
        if not groups_needing_review:
            break
        current_groups = groups_needing_review
    return all_rules


def _resolve_group_target(
    target_spec: str,
    members: list[str],
    entities: dict[str, list[SpeculatedVariant]],
    logger: logging.Logger,
) -> str:
    """Resolve target from group decision."""
    members_set = set(members)
    # Case 1: Pure digit
    if target_spec.isdigit():
        idx = int(target_spec) - 1
        if 0 <= idx < len(members):
            return members[idx]
    # Case 2: Number + name format
    match = re.match(r"^\s*(\d+)[).]?\s+(\w.+?)?\s*$", target_spec)
    if match:
        num_str, name_part = match.groups()
        idx = int(num_str) - 1
        if 0 <= idx < len(members):
            member = members[idx]
            if name_part:
                try:
                    name_match = find_entity_match(name_part, entities)
                except (IndexError, KeyError):
                    name_match = None
                if name_match is None or name_match.canonical != member:
                    resolved = name_match.canonical if name_match else name_part
                    logger.warning(
                        f"Merge target mismatch: member {idx + 1} is '{member}' "
                        f"but '{name_part}' resolves to '{resolved}'; using member number"
                    )
            return member
    # Case 3: Exact member name
    if target_spec in members_set:
        return target_spec
    # Case 4: Fuzzy match
    try:
        entity_match = find_entity_match(target_spec, entities)
    except (IndexError, KeyError):
        entity_match = None
    if entity_match and entity_match.canonical in members_set:
        return entity_match.canonical
    # Case 5: New canonical name
    return target_spec


def _resolve_transitive_merges(
    merge_rules: dict[tuple[str, str], tuple[str, str, str | None]],
) -> dict[tuple[str, str], tuple[str, str, str | None]]:
    """Resolve transitive merge chains (A→B, B→C becomes A→C, B→C)."""
    resolved: dict[tuple[str, str], tuple[str, str, str | None]] = {}
    for (child_norm, kind), (target, trigger, reasoning) in merge_rules.items():
        final_target = target
        final_trigger = trigger
        final_reasoning = reasoning
        visited = {child_norm}
        while True:
            target_norm = normalize_for_comparison(final_target)
            if (target_norm, kind) not in merge_rules:
                break
            if target_norm in visited:
                break
            visited.add(target_norm)
            final_target, final_trigger, final_reasoning = merge_rules[
                (target_norm, kind)
            ]
        resolved[(child_norm, kind)] = (final_target, final_trigger, final_reasoning)
    return resolved


def _apply_merge_rules(
    rules: dict[tuple[str, str], tuple[str, str, str | None]],
    state: State,
) -> None:
    """Apply merge rules to validated_entities_by_resource and track in consolidated."""
    if not rules:
        return
    # Store rules in consolidated structure
    for (norm_name, kind), (target, trigger, reasoning) in rules.items():
        rule = EntityMergeRule(
            source=norm_name, target=target, trigger=trigger, reasoning=reasoning
        )
        kind_merges = state.consolidated.entities.merges.setdefault(
            kind, EntityKindMerges()
        )
        kind_merges.llm_decided.append(rule)
    # Build target lookup
    target_by_norm_and_kind: dict[tuple[str, str], str] = {}
    for (_norm, kind), (target, _, _) in rules.items():
        target_key = (normalize_for_comparison(target), kind)
        target_by_norm_and_kind.setdefault(target_key, target)
    # Apply to each resource
    for resource_id, entities in state.validated_entities_by_resource.items():
        grouped: dict[str, list[EntityMention]] = defaultdict(list)
        for child_name, ref in entities.items():
            # Check if this entity should be merged
            key = (normalize_for_comparison(child_name), ref.kind)
            if key in rules:
                target_name = rules[key][0]
            else:
                # Check if this is a target that should collect mentions
                canonical_target = target_by_norm_and_kind.get(key)
                target_name = canonical_target or child_name
            grouped[target_name].extend(ref.mentions)
        new_entities: dict[str, EntityRef] = {
            canonical: EntityRef(canonical=canonical, mentions=mentions)
            for canonical, mentions in grouped.items()
        }
        merges_for_resource = max(0, len(entities) - len(new_entities))
        state.entities_merged += merges_for_resource
        state.validated_entities_by_resource[resource_id] = new_entities


def _update_pair_entity_references(
    merge_rules: dict[tuple[str, str], tuple[str, str, str | None]],
    state: State,
) -> None:
    """Update EntityRef references in PairAssessments after merging."""
    if not merge_rules:
        return
    for resource_id, assessments in state.pair_assessments_by_resource.items():
        consolidated_entities = state.validated_entities_by_resource.get(
            resource_id, {}
        )
        for assessment in assessments:
            # Update entity1
            e1_key = (
                normalize_for_comparison(assessment.entity1.canonical),
                assessment.entity1.kind,
            )
            if e1_key in merge_rules:
                merged_name = merge_rules[e1_key][0]
                if merged_name in consolidated_entities:
                    assessment.entity1 = consolidated_entities[merged_name]
                else:
                    assessment.entity1 = EntityRef(
                        canonical=merged_name,
                        mentions=assessment.entity1.mentions,
                    )
            # Update entity2
            e2_key = (
                normalize_for_comparison(assessment.entity2.canonical),
                assessment.entity2.kind,
            )
            if e2_key in merge_rules:
                merged_name = merge_rules[e2_key][0]
                if merged_name in consolidated_entities:
                    assessment.entity2 = consolidated_entities[merged_name]
                else:
                    assessment.entity2 = EntityRef(
                        canonical=merged_name,
                        mentions=assessment.entity2.mentions,
                    )


def _rebuild_global_entity_index(state: State) -> None:
    """Rebuild global entity index after merging."""
    mentions_by_canonical: dict[str, list[EntityMention]] = defaultdict(list)
    for entities in state.validated_entities_by_resource.values():
        for canonical, ref in entities.items():
            mentions_by_canonical[canonical].extend(ref.mentions)
    state.global_entities = {
        c: EntityRef(canonical=c, mentions=m) for c, m in mentions_by_canonical.items()
    }
