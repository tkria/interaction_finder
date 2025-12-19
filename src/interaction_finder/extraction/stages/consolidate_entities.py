"""Consolidate entities globally across all documents.

Algorithm:
1. Collect all unique normalized entity forms from all documents
2. Find substring relationships between normalized forms
3. Query LLM for consolidation decisions (skip/merge/rename)
4. Apply merge rules consistently across all documents
5. If renames occurred, re-evaluate for new merge opportunities

This approach ensures:
- Complete coverage (finds all merge opportunities)
- Efficiency (one LLM call per unique normalized pair)
- Consistency (same canonical name across documents)
- Topic-appropriate naming (verbose names can be simplified via rename)
"""

import asyncio
import logging
import re
import secrets
import string
from collections import defaultdict

from interaction_finder.agent_config import AGENT_CALL_ERRORS, agent_getter
from interaction_finder.agent_utils import rename_agent
from interaction_finder.usage import record_usage
from interaction_finder.extraction.consolidate_entities import (
    get_entity_consolidation_agent,
)
from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.entity_matching import (
    extract_entity_variants,
    find_consolidation_candidates,
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
from interaction_finder.extraction.shared import (
    save_checkpoint,
    snapshot_entity_counts,
)
from interaction_finder.extraction.state import MergeCacheForKind, State
from interaction_finder.extraction.utils import (
    extract_all_forms,
    is_obvious_variant,
    normalize_for_comparison,
    osa_distance,
)
from interaction_finder.logging import logfire

# Characters for generating verification tokens
_TOKEN_CHARS = string.ascii_letters + string.digits


def _generate_token(length: int = 4) -> str:
    """Generate a random alphanumeric token for pair verification."""
    return "".join(secrets.choice(_TOKEN_CHARS) for _ in range(length))


def _get_merge_cache(state: State, kind: str) -> MergeCacheForKind:
    """Get or create the merge cache for an entity kind."""
    if kind not in state.merge_cache_by_kind:
        state.merge_cache_by_kind[kind] = MergeCacheForKind()
    return state.merge_cache_by_kind[kind]


def _resolve_pair_from_decision(
    decision_id: int,
    decision_token: str,
    id_to_pair_info: dict[int, tuple[str, str, str]],
    token_to_id: dict[str, int],
    logger,
) -> tuple[str, str] | None:
    """Resolve canonical pair from LLM decision, using token as verification/fallback.

    Returns (parent, child) or None if unresolvable.
    """
    pair_info = id_to_pair_info.get(decision_id)
    token_id = token_to_id.get(decision_token)
    # Both ID and token invalid
    if pair_info is None and token_id is None:
        logger.warning(
            f"LLM decision unresolvable: pair_id={decision_id}, "
            f"token='{decision_token}' - neither found"
        )
        return None
    # ID invalid but token valid - use token
    if pair_info is None:
        logger.warning(
            f"LLM decision ID mismatch: pair_id={decision_id} not found, "
            f"but token '{decision_token}' maps to id={token_id}. Using token."
        )
        info = id_to_pair_info[token_id]
        return (info[0], info[1])
    # ID valid - check token
    parent, child, expected_token = pair_info
    if decision_token == expected_token:
        return (parent, child)
    # Token mismatch - prefer token if it points to a different valid pair
    if token_id is not None and token_id != decision_id:
        logger.warning(
            f"LLM decision conflict: pair_id={decision_id} (token '{expected_token}') "
            f"but got token '{decision_token}' (id={token_id}). Using token."
        )
        info = id_to_pair_info[token_id]
        return (info[0], info[1])
    # Token invalid/same-id but ID valid - trust ID
    logger.warning(
        f"LLM decision token mismatch: pair_id={decision_id} expected "
        f"'{expected_token}' but got '{decision_token}'. Using ID."
    )
    return (parent, child)


async def consolidate_entities(state: State, deps: Deps) -> bool:
    """Consolidate entities globally and update pair references.

    Processes each entity kind concurrently, then applies all merge rules
    together with transitive resolution.
    """
    with logfire.span("consolidate_entities"):
        if deps.progress:
            deps.progress.set_status("Consolidating entities")
        # Capture initial entity state BEFORE any consolidation (once only)
        if not state.consolidated.entities.initial:
            state.consolidated.entities.initial = snapshot_entity_counts(state)
        # Collect entities once
        entities_by_kind, mentions_by_kind = _collect_entity_variants(state)
        # Pre-initialize shared structures before concurrent execution to avoid races
        for kind in entities_by_kind:
            state.consolidated.entities.merges.setdefault(kind, EntityKindMerges())
            _get_merge_cache(state, kind)
        # Process each kind concurrently
        tasks = [
            _consolidate_kind(
                kind, entities, mentions_by_kind.get(kind, {}), state, deps
            )
            for kind, entities in entities_by_kind.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Collect rules, checking for errors
        all_rules: dict[tuple[str, str], tuple[str, str]] = {}
        for kind, result in zip(entities_by_kind.keys(), results):
            if isinstance(result, Exception):
                deps.logger.error(f"{kind} consolidation failed: {result}")
            else:
                all_rules.update(result)
        # Apply all rules with transitive resolution
        if all_rules:
            resolved_rules = _resolve_transitive_merges(all_rules)
            _apply_merge_rules_globally(resolved_rules, state)
            _update_pair_entity_references(resolved_rules, state)
        deps.logger.info(
            f"Entity consolidation: {len(all_rules)} rules applied, "
            f"{state.entities_merged} entities merged",
        )
        # Build global entity index (aggregate mentions across all resources)
        _build_global_entity_index(state)
        # Save checkpoint after entity consolidation
        await save_checkpoint(state, deps, "consolidate_entities")
        return True


def _collect_entity_variants(
    state: State,
) -> tuple[dict[str, dict[str, list[SpeculatedVariant]]], dict[str, dict[str, int]]]:
    """Collect all entities with their speculated variants and mention counts."""
    entities_by_kind: dict[str, dict[str, list[SpeculatedVariant]]] = {}
    mentions_by_kind: dict[str, dict[str, int]] = {}
    for entities in state.validated_entities_by_resource.values():
        for entity_name, entity in entities.items():
            entity_kind = entity.kind
            if entity_kind not in entities_by_kind:
                entities_by_kind[entity_kind] = {}
                mentions_by_kind[entity_kind] = {}
            # Extract variants with speculation tracking
            variants = extract_entity_variants(entity_name, entity.aliases())
            entities_by_kind[entity_kind][entity_name] = variants
            # Count mentions (accumulate across resources)
            current = mentions_by_kind[entity_kind].get(entity_name, 0)
            mentions_by_kind[entity_kind][entity_name] = current + len(entity.mentions)
    return entities_by_kind, mentions_by_kind


async def _consolidate_kind(
    kind: str,
    entities: dict[str, list[SpeculatedVariant]],
    mention_counts: dict[str, int],
    state: State,
    deps: Deps,
) -> dict[tuple[str, str], tuple[str, str]]:
    """Process a single entity kind to completion."""
    with logfire.span(f"Consolidate {kind} entities", kind=kind):
        return await _consolidate_kind_impl(kind, entities, mention_counts, state, deps)


async def _consolidate_kind_impl(
    kind: str,
    entities: dict[str, list[SpeculatedVariant]],
    mention_counts: dict[str, int],
    state: State,
    deps: Deps,
) -> dict[tuple[str, str], tuple[str, str]]:
    """Implementation of per-kind consolidation."""
    max_iterations = deps.config.stage.extraction.max_rename_iterations
    threshold = deps.config.stage.extraction.cluster_token_overlap_threshold
    kind_merges = state.consolidated.entities.merges[kind]
    # Track groups resolved (kept separate) to avoid re-asking
    resolved_groups: set[frozenset[str]] = set()
    all_rules: dict[tuple[str, str], tuple[str, str]] = {}
    # Track auto-merges already applied (keyed by normalized child)
    applied_auto_merges: set[str] = set()
    # Track new names from previous iteration (for targeted re-review)
    previous_new_names: set[str] = set()
    for iteration in range(1, max_iterations + 1):
        # Cluster entities
        deps.logger.info(
            f"Clustering {len(entities)} {kind} entities (threshold={threshold:.2f})"
        )
        candidates = find_consolidation_candidates(
            entities, threshold, mention_counts, deps.logger
        )
        # Filter out groups already resolved
        new_groups = [
            g for g in candidates.agent_review_groups if g not in resolved_groups
        ]
        filtered_resolved = len(candidates.agent_review_groups) - len(new_groups)
        # On subsequent iterations, only review groups containing new names from renames
        filtered_no_new_names = 0
        if previous_new_names:
            groups_with_new_names = [g for g in new_groups if g & previous_new_names]
            filtered_no_new_names = len(new_groups) - len(groups_with_new_names)
            new_groups = groups_with_new_names
        candidates.agent_review_groups = new_groups
        if filtered_resolved > 0 or filtered_no_new_names > 0:
            new_group_set = set(new_groups)
            candidates.merge_trees = [
                t for t in candidates.merge_trees if t.entities in new_group_set
            ]
        # Filter auto-merges to only include new ones
        total_auto_merges = len(candidates.auto_merge)
        new_auto_merges = [
            (child, parent, reasoning)
            for child, parent, reasoning in candidates.auto_merge
            if normalize_for_comparison(child) not in applied_auto_merges
        ]
        filtered_auto_merges = total_auto_merges - len(new_auto_merges)
        # Track clusters presented to LLM
        for cluster in candidates.agent_review_groups:
            kind_merges.clusters.append(list(cluster))
        # Log clustering results
        multi_member_groups = [g for g in new_groups if len(g) > 1]
        skip_parts = []
        if filtered_resolved:
            skip_parts.append(f"{filtered_resolved} groups already resolved")
        if filtered_no_new_names:
            skip_parts.append(f"{filtered_no_new_names} groups without new names")
        if filtered_auto_merges:
            skip_parts.append(f"{filtered_auto_merges} auto-merges already applied")
        skip_info = f" (skipped: {', '.join(skip_parts)})" if skip_parts else ""
        deps.logger.info(
            f"Clustering produced {len(new_auto_merges)} auto-merges, "
            f"{len(candidates.agent_review)} pairwise reviews, "
            f"{len(multi_member_groups)} groups for LLM review{skip_info}"
        )
        # Early exit if no work remains
        if (
            not new_auto_merges
            and not candidates.agent_review
            and not candidates.agent_review_groups
        ):
            deps.logger.debug(f"No work remaining for {kind}, exiting early")
            break
        # Log contested warnings
        _log_contested_warnings(candidates.contested_warnings, kind, entities, deps)
        # Process auto-merge decisions (only new ones)
        iteration_rules: dict[tuple[str, str], tuple[str, str]] = {}
        for child, parent, reasoning in new_auto_merges:
            child_norm = normalize_for_comparison(child)
            iteration_rules[(child_norm, kind)] = (parent, reasoning)
            applied_auto_merges.add(child_norm)
        # Process agent review pairs
        new_names: set[str] = set()
        if candidates.agent_review:
            llm_rules, llm_new_names = await _get_consolidation_decisions_for_kind(
                candidates.agent_review, kind, entities, state, deps
            )
            iteration_rules.update(llm_rules)
            new_names.update(llm_new_names)
        # Process agent review groups with merge trees
        if candidates.agent_review_groups:
            (
                group_rules,
                group_new_names,
                group_resolved,
            ) = await _get_group_consolidation_decisions(
                candidates.agent_review_groups,
                candidates.merge_trees,
                kind,
                entities,
                state,
                deps,
            )
            iteration_rules.update(group_rules)
            new_names.update(group_new_names)
            resolved_groups.update(group_resolved)
        # Accumulate rules
        all_rules.update(iteration_rules)
        # Check if we need another iteration
        if not new_names:
            break
        # Add new names for potential further clustering
        _add_new_names_to_kind(new_names, entities)
        previous_new_names = new_names
        deps.logger.debug(
            f"{kind} consolidation iteration {iteration}: "
            f"{len(new_names)} renames, re-evaluating"
        )
    else:
        if new_names:
            deps.logger.warning(
                f"{kind} consolidation hit max iterations ({max_iterations}) "
                f"with {len(new_names)} pending renames"
            )
    return all_rules


def _log_contested_warnings(
    contested_warnings: list,
    kind: str,
    entities: dict[str, list[SpeculatedVariant]],
    deps: Deps,
) -> None:
    """Log contested variant warnings with full context."""
    if not contested_warnings:
        return

    def fmt_variant(v: SpeculatedVariant) -> str:
        source = (
            f":{v.source}" if v.speculation and v.source != "paren_expansion" else ""
        )
        return f"'{v.form}' [{v.speculation}{source}]"

    contested_lines = [f"Contested variants for {kind}:"]
    structured_data = []
    for norm_form, canonical_to_variants in contested_warnings:
        canonicals = list(canonical_to_variants.keys())
        canonical_list = "' and '".join(canonicals)
        variant_details = [
            f"    - {canonical}: {', '.join(fmt_variant(v) for v in entities.get(canonical, []) if normalize_for_comparison(v.form) == norm_form)}"
            for canonical in canonicals
        ]
        similarity = _analyze_entity_similarity(canonicals)
        contested_lines.extend(
            [
                f"  '{norm_form}' from '{canonical_list}':",
                *variant_details,
                f"    - Similarity: {similarity}",
            ]
        )
        structured_data.append(
            {
                "normalized_form": norm_form,
                "entities": canonicals,
                "similarity": similarity,
            }
        )
    deps.logger.info(
        "\n".join(contested_lines),
        extra={"contested_variants": structured_data, "entity_kind": kind},
    )


def _add_new_names_to_kind(
    new_names: set[str], entities: dict[str, list[SpeculatedVariant]]
) -> None:
    """Add new canonical names from renames to entity dict for re-clustering."""
    # Build normalized lookup to avoid adding case variants of existing entities
    existing_normalized = {normalize_for_comparison(k) for k in entities}
    for new_canonical in new_names:
        if normalize_for_comparison(new_canonical) not in existing_normalized:
            entities[new_canonical] = [
                SpeculatedVariant(
                    form=new_canonical,
                    speculation=0,
                    source="original",
                    is_from_alias=False,
                )
            ]
            existing_normalized.add(normalize_for_comparison(new_canonical))


def _analyze_entity_similarity(canonicals: list[str]) -> str:
    """Analyze why entities with the same normalized form are not safe to merge."""
    if len(canonicals) < 2:
        return "single entity"
    a, b = canonicals[0], canonicals[1]
    norm_a = normalize_for_comparison(a)
    norm_b = normalize_for_comparison(b)
    if norm_a == norm_b:
        return "capitalization differs but unsafe (likely different base forms)"
    if is_obvious_variant(norm_a, norm_b):
        return "obvious variant but unsafe for auto-merge"
    dist = osa_distance(norm_a, norm_b)
    if dist <= 2:
        return f"similar forms (edit distance {dist}) but not safe to merge"
    if a.rstrip("0123456789") == b.rstrip("0123456789"):
        return "same base with different numbers (e.g., SMAD2 vs SMAD3)"
    return f"different base forms ({a} vs {b})"


async def _get_consolidation_decisions_for_kind(
    agent_review_pairs: list[tuple[str, str]],
    kind: str,
    entities: dict[str, list[SpeculatedVariant]],
    state: State,
    deps: Deps,
) -> tuple[dict[tuple[str, str], tuple[str, str]], set[str]]:
    """Query LLM for consolidation decisions, using cache to avoid redundant calls."""
    if not agent_review_pairs:
        return {}, set()
    kind_cache = _get_merge_cache(state, kind)
    # Deduplicate pairs and check cache
    unique_pairs: dict[tuple[str, str], tuple[str, str]] = {}
    for child, parent in agent_review_pairs:
        cache_key = (child, parent)
        if cache_key not in unique_pairs:
            unique_pairs[cache_key] = (child, parent)
    # Separate cached and uncached
    uncached: list[tuple[str, str]] = []
    for cache_key, (child, parent) in unique_pairs.items():
        if cache_key not in kind_cache.cache:
            uncached.append((child, parent))
    # Get LLM decisions for uncached pairs
    if uncached:
        batch_size = deps.config.stage.extraction.merge_batch_size
        for i in range(0, len(uncached), batch_size):
            await _process_consolidation_batch(
                uncached[i : i + batch_size], kind, i // batch_size + 1, state, deps
            )
    # Build rules from cache (now contains all pairs)
    return _build_rules_from_cache(unique_pairs, kind, entities, state)


def _build_rules_from_cache(
    pairs: dict[tuple[str, str], tuple[str, str]],
    kind: str,
    entities: dict[str, list[SpeculatedVariant]],
    state: State,
) -> tuple[dict[tuple[str, str], tuple[str, str]], set[str]]:
    """Build consolidation rules from cached decisions."""
    kind_cache = _get_merge_cache(state, kind)
    rules: dict[tuple[str, str], tuple[str, str]] = {}
    new_names: set[str] = set()
    for cache_key in pairs:
        if cache_key in kind_cache.cache:
            target, reasoning = kind_cache.cache[cache_key]
            kind_cache.hits += 1
            if target:  # merge or rename
                child = cache_key[0]
                child_norm = normalize_for_comparison(child)
                rules[(child_norm, kind)] = (target, reasoning)
                if target not in entities:
                    new_names.add(target)
        else:
            kind_cache.misses += 1
    return rules, new_names


async def _process_consolidation_batch(
    batch: list[tuple[str, str]],
    kind: str,
    batch_num: int,
    state: State,
    deps: Deps,
) -> None:
    """Query LLM for entity pair consolidation decisions and cache results."""
    # Build prompt structures
    pairs_description = []
    id_to_pair: dict[int, tuple[str, str, str]] = {}
    token_to_id: dict[str, int] = {}
    for pair_id, (child, parent) in enumerate(batch, start=1):
        if parent == child:
            continue
        token = _generate_token()
        id_to_pair[pair_id] = (parent, child, token)
        token_to_id[token] = pair_id
        pairs_description.append(f"{pair_id}. [{token}] {child!r} → {parent!r} ?")
    if not pairs_description:
        return
    # Build and execute prompt
    prompt = f"""**Research topic:** {state.topic}

**Target entity types:** {", ".join(state.target_entity_types)}

**Entity pairs to evaluate:**
{chr(10).join(pairs_description)}

Only return pairs that should merge or be renamed. Omit pairs that should remain separate."""
    kind_cache = _get_merge_cache(state, kind)
    agent = get_entity_consolidation_agent(deps.config)
    try:
        with rename_agent(agent, name=f"ConsolidateEntities ({kind}, {batch_num})"):
            async with deps.agent_semaphore:
                result = await agent.run(prompt, deps=deps)
        record_usage(deps.usage, "entity_consolidation", agent, result)
        # Process decisions and populate cache
        returned_ids = {d.pair_id for d in result.output.decisions}
        for decision in result.output.decisions:
            resolved = _resolve_pair_from_decision(
                decision.pair_id,
                decision.confirm_token,
                id_to_pair,
                token_to_id,
                deps.logger,
            )
            if resolved is None:
                continue
            parent, child = resolved
            cache_key = (child, parent)
            target = decision.rename if decision.rename else parent
            kind_cache.cache[cache_key] = (target, decision.reasoning)
        # Cache implicit skips (pairs not returned by LLM)
        for pair_id, (parent, child, token) in id_to_pair.items():
            if pair_id not in returned_ids:
                cache_key = (child, parent)
                kind_cache.cache[cache_key] = (None, "implicit_skip")
    except AGENT_CALL_ERRORS as e:
        deps.logger.error(f"Entity consolidation failed: {type(e).__name__}: {e}")


async def _get_group_consolidation_decisions(
    groups: list[frozenset[str]],
    merge_trees: list,
    kind: str,
    entities: dict[str, list[SpeculatedVariant]],
    state: State,
    deps: Deps,
) -> tuple[dict[tuple[str, str], tuple[str, str]], set[str], set[frozenset[str]]]:
    """Query LLM for group consolidation decisions using tree-based splitting."""
    from interaction_finder.extraction.clustering import Cluster

    if not groups:
        return {}, set(), set()
    max_rounds = deps.config.stage.extraction.cluster_refinement_max_rounds
    all_rules: dict[tuple[str, str], tuple[str, str]] = {}
    all_new_names: set[str] = set()
    all_resolved: set[frozenset[str]] = set()
    # Build map from frozenset → tree for splitting
    group_to_tree: dict[frozenset[str], Cluster] = {
        tree.entities: tree for tree in merge_trees
    }
    get_agent = agent_getter(
        "extraction",
        "entity_group_consolidation",
        ClusterDecisions,
        Deps,
        """You are an expert at consolidating biomedical entity names for systematic literature analysis.

You will be given groups of entities that appear related. For each group, decide if all members should merge to a single canonical entity, based on the research topic.

**Decision framework:**
1. Evaluate each group in the context of the research topic
2. Ask: If a researcher found information about one entity, would they assume it applies to the others? If not, keep them separate.
3. Apply these principles:
   - Entities differing only by clinical subtypes or modifiers → merge to parent term
     (e.g., "Idiopathic/Familial/Heritable [Disease]" → "[Disease]")
   - Entities representing the same underlying condition → merge
     (e.g., "tumour" and "tumor" → consistent spelling)
   - One entity clearly doesn't belong → exclude it (measurements mixed with diseases, etc.)
   - Entities representing genuinely distinct biological phenomena → keep separate
     (e.g., different diseases, different genes)

The research topic tells you what distinctions matter. Merge entities only if a researcher would treat them as interchangeable, not merely because they're related to the topic.

**Actions:**
- **merge**: All members represent the same entity → specify target (member number, name, or new name)
- **exclude**: One specific member doesn't belong → specify which one to remove (member number or name)
- **split**: Cluster mixes unrelated entities but can't identify which → system splits at weakest link

Only return groups that need action (merge/exclude/split). Omit groups that should remain separate.""",
    )
    agent = get_agent(deps.config)
    current_groups = list(groups)
    batch_size = deps.config.stage.extraction.merge_batch_size
    entity_types_str = ", ".join(state.target_entity_types)
    for round_num in range(1, max_rounds + 1):
        if not current_groups:
            break
        num_batches = -(-len(current_groups) // batch_size)
        batch_suffix = f" in {num_batches} batches" if num_batches > 1 else ""
        deps.logger.info(
            f"Entity group consolidation round {round_num}/{max_rounds}: "
            f"{len(current_groups)} groups{batch_suffix}"
        )
        # Process batches, collecting groups that need further review
        groups_needing_review: list[frozenset[str]] = []
        stats = {"merged": 0, "split": 0, "excluded": 0, "kept_separate": 0}
        for batch_start in range(0, len(current_groups), batch_size):
            batch = current_groups[batch_start : batch_start + batch_size]
            # Build prompt data for this batch
            group_data = [
                {
                    "id": _generate_token(),
                    "entities": g,
                    "members": sorted(g, key=lambda e: (len(e), e)),
                }
                for g in batch
            ]
            groups_text = "\n\n".join(
                f"## Group {g['id']}\nMembers:\n"
                + "\n".join(f"  {i + 1}. {m}" for i, m in enumerate(g["members"]))
                for g in group_data
            )
            prompt = f"""**Research topic:** {state.topic}
**Target entity types:** {entity_types_str}

**Entity groups to consolidate:**

{groups_text}

---

Omit groups that should stay separate. Action defaults to "merge" if omitted.

Examples:
- Merge: group_id="abc", target="1", reasoning="All are PAH subtypes"
- Exclude: group_id="def", action="exclude", target="5", reasoning="Member 5 (TAPSE) is a measurement"
- Split: group_id="ghi", action="split", reasoning="Mixes diseases and measurements"
"""
            try:
                async with deps.agent_semaphore:
                    result = await agent.run(prompt, deps=deps)
                record_usage(deps.usage, "entity_group_consolidation", agent, result)
            except AGENT_CALL_ERRORS as e:
                deps.logger.error(
                    f"Entity group consolidation failed for batch: {type(e).__name__}: {e}"
                )
                # Treat all groups in this batch as kept separate
                all_resolved.update(batch)
                stats["kept_separate"] += len(batch)
                continue
            # Index decisions by group_id (strip optional "Group " prefix from LLM response)
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
            groups_with_decisions: set[frozenset[str]] = set()
            for gid, decisions in decisions_by_id.items():
                g = id_to_group[gid]
                groups_with_decisions.add(g["entities"])
                members = g["members"]
                # Categorize decisions
                excludes = [d for d in decisions if d.action == "exclude"]
                merges = [d for d in decisions if d.action == "merge"]
                splits = [d for d in decisions if d.action == "split"]
                if merges and splits:
                    deps.logger.warning(f"Group {gid}: merge+split conflict")
                    continue
                # Apply excludes first
                remaining = set(members)
                for exc in excludes:
                    if exc.target:
                        member = _resolve_group_target(
                            exc.target, members, entities, deps.logger
                        )
                        if member in remaining:
                            remaining.discard(member)
                            stats["excluded"] += 1
                if len(remaining) <= 1:
                    continue
                remaining_entities = frozenset(remaining)
                # Apply merge or split
                if merges:
                    merge = merges[0]
                    if not merge.target:
                        continue
                    target = _resolve_group_target(
                        merge.target, members, entities, deps.logger
                    )
                    if target in members and target not in remaining:
                        deps.logger.warning(
                            f"Group {gid}: merge target '{target}' was excluded"
                        )
                        continue
                    if target not in entities:
                        all_new_names.add(target)
                    for member in remaining:
                        if member != target:
                            all_rules[(normalize_for_comparison(member), kind)] = (
                                target,
                                merge.reasoning or f"Group {gid}",
                            )
                    stats["merged"] += 1
                elif splits:
                    tree = group_to_tree.get(g["entities"]) or group_to_tree.get(
                        remaining_entities
                    )
                    if tree and len(remaining) < len(g["entities"]):
                        tree = tree.find_subtree(remaining_entities) or tree
                    if not tree:
                        deps.logger.warning(f"No tree for group {gid}")
                        continue
                    for sub in tree.split_into_n(2):
                        if len(sub) > 1:
                            groups_needing_review.append(sub)
                            if sub not in group_to_tree and (
                                st := tree.find_subtree(sub)
                            ):
                                group_to_tree[sub] = st
                    stats["split"] += 1
                elif excludes:
                    groups_needing_review.append(remaining_entities)
                    if (tree := group_to_tree.get(g["entities"])) and (
                        st := tree.find_subtree(remaining_entities)
                    ):
                        group_to_tree[remaining_entities] = st
            # Groups without decisions are kept separate - mark as resolved
            kept_separate = set(batch) - groups_with_decisions
            all_resolved.update(kept_separate)
            stats["kept_separate"] += len(kept_separate)
        deps.logger.info(
            f"  Results: {stats['merged']} merged, {stats['split']} split, "
            f"{stats['excluded']} excluded, {stats['kept_separate']} separate"
        )
        if not groups_needing_review:
            break
        current_groups = groups_needing_review
    deps.logger.info(
        f"Entity group consolidation complete: {len(all_rules)} merge rules created, "
        f"{len(all_new_names)} new canonical names"
    )
    return all_rules, all_new_names, all_resolved


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
                # Resolve imprecise responses (β vs beta, hyphens, etc.)
                try:
                    name_match = find_entity_match(name_part, entities)
                except (IndexError, KeyError):
                    name_match = None
                # Compare canonicals case-sensitively (both from entities dict)
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
    merge_rules: dict[tuple[str, str], tuple[str, str]],
) -> dict[tuple[str, str], tuple[str, str]]:
    """Resolve transitive merge chains (A→B, B→C becomes A→C, B→C)."""
    resolved = {}
    for (child_norm, kind), (target, reasoning) in merge_rules.items():
        final_target = target
        final_reasoning = reasoning
        visited = {child_norm}
        while True:
            target_norm = normalize_for_comparison(final_target)
            if (target_norm, kind) not in merge_rules:
                break
            if target_norm in visited:
                break
            visited.add(target_norm)
            final_target, final_reasoning = merge_rules[(target_norm, kind)]
        resolved[(child_norm, kind)] = (final_target, final_reasoning)
    return resolved


def _apply_merge_rules_globally(
    rules: dict[tuple[str, str], tuple[str, str]],
    state: State,
) -> None:
    """Apply merge/rename rules to all documents."""
    if not rules:
        return
    # Store rules in consolidated structure
    for (norm_name, kind), (target, reasoning) in rules.items():
        rule = EntityMergeRule(source=norm_name, target=target, reasoning=reasoning)
        kind_merges = state.consolidated.entities.merges.setdefault(
            kind, EntityKindMerges()
        )
        if reasoning.startswith("auto:"):
            kind_merges.automatic.append(rule)
        else:
            kind_merges.llm_decided.append(rule)
    target_by_norm_and_kind: dict[tuple[str, str], str] = {}
    for (_norm, kind), (target, _) in rules.items():
        target_key = (normalize_for_comparison(target), kind)
        target_by_norm_and_kind.setdefault(target_key, target)
    for resource_id, entities in state.validated_entities_by_resource.items():
        grouped: dict[str, list[EntityMention]] = defaultdict(list)
        for child_name, ref in entities.items():
            target_name = None
            all_forms = extract_all_forms(child_name, ref.aliases())
            for form in all_forms:
                key = (normalize_for_comparison(form), ref.kind)
                if key in rules:
                    target_name = rules[key][0]
                    break
            canonical_target = target_by_norm_and_kind.get(
                (normalize_for_comparison(child_name), ref.kind)
            )
            canonical = target_name or canonical_target or child_name
            grouped[canonical].extend(ref.mentions)
        new_entities: dict[str, EntityRef] = {
            canonical: EntityRef(canonical=canonical, mentions=mentions)
            for canonical, mentions in grouped.items()
        }
        merges_for_resource = max(0, len(entities) - len(new_entities))
        state.entities_merged += merges_for_resource
        state.validated_entities_by_resource[resource_id] = new_entities


def _update_pair_entity_references(
    merge_rules: dict[tuple[str, str], tuple[str, str]],
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


def _build_global_entity_index(state: State) -> None:
    """Aggregate mentions across all resources into global entity index."""
    mentions_by_canonical: dict[str, list[EntityMention]] = defaultdict(list)
    for entities in state.validated_entities_by_resource.values():
        for canonical, ref in entities.items():
            mentions_by_canonical[canonical].extend(ref.mentions)
    state.global_entities = {
        c: EntityRef(canonical=c, mentions=m) for c, m in mentions_by_canonical.items()
    }
