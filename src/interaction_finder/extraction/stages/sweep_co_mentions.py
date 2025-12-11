"""Sweep documents for missed co-mentions of assessed entity pairs.

Searches normalized text for entity co-occurrences that weren't captured
by the initial proximal set extraction, assesses them for relationship
claims, and adds any new assessments to the state.
"""

import asyncio
import re
from collections import defaultdict

from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.models import EvidenceQuality, PairAssessment
from interaction_finder.extraction.shared import save_checkpoint
from interaction_finder.extraction.state import State
from interaction_finder.extraction.sweep_co_mentions import (
    CoMentionRegion,
    CoMentionSweepStats,
    assess_co_mention_region,
    build_entity_search_pattern,
    collect_assessed_pairs,
    collect_global_aliases,
    find_novel_co_mentions_in_resource,
    get_entity_kind,
    merge_co_mentions_into_regions,
    select_co_mentions_to_assess,
)
from interaction_finder.logging import logfire
from interaction_finder.resources import ResourceId


async def sweep_co_mentions(state: State, deps: Deps) -> bool:
    """Scan for missed co-mentions and assess them."""
    with logfire.span("sweep_co_mentions"):
        deps.progress["Candidates"].activate()
        deps.progress.set_status("Sweeping for missed co-mentions")
        # Check if sweep is enabled
        if not deps.config.tools.extraction.sweep_co_mentions:
            deps.logger.info("Co-mention sweep disabled")
            return True
        stats = CoMentionSweepStats()
        # Step 1: Build global alias map
        global_aliases = collect_global_aliases(state.validated_entities_by_resource)
        # Step 2: Collect all assessed pairs
        assessed_pairs = collect_assessed_pairs(state.pair_assessments_by_resource)
        if not assessed_pairs:
            deps.logger.info("No assessed pairs to sweep for co-mentions")
            state.co_mention_sweep_stats = stats
            return True
        # Step 3: Build search patterns for all entities in assessed pairs
        entity_patterns: dict[str, re.Pattern] = {}
        for pair_key in assessed_pairs:
            for name in (pair_key.entity1_name, pair_key.entity2_name):
                if name not in entity_patterns:
                    aliases = global_aliases.get(name, set())
                    entity_patterns[name] = build_entity_search_pattern(name, aliases)
        # Step 4: Find novel co-mentions in all resources
        chunk_distance = deps.config.tools.extraction.proximal_window_chunks
        all_co_mentions = []
        for resource in deps.resource_pool.resources:
            resource_co_mentions = find_novel_co_mentions_in_resource(
                resource,
                resource.id,
                assessed_pairs,
                entity_patterns,
                state.pair_assessments_by_resource,
                chunk_distance,
            )
            all_co_mentions.extend(resource_co_mentions)
        # Update discovery stats (raw counts, before deduplication)
        stats.total_co_mentions_found = len(all_co_mentions)
        for cm in all_co_mentions:
            if cm.priority == "no_existing_assessment":
                stats.co_mentions_no_existing_assessment += 1
            else:
                stats.co_mentions_uncovered_region += 1
        deps.logger.info(
            f"Found {stats.total_co_mentions_found} co-mention occurrences "
            f"({stats.co_mentions_no_existing_assessment} no existing assessment, "
            f"{stats.co_mentions_uncovered_region} uncovered region)"
        )
        if not all_co_mentions:
            state.co_mention_sweep_stats = stats
            return True
        # Step 5: Select co-mentions to assess (policy hook)
        selected = select_co_mentions_to_assess(all_co_mentions, deps.config)
        # Step 6: Build entity kind lookup and merge into regions
        entity_kinds: dict[str, str] = {}
        for name in entity_patterns:
            kind = get_entity_kind(name, state.validated_entities_by_resource)
            if kind:
                entity_kinds[name] = kind
        regions = merge_co_mentions_into_regions(selected, entity_kinds)
        stats.regions_created = len(regions)
        # Count deduplicated pairs
        pairs_to_assess = sum(len(r.candidate_pairs) for r in regions)
        # Update progress display
        deps.progress["Candidates"].completed = pairs_to_assess
        deps.progress["Regions"].total = len(regions)
        deps.progress["Regions"].activate()
        deps.progress["Pairs added"].activate()
        deps.logger.info(
            f"Merged {len(selected)} co-mentions into {len(regions)} regions "
            f"({pairs_to_assess} candidate pairs)"
        )
        # Collect known relationship types for prompt context
        known_relationships = sorted(state.relationship_polarities.keys())
        # Accumulate new pairs by resource
        new_pairs_by_resource: dict[ResourceId, list[tuple]] = defaultdict(list)

        # Step 7: Assess regions concurrently
        async def assess_region(region_index: int, region: CoMentionRegion):
            """Assess all pairs in a region with a single LLM call."""
            try:
                resource = deps.resource_pool.get(region.resource_id)
                if resource is None:
                    return (region, [], [])
                validated_entities = state.validated_entities_by_resource.get(
                    region.resource_id, {}
                )
                assessments, new_pairs = await assess_co_mention_region(
                    region,
                    resource,
                    topic=state.topic,
                    known_relationships=known_relationships,
                    config=deps.config,
                    deps=deps,
                    region_index=region_index,
                    validated_entities=validated_entities,
                    permitted_pairs=state.permitted_pairs,
                )
                return (region, assessments, new_pairs)
            finally:
                deps.progress["Regions"].done()

        tasks = [
            asyncio.create_task(assess_region(idx + 1, r))
            for idx, r in enumerate(regions)
        ]
        for coro in asyncio.as_completed(tasks):
            region, assessments, new_pairs = await coro
            # Update stats
            stats.assessed += len(region.candidate_pairs)
            stats.relationships_found += len(assessments)
            stats.no_relationship_claim += len(region.candidate_pairs) - len(
                assessments
            )
            stats.new_pairs_discovered += len(new_pairs)
            deps.progress["Pairs added"].completed = stats.relationships_found
            deps.progress["Pairs added"].total = stats.assessed
            # Add assessments to state
            for assessment in assessments:
                if region.resource_id not in state.pair_assessments_by_resource:
                    state.pair_assessments_by_resource[region.resource_id] = []
                state.pair_assessments_by_resource[region.resource_id].append(
                    assessment
                )
            # Accumulate new pairs for later assessment
            new_pairs_by_resource[region.resource_id].extend(new_pairs)
        # Mark sweep phase complete
        deps.progress["Regions"].complete()
        deps.progress["Pairs added"].complete()
        deps.logger.info(
            f"Co-mention sweep assessed {stats.assessed} pairs in {stats.regions_created} regions, "
            f"found {stats.relationships_found} relationships, "
            f"discovered {stats.new_pairs_discovered} new pairs"
        )
        # Step 8: Convert discovered pairs to assessments
        for resource_id, raw_pairs in new_pairs_by_resource.items():
            entities = state.validated_entities_by_resource.get(resource_id, {})
            for e1_name, e2_name, rel_types, quotes in raw_pairs:
                entity1, entity2 = entities.get(e1_name), entities.get(e2_name)
                if entity1 is None or entity2 is None:
                    continue
                if resource_id not in state.pair_assessments_by_resource:
                    state.pair_assessments_by_resource[resource_id] = []
                state.pair_assessments_by_resource[resource_id].append(
                    PairAssessment(
                        resource_id=resource_id,
                        entity1=entity1,
                        entity2=entity2,
                        relationship=rel_types[0] if rel_types else "associated_with",
                        quotes=quotes,
                        evidence=EvidenceQuality(
                            directness="implied",
                            source_type="primary",
                            specificity="associative",
                            language="hedged",
                            overall=5,
                        ),
                        topic_relevance=3,
                        reasoning="Discovered during co-mention sweep",
                        source="sweep",
                    )
                )
                stats.new_pairs_assessed += 1
        if stats.new_pairs_assessed > 0:
            deps.logger.info(f"Added {stats.new_pairs_assessed} newly discovered pairs")
        state.co_mention_sweep_stats = stats
        # Save checkpoint after co-mention sweep
        await save_checkpoint(state, deps, "sweep_co_mentions")
        return True
