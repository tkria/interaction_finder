"""Consolidate relationship labels and classify polarities globally.

Unified algorithm:
1. Collect all unique relationship labels from all assessments
2. Query LLM for consolidation + polarity classification (single call)
3. Apply consolidations to all assessments
4. Store polarity mappings in state
5. Filter irrelevant relationship types (optional, on by default)

This approach ensures:
- Vocabulary consolidation (merges synonyms like "linked_to" → "associated_with")
- Polarity classification (positive/negative/neutral/irrelevant)
- Biological direction-aware normalization (context-aware decisions)
- Consistency (same canonical label and polarity across all documents)
"""

from collections import Counter, defaultdict
from statistics import median_low

from interaction_finder.agent_config import AGENT_CALL_ERRORS
from interaction_finder.agent_utils import rename_agent
from interaction_finder.usage import record_usage
from interaction_finder.extraction.consolidate_relationships import (
    get_relationship_consolidation_agent,
)
from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.models import (
    EntityPairKey,
    PairAssessment,
    PairJudgment,
    RelationshipConsolidation,
    SimpleEntity,
    StageEvent,
)
from interaction_finder.extraction.shared import (
    aggregate_evidence,
    get_entity_aliases,
    save_checkpoint,
)
from interaction_finder.extraction.state import State
from interaction_finder.extraction.utils import (
    build_pair_spread,
    make_entity_pair_key,
    normalize_for_comparison,
)
from interaction_finder.logging import logfire


async def consolidate_relationships(state: State, deps: Deps) -> bool:
    """Consolidate relationship labels, classify polarities, and filter irrelevant."""
    with logfire.span("consolidate_relationships"):
        if deps.progress:
            deps.progress.set_status("Consolidating relationships")
        # Record stage boundary
        state.consolidated.entities.stages.append(
            StageEvent(name="consolidate_relationships", event=state.current_event())
        )
        # Step 1: Collect relationship labels with counts
        relationship_counts = _collect_relationship_counts(state)
        if not relationship_counts:
            deps.logger.info("No relationships to consolidate")
            return True
        # Initialize progress counter
        if deps.progress:
            deps.progress["Relationship labels"].total = len(relationship_counts)
            deps.progress["Relationship labels"].activate()
        # Step 2: Get consolidation + polarity from LLM (unified)
        consolidations = await _consolidate_and_classify(
            relationship_counts, state, deps
        )
        if not consolidations:
            deps.logger.warning(
                "Relationship consolidation agent returned no data; "
                "defaulting all relationship polarities to neutral"
            )
            _ensure_polarities_for_all_relationships(
                set(relationship_counts.keys()), state
            )
            return True
        # Step 3: Apply consolidations to assessments
        _apply_consolidations(consolidations, state)
        # Step 4: Store polarity mappings
        _store_polarity_mappings(consolidations, state)
        _ensure_polarities_for_all_relationships(
            set(relationship_counts.keys()), state, deps, log_missing=True
        )
        # Step 4b: Store relationship consolidations in unified structure with events
        for cons in consolidations:
            cons.event = state.next_event()
            state.consolidated.relationships.append(cons)
        deps.logger.info(
            f"Relationship consolidation: {len(consolidations)} relationships processed, "
            f"{state.relationships_merged} assessments updated"
        )
        # Complete progress counter with polarity breakdown
        if deps.progress:
            deps.progress["Relationship labels"].completed = len(consolidations)
            polarity_counts = Counter(c.polarity for c in consolidations)
            breakdown = ", ".join(f"{n} {p}" for p, n in polarity_counts.most_common())
            deps.progress["Relationship labels"].note = f"({breakdown})"
            deps.progress["Relationship labels"].complete()
        # Step 5: Filter irrelevant relationship types (if enabled)
        if deps.config.stage.extraction.filter_irrelevant_relationships:
            _filter_irrelevant_assessments(state, deps)
        # Save checkpoint after relationship consolidation
        await save_checkpoint(state, deps, "consolidate_relationships")
        return True


def _collect_relationship_counts(state: State) -> dict[str, int]:
    """Collect relationship labels with their frequency counts."""
    counts: dict[str, int] = {}
    for assessments in state.pair_assessments_by_resource.values():
        for assessment in assessments:
            counts[assessment.relationship] = counts.get(assessment.relationship, 0) + 1
    return counts


async def _consolidate_and_classify(
    relationship_counts: dict[str, int], state: State, deps: Deps
) -> list[RelationshipConsolidation]:
    """Query LLM for consolidation + polarity classification (unified)."""
    # Sort by frequency (descending) so LLM sees common labels first
    sorted_relationships = sorted(
        relationship_counts.items(), key=lambda x: (-x[1], x[0])
    )
    relationships_str = "\n".join(
        f"- {label!r} ({count})" for label, count in sorted_relationships
    )
    entity_types_str = ", ".join(state.target_entity_types)
    prompt = f"""**Research topic:** {state.topic}

**Target entity types:** {entity_types_str}

**Relationship labels found (with frequency):**
{relationships_str}

For each relationship, provide:
1. Consolidated canonical form (prefer high-frequency labels as canonical forms)
2. Polarity classification relative to this research topic"""
    agent = get_relationship_consolidation_agent(deps.config)
    try:
        with rename_agent(agent, name="consolidate_relationships"):
            async with deps.agent_semaphore:
                result = await agent.run(prompt, deps=deps)
        record_usage(deps.usage, "relationship_consolidation", agent, result)
        return result.output.consolidations
    except AGENT_CALL_ERRORS as e:
        deps.logger.error(f"Relationship consolidation failed: {type(e).__name__}: {e}")
        return []


def _apply_consolidations(
    consolidations: list[RelationshipConsolidation], state: State
) -> None:
    """Apply consolidations to all assessments."""
    consolidation_map: dict[str, str] = {}
    for cons in consolidations:
        norm_orig = normalize_for_comparison(cons.original)
        if cons.original != cons.consolidated:
            consolidation_map[norm_orig] = cons.consolidated
    if not consolidation_map:
        return
    for assessments in state.pair_assessments_by_resource.values():
        for assessment in assessments:
            norm = normalize_for_comparison(assessment.relationship)
            if norm in consolidation_map:
                assessment.relationship = consolidation_map[norm]
                state.relationships_merged += 1
    state.relationship_mappings = consolidation_map


def _store_polarity_mappings(
    consolidations: list[RelationshipConsolidation], state: State
) -> None:
    """Store polarity mappings from consolidations."""
    for cons in consolidations:
        state.relationship_polarities[cons.original] = cons.polarity
        state.relationship_polarities[cons.consolidated] = cons.polarity


def _ensure_polarities_for_all_relationships(
    relationships: set[str],
    state: State,
    deps: Deps | None = None,
    log_missing: bool = False,
) -> None:
    """Ensure every relationship label has a polarity mapping."""
    missing = [rel for rel in relationships if rel not in state.relationship_polarities]
    if not missing:
        return
    for rel in missing:
        state.relationship_polarities[rel] = "neutral"
    if log_missing and deps:
        deps.logger.warning(
            f"Assigned neutral polarity to {len(missing)} "
            "relationships missing classification"
        )


def _filter_irrelevant_assessments(state: State, deps: Deps) -> None:
    """Remove assessments with irrelevant polarity."""
    # Group assessments by pair
    grouped: dict[EntityPairKey, list[PairAssessment]] = defaultdict(list)
    for assessments in state.pair_assessments_by_resource.values():
        for assessment in assessments:
            pair_key = make_entity_pair_key(assessment.entity1, assessment.entity2)
            grouped[pair_key].append(assessment)
    # Create pre-rejected judgments for pairs that are ALL irrelevant
    filtered_pairs: set[EntityPairKey] = set()
    for pair_key, assessments in grouped.items():
        all_irrelevant = all(
            state.relationship_polarities.get(a.relationship) == "irrelevant"
            for a in assessments
        )
        if all_irrelevant:
            spread = build_pair_spread(assessments, state.relationship_polarities)
            first = assessments[0]
            state.pair_judgments[pair_key] = PairJudgment(
                entity1=SimpleEntity(
                    name=first.entity1.canonical,
                    kind=first.entity1.kind,
                    aliases=get_entity_aliases(first.entity1.canonical, state),
                ),
                entity2=SimpleEntity(
                    name=first.entity2.canonical,
                    kind=first.entity2.kind,
                    aliases=get_entity_aliases(first.entity2.canonical, state),
                ),
                relationship=first.relationship,
                spread=spread,
                accepted=False,
                evidence=aggregate_evidence(assessments),
                topic_relevance=median_low(
                    [a.topic_relevance for a in assessments if a.evidence.overall >= 7]
                    or [a.topic_relevance for a in assessments]
                ),
                decision_confidence=0.95,
                reasoning=(
                    "All relationship types for this pair were classified as "
                    "irrelevant to the research question"
                ),
            )
            filtered_pairs.add(pair_key)
    if filtered_pairs:
        _remove_assessments_for_pairs(filtered_pairs, state)
        deps.logger.info(
            f"Filtered {len(filtered_pairs)} pairs with only irrelevant relationships"
        )


def _remove_assessments_for_pairs(pair_keys: set[EntityPairKey], state: State) -> None:
    """Remove assessments for the provided pair keys from state."""
    for resource_id, assessments in list(state.pair_assessments_by_resource.items()):
        filtered = [
            assessment
            for assessment in assessments
            if make_entity_pair_key(assessment.entity1, assessment.entity2)
            not in pair_keys
        ]
        state.pair_assessments_by_resource[resource_id] = filtered
