"""Consolidate relationship labels from new assessments (incremental).

After the co-mention sweep, any new relationship labels need polarity
classification. This stage finds labels not yet in relationship_polarities
and classifies them.
"""

from collections import defaultdict
from statistics import median_low

from pydantic_ai.usage import RunUsage

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


async def consolidate_new_relationships(state: State, deps: Deps) -> bool:
    """Classify polarities for any new relationship labels."""
    with logfire.span("consolidate_new_relationships"):
        if deps.progress:
            deps.progress.set_status("Classifying new relationships")
        # Record stage boundary
        state.consolidated.entities.stages.append(
            StageEvent(
                name="consolidate_new_relationships", event=state.current_event()
            )
        )
        # Collect all current relationship labels
        all_labels: set[str] = set()
        for assessments in state.pair_assessments_by_resource.values():
            for assessment in assessments:
                all_labels.add(assessment.relationship)
        # Find labels not yet classified
        new_labels = {
            label for label in all_labels if label not in state.relationship_polarities
        }
        if not new_labels:
            deps.logger.info("No new relationship labels to classify")
            return True
        deps.logger.info(f"Classifying {len(new_labels)} new relationship labels")
        # Query LLM for polarity classification
        new_labels_list = sorted(new_labels)
        new_labels_str = "\n".join(f"- {r!r}" for r in new_labels_list)
        entity_types_str = ", ".join(state.target_entity_types)
        # Include existing labels so agent can consolidate new ones into them
        existing_labels = sorted(state.relationship_polarities.keys())
        if existing_labels:
            existing_str = "\n".join(
                f"- {r!r} ({state.relationship_polarities[r]})" for r in existing_labels
            )
            existing_section = f"""
**Existing canonical relationship labels (for reference):**
{existing_str}
"""
        else:
            existing_section = ""
        prompt = f"""**Research topic:** {state.topic}

**Target entity types:** {entity_types_str}
{existing_section}
**New relationship labels to classify:**
{new_labels_str}

For each new relationship, provide:
1. Consolidated canonical form (use an existing label if appropriate, or the original if distinct)
2. Polarity classification relative to this research topic"""
        agent = get_relationship_consolidation_agent(deps.config)
        usage = RunUsage()
        try:
            # Initial attempt
            with rename_agent(agent, name="consolidate_new_relationships"):
                async with deps.agent_semaphore:
                    result = await agent.run(prompt, deps=deps, usage=usage)
            _apply_relationship_consolidations(result.output.consolidations, state)
            # Retry up to 3 times for any missing labels
            missing = new_labels - set(state.relationship_polarities.keys())
            for attempt in range(1, 4):
                if not missing:
                    break
                deps.logger.warning(
                    f"LLM omitted {len(missing)} labels; retrying (attempt {attempt}/3)"
                )
                retry_prompt = _build_classification_prompt(
                    state, missing, entity_types_str, existing_section
                )
                try:
                    with rename_agent(
                        agent, name=f"consolidate_new_relationships-retry{attempt}"
                    ):
                        async with deps.agent_semaphore:
                            result = await agent.run(
                                retry_prompt, deps=deps, usage=usage
                            )
                    _apply_relationship_consolidations(
                        result.output.consolidations, state
                    )
                    missing = missing - set(state.relationship_polarities.keys())
                except AGENT_CALL_ERRORS as e:
                    deps.logger.warning(f"Retry {attempt} failed: {type(e).__name__}")
                    break
            # Default any remaining missing labels to neutral
            if missing:
                deps.logger.warning(
                    f"Defaulting {len(missing)} missing labels to neutral"
                )
                for label in missing:
                    state.relationship_polarities[label] = "neutral"
            # Record accumulated usage from all attempts
            record_usage(deps.usage, "relationship_consolidation", agent, usage)
        except AGENT_CALL_ERRORS as e:
            deps.logger.error(
                f"New relationship consolidation failed: {type(e).__name__}: {e}; "
                f"defaulting new labels to neutral polarity"
            )
            for label in new_labels:
                state.relationship_polarities[label] = "neutral"
        # Filter pairs with only irrelevant relationships (if enabled)
        if deps.config.stage.extraction.filter_irrelevant_relationships:
            _filter_irrelevant_pairs(state, deps)
        # Save checkpoint after new relationship consolidation
        await save_checkpoint(state, deps, "consolidate_new_relationships")
        return True


def _build_classification_prompt(
    state: State, labels: set[str], entity_types_str: str, existing_section: str
) -> str:
    """Build prompt for relationship classification."""
    labels_str = "\n".join(f"- {r!r}" for r in sorted(labels))
    return f"""**Research topic:** {state.topic}

**Target entity types:** {entity_types_str}
{existing_section}
**Relationship labels to classify:**
{labels_str}

For each relationship, provide:
1. Consolidated canonical form (use an existing label if appropriate, or the original if distinct)
2. Polarity classification relative to this research topic"""


def _apply_relationship_consolidations(
    consolidations: list[RelationshipConsolidation], state: State
) -> None:
    """Apply relationship consolidations and store polarities."""
    # Store polarities
    for cons in consolidations:
        state.relationship_polarities[cons.original] = cons.polarity
        state.relationship_polarities[cons.consolidated] = cons.polarity
    # Append to unified consolidated structure with event numbers
    for cons in consolidations:
        cons.event = state.next_event()
        state.consolidated.relationships.append(cons)
    # Apply consolidation if label changed
    for cons in consolidations:
        if cons.original != cons.consolidated:
            norm_orig = normalize_for_comparison(cons.original)
            for assessments in state.pair_assessments_by_resource.values():
                for assessment in assessments:
                    if normalize_for_comparison(assessment.relationship) == norm_orig:
                        assessment.relationship = cons.consolidated


def _filter_irrelevant_pairs(state: State, deps: Deps) -> None:
    """Filter pairs that now have only irrelevant relationships after sweep."""
    # Group all assessments by pair
    grouped: dict[EntityPairKey, list[PairAssessment]] = defaultdict(list)
    for assessments in state.pair_assessments_by_resource.values():
        for assessment in assessments:
            pair_key = make_entity_pair_key(assessment.entity1, assessment.entity2)
            grouped[pair_key].append(assessment)
    # Find pairs that are ALL irrelevant and not already judged
    filtered_pairs: set[EntityPairKey] = set()
    for pair_key, assessments in grouped.items():
        if pair_key in state.pair_judgments:
            continue
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
        for resource_id, assessments in list(
            state.pair_assessments_by_resource.items()
        ):
            filtered = [
                a
                for a in assessments
                if make_entity_pair_key(a.entity1, a.entity2) not in filtered_pairs
            ]
            state.pair_assessments_by_resource[resource_id] = filtered
        deps.logger.info(
            f"Filtered {len(filtered_pairs)} pairs with only irrelevant relationships "
            "(from sweep assessments)"
        )
