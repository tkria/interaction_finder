"""Make final accept/reject decisions across documents.

For each unique entity pair:
1. Check if deterministic accept is possible (multiple high-confidence, consistent relationship)
2. Otherwise, investigate by:
   - Collecting all quotes from all assessments
   - Building combined text
   - Calling cross_document_judge_agent
3. Create PairJudgment with decision
4. Store in state
"""

import asyncio
from collections import Counter, defaultdict
from itertools import combinations

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.usage import RunUsage

from interaction_finder.agent_utils import rename_agent
from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.judge_cross_document import (
    get_cross_document_judge_agent,
)
from interaction_finder.extraction.models import (
    EntityPairKey,
    PairAssessment,
    PairJudgment,
    PairSpread,
    SimpleEntity,
)
from interaction_finder.extraction.shared import (
    aggregate_evidence,
    get_entity_aliases,
)
from interaction_finder.extraction.state import State
from interaction_finder.extraction.utils import (
    adjust_heading_levels,
    are_relationships_opposed,
    build_pair_spread,
    collect_relevant_text_for_quotes,
    extract_document_citations,
    make_entity_pair_key,
    opposition_map_from_consolidations,
    validate_document_citations,
)
from interaction_finder.logging import logfire


async def judge_cross_document(state: State, deps: Deps) -> bool:
    """Judge all unique pairs across documents."""
    with logfire.span("judge_cross_document"):
        if deps.progress:
            deps.progress["Unique pairs"].activate()
            deps.progress.set_status("Cross-document validation")
        # Build opposition map once from consolidated relationships
        opposition_map = opposition_map_from_consolidations(
            state.consolidated.relationships
        )
        # Group assessments by entity pair
        assessments_by_pair: dict[EntityPairKey, list[PairAssessment]] = defaultdict(
            list
        )
        for assessments_list in state.pair_assessments_by_resource.values():
            for assessment in assessments_list:
                pair_key = make_entity_pair_key(assessment.entity1, assessment.entity2)
                if pair_key in state.pair_judgments:
                    continue
                assessments_by_pair[pair_key].append(assessment)
        # Update unique pairs count
        if deps.progress:
            deps.progress["Unique pairs"].total = len(assessments_by_pair)
        # Judge each pair
        tasks = []
        for pair_key, assessments in assessments_by_pair.items():
            tasks.append(
                _judge_pair(pair_key, assessments, state, deps, opposition_map)
            )
        # Run all judgments in parallel
        if tasks:
            for coro in asyncio.as_completed(tasks):
                pair_key, judgment = await coro
                state.pair_judgments[pair_key] = judgment
        # Mark judgment phase complete
        if deps.progress:
            deps.progress["Unique pairs"].complete()
        return True


def _can_accept_deterministically(
    assessments: list[PairAssessment], opposition_map: dict[str, set[str]]
) -> tuple[bool, str, str]:
    """Check if we can accept without LLM call."""
    high_conf = [a for a in assessments if a.evidence.overall >= 7]
    if len(high_conf) >= 2:
        relationships = {a.relationship for a in high_conf}
        if len(relationships) == 1:
            relationship = high_conf[0].relationship
            reasoning = _build_consensus_reasoning(high_conf, relationship)
            return (True, relationship, reasoning)
        # Check for opposing relationships
        has_opposition = False
        if opposition_map:
            has_opposition = any(
                are_relationships_opposed(rel1, rel2, opposition_map)
                for rel1, rel2 in combinations(relationships, 2)
            )
        if not has_opposition:
            most_common_rel = Counter(a.relationship for a in high_conf).most_common(1)[
                0
            ][0]
            reasoning = _build_consensus_reasoning(
                high_conf, most_common_rel, multiple_relationships=True
            )
            return (True, most_common_rel, reasoning)
    return (False, "", "")


def _build_consensus_reasoning(
    high_conf_assessments: list[PairAssessment],
    selected_relationship: str,
    multiple_relationships: bool = False,
) -> str:
    """Build detailed reasoning for deterministic consensus cases."""
    by_rel: dict[str, set[str]] = defaultdict(set)
    for a in high_conf_assessments:
        by_rel[a.relationship].add(a.resource_id.id)
    if multiple_relationships:
        lines = [
            f"High-confidence consensus across {len(high_conf_assessments)} sources "
            f"with compatible relationships.\n"
        ]
        all_docs = set()
        for docs in by_rel.values():
            all_docs.update(docs)
        doc_rels: dict[str, set[str]] = defaultdict(set)
        for rel, docs in by_rel.items():
            for doc in docs:
                doc_rels[doc].add(rel)
        rel_groups: dict[frozenset[str], list[str]] = defaultdict(list)
        for doc, rels in doc_rels.items():
            rel_groups[frozenset(rels)].append(doc)
        for rel_set in sorted(rel_groups.keys(), key=lambda rs: (-len(rs), sorted(rs))):
            docs = sorted(rel_groups[rel_set])
            doc_citations = " ".join(f"[{doc}]" for doc in docs)
            if len(rel_set) == 1:
                rel_name = next(iter(rel_set))
                lines.append(
                    f"Found {len(docs)} document{'s' if len(docs) != 1 else ''} "
                    f"that support{'s' if len(docs) == 1 else ''} the relationship '{rel_name}': {doc_citations}"
                )
            else:
                rel_list_local = ", ".join(f"'{r}'" for r in sorted(rel_set))
                lines.append(
                    f"Found {len(docs)} document{'s' if len(docs) != 1 else ''} "
                    f"that support{'s' if len(docs) == 1 else ''} the relationships {rel_list_local}: {doc_citations}"
                )
        return "\n".join(lines) + "."
    else:
        doc_ids = sorted(by_rel[selected_relationship])
        doc_citations = " ".join(f"[{doc}]" for doc in doc_ids)
        return (
            f"High-confidence consensus for '{selected_relationship}' across "
            f"{len(doc_ids)} source{'s' if len(doc_ids) != 1 else ''}: {doc_citations}."
        )


def _sort_assessments_by_date(
    assessments: list[PairAssessment], deps: Deps
) -> list[PairAssessment]:
    """Sort assessments by publication date (newest first), then quote count."""

    def sort_key(a: PairAssessment) -> tuple:
        resource = deps.resource_pool.get(a.resource_id)
        date = resource.publication_date if resource else ""
        return (date or "", -len(a.quotes))

    return sorted(assessments, key=sort_key, reverse=True)


def _build_contentious_prompt(
    pair_key: EntityPairKey, spread: PairSpread, state: State, deps: Deps
) -> str:
    """Build prompt for contentious pairs (supporting + refuting evidence)."""
    padding = getattr(deps.config.tools.extraction, "region_padding_chunks", 1)

    def format_assessments(assessments: list[PairAssessment], label: str) -> str:
        if not assessments:
            return ""
        sorted_assessments = _sort_assessments_by_date(assessments, deps)
        sections = [f"## {label.title()} Evidence\n"]
        for assessment in sorted_assessments:
            resource = deps.resource_pool.get(assessment.resource_id)
            if not resource:
                continue
            text = collect_relevant_text_for_quotes(
                resource, assessment.quotes, padding
            )
            adjusted_text = adjust_heading_levels(text, target_min_level=4)
            doc_id = assessment.resource_id.id
            ev = assessment.evidence
            section = f"""### Document extract [{doc_id}]: {resource.title}
**Relationship:** {assessment.relationship} | **Evidence level:** {ev.overall}/9
**Factors:** {ev.directness}, {ev.source_type}, {ev.specificity}, {ev.language}
**Reasoning:** {assessment.reasoning}

{adjusted_text}"""
            sections.append(section)
        return "\n\n".join(sections)

    positive_text = format_assessments(spread.positive, "Positive")
    negative_text = format_assessments(spread.negative, "Negative")
    neutral_text = format_assessments(spread.neutral, "Neutral")
    all_rels = {
        a.relationship for a in spread.positive + spread.negative + spread.neutral
    }
    relationships_str = ", ".join(f'"{r}"' for r in sorted(all_rels))
    return f"""# Context
Synthesize contradictory evidence for an entity association.

**Topic:** {state.topic}

**Pair:** {pair_key.entity1_name} <-> {pair_key.entity2_name}

**Relationship types found:** {relationships_str}

**Note:** This is a CONTENTIOUS pair with contradictory evidence. This may indicate either:
- Conflicting biological polarities (positive effects vs negative effects)
- Opposing relationships (e.g., "activates" vs "inhibits")
- Context-dependent effects that appear contradictory

# Document Extracts

{positive_text}

{negative_text}

{neutral_text if neutral_text else ""}

# Task
Synthesize the contradictory evidence, considering:
1. Is there genuine disagreement in the literature, or do studies examine different contexts?
2. What is the weight of evidence on each side?
3. Should we accept this pair despite contradictions?

Provide: accepted (true/false), relationship (selected from above), confidence (high/medium/low), and detailed reasoning explaining how you weighed the contradictions. Cite documents using their IDs in square brackets (e.g., [1_abc12345]) when referencing specific evidence."""


def _build_unidirectional_prompt(
    pair_key: EntityPairKey, spread: PairSpread, state: State, deps: Deps
) -> str:
    """Build prompt for unidirectional pairs (no positive+negative conflict)."""
    padding = getattr(deps.config.tools.extraction, "region_padding_chunks", 1)
    all_assessments = spread.positive + spread.negative + spread.neutral
    sorted_assessments = _sort_assessments_by_date(all_assessments, deps)
    document_sections = []
    for assessment in sorted_assessments:
        resource = deps.resource_pool.get(assessment.resource_id)
        if not resource:
            continue
        text = collect_relevant_text_for_quotes(resource, assessment.quotes, padding)
        adjusted_text = adjust_heading_levels(text, target_min_level=3)
        doc_id = assessment.resource_id.id
        ev = assessment.evidence
        section = f"""## Document extract [{doc_id}]: {resource.title}
**Assessment:** level {ev.overall}/9 - {assessment.relationship}
**Factors:** {ev.directness}, {ev.source_type}, {ev.specificity}, {ev.language}
**Reasoning:** {assessment.reasoning}

{adjusted_text}"""
        document_sections.append(section)
    relationships = {a.relationship for a in all_assessments}
    relationships_str = ", ".join(f'"{r}"' for r in sorted(relationships))
    return f"""# Context
Make a final judgment on an entity association.

**Topic:** {state.topic}

**Pair:** {pair_key.entity1_name} <-> {pair_key.entity2_name}

**Relationship types found across documents:** {relationships_str}

# Document Extracts

{chr(10).join(document_sections)}

# Task
Synthesize the evidence across documents, considering consistency, quality, and contradictions.
Select the most accurate relationship overall (from the ones found above).

Provide: accepted (true/false), relationship (selected label), synthesized evidence quality factors, and detailed reasoning. Cite documents using their IDs in square brackets (e.g., [1_abc12345]) when referencing specific evidence."""


def _is_contentious_pair(
    assessments: list[PairAssessment],
    spread: PairSpread,
    opposition_map: dict[str, set[str]],
) -> bool:
    """Detect if a pair has contradictory evidence requiring special handling."""
    if spread.positive and spread.negative:
        return True
    if not opposition_map:
        return False
    relationships = [a.relationship for a in assessments]
    return any(
        are_relationships_opposed(rel1, rel2, opposition_map)
        for rel1, rel2 in combinations(relationships, 2)
    )


async def _judge_pair(
    pair_key: EntityPairKey,
    assessments: list[PairAssessment],
    state: State,
    deps: Deps,
    opposition_map: dict[str, set[str]],
) -> tuple[EntityPairKey, PairJudgment]:
    """Make final judgment on a single pair."""
    try:
        # Build PairSpread by polarity
        spread = build_pair_spread(assessments, state.relationship_polarities)
        # Try deterministic accept
        can_accept, relationship, reasoning = _can_accept_deterministically(
            assessments, opposition_map
        )
        if can_accept:
            if deps.progress:
                deps.progress["Unique pairs"].work()
            first_assessment = assessments[0]
            strong = [a for a in assessments if a.evidence.overall >= 7]
            evidence = aggregate_evidence(strong)
            decision_confidence = 0.95 if len(strong) >= 3 else 0.85
            judgment = PairJudgment(
                entity1=SimpleEntity(
                    name=first_assessment.entity1.canonical,
                    kind=first_assessment.entity1.kind,
                    aliases=get_entity_aliases(
                        first_assessment.entity1.canonical, state
                    ),
                ),
                entity2=SimpleEntity(
                    name=first_assessment.entity2.canonical,
                    kind=first_assessment.entity2.kind,
                    aliases=get_entity_aliases(
                        first_assessment.entity2.canonical, state
                    ),
                ),
                relationship=relationship,
                spread=spread,
                accepted=True,
                evidence=evidence,
                decision_confidence=decision_confidence,
                reasoning=reasoning,
            )
            return (pair_key, judgment)
        # Need LLM investigation
        valid_doc_ids = {a.resource_id.id for a in assessments}
        is_contentious = _is_contentious_pair(assessments, spread, opposition_map)
        if is_contentious:
            prompt = _build_contentious_prompt(pair_key, spread, state, deps)
        else:
            prompt = _build_unidirectional_prompt(pair_key, spread, state, deps)
        agent = get_cross_document_judge_agent(deps.config)
        usage = RunUsage()
        try:
            with rename_agent(
                agent,
                name=f"judge_cross_document: {pair_key.entity1_name} ⇌ {pair_key.entity2_name}",
            ):
                async with deps.agent_semaphore:
                    if deps.progress:
                        deps.progress["Unique pairs"].work()
                    result = await agent.run(prompt, deps=deps, usage=usage)
            cited_ids = extract_document_citations(result.output.reasoning)
            _, invalid_citations = validate_document_citations(cited_ids, valid_doc_ids)
            if invalid_citations:
                deps.logger.warning(
                    f"Invalid document citations in reasoning for {pair_key}: "
                    f"{invalid_citations}"
                )
            relationship = result.output.relationship
            accepted = result.output.accepted
            evidence = result.output.evidence
            decision_confidence = result.output.decision_confidence
            reasoning = result.output.reasoning
        except (TimeoutError, ConnectionError, ValueError, ModelHTTPError) as e:
            deps.logger.error(
                f"Cross-document judgment failed for {pair_key}: "
                f"{type(e).__name__}: {e}"
            )
            if deps.progress:
                deps.progress["Unique pairs"].work()
            evidence = aggregate_evidence(assessments)
            accepted = evidence.overall >= 5
            relationship = Counter(a.relationship for a in assessments).most_common(1)[
                0
            ][0]
            decision_confidence = 0.5
            decision_word = "accepted" if accepted else "rejected"
            reasoning = (
                f"LLM judgment unavailable ({type(e).__name__}); "
                f"decision based on {len(assessments)} per-document assessments. "
                f"Median evidence level {evidence.overall}/9 "
                f"(threshold 5) → {decision_word}."
            )
        # Build judgment
        first_assessment = assessments[0]
        judgment = PairJudgment(
            entity1=SimpleEntity(
                name=first_assessment.entity1.canonical,
                kind=first_assessment.entity1.kind,
                aliases=get_entity_aliases(first_assessment.entity1.canonical, state),
            ),
            entity2=SimpleEntity(
                name=first_assessment.entity2.canonical,
                kind=first_assessment.entity2.kind,
                aliases=get_entity_aliases(first_assessment.entity2.canonical, state),
            ),
            relationship=relationship,
            spread=spread,
            accepted=accepted,
            evidence=evidence,
            decision_confidence=decision_confidence,
            reasoning=reasoning,
        )
        return (pair_key, judgment)
    finally:
        if deps.progress:
            if "judgment" in locals():
                if judgment.accepted:
                    deps.progress["Accepted"].add()
                else:
                    deps.progress["Rejected"].add()
            deps.progress["Unique pairs"].done()
