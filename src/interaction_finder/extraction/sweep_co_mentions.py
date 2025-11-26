"""Co-mention sweep for missed entity pair evidence.

Scans documents for entity pair co-mentions that weren't captured by the initial
proximal set extraction, assesses them for relationship claims, and integrates
results into the extraction pipeline.

This module addresses gaps in recall caused by:
- Chunking boundaries splitting co-occurring entities
- Conservative extraction in the initial proximal pair phase
- Entities mentioned together but not in identified proximal sets
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai.usage import RunUsage

from interaction_finder.agent_config import agent_getter
from interaction_finder.agent_utils import rename_agent
from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.models import (
    EntityMention,
    EntityPairKey,
    PairAssessment,
)
from interaction_finder.extraction.utils import (
    adjust_heading_levels,
    find_best_entity_match,
    make_entity_pair_key,
)
from interaction_finder.resources import (
    QuoteValidationError,
    Resource,
    ResourceId,
    ResourceQuote,
)
from interaction_finder.settings import IfetcherConfig
from interaction_finder.text_mapping import NormalizedTextMapper


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class NovelCoMention:
    """A co-mention of an entity pair that needs assessment.

    Attributes:
        resource_id: Document containing the co-mention
        pair_key: Canonical entity pair identifier
        chunk_range: (start_chunk, end_chunk) spanning both entity mentions
        priority: Category for diagnostics - "no_existing_assessment" means the pair
            has no assessment in this document, "uncovered_region" means the pair
            has assessments but not covering this region
        entity1_pos: Character position of first entity in original text
        entity2_pos: Character position of second entity in original text
        entity1_matched_form: The alias/name that matched for entity1
        entity2_matched_form: The alias/name that matched for entity2
    """

    resource_id: ResourceId
    pair_key: EntityPairKey
    chunk_range: tuple[int, int]
    priority: Literal["no_existing_assessment", "uncovered_region"]
    entity1_pos: int
    entity2_pos: int
    entity1_matched_form: str
    entity2_matched_form: str


@dataclass
class CoMentionSweepStats:
    """Diagnostic statistics from the co-mention sweep.

    Tracks discovery and assessment outcomes to inform future optimizations.
    """

    # Discovery counts
    total_co_mentions_found: int = 0
    co_mentions_no_existing_assessment: int = 0
    co_mentions_uncovered_region: int = 0
    # Batching counts
    regions_created: int = 0
    # Assessment outcomes
    assessed: int = 0
    relationships_found: int = 0
    no_relationship_claim: int = 0


@dataclass
class CandidatePair:
    """Single entity pair candidate within a merged region.

    Attributes:
        pair_key: Canonical entity pair identifier
        entity1_kind: Kind of first entity (gene, disease, etc.)
        entity2_kind: Kind of second entity
    """

    pair_key: EntityPairKey
    entity1_kind: str
    entity2_kind: str


@dataclass
class CoMentionRegion:
    """Merged region containing multiple candidate pairs for batch assessment.

    Created by merging overlapping/adjacent chunk ranges from NovelCoMention
    objects within the same document. Enables assessing multiple pairs with
    a single LLM call.

    Attributes:
        resource_id: Document containing the region
        chunk_range: (start_chunk, end_chunk) covering all merged co-mentions
        candidate_pairs: Unique pairs to assess in this region
    """

    resource_id: ResourceId
    chunk_range: tuple[int, int]
    candidate_pairs: list[CandidatePair]


# =============================================================================
# Pure Functions
# =============================================================================


def collect_global_aliases(
    validated_entities_by_resource: dict[ResourceId, dict[str, EntityMention]],
) -> dict[str, set[str]]:
    """Build canonical_name → all_aliases mapping across all resources.

    Parameters:
        validated_entities_by_resource: Entities extracted from each resource

    Returns:
        Dict mapping each canonical entity name to all known aliases
    """
    global_aliases: dict[str, set[str]] = defaultdict(set)
    for entities in validated_entities_by_resource.values():
        for name, entity in entities.items():
            global_aliases[name].update(entity.aliases)
    return dict(global_aliases)


def build_entity_search_pattern(name: str, aliases: set[str]) -> re.Pattern:
    """Build regex matching entity name or any alias, with optional plural.

    Parameters:
        name: Canonical entity name
        aliases: All known aliases for this entity

    Returns:
        Compiled regex pattern for matching in normalized text
    """
    terms = [name] + list(aliases)
    normalized = [NormalizedTextMapper.normalize(t) for t in terms]
    # Remove empty strings, deduplicate while preserving order
    unique = list(dict.fromkeys(t for t in normalized if t))
    if not unique:
        # Fallback to matching nothing if all terms normalize to empty
        return re.compile(r"(?!)")
    escaped = [re.escape(t) for t in unique]
    pattern = r"\b(?:" + "|".join(escaped) + r")(?:s|es)?\b"
    # Both pattern terms and resource.normalized_text are lowercase
    return re.compile(pattern)


def find_entity_mentions(
    resource: Resource, pattern: re.Pattern
) -> list[tuple[int, int, str]]:
    """Find all matches of entity pattern in normalized text.

    Parameters:
        resource: Document to search
        pattern: Compiled regex from build_entity_search_pattern

    Returns:
        List of (start, end, matched_text) tuples in normalized text
    """
    return [
        (m.start(), m.end(), m.group())
        for m in pattern.finditer(resource.normalized_text)
    ]


def map_normalized_pos_to_original(resource: Resource, norm_pos: int) -> int | None:
    """Map a position in normalized text to original text position.

    Parameters:
        resource: Document with position mapper
        norm_pos: Character position in normalized text

    Returns:
        Corresponding position in original text, or None if mapping fails
    """
    try:
        orig_start, _ = resource.map_normalized_to_original_position(norm_pos, 1)
        return orig_start
    except (ValueError, RuntimeError):
        return None


def is_co_mention_covered(
    pos_a: int,
    pos_b: int,
    pair_assessments: list[PairAssessment],
) -> bool:
    """Check if both positions are within an existing quote span.

    Parameters:
        pos_a: First entity position in original text
        pos_b: Second entity position in original text
        pair_assessments: Existing assessments for this specific pair

    Returns:
        True if both positions fall within the same quote span (bounds inclusive)
    """
    for assessment in pair_assessments:
        for quote in assessment.quotes:
            for span_start, span_end in quote.spans:
                if span_start <= pos_a <= span_end and span_start <= pos_b <= span_end:
                    return True
    return False


def classify_co_mention(
    pos_a: int,
    pos_b: int,
    pair_key: EntityPairKey,
    resource_id: ResourceId,
    assessments_by_resource: dict[ResourceId, list[PairAssessment]],
) -> tuple[bool, Literal["no_existing_assessment", "uncovered_region"]]:
    """Determine if co-mention needs assessment and its priority category.

    Parameters:
        pos_a: First entity position in original text
        pos_b: Second entity position in original text
        pair_key: Canonical entity pair identifier
        resource_id: Document containing the co-mention
        assessments_by_resource: All existing pair assessments

    Returns:
        (is_novel, priority) tuple where is_novel indicates assessment is needed
    """
    assessments = assessments_by_resource.get(resource_id, [])
    pair_assessments = [
        a for a in assessments if make_entity_pair_key(a.entity1, a.entity2) == pair_key
    ]
    # No existing assessment for this pair in this document
    if not pair_assessments:
        return (True, "no_existing_assessment")
    # Has assessment(s) - check if positions are covered
    if is_co_mention_covered(pos_a, pos_b, pair_assessments):
        return (False, "uncovered_region")
    return (True, "uncovered_region")


def find_novel_co_mentions_in_resource(
    resource: Resource,
    resource_id: ResourceId,
    assessed_pairs: set[EntityPairKey],
    entity_patterns: dict[str, re.Pattern],
    assessments_by_resource: dict[ResourceId, list[PairAssessment]],
    chunk_distance: int,
) -> list[NovelCoMention]:
    """Find all novel co-mentions of assessed pairs in a single resource.

    Parameters:
        resource: Document to scan
        resource_id: Document identifier
        assessed_pairs: All entity pairs that have been assessed somewhere
        entity_patterns: Precompiled search patterns for each entity
        assessments_by_resource: Existing assessments for coverage checking
        chunk_distance: Maximum chunk distance for co-mentions

    Returns:
        List of novel co-mentions needing assessment
    """
    # Find all entity mentions in this resource
    # entity_name → [(original_position, matched_form)]
    entity_mentions: dict[str, list[tuple[int, str]]] = {}
    for entity_name, pattern in entity_patterns.items():
        matches = find_entity_mentions(resource, pattern)
        position_forms = []
        for norm_start, _, matched_form in matches:
            orig_pos = map_normalized_pos_to_original(resource, norm_start)
            if orig_pos is not None:
                position_forms.append((orig_pos, matched_form))
        if position_forms:
            entity_mentions[entity_name] = position_forms
    # Find novel co-mentions for each assessed pair
    novel_co_mentions: list[NovelCoMention] = []
    seen_chunk_ranges: dict[EntityPairKey, set[tuple[int, int]]] = defaultdict(set)
    for pair_key in assessed_pairs:
        mentions_a = entity_mentions.get(pair_key.entity1_name, [])
        mentions_b = entity_mentions.get(pair_key.entity2_name, [])
        if not mentions_a or not mentions_b:
            continue
        for pos_a, form_a in mentions_a:
            for pos_b, form_b in mentions_b:
                # Skip self-mentions (same position)
                if pos_a == pos_b:
                    continue
                # Check chunk distance
                chunk_a = resource.get_chunk_for_position(pos_a)
                chunk_b = resource.get_chunk_for_position(pos_b)
                if chunk_a is None or chunk_b is None:
                    continue
                if abs(chunk_a - chunk_b) > chunk_distance:
                    continue
                # Check if novel
                is_novel, priority = classify_co_mention(
                    pos_a, pos_b, pair_key, resource_id, assessments_by_resource
                )
                if not is_novel:
                    continue
                # Deduplicate by chunk range
                chunk_range = (min(chunk_a, chunk_b), max(chunk_a, chunk_b))
                if chunk_range in seen_chunk_ranges[pair_key]:
                    continue
                seen_chunk_ranges[pair_key].add(chunk_range)
                novel_co_mentions.append(
                    NovelCoMention(
                        resource_id=resource_id,
                        pair_key=pair_key,
                        chunk_range=chunk_range,
                        priority=priority,
                        entity1_pos=pos_a,
                        entity2_pos=pos_b,
                        entity1_matched_form=form_a,
                        entity2_matched_form=form_b,
                    )
                )
    return novel_co_mentions


def collect_assessed_pairs(
    assessments_by_resource: dict[ResourceId, list[PairAssessment]],
) -> set[EntityPairKey]:
    """Collect all unique entity pairs that have been assessed.

    Parameters:
        assessments_by_resource: All existing pair assessments

    Returns:
        Set of unique EntityPairKey objects
    """
    pairs = set()
    for assessments in assessments_by_resource.values():
        for assessment in assessments:
            pairs.add(make_entity_pair_key(assessment.entity1, assessment.entity2))
    return pairs


def select_co_mentions_to_assess(
    co_mentions: list[NovelCoMention],
    config: IfetcherConfig,
) -> list[NovelCoMention]:
    """Policy function for selecting which co-mentions to assess.

    Currently returns all co-mentions. This function exists as an extension
    point for future filtering/prioritization strategies.

    Parameters:
        co_mentions: All discovered novel co-mentions
        config: Configuration (for future cap/filter settings)

    Returns:
        Co-mentions to assess
    """
    return co_mentions


def merge_co_mentions_into_regions(
    co_mentions: list[NovelCoMention],
    entity_kinds: dict[str, str],
) -> list[CoMentionRegion]:
    """Merge adjacent co-mentions into batched regions for efficient assessment.

    Groups co-mentions by document and merges overlapping/adjacent chunk ranges,
    then collects unique entity pairs for each merged region. This enables
    assessing multiple pairs with a single LLM call per region.

    Parameters:
        co_mentions: All novel co-mentions discovered
        entity_kinds: Mapping from entity canonical name to kind

    Returns:
        List of CoMentionRegion with merged ranges and deduplicated pairs
    """
    if not co_mentions:
        return []
    # Group by resource_id
    by_resource: dict[ResourceId, list[NovelCoMention]] = defaultdict(list)
    for cm in co_mentions:
        by_resource[cm.resource_id].append(cm)
    regions: list[CoMentionRegion] = []
    for resource_id, resource_cms in by_resource.items():
        # Sort by chunk_range start
        sorted_cms = sorted(resource_cms, key=lambda cm: cm.chunk_range[0])
        # Merge adjacent ranges (gap <= 1)
        current_start, current_end = sorted_cms[0].chunk_range
        current_pair_keys: set[EntityPairKey] = {sorted_cms[0].pair_key}
        for cm in sorted_cms[1:]:
            cm_start, cm_end = cm.chunk_range
            if cm_start <= current_end + 1:
                # Adjacent or overlapping - extend current region
                current_end = max(current_end, cm_end)
                current_pair_keys.add(cm.pair_key)
            else:
                # Gap too large - finalize current region and start new one
                candidate_pairs = [
                    CandidatePair(
                        pair_key=pk,
                        entity1_kind=entity_kinds.get(pk.entity1_name, "entity"),
                        entity2_kind=entity_kinds.get(pk.entity2_name, "entity"),
                    )
                    for pk in current_pair_keys
                ]
                regions.append(
                    CoMentionRegion(
                        resource_id=resource_id,
                        chunk_range=(current_start, current_end),
                        candidate_pairs=candidate_pairs,
                    )
                )
                current_start, current_end = cm_start, cm_end
                current_pair_keys = {cm.pair_key}
        # Finalize last region
        candidate_pairs = [
            CandidatePair(
                pair_key=pk,
                entity1_kind=entity_kinds.get(pk.entity1_name, "entity"),
                entity2_kind=entity_kinds.get(pk.entity2_name, "entity"),
            )
            for pk in current_pair_keys
        ]
        regions.append(
            CoMentionRegion(
                resource_id=resource_id,
                chunk_range=(current_start, current_end),
                candidate_pairs=candidate_pairs,
            )
        )
    return regions


# =============================================================================
# LLM Agent (Batch Assessment)
# =============================================================================


class ConfirmedPair(BaseModel):
    """A confirmed relationship between an entity pair."""

    entity1_name: str = Field(description="Canonical name of first entity")
    entity2_name: str = Field(description="Canonical name of second entity")
    relationship: str = Field(
        description="Relationship type (e.g., 'activates', 'inhibits', 'associated_with')"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in the relationship claim"
    )
    supporting_quotes: list[str] = Field(
        description="Exact verbatim quotes from text supporting this relationship"
    )
    reasoning: str = Field(
        description="Brief explanation of why this relationship exists"
    )


class RegionAssessmentOut(BaseModel):
    """LLM output for batch co-mention region assessment."""

    confirmed_pairs: list[ConfirmedPair] = Field(
        default_factory=list,
        description="Only pairs where a relationship was found. "
        "Pairs not listed are implicitly rejected (no relationship or invalid entities).",
    )


get_region_assessment_agent = agent_getter(
    "extraction",
    "co_mention_region",
    RegionAssessmentOut,
    Deps,
    """Analyze co-occurring entity mentions and assess their relationships.

**Task:**
You will be given a text region and a list of candidate entity pairs to evaluate.
For each pair:
1. Verify that both entity mentions in the text actually refer to the specified canonical entities (not similarly-named entities or false matches)
2. If both entities are valid and relevant to the research topic, determine if the text makes any claim about their relationship

Only include a pair in confirmed_pairs if BOTH conditions are met:
- Both entity mentions are valid (refer to the specified entities and are topic-relevant)
- The text states or strongly implies a relationship between them

**Entity validation:**
Check whether each mention refers to the specified canonical entity:
- "BRCA1" vs "BRCA1-like protein" are different entities
- An entity mentioned in an unrelated context should not be included
- If either entity in a pair fails validation, do not include the pair

**Relationship types:**
Use one of the known relationship types listed in the prompt when possible.
If none fit, use a concise descriptive label (e.g., "activates", "inhibits", "associated_with").

**Quality standards:**
- Be conservative: only report relationships that are clearly stated or strongly implied
- Co-occurrence alone is NOT sufficient - there must be a stated connection
- The relationship must be about these specific entities, not general statements
- Provide exact verbatim quotes from the text, not paraphrases
- If no pairs meet the criteria, return an empty confirmed_pairs list""",
)


# =============================================================================
# Assessment Function
# =============================================================================


def create_minimal_entity_mention(
    canonical_name: str,
    kind: str,
    matched_form: str,
    quotes: list[ResourceQuote],
    reasoning: str,
) -> EntityMention:
    """Create minimal EntityMention for sweep-discovered entity.

    Used when an entity is found via pattern matching but wasn't in the
    document's validated entity set.

    Parameters:
        canonical_name: The canonical entity name
        kind: Entity type (gene, disease, etc.)
        matched_form: The alias form that matched in the text
        quotes: Supporting quotes from the assessment
        reasoning: Entity-specific reasoning from LLM

    Returns:
        A minimal EntityMention with the matched form as alias
    """
    return EntityMention(
        kind=kind,
        name=canonical_name,
        aliases=[matched_form] if matched_form != canonical_name.lower() else [],
        quotes=quotes,
        reasoning=reasoning,
    )


def get_entity_kind(
    canonical_name: str,
    validated_entities_by_resource: dict[ResourceId, dict[str, EntityMention]],
) -> str | None:
    """Get entity kind from any document's validated entities.

    Parameters:
        canonical_name: The canonical entity name
        validated_entities_by_resource: All validated entities by resource

    Returns:
        Entity kind if found, None otherwise
    """
    for entities in validated_entities_by_resource.values():
        if canonical_name in entities:
            return entities[canonical_name].kind
    return None


async def assess_co_mention_region(
    region: CoMentionRegion,
    resource: Resource,
    topic: str,
    known_relationships: list[str],
    config: IfetcherConfig,
    deps: Deps,
    validated_entities: dict[str, EntityMention] | None = None,
) -> list[PairAssessment]:
    """Assess all candidate pairs in a merged region with a single LLM call.

    Parameters:
        region: Merged region with multiple candidate pairs
        resource: Source document
        topic: Research topic for context
        known_relationships: Relationship types already seen in this extraction
        config: Configuration for LLM agent
        deps: Pipeline dependencies
        validated_entities: Document's validated entities (for reuse if available)

    Returns:
        List of PairAssessment for confirmed relationships (may be empty)
    """
    if not region.candidate_pairs:
        return []
    # Build text region from chunk range with padding
    padding = config.tools.extraction.region_padding_chunks
    start_chunk = max(0, region.chunk_range[0] - padding)
    end_chunk = min(len(resource.chunks) - 1, region.chunk_range[1] + padding)
    text_parts = []
    for chunk_idx in range(start_chunk, end_chunk + 1):
        chunk_text = resource.get_chunk_text(chunk_idx)
        if chunk_text:
            text_parts.append(chunk_text)
    text_region = "\n".join(text_parts)
    if not text_region.strip():
        return []
    # Build candidate pairs list for prompt and lookup structures
    pairs_list = []
    pair_lookup: dict[tuple[str, str], CandidatePair] = {}
    all_entity_names: set[str] = set()
    for candidate in region.candidate_pairs:
        pk = candidate.pair_key
        pairs_list.append(
            f"- {pk.entity1_name} ({candidate.entity1_kind}) <-> {pk.entity2_name} ({candidate.entity2_kind})"
        )
        pair_lookup[(pk.entity1_name, pk.entity2_name)] = candidate
        pair_lookup[(pk.entity2_name, pk.entity1_name)] = candidate
        all_entity_names.add(pk.entity1_name)
        all_entity_names.add(pk.entity2_name)
    entity_name_list = list(all_entity_names)
    # Build relationships section
    if known_relationships:
        relationships_section = (
            f"\n**Known relationship types:**\n{', '.join(known_relationships)}\n"
        )
    else:
        relationships_section = ""
    # Build prompt
    adjusted_text_region = adjust_heading_levels(text_region, target_min_level=2)
    prompt = f"""# Context
Evaluate candidate entity pairs for associations in this text region.

**Topic:** {topic}

**Candidate pairs to evaluate:**
{chr(10).join(pairs_list)}
{relationships_section}
# Document Extract
{adjusted_text_region}

# Task
For each candidate pair, determine if there is evidence of a relationship in the text.
Provide supporting quotes for confirmed relationships."""
    # Call LLM
    usage = RunUsage()
    agent = get_region_assessment_agent(config)
    try:
        with rename_agent(
            agent, name=f"AssessRegion: {len(region.candidate_pairs)} pairs"
        ):
            async with deps.agent_semaphore:
                result = await agent.run(prompt, deps=deps, usage=usage)
    except (TimeoutError, ConnectionError, ValueError) as e:
        deps.logger.error(
            f"Co-mention region assessment failed: {type(e).__name__}: {e}"
        )
        return []
    # Process confirmed pairs
    assessments: list[PairAssessment] = []
    for confirmed in result.output.confirmed_pairs:
        # Match entity names using fuzzy matching (handles spelling variants, annotations)
        matched_e1 = find_best_entity_match(confirmed.entity1_name, entity_name_list)
        matched_e2 = find_best_entity_match(confirmed.entity2_name, entity_name_list)
        if matched_e1 is None or matched_e2 is None:
            deps.logger.warning(
                f"LLM returned unknown pair: {confirmed.entity1_name} <-> {confirmed.entity2_name}"
            )
            continue
        # Look up candidate using matched names
        candidate = pair_lookup.get((matched_e1, matched_e2))
        if candidate is None:
            # Matched names don't form a valid pair (shouldn't happen often)
            deps.logger.warning(
                f"Matched entities don't form a candidate pair: {matched_e1} <-> {matched_e2}"
            )
            continue
        # Validate quotes
        validated_quotes = []
        for quote_str in confirmed.supporting_quotes:
            try:
                validated_quotes.append(resource.quote(quote_str))
            except QuoteValidationError:
                pass
        if not validated_quotes:
            deps.logger.debug(
                f"Co-mention assessment for {matched_e1}-{matched_e2} "
                "rejected: no valid quotes"
            )
            continue
        # Get or create EntityMention objects (use matched canonical names)
        if validated_entities and matched_e1 in validated_entities:
            entity1 = validated_entities[matched_e1]
        else:
            entity1 = create_minimal_entity_mention(
                canonical_name=matched_e1,
                kind=candidate.entity1_kind,
                matched_form=confirmed.entity1_name.lower(),
                quotes=validated_quotes,
                reasoning=confirmed.reasoning,
            )
        if validated_entities and matched_e2 in validated_entities:
            entity2 = validated_entities[matched_e2]
        else:
            entity2 = create_minimal_entity_mention(
                canonical_name=matched_e2,
                kind=candidate.entity2_kind,
                matched_form=confirmed.entity2_name.lower(),
                quotes=validated_quotes,
                reasoning=confirmed.reasoning,
            )
        assessments.append(
            PairAssessment(
                resource_id=region.resource_id,
                entity1=entity1,
                entity2=entity2,
                relationship=confirmed.relationship,
                quotes=validated_quotes,
                confidence=confirmed.confidence,
                reasoning=confirmed.reasoning,
                source="sweep",
            )
        )
    return assessments


# =============================================================================
# Node (imported and used in nodes.py)
# =============================================================================
# The SweepCoMentionsNode is defined in nodes.py to keep all nodes together
# and avoid circular imports. This module provides the supporting functions.
