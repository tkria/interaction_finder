"""Per-document processing pipeline functions.

Pure functions for processing a single document through the extraction pipeline:
document analysis (quality + entities) → validation → proximal set identification →
pair extraction → pair assessment.

These functions are extracted from the original node implementations to enable
concurrent per-document processing while maintaining clean separation of concerns.
"""

import asyncio

from interaction_finder.agent_config import AGENT_CALL_ERRORS
from interaction_finder.agent_utils import rename_agent
from interaction_finder.usage import record_usage
from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.extract import get_document_analysis_agent
from interaction_finder.extraction.extract_proximal_pairs import get_proximal_pair_agent
from interaction_finder.extraction.judge_pair_evidence import get_pair_judge_agent
from interaction_finder.extraction.models import (
    EntityMention,
    EntityPairKey,
    EntityRef,
    PairAssessment,
    PaperQualityAssessment,
    ProximalEntitySet,
)
from interaction_finder.extraction.entity_matching import (
    extract_entity_variants,
    find_entity_match,
)
from interaction_finder.extraction.utils import (
    adjust_heading_levels,
    build_text_region,
    collect_relevant_text_for_quotes,
    make_entity_pair_key,
    normalize_for_comparison,
    strip_kind_annotation,
)
from interaction_finder.resources import QuoteValidationError, Resource
from interaction_finder.settings import IfetcherConfig


# Document analysis (combined quality assessment + entity extraction)


def _process_entity_extractions(
    entity_infos: list,
    resource: Resource,
) -> tuple[dict[str, EntityMention], int, int]:
    """Convert LLM entity output to validated EntityMentions.

    Groups entities by normalized name (to merge case variants like "PAH"/"pah"),
    validates quotes against the source document, and returns the processed entities.

    Returns:
        (entities_dict, quotes_validated, quotes_failed)
    """
    # Group by normalized name to merge case variants
    by_normalized: dict[str, list] = {}
    for info in entity_infos:
        key = normalize_for_comparison(strip_kind_annotation(info.name))
        by_normalized.setdefault(key, []).append(info)
    # Convert each group to EntityMention with validated quotes
    entities = {}
    validated, failed = 0, 0
    for infos in by_normalized.values():
        # Validate quotes from all instances
        quotes = []
        for info in infos:
            for q in info.quotes:
                try:
                    quotes.append(resource.quote(q))
                    validated += 1
                except QuoteValidationError:
                    failed += 1
        if not quotes:
            continue  # Skip entities with no valid quotes
        # Use first entity's name (preserves original casing)
        canonical = strip_kind_annotation(infos[0].name)
        # Collect unique aliases
        seen = {normalize_for_comparison(canonical)}
        aliases = []
        for info in infos:
            for name in [strip_kind_annotation(info.name)] + info.aliases:
                norm = normalize_for_comparison(name)
                if norm not in seen and name != canonical:
                    aliases.append(name)
                    seen.add(norm)
        entities[canonical] = EntityMention(
            kind=infos[0].kind,
            name=canonical,
            aliases=aliases,
            quotes=quotes,
            reasoning=" | ".join(i.reasoning for i in infos),
        )
    return entities, validated, failed


async def analyze_document(
    resource: Resource,
    topic: str,
    target_entity_types: list[str],
    config: IfetcherConfig,
    deps: Deps,
) -> tuple[dict[str, EntityMention], PaperQualityAssessment | None, int, int]:
    """Analyze a document: assess quality and extract entities in a single pass.

    Returns (entities, paper_quality, quotes_validated, quotes_failed).
    """
    document_text = adjust_heading_levels(resource.text, target_min_level=2)
    prompt = f"""# Document
**Title:** {resource.title}

{document_text}

# Task
1. First, assess paper quality using the seven-dimension rubric
2. Then, extract entities relevant to: {topic}

**Target entity types:** {", ".join(target_entity_types)}

For each entity of the specified types relevant to the topic, provide:
- Canonical name
- All verbatim name variants from text
- Supporting quotes
- Reasoning for inclusion"""
    agent = get_document_analysis_agent(config)
    try:
        with rename_agent(agent, name=f"AnalyzeDocument: {resource.title[:60]}"):
            async with deps.agent_semaphore:
                if deps.progress:
                    deps.progress["Processed"].work()
                result = await agent.run(prompt, deps=deps)
        record_usage(deps.usage, "document_analysis", agent, result)
    except AGENT_CALL_ERRORS as e:
        deps.logger.error(
            f"Document analysis failed for {resource.id.url}: {type(e).__name__}: {e}"
        )
        return {}, None, 0, 0
    entities, validated, failed = _process_entity_extractions(
        result.output.entities, resource
    )
    return entities, result.output.paper_quality, validated, failed


def validate_entity_kinds(
    entities: dict[str, EntityMention],
    target_entity_types: list[str],
) -> dict[str, EntityMention]:
    """Filter entities to only include target kinds.

    Parameters:
        entities: All extracted entities
        target_entity_types: Entity kinds to keep

    Returns:
        Filtered dict containing only entities with target kinds
    """
    return {
        name: entity
        for name, entity in entities.items()
        if entity.kind in target_entity_types
    }


async def extract_pairs_from_proximal_set(
    proximal_set: ProximalEntitySet,
    entities: dict[str, EntityMention],
    resource: Resource,
    topic: str,
    permitted_pairs: dict[str, set[str]],
    region_padding_chunks: int,
    config: IfetcherConfig,
    deps: Deps,
) -> tuple[list[tuple], int, int]:
    """Extract entity pairs from a single proximal set.

    Parameters:
        proximal_set: Group of co-occurring entities
        entities: All entities in document
        resource: Source document
        topic: Research topic context
        permitted_pairs: Map of allowed entity kind pairings
        region_padding_chunks: Padding for text region around proximal set
        config: Configuration for LLM agents
        deps: Pipeline dependencies

    Returns:
        (pairs, quotes_validated, quotes_failed) where:
        - pairs: List of (entity1_name, entity2_name, relationship_types, quotes) tuples
        - quotes_validated: Count of successfully validated quotes
        - quotes_failed: Count of quotes that failed validation
    """
    # Build text region spanning proximal set with padding
    chunk_start, chunk_end = proximal_set.chunk_range
    text_region = build_text_region(
        resource, chunk_start, chunk_end, region_padding_chunks
    )

    # Build entity list for prompt
    entity_list = []
    for entity_name in proximal_set.entities:
        entity = entities.get(entity_name)
        if entity:
            aliases_str = ", ".join(entity.aliases)
            entity_list.append(
                f"- **{entity_name}** [kind: {entity.kind}, aliases: {aliases_str}]"
            )

    # Build readable description for agent span name
    entity_names = sorted(proximal_set.entities)[:3]
    entities_str = ", ".join(entity_names)
    if len(proximal_set.entities) > 3:
        entities_str += f" (+{len(proximal_set.entities) - 3} more)"

    adjusted_text_region = adjust_heading_levels(text_region, target_min_level=2)
    prompt = f"""# Context
Extract entity associations from this text region.

**Topic:** {topic}

**Entities in this region:**
{chr(10).join(entity_list)}

# Document Extract
{adjusted_text_region}

# Output
For each binary association between these entities that is clearly stated or implied:
- Use the canonical entity names as shown in bold (kind and aliases are metadata only)
- Provide exact supporting quotes"""

    # Call proximal pair extractor LLM agent
    agent = get_proximal_pair_agent(config)
    try:
        with rename_agent(agent, name=f"ExtractProximalPairs: {entities_str}"):
            async with deps.agent_semaphore:
                result = await agent.run(prompt, deps=deps)
        record_usage(deps.usage, "proximal_pair", agent, result)
    except AGENT_CALL_ERRORS as e:
        deps.logger.error(f"Proximal pair extraction failed: {type(e).__name__}: {e}")
        return ([], 0, 0)

    # Process extracted pairs with validation
    pairs = []
    quotes_validated = 0
    quotes_failed = 0

    # Build variant map once for efficient fuzzy matching
    entity_variants = {
        name: extract_entity_variants(name, entities[name].aliases)
        for name in proximal_set.entities
    }

    for pair_info in result.output.pairs:
        # Match LLM-returned entity names to known entities (handles annotations, variants)
        match1 = find_entity_match(pair_info.entity1, entity_variants, allow_fuzzy=True)
        match2 = find_entity_match(pair_info.entity2, entity_variants, allow_fuzzy=True)

        # Skip if entities not found (hallucinations or no close match)
        if match1 is None or match2 is None:
            continue

        entity1 = match1.canonical
        entity2 = match2.canonical
        # Check if pair kinds are permitted by configuration
        e1_obj = entities.get(entity1)
        e2_obj = entities.get(entity2)
        if not e1_obj or not e2_obj:
            continue

        allowed_partners = permitted_pairs.get(e1_obj.kind, set())
        if e2_obj.kind not in allowed_partners:
            deps.logger.debug(
                f"Pair {entity1} ({e1_obj.kind}) - {entity2} ({e2_obj.kind}) "
                f"not permitted by kind constraints"
            )
            continue

        # Validate all supporting quotes
        quotes = []
        for quote_str in pair_info.supporting_quotes:
            try:
                quote = resource.quote(quote_str)
                quotes.append(quote)
                quotes_validated += 1
            except QuoteValidationError:
                quotes_failed += 1

        if quotes:
            # Store as tuple for later assessment
            pairs.append((entity1, entity2, pair_info.relationship_types, quotes))

    return (pairs, quotes_validated, quotes_failed)


async def extract_document_pairs(
    proximal_sets: list[ProximalEntitySet],
    entities: dict[str, EntityMention],
    resource: Resource,
    topic: str,
    permitted_pairs: dict[str, set[str]],
    region_padding_chunks: int,
    config: IfetcherConfig,
    deps: Deps,
) -> tuple[list[tuple], int, int]:
    """Extract pairs from all proximal sets in a document.

    Parameters:
        proximal_sets: All co-occurring entity groups in document
        entities: All entities in document
        resource: Source document
        topic: Research topic context
        permitted_pairs: Map of allowed entity kind pairings
        region_padding_chunks: Padding for text regions
        config: Configuration for LLM agents
        deps: Pipeline dependencies

    Returns:
        (pairs, quotes_validated, quotes_failed) where:
        - pairs: List of (entity1_name, entity2_name, relationship_types, quotes) tuples
        - quotes_validated: Total successfully validated quotes
        - quotes_failed: Total quotes that failed validation
    """
    # Process all proximal sets concurrently
    tasks = [
        extract_pairs_from_proximal_set(
            proximal_set,
            entities,
            resource,
            topic,
            permitted_pairs,
            region_padding_chunks,
            config,
            deps,
        )
        for proximal_set in proximal_sets
    ]

    results = await asyncio.gather(*tasks)

    # Combine results from all proximal sets
    all_pairs = []
    total_validated = 0
    total_failed = 0

    for pairs, validated, failed in results:
        all_pairs.extend(pairs)
        total_validated += validated
        total_failed += failed

    return (all_pairs, total_validated, total_failed)


# Pair assessment


async def assess_single_pair(
    entity1: EntityRef,
    entity2: EntityRef,
    relationship_candidates: set[str],
    quotes: list,
    resource: Resource,
    topic: str,
    region_padding_chunks: int,
    config: IfetcherConfig,
    deps: Deps,
) -> PairAssessment | None:
    """Assess evidence for a single entity pair in a document.

    Parameters:
        entity1: First entity reference in pair
        entity2: Second entity reference in pair
        relationship_candidates: Possible relationship types from extraction
        quotes: Supporting quotes for this pair
        resource: Source document
        topic: Research topic context
        region_padding_chunks: Padding for text regions
        config: Configuration for LLM agents
        deps: Pipeline dependencies

    Returns:
        PairAssessment with evidence quality and reasoning, or None on error
    """
    # Build relevant text region around all quotes
    text_region = collect_relevant_text_for_quotes(
        resource, quotes, region_padding_chunks
    )

    # Build quote list for prompt
    quotes_str = "\n".join(f"[{i}] {q.get_quote_text()}" for i, q in enumerate(quotes))

    # Build relationship candidates string
    candidates = list(relationship_candidates)
    candidates_str = ", ".join(f'"{c}"' for c in candidates)

    adjusted_text_region = adjust_heading_levels(text_region, target_min_level=2)
    prompt = f"""# Context
Assess evidence for an entity association.

**Topic:** {topic}

**Pair:** {entity1.canonical} ({entity1.kind}) <-> {entity2.canonical} ({entity2.kind})

**Relationship type candidates:** {candidates_str}

**Supporting quotes:**
{quotes_str}

# Document Extract
{adjusted_text_region}

# Output
- Select the most appropriate relationship type (from candidates or propose a more specific one)
- Assess evidence quality factors and overall level
- Explain your reasoning"""

    # Call pair judge LLM agent
    agent = get_pair_judge_agent(config)
    try:
        with rename_agent(
            agent, name=f"AssessPair: {entity1.canonical} ⇌ {entity2.canonical}"
        ):
            async with deps.agent_semaphore:
                # Mark pair as in-progress now that we've acquired the semaphore
                if deps.progress:
                    deps.progress["Pairs assessed"].work()
                try:
                    result = await agent.run(prompt, deps=deps)
                finally:
                    # Mark pair as done when assessment completes or fails
                    if deps.progress:
                        deps.progress["Pairs assessed"].done()
        record_usage(deps.usage, "pair_judge", agent, result)
    except AGENT_CALL_ERRORS as e:
        deps.logger.error(
            f"Pair assessment failed for {entity1.canonical}-{entity2.canonical}: "
            f"{type(e).__name__}: {e}"
        )
        return None

    # Extract referenced quotes (LLM provides indices)
    referenced_quotes = [
        quotes[i] for i in result.output.supporting_quote_ids if i < len(quotes)
    ]

    # Create assessment with EntityRef (already in correct format)
    assessment = PairAssessment(
        resource_id=resource.id,
        entity1=entity1,
        entity2=entity2,
        relationship=result.output.relationship,
        quotes=referenced_quotes if referenced_quotes else quotes,
        evidence=result.output.evidence,
        topic_relevance=result.output.topic_relevance,
        reasoning=result.output.reasoning,
    )

    return assessment


async def assess_document_pairs(
    pairs: list[tuple],
    entities: dict[str, EntityRef],
    resource: Resource,
    topic: str,
    region_padding_chunks: int,
    config: IfetcherConfig,
    deps: Deps,
) -> list[PairAssessment]:
    """Deduplicate and assess all pairs from a document.

    Parameters:
        pairs: Raw pairs from extraction (may contain duplicates)
        entities: All entity references in document (canonical names mapped to EntityRefs)
        resource: Source document
        topic: Research topic context
        region_padding_chunks: Padding for text regions
        config: Configuration for LLM agents
        deps: Pipeline dependencies

    Returns:
        List of PairAssessment objects with evidence quality and reasoning
    """
    # Deduplicate pairs by canonical entity pair key
    # Same pair may appear in multiple proximal sets
    pairs_dict: dict[EntityPairKey, dict] = {}

    for entity1_name, entity2_name, rel_types, quotes in pairs:
        # Get entity references (already EntityRefs after validation)
        entity1_ref = entities[entity1_name]
        entity2_ref = entities[entity2_name]

        # Create canonical ordered key
        pair_key = make_entity_pair_key(entity1_ref, entity2_ref)

        if pair_key not in pairs_dict:
            # Store entity references in canonical order determined by pair_key
            ordered_entity1_ref = entities[pair_key.entity1_name]
            ordered_entity2_ref = entities[pair_key.entity2_name]

            pairs_dict[pair_key] = {
                "entity1": ordered_entity1_ref,
                "entity2": ordered_entity2_ref,
                "relationship_candidates": set(),
                "quotes": [],
            }

        # Accumulate relationship types and quotes from all occurrences
        pairs_dict[pair_key]["relationship_candidates"].update(rel_types)
        pairs_dict[pair_key]["quotes"].extend(quotes)
    # Update progress total after deduplication (actual work to be done)
    if deps.progress:
        deps.progress["Pairs assessed"].total += len(pairs_dict)
        deps.progress["Pairs assessed"].activate()
    # Assess each unique pair concurrently
    tasks = [
        assess_single_pair(
            pair_info["entity1"],
            pair_info["entity2"],
            pair_info["relationship_candidates"],
            pair_info["quotes"],
            resource,
            topic,
            region_padding_chunks,
            config,
            deps,
        )
        for pair_info in pairs_dict.values()
    ]

    assessments = await asyncio.gather(*tasks)

    # Filter out None results (assessment failures)
    return [a for a in assessments if a is not None]
