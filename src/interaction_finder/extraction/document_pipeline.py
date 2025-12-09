"""Per-document processing pipeline functions.

Pure functions for processing a single document through the extraction pipeline:
entity extraction → validation → proximal set identification → pair extraction →
pair assessment.

These functions are extracted from the original node implementations to enable
concurrent per-document processing while maintaining clean separation of concerns.
"""

import asyncio

from pydantic_ai.usage import RunUsage

from interaction_finder.agent_utils import rename_agent
from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.extract import get_entity_extractor_agent
from interaction_finder.extraction.extract_proximal_pairs import get_proximal_pair_agent
from interaction_finder.extraction.judge_pair_evidence import get_pair_judge_agent
from interaction_finder.extraction.models import (
    EntityMention,
    EntityPairKey,
    EntityRef,
    PairAssessment,
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


# Entity extraction and validation


async def extract_document_entities(
    resource: Resource,
    topic: str,
    target_entity_types: list[str],
    config: IfetcherConfig,
    deps: Deps,
) -> tuple[dict[str, EntityMention], int, int]:
    """Extract entities from a single document with quote validation.

    Parameters:
        resource: Document to process
        topic: Research topic context
        target_entity_types: Entity kinds to extract (e.g., ["gene", "disease"])
        config: Configuration for LLM agents
        deps: Pipeline dependencies (logger, etc.)

    Returns:
        (entities, quotes_validated, quotes_failed) where:
        - entities: Dict of {canonical_name: EntityMention} with validated quotes
        - quotes_validated: Count of successfully validated quotes
        - quotes_failed: Count of quotes that failed validation
    """
    usage = RunUsage()
    # Build extraction prompt
    entity_types_str = ", ".join(target_entity_types)
    document_text = adjust_heading_levels(resource.text[:15000], target_min_level=2)
    prompt = f"""# Context
Extract entities relevant to: {topic}

**Target entity types:** {entity_types_str}

# Document Extract
**Title:** {resource.title}

{document_text}

# Output
For each entity of the specified types relevant to the topic, provide:
- Canonical name
- All verbatim name variants from text
- Supporting quotes
- Reasoning for inclusion"""

    # Call entity extractor LLM agent
    agent = get_entity_extractor_agent(config)
    try:
        with rename_agent(
            agent, name=f"ExtractDocumentEntities: {resource.title[:60]}"
        ):
            async with deps.agent_semaphore:
                # Mark document as in-progress now that we've acquired the semaphore
                if deps.progress:
                    deps.progress["Processed"].work()
                result = await agent.run(prompt, deps=deps, usage=usage)
    except (TimeoutError, ConnectionError, ValueError) as e:
        deps.logger.error(
            f"Entity extraction failed for {resource.id.url}: {type(e).__name__}: {e}"
        )
        return ({}, 0, 0)

    # Convert LLM output to EntityMention objects with validated ResourceQuotes
    # Group by normalized name to merge case variants (e.g., "PAH" and "pah")
    entities_by_normalized_name: dict[str, list] = {}
    for entity_info in result.output.entities:
        # Normalize: strip kind annotation, then apply text normalization
        normalized_name = normalize_for_comparison(
            strip_kind_annotation(entity_info.name)
        )
        if normalized_name not in entities_by_normalized_name:
            entities_by_normalized_name[normalized_name] = []
        entities_by_normalized_name[normalized_name].append(entity_info)

    # Convert and merge each normalized entity group
    entities_dict = {}
    quotes_validated = 0
    quotes_failed = 0

    for normalized_name, entity_infos in entities_by_normalized_name.items():
        # Validate all quotes from all instances of this entity
        all_quotes = []
        for entity_info in entity_infos:
            for quote_str in entity_info.quotes:
                try:
                    quote = resource.quote(quote_str)
                    all_quotes.append(quote)
                    quotes_validated += 1
                except QuoteValidationError:
                    quotes_failed += 1

        if all_quotes:  # Only store entity if we have at least one valid quote
            # Use first entity's name (after stripping kind) as canonical name
            # This preserves original casing (e.g., "BRCA1" not "brca1")
            canonical_name = strip_kind_annotation(entity_infos[0].name)

            # Collect all alternative names as aliases
            all_aliases = []
            seen_aliases_normalized = set()
            for entity_info in entity_infos:
                # Add original name as alias if it differs from canonical
                stripped_name = strip_kind_annotation(entity_info.name)
                if stripped_name != canonical_name:
                    name_normalized = normalize_for_comparison(stripped_name)
                    if name_normalized not in seen_aliases_normalized:
                        all_aliases.append(stripped_name)
                        seen_aliases_normalized.add(name_normalized)
                # Add all aliases from this entity
                for alias in entity_info.aliases:
                    alias_normalized = normalize_for_comparison(alias)
                    if alias_normalized not in seen_aliases_normalized:
                        all_aliases.append(alias)
                        seen_aliases_normalized.add(alias_normalized)

            # Merge reasoning from all instances
            merged_reasoning = " | ".join(
                entity_info.reasoning for entity_info in entity_infos
            )

            # Store entity with canonical name
            entities_dict[canonical_name] = EntityMention(
                kind=entity_infos[0].kind,
                name=canonical_name,
                aliases=all_aliases,
                quotes=all_quotes,
                reasoning=merged_reasoning,
            )

    return (entities_dict, quotes_validated, quotes_failed)


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
    usage = RunUsage()

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
                result = await agent.run(prompt, deps=deps, usage=usage)
    except (TimeoutError, ConnectionError, ValueError) as e:
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
    usage = RunUsage()

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
                    result = await agent.run(prompt, deps=deps, usage=usage)
                finally:
                    # Mark pair as done when assessment completes or fails
                    if deps.progress:
                        deps.progress["Pairs assessed"].done()
    except (TimeoutError, ConnectionError, ValueError) as e:
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
