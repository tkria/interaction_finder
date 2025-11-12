"""Graph nodes for association extraction pipeline.

Each node represents a stage in the pipeline. Nodes control flow via
return type annotations. All branching and looping happens in nodes;
agents only produce typed data.

Pipeline stages:
1. ExtractEntitiesNode - Extract entities from documents
2. ValidateEntitiesNode - Validate entity kinds and merge substring duplicates
3. IdentifyProximalSetsNode - Find groups of proximal entities
4. ExtractPairsFromProximalSetsNode - Extract pairs from proximal regions
5. AssessPairsNode - Judge evidence for each pair per document
6. JudgeCrossDocumentNode - Make final accept/reject decisions
7. FinalizeNode - Build final output
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Union

from pydantic_ai.usage import RunUsage
from pydantic_graph import BaseNode, End, GraphRunContext

from interaction_finder.agent_utils import rename_agent
from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.extract import get_entity_extractor_agent
from interaction_finder.extraction.extract_proximal_pairs import get_proximal_pair_agent
from interaction_finder.extraction.judge_cross_document import (
    get_cross_document_judge_agent,
)
from interaction_finder.extraction.judge_pair_evidence import get_pair_judge_agent
from interaction_finder.extraction.merge_entities import get_entity_merge_agent
from interaction_finder.extraction.models import (
    EntityMention,
    EntityPairKey,
    ExtractionMetadata,
    ExtractionResult,
    PairAssessment,
    PairJudgment,
    SimpleEntity,
)
from interaction_finder.extraction.state import State
from interaction_finder.extraction.utils import (
    build_text_region,
    collect_relevant_text_for_quotes,
    find_substring_entities,
    identify_proximal_sets,
    make_entity_pair_key,
    normalize_for_comparison,
    strip_kind_annotation,
)
from interaction_finder.logging import logfire
from interaction_finder.resources import QuoteValidationError, Resource


def _classify_quote_error(error: Exception) -> str:
    """Classify quote validation error by type."""
    error_name = type(error).__name__
    if "Paraphrase" in error_name:
        return "paraphrased"
    elif "Missing" in error_name or "NotFound" in error_name:
        return "missing"
    else:
        return "fuzzy_match_failed"


def _extract_similarity(error: Exception) -> float | None:
    """Extract similarity percentage from error message if available."""
    error_str = str(error)
    # Look for "Similarity: XX.X%" pattern
    import re

    match = re.search(r"Similarity:\s*([\d.]+)%", error_str)
    if match:
        return float(match.group(1))
    return None


@dataclass
class ExtractEntitiesNode(BaseNode[State, Deps, ExtractionResult]):
    """Extract entities from all resources in parallel.

    For each resource:
    1. Call entity_extractor_agent to get entities with quotes
    2. Convert all quote strings to ResourceQuote objects via fuzzy matching
    3. Merge entities with same canonical name
    4. Store validated results in State.entities_by_resource
    """

    async def run(
        self, ctx: GraphRunContext[State, Deps]
    ) -> Union["ValidateEntitiesNode", End[ExtractionResult]]:
        """Process all resources in parallel."""
        with logfire.span("ExtractEntitiesNode"):
            resources = ctx.deps.resource_pool.resources

            if not resources:
                ctx.deps.logger.warning("No resources to process")
                return End(self._empty_result(ctx))

            # Set up progress tracking
            if ctx.deps.progress:
                ctx.deps.progress.documents_total = len(resources)
                ctx.deps.progress.set_phase_extracting()

            # Process all resources in parallel
            tasks = [self._process_resource(resource, ctx) for resource in resources]
            await asyncio.gather(*tasks)

            # Check if we found any entities
            if not ctx.state.entities_by_resource:
                ctx.deps.logger.warning("No entities extracted from documents")
                return End(self._empty_result(ctx))

            return ValidateEntitiesNode()

    def _empty_result(self, ctx: GraphRunContext[State, Deps]) -> ExtractionResult:
        """Create empty result for early termination."""
        return ExtractionResult(
            resources=ctx.deps.resource_pool,
            judgments=[],
            metadata=ExtractionMetadata(
                topic=ctx.state.topic,
                resource_count=len(ctx.deps.resource_pool.resources),
                total_entities_found=0,
                entities_after_validation=0,
                entities_merged=0,
                merge_cache_hits=0,
                merge_cache_misses=0,
                proximal_sets_found=0,
                total_pairs_found=0,
                pairs_accepted=0,
                pairs_rejected=0,
                quotes_validated=0,
                quotes_failed=0,
            ),
        )

    async def _process_resource(
        self, resource: Resource, ctx: GraphRunContext[State, Deps]
    ):
        """Process a single resource: extract entities."""
        usage = RunUsage()
        # Build prompt
        entity_types_str = ", ".join(ctx.state.target_entity_types)
        prompt = f"""Extract entities from this document relevant to: {ctx.state.topic}

**Target entity types:** {entity_types_str}

**Document title:** {resource.title}

**Document text:**
{resource.text[:15000]}

Extract all entities of the specified types that are relevant to the topic.
For each entity, provide: canonical name, all verbatim names from text, supporting quotes, and reasoning."""
        # Extract entities with renamed span
        agent = get_entity_extractor_agent(ctx.deps.config)
        try:
            with rename_agent(
                agent, name=f"ExtractEntitiesNode: {resource.title[:60]}"
            ):
                result = await agent.run(prompt, deps=ctx.deps, usage=usage)
        except (TimeoutError, ConnectionError, ValueError) as e:
            ctx.deps.logger.error(
                f"Entity extraction failed for {resource.id.url}: {type(e).__name__}: {e}"
            )
            return
        # Convert entities to EntityMention with ResourceQuotes, merging duplicates by normalized name
        # Group by normalized name to catch "PAH" vs "pah", "BRCA1" vs "brca1", etc.
        entities_by_normalized_name: dict[str, list] = {}
        for entity_info in result.output.entities:
            # Normalize: strip kind annotation, then apply text normalization
            normalized_name = normalize_for_comparison(
                strip_kind_annotation(entity_info.name)
            )
            if normalized_name not in entities_by_normalized_name:
                entities_by_normalized_name[normalized_name] = []
            entities_by_normalized_name[normalized_name].append(entity_info)
        # Convert and merge each entity name
        entities_dict = {}
        for normalized_name, entity_infos in entities_by_normalized_name.items():
            # Validate all quotes from all instances
            all_quotes = []
            for entity_info in entity_infos:
                for quote_str in entity_info.quotes:
                    try:
                        quote = resource.quote(quote_str)
                        all_quotes.append(quote)
                        ctx.state.quotes_validated += 1
                    except QuoteValidationError:
                        ctx.state.quotes_failed += 1
            if all_quotes:  # Only store entity if we have valid quotes
                # Use first entity's name (after stripping kind annotation) as canonical name
                # This preserves original casing (e.g., "BRCA1" not "brca1")
                canonical_name = strip_kind_annotation(entity_infos[0].name)
                # Collect all original names and aliases
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
                # Merge reasoning
                merged_reasoning = " | ".join(
                    entity_info.reasoning for entity_info in entity_infos
                )
                # Store with canonical name as both key and in entity object
                # Normalized name is only used for grouping during extraction
                entities_dict[canonical_name] = EntityMention(
                    kind=entity_infos[0].kind,
                    name=canonical_name,
                    aliases=all_aliases,
                    quotes=all_quotes,
                    reasoning=merged_reasoning,
                )
        # Store entities for this resource
        if entities_dict:
            ctx.state.entities_by_resource[resource.id] = entities_dict

            # Update progress
            if ctx.deps.progress:
                ctx.deps.progress.documents_processed += 1
                ctx.deps.progress.entities_found = sum(
                    len(e) for e in ctx.state.entities_by_resource.values()
                )
                ctx.deps.progress.quotes_validated = ctx.state.quotes_validated
                ctx.deps.progress.quotes_failed = ctx.state.quotes_failed
                ctx.deps.progress.update()


@dataclass
class ValidateEntitiesNode(BaseNode[State, Deps, ExtractionResult]):
    """Validate entity kinds and merge substring duplicates.

    For each resource:
    1. Remove entities not matching target_entity_types
    2. Identify entity pairs where one name is substring of another
    3. Batch LLM calls to decide which should be merged
    4. Apply merge decisions (move child name to parent aliases)
    5. Store validated entities
    """

    async def run(
        self, ctx: GraphRunContext[State, Deps]
    ) -> Union["IdentifyProximalSetsNode", End[ExtractionResult]]:
        """Validate entities across all resources."""
        with logfire.span("ValidateEntitiesNode"):
            # Process each resource
            for resource_id, entities in ctx.state.entities_by_resource.items():
                # Filter by kind
                valid_entities = {
                    name: entity
                    for name, entity in entities.items()
                    if entity.kind in ctx.state.target_entity_types
                }

                if not valid_entities:
                    continue

                # Find substring entity pairs
                substring_pairs = find_substring_entities(valid_entities)

                if substring_pairs:
                    # Batch merge decisions via LLM
                    merge_decisions = await self._get_merge_decisions(
                        substring_pairs, valid_entities, ctx
                    )

                    # Apply merges
                    for decision in merge_decisions:
                        if decision.should_merge:
                            parent = valid_entities.get(decision.parent_entity)
                            child = valid_entities.get(decision.child_entity)

                            if parent and child:
                                # Merge child into parent
                                # Add child name to parent aliases
                                if child.name not in parent.aliases:
                                    parent.aliases.append(child.name)
                                # Combine quotes
                                parent.quotes.extend(child.quotes)
                                # Merge reasoning
                                parent.reasoning += (
                                    f" | MERGED({child.name}): {child.reasoning}"
                                )
                                # Add child aliases to parent
                                for alias in child.aliases:
                                    if alias not in parent.aliases:
                                        parent.aliases.append(alias)

                                # Remove child
                                del valid_entities[decision.child_entity]
                                ctx.state.entities_merged += 1

                # Store validated entities
                if valid_entities:
                    ctx.state.validated_entities_by_resource[resource_id] = (
                        valid_entities
                    )

            # Check if we have any validated entities
            if not ctx.state.validated_entities_by_resource:
                ctx.deps.logger.warning("No entities after validation")
                total_found = sum(
                    len(entities)
                    for entities in ctx.state.entities_by_resource.values()
                )
                return End(
                    ExtractionResult(
                        resources=ctx.deps.resource_pool,
                        judgments=[],
                        metadata=ExtractionMetadata(
                            topic=ctx.state.topic,
                            resource_count=len(ctx.deps.resource_pool.resources),
                            total_entities_found=total_found,
                            entities_after_validation=0,
                            entities_merged=ctx.state.entities_merged,
                            merge_cache_hits=ctx.state.merge_cache_hits,
                            merge_cache_misses=ctx.state.merge_cache_misses,
                            proximal_sets_found=0,
                            total_pairs_found=0,
                            pairs_accepted=0,
                            pairs_rejected=0,
                            quotes_validated=ctx.state.quotes_validated,
                            quotes_failed=ctx.state.quotes_failed,
                        ),
                    )
                )

            # Apply merge decisions consistently across all documents
            self._normalize_merges_across_documents(ctx)

            return IdentifyProximalSetsNode()

    def _normalize_merges_across_documents(
        self, ctx: GraphRunContext[State, Deps]
    ) -> None:
        """Apply merge decisions consistently across all documents.

        After per-document merging, this ensures consistency by finding
        any entities across documents that should be merged based on cached
        decisions and applying those merges globally.

        This handles cases where:
        - Doc1 has "PAH" alone, Doc2 has "PAH" and "pulmonary arterial hypertension pah"
        - The LLM decided to merge them in Doc2
        - We need to apply same merge in Doc1 if we later extract the longer variant

        Algorithm:
        1. Collect all unique entity names across all documents (by normalized name)
        2. For each document, check if any entities should be merged based on cache
        3. Apply merges within each document, maintaining provenance
        """
        # Collect all unique entity canonical names (normalized) by kind
        all_entities_by_kind: dict[str, set[str]] = {}
        for entities in ctx.state.validated_entities_by_resource.values():
            for entity_name, entity in entities.items():
                if entity.kind not in all_entities_by_kind:
                    all_entities_by_kind[entity.kind] = set()
                all_entities_by_kind[entity.kind].add(entity_name)

        # For each kind, build a mapping of entities that should be merged
        # based on cached decisions: child_name → parent_name
        merge_mapping: dict[str, str] = {}

        for kind, entity_names in all_entities_by_kind.items():
            entity_list = list(entity_names)

            # Check all pairs for cached merge decisions
            for i, name1 in enumerate(entity_list):
                norm1 = normalize_for_comparison(name1)

                for name2 in entity_list[i + 1 :]:
                    norm2 = normalize_for_comparison(name2)

                    # Check if one is substring of other
                    if norm1 == norm2:
                        # Exact match - shouldn't happen after normalization, but handle it
                        continue
                    elif norm1 in norm2:
                        # name1 is parent (general), name2 is child (specific)
                        cache_key = (norm1, norm2, kind)
                        if (
                            cache_key in ctx.state.merge_decision_cache
                            and ctx.state.merge_decision_cache[cache_key]
                        ):
                            # This pair should be merged
                            merge_mapping[name2] = name1
                    elif norm2 in norm1:
                        # name2 is parent (general), name1 is child (specific)
                        cache_key = (norm2, norm1, kind)
                        if (
                            cache_key in ctx.state.merge_decision_cache
                            and ctx.state.merge_decision_cache[cache_key]
                        ):
                            # This pair should be merged
                            merge_mapping[name1] = name2

        # Apply merge mapping to all documents
        if merge_mapping:
            for (
                resource_id,
                entities,
            ) in ctx.state.validated_entities_by_resource.items():
                # Find entities in this document that need merging
                entities_to_merge = []
                for child_name in list(entities.keys()):
                    if child_name in merge_mapping:
                        parent_name = merge_mapping[child_name]
                        # Only merge if parent exists in this document
                        if parent_name in entities:
                            entities_to_merge.append((parent_name, child_name))

                # Apply merges
                for parent_name, child_name in entities_to_merge:
                    parent = entities[parent_name]
                    child = entities[child_name]

                    # Merge child into parent
                    if child.name not in parent.aliases:
                        parent.aliases.append(child.name)
                    parent.quotes.extend(child.quotes)
                    parent.reasoning += (
                        f" | MERGED_POSTHOC({child.name}): {child.reasoning}"
                    )
                    for alias in child.aliases:
                        if alias not in parent.aliases:
                            parent.aliases.append(alias)

                    # Remove child
                    del entities[child_name]
                    ctx.state.entities_merged += 1

            ctx.deps.logger.info(
                f"Post-hoc normalization: applied {len(merge_mapping)} merge rules across documents"
            )

    async def _get_merge_decisions(
        self,
        substring_pairs: list[tuple[str, str]],
        entities: dict[str, EntityMention],
        ctx: GraphRunContext[State, Deps],
    ) -> list:
        """Get merge decisions from cache or LLM in batches.

        Checks cache first using normalized entity names. For cache misses,
        queries LLM and updates cache with results.
        """
        from interaction_finder.extraction.models import EntityMergeDecision

        # Separate pairs into cached and uncached
        cached_decisions = []
        uncached_pairs = []

        for parent_name, child_name in substring_pairs:
            parent = entities[parent_name]
            child = entities[child_name]
            # Create cache key: (norm_parent, norm_child, kind)
            # Use same kind for both since we only merge entities of same kind
            cache_key = (
                normalize_for_comparison(parent_name),
                normalize_for_comparison(child_name),
                parent.kind,
            )

            if cache_key in ctx.state.merge_decision_cache:
                # Cache hit - reuse decision
                should_merge = ctx.state.merge_decision_cache[cache_key]
                cached_decisions.append(
                    EntityMergeDecision(
                        parent_entity=parent_name,
                        child_entity=child_name,
                        should_merge=should_merge,
                        reasoning="[Cached decision from previous document]",
                    )
                )
                ctx.state.merge_cache_hits += 1
            else:
                # Cache miss - need LLM decision
                uncached_pairs.append((parent_name, child_name))
                ctx.state.merge_cache_misses += 1

        all_decisions = cached_decisions

        # Query LLM for uncached pairs only
        if uncached_pairs:
            # Get batch size from config (default 50)
            batch_size = getattr(
                ctx.deps.config.tools.extraction, "merge_batch_size", 50
            )

            # Process uncached pairs in batches
            for i in range(0, len(uncached_pairs), batch_size):
                batch = uncached_pairs[i : i + batch_size]

                # Build prompt describing all pairs in batch
                pairs_description = []
                for parent_name, child_name in batch:
                    parent = entities[parent_name]
                    child = entities[child_name]
                    pairs_description.append(
                        f"- Parent: '{parent_name}' (type: {parent.kind})\n"
                        f"  Child: '{child_name}' (type: {child.kind})"
                    )

                # Format target entity types
                entity_types_str = ", ".join(ctx.state.target_entity_types)

                prompt = f"""**Research topic:** {ctx.state.topic}

**Target entity types for this research:** {entity_types_str}

**Task:** Decide whether the following entity pairs should be merged.
Each pair has one entity whose name is a substring of the other.

Consider the research context: we're looking for associations between {entity_types_str}.
Merge entities that represent the same concept for this research question, even if they
differ in biological specificity (e.g., merge disease subtypes into the main disease,
merge gene variants into the gene name).

**Entity pairs to evaluate:**
{chr(10).join(pairs_description)}

For each pair, decide if they should be merged (child absorbed into parent) or kept separate."""

                # Call merge agent with renamed span
                agent = get_entity_merge_agent(ctx.deps.config)
                usage = RunUsage()
                try:
                    with rename_agent(
                        agent,
                        name=f"ValidateEntitiesNode (merge batch {i // batch_size + 1})",
                    ):
                        result = await agent.run(prompt, deps=ctx.deps, usage=usage)
                    # Store decisions in cache for future use
                    for decision in result.output.decisions:
                        parent = entities[decision.parent_entity]
                        cache_key = (
                            normalize_for_comparison(decision.parent_entity),
                            normalize_for_comparison(decision.child_entity),
                            parent.kind,
                        )
                        ctx.state.merge_decision_cache[cache_key] = (
                            decision.should_merge
                        )

                    all_decisions.extend(result.output.decisions)
                except (TimeoutError, ConnectionError, ValueError) as e:
                    ctx.deps.logger.error(
                        f"Entity merge decision failed: {type(e).__name__}: {e}"
                    )
                    # Continue without merging this batch

        return all_decisions


@dataclass
class IdentifyProximalSetsNode(BaseNode[State, Deps, ExtractionResult]):
    """Identify groups of entities found in close proximity.

    For each resource:
    1. Use sliding window algorithm to find entity co-occurrence regions
    2. Store ProximalEntitySet objects
    """

    async def run(
        self, ctx: GraphRunContext[State, Deps]
    ) -> "ExtractPairsFromProximalSetsNode":
        """Identify proximal sets in all resources."""
        with logfire.span("IdentifyProximalSetsNode"):
            # Get proximity threshold from config (default 2)
            threshold = getattr(
                ctx.deps.config.tools.extraction, "proximal_window_chunks", 2
            )

            # Process each resource
            for (
                resource_id,
                entities,
            ) in ctx.state.validated_entities_by_resource.items():
                resource = ctx.deps.resource_pool.get(resource_id)
                if not resource:
                    continue

                # Identify proximal sets
                proximal_sets = identify_proximal_sets(entities, threshold, resource)

                if proximal_sets:
                    ctx.state.proximal_sets_by_resource[resource_id] = proximal_sets

            return ExtractPairsFromProximalSetsNode()


@dataclass
class ExtractPairsFromProximalSetsNode(BaseNode[State, Deps, ExtractionResult]):
    """Extract pairs from proximal entity regions.

    For each proximal set:
    1. Build text region (first quote chunk to last + padding)
    2. Call proximal_pair_agent to extract associations
    3. Validate entities in response are in proximal set
    4. Validate quotes and create pair objects
    5. Store pairs temporarily per resource (deduplication happens in next node)
    """

    async def run(self, ctx: GraphRunContext[State, Deps]) -> "AssessPairsNode":
        """Extract pairs from all proximal sets in parallel."""
        with logfire.span("ExtractPairsFromProximalSetsNode"):
            # Collect all proximal sets with their resource context
            tasks = []
            for (
                resource_id,
                proximal_sets,
            ) in ctx.state.proximal_sets_by_resource.items():
                resource = ctx.deps.resource_pool.get(resource_id)
                if not resource:
                    continue

                entities = ctx.state.validated_entities_by_resource[resource_id]

                for proximal_set in proximal_sets:
                    tasks.append(
                        self._process_proximal_set(
                            proximal_set, entities, resource, resource_id, ctx
                        )
                    )

            # Run all extractions in parallel
            if tasks:
                # Results is list of (resource_id, list[pair_tuples])
                results = await asyncio.gather(*tasks)

                # Organize by resource for next stage
                pairs_by_resource = defaultdict(list)
                for resource_id, pairs in results:
                    if pairs:
                        pairs_by_resource[resource_id].extend(pairs)

                # Store temporarily (will be deduplicated and assessed in next node)
                # Store as list of (entity1, entity2, relationship_candidates, quotes) tuples
                ctx.state._temp_pairs_by_resource = dict(pairs_by_resource)

            return AssessPairsNode()

    async def _process_proximal_set(
        self,
        proximal_set,
        entities: dict[str, EntityMention],
        resource: Resource,
        resource_id,
        ctx: GraphRunContext[State, Deps],
    ) -> tuple:
        """Process a single proximal set to extract pairs."""
        # Build readable entity list for span name
        entity_names = sorted(proximal_set.entities)[:3]  # Show up to 3 entities
        entities_str = ", ".join(entity_names)
        if len(proximal_set.entities) > 3:
            entities_str += f" (+{len(proximal_set.entities) - 3} more)"
        # Get padding from config (default 1)
        padding = getattr(ctx.deps.config.tools.extraction, "region_padding_chunks", 1)
        # Build text region
        chunk_start, chunk_end = proximal_set.chunk_range
        text_region = build_text_region(resource, chunk_start, chunk_end, padding)
        # Build entity list with aliases
        entity_list = []
        for entity_name in proximal_set.entities:
            entity = entities.get(entity_name)
            if entity:
                aliases_str = ", ".join(entity.aliases)
                entity_list.append(
                    f"- **{entity_name}** [kind: {entity.kind}, aliases: {aliases_str}]"
                )
        prompt = f"""Topic: {ctx.state.topic}

**Entities in this region:**
{chr(10).join(entity_list)}

**Text region:**
{text_region}

Extract all binary associations between these entities that are clearly stated or implied in the text.
Use the canonical entity names as shown in bold (kind and aliases are metadata only).
Provide exact supporting quotes."""
        # Call proximal pair agent with renamed span
        agent = get_proximal_pair_agent(ctx.deps.config)
        usage = RunUsage()
        try:
            with rename_agent(
                agent, name=f"ExtractPairsFromProximalSetsNode: {entities_str}"
            ):
                result = await agent.run(prompt, deps=ctx.deps, usage=usage)
        except (TimeoutError, ConnectionError, ValueError) as e:
            ctx.deps.logger.error(
                f"Proximal pair extraction failed: {type(e).__name__}: {e}"
            )
            return (resource_id, [])
        # Process extracted pairs
        pairs = []
        for pair_info in result.output.pairs:
            # Sanitize entity names (strip kind annotations if present)
            entity1 = strip_kind_annotation(pair_info.entity1)
            entity2 = strip_kind_annotation(pair_info.entity2)
            # Verify entities are in proximal set
            if (
                entity1 not in proximal_set.entities
                or entity2 not in proximal_set.entities
            ):
                ctx.deps.logger.warning(
                    f"Pair references entity not in proximal set: {entity1}-{entity2}"
                )
                continue
            # Validate pair kinds are permitted
            e1_obj = entities.get(entity1)
            e2_obj = entities.get(entity2)
            if not e1_obj or not e2_obj:
                ctx.deps.logger.warning(
                    f"Pair references unknown entity: {entity1}-{entity2}"
                )
                continue
            # Check if this pair combination is permitted
            allowed_partners = ctx.state.permitted_pairs.get(e1_obj.kind, set())
            if e2_obj.kind not in allowed_partners:
                ctx.deps.logger.debug(
                    f"Pair {entity1} ({e1_obj.kind}) - {entity2} ({e2_obj.kind}) "
                    f"not permitted by kind constraints"
                )
                continue
            # Validate quotes
            quotes = []
            for quote_str in pair_info.supporting_quotes:
                try:
                    quote = resource.quote(quote_str)
                    quotes.append(quote)
                    ctx.state.quotes_validated += 1
                except Exception as e:
                    ctx.state.quotes_failed += 1
                    ctx.deps.logger.warning(
                        f"Failed to validate pair quote: {type(e).__name__}: {e}"
                    )
            if quotes:
                # Store as tuple (will be converted to PairAssessment after dedup/assessment)
                # Use sanitized entity names
                pairs.append(
                    (
                        entity1,
                        entity2,
                        pair_info.relationship_types,
                        quotes,
                    )
                )
        return (resource_id, pairs)


@dataclass
class AssessPairsNode(BaseNode[State, Deps, ExtractionResult]):
    """Deduplicate pairs and assess evidence per document.

    For each resource:
    1. Deduplicate pairs (same entity pair may appear in multiple proximal sets)
    2. For each unique pair:
       - Collect all quotes
       - Build relevant text region
       - Call pair_judge_agent
       - Create PairAssessment with full EntityMention objects
    3. Store assessments
    """

    async def run(self, ctx: GraphRunContext[State, Deps]) -> "JudgeCrossDocumentNode":
        """Assess all pairs per resource."""
        with logfire.span("AssessPairsNode"):
            # Get temporary pairs from previous node
            temp_pairs = getattr(ctx.state, "_temp_pairs_by_resource", {})

            # Process each resource
            tasks = []
            for resource_id, pairs_list in temp_pairs.items():
                resource = ctx.deps.resource_pool.get(resource_id)
                if not resource:
                    continue

                entities = ctx.state.validated_entities_by_resource[resource_id]

                # Deduplicate pairs by (entity1, entity2)
                pairs_dict = {}
                for entity1_name, entity2_name, rel_types, quotes in pairs_list:
                    # Get entity objects
                    entity1 = entities[entity1_name]
                    entity2 = entities[entity2_name]

                    # Create ordered key (determines canonical ordering)
                    pair_key = make_entity_pair_key(entity1, entity2)

                    if pair_key not in pairs_dict:
                        # Store entities in the canonical order determined by pair_key
                        ordered_entity1 = entities[pair_key.entity1_name]
                        ordered_entity2 = entities[pair_key.entity2_name]

                        pairs_dict[pair_key] = {
                            "entity1": ordered_entity1,
                            "entity2": ordered_entity2,
                            "relationship_candidates": set(),
                            "quotes": [],
                        }

                    pairs_dict[pair_key]["relationship_candidates"].update(rel_types)
                    pairs_dict[pair_key]["quotes"].extend(quotes)

                # Assess each unique pair
                for pair_info in pairs_dict.values():
                    tasks.append(
                        self._assess_pair(pair_info, resource, resource_id, ctx)
                    )

            # Run all assessments in parallel
            if tasks:
                assessments = await asyncio.gather(*tasks)

                # Organize by resource
                for resource_id, assessment in assessments:
                    if assessment:
                        if resource_id not in ctx.state.pair_assessments_by_resource:
                            ctx.state.pair_assessments_by_resource[resource_id] = []
                        ctx.state.pair_assessments_by_resource[resource_id].append(
                            assessment
                        )

            # Clean up temporary data
            if hasattr(ctx.state, "_temp_pairs_by_resource"):
                delattr(ctx.state, "_temp_pairs_by_resource")

            return JudgeCrossDocumentNode()

    async def _assess_pair(
        self,
        pair_info: dict,
        resource: Resource,
        resource_id,
        ctx: GraphRunContext[State, Deps],
    ) -> tuple:
        """Assess a single pair in a single resource."""
        entity1 = pair_info["entity1"]
        entity2 = pair_info["entity2"]
        # Get padding from config
        padding = getattr(ctx.deps.config.tools.extraction, "region_padding_chunks", 1)
        # Build relevant text for all quotes
        all_quotes = pair_info["quotes"]
        text_region = collect_relevant_text_for_quotes(resource, all_quotes, padding)
        # Build quote list for prompt
        quotes_str = "\n".join(
            f"[{i}] {q.get_quote_text()}" for i, q in enumerate(all_quotes)
        )
        # Build relationship candidates string
        candidates = list(pair_info["relationship_candidates"])
        candidates_str = ", ".join(f'"{c}"' for c in candidates)
        prompt = f"""Topic: {ctx.state.topic}

**Pair:** {entity1.name} ({entity1.kind}) <-> {entity2.name} ({entity2.kind})

**Relationship type candidates:** {candidates_str}

**Relevant text from document:**
{text_region}

**Supporting quotes:**
{quotes_str}

Assess the strength of evidence for this association in this document.
Select the most appropriate relationship type (from candidates or propose a more specific one).
Assign a confidence level (high/medium/low) and explain your reasoning."""
        # Call pair judge agent with renamed span
        agent = get_pair_judge_agent(ctx.deps.config)
        usage = RunUsage()
        try:
            with rename_agent(
                agent, name=f"AssessPairsNode: {entity1.name} ⇌ {entity2.name}"
            ):
                result = await agent.run(prompt, deps=ctx.deps, usage=usage)
        except (TimeoutError, ConnectionError, ValueError) as e:
            ctx.deps.logger.error(
                f"Pair assessment failed for {entity1.name}-{entity2.name}: "
                f"{type(e).__name__}: {e}"
            )
            return (resource_id, None)
        # Extract referenced quotes
        referenced_quotes = [
            all_quotes[i]
            for i in result.output.supporting_quote_ids
            if i < len(all_quotes)
        ]
        # Create assessment
        assessment = PairAssessment(
            resource_id=resource_id,
            entity1=entity1,
            entity2=entity2,
            relationship=result.output.relationship,
            quotes=referenced_quotes if referenced_quotes else all_quotes,
            confidence=result.output.confidence,
            reasoning=result.output.reasoning,
        )
        return (resource_id, assessment)


@dataclass
class JudgeCrossDocumentNode(BaseNode[State, Deps, ExtractionResult]):
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

    async def run(self, ctx: GraphRunContext[State, Deps]) -> "FinalizeNode":
        """Judge all unique pairs across documents."""
        with logfire.span("JudgeCrossDocumentNode"):
            # Set phase
            if ctx.deps.progress:
                ctx.deps.progress.set_phase_judging()

            # Group assessments by entity pair
            assessments_by_pair: dict[EntityPairKey, list[PairAssessment]] = (
                defaultdict(list)
            )

            for assessments_list in ctx.state.pair_assessments_by_resource.values():
                for assessment in assessments_list:
                    pair_key = make_entity_pair_key(
                        assessment.entity1, assessment.entity2
                    )
                    assessments_by_pair[pair_key].append(assessment)

            # Update unique pairs count
            if ctx.deps.progress:
                ctx.deps.progress.unique_pairs = len(assessments_by_pair)
                ctx.deps.progress.update()

            # Judge each pair
            tasks = []
            for pair_key, assessments in assessments_by_pair.items():
                tasks.append(self._judge_pair(pair_key, assessments, ctx))

            # Run all judgments in parallel
            if tasks:
                judgments = await asyncio.gather(*tasks)

                for pair_key, judgment in judgments:
                    ctx.state.pair_judgments[pair_key] = judgment
                    # Update accepted/rejected counts
                    if ctx.deps.progress:
                        if judgment.accepted:
                            ctx.deps.progress.accepted += 1
                        else:
                            ctx.deps.progress.rejected += 1
                        ctx.deps.progress.update()

            return FinalizeNode()

    def _can_accept_deterministically(
        self, assessments: list[PairAssessment]
    ) -> tuple[bool, str, str]:
        """Check if we can accept without LLM call.

        Returns:
            (can_accept, relationship, reasoning) or (False, "", "")
        """
        # Need multiple high-confidence assessments with same relationship
        high_conf = [a for a in assessments if a.confidence == "high"]

        if len(high_conf) >= 2:
            # Check if all have same relationship
            relationships = {a.relationship for a in high_conf}
            if len(relationships) == 1:
                relationship = high_conf[0].relationship
                return (
                    True,
                    relationship,
                    f"Multiple high-confidence assessments ({len(high_conf)}) "
                    f"with consistent relationship '{relationship}' provide strong evidence.",
                )

        return (False, "", "")

    def _should_investigate(self, assessments: list[PairAssessment]) -> bool:
        """Determine if we need LLM investigation.

        Always investigate except for the deterministic accept case.
        """
        # Check for deterministic accept
        can_accept, _, _ = self._can_accept_deterministically(assessments)
        if can_accept:
            return False

        # All other cases need investigation
        return True

    async def _judge_pair(
        self,
        pair_key: EntityPairKey,
        assessments: list[PairAssessment],
        ctx: GraphRunContext[State, Deps],
    ) -> tuple[EntityPairKey, PairJudgment]:
        """Make final judgment on a single pair."""
        # Try deterministic accept
        can_accept, relationship, reasoning = self._can_accept_deterministically(
            assessments
        )
        if can_accept:
            # Create judgment without LLM call
            # Get entity info from first assessment
            first_assessment = assessments[0]
            judgment = PairJudgment(
                entity1=SimpleEntity(
                    name=first_assessment.entity1.name,
                    kind=first_assessment.entity1.kind,
                    aliases=first_assessment.entity1.aliases,
                ),
                entity2=SimpleEntity(
                    name=first_assessment.entity2.name,
                    kind=first_assessment.entity2.kind,
                    aliases=first_assessment.entity2.aliases,
                ),
                relationship=relationship,
                assessments=assessments,
                accepted=True,
                confidence="high",
                reasoning=reasoning,
            )
            return (pair_key, judgment)
        # Need LLM investigation
        # Build per-document sections with assessment + content together
        padding = getattr(ctx.deps.config.tools.extraction, "region_padding_chunks", 1)
        document_sections = []
        for i, assessment in enumerate(assessments, 1):
            resource = ctx.deps.resource_pool.get(assessment.resource_id)
            if not resource:
                continue
            # Build document section with assessment first, then content
            text = collect_relevant_text_for_quotes(
                resource, assessment.quotes, padding
            )
            section = f"""**Document {i}:** {resource.title}

**Assessment:** {assessment.confidence} confidence - {assessment.relationship}
**Reasoning:** {assessment.reasoning}

**Evidence from document:**
{text}"""
            document_sections.append(section)
        # Get all relationship types mentioned
        relationships = {a.relationship for a in assessments}
        relationships_str = ", ".join(f'"{r}"' for r in sorted(relationships))
        prompt = f"""Topic: {ctx.state.topic}

**Pair:** {pair_key.entity1_name} <-> {pair_key.entity2_name}

**Relationship types found across documents:** {relationships_str}

{chr(10).join(f"{chr(10)}---{chr(10)}{chr(10)}" + section for section in document_sections)}

---

**Task:**
Make a final judgment on whether to accept this association.
Synthesize the evidence across documents, considering consistency, quality, and contradictions.
Decide: accept or reject, with confidence level (high/medium/low) and detailed reasoning."""
        # Call cross-document judge with renamed span
        agent = get_cross_document_judge_agent(ctx.deps.config)
        usage = RunUsage()
        try:
            with rename_agent(
                agent,
                name=f"JudgeCrossDocumentNode: {pair_key.entity1_name} ⇌ {pair_key.entity2_name}",
            ):
                result = await agent.run(prompt, deps=ctx.deps, usage=usage)
        except (TimeoutError, ConnectionError, ValueError) as e:
            ctx.deps.logger.error(
                f"Cross-document judgment failed for {pair_key}: "
                f"{type(e).__name__}: {e}"
            )
            # Default to rejection with low confidence
            first_assessment = assessments[0]
            return (
                pair_key,
                PairJudgment(
                    entity1=SimpleEntity(
                        name=first_assessment.entity1.name,
                        kind=first_assessment.entity1.kind,
                        aliases=first_assessment.entity1.aliases,
                    ),
                    entity2=SimpleEntity(
                        name=first_assessment.entity2.name,
                        kind=first_assessment.entity2.kind,
                        aliases=first_assessment.entity2.aliases,
                    ),
                    relationship=first_assessment.relationship,
                    assessments=assessments,
                    accepted=False,
                    confidence="low",
                    reasoning=f"Judgment failed due to error: {e}",
                ),
            )
        # Select most common relationship type
        relationship_counts = {}
        for assessment in assessments:
            relationship_counts[assessment.relationship] = (
                relationship_counts.get(assessment.relationship, 0) + 1
            )
        most_common_relationship = max(relationship_counts.items(), key=lambda x: x[1])[
            0
        ]
        # Create judgment
        first_assessment = assessments[0]
        judgment = PairJudgment(
            entity1=SimpleEntity(
                name=first_assessment.entity1.name,
                kind=first_assessment.entity1.kind,
                aliases=first_assessment.entity1.aliases,
            ),
            entity2=SimpleEntity(
                name=first_assessment.entity2.name,
                kind=first_assessment.entity2.kind,
                aliases=first_assessment.entity2.aliases,
            ),
            relationship=most_common_relationship,
            assessments=assessments,
            accepted=result.output.accepted,
            confidence=result.output.confidence,
            reasoning=result.output.reasoning,
        )
        return (pair_key, judgment)


@dataclass
class FinalizeNode(BaseNode[State, Deps, ExtractionResult]):
    """Build final output with all judgments and metadata."""

    async def run(self, ctx: GraphRunContext[State, Deps]) -> End[ExtractionResult]:
        """Gather all judgments and build final result."""
        with logfire.span("FinalizeNode"):
            # Convert judgments dict to list
            all_judgments = list(ctx.state.pair_judgments.values())

            # Calculate metadata
            total_entities_found = sum(
                len(entities) for entities in ctx.state.entities_by_resource.values()
            )
            entities_after_validation = sum(
                len(entities)
                for entities in ctx.state.validated_entities_by_resource.values()
            )
            proximal_sets_found = sum(
                len(sets) for sets in ctx.state.proximal_sets_by_resource.values()
            )
            total_pairs_found = len(all_judgments)
            pairs_accepted = sum(1 for j in all_judgments if j.accepted)
            pairs_rejected = total_pairs_found - pairs_accepted

            metadata = ExtractionMetadata(
                topic=ctx.state.topic,
                resource_count=len(ctx.deps.resource_pool.resources),
                total_entities_found=total_entities_found,
                entities_after_validation=entities_after_validation,
                entities_merged=ctx.state.entities_merged,
                merge_cache_hits=ctx.state.merge_cache_hits,
                merge_cache_misses=ctx.state.merge_cache_misses,
                proximal_sets_found=proximal_sets_found,
                total_pairs_found=total_pairs_found,
                pairs_accepted=pairs_accepted,
                pairs_rejected=pairs_rejected,
                quotes_validated=ctx.state.quotes_validated,
                quotes_failed=ctx.state.quotes_failed,
            )

            result = ExtractionResult(
                resources=ctx.deps.resource_pool,
                judgments=all_judgments,
                metadata=metadata,
            )

            return End(result)
