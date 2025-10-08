"""
Pipeline nodes for extraction graph V3.

Implements the three-phase extraction pipeline:
1. ExtractEntities: Extract entities from full documents with semantic caching
2. AssessIndividually: Assess each entity's relationship potential (task 06)
3. GeneratePairs: Generate and evaluate entity pairs (tasks 07-08)
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING

from pydantic_graph import BaseNode, GraphRunContext

from .state import ExtractionStateV3
from .deps import ExtractionDepsV3
from .agents import create_entity_extractor_v3, create_assessment_agent_v3
from .cache import compute_extraction_cache_key, compute_assessment_cache_key
from ..extraction_graph_v2.models import (
    EntityWithQuotes,
    SimpleEntityListOut,
    AssessmentOut,
    IndividualAssessment,
)
from ..extraction_graph_v2.parallelism import with_parallelism_control
from ..resources import (
    ResourceId,
    Resource,
    find_quote_with_fuzzy_matching,
    FuzzySuggestion,
)

if TYPE_CHECKING:
    from .models import PairCandidate

logger = logging.getLogger(__name__)


@dataclass
class ExtractEntities(BaseNode[ExtractionStateV3, ExtractionDepsV3]):
    """
    Extract entities from documents using full document context.

    Uses semantic caching to avoid redundant LLM calls and fuzzy quote matching
    for robust quote validation. Processes each document independently with
    configurable parallelism.

    Phase 1 of V3 pipeline.
    """

    async def run(
        self, ctx: GraphRunContext[ExtractionStateV3, ExtractionDepsV3]
    ) -> None:
        """
        Extract entities from all documents with parallel processing.

        Orchestrates per-document extraction with cache checking, LLM calls,
        quote validation, and entity deduplication across documents.
        """
        state, deps = ctx.state, ctx.deps

        resources = state.resource_pool.resources
        logger.info(f"Starting entity extraction from {len(resources)} documents")
        logger.info(f"Entity kinds: {deps.get_entity_kinds()}")
        logger.info(
            f"Cache enabled: {deps.semantic_cache_enabled and state.cache is not None}"
        )

        # Execute extractions with parallelism control
        parallelism_desc = (
            "unlimited"
            if deps.extraction_parallelism == 0
            else f"limit={deps.extraction_parallelism}"
        )
        logger.info(f"Processing with parallelism {parallelism_desc}")

        results = await with_parallelism_control(
            resources,
            lambda resource: self._extract_from_document(resource.id, ctx),
            parallelism=deps.extraction_parallelism,
            description="document extraction",
        )

        # Merge entities across documents
        for i, doc_entities in enumerate(results):
            if isinstance(doc_entities, Exception):
                resource = resources[i]
                logger.error(
                    f"Extraction failed for document {resource.id.id}: {doc_entities}"
                )
                continue

            self._merge_entities_into_state(doc_entities, state)

        logger.info(
            f"Extraction complete: {len(state.entities_found)} unique entities found"
        )
        logger.info(
            f"Cache stats - hits: {state.metrics.cache_hits_extraction}, "
            f"misses: {state.metrics.cache_misses_extraction}"
        )

        # Save checkpoint if callback provided
        if hasattr(deps, "checkpoint_callback") and deps.checkpoint_callback:
            await deps.checkpoint_callback("extraction", state)

    async def _extract_from_document(
        self,
        resource_id: ResourceId,
        ctx: GraphRunContext[ExtractionStateV3, ExtractionDepsV3],
    ) -> List[EntityWithQuotes]:
        """
        Extract entities from a single document with caching.

        Checks semantic cache before calling LLM, converts agent output
        to EntityWithQuotes with validated quotes, and saves to cache.

        Args:
            resource_id: Document identifier
            ctx: Graph context with state and deps

        Returns:
            List of entities with validated quotes from this document
        """
        state, deps = ctx.state, ctx.deps
        cache = state.cache

        resource = state.resource_pool.get(resource_id)
        logger.debug(f"Processing document: {resource.title}")

        # Compute cache key
        model_str = str(deps.model) if hasattr(deps.model, "__str__") else deps.model
        cache_key = compute_extraction_cache_key(
            full_doc_text=resource.text,
            entity_kinds=deps.get_entity_kinds(),
            model_version=model_str,
            prompt_version="v3_2025-10",  # Update when prompts change
        )

        # Check cache if enabled
        if cache and deps.semantic_cache_enabled:
            cached = await cache.get_extraction(cache_key)
            if cached is not None:
                state.metrics.record_cache_hit("extraction")
                logger.debug(f"Cache hit for {resource.title} ({len(cached)} entities)")
                return cached

            state.metrics.record_cache_miss("extraction")

        # Cache miss - extract entities via LLM
        logger.debug(f"Cache miss for {resource.title} - calling LLM")

        # Create extraction agent
        from ..models import Term

        target_term = (
            Term(name=deps.target_term or "unknown", kind=None)
            if isinstance(deps.target_term, str)
            else deps.target_term
        )

        agent = create_entity_extractor_v3(
            model=deps.model,
            entity_kinds=deps.get_entity_kinds(),
            task_context=deps.get_task_context(),
            target_term=target_term,
        )

        # Run extraction with timing
        start_time = time.time()
        try:
            result = await agent.run(resource.text, deps=deps)
            duration = time.time() - start_time

            state.metrics.entities_extraction_calls += 1
            state.metrics.entities_extraction_successes += 1
            state.metrics.extraction_time += duration

            logger.debug(
                f"Extracted {len(result.entities)} entities from {resource.title} "
                f"in {duration:.2f}s"
            )
        except Exception as e:
            duration = time.time() - start_time
            state.metrics.entities_extraction_calls += 1
            state.metrics.extraction_time += duration

            logger.error(f"Extraction failed for {resource.title}: {e}")
            return []

        # Convert to EntityWithQuotes with validated quotes
        entities = self._create_entities_with_quotes(result, resource, deps)

        # Save to cache if enabled
        if cache and deps.semantic_cache_enabled and entities:
            await cache.set_extraction(cache_key, entities)
            logger.debug(
                f"Saved {len(entities)} entities to cache for {resource.title}"
            )

        return entities

    def _create_entities_with_quotes(
        self,
        agent_output: "SimpleEntityListOut",
        resource: Resource,
        deps: ExtractionDepsV3,
    ) -> List[EntityWithQuotes]:
        """
        Convert agent output to EntityWithQuotes with validated quotes.

        Uses fuzzy matching to validate quotes against resource text.
        Entities without valid quotes are excluded (quality over quantity).

        Args:
            agent_output: Raw agent output with entity data and quote strings
            resource: Source document
            deps: Dependencies for quote validation config

        Returns:
            List of entities with validated ResourceQuote objects
        """
        entities = []

        for entity_out in agent_output.entities:
            # Validate quotes using fuzzy matching (task 03)
            validated_quotes = []

            for quote_text in entity_out.quotes:
                if not quote_text or not quote_text.strip():
                    continue

                quote_text = quote_text.strip()

                # Try fuzzy matching with configurable thresholds
                result = find_quote_with_fuzzy_matching(
                    resource=resource,
                    quote_text=quote_text,
                    auto_correct_threshold=deps.fuzzy_auto_correct_threshold,
                    suggest_threshold=deps.fuzzy_suggest_threshold,
                )

                if isinstance(result, FuzzySuggestion):
                    # Log suggestion but don't include (below auto-correct threshold)
                    logger.warning(
                        f"Quote validation for {entity_out.name} below threshold "
                        f"({result.similarity:.2f}): {quote_text[:50]}..."
                    )
                    # Record quote error (treat as paraphrased)
                    from ..extraction_graph_v2.models import QuoteErrorRecord

                    deps.quote_error_log.append(
                        QuoteErrorRecord(
                            entity_name=entity_out.name,
                            entity_kind=entity_out.kind,
                            original_quote=quote_text,
                            error_type="paraphrased",
                            suggested_corrections=[result.suggested_quote],
                            matched_percentage=result.similarity,
                        )
                    )
                elif result is None:
                    # No match found
                    logger.warning(
                        f"Quote validation failed for {entity_out.name}: {quote_text[:50]}..."
                    )
                    # Record quote error
                    from ..extraction_graph_v2.models import QuoteErrorRecord

                    deps.quote_error_log.append(
                        QuoteErrorRecord(
                            entity_name=entity_out.name,
                            entity_kind=entity_out.kind,
                            original_quote=quote_text,
                            error_type="not_found",
                        )
                    )
                else:
                    # Valid ResourceQuote
                    validated_quotes.append(result)
                    logger.debug(
                        f"Validated quote for {entity_out.name}: {quote_text[:50]}..."
                    )

            # Only include entities with at least one valid quote
            if validated_quotes:
                entity = EntityWithQuotes(
                    name=entity_out.name,
                    kind=entity_out.kind,
                    aliases=entity_out.aliases,
                    quotes=validated_quotes,
                    confidence=1.0,
                )

                # Final validation
                if entity.validate():
                    entities.append(entity)
                    logger.debug(
                        f"Created entity '{entity.name}' with {len(validated_quotes)} quotes"
                    )
                else:
                    logger.warning(f"Entity '{entity.name}' failed validation")

        return entities

    def _merge_entities_into_state(
        self,
        doc_entities: List[EntityWithQuotes],
        state: ExtractionStateV3,
    ) -> None:
        """
        Merge entities from one document into state, handling duplicates.

        Entities are deduplicated by name (case-sensitive). Quotes from
        multiple documents are aggregated, and aliases are merged.

        Args:
            doc_entities: Entities extracted from one document
            state: Extraction state to update
        """
        for entity in doc_entities:
            key = entity.name  # Use name as key (case-sensitive)

            if key in state.entities_found:
                # Merge quotes from this document
                existing = state.entities_found[key]
                existing.quotes.extend(entity.quotes)

                # Merge aliases (deduplicate)
                for alias in entity.aliases:
                    if alias not in existing.aliases:
                        existing.aliases.append(alias)

                logger.debug(
                    f"Merged entity '{key}' - now has {len(existing.quotes)} total quotes"
                )
            else:
                # New entity
                state.entities_found[key] = entity
                logger.debug(
                    f"Added new entity '{key}' with {len(entity.quotes)} quotes"
                )


@dataclass
class AssessIndividually(BaseNode[ExtractionStateV3, ExtractionDepsV3]):
    """
    Assess each entity's relationship potential individually.

    Processes all entities concurrently (I/O bound) using semantic caching
    to avoid redundant LLM calls. Extracts related_entities for candidate
    generation and validates evidence quotes using fuzzy matching.

    Phase 2 of V3 pipeline.
    """

    async def run(
        self, ctx: GraphRunContext[ExtractionStateV3, ExtractionDepsV3]
    ) -> None:
        """
        Assess all entities concurrently with cache checking.

        Orchestrates per-entity assessment with cache checking, LLM calls,
        evidence quote validation, and storage in state.individual_assessments.
        """
        state, deps = ctx.state, ctx.deps

        logger.info(
            f"Starting individual assessment of {len(state.entities_found)} entities"
        )
        logger.info(
            f"Cache enabled: {deps.semantic_cache_enabled and state.cache is not None}"
        )

        # Process all entities concurrently (unlimited parallelism - I/O bound)
        assessment_tasks = []
        entity_names = []
        for entity_name, entity in state.entities_found.items():
            assessment_tasks.append(self._assess_one_entity(entity, ctx))
            entity_names.append(entity_name)

        results = await asyncio.gather(*assessment_tasks, return_exceptions=True)

        # Store results in dict for O(1) lookup
        for entity_name, result in zip(entity_names, results):
            if isinstance(result, Exception):
                logger.error(f"Assessment failed for {entity_name}: {result}")
                continue
            state.individual_assessments[entity_name] = result

        logger.info(
            f"Assessment complete: {len(state.individual_assessments)} entities assessed"
        )
        logger.info(
            f"Cache stats - hits: {state.metrics.cache_hits_assessment}, "
            f"misses: {state.metrics.cache_misses_assessment}"
        )

        # Save checkpoint if callback provided
        if hasattr(deps, "checkpoint_callback") and deps.checkpoint_callback:
            await deps.checkpoint_callback("assessment", state)

    async def _assess_one_entity(
        self,
        entity: EntityWithQuotes,
        ctx: GraphRunContext[ExtractionStateV3, ExtractionDepsV3],
    ) -> IndividualAssessment:
        """
        Assess a single entity's relationship potential with caching.

        Checks semantic cache before calling LLM, validates evidence quotes
        using fuzzy matching, and converts potential level to confidence score.

        Args:
            entity: Entity to assess
            ctx: Graph context with state and deps

        Returns:
            IndividualAssessment with validated evidence and related entities
        """
        state, deps = ctx.state, ctx.deps
        cache = state.cache

        logger.debug(f"Assessing entity: {entity.name} ({entity.kind})")

        # Compute cache key
        model_str = str(deps.model) if hasattr(deps.model, "__str__") else deps.model
        cache_key = compute_assessment_cache_key(
            entity_name=entity.name,
            entity_kind=entity.kind,
            task_context=deps.get_task_context(),
            model_version=model_str,
            prompt_version="v3_2025-10",  # Update when prompts change
        )

        # Check cache if enabled
        if cache and deps.semantic_cache_enabled:
            cached = await cache.get_assessment(cache_key)
            if cached is not None:
                state.metrics.record_cache_hit("assessment")
                logger.debug(f"Cache hit for {entity.name}")
                return cached

            state.metrics.record_cache_miss("assessment")

        # Cache miss - assess entity via LLM
        logger.debug(f"Cache miss for {entity.name} - calling LLM")

        # Get contexts from entity quotes (limit to 10 for token efficiency)
        contexts = entity.all_contexts[:10]

        # Create assessment agent
        agent = create_assessment_agent_v3(
            model=deps.model,
            entity_kinds=deps.get_entity_kinds(),
            relationship_type=deps.get_relation_type(),
        )

        # Build prompt with entity contexts
        prompt = self._build_assessment_prompt(entity, contexts, deps)

        # Run assessment with timing
        start_time = time.time()
        try:
            result = await agent.run(prompt, deps=deps)
            duration = time.time() - start_time

            state.metrics.assessment_calls += 1
            state.metrics.assessment_successes += 1
            state.metrics.assessment_time += duration

            logger.debug(
                f"Assessed {entity.name} in {duration:.2f}s: "
                f"potential={result.potential}, related={len(result.related)}"
            )
        except Exception as e:
            duration = time.time() - start_time
            state.metrics.assessment_calls += 1
            state.metrics.assessment_time += duration

            logger.error(f"Assessment failed for {entity.name}: {e}")
            # Return minimal assessment on failure
            return IndividualAssessment(
                entity=entity,
                relationship_potential="none",
                related_entities=[],
                evidence_quotes=[],
                reasoning=f"Assessment failed: {str(e)}",
                confidence=0.0,
            )

        # Convert to IndividualAssessment with validated quotes
        assessment = self._create_assessment_with_quotes(entity, result, state)

        # Save to cache if enabled
        if cache and deps.semantic_cache_enabled:
            await cache.set_assessment(cache_key, assessment)
            logger.debug(f"Saved assessment to cache for {entity.name}")

        return assessment

    def _build_assessment_prompt(
        self,
        entity: EntityWithQuotes,
        contexts: List[str],
        deps: ExtractionDepsV3,
    ) -> str:
        """
        Build assessment prompt with entity contexts.

        Constructs a prompt emphasizing the need for specific entity names
        in the related_entities field (not entity types).

        Args:
            entity: Entity being assessed
            contexts: Context excerpts where entity appears
            deps: Dependencies with task configuration

        Returns:
            Formatted prompt string for assessment agent
        """
        contexts_text = "\n\n".join(
            f"Context {i + 1}:\n{ctx}" for i, ctx in enumerate(contexts[:10])
        )

        return f"""Entity: {entity.name} ({entity.kind})

Contexts where this entity appears:
{contexts_text}

Assess the relationship potential of this entity for {deps.get_relation_type()}.

IMPORTANT: In the 'related' field, list specific entity names (not types) that this entity might relate to.
Examples:
- Good: "BRCA1", "breast cancer", "TP53"
- Bad: "genes", "diseases", "proteins"

These specific names will be used to generate pair candidates for evaluation.
"""

    def _create_assessment_with_quotes(
        self,
        entity: EntityWithQuotes,
        agent_output: AssessmentOut,
        state: ExtractionStateV3,
    ) -> IndividualAssessment:
        """
        Convert agent output to IndividualAssessment with validated quotes.

        Validates evidence quotes using fuzzy matching across all resources
        in the resource pool. Maps potential level to confidence score.

        Args:
            entity: Entity being assessed
            agent_output: Raw agent output
            state: Extraction state with resource pool

        Returns:
            IndividualAssessment with validated evidence quotes
        """
        # Validate evidence quotes by searching all resources
        evidence_quotes = []
        for evidence_text in agent_output.evidence:
            if not evidence_text or not evidence_text.strip():
                continue

            evidence_text = evidence_text.strip()

            # Search all resources for this evidence
            for resource in state.resource_pool.resources:
                quote = find_quote_with_fuzzy_matching(
                    resource=resource,
                    quote_text=evidence_text,
                    auto_correct_threshold=0.90,
                    suggest_threshold=0.75,
                )
                if quote and not isinstance(quote, FuzzySuggestion):
                    # Found valid quote in this resource
                    evidence_quotes.append(quote)
                    logger.debug(
                        f"Validated evidence for {entity.name}: {evidence_text[:50]}..."
                    )
                    break  # Found in this resource, move to next evidence

        # Map potential to confidence score
        confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.5, "none": 0.0}
        confidence = confidence_map.get(agent_output.potential, 0.5)

        return IndividualAssessment(
            entity=entity,
            relationship_potential=agent_output.potential,
            related_entities=agent_output.related,  # CRITICAL for task 07
            evidence_quotes=evidence_quotes,
            reasoning=agent_output.reasoning,
            confidence=confidence,
        )


@dataclass
class GeneratePairCandidates(BaseNode[ExtractionStateV3, ExtractionDepsV3]):
    """
    Generate candidate entity pairs using hybrid strategy.

    Combines tiered co-occurrence analysis (chunk-based, adjacent, document-level)
    with assessment-suggested pairs to intelligently reduce the N² pair space.
    This is the key innovation separating V3 from V2.

    Phase 3a of V3 pipeline (before EvaluatePairs in task 08).
    """

    async def run(
        self, ctx: GraphRunContext[ExtractionStateV3, ExtractionDepsV3]
    ) -> None:
        """
        Generate pair candidates using co-occurrence and assessment suggestions.

        Orchestrates candidate generation from both strategies, deduplicates,
        filters same-kind pairs if configured, and saves checkpoint.
        """
        state, deps = ctx.state, ctx.deps

        logger.info(
            f"Starting pair candidate generation from {len(state.entities_found)} entities"
        )
        logger.info(
            f"Co-occurrence strategy: same_chunk={deps.enable_same_chunk}, "
            f"adjacent={deps.enable_adjacent_chunks}, document={deps.enable_document_level}"
        )

        candidates = {}

        # Strategy 1: Co-occurrence based
        cooccurrence_candidates = self._find_cooccurrence_pairs(state, deps)
        for candidate in cooccurrence_candidates:
            key = self._make_pair_key(candidate.entity_a.name, candidate.entity_b.name)
            candidates[key] = candidate
            state.metrics.candidates_from_cooccurrence += 1

        logger.info(
            f"Generated {len(cooccurrence_candidates)} candidates from co-occurrence"
        )

        # Strategy 2: Assessment-suggested
        assessment_candidates = self._find_assessment_suggested_pairs(state, deps)
        for candidate in assessment_candidates:
            key = self._make_pair_key(candidate.entity_a.name, candidate.entity_b.name)
            if key in candidates:
                # Mark as "both" - found by both strategies
                candidates[key].generation_strategy = "both"
            else:
                candidates[key] = candidate
            state.metrics.candidates_from_assessment += 1

        logger.info(
            f"Generated {len(assessment_candidates)} candidates from assessments"
        )

        # Filter same-kind pairs if configured
        if not deps.include_same_kind_pairs:
            before_filter = len(candidates)
            candidates = {
                k: v
                for k, v in candidates.items()
                if v.entity_a.kind != v.entity_b.kind
            }
            filtered_count = before_filter - len(candidates)
            if filtered_count > 0:
                logger.info(f"Filtered {filtered_count} same-kind pairs")

        state.pair_candidates = candidates
        state.metrics.candidates_generated = len(candidates)

        logger.info(f"Total candidates generated: {len(candidates)}")

        # Save checkpoint if callback provided
        if hasattr(deps, "checkpoint_callback") and deps.checkpoint_callback:
            await deps.checkpoint_callback("candidates", state)

    def _find_cooccurrence_pairs(
        self, state: ExtractionStateV3, deps: ExtractionDepsV3
    ) -> List["PairCandidate"]:
        """
        Find entity pairs using tiered chunk-based co-occurrence.

        Implements three tiers of co-occurrence:
        - Tier 1: Same semantic chunk (70% of relations, 59% precision)
        - Tier 2: Adjacent chunks ±1 (25% of relations, 48% precision)
        - Tier 3: Document-level (remaining, lower precision, optional)

        Args:
            state: Extraction state with entities and resource pool
            deps: Dependencies with co-occurrence configuration

        Returns:
            List of PairCandidate objects from co-occurrence analysis
        """
        from collections import defaultdict
        from .models import PairCandidate

        candidates = {}

        # Build resource → entities mapping
        resource_to_entities = defaultdict(list)
        for entity in state.entities_found.values():
            for quote in entity.quotes:
                resource_to_entities[quote.resource.id].append(entity)

        # For each resource, find co-occurring pairs
        for resource_id, entities in resource_to_entities.items():
            for i, entity_a in enumerate(entities):
                for entity_b in entities[i + 1 :]:
                    # Skip same-kind pairs if configured
                    if (
                        entity_a.kind == entity_b.kind
                        and not deps.include_same_kind_pairs
                    ):
                        continue

                    # Get chunk indices for each entity in this resource
                    a_chunks = set()
                    for quote in entity_a.quotes:
                        if quote.resource.id == resource_id:
                            a_chunks.update(quote.chunk_indices)

                    b_chunks = set()
                    for quote in entity_b.quotes:
                        if quote.resource.id == resource_id:
                            b_chunks.update(quote.chunk_indices)

                    # Check tiered co-occurrence
                    co_occurs = False
                    strategy = None

                    # Tier 1: Same chunk (highest precision)
                    if deps.enable_same_chunk and (a_chunks & b_chunks):
                        co_occurs = True
                        strategy = "same_chunk"

                    # Tier 2: Adjacent chunks (±1)
                    if not co_occurs and deps.enable_adjacent_chunks:
                        for a_idx in a_chunks:
                            for b_idx in b_chunks:
                                if abs(a_idx - b_idx) == 1:
                                    co_occurs = True
                                    strategy = "adjacent_chunks"
                                    break
                            if co_occurs:
                                break

                    # Tier 3: Document-level (optional, lowest precision)
                    if not co_occurs and deps.enable_document_level:
                        co_occurs = True
                        strategy = "document_level"

                    # Create or update candidate
                    if co_occurs:
                        key = tuple(sorted([entity_a.name, entity_b.name]))
                        if key in candidates:
                            # Same pair found in another resource
                            candidates[key].co_occurrence_count += 1
                            candidates[key].shared_resources.append(resource_id.id)
                        else:
                            candidates[key] = PairCandidate(
                                entity_a=entity_a,
                                entity_b=entity_b,
                                co_occurrence_count=1,
                                shared_resources=[resource_id.id],
                                generation_strategy=strategy,
                            )

        return list(candidates.values())

    def _find_assessment_suggested_pairs(
        self, state: ExtractionStateV3, deps: ExtractionDepsV3
    ) -> List["PairCandidate"]:
        """
        Find pairs suggested by individual assessments.

        Uses the related_entities field from each assessment to identify
        potential relationships. Fuzzy matches suggested names to actual
        entities found.

        Args:
            state: Extraction state with assessments and entities
            deps: Dependencies with pairing configuration

        Returns:
            List of PairCandidate objects from assessment suggestions
        """
        from .models import PairCandidate

        candidates = []

        for entity_name, assessment in state.individual_assessments.items():
            # Skip low/none potential entities
            if assessment.relationship_potential in ["low", "none"]:
                continue

            # Match related_entities to actual entities_found
            for related_name in assessment.related_entities:
                matched_entity = self._fuzzy_match_entity(
                    related_name, state.entities_found
                )
                if not matched_entity:
                    logger.debug(
                        f"Could not match related entity '{related_name}' "
                        f"suggested by {entity_name}"
                    )
                    continue

                # Skip same-kind pairs if configured
                if (
                    assessment.entity.kind == matched_entity.kind
                    and not deps.include_same_kind_pairs
                ):
                    continue

                candidates.append(
                    PairCandidate(
                        entity_a=assessment.entity,
                        entity_b=matched_entity,
                        co_occurrence_count=0,  # Not from co-occurrence
                        shared_resources=[],
                        generation_strategy="assessment_suggested",
                    )
                )

        return candidates

    def _fuzzy_match_entity(
        self, related_name: str, entities_found: Dict[str, EntityWithQuotes]
    ) -> Optional[EntityWithQuotes]:
        """
        Fuzzy match related entity name to actual entities.

        Tries in order:
        1. Exact match (case-sensitive)
        2. Case-insensitive match
        3. Alias match (case-sensitive)
        4. Alias match (case-insensitive)

        Args:
            related_name: Entity name from assessment's related_entities
            entities_found: Dict of all entities found (name → entity)

        Returns:
            Matched EntityWithQuotes or None if no match found
        """
        # Exact match (case-sensitive)
        if related_name in entities_found:
            return entities_found[related_name]

        # Case-insensitive match
        related_lower = related_name.lower()
        for name, entity in entities_found.items():
            if name.lower() == related_lower:
                return entity

        # Alias match (case-sensitive)
        for entity in entities_found.values():
            if related_name in entity.aliases:
                return entity

        # Alias match (case-insensitive)
        for entity in entities_found.values():
            if related_lower in [alias.lower() for alias in entity.aliases]:
                return entity

        return None

    def _make_pair_key(self, name_a: str, name_b: str) -> tuple[str, str]:
        """
        Create normalized pair key for deduplication.

        Args:
            name_a: First entity name
            name_b: Second entity name

        Returns:
            Sorted tuple of entity names for consistent lookup
        """
        return tuple(sorted([name_a, name_b]))
