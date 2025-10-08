"""
Pipeline nodes for extraction graph V3.

Implements the three-phase extraction pipeline:
1. ExtractEntities: Extract entities from full documents with semantic caching
2. AssessIndividually: Assess each entity's relationship potential (task 06)
3. GeneratePairs: Generate and evaluate entity pairs (tasks 07-08)
"""

import logging
import time
from dataclasses import dataclass
from typing import List, TYPE_CHECKING

from pydantic_graph import BaseNode, GraphRunContext

from .state import ExtractionStateV3
from .deps import ExtractionDepsV3
from .agents import create_entity_extractor_v3
from .cache import compute_extraction_cache_key
from ..extraction_graph_v2.models import EntityWithQuotes, SimpleEntityListOut
from ..extraction_graph_v2.parallelism import with_parallelism_control
from ..resources import (
    ResourceId,
    Resource,
    find_quote_with_fuzzy_matching,
    FuzzySuggestion,
)

if TYPE_CHECKING:
    pass

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
