"""Process all documents concurrently through the per-document pipeline.

For each document (in parallel):
1. Extract entities with quote validation
2. Filter by target entity kinds
3. Identify proximal entity sets
4. Extract pairs from proximal regions
5. Deduplicate and assess pairs

Documents that fail processing are logged but don't block other documents.
"""

import asyncio

from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.document_pipeline import (
    analyze_document,
    assess_document_pairs,
    extract_document_pairs,
    validate_entity_kinds,
)
from interaction_finder.extraction.models import EntityRef
from interaction_finder.extraction.shared import save_checkpoint
from interaction_finder.extraction.state import State
from interaction_finder.extraction.utils import identify_proximal_sets
from interaction_finder.logging import logfire
from interaction_finder.resources import Resource


async def process_documents(state: State, deps: Deps) -> bool:
    """Process all resources concurrently.

    Returns False if no entities were extracted (pipeline should abort).
    """
    with logfire.span("process_documents"):
        resources = deps.resource_pool.resources
        if not resources:
            deps.logger.warning("No resources to process")
            return False
        # Set up progress tracking
        if deps.progress:
            deps.progress["Processed"].total = len(resources)
            deps.progress["Processed"].activate()
            deps.progress.set_status("Extracting entities")
        # Process all documents in parallel, tracking progress as they complete
        tasks = [_process_document(resource, state, deps) for resource in resources]
        for coro in asyncio.as_completed(tasks):
            await coro
        # Mark document processing phase complete
        if deps.progress:
            deps.progress["Processed"].complete()
            deps.progress["Pairs assessed"].complete()
        # Check if we found any validated entities
        if not state.validated_entities_by_resource:
            deps.logger.warning("No entities extracted from documents")
            return False
        # Save partial checkpoint after expensive document processing stage
        await save_checkpoint(state, deps, "process_documents")
        return True


async def _process_document(resource: Resource, state: State, deps: Deps):
    """Process a single document through the full per-document pipeline."""
    with logfire.span(
        "Document {resource_id}",
        resource_id=resource.id.id,
        title=resource.title[:60],
        url=resource.id.url,
    ):
        try:
            # Stage 1: Analyze document (quality assessment + entity extraction)
            (
                entities,
                quality_assessment,
                quotes_validated,
                quotes_failed,
            ) = await analyze_document(
                resource,
                state.topic,
                state.target_entity_types,
                deps.config,
                deps,
            )
            # Store paper quality assessment (even if entity extraction found nothing)
            if quality_assessment is not None:
                state.paper_quality[resource.id] = quality_assessment
            # Update quote counters (single-threaded async, += is atomic)
            state.quotes_validated += quotes_validated
            state.quotes_failed += quotes_failed
            if not entities:
                return  # No entities found in this document
            # Store raw entities
            state.entities_by_resource[resource.id] = entities
            # Stage 2: Validate entity kinds
            if deps.progress:
                deps.progress.set_status("Validating entities")
            validated = validate_entity_kinds(entities, state.target_entity_types)
            if not validated:
                return  # No valid entities after kind filtering
            # Store validated entities as EntityRefs (preserve original mentions)
            state.validated_entities_by_resource[resource.id] = {
                name: EntityRef(canonical=name, mentions=[mention])
                for name, mention in validated.items()
            }
            # Update progress
            if deps.progress:
                deps.progress["Entities"].completed = sum(
                    len(e) for e in state.entities_by_resource.values()
                )
                deps.progress["Quotes"].completed = state.quotes_validated
                if state.quotes_failed > 0:
                    deps.progress["Quotes"].note = f"({state.quotes_failed} invalid)"
            # Stage 3: Identify proximal entity sets
            if deps.progress:
                deps.progress.set_status("Finding proximal pairs")
            proximal_sets = identify_proximal_sets(
                validated,
                deps.config.stage.extraction.proximal_window_chunks,
                resource,
            )
            if not proximal_sets:
                return  # No co-occurring entities found
            state.proximal_sets_by_resource[resource.id] = proximal_sets
            # Stage 4: Extract pairs from proximal sets
            if deps.progress:
                deps.progress.set_status("Extracting relationships")
            pairs, pairs_validated, pairs_failed = await extract_document_pairs(
                proximal_sets,
                validated,
                resource,
                state.topic,
                state.permitted_pairs,
                deps.config.stage.extraction.region_padding_chunks,
                deps.config,
                deps,
            )
            # Update quote counters from pair extraction
            state.quotes_validated += pairs_validated
            state.quotes_failed += pairs_failed
            if not pairs:
                return  # No pairs found in proximal sets
            # Stage 5: Deduplicate and assess pairs
            if deps.progress:
                deps.progress.set_status("Assessing pairs")
            assessments = await assess_document_pairs(
                pairs,
                state.validated_entities_by_resource[resource.id],
                resource,
                state.topic,
                deps.config.stage.extraction.region_padding_chunks,
                deps.config,
                deps,
            )
            if assessments:
                state.pair_assessments_by_resource[resource.id] = assessments
        except Exception as e:
            # Log error but don't fail entire pipeline
            deps.logger.error(
                f"[red]Document processing failed for {resource.title}: {type(e).__name__}: {e}[/red]",
                extra={"markup": True},
            )
        finally:
            # Update counters when work completes (tight scoping, even on error)
            if deps.progress:
                deps.progress["Processed"].done()
