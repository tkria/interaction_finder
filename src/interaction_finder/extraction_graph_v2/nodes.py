"""
Simplified 3-node implementation for extraction graph V2.

Following pydantic-graph patterns with ResourceQuote integration
for complete provenance tracking.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_graph import BaseNode, GraphRunContext, End
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .state import ExtractionState
from .deps import ExtractionDeps
from .models import EntityWithQuotes, IndividualAssessment, EntityPairOut
from .agents import create_entity_extractor, create_assessment_agent
from .directextract import extract_cited_entities
from .parallelism import with_parallelism_control, ParallelismController
from .quote_logging import save_quote_errors_incremental

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)
console = Console()


def log_resourcequote_failure(
    quote_text: str, resource_id: str, resource_url: str, error: str = None
):
    """Log ResourceQuote failure with rich styling."""
    # Create the main content
    content = Text()
    content.append("ResourceQuote Creation Failed\n\n", style="bold red")

    content.append("Quote: ", style="bold")
    content.append(f'"{quote_text}"\n', style="yellow")

    content.append("Resource ID: ", style="bold")
    content.append(f"{resource_id}\n", style="cyan")

    content.append("Resource URL: ", style="bold")
    content.append(f"{resource_url}\n", style="blue")

    if error:
        content.append("Error: ", style="bold")
        content.append(error, style="red")

    # Create panel with the content
    panel = Panel(
        content,
        title="[bold red]Quote Processing Error[/bold red]",
        border_style="red",
        padding=(1, 2),
    )

    console.print(panel)


@dataclass
class ExtractEntities(BaseNode[ExtractionState, ExtractionDeps]):
    """
    Extract entities and create ResourceQuotes for each.

    This is the first node in our simplified 3-node pipeline.
    For now, we'll start with BRCA1 hard-coded to establish
    the end-to-end pipeline.
    """

    async def run(self, ctx: GraphRunContext[ExtractionState, ExtractionDeps]) -> None:
        """
        Extract entities using LLM agent and create ResourceQuotes.

        Processes each document individually for better reliability and simpler quote validation.
        """
        logger.info(
            "Starting individual document entity extraction with ResourceQuote creation"
        )

        # Get configuration
        entity_kinds = ctx.deps.get_entity_kinds()
        task_context = ctx.deps.get_task_context()
        target_term = getattr(ctx.deps, "target_term", None)
        logger.info(f"Looking for entity kinds: {entity_kinds}")

        # Get all documents directly (no grouping needed for individual processing)
        all_resources = list(ctx.state.resource_pool.resources)
        logger.info(f"Processing {len(all_resources)} documents individually")

        # Process each document individually
        document_results = await self._process_documents_individually(
            all_resources, entity_kinds, task_context, target_term, ctx
        )

        # Merge entities found across all documents
        entities_found = self._merge_document_entities(document_results, ctx)

        logger.info(
            f"ExtractEntities complete: {entities_found} unique entities found across all documents"
        )

    async def _process_documents_individually(
        self,
        resources,
        entity_kinds,
        task_context,
        target_term,
        ctx: GraphRunContext[ExtractionState, ExtractionDeps],
    ):
        """
        Process each document individually with configurable parallelism.

        Returns:
            List of (resource, extracted_entities) tuples
        """
        if not resources:
            return []

        # Create extraction function for each resource
        async def extract_from_resource(resource):
            return await self._extract_entities_from_document(
                resource, entity_kinds, task_context, target_term, ctx
            )

        # Use parallelism control from deps
        parallelism = ctx.deps.parallelism
        parallelism_desc = f"unlimited" if parallelism == 0 else f"limit={parallelism}"
        logger.info(
            f"Processing {len(resources)} documents with parallelism {parallelism_desc}"
        )

        # Process documents with parallelism control
        results = await with_parallelism_control(
            resources,
            extract_from_resource,
            parallelism=parallelism,
            description="document extraction",
        )

        # Handle results and exceptions
        document_results = []
        for i, result in enumerate(results):
            resource = resources[i]
            if isinstance(result, Exception):
                logger.error(
                    f"Document processing failed for {resource.title}: {result}"
                )
                continue
            if result:
                document_results.append((resource, result))

        return document_results

    async def _extract_entities_from_document(
        self,
        resource,
        entity_kinds,
        task_context,
        target_term,
        ctx: GraphRunContext[ExtractionState, ExtractionDeps],
    ):
        """
        Extract entities from a single document.

        Returns:
            List of EntityWithQuotes objects from this document
        """
        logger.debug(f"Processing document: {resource.title}")

        # Build focused prompt for this document
        prompt = f"Extract ALL {', '.join(entity_kinds)} entities from the following scientific document:\n\n{resource.text}"

        # Create agent for this extraction
        agent = create_entity_extractor(
            ctx.deps.model, entity_kinds, task_context, target_term
        )

        # Set current resource for validation (single document)
        ctx.deps.current_resources = [resource]

        # Run extraction with timing
        start_time = time.time()
        try:
            result = await agent.run(prompt, deps=ctx.deps)
            agent_duration = time.time() - start_time
            ctx.state.metrics.record_extraction_call(
                success=True, duration=agent_duration
            )
            logger.debug(
                f"Extracted {len(result.output.entities)} entities from {resource.title}"
            )
        except Exception as e:
            agent_duration = time.time() - start_time
            ctx.state.metrics.record_extraction_call(
                success=False, duration=agent_duration
            )
            logger.error(f"Entity extraction failed for {resource.title}: {e}")
            return []

        # Convert to EntityWithQuotes objects
        document_entities = []
        if result and result.output and result.output.entities:
            for entity_data in result.output.entities:
                # Create quotes directly from this single document
                quotes = self._create_quotes_from_document(entity_data, resource)

                if quotes:
                    entity = EntityWithQuotes(
                        name=entity_data.name,
                        kind=entity_data.kind,
                        aliases=entity_data.aliases,
                        quotes=quotes,
                        confidence=0.8,
                    )

                    # Validate entity
                    if entity.validate():
                        document_entities.append(entity)
                        logger.debug(
                            f"Created entity '{entity.name}' with {len(quotes)} quotes from {resource.title}"
                        )
                    else:
                        logger.warning(f"Entity '{entity.name}' failed validation")

        # Save quote errors incrementally after processing this document
        if ctx.deps.output_dir and ctx.deps.quote_error_log:
            # Only save new errors since last save
            ctx.deps.saved_error_count = save_quote_errors_incremental(
                ctx.deps.quote_error_log,
                ctx.deps.output_dir,
                append_mode=True,
                saved_count=ctx.deps.saved_error_count,
            )

        return document_entities

    def _create_quotes_from_document(self, entity_data, resource):
        """
        Create ResourceQuotes from entity data for a single document.

        Much simpler than the old _process_llm_quotes since we only have one document.
        """
        quotes = []

        # Process quotes directly from the simple list structure
        if entity_data.quotes:
            for quote_text in entity_data.quotes:
                if not quote_text or not quote_text.strip():
                    continue

                quote_text = quote_text.strip()
                try:
                    resource_quote = resource.quote(quote_text)
                    quotes.append(resource_quote)
                    logger.debug(
                        f"Created quote for '{entity_data.name}': '{quote_text[:50]}...'"
                    )
                except ValueError as e:
                    log_resourcequote_failure(
                        quote_text, resource.id.id, resource.id.url, str(e)
                    )

        return quotes

    def _merge_document_entities(self, document_results, ctx):
        """
        Merge entities found across multiple documents, combining quotes.

        Args:
            document_results: List of (resource, entities_list) tuples
            ctx: Graph context

        Returns:
            Number of unique entities found
        """
        entity_map = {}  # name -> EntityWithQuotes

        for resource, entities in document_results:
            for entity in entities:
                entity_name = entity.name

                if entity_name in entity_map:
                    # Merge with existing entity
                    existing = entity_map[entity_name]
                    # Combine quotes from both entities
                    existing.quotes.extend(entity.quotes)
                    # Combine aliases
                    existing.aliases = list(set(existing.aliases + entity.aliases))
                    logger.debug(
                        f"Merged entity '{entity_name}' - now has {len(existing.quotes)} total quotes"
                    )
                else:
                    # New entity
                    entity_map[entity_name] = entity

        # Store merged entities in state
        entities_found = 0
        for entity_name, entity in entity_map.items():
            # Record final validation
            ctx.state.metrics.record_entity_validation(entity)
            if entity.validate():
                ctx.state.entities_found[entity_name] = entity
                entities_found += 1
                logger.info(
                    f"Final entity '{entity_name}' with {entity.total_occurrences} total occurrences"
                )
            else:
                logger.warning(f"Final entity '{entity_name}' failed validation")

        return entities_found


@dataclass
class AssessIndividually(BaseNode[ExtractionState, ExtractionDeps]):
    """
    Assess each entity's relationship potential individually.

    This captures the key insight from the original graph:
    process each entity with its specific context.
    """

    async def run(self, ctx: GraphRunContext[ExtractionState, ExtractionDeps]) -> None:
        """
        Process each entity individually with its contexts.

        Phase 2 implementation: Mock assessments to test aggregation.
        """
        logger.info(
            f"Starting individual assessment of {len(ctx.state.entities_found)} entities"
        )

        # Create assessment tasks for each entity
        tasks = []
        for entity_name, entity in ctx.state.entities_found.items():
            tasks.append(self._assess_one_entity(entity, ctx))

        # Run assessments concurrently
        try:
            assessments = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error during parallel assessment: {e}")
            raise

        # Process results
        valid_assessments = []
        for i, assessment in enumerate(assessments):
            entity_name = list(ctx.state.entities_found.keys())[i]

            if isinstance(assessment, Exception):
                logger.error(f"Assessment failed for {entity_name}: {assessment}")
                continue

            if assessment and assessment.relationship_potential != "none":
                valid_assessments.append(assessment)
                logger.info(
                    f"Assessment for {entity_name}: {assessment.relationship_potential} potential"
                )
            else:
                logger.info(f"No relationships found for {entity_name}")

        ctx.state.individual_assessments = valid_assessments
        logger.info(
            f"Individual assessment complete: {len(valid_assessments)} entities with relationship potential"
        )

    async def _assess_one_entity(
        self,
        entity: EntityWithQuotes,
        ctx: GraphRunContext[ExtractionState, ExtractionDeps],
    ) -> IndividualAssessment:
        """
        Assess a single entity with its specific context using LLM agent.

        Phase 3: Real LLM-powered assessment of relationship potential.
        """
        logger.debug(f"Assessing entity: {entity.name}")

        # Get all contexts where this entity appears
        contexts = entity.all_contexts
        logger.debug(f"Found {len(contexts)} contexts for {entity.name}")

        # Build prompt with entity and its contexts
        entity_kinds = ctx.deps.get_entity_kinds()
        contexts_text = "\n\n".join(
            [f"Context {i + 1}: {context}" for i, context in enumerate(contexts)]
        )

        prompt = f"""Assess the relationship potential for the entity "{entity.name}" (type: {entity.kind}).

ENTITY TO ASSESS: {entity.name}
ENTITY TYPE: {entity.kind}
LOOKING FOR RELATIONSHIPS WITH: {", ".join(entity_kinds)}

CONTEXTS WHERE "{entity.name}" APPEARS:
{contexts_text}

Analyze these contexts to determine if "{entity.name}" has potential relationships with other {", ".join(entity_kinds)} entities."""

        # Create and run assessment agent
        agent = create_assessment_agent(ctx.deps.model, entity_kinds)

        start_time = time.time()
        try:
            result = await agent.run(prompt, deps=ctx.deps)
            agent_duration = time.time() - start_time
            ctx.state.metrics.record_assessment_call(
                success=True, duration=agent_duration
            )
            logger.debug(
                f"Agent assessment for {entity.name}: {result.output.potential}"
            )
        except Exception as e:
            agent_duration = time.time() - start_time
            ctx.state.metrics.record_assessment_call(
                success=False, duration=agent_duration
            )
            logger.error(f"Assessment agent failed for {entity.name}: {e}")
            # Fall back to safe default
            assessment = IndividualAssessment(
                entity=entity,
                relationship_potential="none",
                related_entities=[],
                evidence_quotes=[],
                reasoning=f"Assessment failed: {str(e)}",
                confidence=0.0,
            )
            return assessment

        # Convert agent evidence strings to ResourceQuotes
        evidence_quotes = []
        evidence_requested = (
            len(result.output.evidence) if result.output.evidence else 0
        )
        evidence_found = 0

        if result.output.evidence:
            for evidence_text in result.output.evidence:
                # Try to find this evidence text in our resources
                for resource in ctx.state.resource_pool.resources:
                    try:
                        quote = resource.quote(evidence_text)
                        evidence_quotes.append(quote)
                        evidence_found += 1
                        logger.debug(
                            f"Found evidence quote for {entity.name}: '{evidence_text[:50]}...'"
                        )
                        break
                    except ValueError:
                        # Quote not found in this resource, try next
                        continue
                else:
                    logger.warning(
                        f"Could not locate evidence text in resources: '{evidence_text[:50]}...'"
                    )

        # Record evidence search metrics
        ctx.state.metrics.record_evidence_search(evidence_requested, evidence_found)

        # Map agent confidence levels
        confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.4, "none": 0.1}

        assessment = IndividualAssessment(
            entity=entity,
            relationship_potential=result.output.potential,
            related_entities=result.output.related,
            evidence_quotes=evidence_quotes,
            reasoning=result.output.reasoning,
            confidence=confidence_map[result.output.potential],
        )

        logger.debug(
            f"Assessment complete for {entity.name}: {result.output.potential} potential"
        )
        return assessment


@dataclass
class AggregateIntoPairs(BaseNode[ExtractionState, ExtractionDeps]):
    """
    Combine individual assessments into verified pairs.

    This is the final node that produces the output pairs
    with complete ResourceQuote provenance.
    """

    async def run(self, ctx: GraphRunContext[ExtractionState, ExtractionDeps]) -> None:
        """
        Aggregate individual assessments into entity pairs.
        """
        logger.info(
            f"Starting aggregation of {len(ctx.state.individual_assessments)} assessments"
        )

        pairs_created = 0

        # Smart pairing based on assessments
        for assessment in ctx.state.individual_assessments:
            if assessment.relationship_potential in ["high", "medium"]:
                # Find related entities that were also extracted
                for related_name in assessment.related_entities:
                    if related_entity := ctx.state.entities_found.get(related_name):
                        # Skip same-kind pairs (gene-gene, disease-disease)
                        if assessment.entity.kind == related_entity.kind:
                            logger.debug(
                                f"Skipping same-kind pair: {assessment.entity.name}-{related_name}"
                            )
                            continue

                        # Check if we already have this pair (avoid duplicates)
                        existing_pair = self._find_existing_pair(
                            ctx.state.final_pairs, assessment.entity.name, related_name
                        )
                        if existing_pair:
                            logger.debug(
                                f"Pair already exists: {assessment.entity.name}-{related_name}"
                            )
                            continue

                        # Create pair with full provenance
                        try:
                            confidence_map = {
                                "high": "high",
                                "medium": "medium",
                                "low": "low",
                            }
                            pair = EntityPairOut(
                                entity_a=assessment.entity,
                                entity_b=related_entity,
                                relationship=ctx.deps.get_relation_type(),
                                confidence=confidence_map[
                                    assessment.relationship_potential
                                ],
                                evidence_quotes=assessment.evidence_quotes,
                                reasoning=f"Individual assessment indicates {assessment.relationship_potential} relationship potential",
                            )

                            # Validate pair has evidence
                            if pair.validate_provenance():
                                ctx.state.final_pairs.append(pair)
                                pairs_created += 1
                                logger.info(
                                    f"Created pair: {assessment.entity.name} - {related_name} ({pair.confidence})"
                                )
                            else:
                                logger.warning(
                                    f"Pair failed provenance validation: {assessment.entity.name}-{related_name}"
                                )

                        except ValueError as e:
                            logger.error(
                                f"Failed to create pair {assessment.entity.name}-{related_name}: {e}"
                            )

        logger.info(
            f"Aggregation complete: {pairs_created} pairs created with full provenance"
        )

    def _find_existing_pair(
        self, existing_pairs: list[EntityPairOut], name_a: str, name_b: str
    ) -> EntityPairOut | None:
        """Find if a pair already exists (order-insensitive)."""
        for pair in existing_pairs:
            pair_names = {pair.entity_a.name, pair.entity_b.name}
            if pair_names == {name_a, name_b}:
                return pair
        return None
