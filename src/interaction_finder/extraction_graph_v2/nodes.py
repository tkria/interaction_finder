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

    async def run(
        self, ctx: GraphRunContext[ExtractionState, ExtractionDeps]
    ) -> "AssessIndividually":
        """
        Extract entities using LLM agent and create ResourceQuotes.

        Phase 3 implementation: Real LLM-powered entity extraction.
        """
        logger.info("Starting LLM entity extraction with ResourceQuote creation")

        # Get configuration
        entity_kinds = ctx.deps.get_entity_kinds()
        task_context = ctx.deps.get_task_context()
        logger.info(f"Looking for entity kinds: {entity_kinds}")

        # Build prompt with current group's document content
        group_resources = ctx.state.get_current_group_resources()
        logger.info(
            f"Processing group {ctx.state.current_group_index + 1} with {len(group_resources)} documents"
        )

        # Format documents with resource IDs for the LLM
        all_text_parts = []
        for resource in group_resources:
            doc_header = f"Resource {resource.id.id}: {resource.title}"
            all_text_parts.append(f"[{doc_header}]\n{resource.text}")
        all_text = "\n\n---\n\n".join(all_text_parts)

        # Build prompt to extract ALL entities (target term is handled by agent internally)
        prompt = f"Extract ALL {', '.join(entity_kinds)} entities from the following scientific text. Documents are labeled with Resource IDs - use these exact IDs in your output:\n\n{all_text}"

        # Create and run entity extraction agent
        logger.info("Running entity extraction agent")
        target_term = getattr(ctx.deps, "target_term", None)
        agent = create_entity_extractor(
            ctx.deps.model, entity_kinds, task_context, target_term
        )

        # Set current resources in deps for validation
        ctx.deps.current_resources = group_resources

        # Record timing and success
        start_time = time.time()
        try:
            result = await agent.run(prompt, deps=ctx.deps)
            agent_duration = time.time() - start_time
            ctx.state.metrics.record_extraction_call(
                success=True, duration=agent_duration
            )
            logger.info(f"Agent extracted {len(result.output.entities)} entities")
        except Exception as e:
            agent_duration = time.time() - start_time
            ctx.state.metrics.record_extraction_call(
                success=False, duration=agent_duration
            )
            logger.error(f"Entity extraction agent failed: {e}")
            # Continue with empty entities rather than crash
            result = None

        # Convert agent output to ResourceQuotes
        entities_found = 0
        if result and result.output:
            # Debug: Log what the agent returned
            logger.info(
                f"Agent output - entities: {len(result.output.entities)}, reasoning length: {len(result.output.reasoning)}"
            )

            if not result.output.entities:
                logger.warning("Agent returned no entities in the entities list")
                if result.output.reasoning:
                    logger.info(
                        f"Agent reasoning (first 200 chars): {result.output.reasoning[:200]}..."
                    )
                    # Look for potential entities mentioned in reasoning
                    reasoning_text = result.output.reasoning
                    if len(reasoning_text) > 50:  # If reasoning is substantial
                        logger.warning(
                            "Entities may have been incorrectly placed in reasoning field instead of entities list"
                        )

            for entity_data in result.output.entities:
                entity_name = entity_data.name
                entity_kind = entity_data.kind
                llm_quotes = entity_data.quotes

                # Process LLM-provided quotes to create ResourceQuotes
                # Since validation now ensures quotes are valid, this should succeed
                quotes = []
                if llm_quotes:
                    quotes = self._process_llm_quotes(
                        entity_name, llm_quotes, group_resources
                    )
                    if quotes:
                        logger.info(
                            f"Created ResourceQuotes for '{entity_name}' from validated LLM quotes: {len(quotes)} total quotes"
                        )

                # Minimal fallback for edge cases that slip through validation
                if not quotes:
                    logger.warning(
                        f"Unexpected: no quotes after validation for '{entity_name}', trying exact match fallback"
                    )
                    for resource in group_resources:
                        try:
                            quote = resource.quote(entity_name)
                            if quote and quote.count > 0:
                                quotes.append(quote)
                                logger.info(
                                    f"Fallback: Created ResourceQuote for '{entity_name}' with {quote.count} occurrences in {resource.id.url}"
                                )
                                break  # Only need one successful match
                        except ValueError:
                            continue

                if quotes:
                    # Create EntityWithQuotes and store in state
                    entity = EntityWithQuotes(
                        name=entity_name,
                        kind=entity_kind,
                        quotes=quotes,
                        confidence=0.8,  # Default confidence since not in EntityOut model
                    )

                    # Validate the entity has valid quotes and record metrics
                    ctx.state.metrics.record_entity_validation(entity)
                    if entity.validate():
                        ctx.state.entities_found[entity_name] = entity
                        entities_found += 1
                        logger.info(
                            f"Successfully added entity '{entity_name}' with {entity.total_occurrences} total occurrences"
                        )
                    else:
                        logger.warning(f"Entity '{entity_name}' failed validation")
                else:
                    logger.warning(
                        f"No ResourceQuotes found for LLM-extracted entity '{entity_name}'"
                    )
        else:
            logger.warning("No entities extracted by LLM or extraction failed")

        logger.info(
            f"Entity extraction complete: {entities_found} entities with valid quotes"
        )

        # Control flow decision stays in node
        if not ctx.state.entities_found:
            raise ValueError("No entities found with verifiable quotes")

        return AssessIndividually()

    def _process_llm_quotes(self, entity_name, llm_quotes, group_resources):
        """
        Process LLM-provided quotes to create ResourceQuotes.

        Args:
            entity_name: Name of the entity
            llm_quotes: List of quotes from LLM with text and source
            group_resources: List of Resource objects from current group

        Returns:
            List of ResourceQuote objects
        """
        quotes = []

        # Create a mapping of resource identifiers to resources
        source_to_resource = {}
        for resource in group_resources:
            # Map various formats the LLM might use for the resource
            resource_id = resource.id.id
            title = resource.title

            source_to_resource[resource_id] = resource
            source_to_resource[title] = resource
            source_to_resource[f"Resource {resource_id}"] = resource
            source_to_resource[f"Resource {resource_id}: {title}"] = resource

        for quote_data in llm_quotes:
            quote_text = quote_data.text.strip()
            quote_source = quote_data.source.strip()

            # Extract resource ID from various possible formats
            original_source = quote_source
            if quote_source.startswith("Resource "):
                quote_source = quote_source[9:]  # Remove "Resource " (9 characters)
                # Also strip any trailing colon and title if present
                if ":" in quote_source:
                    quote_source = quote_source.split(":")[0].strip()

            if not quote_text:
                logger.warning(f"Empty quote text for entity '{entity_name}', skipping")
                continue

            logger.debug(
                f"Looking for source '{quote_source}' (original: '{original_source}') in mapping with keys: {list(source_to_resource.keys())}"
            )

            # Find the resource by matching the source
            resource = source_to_resource.get(quote_source)
            if not resource:
                logger.warning(
                    f"Could not find resource for source '{quote_source}' (original: '{original_source}'), trying fuzzy match"
                )
                # Try fuzzy matching on source
                for source_key, res in source_to_resource.items():
                    if quote_source in source_key or source_key in quote_source:
                        resource = res
                        break

            if resource:
                # Try to create ResourceQuote directly
                try:
                    resource_quote = resource.quote(quote_text)
                    if resource_quote and resource_quote.count > 0:
                        quotes.append(resource_quote)
                        logger.info(
                            f"Created ResourceQuote from LLM quote for '{entity_name}': '{quote_text[:50]}...'"
                        )
                    else:
                        log_resourcequote_failure(
                            quote_text,
                            resource.id.id,
                            resource.id.url,
                            "ResourceQuote created but found no occurrences",
                        )
                except ValueError as e:
                    log_resourcequote_failure(
                        quote_text, resource.id.id, resource.id.url, str(e)
                    )
            else:
                logger.warning(
                    f"Could not match source '{quote_source}' to any document"
                )

        return quotes


@dataclass
class AssessIndividually(BaseNode[ExtractionState, ExtractionDeps]):
    """
    Assess each entity's relationship potential individually.

    This captures the key insight from the original graph:
    process each entity with its specific context.
    """

    async def run(
        self, ctx: GraphRunContext[ExtractionState, ExtractionDeps]
    ) -> "AggregateIntoPairs":
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

        return AggregateIntoPairs()

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
                        if quote and quote.count > 0:
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
class AggregateIntoPairs(
    BaseNode[ExtractionState, ExtractionDeps, list[EntityPairOut]]
):
    """
    Combine individual assessments into verified pairs.

    This is the final node that produces the output pairs
    with complete ResourceQuote provenance.
    """

    async def run(
        self, ctx: GraphRunContext[ExtractionState, ExtractionDeps]
    ) -> End[list[EntityPairOut]]:
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

        return End(ctx.state.final_pairs)

    def _find_existing_pair(
        self, existing_pairs: list[EntityPairOut], name_a: str, name_b: str
    ) -> EntityPairOut | None:
        """Find if a pair already exists (order-insensitive)."""
        for pair in existing_pairs:
            pair_names = {pair.entity_a.name, pair.entity_b.name}
            if pair_names == {name_a, name_b}:
                return pair
        return None
