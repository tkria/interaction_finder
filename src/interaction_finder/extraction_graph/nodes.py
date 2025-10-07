"""
Graph nodes for the extraction pipeline.

This module defines the control flow nodes that orchestrate the entity
extraction process, following the pydantic-graph pattern.
"""

import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any
from pydantic_graph import BaseNode, GraphRunContext, End

try:
    import logfire
except ImportError:
    # Create a no-op logfire if not available
    class _NoOpLogfire:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

    logfire = _NoOpLogfire()

from .deps import Deps
from .state import EntityExtractionState, DocumentGroup
from .models import EntityInfo, EntityAssessmentOut, IndividualEntityContext


def get_progress_tracker(state: EntityExtractionState):
    """Extract progress tracker from state if available."""
    return state.processing_notes.get("progress_tracker")


@dataclass
class InitialRouter(BaseNode[EntityExtractionState, Deps]):
    """
    Initial routing node that determines if content is suitable for extraction.

    Routes scientific content to EntityExtraction, terminates non-scientific content.
    """

    async def run(
        self, ctx: GraphRunContext[EntityExtractionState, Deps]
    ) -> "EntityExtraction" | End[str]:
        ctx.state.log_node_entry("InitialRouter")

        current_group = ctx.state.get_current_group()
        logfire.info(
            f"Starting InitialRouter for group with {len(current_group.urls) if current_group else 0} documents"
        )

        # Update progress tracker
        progress_tracker = get_progress_tracker(ctx.state)
        if progress_tracker:
            progress_tracker.update_node("InitialRouter")
            progress_tracker.update_stage("Classifying document content")

        if not current_group:
            logfire.warning("No document group to process")
            return End("No document group to process")

        # Get routing agent
        from .agents import create_router_agent, run_agent_with_tracking

        router_agent = create_router_agent(
            ctx.deps.model, ctx.deps.entity_kinds, ctx.deps.task_context
        )

        # Combine text from document group with attribution
        combined_text = current_group.combine_all_text(max_chars=20000)

        # Run classification
        prompt = f"""Classify the following document group for {", ".join(ctx.deps.entity_kinds)} extraction:

{combined_text}

Determine if this content is suitable for scientific entity extraction."""

        result = await run_agent_with_tracking(
            router_agent, prompt, str(ctx.deps.model), deps=ctx.deps
        )

        # Log decision
        ctx.state.add_processing_note(
            "routing_decision",
            {
                "classification": result.output.classification,
                "reasoning": result.output.reasoning,
                "confidence": result.output.confidence,
            },
        )

        if result.output.classification == "scientific":
            logfire.info(
                f"Content classified as scientific (confidence: {result.output.confidence})"
            )
            return EntityExtraction()
        else:
            logfire.info(
                f"Content classified as non-scientific: {result.output.reasoning}"
            )
            return End(f"Non-scientific content: {result.output.reasoning}")


@dataclass
class EntityExtraction(BaseNode[EntityExtractionState, Deps]):
    """
    Extract entities from the document group with full attribution.
    """

    async def run(
        self, ctx: GraphRunContext[EntityExtractionState, Deps]
    ) -> "ParallelEntityProcessor" | End[str]:
        ctx.state.log_node_entry("EntityExtraction")

        current_group = ctx.state.get_current_group()
        logfire.info(
            f"Starting EntityExtraction for {len(current_group.urls) if current_group else 0} documents"
        )

        # Update progress tracker
        progress_tracker = get_progress_tracker(ctx.state)
        if progress_tracker:
            progress_tracker.update_node("EntityExtraction")
            progress_tracker.update_stage("Extracting entities from documents")

        if not current_group:
            logfire.warning("No document group to process in EntityExtraction")
            return End("No document group to process")

        # Use direct extraction approach instead of agent-based approach
        from ..extraction_graph_v2.directextract import extract_cited_entities

        # Populate ResourcePool with document content for resource tracking
        resources = current_group.populate_resource_pool(ctx.deps.resource_pool)
        logfire.debug(f"Populated ResourcePool with {len(resources)} resources")

        # Convert resources to list format expected by directextract
        resource_list = list(resources.values())

        try:
            # Use directextract pipeline for complete extraction + validation
            entities_with_quotes, metrics = await extract_cited_entities(
                resource_list, ctx.deps.model, ctx.deps.entity_kinds
            )

            # Convert DirectExtraction results to v1 state format
            for entity in entities_with_quotes:
                # Convert to EntityInfo dict format expected by v1 pipeline
                entity_dict = {
                    "name": entity.name,
                    "kind": entity.kind,
                    "aliases": [],
                    "source_url": entity.quotes[0].source.url if entity.quotes else "",
                }

                ctx.state.add_entity(entity.kind, entity_dict)

            logfire.info(
                f"Direct extraction completed: {len(entities_with_quotes)} entities found"
            )

        except Exception as e:
            logfire.error(f"Direct entity extraction failed: {e}")
            return End(f"Direct entity extraction failed: {e}")

        # Check if any entities were found
        total_entities = ctx.state.get_entity_count()
        logfire.info(f"Entity extraction completed: {total_entities} entities found")

        if total_entities == 0:
            logfire.warning("No entities found in document group")
            return End("No entities found in document group")

        ctx.state.add_processing_note(
            "extraction_summary",
            {
                "total_entities": total_entities,
                "entities_by_kind": {
                    kind: ctx.state.get_entity_count(kind)
                    for kind in ctx.deps.entity_kinds
                },
                "documents_processed": len(current_group.urls),
            },
        )

        return ParallelEntityProcessor()


@dataclass
class ParallelEntityAssessment(BaseNode[EntityExtractionState, Deps]):
    """
    Assess entity relationships in parallel batches.
    """

    async def run(
        self, ctx: GraphRunContext[EntityExtractionState, Deps]
    ) -> "EntityPairGeneration":
        ctx.state.log_node_entry("ParallelEntityAssessment")

        # Get assessment agent
        from .agents import create_assessment_agent

        assessor_agent = create_assessment_agent(
            ctx.deps.model,
            ctx.deps.entity_kinds,
            ctx.deps.relation_type,
            ctx.deps.task_context,
        )

        # Handle binary entity relationships (e.g., gene-disease, celltype-biomarker)
        if len(ctx.deps.entity_kinds) == 2:
            await self._assess_binary_relationships(ctx, assessor_agent)
        else:
            # Handle other relationship patterns
            await self._assess_general_relationships(ctx, assessor_agent)

        ctx.state.add_processing_note(
            "assessment_summary",
            {
                "total_assessments": len(ctx.state.individual_assessments),
                "batch_size": batch_size,
            },
        )

        return EntityPairGeneration()

    async def _assess_binary_relationships(
        self, ctx: GraphRunContext[EntityExtractionState, Deps], agent
    ):
        """Assess relationships between two entity types."""
        kind_a, kind_b = ctx.deps.entity_kinds
        entities_a = ctx.state.extracted_entities.get(kind_a, [])
        entities_b = ctx.state.extracted_entities.get(kind_b, [])

        # Create assessment tasks for entities from same documents
        tasks = []
        for entity_a_dict in entities_a:
            for entity_b_dict in entities_b:
                # Only assess if entities are from the same document
                if entity_a_dict["source_url"] == entity_b_dict["source_url"]:
                    tasks.append((entity_a_dict, entity_b_dict))

        # Process in batches using configured batch size
        batch_size = ctx.deps.batch_size
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i : i + batch_size]
            batch_results = await asyncio.gather(
                *[
                    self._assess_entity_pair(ctx, agent, entity_a, entity_b)
                    for entity_a, entity_b in batch
                ],
                return_exceptions=True,
            )

            # Store successful assessments
            for result in batch_results:
                if isinstance(result, dict):
                    ctx.state.individual_assessments.append(result)

    async def _assess_general_relationships(
        self, ctx: GraphRunContext[EntityExtractionState, Deps], agent
    ):
        """Assess relationships for general entity patterns."""
        # Simplified implementation - could be extended for complex patterns
        all_entities = []
        for kind, entities in ctx.state.extracted_entities.items():
            all_entities.extend(entities)

        # For now, just create self-assessments
        for entity in all_entities[:10]:  # Limit for demonstration
            assessment = {
                "entity": entity,
                "assessment_type": "self",
                "confidence": "moderate",
                "notes": "Self-assessment for single entity type",
            }
            ctx.state.individual_assessments.append(assessment)

    async def _assess_entity_pair(
        self,
        ctx: GraphRunContext[EntityExtractionState, Deps],
        agent,
        entity_a_dict: Dict[str, Any],
        entity_b_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assess relationship between a specific entity pair."""

        # Convert dicts back to EntityInfo models for agent
        entity_a = EntityInfo(**entity_a_dict)
        entity_b = EntityInfo(**entity_b_dict)

        # Get context from document group
        current_group = ctx.state.get_current_group()

        # Get text context for the assessment
        context_text = ""
        if current_group and entity_a.source_url in current_group.chunks:
            chunks = current_group.chunks[entity_a.source_url]
            context_text = "\n".join([chunk.get("text", "") for chunk in chunks])

        # Create assessment prompt
        prompt = f"""Assess the {ctx.deps.relation_type} relationship between these entities:

Entity A: {entity_a.name} (type: {entity_a.kind})
Entity B: {entity_b.name} (type: {entity_b.kind})

Context from source document:
{context_text[:10000]}

Determine if there is evidence of a {ctx.deps.relation_type} relationship."""

        try:
            from .agents import run_agent_with_tracking

            result = await run_agent_with_tracking(
                agent, prompt, str(ctx.deps.model), deps=ctx.deps
            )
            return result.output.model_dump()
        except Exception as e:
            # Return error information for debugging
            return {
                "error": str(e),
                "entity_a": entity_a_dict,
                "entity_b": entity_b_dict,
                "assessment_type": "failed",
            }


@dataclass
class ParallelEntityProcessor(BaseNode[EntityExtractionState, Deps]):
    """
    Process entities individually with parallel context extraction and assessment.

    This node implements the individual entity processing pattern where each
    entity gets its own context extraction and relationship assessment.
    """

    async def run(
        self, ctx: GraphRunContext[EntityExtractionState, Deps]
    ) -> "EntityAggregation" | End[str]:
        ctx.state.log_node_entry("ParallelEntityProcessor")

        import time

        start_time = time.time()

        # Update progress tracker
        progress_tracker = get_progress_tracker(ctx.state)
        if progress_tracker:
            progress_tracker.update_node("ParallelEntityProcessor")
            progress_tracker.update_stage("Processing individual entities")

        # Get all entities to process
        all_entities = []
        for kind, entities in ctx.state.extracted_entities.items():
            for entity_dict in entities:
                entity_info = EntityInfo(**entity_dict)
                all_entities.append(entity_info)

        if not all_entities:
            logfire.warning("No entities to process individually")
            return End("No entities to process individually")

        total_entities = len(all_entities)
        batch_size = ctx.deps.batch_size
        total_batches = (total_entities + batch_size - 1) // batch_size

        logfire.info(
            f"Starting parallel entity processing: {total_entities} entities in {total_batches} batches (batch_size={batch_size})"
        )

        for batch_idx, i in enumerate(range(0, len(all_entities), batch_size)):
            batch_start_time = time.time()
            batch = all_entities[i : i + batch_size]
            batch_end = min(i + batch_size, total_entities)

            # Update progress tracker
            if progress_tracker:
                progress_tracker.update_batch_progress(
                    f"Processing entities {i + 1}-{batch_end} of {total_entities}"
                )

            logfire.info(
                f"Processing batch {batch_idx + 1}/{total_batches}: entities {i + 1}-{batch_end}"
            )

            # Process context extraction in parallel for this batch
            context_start = time.time()
            contexts = await self._extract_contexts_batch(ctx, batch)
            context_time = time.time() - context_start
            ctx.state.individual_entity_contexts.extend(contexts)

            logfire.debug(
                f"Context extraction for batch {batch_idx + 1} took {context_time:.1f}s"
            )

            # Process individual assessments in parallel for this batch
            assessment_start = time.time()
            assessments = await self._assess_entities_batch(ctx, contexts)
            assessment_time = time.time() - assessment_start
            ctx.state.individual_entity_assessments.extend(assessments)

            logfire.debug(
                f"Assessment for batch {batch_idx + 1} took {assessment_time:.1f}s"
            )

            ctx.state.entities_processed += len(batch)

            batch_time = time.time() - batch_start_time
            logfire.info(
                f"Batch {batch_idx + 1}/{total_batches} completed in {batch_time:.1f}s"
            )

        ctx.state.add_processing_note(
            "individual_processing_summary",
            {
                "total_entities": len(all_entities),
                "entities_processed": ctx.state.entities_processed,
                "individual_contexts": len(ctx.state.individual_entity_contexts),
                "individual_assessments": len(ctx.state.individual_entity_assessments),
                "batch_size": batch_size,
            },
        )

        total_time = time.time() - start_time
        logfire.info(
            f"ParallelEntityProcessor completed in {total_time:.1f}s: processed {total_entities} entities in {total_batches} batches"
        )

        return EntityAggregation()

    async def _extract_contexts_batch(
        self, ctx: GraphRunContext[EntityExtractionState, Deps], batch: List[EntityInfo]
    ) -> List[IndividualEntityContext]:
        """Extract context for each entity in parallel."""
        from .agents import create_context_extractor_agent, run_agent_with_tracking

        context_agent = create_context_extractor_agent(
            ctx.deps.model, ctx.deps.entity_kinds, ctx.deps.task_context
        )

        model_name = str(ctx.deps.model)

        # Get document group for context
        current_group = ctx.state.get_current_group()
        if not current_group:
            return []

        tasks = []
        for entity in batch:
            # Get context text for this entity's document
            context_text = ""
            if entity.source_url in current_group.chunks:
                chunks = current_group.chunks[entity.source_url]
                context_text = "\n".join([chunk.get("text", "") for chunk in chunks])

            # Create prompt for context extraction
            prompt = f"""Extract detailed context for this entity:

Entity: {entity.name} (type: {entity.kind})
Source Document: {entity.source_url}

Document Content:
{context_text[:20000]}  # Limit content to avoid token limits

Focus on finding mentions, claims, and evidence specifically about '{entity.name}'."""

            tasks.append(
                run_agent_with_tracking(
                    context_agent,
                    prompt,
                    str(ctx.deps.model),
                    deps=ctx.deps,
                )
            )

        # Execute context extractions in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        contexts = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Report error with context if verbose mode
                progress_tracker = get_progress_tracker(ctx.state)
                if progress_tracker:
                    entity = batch[i]
                    progress_tracker.report_error(
                        str(result),
                        f"Context extraction failed for entity '{entity.name}'",
                    )

                # Create fallback context for failed extractions
                entity = batch[i]
                fallback_context = IndividualEntityContext(
                    entity=entity,
                    mentions=[f"Entity '{entity.name}' found in document"],
                    claims=[],
                    evidence=[],
                    chunk_contexts=[],
                    context_confidence=0.1,
                )
                contexts.append(fallback_context)
            else:
                # Update the context with the actual entity info
                context = result.output
                context.entity = batch[i]  # Ensure entity info is complete
                contexts.append(context)

        return contexts

    async def _assess_entities_batch(
        self,
        ctx: GraphRunContext[EntityExtractionState, Deps],
        contexts: List[IndividualEntityContext],
    ) -> List["IndividualEntityAssessment"]:
        """Assess each entity individually in parallel."""
        from .agents import create_individual_assessor_agent, run_agent_with_tracking
        from .models import IndividualEntityAssessment

        assessor_agent = create_individual_assessor_agent(
            ctx.deps.model,
            ctx.deps.entity_kinds,
            ctx.deps.relation_type,
            ctx.deps.task_context,
        )

        tasks = []
        for context in contexts:
            # Create assessment prompt
            mentions_str = "\n".join(f"- {mention}" for mention in context.mentions)
            claims_str = "\n".join(f"- {claim}" for claim in context.claims)
            evidence_str = "\n".join(f"- {evidence}" for evidence in context.evidence)

            prompt = f"""Assess the relationship potential for this entity:

Entity: {context.entity.name} (type: {context.entity.kind})

Extracted Context:
Mentions:
{mentions_str}

Claims:
{claims_str}

Evidence:
{evidence_str}

Assess this entity's potential for {ctx.deps.relation_type} relationships and identify potential partners."""

            tasks.append(
                run_agent_with_tracking(
                    assessor_agent,
                    prompt,
                    str(ctx.deps.model),
                    deps=ctx.deps,
                )
            )

        # Execute assessments in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        assessments = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Report error with context if verbose mode
                progress_tracker = get_progress_tracker(ctx.state)
                if progress_tracker:
                    context = contexts[i]
                    progress_tracker.report_error(
                        str(result),
                        f"Assessment failed for entity '{context.entity.name}'",
                    )

                # Create fallback assessment for failed assessments
                context = contexts[i]
                fallback_assessment = IndividualEntityAssessment(
                    entity=context.entity,
                    context=context,
                    relationship_potential="none",
                    target_entity_hints=[],
                    reasoning="Assessment failed due to processing error",
                    confidence=0.1,
                    processing_notes=f"Error: {str(result)}",
                )
                assessments.append(fallback_assessment)
            else:
                # Update the assessment with the actual context
                assessment = result.output
                assessment.context = contexts[i]  # Ensure context is properly linked
                assessments.append(assessment)

        # Update entities processed count for progress tracking
        progress_tracker = get_progress_tracker(ctx.state)
        if progress_tracker:
            ctx.state.entities_processed += len(assessments)
            progress_tracker.update_processed(ctx.state.entities_processed)

        return assessments


@dataclass
class EntityAggregation(BaseNode[EntityExtractionState, Deps]):
    """
    Aggregate individual entity assessments into entity pairs.

    This node takes all individual entity assessments and creates
    entity pairs based on relationship potential and hints.
    """

    async def run(
        self, ctx: GraphRunContext[EntityExtractionState, Deps]
    ) -> "EntityPairGeneration" | End[List[Dict[str, Any]]]:
        ctx.state.log_node_entry("EntityAggregation")

        # Update progress tracker
        progress_tracker = get_progress_tracker(ctx.state)
        if progress_tracker:
            progress_tracker.update_node("EntityAggregation")
            progress_tracker.update_stage(
                "Aggregating individual assessments into potential pairs"
            )

        if not ctx.state.individual_entity_assessments:
            return End([])  # No individual assessments to aggregate

        # Get aggregation agent
        from .agents import create_aggregation_agent, run_agent_with_tracking

        aggregator_agent = create_aggregation_agent(
            ctx.deps.model,
            ctx.deps.entity_kinds,
            ctx.deps.relation_type,
            ctx.deps.task_context,
        )

        # Build aggregation prompt
        prompt = self._build_aggregation_prompt(ctx.state)

        try:
            result = await run_agent_with_tracking(
                aggregator_agent,
                prompt,
                str(ctx.deps.model),
                deps=ctx.deps,
            )
            ctx.state.aggregation_result = result.output

            # Convert high-confidence pairs to entity pairs for the next stage
            for pair in result.output.high_confidence_pairs:
                # Filter out same-type pairs (e.g., disease-disease)
                if pair.entity_a.kind == pair.entity_b.kind:
                    continue  # Skip same-type pairs

                pair_dict = pair.model_dump()
                # Ensure source_documents is populated from entity source URLs
                if not pair_dict.get("source_documents"):
                    source_urls = set()
                    if pair.entity_a and pair.entity_a.source_url:
                        source_urls.add(pair.entity_a.source_url)
                    if pair.entity_b and pair.entity_b.source_url:
                        source_urls.add(pair.entity_b.source_url)
                    pair_dict["source_documents"] = list(source_urls)
                ctx.state.entity_pairs.append(pair_dict)

            ctx.state.add_processing_note(
                "aggregation_summary",
                {
                    "total_individuals_assessed": len(
                        ctx.state.individual_entity_assessments
                    ),
                    "high_confidence_pairs": len(result.output.high_confidence_pairs),
                    "moderate_confidence_pairs": len(
                        result.output.moderate_confidence_pairs
                    ),
                    "unmatched_entities": len(result.output.unmatched_entities),
                    "aggregation_confidence": result.output.aggregation_confidence,
                },
            )

            if ctx.state.entity_pairs:
                return EntityPairGeneration()
            else:
                return End([])  # No pairs were generated

        except Exception as e:
            ctx.state.add_processing_note("aggregation_error", str(e))
            return End([])  # Fail gracefully

    def _build_aggregation_prompt(self, state: EntityExtractionState) -> str:
        """Build comprehensive prompt for aggregation agent."""

        # Organize assessments by relationship potential
        high_potential = [
            a
            for a in state.individual_entity_assessments
            if a.relationship_potential == "high"
        ]
        moderate_potential = [
            a
            for a in state.individual_entity_assessments
            if a.relationship_potential == "moderate"
        ]
        low_potential = [
            a
            for a in state.individual_entity_assessments
            if a.relationship_potential in ["low", "none"]
        ]

        # Build assessments summary
        assessments_summary = []

        for assessment in state.individual_entity_assessments:
            entity_summary = f"""
Entity: {assessment.entity.name} ({assessment.entity.kind})
Source URL: {assessment.entity.source_url}
Relationship Potential: {assessment.relationship_potential}
Confidence: {assessment.confidence:.2f}
Target Hints: {", ".join(assessment.target_entity_hints) if assessment.target_entity_hints else "None"}
Reasoning: {assessment.reasoning}
Context Summary: {len(assessment.context.mentions)} mentions, {len(assessment.context.claims)} claims, {len(assessment.context.evidence)} evidence pieces
"""
            assessments_summary.append(entity_summary)

        prompt = f"""Analyze these individual entity assessments and create {state.relation_type} pairs:

INDIVIDUAL ASSESSMENTS:
{"".join(assessments_summary)}

AGGREGATION TASK:
1. Match entities that show mutual relationship potential
2. Look for target entity hints that match actual entities
3. Consider confidence levels and evidence quality
4. Create pairs with appropriate confidence levels
5. CRITICAL: For each EntityPairOut, populate source_documents with the Source URLs from BOTH entities
6. Identify entities that don't have clear matches

STATISTICS:
- Total entities assessed: {len(state.individual_entity_assessments)}
- High potential: {len(high_potential)}
- Moderate potential: {len(moderate_potential)}
- Low/None potential: {len(low_potential)}

Focus on creating well-evidenced pairs while being conservative about relationship quality."""

        return prompt


@dataclass
class EntityPairGeneration(BaseNode[EntityExtractionState, Deps]):
    """
    Generate final entity pairs from individual assessments.
    """

    async def run(
        self, ctx: GraphRunContext[EntityExtractionState, Deps]
    ) -> "QualityValidation" | End[List[Dict[str, Any]]]:
        ctx.state.log_node_entry("EntityPairGeneration")

        # Update progress tracker
        progress_tracker = get_progress_tracker(ctx.state)
        if progress_tracker:
            progress_tracker.update_node("EntityPairGeneration")
            progress_tracker.update_stage(
                "Generating final entity pairs from assessments"
            )

        # Get pair generation agent
        from .agents import create_pair_generator_agent, run_agent_with_tracking

        pair_agent = create_pair_generator_agent(
            ctx.deps.model,
            ctx.deps.entity_kinds,
            ctx.deps.relation_type,
            ctx.deps.task_context,
        )

        # Filter assessments for high-confidence pairs
        # Use the correct field: individual_entity_assessments (not individual_assessments)
        valid_assessments = [
            assessment
            for assessment in ctx.state.individual_entity_assessments
            if assessment.confidence
            > 0.5  # Use actual IndividualEntityAssessment objects
            and assessment.relationship_potential in ["high", "moderate"]
        ]

        # Debug: Show what we're working with
        progress_tracker = get_progress_tracker(ctx.state)
        if progress_tracker:
            progress_tracker.update_stage(
                f"Found {len(valid_assessments)} high-confidence assessments from {len(ctx.state.individual_entity_assessments)} total"
            )

        if not valid_assessments:
            return End([])  # No valid pairs found

        # Generate pairs from assessments (simplified)
        for assessment in valid_assessments[:5]:  # Limit for demonstration
            if hasattr(assessment, "processing_notes") and "Error:" in str(
                assessment.processing_notes
            ):
                continue

            try:
                # Create prompt for pair generation
                prompt = f"""Generate an entity pair from this assessment:

Assessment: {assessment}

Create a high-quality entity pair with complete provenance."""

                result = await run_agent_with_tracking(
                    pair_agent,
                    prompt,
                    str(ctx.deps.model),
                    deps=ctx.deps,
                )
                pair_dict = result.output.model_dump()

                # Filter out same-type pairs (e.g., disease-disease)
                if (
                    hasattr(result.output, "entity_a")
                    and hasattr(result.output, "entity_b")
                    and result.output.entity_a.kind == result.output.entity_b.kind
                ):
                    continue  # Skip same-type pairs

                # Ensure source_documents is populated from entity source URLs
                if not pair_dict.get("source_documents"):
                    source_urls = set()
                    if (
                        hasattr(result.output, "entity_a")
                        and result.output.entity_a.source_url
                    ):
                        source_urls.add(result.output.entity_a.source_url)
                    if (
                        hasattr(result.output, "entity_b")
                        and result.output.entity_b.source_url
                    ):
                        source_urls.add(result.output.entity_b.source_url)
                    if source_urls:
                        pair_dict["source_documents"] = list(source_urls)
                    elif hasattr(assessment, "entity") and assessment.entity.source_url:
                        # Fallback to assessment entity if pair entities not available
                        pair_dict["source_documents"] = [assessment.entity.source_url]

                ctx.state.entity_pairs.append(pair_dict)

            except Exception:
                continue  # Skip failed pair generation

        if not ctx.state.entity_pairs:
            return End([])  # No pairs generated successfully

        return QualityValidation()


@dataclass
class QualityValidation(BaseNode[EntityExtractionState, Deps, List[Dict[str, Any]]]):
    """
    Final quality validation with potential revision loop.
    """

    max_revisions: int = 2

    async def run(
        self, ctx: GraphRunContext[EntityExtractionState, Deps]
    ) -> "EntityPairGeneration" | End[List[Dict[str, Any]]]:
        ctx.state.log_node_entry("QualityValidation")

        # Update progress tracker
        progress_tracker = get_progress_tracker(ctx.state)
        if progress_tracker:
            progress_tracker.update_node("QualityValidation")
            progress_tracker.update_stage(
                "Performing quality validation on entity pairs"
            )

        # Get validation agent
        from .agents import create_validation_agent, run_agent_with_tracking

        validator_agent = create_validation_agent(
            ctx.deps.model,
            ctx.deps.entity_kinds,
            ctx.deps.relation_type,
            ctx.deps.task_context,
        )

        validated_pairs = []
        revision_count = ctx.state.processing_notes.get("revision_count", 0)

        # Debug: Show validation input
        progress_tracker = get_progress_tracker(ctx.state)
        if progress_tracker:
            progress_tracker.update_stage(
                f"Validating {len(ctx.state.entity_pairs)} entity pairs"
            )

        for pair in ctx.state.entity_pairs:
            # Create validation prompt
            prompt = f"""Validate this entity pair for quality:

Pair: {pair}

Check evidence quality, relationship clarity, and completeness."""

            try:
                result = await run_agent_with_tracking(
                    validator_agent,
                    prompt,
                    str(ctx.deps.model),
                    deps=ctx.deps,
                )

                if result.output.verdict == "approve":
                    validated_pairs.append(pair)
                elif (
                    result.output.verdict == "revise"
                    and revision_count < self.max_revisions
                ):
                    # Add revision feedback to processing notes
                    ctx.state.add_processing_note(
                        "revision_feedback", result.output.feedback
                    )
                    ctx.state.add_processing_note("revision_count", revision_count + 1)

                    # Return to pair generation for revision
                    return EntityPairGeneration()
                else:
                    # Debug: Track rejected pairs
                    if progress_tracker:
                        verdict = result.output.verdict
                        reason = getattr(
                            result.output, "reasoning", "No reason provided"
                        )
                        progress_tracker.update_stage(
                            f"Pair rejected: {verdict} - {reason[:50]}..."
                        )

                # "reject" verdict or max revisions reached - skip pair

            except Exception:
                continue  # Skip pairs that fail validation

        ctx.state.add_processing_note(
            "validation_summary",
            {
                "total_pairs": len(ctx.state.entity_pairs),
                "validated_pairs": len(validated_pairs),
                "revision_count": revision_count,
            },
        )

        return End(validated_pairs)
