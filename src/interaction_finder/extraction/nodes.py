"""Graph nodes for association extraction pipeline.

Each node represents a stage in the pipeline. Nodes control flow via
return type annotations. All branching and looping happens in nodes;
agents only produce typed data.
"""

import asyncio
from dataclasses import dataclass
from typing import Union

from pydantic_ai.usage import RunUsage
from pydantic_graph import BaseNode, End, GraphRunContext

from interaction_finder.extraction.assess_entity import get_entity_assessor_agent
from interaction_finder.extraction.assess_pair import get_pair_assessor_agent
from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.extract import get_entity_extractor_agent
from interaction_finder.extraction.extract_pairs import get_pair_agent
from interaction_finder.extraction.judge import get_judge_agent
from interaction_finder.extraction.models import (
    EntityAssessment,
    EntityMention,
    ExtractionMetadata,
    ExtractionResult,
    FinalJudgment,
    PairAssessment,
    PairKey,
    PairMention,
    PairWithProvenance,
)
from interaction_finder.extraction.state import State
from interaction_finder.logging import logfire
from interaction_finder.resources import Resource


@dataclass
class ExtractFromDocumentsNode(BaseNode[State, Deps, ExtractionResult]):
    """Extract entities and pairs from all resources in parallel.

    For each resource:
    1. Call entity_extractor_agent to get entities with quotes
    2. Call pair_extractor_agent to get pairs with quotes
    3. Convert all quote strings to ResourceQuote objects
    4. Store validated results in State
    """

    async def run(
        self, ctx: GraphRunContext[State, Deps]
    ) -> Union["AssessEntitiesNode", End[ExtractionResult]]:
        """Process all resources in parallel."""
        with logfire.span("ExtractFromDocumentsNode"):
            resources = ctx.deps.resource_pool.resources

            if not resources:
                ctx.deps.logger.warning("No resources to process")
                # Return empty result
                return End(
                    ExtractionResult(
                        accepted_pairs=[],
                        metadata=ExtractionMetadata(
                            topic=ctx.state.topic,
                            resource_count=0,
                            total_entities_found=0,
                            total_pairs_found=0,
                            pairs_accepted=0,
                            pairs_rejected=0,
                            quotes_validated=0,
                            quotes_failed=0,
                        ),
                    )
                )

            # Process all resources in parallel
            tasks = [self._process_resource(resource, ctx) for resource in resources]
            await asyncio.gather(*tasks)

            # Check if we found any entities or pairs
            if not ctx.state.entities_by_resource and not ctx.state.pairs_by_resource:
                ctx.deps.logger.warning("No entities or pairs extracted from documents")
                return End(
                    ExtractionResult(
                        accepted_pairs=[],
                        metadata=ExtractionMetadata(
                            topic=ctx.state.topic,
                            resource_count=len(resources),
                            total_entities_found=0,
                            total_pairs_found=0,
                            pairs_accepted=0,
                            pairs_rejected=0,
                            quotes_validated=ctx.state.quotes_validated,
                            quotes_failed=ctx.state.quotes_failed,
                        ),
                    )
                )

            return AssessEntitiesNode()

    async def _process_resource(
        self, resource: Resource, ctx: GraphRunContext[State, Deps]
    ):
        """Process a single resource: extract entities and pairs."""
        with logfire.span("process_resource", resource_url=resource.id.url):
            usage = RunUsage()

            # Build prompts - keep focused and structured
            entity_types_str = ", ".join(ctx.state.target_entity_types)
            entity_prompt = f"""Extract entities from this document relevant to: {ctx.state.topic}

**Target entity types:** {entity_types_str}

**Document title:** {resource.title}

**Document text:**
{resource.text[:15000]}

Extract all entities of the specified types that are relevant to the topic.
For each entity, provide: canonical name, all verbatim names from text, supporting quotes, and reasoning."""

            pair_prompt = f"""Extract associations from this document relevant to: {ctx.state.topic}

**Document title:** {resource.title}

**Document text:**
{resource.text[:15000]}

Extract binary entity-entity associations relevant to the topic.
Use canonical entity names and provide exact quotes supporting each association."""

            # Extract entities using configured extraction model
            try:
                entity_result = await get_entity_extractor_agent(ctx.deps.config).run(
                    entity_prompt, deps=ctx.deps, usage=usage
                )
            except (
                # Catch expected failures from LLM operations
                TimeoutError,
                ConnectionError,
                ValueError,  # Model/validation errors from pydantic-ai
            ) as e:
                ctx.deps.logger.error(
                    f"Entity extraction failed for {resource.id.url}: {type(e).__name__}: {e}"
                )
                # Continue without entities from this resource
                entity_result = None

            # Extract pairs using configured extraction model
            try:
                pair_result = await get_pair_agent(ctx.deps.config).run(
                    pair_prompt, deps=ctx.deps, usage=usage
                )
            except (
                # Catch expected failures from LLM operations
                TimeoutError,
                ConnectionError,
                ValueError,  # Model/validation errors from pydantic-ai
            ) as e:
                ctx.deps.logger.error(
                    f"Pair extraction failed for {resource.id.url}: {type(e).__name__}: {e}"
                )
                # Continue without pairs from this resource
                pair_result = None

            # Convert entities to EntityMention with ResourceQuotes, merging duplicates
            entities_dict = {}
            if entity_result is not None:
                # Group entities by name to handle duplicates
                entities_by_name: dict[str, list] = {}
                for entity_info in entity_result.output.entities:
                    if entity_info.name not in entities_by_name:
                        entities_by_name[entity_info.name] = []
                    entities_by_name[entity_info.name].append(entity_info)

                # Convert and merge each entity name
                for entity_name, entity_infos in entities_by_name.items():
                    # Validate all quotes from all instances
                    all_quotes = []
                    for entity_info in entity_infos:
                        for quote_str in entity_info.quotes:
                            try:
                                quote = resource.quote(quote_str)
                                all_quotes.append(quote)
                                ctx.state.quotes_validated += 1
                            except Exception as e:
                                ctx.state.quotes_failed += 1
                                ctx.deps.logger.warning(
                                    f"Failed to validate entity quote for '{entity_name}' "
                                    f"in {resource.id.url}: {type(e).__name__}: {e}"
                                )

                    if all_quotes:  # Only store entity if we have valid quotes
                        # Merge aliases from all instances (deduplicate)
                        all_aliases = []
                        seen_aliases = set()
                        for entity_info in entity_infos:
                            for alias in entity_info.aliases:
                                if alias not in seen_aliases:
                                    all_aliases.append(alias)
                                    seen_aliases.add(alias)

                        # Merge reasoning (join with separator if multiple)
                        merged_reasoning = " | ".join(
                            entity_info.reasoning for entity_info in entity_infos
                        )

                        entities_dict[entity_name] = EntityMention(
                            kind=entity_infos[0].kind,  # Should be consistent
                            name=entity_name,
                            aliases=all_aliases,
                            quotes=all_quotes,
                            reasoning=merged_reasoning,
                        )

            # Store entities for this resource
            if entities_dict:
                ctx.state.entities_by_resource[resource.id] = entities_dict

            # Convert pairs to PairMention with ResourceQuotes
            pairs_list = []
            if pair_result is not None:
                for pair_info in pair_result.output.pairs:
                    # Convert quote strings to ResourceQuotes
                    quotes = []
                    for quote_str in pair_info.supporting_quotes:
                        try:
                            quote = resource.quote(quote_str)
                            quotes.append(quote)
                            ctx.state.quotes_validated += 1
                        except Exception as e:
                            ctx.state.quotes_failed += 1
                            ctx.deps.logger.warning(
                                f"Failed to validate pair quote for "
                                f"'{pair_info.entity1}-{pair_info.entity2}' "
                                f"in {resource.id.url}: {type(e).__name__}: {e}"
                            )

                    if quotes:  # Only store pair if we have valid quotes
                        pairs_list.append(
                            PairMention(
                                entity1=pair_info.entity1,
                                entity2=pair_info.entity2,
                                relationship_type=pair_info.relationship_type,
                                quotes=quotes,
                            )
                        )

            # Store pairs for this resource
            if pairs_list:
                ctx.state.pairs_by_resource[resource.id] = pairs_list


@dataclass
class AssessEntitiesNode(BaseNode[State, Deps, ExtractionResult]):
    """Assess entity relevance per resource in parallel.

    For each (entity, resource) combination, call entity_assessor_agent
    to evaluate evidence strength.
    """

    async def run(self, ctx: GraphRunContext[State, Deps]) -> "AssessPairsNode":
        """Assess all entity mentions in parallel."""
        with logfire.span("AssessEntitiesNode"):
            # Build list of (entity_name, resource_id, quotes) tuples to assess
            assessment_tasks = []
            for resource_id, entities_dict in ctx.state.entities_by_resource.items():
                for entity_name, entity_mention in entities_dict.items():
                    assessment_tasks.append(
                        self._assess_entity(
                            entity_name, entity_mention, resource_id, ctx
                        )
                    )

            # Run all assessments in parallel
            if assessment_tasks:
                await asyncio.gather(*assessment_tasks)

            return AssessPairsNode()

    async def _assess_entity(
        self,
        entity_name: str,
        entity_mention: EntityMention,
        resource_id,
        ctx: GraphRunContext[State, Deps],
    ):
        """Assess a single entity in a single resource."""
        with logfire.span("assess_entity", entity=entity_name):
            usage = RunUsage()

            # Build quote list for prompt
            quotes_str = "\n".join(
                f"[{i}] {q.get_quote_text()}"
                for i, q in enumerate(entity_mention.quotes)
            )

            prompt = f"""Assess the evidence strength for this entity's relevance to the topic.

**Topic:** {ctx.state.topic}

**Entity:** {entity_name} (type: {entity_mention.kind})

**Quotes from document:**
{quotes_str}

Evaluate how strongly these quotes support the entity's relevance to the topic.
Reference specific quote indices in your assessment."""

            # Call assessment agent using configured extraction model
            result = await get_entity_assessor_agent(ctx.deps.config).run(
                prompt, deps=ctx.deps, usage=usage
            )

            # Extract referenced quotes
            referenced_quotes = [
                entity_mention.quotes[i]
                for i in result.output.supporting_quote_ids
                if i < len(entity_mention.quotes)
            ]

            # Create assessment
            assessment = EntityAssessment(
                resource_id=resource_id,
                strength=result.output.strength,
                rationale=result.output.rationale,
                quotes=referenced_quotes,
            )

            # Store in state
            if entity_name not in ctx.state.entity_assessments:
                ctx.state.entity_assessments[entity_name] = []
            ctx.state.entity_assessments[entity_name].append(assessment)


@dataclass
class AssessPairsNode(BaseNode[State, Deps, ExtractionResult]):
    """Assess pair validity per resource in parallel.

    For each (pair, resource) combination, gather pair quotes and entity quotes,
    then call pair_assessor_agent to evaluate evidence strength.
    """

    async def run(self, ctx: GraphRunContext[State, Deps]) -> "JudgePairsNode":
        """Assess all pair mentions in parallel."""
        with logfire.span("AssessPairsNode"):
            # Build list of (pair, resource_id) tuples to assess
            assessment_tasks = []
            for resource_id, pairs_list in ctx.state.pairs_by_resource.items():
                for pair_mention in pairs_list:
                    assessment_tasks.append(
                        self._assess_pair(pair_mention, resource_id, ctx)
                    )

            # Run all assessments in parallel
            if assessment_tasks:
                await asyncio.gather(*assessment_tasks)

            return JudgePairsNode()

    async def _assess_pair(
        self,
        pair_mention: PairMention,
        resource_id,
        ctx: GraphRunContext[State, Deps],
    ):
        """Assess a single pair in a single resource."""
        pair_key: PairKey = (
            pair_mention.entity1,
            pair_mention.entity2,
            pair_mention.relationship_type,
        )

        with logfire.span("assess_pair", pair=f"{pair_key[0]}-{pair_key[1]}"):
            usage = RunUsage()

            # Gather all relevant quotes: pair quotes + entity quotes from this resource
            all_quotes = list(pair_mention.quotes)

            # Add entity quotes if entities exist in this resource
            entities_in_resource = ctx.state.entities_by_resource.get(resource_id, {})
            if pair_mention.entity1 in entities_in_resource:
                all_quotes.extend(entities_in_resource[pair_mention.entity1].quotes)
            if pair_mention.entity2 in entities_in_resource:
                all_quotes.extend(entities_in_resource[pair_mention.entity2].quotes)

            # Build quote list for prompt
            quotes_str = "\n".join(
                f"[{i}] {q.get_quote_text()}" for i, q in enumerate(all_quotes)
            )

            # Include entity assessment context if available
            entity1_context = ""
            entity2_context = ""
            if pair_mention.entity1 in ctx.state.entity_assessments:
                assessments = [
                    a
                    for a in ctx.state.entity_assessments[pair_mention.entity1]
                    if a.resource_id == resource_id
                ]
                if assessments:
                    entity1_context = f"\n**{pair_mention.entity1} relevance:** {assessments[0].strength} - {assessments[0].rationale}"

            if pair_mention.entity2 in ctx.state.entity_assessments:
                assessments = [
                    a
                    for a in ctx.state.entity_assessments[pair_mention.entity2]
                    if a.resource_id == resource_id
                ]
                if assessments:
                    entity2_context = f"\n**{pair_mention.entity2} relevance:** {assessments[0].strength} - {assessments[0].rationale}"

            prompt = f"""Assess the evidence strength for this association.

**Topic:** {ctx.state.topic}

**Association:** {pair_mention.entity1} {pair_mention.relationship_type} {pair_mention.entity2}
{entity1_context}{entity2_context}

**Quotes from document:**
{quotes_str}

Evaluate how strongly these quotes support the validity of this association.
Consider both the relevance of the individual entities and the strength of their relationship.
Reference specific quote indices in your assessment."""

            # Call assessment agent using configured extraction model
            result = await get_pair_assessor_agent(ctx.deps.config).run(
                prompt, deps=ctx.deps, usage=usage
            )

            # Extract referenced quotes
            referenced_quotes = [
                all_quotes[i]
                for i in result.output.supporting_quote_ids
                if i < len(all_quotes)
            ]

            # Create assessment
            assessment = PairAssessment(
                resource_id=resource_id,
                strength=result.output.strength,
                rationale=result.output.rationale,
                quotes=referenced_quotes,
            )

            # Store in state
            if pair_key not in ctx.state.pair_assessments:
                ctx.state.pair_assessments[pair_key] = []
            ctx.state.pair_assessments[pair_key].append(assessment)


@dataclass
class JudgePairsNode(BaseNode[State, Deps, ExtractionResult]):
    """Make final judgments on pairs (conditional LLM calls).

    For each unique pair across all resources:
    1. Check if high-confidence judgment possible from assessments alone
    2. If yes: make deterministic judgment
    3. If no: gather all quotes and call final_judge_agent
    """

    async def run(self, ctx: GraphRunContext[State, Deps]) -> "FinalizeNode":
        """Make final judgments on all pairs."""
        with logfire.span("JudgePairsNode"):
            # Process each unique pair
            judgment_tasks = []
            for pair_key in ctx.state.pair_assessments.keys():
                judgment_tasks.append(self._judge_pair(pair_key, ctx))

            # Run all judgments in parallel
            if judgment_tasks:
                await asyncio.gather(*judgment_tasks)

            return FinalizeNode()

    def _can_judge_deterministically(
        self, assessments: list[PairAssessment]
    ) -> tuple[bool, FinalJudgment | None]:
        """Check if we can make high-confidence judgment without LLM.

        Returns:
            (can_judge, judgment) - judgment is None if can't judge deterministically
        """
        strong_count = sum(1 for a in assessments if a.strength == "strong")
        none_count = sum(1 for a in assessments if a.strength == "none")

        # High-confidence accept: multiple strong assessments
        if strong_count >= 2:
            return (
                True,
                FinalJudgment(
                    accepted=True,
                    confidence="high",
                    rationale=f"Multiple strong assessments ({strong_count}) "
                    "provide high-confidence evidence for this association.",
                ),
            )

        # High-confidence reject: multiple none assessments
        if none_count >= 2:
            return (
                True,
                FinalJudgment(
                    accepted=False,
                    confidence="high",
                    rationale=f"Multiple assessments ({none_count}) found no "
                    "meaningful evidence for this association.",
                ),
            )

        # Can't judge deterministically
        return (False, None)

    async def _judge_pair(self, pair_key: PairKey, ctx: GraphRunContext[State, Deps]):
        """Make final judgment on a single pair."""
        with logfire.span("judge_pair", pair=f"{pair_key[0]}-{pair_key[1]}"):
            assessments = ctx.state.pair_assessments[pair_key]

            # Try deterministic judgment first
            can_judge, judgment = self._can_judge_deterministically(assessments)

            if can_judge:
                ctx.state.final_pair_judgments[pair_key] = judgment
            else:
                # Need LLM judgment - gather all quotes from all resources
                all_quotes = []
                for assessment in assessments:
                    all_quotes.extend(assessment.quotes)

                # Also gather quotes from pair mentions
                for resource_id, pairs_list in ctx.state.pairs_by_resource.items():
                    for pair_mention in pairs_list:
                        if (
                            pair_mention.entity1 == pair_key[0]
                            and pair_mention.entity2 == pair_key[1]
                            and pair_mention.relationship_type == pair_key[2]
                        ):
                            all_quotes.extend(pair_mention.quotes)

                # Build assessment summary
                assessment_summary = "\n".join(
                    f"- Resource {i + 1}: {a.strength} - {a.rationale}"
                    for i, a in enumerate(assessments)
                )

                # Include entity assessment summary across all documents
                entity1_summary = ""
                entity2_summary = ""
                if pair_key[0] in ctx.state.entity_assessments:
                    entity1_assessments = ctx.state.entity_assessments[pair_key[0]]
                    strengths = [a.strength for a in entity1_assessments]
                    entity1_summary = f"\n\n**{pair_key[0]} relevance across documents:** {', '.join(strengths)}"

                if pair_key[1] in ctx.state.entity_assessments:
                    entity2_assessments = ctx.state.entity_assessments[pair_key[1]]
                    strengths = [a.strength for a in entity2_assessments]
                    entity2_summary = f"\n**{pair_key[1]} relevance across documents:** {', '.join(strengths)}"

                # Build quote list
                quotes_str = "\n".join(
                    f"[{i}] {q.get_quote_text()}" for i, q in enumerate(all_quotes)
                )

                prompt = f"""Make a final judgment on this association across all documents.

**Topic:** {ctx.state.topic}

**Association:** {pair_key[0]} {pair_key[2]} {pair_key[1]}

**Per-document pair assessments:**
{assessment_summary}{entity1_summary}{entity2_summary}

**All supporting quotes:**
{quotes_str}

Based on the evidence across all documents, decide whether to accept this association
and your confidence level. Consider:
1. The strength and consistency of pair-level evidence
2. The relevance of both entities to the topic
3. The quality and quantity of supporting quotes"""

                # Call final judge agent using configured judge model
                usage = RunUsage()
                result = await get_judge_agent(ctx.deps.config).run(
                    prompt, deps=ctx.deps, usage=usage
                )

                judgment = FinalJudgment(
                    accepted=result.output.accepted,
                    confidence=result.output.confidence,
                    rationale=result.output.rationale,
                )

                ctx.state.final_pair_judgments[pair_key] = judgment


@dataclass
class FinalizeNode(BaseNode[State, Deps, ExtractionResult]):
    """Build final output with accepted pairs and provenance."""

    async def run(self, ctx: GraphRunContext[State, Deps]) -> End[ExtractionResult]:
        """Gather accepted pairs and build final result."""
        with logfire.span("FinalizeNode"):
            accepted_pairs = []

            for pair_key, judgment in ctx.state.final_pair_judgments.items():
                if judgment.accepted:
                    # Gather all quotes for this pair
                    all_quotes = []
                    for resource_id, pairs_list in ctx.state.pairs_by_resource.items():
                        for pair_mention in pairs_list:
                            if (
                                pair_mention.entity1 == pair_key[0]
                                and pair_mention.entity2 == pair_key[1]
                                and pair_mention.relationship_type == pair_key[2]
                            ):
                                all_quotes.extend(pair_mention.quotes)

                    # Get entity types from entity mentions
                    entity1_type = "unknown"
                    entity2_type = "unknown"
                    for entities_dict in ctx.state.entities_by_resource.values():
                        if pair_key[0] in entities_dict:
                            entity1_type = entities_dict[pair_key[0]].kind
                        if pair_key[1] in entities_dict:
                            entity2_type = entities_dict[pair_key[1]].kind

                    # Build PairWithProvenance
                    pair_with_prov = PairWithProvenance(
                        entity1=pair_key[0],
                        entity2=pair_key[1],
                        relationship_type=pair_key[2],
                        entity1_type=entity1_type,
                        entity2_type=entity2_type,
                        all_quotes=all_quotes,
                        assessments=ctx.state.pair_assessments[pair_key],
                        final_judgment=judgment,
                    )
                    accepted_pairs.append(pair_with_prov)

            # Build metadata
            total_entities = sum(
                len(entities) for entities in ctx.state.entities_by_resource.values()
            )
            total_pairs = sum(
                len(pairs) for pairs in ctx.state.pairs_by_resource.values()
            )
            pairs_accepted = len(accepted_pairs)
            pairs_rejected = len(ctx.state.final_pair_judgments) - pairs_accepted

            metadata = ExtractionMetadata(
                topic=ctx.state.topic,
                resource_count=len(ctx.deps.resource_pool.resources),
                total_entities_found=total_entities,
                total_pairs_found=total_pairs,
                pairs_accepted=pairs_accepted,
                pairs_rejected=pairs_rejected,
                quotes_validated=ctx.state.quotes_validated,
                quotes_failed=ctx.state.quotes_failed,
            )

            result = ExtractionResult(accepted_pairs=accepted_pairs, metadata=metadata)

            return End(result)
