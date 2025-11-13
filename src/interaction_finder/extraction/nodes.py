"""Graph nodes for association extraction pipeline.

Each node represents a stage in the pipeline. Nodes control flow via
return type annotations. All branching and looping happens in nodes;
agents only produce typed data.

Pipeline stages:
1. ProcessDocumentsNode - Process all documents concurrently (entities → pairs → assessments)
2. MergeEntitiesNode - Merge entities globally + update pair references
3. JudgeCrossDocumentNode - Make final accept/reject decisions
4. FinalizeNode - Build final output
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Union

from pydantic_ai.usage import RunUsage
from pydantic_graph import BaseNode, End, GraphRunContext

from interaction_finder.agent_utils import rename_agent
from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.document_pipeline import (
    assess_document_pairs,
    extract_document_entities,
    extract_document_pairs,
    identify_document_proximal_sets,
    validate_entity_kinds,
)
from interaction_finder.extraction.judge_cross_document import (
    get_cross_document_judge_agent,
)
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
    collect_relevant_text_for_quotes,
    make_entity_pair_key,
    normalize_for_comparison,
)
from interaction_finder.logging import logfire
from interaction_finder.resources import Resource


@dataclass
class ProcessDocumentsNode(BaseNode[State, Deps, ExtractionResult]):
    """Process all documents concurrently through the per-document pipeline.

    For each document (in parallel):
    1. Extract entities with quote validation
    2. Filter by target entity kinds
    3. Identify proximal entity sets
    4. Extract pairs from proximal regions
    5. Deduplicate and assess pairs

    Documents that fail processing are logged but don't block other documents.
    """

    async def run(
        self, ctx: GraphRunContext[State, Deps]
    ) -> Union["MergeEntitiesNode", End[ExtractionResult]]:
        """Process all resources concurrently."""
        with logfire.span("ProcessDocumentsNode"):
            resources = ctx.deps.resource_pool.resources

            if not resources:
                ctx.deps.logger.warning("No resources to process")
                return End(self._empty_result(ctx))

            # Set up progress tracking
            if ctx.deps.progress:
                ctx.deps.progress.documents_total = len(resources)
                ctx.deps.progress.documents_in_progress = len(resources)
                ctx.deps.progress.set_phase_extracting()

            # Process all documents in parallel, tracking progress as they complete
            tasks = [self._process_document(resource, ctx) for resource in resources]
            for coro in asyncio.as_completed(tasks):
                await coro
                # Update progress: decrement in-progress as each document completes
                if ctx.deps.progress:
                    # documents_processed is incremented inside _process_document
                    # Here we just need to decrement in_progress
                    ctx.deps.progress.documents_in_progress = max(
                        0, ctx.deps.progress.documents_in_progress - 1
                    )
                    ctx.deps.progress.update()

            # Check if we found any validated entities
            if not ctx.state.validated_entities_by_resource:
                ctx.deps.logger.warning("No entities extracted from documents")
                return End(self._empty_result(ctx))

            return MergeEntitiesNode()

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

    async def _process_document(
        self, resource: Resource, ctx: GraphRunContext[State, Deps]
    ):
        """Process a single document through the full per-document pipeline."""
        with logfire.span(
            f"Document {resource.id.id}: {resource.title[:60]}",
            url=resource.id.url,
        ):
            try:
                # Stage 1: Extract entities with quote validation
                (
                    entities,
                    quotes_validated,
                    quotes_failed,
                ) = await extract_document_entities(
                    resource,
                    ctx.state.topic,
                    ctx.state.target_entity_types,
                    ctx.deps.config,
                    ctx.deps,
                )

                # Update quote counters
                ctx.state.quotes_validated += quotes_validated
                ctx.state.quotes_failed += quotes_failed

                if not entities:
                    return  # No entities found in this document

                # Store raw entities
                ctx.state.entities_by_resource[resource.id] = entities

                # Stage 2: Validate entity kinds
                if ctx.deps.progress:
                    ctx.deps.progress.set_phase_validating()

                validated = validate_entity_kinds(
                    entities, ctx.state.target_entity_types
                )

                if not validated:
                    return  # No valid entities after kind filtering

                ctx.state.validated_entities_by_resource[resource.id] = validated

                # Update progress
                if ctx.deps.progress:
                    ctx.deps.progress.documents_processed += 1
                    ctx.deps.progress.entities_found = sum(
                        len(e) for e in ctx.state.entities_by_resource.values()
                    )
                    ctx.deps.progress.quotes_validated = ctx.state.quotes_validated
                    ctx.deps.progress.quotes_failed = ctx.state.quotes_failed
                    ctx.deps.progress.update()

                # Stage 3: Identify proximal entity sets
                if ctx.deps.progress:
                    ctx.deps.progress.set_phase_proximal()

                proximal_sets = identify_document_proximal_sets(
                    validated,
                    resource,
                    ctx.deps.config.tools.extraction.proximal_window_chunks,
                )

                if not proximal_sets:
                    return  # No co-occurring entities found

                ctx.state.proximal_sets_by_resource[resource.id] = proximal_sets

                # Stage 4: Extract pairs from proximal sets
                if ctx.deps.progress:
                    ctx.deps.progress.set_phase_extracting_pairs()

                pairs, pairs_validated, pairs_failed = await extract_document_pairs(
                    proximal_sets,
                    validated,
                    resource,
                    ctx.state.topic,
                    ctx.state.permitted_pairs,
                    ctx.deps.config.tools.extraction.region_padding_chunks,
                    ctx.deps.config,
                    ctx.deps,
                )

                # Update quote counters from pair extraction
                ctx.state.quotes_validated += pairs_validated
                ctx.state.quotes_failed += pairs_failed

                if not pairs:
                    return  # No pairs found in proximal sets

                # Update pairs_found counter
                if ctx.deps.progress:
                    ctx.deps.progress.pairs_found += len(pairs)
                    ctx.deps.progress.pairs_in_progress += len(pairs)
                    ctx.deps.progress.update()

                # Stage 5: Deduplicate and assess pairs
                if ctx.deps.progress:
                    ctx.deps.progress.set_phase_assessing()
                assessments = await assess_document_pairs(
                    pairs,
                    validated,
                    resource,
                    ctx.state.topic,
                    ctx.deps.config.tools.extraction.region_padding_chunks,
                    ctx.deps.config,
                    ctx.deps,
                )

                if assessments:
                    ctx.state.pair_assessments_by_resource[resource.id] = assessments

                    # Update progress: increment assessed, decrement in-progress
                    if ctx.deps.progress:
                        ctx.deps.progress.pairs_assessed += len(assessments)
                        ctx.deps.progress.pairs_in_progress = max(
                            0, ctx.deps.progress.pairs_in_progress - len(assessments)
                        )
                        ctx.deps.progress.update()
                else:
                    # No assessments means pairs were filtered out, still need to decrement in-progress
                    if ctx.deps.progress:
                        ctx.deps.progress.pairs_in_progress = max(
                            0, ctx.deps.progress.pairs_in_progress - len(pairs)
                        )
                        ctx.deps.progress.update()

            except Exception as e:
                # Log error but don't fail entire pipeline
                ctx.deps.logger.error(
                    f"[red]Document processing failed for {resource.title}: {type(e).__name__}: {e}[/red]",
                    extra={"markup": True},
                )


@dataclass
class MergeEntitiesNode(BaseNode[State, Deps, ExtractionResult]):
    """Merge entities globally across all documents.

    Algorithm:
    1. Collect all unique normalized entity forms from all documents
    2. Find substring relationships between normalized forms
    3. Query LLM for merge decisions on unique pairs (with caching)
    4. Apply merge rules consistently across all documents

    This approach ensures:
    - Complete coverage (finds all merge opportunities)
    - Efficiency (one LLM call per unique normalized pair)
    - Consistency (same canonical name across documents)
    """

    async def run(self, ctx: GraphRunContext[State, Deps]) -> "JudgeCrossDocumentNode":
        """Merge entities globally and update pair references."""
        with logfire.span("MergeEntitiesNode"):
            # Step 1: Collect unique normalized entities with canonical variants
            unique_entities = self._collect_unique_entities(ctx)

            # Step 2: Find substring pairs globally
            substring_pairs = self._find_global_substring_pairs(unique_entities)

            # Step 3: Get merge decisions from LLM (with caching)
            merge_rules = await self._get_global_merge_decisions(substring_pairs, ctx)

            # Step 4: Resolve transitive merge chains (A→B→C becomes A→C)
            resolved_rules = self._resolve_transitive_merges(merge_rules)

            # Step 5: Apply merge rules to entity dictionaries
            self._apply_merge_rules_globally(resolved_rules, ctx)

            # Step 6: Update entity references in all pair assessments
            self._update_pair_entity_references(resolved_rules, ctx)

            ctx.deps.logger.info(
                f"Global merging: {len(resolved_rules)} rules applied, "
                f"{ctx.state.entities_merged} entities merged"
            )

            return JudgeCrossDocumentNode()

    def _collect_unique_entities(
        self, ctx: GraphRunContext[State, Deps]
    ) -> dict[str, dict[str, set[str]]]:
        """Collect all unique normalized entities with their canonical variants.

        Returns:
            {kind: {normalized_name: {canonical_variant1, canonical_variant2, ...}}}
        """
        unique_by_kind: dict[str, dict[str, set[str]]] = {}

        for entities in ctx.state.validated_entities_by_resource.values():
            for entity_name, entity in entities.items():
                if entity.kind not in unique_by_kind:
                    unique_by_kind[entity.kind] = {}

                norm_name = normalize_for_comparison(entity_name)
                if norm_name not in unique_by_kind[entity.kind]:
                    unique_by_kind[entity.kind][norm_name] = set()

                unique_by_kind[entity.kind][norm_name].add(entity_name)

                # Track canonical variant globally
                norm_key = (norm_name, entity.kind)
                if norm_key not in ctx.state.canonical_name_variants:
                    ctx.state.canonical_name_variants[norm_key] = set()
                ctx.state.canonical_name_variants[norm_key].add(entity_name)

        return unique_by_kind

    def _find_global_substring_pairs(
        self, unique_entities: dict[str, dict[str, set[str]]]
    ) -> dict[str, list[tuple[str, str]]]:
        """Find substring relationships between normalized entity names.

        Returns:
            {kind: [(norm_parent, norm_child), ...]}
        """
        pairs_by_kind: dict[str, list[tuple[str, str]]] = {}

        for kind, normalized_entities in unique_entities.items():
            norm_list = list(normalized_entities.keys())
            pairs = []

            for i, norm1 in enumerate(norm_list):
                for norm2 in norm_list[i + 1 :]:
                    if norm1 == norm2:
                        continue
                    elif norm1 in norm2:
                        # norm1 is parent (general), norm2 is child (specific)
                        pairs.append((norm1, norm2))
                    elif norm2 in norm1:
                        # norm2 is parent (general), norm1 is child (specific)
                        pairs.append((norm2, norm1))

            if pairs:
                pairs_by_kind[kind] = pairs

        return pairs_by_kind

    async def _get_global_merge_decisions(
        self,
        substring_pairs_by_kind: dict[str, list[tuple[str, str]]],
        ctx: GraphRunContext[State, Deps],
    ) -> dict[tuple[str, str], str]:
        """Query LLM for all unique normalized pairs.

        Returns:
            {(norm_child, kind): norm_parent} for pairs that should merge
        """
        from interaction_finder.extraction.models import EntityMergeDecision

        merge_rules: dict[tuple[str, str], str] = {}

        for kind, pairs in substring_pairs_by_kind.items():
            # Separate cached and uncached pairs
            uncached_pairs = []

            for norm_parent, norm_child in pairs:
                cache_key = (norm_parent, norm_child, kind)

                if cache_key in ctx.state.merge_decision_cache:
                    # Cache hit
                    if ctx.state.merge_decision_cache[cache_key]:
                        merge_rules[(norm_child, kind)] = norm_parent
                    ctx.state.merge_cache_hits += 1
                else:
                    # Cache miss
                    uncached_pairs.append((norm_parent, norm_child))
                    ctx.state.merge_cache_misses += 1

            # Query LLM for uncached pairs in batches
            if uncached_pairs:
                batch_size = ctx.deps.config.tools.extraction.merge_batch_size

                for i in range(0, len(uncached_pairs), batch_size):
                    batch = uncached_pairs[i : i + batch_size]

                    # Build prompt
                    pairs_description = []
                    for norm_parent, norm_child in batch:
                        pairs_description.append(
                            f"- Parent: '{norm_parent}' (type: {kind})\n"
                            f"  Child: '{norm_child}' (type: {kind})"
                        )

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

                    # Call LLM
                    agent = get_entity_merge_agent(ctx.deps.config)
                    usage = RunUsage()
                    try:
                        with rename_agent(
                            agent,
                            name=f"MergeEntitiesNode (kind={kind}, batch {i // batch_size + 1})",
                        ):
                            result = await agent.run(prompt, deps=ctx.deps, usage=usage)

                        # Store decisions in cache
                        for decision in result.output.decisions:
                            cache_key = (
                                normalize_for_comparison(decision.parent_entity),
                                normalize_for_comparison(decision.child_entity),
                                kind,
                            )
                            ctx.state.merge_decision_cache[cache_key] = (
                                decision.should_merge
                            )

                            if decision.should_merge:
                                norm_child = normalize_for_comparison(
                                    decision.child_entity
                                )
                                norm_parent = normalize_for_comparison(
                                    decision.parent_entity
                                )
                                merge_rules[(norm_child, kind)] = norm_parent

                    except (TimeoutError, ConnectionError, ValueError) as e:
                        ctx.deps.logger.error(
                            f"Entity merge decision failed: {type(e).__name__}: {e}"
                        )
                        # Continue without merging this batch

        return merge_rules

    def _resolve_transitive_merges(
        self, merge_rules: dict[tuple[str, str], str]
    ) -> dict[tuple[str, str], str]:
        """Resolve transitive merge chains.

        If A→B and B→C, resolve to A→C (and B→C).
        This ensures all entities in a chain ultimately point to the final parent.

        Algorithm:
        1. For each child in merge_rules, follow parent chain until we find
           a parent that isn't itself a child
        2. Update rule to point directly to final parent

        Returns:
            Resolved merge rules with transitive chains collapsed
        """
        resolved = {}

        for (child_norm, kind), parent_norm in merge_rules.items():
            # Follow the chain: child → parent → parent's parent → ...
            final_parent = parent_norm
            visited = {child_norm}  # Prevent infinite loops

            while (final_parent, kind) in merge_rules:
                if final_parent in visited:
                    # Cycle detected - stop here
                    break
                visited.add(final_parent)
                final_parent = merge_rules[(final_parent, kind)]

            resolved[(child_norm, kind)] = final_parent

        return resolved

    def _apply_merge_rules_globally(
        self,
        merge_rules: dict[tuple[str, str], str],
        ctx: GraphRunContext[State, Deps],
    ) -> None:
        """Apply merge rules to all documents using normalized lookups.

        Merge rules are resolved transitively before application, so if A→B→C,
        we apply A→C directly.
        """
        if not merge_rules:
            return

        # Resolve transitive chains first
        resolved_rules = self._resolve_transitive_merges(merge_rules)

        for resource_id, entities in ctx.state.validated_entities_by_resource.items():
            entities_to_merge = []

            for child_canonical in list(entities.keys()):
                child_entity = entities[child_canonical]
                child_norm = normalize_for_comparison(child_canonical)
                rule_key = (child_norm, child_entity.kind)

                if rule_key in resolved_rules:
                    parent_norm = resolved_rules[rule_key]

                    # Find parent in this document by normalized lookup
                    parent_canonical = None
                    for candidate_name, candidate_entity in entities.items():
                        if (
                            normalize_for_comparison(candidate_name) == parent_norm
                            and candidate_entity.kind == child_entity.kind
                        ):
                            parent_canonical = candidate_name
                            break

                    # Merge if parent exists
                    if parent_canonical and parent_canonical != child_canonical:
                        entities_to_merge.append((parent_canonical, child_canonical))

            # Apply merges - with transitive resolution, no entity should be missing
            for parent_name, child_name in entities_to_merge:
                if parent_name not in entities or child_name not in entities:
                    # This shouldn't happen with transitive resolution, but check anyway
                    continue

                parent = entities[parent_name]
                child = entities[child_name]

                # Merge child into parent
                if child.name not in parent.aliases:
                    parent.aliases.append(child.name)
                parent.quotes.extend(child.quotes)
                parent.reasoning += f" | MERGED_GLOBAL({child.name}): {child.reasoning}"
                for alias in child.aliases:
                    if alias not in parent.aliases:
                        parent.aliases.append(alias)

                # Remove child
                del entities[child_name]
                ctx.state.entities_merged += 1

    def _update_pair_entity_references(
        self,
        merge_rules: dict[tuple[str, str], str],
        ctx: GraphRunContext[State, Deps],
    ) -> None:
        """Update EntityMention references in PairAssessments after merging.

        After entities are merged, pair assessments may reference old entity names.
        This method updates those references to use the canonical merged names.

        Parameters:
            merge_rules: Mapping of (normalized_child, kind) → normalized_parent
            ctx: Graph run context with state containing assessments
        """
        if not merge_rules:
            return

        for resource_id, assessments in ctx.state.pair_assessments_by_resource.items():
            entities = ctx.state.validated_entities_by_resource.get(resource_id, {})

            for assessment in assessments:
                # Check if entity1 was merged
                e1_norm = normalize_for_comparison(assessment.entity1.name)
                e1_key = (e1_norm, assessment.entity1.kind)

                if e1_key in merge_rules:
                    merged_name = self._find_canonical_name(
                        merge_rules[e1_key], entities, assessment.entity1.kind
                    )
                    if merged_name and merged_name != assessment.entity1.name:
                        self._update_entity_in_assessment(
                            assessment, "entity1", merged_name
                        )

                # Check if entity2 was merged
                e2_norm = normalize_for_comparison(assessment.entity2.name)
                e2_key = (e2_norm, assessment.entity2.kind)

                if e2_key in merge_rules:
                    merged_name = self._find_canonical_name(
                        merge_rules[e2_key], entities, assessment.entity2.kind
                    )
                    if merged_name and merged_name != assessment.entity2.name:
                        self._update_entity_in_assessment(
                            assessment, "entity2", merged_name
                        )

    def _find_canonical_name(
        self,
        normalized_target: str,
        entities: dict[str, EntityMention],
        kind: str,
    ) -> str | None:
        """Find canonical entity name matching normalized target.

        Parameters:
            normalized_target: Normalized name to find
            entities: Entity dictionary to search
            kind: Entity kind to match

        Returns:
            Canonical name if found, None otherwise
        """
        for name, entity in entities.items():
            if (
                normalize_for_comparison(name) == normalized_target
                and entity.kind == kind
            ):
                return name
        return None

    def _update_entity_in_assessment(
        self,
        assessment: PairAssessment,
        attr: str,
        new_name: str,
    ) -> None:
        """Update entity reference in assessment (mutates in place).

        Parameters:
            assessment: PairAssessment to update
            attr: Attribute name ("entity1" or "entity2")
            new_name: New canonical name to use
        """
        entity = getattr(assessment, attr)
        old_name = entity.name

        # Add old name to aliases if not already present
        if old_name not in entity.aliases:
            entity.aliases.append(old_name)

        # Update name
        entity.name = new_name

        # Update reasoning to show merge
        entity.reasoning += f" | MERGED_FROM({old_name})"


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

            # Update unique pairs count and set in-progress
            if ctx.deps.progress:
                ctx.deps.progress.unique_pairs = len(assessments_by_pair)
                ctx.deps.progress.judgments_in_progress = len(assessments_by_pair)
                ctx.deps.progress.update()

            # Judge each pair
            tasks = []
            for pair_key, assessments in assessments_by_pair.items():
                tasks.append(self._judge_pair(pair_key, assessments, ctx))

            # Run all judgments in parallel, tracking progress as they complete
            if tasks:
                for coro in asyncio.as_completed(tasks):
                    pair_key, judgment = await coro
                    ctx.state.pair_judgments[pair_key] = judgment
                    # Update accepted/rejected counts and decrement in-progress
                    if ctx.deps.progress:
                        if judgment.accepted:
                            ctx.deps.progress.accepted += 1
                        else:
                            ctx.deps.progress.rejected += 1
                        ctx.deps.progress.judgments_in_progress = max(
                            0, ctx.deps.progress.judgments_in_progress - 1
                        )
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
