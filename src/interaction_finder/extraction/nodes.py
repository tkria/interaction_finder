"""Graph nodes for association extraction pipeline.

Each node represents a stage in the pipeline. Nodes control flow via
return type annotations. All branching and looping happens in nodes;
agents only produce typed data.

Pipeline stages:
1. ProcessDocumentsNode - Process all documents concurrently (entities → pairs → assessments)
2. ConsolidateEntitiesNode - Consolidate entities globally + update pair references
3. ConsolidateRelationshipsNode - Consolidate relationship labels + classify polarities
4. SweepCoMentionsNode - Find missed co-mentions of assessed pairs
5. ConsolidateNewRelationshipsNode - Classify polarities for new relationship labels
6. JudgeCrossDocumentNode - Make final accept/reject decisions
7. FinalizeNode - Build final output
"""

import asyncio
import logging
import re
import secrets
import string
from collections import defaultdict
from dataclasses import dataclass
from typing import Union

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.usage import RunUsage
from pydantic_graph import BaseNode, End, GraphRunContext

from interaction_finder.agent_utils import rename_agent
from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.document_pipeline import (
    analyze_document,
    assess_document_pairs,
    extract_document_pairs,
    validate_entity_kinds,
)
from interaction_finder.extraction.judge_cross_document import (
    get_cross_document_judge_agent,
)
from interaction_finder.extraction.consolidate_entities import (
    get_entity_consolidation_agent,
)
from interaction_finder.extraction.models import (
    ClusterDecision,
    EntityKindMerges,
    EntityMention,
    EntityMergeRule,
    EntityPairKey,
    EntityRef,
    EvidenceQuality,
    ExtractionMetadata,
    ExtractionResult,
    PairAssessment,
    PairJudgment,
    PairSpread,
    RelationshipConsolidation,
    SimpleEntity,
)
from interaction_finder.extraction.state import MergeCacheForKind, State
from interaction_finder.extraction.entity_matching import (
    extract_entity_variants,
    find_consolidation_candidates,
    find_entity_match,
    SpeculatedVariant,
)
from interaction_finder.extraction.utils import (
    adjust_heading_levels,
    are_relationships_opposed,
    build_pair_spread,
    collect_relevant_text_for_quotes,
    extract_all_forms,
    extract_document_citations,
    identify_proximal_sets,
    is_obvious_variant,
    make_entity_pair_key,
    normalize_for_comparison,
    opposition_map_from_consolidations,
    osa_distance,
    validate_document_citations,
)
from interaction_finder.logging import logfire
from interaction_finder.resources import Resource, ResourceId

# Characters for generating verification tokens (alphanumeric, mixed case)
_TOKEN_CHARS = string.ascii_letters + string.digits


def _generate_token(length: int = 4) -> str:
    """Generate a random alphanumeric token for pair verification."""
    return "".join(secrets.choice(_TOKEN_CHARS) for _ in range(length))


def _get_merge_cache(state: State, kind: str) -> MergeCacheForKind:
    """Get or create the merge cache for an entity kind."""
    if kind not in state.merge_cache_by_kind:
        state.merge_cache_by_kind[kind] = MergeCacheForKind()
    return state.merge_cache_by_kind[kind]


def _aggregate_cache_stats(state: State) -> tuple[int, int]:
    """Aggregate cache hits and misses across all kinds."""
    hits = sum(c.hits for c in state.merge_cache_by_kind.values())
    misses = sum(c.misses for c in state.merge_cache_by_kind.values())
    return hits, misses


def _resolve_pair_from_decision(
    decision_id: int,
    decision_token: str,
    id_to_pair_info: dict[int, tuple[tuple[str, str], str, str]],
    token_to_id: dict[str, int],
    logger,
) -> tuple[tuple[str, str], str] | None:
    """Resolve canonical pair from LLM decision, using token as verification/fallback.

    Returns (canonical_pair, norm_child) or None if unresolvable.
    The norm_child is used to key rules by normalized form for cross-document consistency.
    Logs warnings for any mismatches.
    """
    pair_info = id_to_pair_info.get(decision_id)
    token_id = token_to_id.get(decision_token)
    # Both ID and token invalid
    if pair_info is None and token_id is None:
        logger.warning(
            f"LLM decision unresolvable: pair_id={decision_id}, "
            f"token='{decision_token}' - neither found"
        )
        return None
    # ID invalid but token valid - use token
    if pair_info is None:
        logger.warning(
            f"LLM decision ID mismatch: pair_id={decision_id} not found, "
            f"but token '{decision_token}' maps to id={token_id}. Using token."
        )
        info = id_to_pair_info[token_id]
        return (info[0], info[1])
    # ID valid - check token
    canonical_pair, norm_child, expected_token = pair_info
    if decision_token == expected_token:
        return (canonical_pair, norm_child)  # Perfect match
    # Token mismatch - prefer token if it points to a different valid pair
    if token_id is not None and token_id != decision_id:
        logger.warning(
            f"LLM decision conflict: pair_id={decision_id} (token '{expected_token}') "
            f"but got token '{decision_token}' (id={token_id}). Using token."
        )
        info = id_to_pair_info[token_id]
        return (info[0], info[1])
    # Token invalid/same-id but ID valid - trust ID
    logger.warning(
        f"LLM decision token mismatch: pair_id={decision_id} expected "
        f"'{expected_token}' but got '{decision_token}'. Using ID."
    )
    return (canonical_pair, norm_child)


def _get_entity_aliases(canonical: str, state: "State") -> list[str]:
    """Get aliases for an entity from the global index."""
    if ref := state.global_entities.get(canonical):
        return ref.aliases()
    return []


def _aggregate_evidence(assessments: list[PairAssessment]) -> EvidenceQuality:
    """Aggregate evidence from multiple assessments into a single EvidenceQuality.

    Uses median for overall level (robust to outliers) and mode for factor values.
    For even counts, takes the lower of the two middle values (conservative).
    """
    from collections import Counter
    from statistics import median_low

    if not assessments:
        return EvidenceQuality(
            directness="tangential",
            source_type="other",
            specificity="vague",
            language="speculative",
            overall=1,
        )
    # Use median evidence level (robust, conservative tie-break)
    levels = [a.evidence.overall for a in assessments]
    median_level = median_low(levels)
    # Use most common factor values
    directness = Counter(a.evidence.directness for a in assessments).most_common(1)[0][
        0
    ]
    source_type = Counter(a.evidence.source_type for a in assessments).most_common(1)[
        0
    ][0]
    specificity = Counter(a.evidence.specificity for a in assessments).most_common(1)[
        0
    ][0]
    language = Counter(a.evidence.language for a in assessments).most_common(1)[0][0]
    return EvidenceQuality(
        directness=directness,
        source_type=source_type,
        specificity=specificity,
        language=language,
        overall=median_level,
    )


async def _save_partial_checkpoint(
    ctx: GraphRunContext[State, Deps], stage: str
) -> None:
    """Save partial checkpoint with current state for resumption.

    Parameters:
        ctx: Graph context with state and dependencies
        stage: Name of the stage just completed
    """
    if not ctx.deps.checkpoint_path:
        return

    from pathlib import Path

    from interaction_finder.checkpoint import ExtractionStageData, PipelineCheckpoint
    from interaction_finder.version import get_version_string

    # Helper to count items in nested dicts
    def count_nested(d):
        return sum(len(v) for v in d.values())

    # Build partial checkpoint with resume state
    checkpoint = PipelineCheckpoint(
        topic=ctx.state.topic,
        resources=ctx.deps.resource_pool,
        created_by=get_version_string(),
        keywords=ctx.deps.input_checkpoint.keywords,
        search=ctx.deps.input_checkpoint.search,
        extraction=ExtractionStageData(
            target_entity_types=ctx.state.target_entity_types,
            permitted_pairs={k: list(v) for k, v in ctx.state.permitted_pairs.items()},
            judgments=[],
            consolidated=ctx.state.consolidated,
            paper_quality={
                rid.url: assessment
                for rid, assessment in ctx.state.paper_quality.items()
            },
            metadata=ExtractionMetadata(
                topic=ctx.state.topic,
                resource_count=len(ctx.deps.resource_pool.resources),
                total_entities_found=count_nested(ctx.state.entities_by_resource),
                entities_after_validation=count_nested(
                    ctx.state.validated_entities_by_resource
                ),
                entities_merged=ctx.state.entities_merged,
                merge_cache_hits=sum(
                    c.hits for c in ctx.state.merge_cache_by_kind.values()
                ),
                merge_cache_misses=sum(
                    c.misses for c in ctx.state.merge_cache_by_kind.values()
                ),
                proximal_sets_found=count_nested(ctx.state.proximal_sets_by_resource),
                total_pairs_found=0,
                pairs_accepted=0,
                pairs_rejected=0,
                quotes_validated=ctx.state.quotes_validated,
                quotes_failed=ctx.state.quotes_failed,
                resume_from=stage,
                resume_state=ctx.state.to_dict(),
            ),
        ),
    )

    Path(ctx.deps.checkpoint_path).write_text(checkpoint.model_dump_json(indent=2))
    ctx.deps.logger.info(f"Saved partial checkpoint after {stage}")


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
    ) -> Union["ConsolidateEntitiesNode", End[ExtractionResult]]:
        """Process all resources concurrently."""
        with logfire.span("ProcessDocumentsNode"):
            resources = ctx.deps.resource_pool.resources

            if not resources:
                ctx.deps.logger.warning("No resources to process")
                return End(self._empty_result(ctx))

            # Set up progress tracking
            if ctx.deps.progress:
                ctx.deps.progress["Processed"].total = len(resources)
                ctx.deps.progress["Processed"].activate()
                ctx.deps.progress.set_status("Extracting entities")

            # Process all documents in parallel, tracking progress as they complete
            # Note: counter updates happen inside _process_document for tight scoping
            tasks = [self._process_document(resource, ctx) for resource in resources]
            for coro in asyncio.as_completed(tasks):
                await coro
            # Mark document processing phase complete
            if ctx.deps.progress:
                ctx.deps.progress["Processed"].complete()
                ctx.deps.progress["Pairs assessed"].complete()
            # Check if we found any validated entities
            if not ctx.state.validated_entities_by_resource:
                ctx.deps.logger.warning("No entities extracted from documents")
                return End(self._empty_result(ctx))

            # Save partial checkpoint after expensive document processing stage
            await _save_partial_checkpoint(ctx, "process_documents")

            return ConsolidateEntitiesNode()

    def _empty_result(self, ctx: GraphRunContext[State, Deps]) -> ExtractionResult:
        """Create empty result for early termination."""
        return ExtractionResult(
            topic=ctx.state.topic,
            target_entity_types=ctx.state.target_entity_types,
            permitted_pairs={k: list(v) for k, v in ctx.state.permitted_pairs.items()},
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
                # Stage 1: Analyze document (quality assessment + entity extraction)
                # Combined into single LLM call - quality assessed first, then entities
                # Note: .work() is called inside analyze_document after semaphore acquisition
                (
                    entities,
                    quality_assessment,
                    quotes_validated,
                    quotes_failed,
                ) = await analyze_document(
                    resource,
                    ctx.state.topic,
                    ctx.state.target_entity_types,
                    ctx.deps.config,
                    ctx.deps,
                )
                # Store paper quality assessment (even if entity extraction found nothing)
                if quality_assessment is not None:
                    ctx.state.paper_quality[resource.id] = quality_assessment
                # Update quote counters
                ctx.state.quotes_validated += quotes_validated
                ctx.state.quotes_failed += quotes_failed
                if not entities:
                    return  # No entities found in this document

                # Store raw entities
                ctx.state.entities_by_resource[resource.id] = entities

                # Stage 2: Validate entity kinds
                if ctx.deps.progress:
                    ctx.deps.progress.set_status("Validating entities")

                validated = validate_entity_kinds(
                    entities, ctx.state.target_entity_types
                )

                if not validated:
                    return  # No valid entities after kind filtering

                # Store validated entities as EntityRefs (preserve original mentions)
                ctx.state.validated_entities_by_resource[resource.id] = {
                    name: EntityRef(canonical=name, mentions=[mention])
                    for name, mention in validated.items()
                }

                # Update progress (documents_processed incremented in main loop for atomicity)
                if ctx.deps.progress:
                    ctx.deps.progress["Entities"].completed = sum(
                        len(e) for e in ctx.state.entities_by_resource.values()
                    )
                    ctx.deps.progress["Quotes"].completed = ctx.state.quotes_validated
                    # Set note for failed quotes
                    if ctx.state.quotes_failed > 0:
                        ctx.deps.progress[
                            "Quotes"
                        ].note = f"({ctx.state.quotes_failed} invalid)"

                # Stage 3: Identify proximal entity sets
                if ctx.deps.progress:
                    ctx.deps.progress.set_status("Finding proximal pairs")

                proximal_sets = identify_proximal_sets(
                    validated,
                    ctx.deps.config.tools.extraction.proximal_window_chunks,
                    resource,
                )

                if not proximal_sets:
                    return  # No co-occurring entities found

                ctx.state.proximal_sets_by_resource[resource.id] = proximal_sets

                # Stage 4: Extract pairs from proximal sets
                if ctx.deps.progress:
                    ctx.deps.progress.set_status("Extracting relationships")

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

                # Stage 5: Deduplicate and assess pairs
                # Note: total, .work() and .done() are updated inside assess_document_pairs
                if ctx.deps.progress:
                    ctx.deps.progress.set_status("Assessing pairs")
                assessments = await assess_document_pairs(
                    pairs,
                    ctx.state.validated_entities_by_resource[resource.id],
                    resource,
                    ctx.state.topic,
                    ctx.deps.config.tools.extraction.region_padding_chunks,
                    ctx.deps.config,
                    ctx.deps,
                )

                if assessments:
                    ctx.state.pair_assessments_by_resource[resource.id] = assessments

            except Exception as e:
                # Log error but don't fail entire pipeline
                ctx.deps.logger.error(
                    f"[red]Document processing failed for {resource.title}: {type(e).__name__}: {e}[/red]",
                    extra={"markup": True},
                )
            finally:
                # Update counters when work completes (tight scoping, even on error)
                if ctx.deps.progress:
                    ctx.deps.progress["Processed"].done()


@dataclass
class ConsolidateEntitiesNode(BaseNode[State, Deps, ExtractionResult]):
    """Consolidate entities globally across all documents.

    Algorithm:
    1. Collect all unique normalized entity forms from all documents
    2. Find substring relationships between normalized forms
    3. Query LLM for consolidation decisions (skip/merge/rename)
    4. Apply merge rules consistently across all documents
    5. If renames occurred, re-evaluate for new merge opportunities

    This approach ensures:
    - Complete coverage (finds all merge opportunities)
    - Efficiency (one LLM call per unique normalized pair)
    - Consistency (same canonical name across documents)
    - Topic-appropriate naming (verbose names can be simplified via rename)
    """

    async def run(
        self, ctx: GraphRunContext[State, Deps]
    ) -> "ConsolidateRelationshipsNode":
        """Consolidate entities globally and update pair references.

        Processes each entity kind concurrently, then applies all merge rules
        together with transitive resolution.
        """
        with logfire.span("ConsolidateEntitiesNode"):
            if ctx.deps.progress:
                ctx.deps.progress.set_status("Consolidating entities")
            # Capture initial entity state BEFORE any consolidation (once only)
            if not ctx.state.consolidated.entities.initial:
                ctx.state.consolidated.entities.initial = _snapshot_entity_counts(
                    ctx.state
                )
            # Collect entities once
            entities_by_kind, mentions_by_kind = self._collect_entity_variants(ctx)
            # Pre-initialize shared structures before concurrent execution to avoid races
            for kind in entities_by_kind:
                ctx.state.consolidated.entities.merges.setdefault(
                    kind, EntityKindMerges()
                )
                _get_merge_cache(ctx.state, kind)
            # Process each kind concurrently. This is safe because:
            # - merge_cache_by_kind is keyed by kind, so each kind has isolated cache state
            # - consolidated.entities.merges is keyed by kind (pre-initialized above)
            tasks = [
                self._consolidate_kind(
                    kind, entities, mentions_by_kind.get(kind, {}), ctx
                )
                for kind, entities in entities_by_kind.items()
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Collect rules, checking for errors
            all_rules: dict[tuple[str, str], tuple[str, str]] = {}
            for kind, result in zip(entities_by_kind.keys(), results):
                if isinstance(result, Exception):
                    ctx.deps.logger.error(f"{kind} consolidation failed: {result}")
                else:
                    all_rules.update(result)
            # Apply all rules with transitive resolution
            if all_rules:
                resolved_rules = self._resolve_transitive_merges(all_rules)
                self._apply_merge_rules_globally(resolved_rules, ctx)
                self._update_pair_entity_references(resolved_rules, ctx)
            ctx.deps.logger.info(
                f"Entity consolidation: {len(all_rules)} rules applied, "
                f"{ctx.state.entities_merged} entities merged",
            )
            # Build global entity index (aggregate mentions across all resources)
            self._build_global_entity_index(ctx)
            # Save checkpoint after entity consolidation
            await _save_partial_checkpoint(ctx, "consolidate_entities")
            return ConsolidateRelationshipsNode()

    async def _consolidate_kind(
        self,
        kind: str,
        entities: dict[str, list[SpeculatedVariant]],
        mention_counts: dict[str, int],
        ctx: GraphRunContext[State, Deps],
    ) -> dict[tuple[str, str], tuple[str, str]]:
        """Process a single entity kind to completion.

        Re-clusters on each iteration but filters results to avoid redundant work:
        - Auto-merges: skip if already applied (tracked by normalized child name)
        - Groups: skip if already resolved, or if they don't contain new names from renames
        - Pairwise reviews: handled by LLM cache in _get_consolidation_decisions_for_kind

        Iteration continues only when renames create new canonical names that might
        cluster with other entities.

        Returns all merge rules for this kind as {(norm_name, kind): (target, reasoning)}.
        """
        with logfire.span(f"Consolidate {kind} entities", kind=kind):
            return await self._consolidate_kind_impl(
                kind, entities, mention_counts, ctx
            )

    async def _consolidate_kind_impl(
        self,
        kind: str,
        entities: dict[str, list[SpeculatedVariant]],
        mention_counts: dict[str, int],
        ctx: GraphRunContext[State, Deps],
    ) -> dict[tuple[str, str], tuple[str, str]]:
        """Implementation of per-kind consolidation (wrapped by _consolidate_kind)."""
        max_iterations = ctx.deps.config.tools.extraction.max_rename_iterations
        threshold = ctx.deps.config.tools.extraction.cluster_token_overlap_threshold
        kind_merges = ctx.state.consolidated.entities.merges[kind]
        # Track groups resolved (kept separate) to avoid re-asking
        resolved_groups: set[frozenset[str]] = set()
        all_rules: dict[tuple[str, str], tuple[str, str]] = {}
        # Track auto-merges already applied (keyed by normalized child)
        applied_auto_merges: set[str] = set()
        # Track new names from previous iteration (for targeted re-review)
        previous_new_names: set[str] = set()
        for iteration in range(1, max_iterations + 1):
            # Cluster entities
            ctx.deps.logger.info(
                f"Clustering {len(entities)} {kind} entities (threshold={threshold:.2f})"
            )
            candidates = find_consolidation_candidates(
                entities, threshold, mention_counts, ctx.deps.logger
            )
            # Filter out groups already resolved
            new_groups = [
                g for g in candidates.agent_review_groups if g not in resolved_groups
            ]
            filtered_resolved = len(candidates.agent_review_groups) - len(new_groups)
            # On subsequent iterations, only review groups containing new names from renames
            filtered_no_new_names = 0
            if previous_new_names:
                groups_with_new_names = [
                    g for g in new_groups if g & previous_new_names
                ]
                filtered_no_new_names = len(new_groups) - len(groups_with_new_names)
                new_groups = groups_with_new_names
            candidates.agent_review_groups = new_groups
            if filtered_resolved > 0 or filtered_no_new_names > 0:
                new_group_set = set(new_groups)
                candidates.merge_trees = [
                    t for t in candidates.merge_trees if t.entities in new_group_set
                ]
            # Filter auto-merges to only include new ones
            total_auto_merges = len(candidates.auto_merge)
            new_auto_merges = [
                (child, parent, reasoning)
                for child, parent, reasoning in candidates.auto_merge
                if normalize_for_comparison(child) not in applied_auto_merges
            ]
            filtered_auto_merges = total_auto_merges - len(new_auto_merges)
            # Track clusters presented to LLM
            for cluster in candidates.agent_review_groups:
                kind_merges.clusters.append(list(cluster))
            # Log clustering results
            multi_member_groups = [g for g in new_groups if len(g) > 1]
            skip_parts = []
            if filtered_resolved:
                skip_parts.append(f"{filtered_resolved} groups already resolved")
            if filtered_no_new_names:
                skip_parts.append(f"{filtered_no_new_names} groups without new names")
            if filtered_auto_merges:
                skip_parts.append(f"{filtered_auto_merges} auto-merges already applied")
            skip_info = f" (skipped: {', '.join(skip_parts)})" if skip_parts else ""
            ctx.deps.logger.info(
                f"Clustering produced {len(new_auto_merges)} auto-merges, "
                f"{len(candidates.agent_review)} pairwise reviews, "
                f"{len(multi_member_groups)} groups for LLM review{skip_info}"
            )
            # Early exit if no work remains
            if (
                not new_auto_merges
                and not candidates.agent_review
                and not candidates.agent_review_groups
            ):
                ctx.deps.logger.debug(f"No work remaining for {kind}, exiting early")
                break
            # Log contested warnings
            self._log_contested_warnings(
                candidates.contested_warnings, kind, entities, ctx
            )
            # Process auto-merge decisions (only new ones)
            iteration_rules: dict[tuple[str, str], tuple[str, str]] = {}
            for child, parent, reasoning in new_auto_merges:
                child_norm = normalize_for_comparison(child)
                iteration_rules[(child_norm, kind)] = (parent, reasoning)
                applied_auto_merges.add(child_norm)
            # Process agent review pairs
            new_names: set[str] = set()
            if candidates.agent_review:
                (
                    llm_rules,
                    llm_new_names,
                ) = await self._get_consolidation_decisions_for_kind(
                    candidates.agent_review, kind, entities, ctx
                )
                iteration_rules.update(llm_rules)
                new_names.update(llm_new_names)
            # Process agent review groups with merge trees
            if candidates.agent_review_groups:
                (
                    group_rules,
                    group_new_names,
                    group_resolved,
                ) = await self._get_group_consolidation_decisions(
                    candidates.agent_review_groups,
                    candidates.merge_trees,
                    kind,
                    entities,
                    ctx,
                )
                iteration_rules.update(group_rules)
                new_names.update(group_new_names)
                resolved_groups.update(group_resolved)
            # Accumulate rules
            all_rules.update(iteration_rules)
            # Check if we need another iteration
            if not new_names:
                break
            # Add new names for potential further clustering
            self._add_new_names_to_kind(new_names, entities)
            previous_new_names = new_names
            ctx.deps.logger.debug(
                f"{kind} consolidation iteration {iteration}: "
                f"{len(new_names)} renames, re-evaluating"
            )
        else:
            if new_names:
                ctx.deps.logger.warning(
                    f"{kind} consolidation hit max iterations ({max_iterations}) "
                    f"with {len(new_names)} pending renames"
                )
        return all_rules

    def _log_contested_warnings(
        self,
        contested_warnings: list,
        kind: str,
        entities: dict[str, list[SpeculatedVariant]],
        ctx: GraphRunContext[State, Deps],
    ) -> None:
        """Log contested variant warnings with full context."""
        if not contested_warnings:
            return

        def fmt_variant(v: SpeculatedVariant) -> str:
            source = (
                f":{v.source}"
                if v.speculation and v.source != "paren_expansion"
                else ""
            )
            return f"'{v.form}' [{v.speculation}{source}]"

        contested_lines = [f"Contested variants for {kind}:"]
        structured_data = []
        for norm_form, canonical_to_variants in contested_warnings:
            canonicals = list(canonical_to_variants.keys())
            canonical_list = "' and '".join(canonicals)
            variant_details = [
                f"    - {canonical}: {', '.join(fmt_variant(v) for v in entities.get(canonical, []) if normalize_for_comparison(v.form) == norm_form)}"
                for canonical in canonicals
            ]
            similarity = self._analyze_entity_similarity(canonicals)
            contested_lines.extend(
                [
                    f"  '{norm_form}' from '{canonical_list}':",
                    *variant_details,
                    f"    - Similarity: {similarity}",
                ]
            )
            structured_data.append(
                {
                    "normalized_form": norm_form,
                    "entities": canonicals,
                    "similarity": similarity,
                }
            )
        ctx.deps.logger.info(
            "\n".join(contested_lines),
            extra={"contested_variants": structured_data, "entity_kind": kind},
        )

    def _add_new_names_to_kind(
        self, new_names: set[str], entities: dict[str, list[SpeculatedVariant]]
    ) -> None:
        """Add new canonical names from renames to entity dict for re-clustering."""
        for new_canonical in new_names:
            if new_canonical not in entities:
                entities[new_canonical] = [
                    SpeculatedVariant(
                        form=new_canonical,
                        speculation=0,
                        source="original",
                        is_from_alias=False,
                    )
                ]

    def _collect_entity_variants(
        self, ctx: GraphRunContext[State, Deps]
    ) -> tuple[
        dict[str, dict[str, list[SpeculatedVariant]]], dict[str, dict[str, int]]
    ]:
        """Collect all entities with their speculated variants and mention counts.

        Uses the new extract_entity_variants() system which tracks speculation
        levels for each variant form. This enables automatic merge decisions
        based on confidence thresholds and provides rich provenance.

        Returns:
            Tuple of:
            - {kind: {canonical_name: [SpeculatedVariant, ...]}}
            - {kind: {canonical_name: mention_count}}
        """
        entities_by_kind: dict[str, dict[str, list[SpeculatedVariant]]] = {}
        mentions_by_kind: dict[str, dict[str, int]] = {}

        for entities in ctx.state.validated_entities_by_resource.values():
            for entity_name, entity in entities.items():
                entity_kind = entity.kind
                if entity_kind not in entities_by_kind:
                    entities_by_kind[entity_kind] = {}
                    mentions_by_kind[entity_kind] = {}
                # Extract variants with speculation tracking
                variants = extract_entity_variants(entity_name, entity.aliases())
                entities_by_kind[entity_kind][entity_name] = variants
                # Count mentions (accumulate across resources)
                current = mentions_by_kind[entity_kind].get(entity_name, 0)
                mentions_by_kind[entity_kind][entity_name] = current + len(
                    entity.mentions
                )

        return entities_by_kind, mentions_by_kind

    def _analyze_entity_similarity(self, canonicals: list[str]) -> str:
        """Analyze why entities with the same normalized form are not safe to merge.

        Returns a human-readable description of their relationship.
        """
        if len(canonicals) < 2:
            return "single entity"

        # Compare first two (most common case)
        a, b = canonicals[0], canonicals[1]
        norm_a = normalize_for_comparison(a)
        norm_b = normalize_for_comparison(b)

        # Check various types of differences
        if norm_a == norm_b:
            # Normalized forms match but originals differ (e.g., BRCA1 vs Brca1)
            # This was rejected by are_safe_capitalization_variants
            return "capitalization differs but unsafe (likely different base forms)"

        if is_obvious_variant(norm_a, norm_b):
            return "obvious variant but unsafe for auto-merge"

        # Check OSA distance for fuzzy similarity
        dist = osa_distance(norm_a, norm_b)
        if dist <= 2:
            return f"similar forms (edit distance {dist}) but not safe to merge"

        # Check for common patterns
        if a.rstrip("0123456789") == b.rstrip("0123456789"):
            return "same base with different numbers (e.g., SMAD2 vs SMAD3)"

        # Generic case
        return f"different base forms ({a} vs {b})"

    async def _get_consolidation_decisions_for_kind(
        self,
        agent_review_pairs: list[tuple[str, str]],
        kind: str,
        entities: dict[str, list[SpeculatedVariant]],
        ctx: GraphRunContext[State, Deps],
    ) -> tuple[dict[tuple[str, str], tuple[str, str]], set[str]]:
        """Query LLM for consolidation decisions, using cache to avoid redundant calls.

        Parameters:
            agent_review_pairs: List of (child_canonical, parent_canonical) pairs
            kind: Entity kind
            entities: Dict of canonical_name → list of SpeculatedVariants
            ctx: Graph run context

        Returns:
            Tuple of (rules, new_names) where:
            - rules maps (norm_child, kind) → (target, reasoning)
            - new_names is set of new canonical names (for renames)
        """
        if not agent_review_pairs:
            return {}, set()

        kind_cache = _get_merge_cache(ctx.state, kind)
        # Deduplicate pairs and check cache
        unique_pairs: dict[tuple[str, str], tuple[str, str]] = {}
        for child, parent in agent_review_pairs:
            cache_key = (child, parent)
            if cache_key not in unique_pairs:
                unique_pairs[cache_key] = (child, parent)
        # Separate cached and uncached
        uncached: list[tuple[str, str]] = []
        for cache_key, (child, parent) in unique_pairs.items():
            if cache_key not in kind_cache.cache:
                uncached.append((child, parent))
        # Get LLM decisions for uncached pairs
        if uncached:
            batch_size = ctx.deps.config.tools.extraction.merge_batch_size
            for i in range(0, len(uncached), batch_size):
                await self._process_consolidation_batch_new(
                    uncached[i : i + batch_size], kind, i // batch_size + 1, ctx
                )
        # Build rules from cache (now contains all pairs)
        return self._build_rules_from_cache(unique_pairs, kind, entities, ctx)

    def _build_rules_from_cache(
        self,
        pairs: dict[tuple[str, str], tuple[str, str]],
        kind: str,
        entities: dict[str, list[SpeculatedVariant]],
        ctx: GraphRunContext[State, Deps],
    ) -> tuple[dict[tuple[str, str], tuple[str, str]], set[str]]:
        """Build consolidation rules from cached decisions.

        Parameters:
            pairs: Map of (child, parent) → (child, parent)
            kind: Entity kind
            entities: Dict of canonical_name → variants
            ctx: Graph run context

        Returns:
            Tuple of (rules, new_names)
        """
        kind_cache = _get_merge_cache(ctx.state, kind)
        rules: dict[tuple[str, str], tuple[str, str]] = {}
        new_names: set[str] = set()
        for cache_key in pairs:
            if cache_key in kind_cache.cache:
                target, reasoning = kind_cache.cache[cache_key]
                kind_cache.hits += 1
                if target:  # merge or rename
                    child = cache_key[0]
                    child_norm = normalize_for_comparison(child)
                    rules[(child_norm, kind)] = (target, reasoning)
                    if target not in entities:
                        new_names.add(target)
                # else: skip - no rule
            else:
                kind_cache.misses += 1
        return rules, new_names

    async def _process_consolidation_batch_new(
        self,
        batch: list[tuple[str, str]],
        kind: str,
        batch_num: int,
        ctx: GraphRunContext[State, Deps],
    ) -> None:
        """Query LLM for entity pair consolidation decisions and cache results.

        Parameters:
            batch: List of (child_canonical, parent_canonical) pairs
            kind: Entity kind
            batch_num: Batch number for logging
            ctx: Graph run context (cache is populated in merge_cache_by_kind[kind])
        """
        # Build prompt structures
        pairs_description = []
        id_to_pair: dict[int, tuple[str, str, str]] = {}  # id → (parent, child, token)
        token_to_id: dict[str, int] = {}
        for pair_id, (child, parent) in enumerate(batch, start=1):
            if parent == child:
                continue
            token = _generate_token()
            id_to_pair[pair_id] = (parent, child, token)
            token_to_id[token] = pair_id
            pairs_description.append(f"{pair_id}. [{token}] {child!r} → {parent!r} ?")
        if not pairs_description:
            return
        # Build and execute prompt
        prompt = f"""**Research topic:** {ctx.state.topic}

**Target entity types:** {", ".join(ctx.state.target_entity_types)}

**Entity pairs to evaluate:**
{chr(10).join(pairs_description)}

Only return pairs that should merge or be renamed. Omit pairs that should remain separate."""
        kind_cache = _get_merge_cache(ctx.state, kind)
        agent = get_entity_consolidation_agent(ctx.deps.config)
        try:
            with rename_agent(agent, name=f"ConsolidateEntities ({kind}, {batch_num})"):
                async with ctx.deps.agent_semaphore:
                    result = await agent.run(prompt, deps=ctx.deps, usage=RunUsage())
            # Process decisions and populate cache
            returned_ids = {d.pair_id for d in result.output.decisions}
            for decision in result.output.decisions:
                resolved = _resolve_pair_from_decision(
                    decision.pair_id,
                    decision.confirm_token,
                    id_to_pair,
                    token_to_id,
                    ctx.deps.logger,
                )
                if resolved is None:
                    continue
                parent, child = resolved
                cache_key = (child, parent)
                # Cache decision: target for merge/rename
                target = decision.rename if decision.rename else parent
                kind_cache.cache[cache_key] = (target, decision.reasoning)
            # Cache implicit skips (pairs not returned by LLM)
            for pair_id, (parent, child, token) in id_to_pair.items():
                if pair_id not in returned_ids:
                    cache_key = (child, parent)
                    kind_cache.cache[cache_key] = (None, "implicit_skip")
        except (TimeoutError, ConnectionError, ValueError, ModelHTTPError) as e:
            ctx.deps.logger.error(f"Entity consolidation failed: {e}")

    async def _get_group_consolidation_decisions(
        self,
        groups: list[frozenset[str]],
        merge_trees: list,  # List of Cluster objects
        kind: str,
        entities: dict[str, list[SpeculatedVariant]],
        ctx: GraphRunContext[State, Deps],
    ) -> tuple[dict[tuple[str, str], tuple[str, str]], set[str], set[frozenset[str]]]:
        """Query LLM for group consolidation decisions using tree-based splitting.

        Uses hierarchical merge trees for surgical splits instead of re-clustering:
        1. LLM reviews clusters with tree structure context
        2. Split requests use tree.split_into_n() for precise subdivision
        3. Repeat until no splits or max rounds reached

        Args:
            groups: List of entity groups (each is a frozenset of entity names)
            merge_trees: Corresponding Cluster trees for each group
            kind: Entity kind
            entities: Dict of canonical_name → variants
            ctx: Graph run context

        Returns:
            Tuple of (rules, new_names, resolved_groups) where:
            - rules maps (norm_member, kind) → (target, reasoning)
            - new_names is set of new canonical names (for renames)
            - resolved_groups is set of groups that were fully resolved (merged or kept separate)
        """
        from interaction_finder.extraction.models import ClusterDecisions
        from interaction_finder.agent_config import agent_getter
        from interaction_finder.extraction.clustering import Cluster

        if not groups:
            return {}, set(), set()

        max_rounds = ctx.deps.config.tools.extraction.cluster_refinement_max_rounds
        all_rules: dict[tuple[str, str], tuple[str, str]] = {}
        all_new_names: set[str] = set()
        all_resolved: set[frozenset[str]] = (
            set()
        )  # Groups fully resolved (merged/kept separate)
        # Build map from frozenset → tree for splitting
        group_to_tree: dict[frozenset[str], Cluster] = {
            tree.entities: tree for tree in merge_trees
        }
        get_agent = agent_getter(
            "extraction",
            "entity_group_consolidation",
            ClusterDecisions,
            Deps,
            """You are an expert at consolidating biomedical entity names for systematic literature analysis.

You will be given groups of entities that appear related. For each group, decide if all members should merge to a single canonical entity, based on the research topic.

**Decision framework:**
1. Evaluate each group in the context of the research topic
2. Ask: Does the topic require distinguishing between these entities?
3. Apply these principles:
   - Entities differing only by clinical subtypes or modifiers → merge to parent term
     (e.g., "Idiopathic/Familial/Heritable [Disease]" → "[Disease]")
   - Entities representing the same underlying condition → merge
     (e.g., "tumour" and "tumor" → consistent spelling)
   - One entity clearly doesn't belong → exclude it (measurements mixed with diseases, etc.)
   - Entities representing genuinely distinct biological phenomena → keep separate
     (e.g., different diseases, different genes)

The research topic determines what distinctions matter. If entities are variants of the same concept relevant to the topic, merge them.

**Actions:**
- **merge**: All members represent the same entity → specify target (member number, name, or new name)
- **exclude**: One specific member doesn't belong → specify which one to remove (member number or name)
- **split**: Cluster mixes unrelated entities but can't identify which → system splits at weakest link

Only return groups that need action (merge/exclude/split). Omit groups that should remain separate.""",
        )
        agent = get_agent(ctx.deps.config)
        current_groups = list(groups)
        batch_size = ctx.deps.config.tools.extraction.merge_batch_size
        entity_types_str = ", ".join(ctx.state.target_entity_types)

        for round_num in range(1, max_rounds + 1):
            if not current_groups:
                break
            num_batches = -(-len(current_groups) // batch_size)  # ceil division
            batch_suffix = f" in {num_batches} batches" if num_batches > 1 else ""
            ctx.deps.logger.info(
                f"Entity group consolidation round {round_num}/{max_rounds}: "
                f"{len(current_groups)} groups{batch_suffix}"
            )
            # Process batches, collecting groups that need further review
            groups_needing_review: list[frozenset[str]] = []
            stats = {"merged": 0, "split": 0, "excluded": 0, "kept_separate": 0}
            for batch_start in range(0, len(current_groups), batch_size):
                batch = current_groups[batch_start : batch_start + batch_size]
                # Build prompt data for this batch
                group_data = [
                    {
                        "id": _generate_token(),
                        "entities": g,
                        "members": sorted(g, key=lambda e: (len(e), e)),
                    }
                    for g in batch
                ]
                groups_text = "\n\n".join(
                    f"## Group {g['id']}\nMembers:\n"
                    + "\n".join(f"  {i + 1}. {m}" for i, m in enumerate(g["members"]))
                    for g in group_data
                )
                prompt = f"""**Research topic:** {ctx.state.topic}
**Target entity types:** {entity_types_str}

**Entity groups to consolidate:**

{groups_text}

---

Omit groups that should stay separate. Action defaults to "merge" if omitted.

Examples:
- Merge: group_id="abc", target="1", reasoning="All are PAH subtypes"
- Exclude: group_id="def", action="exclude", target="5", reasoning="Member 5 (TAPSE) is a measurement"
- Split: group_id="ghi", action="split", reasoning="Mixes diseases and measurements"
"""
                async with ctx.deps.agent_semaphore:
                    result = await agent.run(prompt)
                # Index decisions by group_id
                id_to_group = {g["id"]: g for g in group_data}
                decisions_by_id: dict[str, list[ClusterDecision]] = {}
                for d in result.output.decisions:
                    if d.group_id not in id_to_group:
                        ctx.deps.logger.warning(f"Unknown group_id '{d.group_id}'")
                        continue
                    decisions_by_id.setdefault(d.group_id, []).append(d)
                # Process decisions
                groups_with_decisions: set[frozenset[str]] = set()
                for gid, decisions in decisions_by_id.items():
                    g = id_to_group[gid]
                    groups_with_decisions.add(g["entities"])
                    members = g["members"]
                    # Categorize decisions
                    excludes = [d for d in decisions if d.action == "exclude"]
                    merges = [d for d in decisions if d.action == "merge"]
                    splits = [d for d in decisions if d.action == "split"]
                    if merges and splits:
                        ctx.deps.logger.warning(f"Group {gid}: merge+split conflict")
                        continue
                    # Apply excludes first
                    remaining = set(members)
                    for exc in excludes:
                        if exc.target:
                            member = self._resolve_group_target(
                                exc.target, members, entities, ctx.deps.logger
                            )
                            if member in remaining:
                                remaining.discard(member)
                                stats["excluded"] += 1
                    if len(remaining) <= 1:
                        continue
                    remaining_entities = frozenset(remaining)
                    # Apply merge or split (resolve target against original members list)
                    if merges:
                        merge = merges[0]
                        if not merge.target:
                            continue
                        target = self._resolve_group_target(
                            merge.target, members, entities, ctx.deps.logger
                        )
                        # Check if target was an existing member that got excluded
                        if target in members and target not in remaining:
                            ctx.deps.logger.warning(
                                f"Group {gid}: merge target '{target}' was excluded"
                            )
                            continue
                        if target not in entities:
                            all_new_names.add(target)
                        for member in remaining:
                            if member != target:
                                all_rules[(normalize_for_comparison(member), kind)] = (
                                    target,
                                    merge.reasoning or f"Group {gid}",
                                )
                        stats["merged"] += 1
                    elif splits:
                        tree = group_to_tree.get(g["entities"]) or group_to_tree.get(
                            remaining_entities
                        )
                        if tree and len(remaining) < len(g["entities"]):
                            tree = tree.find_subtree(remaining_entities) or tree
                        if not tree:
                            ctx.deps.logger.warning(f"No tree for group {gid}")
                            continue
                        for sub in tree.split_into_n(2):
                            if len(sub) > 1:
                                groups_needing_review.append(sub)
                                if sub not in group_to_tree and (
                                    st := tree.find_subtree(sub)
                                ):
                                    group_to_tree[sub] = st
                        stats["split"] += 1
                    elif excludes:
                        # Only excludes: queue remainder for review
                        groups_needing_review.append(remaining_entities)
                        if (tree := group_to_tree.get(g["entities"])) and (
                            st := tree.find_subtree(remaining_entities)
                        ):
                            group_to_tree[remaining_entities] = st
                # Groups without decisions are kept separate - mark as resolved
                kept_separate = set(batch) - groups_with_decisions
                all_resolved.update(kept_separate)
                stats["kept_separate"] += len(kept_separate)
            ctx.deps.logger.info(
                f"  Results: {stats['merged']} merged, {stats['split']} split, "
                f"{stats['excluded']} excluded, {stats['kept_separate']} separate"
            )
            if not groups_needing_review:
                break
            current_groups = groups_needing_review

        ctx.deps.logger.info(
            f"Entity group consolidation complete: {len(all_rules)} merge rules created, "
            f"{len(all_new_names)} new canonical names"
        )
        return all_rules, all_new_names, all_resolved

    def _resolve_group_target(
        self,
        target_spec: str,
        members: list[str],
        entities: dict[str, list[SpeculatedVariant]],
        logger: logging.Logger,
    ) -> str:
        """Resolve target from group decision.

        Handles multiple formats that LLMs may return:
        - Pure digit: "1", "2" (member number)
        - Number + name: "2. Entity Name", "2) Entity Name" (common LLM pattern)
        - Exact name: "Entity Name"
        - Fuzzy name: variant spelling that matches a member
        - New name: creates a new canonical

        Args:
            target_spec: Member number, member name, or new canonical name.
                Also handles combined formats like "2. Name" or "2) Name".
            members: Group members in order
            entities: All entities (for fuzzy matching)
            logger: Logger for warnings

        Returns:
            Resolved target entity name
        """
        members_set = set(members)
        # Case 1: Pure digit like "1", "2", etc.
        if target_spec.isdigit():
            idx = int(target_spec) - 1
            if 0 <= idx < len(members):
                return members[idx]
        # Case 2: Number + name format ("2. Name", "2) Name", "2 Name")
        # Pattern: optional whitespace, digits, optional ).  delimiter, required
        # space, then word-starting name
        match = re.match(r"^\s*(\d+)[).]?\s+(\w.+?)?\s*$", target_spec)
        if match:
            num_str, name_part = match.groups()
            idx = int(num_str) - 1
            if 0 <= idx < len(members):
                member = members[idx]
                # Validate name part matches the member using entity matching
                if name_part:
                    try:
                        name_match = find_entity_match(name_part, entities)
                    except (IndexError, KeyError):
                        name_match = None  # Graceful fallback for malformed entities
                    if name_match is None or name_match.canonical != member:
                        resolved_to = name_match.canonical if name_match else "nothing"
                        logger.warning(
                            f"Merge target mismatch: member {idx + 1} is '{member}' "
                            f"but specified name '{name_part}' resolves to "
                            f"'{resolved_to}'; using member number"
                        )
                return member
        # Case 3: Exact member name (fast path, no entity matching needed)
        if target_spec in members_set:
            return target_spec
        # Case 4: Name that fuzzy-matches a member
        try:
            entity_match = find_entity_match(target_spec, entities)
        except (IndexError, KeyError):
            entity_match = None  # Graceful fallback for malformed entities
        if entity_match and entity_match.canonical in members_set:
            return entity_match.canonical
        # Case 5: New canonical name
        return target_spec

    def _resolve_transitive_merges(
        self, merge_rules: dict[tuple[str, str], tuple[str, str]]
    ) -> dict[tuple[str, str], tuple[str, str]]:
        """Resolve transitive merge chains.

        If A→B and B→C, resolve to A→C (and B→C).
        This ensures all entities in a chain ultimately point to the final parent.

        Rules are keyed by (normalized_form, kind). Values are (target, reasoning).
        When following chains, we normalize targets to check for further rules.
        The final reasoning is taken from the last link in the chain.

        Returns:
            Resolved merge rules with transitive chains collapsed
        """
        resolved = {}
        for (child_norm, kind), (target, reasoning) in merge_rules.items():
            # Follow the chain: child → target → target's target → ...
            final_target = target
            final_reasoning = reasoning
            visited = {child_norm}  # Prevent infinite loops
            # Normalize target to check if it's also a key in the rules
            while True:
                target_norm = normalize_for_comparison(final_target)
                if (target_norm, kind) not in merge_rules:
                    break
                if target_norm in visited:
                    # Cycle detected - stop here
                    break
                visited.add(target_norm)
                final_target, final_reasoning = merge_rules[(target_norm, kind)]
            resolved[(child_norm, kind)] = (final_target, final_reasoning)
        return resolved

    def _apply_merge_rules_globally(
        self,
        rules: dict[tuple[str, str], tuple[str, str]],
        ctx: GraphRunContext[State, Deps],
    ) -> None:
        """Apply merge/rename rules to all documents.

        Rules map (norm_child, kind) → (target, reasoning). Keys are normalized forms
        for cross-document consistency. Target may be a normalized form (for merge)
        or a canonical name (for rename). Rules should be transitively resolved
        before calling.

        Stores rules directly in consolidated.entities.merges for provenance.
        """
        if not rules:
            return
        # Store rules directly in consolidated structure, categorized by type and kind
        for (norm_name, kind), (target, reasoning) in rules.items():
            rule = EntityMergeRule(source=norm_name, target=target, reasoning=reasoning)
            kind_merges = ctx.state.consolidated.entities.merges.setdefault(
                kind, EntityKindMerges()
            )
            if reasoning.startswith("auto:"):
                kind_merges.automatic.append(rule)
            else:
                kind_merges.llm_decided.append(rule)
        target_by_norm_and_kind: dict[tuple[str, str], str] = {}
        for (_norm, kind), (target, _) in rules.items():
            target_key = (normalize_for_comparison(target), kind)
            target_by_norm_and_kind.setdefault(target_key, target)
        for resource_id, entities in ctx.state.validated_entities_by_resource.items():
            # Build new EntityRefs without mutating underlying mentions
            from interaction_finder.extraction.models import EntityMention

            grouped: dict[str, list[EntityMention]] = defaultdict(list)

            for child_name, ref in entities.items():
                target_name = None
                all_forms = extract_all_forms(child_name, ref.aliases())
                for form in all_forms:
                    key = (normalize_for_comparison(form), ref.kind)
                    if key in rules:
                        target_name = rules[key][0]
                        break
                canonical_target = target_by_norm_and_kind.get(
                    (normalize_for_comparison(child_name), ref.kind)
                )
                canonical = target_name or canonical_target or child_name
                grouped[canonical].extend(ref.mentions)

            new_entities: dict[str, EntityRef] = {
                canonical: EntityRef(canonical=canonical, mentions=mentions)
                for canonical, mentions in grouped.items()
            }

            # Count actual merges within the resource (renames without consolidation
            # should not contribute to the merge tally).
            merges_for_resource = max(0, len(entities) - len(new_entities))
            ctx.state.entities_merged += merges_for_resource
            ctx.state.validated_entities_by_resource[resource_id] = new_entities

    def _update_pair_entity_references(
        self,
        merge_rules: dict[tuple[str, str], tuple[str, str]],
        ctx: GraphRunContext[State, Deps],
    ) -> None:
        """Update EntityRef references in PairAssessments after merging.

        After entities are merged, pair assessments may reference old entity names.
        This method replaces the EntityRef objects with the consolidated versions from
        validated_entities_by_resource, which have the updated canonical names and
        aggregated mentions from all merged entities.

        Rules are keyed by (normalized_form, kind). Values are (target, reasoning).

        Parameters:
            merge_rules: Mapping of (norm_child, kind) → (target, reasoning)
            ctx: Graph run context with state containing assessments
        """
        if not merge_rules:
            return
        for resource_id, assessments in ctx.state.pair_assessments_by_resource.items():
            # Get consolidated entities for this resource
            consolidated_entities = ctx.state.validated_entities_by_resource.get(
                resource_id, {}
            )
            for assessment in assessments:
                # Update entity1: check if it was merged, then look up consolidated version
                e1_key = (
                    normalize_for_comparison(assessment.entity1.canonical),
                    assessment.entity1.kind,
                )
                if e1_key in merge_rules:
                    # Entity was merged - get target name and look up consolidated EntityRef
                    merged_name = merge_rules[e1_key][0]
                    if merged_name in consolidated_entities:
                        # Use consolidated entity with all merged mentions
                        assessment.entity1 = consolidated_entities[merged_name]
                    else:
                        # Fallback: update canonical but preserve existing mentions
                        # (happens when validated_entities_by_resource not yet updated)
                        assessment.entity1 = EntityRef(
                            canonical=merged_name,
                            mentions=assessment.entity1.mentions,
                        )
                # Update entity2: check if it was merged, then look up consolidated version
                e2_key = (
                    normalize_for_comparison(assessment.entity2.canonical),
                    assessment.entity2.kind,
                )
                if e2_key in merge_rules:
                    # Entity was merged - get target name and look up consolidated EntityRef
                    merged_name = merge_rules[e2_key][0]
                    if merged_name in consolidated_entities:
                        # Use consolidated entity with all merged mentions
                        assessment.entity2 = consolidated_entities[merged_name]
                    else:
                        # Fallback: update canonical but preserve existing mentions
                        # (happens when validated_entities_by_resource not yet updated)
                        assessment.entity2 = EntityRef(
                            canonical=merged_name,
                            mentions=assessment.entity2.mentions,
                        )

    def _build_global_entity_index(self, ctx: GraphRunContext[State, Deps]) -> None:
        """Aggregate mentions across all resources into global entity index."""
        from collections import defaultdict

        mentions_by_canonical: dict[str, list[EntityMention]] = defaultdict(list)
        for entities in ctx.state.validated_entities_by_resource.values():
            for canonical, ref in entities.items():
                mentions_by_canonical[canonical].extend(ref.mentions)
        ctx.state.global_entities = {
            c: EntityRef(canonical=c, mentions=m)
            for c, m in mentions_by_canonical.items()
        }


@dataclass
class ConsolidateRelationshipsNode(BaseNode[State, Deps, ExtractionResult]):
    """Consolidate relationship labels and classify polarities globally.

    Unified algorithm:
    1. Collect all unique relationship labels from all assessments
    2. Query LLM for consolidation + polarity classification (single call)
    3. Apply consolidations to all assessments
    4. Store polarity mappings in state
    5. Filter irrelevant relationship types (optional, on by default)

    This approach ensures:
    - Vocabulary consolidation (merges synonyms like "linked_to" → "associated_with")
    - Polarity classification (positive/negative/neutral/irrelevant)
    - Biological direction-aware normalization (context-aware decisions)
    - Consistency (same canonical label and polarity across all documents)
    """

    async def run(self, ctx: GraphRunContext[State, Deps]) -> "SweepCoMentionsNode":
        """Consolidate relationship labels, classify polarities, and filter irrelevant."""
        with logfire.span("ConsolidateRelationshipsNode"):
            if ctx.deps.progress:
                ctx.deps.progress.set_status("Consolidating relationships")
            # Step 1: Collect unique relationship labels
            unique_relationships = self._collect_unique_relationships(ctx)

            if not unique_relationships:
                ctx.deps.logger.info("No relationships to consolidate")
                return SweepCoMentionsNode()

            # Step 2: Get consolidation + polarity from LLM (unified)
            consolidations = await self._consolidate_and_classify(
                unique_relationships, ctx
            )

            if not consolidations:
                ctx.deps.logger.warning(
                    "Relationship consolidation agent returned no data; "
                    "defaulting all relationship polarities to neutral"
                )
                self._ensure_polarities_for_all_relationships(unique_relationships, ctx)
                return SweepCoMentionsNode()

            # Step 3: Apply consolidations to assessments
            self._apply_consolidations(consolidations, ctx)

            # Step 4: Store polarity mappings
            self._store_polarity_mappings(consolidations, ctx)
            self._ensure_polarities_for_all_relationships(
                unique_relationships, ctx, log_missing=True
            )

            # Step 4b: Store relationship consolidations in unified structure
            if consolidations:
                ctx.state.consolidated.relationships.extend(consolidations)

            ctx.deps.logger.info(
                f"Relationship consolidation: {len(consolidations)} relationships processed, "
                f"{ctx.state.relationships_merged} assessments updated"
            )

            # Step 5: Filter irrelevant relationship types (if enabled)
            if ctx.deps.config.tools.extraction.filter_irrelevant_relationships:
                self._filter_irrelevant_assessments(ctx)

            # Save checkpoint after relationship consolidation
            await _save_partial_checkpoint(ctx, "consolidate_relationships")

            return SweepCoMentionsNode()

    def _collect_unique_relationships(
        self, ctx: GraphRunContext[State, Deps]
    ) -> set[str]:
        """Collect all unique relationship labels from assessments.

        Returns:
            Set of unique relationship labels (unnormalized)
        """
        relationships = set()

        for assessments in ctx.state.pair_assessments_by_resource.values():
            for assessment in assessments:
                relationships.add(assessment.relationship)

        return relationships

    async def _consolidate_and_classify(
        self, relationships: set[str], ctx: GraphRunContext[State, Deps]
    ) -> list[RelationshipConsolidation]:
        """Query LLM for consolidation + polarity classification (unified).

        Returns:
            List of RelationshipConsolidation objects
        """
        from interaction_finder.extraction.consolidate_relationships import (
            get_relationship_consolidation_agent,
        )

        # Build prompt
        relationships_list = sorted(relationships)
        relationships_str = "\n".join(f"- {r!r}" for r in relationships_list)
        entity_types_str = ", ".join(ctx.state.target_entity_types)

        prompt = f"""**Research topic:** {ctx.state.topic}

**Target entity types:** {entity_types_str}

**Relationship labels found:**
{relationships_str}

For each relationship, provide:
1. Consolidated canonical form (may equal original)
2. Polarity classification relative to this research topic"""

        # Call unified agent
        agent = get_relationship_consolidation_agent(ctx.deps.config)
        usage = RunUsage()
        try:
            with rename_agent(agent, name="ConsolidateRelationshipsNode"):
                async with ctx.deps.agent_semaphore:
                    result = await agent.run(prompt, deps=ctx.deps, usage=usage)
            return result.output.consolidations
        except (TimeoutError, ConnectionError, ValueError, ModelHTTPError) as e:
            ctx.deps.logger.error(
                f"Relationship consolidation failed: {type(e).__name__}: {e}"
            )
            return []

    def _apply_consolidations(
        self,
        consolidations: list[RelationshipConsolidation],
        ctx: GraphRunContext[State, Deps],
    ) -> None:
        """Apply consolidations to all assessments.

        Updates assessment.relationship in place when consolidation changes the label.
        """
        # Build mapping: normalized_original → consolidated_label
        consolidation_map: dict[str, str] = {}
        for cons in consolidations:
            norm_orig = normalize_for_comparison(cons.original)
            # If consolidation changed the label, store mapping
            if cons.original != cons.consolidated:
                consolidation_map[norm_orig] = cons.consolidated

        if not consolidation_map:
            return

        # Apply consolidations to all assessments
        for assessments in ctx.state.pair_assessments_by_resource.values():
            for assessment in assessments:
                norm = normalize_for_comparison(assessment.relationship)
                if norm in consolidation_map:
                    assessment.relationship = consolidation_map[norm]
                    ctx.state.relationships_merged += 1

        # Store in state for provenance
        ctx.state.relationship_mappings = consolidation_map

    def _store_polarity_mappings(
        self,
        consolidations: list[RelationshipConsolidation],
        ctx: GraphRunContext[State, Deps],
    ) -> None:
        """Store polarity mappings from consolidations.

        Creates mapping: relationship_label → polarity
        Stores for BOTH original and consolidated labels to handle cases where
        relationships weren't merged.
        """
        for cons in consolidations:
            # Store polarity for both original and consolidated labels
            # This ensures lookup works even if consolidation wasn't applied
            ctx.state.relationship_polarities[cons.original] = cons.polarity
            ctx.state.relationship_polarities[cons.consolidated] = cons.polarity

    def _ensure_polarities_for_all_relationships(
        self,
        relationships: set[str],
        ctx: GraphRunContext[State, Deps],
        log_missing: bool = False,
    ) -> None:
        """Ensure every relationship label has a polarity mapping.

        Used both for total consolidation failures (all defaults) and partial
        responses where the LLM omitted some labels. Missing entries default to
        "neutral" so downstream spread grouping never raises KeyError.
        """
        missing = [
            rel for rel in relationships if rel not in ctx.state.relationship_polarities
        ]
        if not missing:
            return

        for rel in missing:
            ctx.state.relationship_polarities[rel] = "neutral"

        if log_missing:
            ctx.deps.logger.warning(
                f"Assigned neutral polarity to {len(missing)} "
                "relationships missing classification"
            )

    def _filter_irrelevant_assessments(self, ctx: GraphRunContext[State, Deps]) -> None:
        """Remove assessments with irrelevant polarity.

        Uses the stored polarity mappings to identify and remove irrelevant assessments.
        """
        # Group assessments by pair
        grouped: dict[EntityPairKey, list[PairAssessment]] = defaultdict(list)
        for assessments in ctx.state.pair_assessments_by_resource.values():
            for assessment in assessments:
                pair_key = make_entity_pair_key(assessment.entity1, assessment.entity2)
                grouped[pair_key].append(assessment)

        # Create pre-rejected judgments for pairs that are ALL irrelevant
        filtered_pairs: set[EntityPairKey] = set()
        for pair_key, assessments in grouped.items():
            # Check if all assessments are irrelevant
            all_irrelevant = all(
                ctx.state.relationship_polarities.get(a.relationship) == "irrelevant"
                for a in assessments
            )

            if all_irrelevant:
                # Create rejected judgment with empty spread (or all in irrelevant)
                spread = build_pair_spread(
                    assessments, ctx.state.relationship_polarities
                )
                first = assessments[0]
                ctx.state.pair_judgments[pair_key] = PairJudgment(
                    entity1=SimpleEntity(
                        name=first.entity1.canonical,
                        kind=first.entity1.kind,
                        aliases=_get_entity_aliases(first.entity1.canonical, ctx.state),
                    ),
                    entity2=SimpleEntity(
                        name=first.entity2.canonical,
                        kind=first.entity2.kind,
                        aliases=_get_entity_aliases(first.entity2.canonical, ctx.state),
                    ),
                    relationship=first.relationship,
                    spread=spread,
                    accepted=False,
                    evidence=_aggregate_evidence(assessments),
                    decision_confidence=0.95,  # High confidence in rejection decision
                    reasoning=(
                        "All relationship types for this pair were classified as "
                        "irrelevant to the research question"
                    ),
                )
                filtered_pairs.add(pair_key)
        if filtered_pairs:
            self._remove_assessments_for_pairs(filtered_pairs, ctx)
            ctx.deps.logger.info(
                f"Filtered {len(filtered_pairs)} pairs with only irrelevant relationships"
            )

    def _remove_assessments_for_pairs(
        self,
        pair_keys: set[EntityPairKey],
        ctx: GraphRunContext[State, Deps],
    ) -> None:
        """Remove assessments for the provided pair keys from state."""

        for resource_id, assessments in list(
            ctx.state.pair_assessments_by_resource.items()
        ):
            filtered = [
                assessment
                for assessment in assessments
                if make_entity_pair_key(assessment.entity1, assessment.entity2)
                not in pair_keys
            ]
            ctx.state.pair_assessments_by_resource[resource_id] = filtered


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

    def _build_opposition_map(
        self, ctx: GraphRunContext[State, Deps]
    ) -> dict[str, set[str]]:
        """Build opposition map from consolidated relationships."""
        return opposition_map_from_consolidations(ctx.state.consolidated.relationships)

    async def run(self, ctx: GraphRunContext[State, Deps]) -> "FinalizeNode":
        """Judge all unique pairs across documents."""
        with logfire.span("JudgeCrossDocumentNode"):
            # Set phase
            if ctx.deps.progress:
                ctx.deps.progress["Unique pairs"].activate()
                ctx.deps.progress.set_status("Cross-document validation")
            # Build opposition map once from consolidated relationships
            opposition_map = self._build_opposition_map(ctx)
            # Group assessments by entity pair
            assessments_by_pair: dict[EntityPairKey, list[PairAssessment]] = (
                defaultdict(list)
            )

            for assessments_list in ctx.state.pair_assessments_by_resource.values():
                for assessment in assessments_list:
                    pair_key = make_entity_pair_key(
                        assessment.entity1, assessment.entity2
                    )
                    if pair_key in ctx.state.pair_judgments:
                        continue
                    assessments_by_pair[pair_key].append(assessment)

            # Update unique pairs count
            if ctx.deps.progress:
                ctx.deps.progress["Unique pairs"].total = len(assessments_by_pair)

            # Judge each pair (counter updates happen inside _judge_pair for tight scoping)
            tasks = []
            for pair_key, assessments in assessments_by_pair.items():
                tasks.append(
                    self._judge_pair(pair_key, assessments, ctx, opposition_map)
                )

            # Run all judgments in parallel
            if tasks:
                for coro in asyncio.as_completed(tasks):
                    pair_key, judgment = await coro
                    ctx.state.pair_judgments[pair_key] = judgment
            # Mark judgment phase complete
            if ctx.deps.progress:
                ctx.deps.progress["Unique pairs"].complete()
            return FinalizeNode()

    def _can_accept_deterministically(
        self, assessments: list[PairAssessment], opposition_map: dict[str, set[str]]
    ) -> tuple[bool, str, str]:
        """Check if we can accept without LLM call.

        Accepts if there are multiple strong-evidence assessments (overall >= 7)
        with no opposing relationships among them.

        Returns:
            (can_accept, relationship, reasoning) or (False, "", "")
        """
        # Need multiple strong-evidence assessments (level 7 = ~80% probability)
        high_conf = [a for a in assessments if a.evidence.overall >= 7]

        if len(high_conf) >= 2:
            # Check if all have same relationship (fast path)
            relationships = {a.relationship for a in high_conf}
            if len(relationships) == 1:
                relationship = high_conf[0].relationship
                reasoning = self._build_consensus_reasoning(high_conf, relationship)
                return (True, relationship, reasoning)

            # Check for opposing relationships (if we have opposition data)
            has_opposition = False
            if opposition_map:
                from itertools import combinations

                has_opposition = any(
                    are_relationships_opposed(rel1, rel2, opposition_map)
                    for rel1, rel2 in combinations(relationships, 2)
                )

            if not has_opposition:
                # Multiple high-conf, different but compatible relationships
                # Pick most common
                from collections import Counter

                most_common_rel = Counter(
                    a.relationship for a in high_conf
                ).most_common(1)[0][0]
                reasoning = self._build_consensus_reasoning(
                    high_conf, most_common_rel, multiple_relationships=True
                )
                return (True, most_common_rel, reasoning)

        return (False, "", "")

    def _build_consensus_reasoning(
        self,
        high_conf_assessments: list[PairAssessment],
        selected_relationship: str,
        multiple_relationships: bool = False,
    ) -> str:
        """Build detailed reasoning for deterministic consensus cases.

        Generates prose that cites documents and explains the consensus,
        providing better provenance than generic templates.

        Parameters:
            high_conf_assessments: High-confidence assessments (≥2)
            selected_relationship: The relationship to report
            multiple_relationships: Whether assessments have compatible but different relationships

        Returns:
            Detailed reasoning text with document citations
        """
        # Group assessments by relationship, tracking unique documents per relationship
        from collections import defaultdict

        by_rel: dict[str, set[str]] = defaultdict(set)
        for a in high_conf_assessments:
            by_rel[a.relationship].add(a.resource_id.id)
        # Build structured reasoning with header and grouped citations
        if multiple_relationships:
            lines = [
                f"High-confidence consensus across {len(high_conf_assessments)} sources "
                f"with compatible relationships.\n"
            ]
            # Find documents that appear in multiple relationship groups
            all_docs = set()
            for docs in by_rel.values():
                all_docs.update(docs)
            doc_rels: dict[str, set[str]] = defaultdict(set)
            for rel, docs in by_rel.items():
                for doc in docs:
                    doc_rels[doc].add(rel)
            # Group documents by their relationship set
            rel_groups: dict[frozenset[str], list[str]] = defaultdict(list)
            for doc, rels in doc_rels.items():
                rel_groups[frozenset(rels)].append(doc)
            # Build citation lines for each relationship group
            for rel_set in sorted(
                rel_groups.keys(), key=lambda rs: (-len(rs), sorted(rs))
            ):
                docs = sorted(rel_groups[rel_set])
                doc_citations = " ".join(f"[{doc}]" for doc in docs)
                if len(rel_set) == 1:
                    rel_name = next(iter(rel_set))
                    lines.append(
                        f"Found {len(docs)} document{'s' if len(docs) != 1 else ''} "
                        f"that support{'s' if len(docs) == 1 else ''} the relationship '{rel_name}': {doc_citations}"
                    )
                else:
                    rel_list_local = ", ".join(f"'{r}'" for r in sorted(rel_set))
                    lines.append(
                        f"Found {len(docs)} document{'s' if len(docs) != 1 else ''} "
                        f"that support{'s' if len(docs) == 1 else ''} the relationships {rel_list_local}: {doc_citations}"
                    )
            return "\n".join(lines) + "."
        else:
            # Single relationship case - simpler format
            doc_ids = sorted(by_rel[selected_relationship])
            doc_citations = " ".join(f"[{doc}]" for doc in doc_ids)
            return (
                f"High-confidence consensus for '{selected_relationship}' across "
                f"{len(doc_ids)} source{'s' if len(doc_ids) != 1 else ''}: {doc_citations}."
            )

    def _should_investigate(
        self, assessments: list[PairAssessment], opposition_map: dict[str, set[str]]
    ) -> bool:
        """Determine if we need LLM investigation.

        Always investigate except for the deterministic accept case.
        """
        # Check for deterministic accept
        can_accept, _, _ = self._can_accept_deterministically(
            assessments, opposition_map
        )
        if can_accept:
            return False

        # All other cases need investigation
        return True

    def _sort_assessments_by_date(
        self,
        assessments: list[PairAssessment],
        ctx: GraphRunContext[State, Deps],
    ) -> list[PairAssessment]:
        """Sort assessments by publication date (newest first), then quote count.

        Ensures documents are presented to the LLM in the same order they'll appear
        in the report, making citations consistent with visual display order.
        """

        def sort_key(a: PairAssessment) -> tuple:
            resource = ctx.deps.resource_pool.get(a.resource_id)
            date = resource.publication_date if resource else ""
            # Empty dates sort last; newer dates first; more quotes first
            return (date or "", -len(a.quotes))

        return sorted(assessments, key=sort_key, reverse=True)

    def _build_contentious_prompt(
        self,
        pair_key: EntityPairKey,
        spread: PairSpread,
        ctx: GraphRunContext[State, Deps],
    ) -> str:
        """Build prompt for contentious pairs (supporting + refuting evidence)."""
        padding = getattr(ctx.deps.config.tools.extraction, "region_padding_chunks", 1)

        def format_assessments(assessments: list[PairAssessment], label: str) -> str:
            """Format a list of assessments with document evidence."""
            if not assessments:
                return ""
            # Sort by publication date (newest first) for consistent display order
            sorted_assessments = self._sort_assessments_by_date(assessments, ctx)
            sections = [f"## {label.title()} Evidence\n"]
            for assessment in sorted_assessments:
                resource = ctx.deps.resource_pool.get(assessment.resource_id)
                if not resource:
                    continue
                text = collect_relevant_text_for_quotes(
                    resource, assessment.quotes, padding
                )
                adjusted_text = adjust_heading_levels(text, target_min_level=4)
                doc_id = assessment.resource_id.id
                ev = assessment.evidence
                section = f"""### Document extract [{doc_id}]: {resource.title}
**Relationship:** {assessment.relationship} | **Evidence level:** {ev.overall}/9
**Factors:** {ev.directness}, {ev.source_type}, {ev.specificity}, {ev.language}
**Reasoning:** {assessment.reasoning}

{adjusted_text}"""
                sections.append(section)
            return "\n\n".join(sections)

        positive_text = format_assessments(spread.positive, "Positive")
        negative_text = format_assessments(spread.negative, "Negative")
        neutral_text = format_assessments(spread.neutral, "Neutral")

        # Get all unique relationships
        all_rels = {
            a.relationship for a in spread.positive + spread.negative + spread.neutral
        }
        relationships_str = ", ".join(f'"{r}"' for r in sorted(all_rels))

        return f"""# Context
Synthesize contradictory evidence for an entity association.

**Topic:** {ctx.state.topic}

**Pair:** {pair_key.entity1_name} <-> {pair_key.entity2_name}

**Relationship types found:** {relationships_str}

**Note:** This is a CONTENTIOUS pair with contradictory evidence. This may indicate either:
- Conflicting biological polarities (positive effects vs negative effects)
- Opposing relationships (e.g., "activates" vs "inhibits")
- Context-dependent effects that appear contradictory

# Document Extracts

{positive_text}

{negative_text}

{neutral_text if neutral_text else ""}

# Task
Synthesize the contradictory evidence, considering:
1. Is there genuine disagreement in the literature, or do studies examine different contexts?
2. What is the weight of evidence on each side?
3. Should we accept this pair despite contradictions?

Provide: accepted (true/false), relationship (selected from above), confidence (high/medium/low), and detailed reasoning explaining how you weighed the contradictions. Cite documents using their IDs in square brackets (e.g., [1_abc12345]) when referencing specific evidence."""

    def _build_unidirectional_prompt(
        self,
        pair_key: EntityPairKey,
        spread: PairSpread,
        ctx: GraphRunContext[State, Deps],
    ) -> str:
        """Build prompt for unidirectional pairs (no positive+negative conflict)."""
        padding = getattr(ctx.deps.config.tools.extraction, "region_padding_chunks", 1)

        # Combine all assessments (one polarity category will dominate)
        all_assessments = spread.positive + spread.negative + spread.neutral
        # Sort by publication date (newest first) for consistent display order
        sorted_assessments = self._sort_assessments_by_date(all_assessments, ctx)
        document_sections = []
        for assessment in sorted_assessments:
            resource = ctx.deps.resource_pool.get(assessment.resource_id)
            if not resource:
                continue
            text = collect_relevant_text_for_quotes(
                resource, assessment.quotes, padding
            )
            adjusted_text = adjust_heading_levels(text, target_min_level=3)
            doc_id = assessment.resource_id.id
            ev = assessment.evidence
            section = f"""## Document extract [{doc_id}]: {resource.title}
**Assessment:** level {ev.overall}/9 - {assessment.relationship}
**Factors:** {ev.directness}, {ev.source_type}, {ev.specificity}, {ev.language}
**Reasoning:** {assessment.reasoning}

{adjusted_text}"""
            document_sections.append(section)

        relationships = {a.relationship for a in all_assessments}
        relationships_str = ", ".join(f'"{r}"' for r in sorted(relationships))

        return f"""# Context
Make a final judgment on an entity association.

**Topic:** {ctx.state.topic}

**Pair:** {pair_key.entity1_name} <-> {pair_key.entity2_name}

**Relationship types found across documents:** {relationships_str}

# Document Extracts

{chr(10).join(document_sections)}

# Task
Synthesize the evidence across documents, considering consistency, quality, and contradictions.
Select the most accurate relationship overall (from the ones found above).

Provide: accepted (true/false), relationship (selected label), synthesized evidence quality factors, and detailed reasoning. Cite documents using their IDs in square brackets (e.g., [1_abc12345]) when referencing specific evidence."""

    def _is_contentious_pair(
        self,
        assessments: list[PairAssessment],
        spread: PairSpread,
        opposition_map: dict[str, set[str]],
    ) -> bool:
        """Detect if a pair has contradictory evidence requiring special handling.

        A pair is contentious if either:
        1. It has both positive AND negative polarity assessments
        2. It has opposing relationships (e.g., "activates" vs "inhibits")
        """
        # Check 1: Contradictory polarity (positive + negative)
        if spread.positive and spread.negative:
            return True
        # Check 2: Opposing relationships
        if not opposition_map:
            return False
        from itertools import combinations

        relationships = [a.relationship for a in assessments]
        return any(
            are_relationships_opposed(rel1, rel2, opposition_map)
            for rel1, rel2 in combinations(relationships, 2)
        )

    async def _judge_pair(
        self,
        pair_key: EntityPairKey,
        assessments: list[PairAssessment],
        ctx: GraphRunContext[State, Deps],
        opposition_map: dict[str, set[str]],
    ) -> tuple[EntityPairKey, PairJudgment]:
        """Make final judgment on a single pair."""
        try:
            # Build PairSpread by polarity
            spread = build_pair_spread(assessments, ctx.state.relationship_polarities)

            # Try deterministic accept
            can_accept, relationship, reasoning = self._can_accept_deterministically(
                assessments, opposition_map
            )
            if can_accept:
                # Create judgment without LLM call (no semaphore needed)
                # Mark as in-progress for consistent counter behavior
                if ctx.deps.progress:
                    ctx.deps.progress["Unique pairs"].work()
                first_assessment = assessments[0]
                # Aggregate evidence from strong assessments
                strong = [a for a in assessments if a.evidence.overall >= 7]
                evidence = _aggregate_evidence(strong)
                # High decision confidence for deterministic consensus
                decision_confidence = 0.95 if len(strong) >= 3 else 0.85
                judgment = PairJudgment(
                    entity1=SimpleEntity(
                        name=first_assessment.entity1.canonical,
                        kind=first_assessment.entity1.kind,
                        aliases=_get_entity_aliases(
                            first_assessment.entity1.canonical, ctx.state
                        ),
                    ),
                    entity2=SimpleEntity(
                        name=first_assessment.entity2.canonical,
                        kind=first_assessment.entity2.kind,
                        aliases=_get_entity_aliases(
                            first_assessment.entity2.canonical, ctx.state
                        ),
                    ),
                    relationship=relationship,
                    spread=spread,
                    accepted=True,
                    evidence=evidence,
                    decision_confidence=decision_confidence,
                    reasoning=reasoning,
                )
                return (pair_key, judgment)
            # Need LLM investigation
            # Collect valid document IDs from all assessments for citation validation
            valid_doc_ids = {a.resource_id.id for a in assessments}
            # Detect contentious pairs (positive + negative polarity OR opposing relationships)
            is_contentious = self._is_contentious_pair(
                assessments, spread, opposition_map
            )

            if is_contentious:
                prompt = self._build_contentious_prompt(pair_key, spread, ctx)
            else:
                prompt = self._build_unidirectional_prompt(pair_key, spread, ctx)
            # Call cross-document judge with renamed span
            agent = get_cross_document_judge_agent(ctx.deps.config)
            usage = RunUsage()
            try:
                with rename_agent(
                    agent,
                    name=f"JudgeCrossDocumentNode: {pair_key.entity1_name} ⇌ {pair_key.entity2_name}",
                ):
                    async with ctx.deps.agent_semaphore:
                        # Mark pair as in-progress now that we've acquired the semaphore
                        if ctx.deps.progress:
                            ctx.deps.progress["Unique pairs"].work()
                        result = await agent.run(prompt, deps=ctx.deps, usage=usage)
            except (TimeoutError, ConnectionError, ValueError, ModelHTTPError) as e:
                ctx.deps.logger.error(
                    f"Cross-document judgment failed for {pair_key}: "
                    f"{type(e).__name__}: {e}"
                )
                # Mark as in-progress for error case (will be marked done in finally)
                if ctx.deps.progress:
                    ctx.deps.progress["Unique pairs"].work()
                # Default to rejection with low decision confidence (error case)
                first_assessment = assessments[0]
                return (
                    pair_key,
                    PairJudgment(
                        entity1=SimpleEntity(
                            name=first_assessment.entity1.canonical,
                            kind=first_assessment.entity1.kind,
                            aliases=_get_entity_aliases(
                                first_assessment.entity1.canonical, ctx.state
                            ),
                        ),
                        entity2=SimpleEntity(
                            name=first_assessment.entity2.canonical,
                            kind=first_assessment.entity2.kind,
                            aliases=_get_entity_aliases(
                                first_assessment.entity2.canonical, ctx.state
                            ),
                        ),
                        relationship=first_assessment.relationship,
                        spread=spread,
                        accepted=False,
                        evidence=_aggregate_evidence(assessments),
                        decision_confidence=0.3,  # Low confidence due to error
                        reasoning=f"Judgment failed due to error: {e}",
                    ),
                )

            # Validate document citations in reasoning (log warnings for invalid ones)
            cited_ids = extract_document_citations(result.output.reasoning)
            _, invalid_citations = validate_document_citations(cited_ids, valid_doc_ids)
            if invalid_citations:
                ctx.deps.logger.warning(
                    f"Invalid document citations in reasoning for {pair_key}: {invalid_citations}"
                )
            # Create judgment using LLM-selected relationship
            first_assessment = assessments[0]
            judgment = PairJudgment(
                entity1=SimpleEntity(
                    name=first_assessment.entity1.canonical,
                    kind=first_assessment.entity1.kind,
                    aliases=_get_entity_aliases(
                        first_assessment.entity1.canonical, ctx.state
                    ),
                ),
                entity2=SimpleEntity(
                    name=first_assessment.entity2.canonical,
                    kind=first_assessment.entity2.kind,
                    aliases=_get_entity_aliases(
                        first_assessment.entity2.canonical, ctx.state
                    ),
                ),
                relationship=result.output.relationship,
                spread=spread,
                accepted=result.output.accepted,
                evidence=result.output.evidence,
                decision_confidence=result.output.decision_confidence,
                reasoning=result.output.reasoning,
            )
            return (pair_key, judgment)
        finally:
            # Update counters when work completes (tight scoping, even on error)
            if ctx.deps.progress:
                # judgment is in locals() if we got to the return statement
                if "judgment" in locals():
                    if judgment.accepted:
                        ctx.deps.progress["Accepted"].add()
                    else:
                        ctx.deps.progress["Rejected"].add()
                ctx.deps.progress["Unique pairs"].done()


@dataclass
class SweepCoMentionsNode(BaseNode[State, Deps, ExtractionResult]):
    """Sweep documents for missed co-mentions of assessed entity pairs.

    Searches normalized text for entity co-occurrences that weren't captured
    by the initial proximal set extraction, assesses them for relationship
    claims, and adds any new assessments to the state.
    """

    async def run(
        self, ctx: GraphRunContext[State, Deps]
    ) -> "ConsolidateNewRelationshipsNode":
        """Scan for missed co-mentions and assess them."""
        from interaction_finder.extraction.sweep_co_mentions import (
            CoMentionRegion,
            CoMentionSweepStats,
            assess_co_mention_region,
            build_entity_search_pattern,
            collect_assessed_pairs,
            collect_global_aliases,
            find_novel_co_mentions_in_resource,
            get_entity_kind,
            merge_co_mentions_into_regions,
            select_co_mentions_to_assess,
        )

        with logfire.span("SweepCoMentionsNode"):
            ctx.deps.progress["Candidates"].activate()
            ctx.deps.progress.set_status("Sweeping for missed co-mentions")

            # Check if sweep is enabled
            if not ctx.deps.config.tools.extraction.sweep_co_mentions:
                ctx.deps.logger.info("Co-mention sweep disabled")
                return ConsolidateNewRelationshipsNode()

            stats = CoMentionSweepStats()

            # Step 1: Build global alias map
            global_aliases = collect_global_aliases(
                ctx.state.validated_entities_by_resource
            )

            # Step 2: Collect all assessed pairs
            assessed_pairs = collect_assessed_pairs(
                ctx.state.pair_assessments_by_resource
            )

            if not assessed_pairs:
                ctx.deps.logger.info("No assessed pairs to sweep for co-mentions")
                ctx.state.co_mention_sweep_stats = stats
                return ConsolidateNewRelationshipsNode()

            # Step 3: Build search patterns for all entities in assessed pairs
            entity_patterns: dict[str, "re.Pattern"] = {}
            for pair_key in assessed_pairs:
                for name in (pair_key.entity1_name, pair_key.entity2_name):
                    if name not in entity_patterns:
                        aliases = global_aliases.get(name, set())
                        entity_patterns[name] = build_entity_search_pattern(
                            name, aliases
                        )

            # Step 4: Find novel co-mentions in all resources
            chunk_distance = ctx.deps.config.tools.extraction.proximal_window_chunks
            all_co_mentions = []

            for resource in ctx.deps.resource_pool.resources:
                resource_co_mentions = find_novel_co_mentions_in_resource(
                    resource,
                    resource.id,
                    assessed_pairs,
                    entity_patterns,
                    ctx.state.pair_assessments_by_resource,
                    chunk_distance,
                )
                all_co_mentions.extend(resource_co_mentions)

            # Update discovery stats (raw counts, before deduplication)
            stats.total_co_mentions_found = len(all_co_mentions)
            for cm in all_co_mentions:
                if cm.priority == "no_existing_assessment":
                    stats.co_mentions_no_existing_assessment += 1
                else:
                    stats.co_mentions_uncovered_region += 1
            ctx.deps.logger.info(
                f"Found {stats.total_co_mentions_found} co-mention occurrences "
                f"({stats.co_mentions_no_existing_assessment} no existing assessment, "
                f"{stats.co_mentions_uncovered_region} uncovered region)"
            )

            if not all_co_mentions:
                ctx.state.co_mention_sweep_stats = stats
                return ConsolidateNewRelationshipsNode()

            # Step 5: Select co-mentions to assess (policy hook)
            selected = select_co_mentions_to_assess(all_co_mentions, ctx.deps.config)

            # Step 6: Build entity kind lookup and merge into regions
            entity_kinds: dict[str, str] = {}
            for name in entity_patterns:
                kind = get_entity_kind(name, ctx.state.validated_entities_by_resource)
                if kind:
                    entity_kinds[name] = kind

            regions = merge_co_mentions_into_regions(selected, entity_kinds)
            stats.regions_created = len(regions)
            # Count deduplicated pairs (unique pairs per region, summed across regions)
            pairs_to_assess = sum(len(r.candidate_pairs) for r in regions)
            # Update progress display with deduplicated count
            ctx.deps.progress["Candidates"].completed = pairs_to_assess
            ctx.deps.progress["Regions"].total = len(regions)
            ctx.deps.progress["Regions"].activate()
            ctx.deps.progress["Pairs added"].activate()
            ctx.deps.logger.info(
                f"Merged {len(selected)} co-mentions into {len(regions)} regions "
                f"({pairs_to_assess} candidate pairs)"
            )

            # Collect known relationship types for prompt context
            known_relationships = sorted(ctx.state.relationship_polarities.keys())
            # Accumulate new pairs by resource for deduplication and batch assessment
            new_pairs_by_resource: dict[ResourceId, list[tuple]] = defaultdict(list)

            # Step 7: Assess regions concurrently (with 1-based indexing for diagnostics)
            async def assess_region(region_index: int, region: CoMentionRegion):
                """Assess all pairs in a region with a single LLM call."""
                # Note: .work() is called inside assess_co_mention_region after semaphore acquisition
                try:
                    resource = ctx.deps.resource_pool.get(region.resource_id)
                    if resource is None:
                        return (region, [], [])
                    validated_entities = ctx.state.validated_entities_by_resource.get(
                        region.resource_id, {}
                    )
                    assessments, new_pairs = await assess_co_mention_region(
                        region,
                        resource,
                        topic=ctx.state.topic,
                        known_relationships=known_relationships,
                        config=ctx.deps.config,
                        deps=ctx.deps,
                        region_index=region_index,
                        validated_entities=validated_entities,
                        permitted_pairs=ctx.state.permitted_pairs,
                    )
                    return (region, assessments, new_pairs)
                finally:
                    # Mark region as done (moves from in-progress to completed)
                    ctx.deps.progress["Regions"].done()

            tasks = [
                asyncio.create_task(assess_region(idx + 1, r))
                for idx, r in enumerate(regions)
            ]
            for coro in asyncio.as_completed(tasks):
                region, assessments, new_pairs = await coro
                # Update stats: count pairs assessed, not regions
                stats.assessed += len(region.candidate_pairs)
                stats.relationships_found += len(assessments)
                stats.no_relationship_claim += len(region.candidate_pairs) - len(
                    assessments
                )
                stats.new_pairs_discovered += len(new_pairs)
                ctx.deps.progress["Pairs added"].completed = stats.relationships_found
                ctx.deps.progress["Pairs added"].total = stats.assessed
                # Add assessments to state
                for assessment in assessments:
                    if region.resource_id not in ctx.state.pair_assessments_by_resource:
                        ctx.state.pair_assessments_by_resource[region.resource_id] = []
                    ctx.state.pair_assessments_by_resource[region.resource_id].append(
                        assessment
                    )
                # Accumulate new pairs for later assessment
                new_pairs_by_resource[region.resource_id].extend(new_pairs)
            # Mark sweep phase complete
            ctx.deps.progress["Regions"].complete()
            ctx.deps.progress["Pairs added"].complete()
            ctx.deps.logger.info(
                f"Co-mention sweep assessed {stats.assessed} pairs in {stats.regions_created} regions, "
                f"found {stats.relationships_found} relationships, "
                f"discovered {stats.new_pairs_discovered} new pairs"
            )
            # Step 8: Convert discovered pairs to assessments
            # (already have validated quotes and relationships from sweep)
            for resource_id, raw_pairs in new_pairs_by_resource.items():
                entities = ctx.state.validated_entities_by_resource.get(resource_id, {})
                for e1_name, e2_name, rel_types, quotes in raw_pairs:
                    entity1, entity2 = entities.get(e1_name), entities.get(e2_name)
                    if entity1 is None or entity2 is None:
                        continue
                    if resource_id not in ctx.state.pair_assessments_by_resource:
                        ctx.state.pair_assessments_by_resource[resource_id] = []
                    ctx.state.pair_assessments_by_resource[resource_id].append(
                        PairAssessment(
                            resource_id=resource_id,
                            entity1=entity1,
                            entity2=entity2,
                            relationship=rel_types[0]
                            if rel_types
                            else "associated_with",
                            quotes=quotes,
                            evidence=EvidenceQuality(
                                directness="implied",
                                source_type="primary",
                                specificity="associative",
                                language="hedged",
                                overall=5,
                            ),
                            reasoning="Discovered during co-mention sweep",
                            source="sweep",
                        )
                    )
                    stats.new_pairs_assessed += 1
            if stats.new_pairs_assessed > 0:
                ctx.deps.logger.info(
                    f"Added {stats.new_pairs_assessed} newly discovered pairs"
                )

            ctx.state.co_mention_sweep_stats = stats

            # Save checkpoint after co-mention sweep
            await _save_partial_checkpoint(ctx, "sweep_co_mentions")

            return ConsolidateNewRelationshipsNode()


@dataclass
class ConsolidateNewRelationshipsNode(BaseNode[State, Deps, ExtractionResult]):
    """Consolidate relationship labels from new assessments (incremental).

    After the co-mention sweep, any new relationship labels need polarity
    classification. This node finds labels not yet in relationship_polarities
    and classifies them.
    """

    async def run(self, ctx: GraphRunContext[State, Deps]) -> "JudgeCrossDocumentNode":
        """Classify polarities for any new relationship labels."""
        from interaction_finder.extraction.consolidate_relationships import (
            get_relationship_consolidation_agent,
        )

        with logfire.span("ConsolidateNewRelationshipsNode"):
            if ctx.deps.progress:
                ctx.deps.progress.set_status("Classifying new relationships")
            # Collect all current relationship labels
            all_labels: set[str] = set()
            for assessments in ctx.state.pair_assessments_by_resource.values():
                for assessment in assessments:
                    all_labels.add(assessment.relationship)

            # Find labels not yet classified
            new_labels = {
                label
                for label in all_labels
                if label not in ctx.state.relationship_polarities
            }

            if not new_labels:
                ctx.deps.logger.info("No new relationship labels to classify")
                return JudgeCrossDocumentNode()

            ctx.deps.logger.info(
                f"Classifying {len(new_labels)} new relationship labels"
            )

            # Query LLM for polarity classification
            new_labels_list = sorted(new_labels)
            new_labels_str = "\n".join(f"- {r!r}" for r in new_labels_list)
            entity_types_str = ", ".join(ctx.state.target_entity_types)
            # Include existing labels so agent can consolidate new ones into them
            existing_labels = sorted(ctx.state.relationship_polarities.keys())
            if existing_labels:
                existing_str = "\n".join(
                    f"- {r!r} ({ctx.state.relationship_polarities[r]})"
                    for r in existing_labels
                )
                existing_section = f"""
**Existing canonical relationship labels (for reference):**
{existing_str}
"""
            else:
                existing_section = ""

            prompt = f"""**Research topic:** {ctx.state.topic}

**Target entity types:** {entity_types_str}
{existing_section}
**New relationship labels to classify:**
{new_labels_str}

For each new relationship, provide:
1. Consolidated canonical form (use an existing label if appropriate, or the original if distinct)
2. Polarity classification relative to this research topic"""

            agent = get_relationship_consolidation_agent(ctx.deps.config)
            usage = RunUsage()
            try:
                # Initial attempt
                with rename_agent(agent, name="ConsolidateNewRelationshipsNode"):
                    async with ctx.deps.agent_semaphore:
                        result = await agent.run(prompt, deps=ctx.deps, usage=usage)
                self._apply_relationship_consolidations(
                    result.output.consolidations, ctx
                )

                # Retry up to 3 times for any missing labels
                missing = new_labels - set(ctx.state.relationship_polarities.keys())
                for attempt in range(1, 4):
                    if not missing:
                        break
                    ctx.deps.logger.warning(
                        f"LLM omitted {len(missing)} labels; retrying (attempt {attempt}/3)"
                    )
                    retry_prompt = self._build_classification_prompt(
                        ctx, missing, entity_types_str, existing_section
                    )
                    try:
                        with rename_agent(
                            agent,
                            name=f"ConsolidateNewRelationshipsNode-retry{attempt}",
                        ):
                            async with ctx.deps.agent_semaphore:
                                result = await agent.run(
                                    retry_prompt, deps=ctx.deps, usage=usage
                                )
                        self._apply_relationship_consolidations(
                            result.output.consolidations, ctx
                        )
                        missing = missing - set(
                            ctx.state.relationship_polarities.keys()
                        )
                    except (
                        TimeoutError,
                        ConnectionError,
                        ValueError,
                        ModelHTTPError,
                    ) as e:
                        ctx.deps.logger.warning(
                            f"Retry {attempt} failed: {type(e).__name__}"
                        )
                        break

                # Default any remaining missing labels to neutral
                if missing:
                    ctx.deps.logger.warning(
                        f"Defaulting {len(missing)} missing labels to neutral"
                    )
                    for label in missing:
                        ctx.state.relationship_polarities[label] = "neutral"

            except (TimeoutError, ConnectionError, ValueError, ModelHTTPError) as e:
                ctx.deps.logger.warning(
                    f"New relationship consolidation failed: {type(e).__name__}: {e}; "
                    f"defaulting new labels to neutral polarity"
                )
                for label in new_labels:
                    ctx.state.relationship_polarities[label] = "neutral"

            # Filter pairs with only irrelevant relationships (if enabled)
            if ctx.deps.config.tools.extraction.filter_irrelevant_relationships:
                self._filter_irrelevant_pairs(ctx)

            # Save checkpoint after new relationship consolidation
            await _save_partial_checkpoint(ctx, "consolidate_new_relationships")

            return JudgeCrossDocumentNode()

    def _build_classification_prompt(
        self,
        ctx: GraphRunContext[State, Deps],
        labels: set[str],
        entity_types_str: str,
        existing_section: str,
    ) -> str:
        """Build prompt for relationship classification."""
        labels_str = "\n".join(f"- {r!r}" for r in sorted(labels))
        return f"""**Research topic:** {ctx.state.topic}

**Target entity types:** {entity_types_str}
{existing_section}
**Relationship labels to classify:**
{labels_str}

For each relationship, provide:
1. Consolidated canonical form (use an existing label if appropriate, or the original if distinct)
2. Polarity classification relative to this research topic"""

    def _apply_relationship_consolidations(
        self,
        consolidations: list[RelationshipConsolidation],
        ctx: GraphRunContext[State, Deps],
    ) -> None:
        """Apply relationship consolidations and store polarities."""
        # Store polarities
        for cons in consolidations:
            ctx.state.relationship_polarities[cons.original] = cons.polarity
            ctx.state.relationship_polarities[cons.consolidated] = cons.polarity
        # Append to unified consolidated structure
        if consolidations:
            ctx.state.consolidated.relationships.extend(consolidations)
        # Apply consolidation if label changed
        for cons in consolidations:
            if cons.original != cons.consolidated:
                norm_orig = normalize_for_comparison(cons.original)
                for assessments in ctx.state.pair_assessments_by_resource.values():
                    for assessment in assessments:
                        if (
                            normalize_for_comparison(assessment.relationship)
                            == norm_orig
                        ):
                            assessment.relationship = cons.consolidated

    def _filter_irrelevant_pairs(self, ctx: GraphRunContext[State, Deps]) -> None:
        """Filter pairs that now have only irrelevant relationships after sweep.

        Similar to ConsolidateRelationshipsNode._filter_irrelevant_assessments,
        but only processes pairs that had new sweep assessments. Creates
        pre-rejected judgments for pairs where ALL assessments are irrelevant.
        """
        # Group all assessments by pair (including sweep-added ones)
        grouped: dict[EntityPairKey, list[PairAssessment]] = defaultdict(list)
        for assessments in ctx.state.pair_assessments_by_resource.values():
            for assessment in assessments:
                pair_key = make_entity_pair_key(assessment.entity1, assessment.entity2)
                grouped[pair_key].append(assessment)

        # Find pairs that are ALL irrelevant and not already judged
        filtered_pairs: set[EntityPairKey] = set()
        for pair_key, assessments in grouped.items():
            # Skip if already has a judgment
            if pair_key in ctx.state.pair_judgments:
                continue

            # Check if all assessments are irrelevant
            all_irrelevant = all(
                ctx.state.relationship_polarities.get(a.relationship) == "irrelevant"
                for a in assessments
            )

            if all_irrelevant:
                spread = build_pair_spread(
                    assessments, ctx.state.relationship_polarities
                )
                first = assessments[0]
                ctx.state.pair_judgments[pair_key] = PairJudgment(
                    entity1=SimpleEntity(
                        name=first.entity1.canonical,
                        kind=first.entity1.kind,
                        aliases=_get_entity_aliases(first.entity1.canonical, ctx.state),
                    ),
                    entity2=SimpleEntity(
                        name=first.entity2.canonical,
                        kind=first.entity2.kind,
                        aliases=_get_entity_aliases(first.entity2.canonical, ctx.state),
                    ),
                    relationship=first.relationship,
                    spread=spread,
                    accepted=False,
                    evidence=_aggregate_evidence(assessments),
                    decision_confidence=0.95,  # High confidence in rejection decision
                    reasoning=(
                        "All relationship types for this pair were classified as "
                        "irrelevant to the research question"
                    ),
                )
                filtered_pairs.add(pair_key)
        if filtered_pairs:
            # Remove assessments for filtered pairs
            for resource_id, assessments in list(
                ctx.state.pair_assessments_by_resource.items()
            ):
                filtered = [
                    a
                    for a in assessments
                    if make_entity_pair_key(a.entity1, a.entity2) not in filtered_pairs
                ]
                ctx.state.pair_assessments_by_resource[resource_id] = filtered

            ctx.deps.logger.info(
                f"Filtered {len(filtered_pairs)} pairs with only irrelevant relationships "
                "(from sweep assessments)"
            )


def _snapshot_entity_counts(state: State) -> dict[str, dict[str, int]]:
    """Capture entity mention counts as {kind: {name: count}}.

    Shared helper for capturing entity state at different pipeline stages.
    """
    snapshot: dict[str, dict[str, int]] = {}
    for entities in state.validated_entities_by_resource.values():
        for name, ref in entities.items():
            if ref.kind not in snapshot:
                snapshot[ref.kind] = {}
            snapshot[ref.kind][name] = snapshot[ref.kind].get(name, 0) + len(
                ref.mentions
            )
    return snapshot


@dataclass
class FinalizeNode(BaseNode[State, Deps, ExtractionResult]):
    """Build final output with all judgments and metadata."""

    def _finalize_consolidated(self, ctx: GraphRunContext[State, Deps]) -> None:
        """Finalize consolidated data: set final entity counts."""
        ctx.state.consolidated.entities.final = _snapshot_entity_counts(ctx.state)

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

            cache_hits, cache_misses = _aggregate_cache_stats(ctx.state)
            metadata = ExtractionMetadata(
                topic=ctx.state.topic,
                resource_count=len(ctx.deps.resource_pool.resources),
                total_entities_found=total_entities_found,
                entities_after_validation=entities_after_validation,
                entities_merged=ctx.state.entities_merged,
                merge_cache_hits=cache_hits,
                merge_cache_misses=cache_misses,
                proximal_sets_found=proximal_sets_found,
                total_pairs_found=total_pairs_found,
                pairs_accepted=pairs_accepted,
                pairs_rejected=pairs_rejected,
                quotes_validated=ctx.state.quotes_validated,
                quotes_failed=ctx.state.quotes_failed,
            )
            # Finalize consolidated data: set final counts and categorize merges
            self._finalize_consolidated(ctx)
            # Convert paper_quality keys from ResourceId to URL strings
            paper_quality_by_url = {
                rid.url: assessment
                for rid, assessment in ctx.state.paper_quality.items()
            }
            result = ExtractionResult(
                topic=ctx.state.topic,
                target_entity_types=ctx.state.target_entity_types,
                permitted_pairs={
                    k: list(v) for k, v in ctx.state.permitted_pairs.items()
                },
                resources=ctx.deps.resource_pool,
                judgments=all_judgments,
                metadata=metadata,
                consolidated=ctx.state.consolidated,
                entities=ctx.state.global_entities,
                paper_quality=paper_quality_by_url,
            )

            return End(result)
