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
import re
import secrets
import string
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
    validate_entity_kinds,
)
from interaction_finder.extraction.judge_cross_document import (
    get_cross_document_judge_agent,
)
from interaction_finder.extraction.consolidate_entities import (
    get_entity_consolidation_agent,
)
from interaction_finder.extraction.models import (
    ConsolidationRule,
    EntityPairKey,
    EntityRef,
    ExtractionMetadata,
    ExtractionResult,
    PairAssessment,
    PairJudgment,
    PairSpread,
    RelationshipConsolidation,
    SimpleEntity,
)
from interaction_finder.extraction.state import State
from interaction_finder.extraction.utils import (
    adjust_heading_levels,
    collect_relevant_text_for_quotes,
    extract_all_forms,
    extract_document_citations,
    identify_proximal_sets,
    is_obvious_variant,
    make_entity_pair_key,
    normalize_for_comparison,
    osa_distance,
    validate_document_citations,
)
from interaction_finder.logging import logfire
from interaction_finder.resources import Resource

# Characters for generating verification tokens (alphanumeric, mixed case)
_TOKEN_CHARS = string.ascii_letters + string.digits


def _generate_token(length: int = 4) -> str:
    """Generate a random alphanumeric token for pair verification."""
    return "".join(secrets.choice(_TOKEN_CHARS) for _ in range(length))


def _convert_consolidation_rules(
    state_rules: dict[tuple[str, str], tuple[str, str]],
) -> list[ConsolidationRule]:
    """Convert state consolidation rules to serializable ConsolidationRule objects."""
    return [
        ConsolidationRule(source=source, kind=kind, target=target, reasoning=reasoning)
        for (source, kind), (target, reasoning) in state_rules.items()
    ]


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
            consolidation_rules=[],
        )

    async def _process_document(
        self, resource: Resource, ctx: GraphRunContext[State, Deps]
    ):
        """Process a single document through the full per-document pipeline."""
        # Increment in-progress counter when work starts (tight scoping)
        if ctx.deps.progress:
            ctx.deps.progress["Processed"].work()

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
                if ctx.deps.progress:
                    ctx.deps.progress["Pairs assessed"].total += len(pairs)
                    ctx.deps.progress["Pairs assessed"].work(len(pairs))
                    ctx.deps.progress["Pairs assessed"].activate()
                    ctx.deps.progress.set_status("Assessing pairs")

                try:
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
                        ctx.state.pair_assessments_by_resource[resource.id] = (
                            assessments
                        )
                finally:
                    # Mark pairs as done (moves from in_progress to completed)
                    if ctx.deps.progress:
                        ctx.deps.progress["Pairs assessed"].done(len(pairs))

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
        """Consolidate entities globally and update pair references."""
        with logfire.span("ConsolidateEntitiesNode"):
            if ctx.deps.progress:
                ctx.deps.progress.set_status("Consolidating entities")
            max_iterations = ctx.deps.config.tools.extraction.max_rename_iterations
            total_rules = 0
            iterations_completed = 0
            # Collect entities once, then iteratively find/process merge opportunities
            unique_entities = self._collect_unique_entities(ctx)
            for iteration in range(1, max_iterations + 1):
                iterations_completed = iteration
                # Find auto-merge rules and LLM candidate pairs
                exact_rules, candidate_pairs, canonical_lookup = (
                    self._find_merge_candidates(unique_entities)
                )
                llm_rules, new_names = await self._get_consolidation_decisions(
                    candidate_pairs, canonical_lookup, unique_entities, ctx
                )
                # Combine and apply all rules
                all_rules = {**exact_rules, **llm_rules}
                if not all_rules:
                    break  # No more work to do
                resolved_rules = self._resolve_transitive_merges(all_rules)
                self._apply_merge_rules_globally(resolved_rules, ctx)
                self._update_pair_entity_references(resolved_rules, ctx)
                total_rules += len(resolved_rules)
                # If renames created new names, add them to unique_entities and re-check
                if not new_names:
                    break
                self._add_new_names_to_entities(new_names, unique_entities)
                ctx.deps.logger.debug(
                    f"Consolidation iteration {iteration}: {len(new_names)} renames, re-evaluating"
                )
            else:
                # Loop completed without break - hit max_iterations with pending renames
                if new_names:
                    ctx.deps.logger.warning(
                        f"Entity consolidation hit max iterations ({max_iterations}) "
                        f"with {len(new_names)} pending renames"
                    )
            iterations_suffix = (
                f" in {iterations_completed} iteration{'s' if iterations_completed != 1 else ''}"
                if iterations_completed > 1
                else ""
            )
            ctx.deps.logger.info(
                f"Entity consolidation: {total_rules} rules applied, "
                f"{ctx.state.entities_merged} entities merged{iterations_suffix}",
            )
            return ConsolidateRelationshipsNode()

    def _collect_unique_entities(
        self, ctx: GraphRunContext[State, Deps]
    ) -> dict[str, dict[str, set[str]]]:
        """Collect all unique normalized entity forms with their canonical names.

        Expands each entity to all its variant forms (name, aliases, parenthetical
        expansions) and creates a mapping from normalized forms to canonical entity
        names. When multiple entities map to the same normalized form, they become
        merge candidates.

        Also populates ctx.state.canonical_name_variants for debugging/introspection.
        This is cleared at the start of each collection to prevent stale data.

        Returns:
            {kind: {normalized_form: {canonical_name1, canonical_name2, ...}}}
        """
        # Clear canonical_name_variants to prevent accumulation across multiple runs
        ctx.state.canonical_name_variants.clear()

        unique_by_kind: dict[str, dict[str, set[str]]] = {}

        for entities in ctx.state.validated_entities_by_resource.values():
            for entity_name, entity in entities.items():
                entity_kind = entity.kind
                if entity_kind not in unique_by_kind:
                    unique_by_kind[entity_kind] = {}

                # Extract all forms: name + aliases + parenthetical expansions
                all_forms = extract_all_forms(entity_name, entity.aliases())

                # Normalize each form and map it to this entity's canonical name
                for form in all_forms:
                    norm_form = normalize_for_comparison(form)
                    if norm_form not in unique_by_kind[entity_kind]:
                        unique_by_kind[entity_kind][norm_form] = set()

                    unique_by_kind[entity_kind][norm_form].add(entity_name)

                    # Track canonical variant globally
                    norm_key = (norm_form, entity_kind)
                    if norm_key not in ctx.state.canonical_name_variants:
                        ctx.state.canonical_name_variants[norm_key] = set()
                    ctx.state.canonical_name_variants[norm_key].add(entity_name)

        return unique_by_kind

    def _select_best_canonical_name(self, canonical_names: set[str]) -> str:
        """Select the best canonical name from a set of capitalization variants.

        Uses a multi-tier heuristic to prefer mixed-case over all-uppercase or
        all-lowercase:
        1. Primary: Maximize mixed-case complexity (lowercase_count * uppercase_count)
        2. Tie-breaker 1: Prefer more uppercase letters
        3. Tie-breaker 2: Minimize lowercase letters

        This prefers: "Pulmonary Arterial Hypertension" > "PULMONARY" > "pulmonary"

        Parameters:
            canonical_names: Set of canonical name variants (same normalized form)

        Returns:
            The canonical name with best capitalization
        """
        return max(
            canonical_names,
            key=lambda v: (
                sum(1 for c in v if c.islower())
                * sum(1 for c in v if c.isupper()),  # Mixed-case score
                sum(1 for c in v if c.isupper()),  # Uppercase count
                -sum(1 for c in v if c.islower()),  # Minimize lowercase
            ),
        )

    def _find_merge_candidates(
        self, unique_entities: dict[str, dict[str, set[str]]]
    ) -> tuple[
        dict[tuple[str, str], tuple[str, str]],
        dict[str, list[tuple[str, str]]],
        dict[tuple[str, str], str],
    ]:
        """Find merge candidates: exact matches, fuzzy matches, and substring pairs.

        Three-phase approach:
        1. Auto-merge exact normalized matches (capitalization variants)
        2. Auto-merge fuzzy Tier 1 matches (OSA distance=1 + obvious patterns)
        3. Find substring/fuzzy Tier 2 pairs for LLM review

        After Phases 1 & 2, builds a canonical lookup mapping every normalized
        form to its final canonical name (after applying all merge rules). This
        ensures Phase 3 lookups are consistent.

        Returns:
            Tuple of:
            - {(norm_child, kind): (target, reasoning)} for auto-merge rules with provenance
            - {kind: [(norm_parent, norm_child), ...]} for LLM review pairs
            - {(normalized_form, kind): canonical_name} for Phase 3 lookups
        """
        auto_merge_rules: dict[tuple[str, str], tuple[str, str]] = {}

        # PHASE 1: Handle exact normalized matches (capitalization variants)
        # When multiple canonical names map to the same normalized form, merge them
        # Rules are keyed by normalized form (not canonical) for cross-document consistency
        phase1_rule_count = 0
        for kind, normalized_entities in unique_entities.items():
            for norm_form, canonical_names in list(normalized_entities.items()):
                if len(canonical_names) > 1:
                    # Select best canonical name using capitalization complexity
                    canonical = self._select_best_canonical_name(canonical_names)
                    # Create merge rules keyed by normalized form of child
                    for child_name in canonical_names:
                        if child_name != canonical:
                            child_norm = normalize_for_comparison(child_name)
                            auto_merge_rules[(child_norm, kind)] = (
                                canonical,
                                "auto:cap",
                            )
                            phase1_rule_count += 1
                    # CONSOLIDATE: Update unique_entities to have single canonical
                    # This ensures phase 2 works with consolidated 1:1 mappings
                    normalized_entities[norm_form] = {canonical}

        # PHASE 1.5: Propagate merge rules to ALL normalized forms
        # If "BRCA1" was merged into "Brca1", update all normalized forms
        # that reference "BRCA1" to point to "Brca1" instead
        if phase1_rule_count > 0:
            self._propagate_rules_to_unique_entities(auto_merge_rules, unique_entities)
        # PHASE 2: Handle fuzzy matches (spelling variants, pluralization)
        # Now that capitalization variants are consolidated, check for OSA patterns
        # Rules are keyed by normalized form for consistency
        # Track which normalized forms had Phase 1 rules (multiple cap variants = established)
        phase1_winners = {key[0] for key in auto_merge_rules.keys()}
        phase2_rule_count = 0
        for kind, normalized_entities in unique_entities.items():
            norm_list = list(normalized_entities.keys())
            for i, norm1 in enumerate(norm_list):
                for norm2 in norm_list[i + 1 :]:
                    # Only check if entity names are long enough to avoid false positives
                    max_len = max(len(norm1), len(norm2))
                    if max_len >= 10:
                        distance = osa_distance(norm1, norm2)
                        # Tier 1: Distance 1 + obvious pattern → auto-merge
                        if distance == 1 and is_obvious_variant(norm1, norm2):
                            # Get canonical names (now 1:1 after phase 1 consolidation)
                            canonical1 = next(iter(normalized_entities[norm1]))
                            canonical2 = next(iter(normalized_entities[norm2]))
                            # Determine parent: prefer Phase 1 winner, else shorter norm
                            # Phase 1 winners had multiple cap variants = more established
                            norm1_established = norm1 in phase1_winners
                            norm2_established = norm2 in phase1_winners
                            if norm1_established and not norm2_established:
                                parent_norm, child_norm = norm1, norm2
                                parent_canonical = canonical1
                            elif norm2_established and not norm1_established:
                                parent_norm, child_norm = norm2, norm1
                                parent_canonical = canonical2
                            elif len(norm1) <= len(norm2):
                                # Neither or both established: fall back to shorter = parent
                                parent_norm, child_norm = norm1, norm2
                                parent_canonical = canonical1
                            else:
                                parent_norm, child_norm = norm2, norm1
                                parent_canonical = canonical2
                            auto_merge_rules[(child_norm, kind)] = (
                                parent_canonical,
                                "auto:fuzzy",
                            )
                            del normalized_entities[child_norm]
                            phase2_rule_count += 1
                            # Skip to next pair (one norm_form is now gone)
                            break
        # PHASE 2.5: Propagate Phase 2 merge rules to ALL normalized forms
        if phase2_rule_count > 0:
            self._propagate_rules_to_unique_entities(auto_merge_rules, unique_entities)
        # Build canonical lookup: (normalized_form, kind) → canonical_name
        # This is the single source of truth for Phase 3 lookups
        canonical_lookup: dict[tuple[str, str], str] = {}
        for kind, normalized_entities in unique_entities.items():
            for norm_form, canonical_names in normalized_entities.items():
                # After Phase 1 & 2 propagation, should be exactly one canonical
                assert len(canonical_names) == 1, (
                    f"Expected 1 canonical after consolidation, got {len(canonical_names)}: {canonical_names}"
                )
                canonical_lookup[(norm_form, kind)] = next(iter(canonical_names))

        # PHASE 3: Find substring and fuzzy Tier 2 pairs for LLM review
        # These work on fully consolidated entities (no capitalization variants)
        candidate_pairs: dict[str, list[tuple[str, str]]] = {}

        for kind, normalized_entities in unique_entities.items():
            norm_list = list(normalized_entities.keys())
            pairs = []

            for i, norm1 in enumerate(norm_list):
                for norm2 in norm_list[i + 1 :]:
                    # Check substring relationship
                    if norm1 in norm2 and norm1 != norm2:
                        pairs.append((norm1, norm2))
                    elif norm2 in norm1:
                        pairs.append((norm2, norm1))
                    else:
                        # Check fuzzy Tier 2: 2-10% edit distance → ask LLM
                        max_len = max(len(norm1), len(norm2))
                        if max_len >= 10:
                            distance = osa_distance(norm1, norm2)
                            ratio = distance / max_len

                            if ratio <= 0.10:
                                # Order by length (shorter = more general)
                                if len(norm1) <= len(norm2):
                                    pairs.append((norm1, norm2))
                                else:
                                    pairs.append((norm2, norm1))

            if pairs:
                candidate_pairs[kind] = pairs

        return auto_merge_rules, candidate_pairs, canonical_lookup

    def _propagate_rules_to_unique_entities(
        self,
        rules: dict[tuple[str, str], tuple[str, str]],
        unique_entities: dict[str, dict[str, set[str]]],
    ) -> None:
        """Propagate merge rules to update canonical names in unique_entities.

        After creating merge rules, canonical names in unique_entities may reference
        entities that have been merged away. This follows rule chains to update all
        references to point to the final canonical name.

        Rules are (norm_child, kind) → (target, reasoning). Only target is used here.

        Mutates unique_entities in place.
        """
        for kind, normalized_entities in unique_entities.items():
            for norm_form, canonical_names in list(normalized_entities.items()):
                updated_canonicals = set()
                for canonical in canonical_names:
                    # Follow redirect chain using normalized keys
                    current = canonical
                    visited = {normalize_for_comparison(current)}
                    while (normalize_for_comparison(current), kind) in rules:
                        current = rules[(normalize_for_comparison(current), kind)][0]
                        current_norm = normalize_for_comparison(current)
                        if current_norm in visited:
                            break  # Cycle detection
                        visited.add(current_norm)
                    updated_canonicals.add(current)
                normalized_entities[norm_form] = updated_canonicals

    def _add_new_names_to_entities(
        self,
        new_names: set[tuple[str, str]],
        unique_entities: dict[str, dict[str, set[str]]],
    ) -> None:
        """Add new canonical names (from renames) to unique_entities for re-evaluation.

        This allows the existing _find_merge_candidates logic to detect any new
        merge opportunities created by the renames.
        """
        for new_canonical, kind in new_names:
            if kind not in unique_entities:
                unique_entities[kind] = {}
            new_norm = normalize_for_comparison(new_canonical)
            if new_norm not in unique_entities[kind]:
                unique_entities[kind][new_norm] = set()
            unique_entities[kind][new_norm].add(new_canonical)

    async def _get_consolidation_decisions(
        self,
        candidate_pairs_by_kind: dict[str, list[tuple[str, str]]],
        canonical_lookup: dict[tuple[str, str], str],
        unique_entities: dict[str, dict[str, set[str]]],
        ctx: GraphRunContext[State, Deps],
    ) -> tuple[dict[tuple[str, str], tuple[str, str]], set[tuple[str, str]]]:
        """Query LLM for consolidation decisions on entity pairs.

        Returns:
            Tuple of (rules, new_names) where:
            - rules maps (norm_child, kind) → (target, reasoning)
            - new_names is set of (new_canonical, kind) for targets that don't
              exist in unique_entities (i.e., renames to new names)
        """
        rules: dict[tuple[str, str], tuple[str, str]] = {}
        new_names: set[tuple[str, str]] = set()
        for kind, pairs in candidate_pairs_by_kind.items():
            kind_entities = unique_entities.get(kind, {})
            # Separate cached and uncached pairs
            uncached_pairs = []
            for norm_parent, norm_child in pairs:
                cache_key = (norm_parent, norm_child, kind)
                if cache_key in ctx.state.merge_decision_cache:
                    # Cache stores target canonical name (merge/rename) or False (skip)
                    target = ctx.state.merge_decision_cache[cache_key]
                    if target:
                        # Cached decisions use "cached" as reasoning (original LLM reasoning not stored)
                        rules[(norm_child, kind)] = (target, "cached")
                        # Check if target is a new name (rename to non-existent entity)
                        target_norm = normalize_for_comparison(target)
                        if target_norm not in kind_entities:
                            new_names.add((target, kind))
                    ctx.state.merge_cache_hits += 1
                else:
                    uncached_pairs.append((norm_parent, norm_child))
                    ctx.state.merge_cache_misses += 1
            # Query LLM for uncached pairs in batches
            batch_size = ctx.deps.config.tools.extraction.merge_batch_size
            for i in range(0, len(uncached_pairs), batch_size):
                batch = uncached_pairs[i : i + batch_size]
                batch_rules, batch_new_names = await self._process_consolidation_batch(
                    batch,
                    kind,
                    canonical_lookup,
                    kind_entities,
                    i // batch_size + 1,
                    ctx,
                )
                rules.update(batch_rules)
                new_names.update(batch_new_names)
        return rules, new_names

    async def _process_consolidation_batch(
        self,
        batch: list[tuple[str, str]],
        kind: str,
        canonical_lookup: dict[tuple[str, str], str],
        kind_entities: dict[str, set[str]],
        batch_num: int,
        ctx: GraphRunContext[State, Deps],
    ) -> tuple[dict[tuple[str, str], tuple[str, str]], set[tuple[str, str]]]:
        """Process a batch of entity pairs for consolidation decisions.

        Returns:
            Tuple of (rules, new_names) where:
            - rules maps (norm_child, kind) → (target, reasoning)
            - new_names is set of (new_canonical, kind) for targets not in kind_entities
        """
        rules: dict[tuple[str, str], tuple[str, str]] = {}
        new_names: set[tuple[str, str]] = set()
        # Build prompt structures
        pairs_description = []
        seen_canonical: set[tuple[str, str]] = set()
        canonical_to_norm: dict[tuple[str, str], list[tuple[str, str]]] = {}
        # id → (canonical_pair, norm_child, token) - norm_child used to key rules
        id_to_pair: dict[int, tuple[tuple[str, str], str, str]] = {}
        token_to_id: dict[str, int] = {}
        pair_id = 0
        for norm_parent, norm_child in batch:
            parent = canonical_lookup[(norm_parent, kind)]
            child = canonical_lookup[(norm_child, kind)]
            # Skip identical canonical names (auto-cache as merge to parent)
            if parent == child:
                ctx.state.merge_decision_cache[(norm_parent, norm_child, kind)] = parent
                continue
            # Track norm→canonical mapping
            canonical_pair = (parent, child)
            canonical_to_norm.setdefault(canonical_pair, []).append(
                (norm_parent, norm_child)
            )
            # Skip duplicate canonical pairs
            if canonical_pair in seen_canonical:
                continue
            seen_canonical.add(canonical_pair)
            # Build prompt entry
            pair_id += 1
            token = _generate_token()
            id_to_pair[pair_id] = (canonical_pair, norm_child, token)
            token_to_id[token] = pair_id
            pairs_description.append(f"{pair_id}. [{token}] {child!r} → {parent!r}")
        if not pairs_description:
            return rules, new_names
        # Build and execute prompt
        prompt = f"""**Research topic:** {ctx.state.topic}

**Target entity types:** {", ".join(ctx.state.target_entity_types)}

**Entity pairs to evaluate:**
{chr(10).join(pairs_description)}

For each pair, decide the appropriate action (skip, merge, or rename)."""
        agent = get_entity_consolidation_agent(ctx.deps.config)
        try:
            with rename_agent(agent, name=f"ConsolidateEntities ({kind}, {batch_num})"):
                async with ctx.deps.agent_semaphore:
                    result = await agent.run(prompt, deps=ctx.deps, usage=RunUsage())
            # Process decisions
            for decision in result.output.decisions:
                resolved = _resolve_pair_from_decision(
                    decision.pair_id,
                    decision.pair_token,
                    id_to_pair,
                    token_to_id,
                    ctx.deps.logger,
                )
                if resolved is None:
                    continue
                canonical_pair, norm_child = resolved
                parent, child = canonical_pair
                norm_pairs = canonical_to_norm[canonical_pair]
                # Determine target based on action
                if decision.action == "skip":
                    for np, nc in norm_pairs:
                        ctx.state.merge_decision_cache[(np, nc, kind)] = False
                    continue
                elif decision.action == "merge":
                    target = parent
                elif decision.action == "rename" and decision.target:
                    target = decision.target
                else:
                    continue
                # Cache decision and create rule (same for merge and rename)
                for np, nc in norm_pairs:
                    ctx.state.merge_decision_cache[(np, nc, kind)] = target
                rules[(norm_child, kind)] = (target, decision.reasoning)
                # Check if target is a new name (not in existing entities)
                target_norm = normalize_for_comparison(target)
                if target_norm not in kind_entities:
                    new_names.add((target, kind))
        except (TimeoutError, ConnectionError, ValueError) as e:
            ctx.deps.logger.error(f"Entity consolidation failed: {e}")
        return rules, new_names

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

        Also stores rules in state.consolidation_rules for provenance tracking.
        """
        if not rules:
            return
        # Store rules for provenance (includes reasoning)
        ctx.state.consolidation_rules.update(rules)
        target_by_norm_and_kind: dict[tuple[str, str], str] = {}
        for (_norm, kind), (target, _) in rules.items():
            target_key = (normalize_for_comparison(target), kind)
            target_by_norm_and_kind.setdefault(target_key, target)
        for resource_id, entities in ctx.state.validated_entities_by_resource.items():
            # Build new EntityRefs without mutating underlying mentions
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
    - Polarity classification (supporting/refuting/neutral/irrelevant)
    - Topic-appropriate normalization (context-aware decisions)
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

            ctx.deps.logger.info(
                f"Relationship consolidation: {len(consolidations)} relationships processed, "
                f"{ctx.state.relationships_merged} assessments updated"
            )

            # Step 5: Filter irrelevant relationship types (if enabled)
            if ctx.deps.config.tools.extraction.filter_irrelevant_relationships:
                self._filter_irrelevant_assessments(ctx)

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
        relationships_str = "\n".join(f"- {r}" for r in relationships_list)
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
        except (TimeoutError, ConnectionError, ValueError) as e:
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
        from interaction_finder.extraction.utils import normalize_for_comparison

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
        from interaction_finder.extraction.utils import (
            build_pair_spread,
            make_entity_pair_key,
        )

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
                        aliases=first.entity1.aliases(),
                    ),
                    entity2=SimpleEntity(
                        name=first.entity2.canonical,
                        kind=first.entity2.kind,
                        aliases=first.entity2.aliases(),
                    ),
                    relationship=first.relationship,
                    spread=spread,
                    accepted=False,
                    confidence="high",
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

    async def run(self, ctx: GraphRunContext[State, Deps]) -> "FinalizeNode":
        """Judge all unique pairs across documents."""
        with logfire.span("JudgeCrossDocumentNode"):
            # Set phase
            if ctx.deps.progress:
                ctx.deps.progress["Unique pairs"].activate()
                ctx.deps.progress.set_status("Cross-document validation")

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
                tasks.append(self._judge_pair(pair_key, assessments, ctx))

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
            sections = [f"## {label.title()} Evidence\n"]
            for assessment in assessments:
                resource = ctx.deps.resource_pool.get(assessment.resource_id)
                if not resource:
                    continue
                text = collect_relevant_text_for_quotes(
                    resource, assessment.quotes, padding
                )
                adjusted_text = adjust_heading_levels(text, target_min_level=4)
                doc_id = assessment.resource_id.id
                section = f"""### Document extract [{doc_id}]: {resource.title}
**Relationship:** {assessment.relationship} | **Confidence:** {assessment.confidence}
**Reasoning:** {assessment.reasoning}

{adjusted_text}"""
                sections.append(section)
            return "\n\n".join(sections)

        supporting_text = format_assessments(spread.supporting, "Supporting")
        refuting_text = format_assessments(spread.refuting, "Refuting")
        neutral_text = format_assessments(spread.neutral, "Neutral")

        # Get all unique relationships
        all_rels = {
            a.relationship for a in spread.supporting + spread.refuting + spread.neutral
        }
        relationships_str = ", ".join(f'"{r}"' for r in sorted(all_rels))

        return f"""# Context
Synthesize contradictory evidence for an entity association.

**Topic:** {ctx.state.topic}

**Pair:** {pair_key.entity1_name} <-> {pair_key.entity2_name}

**Relationship types found:** {relationships_str}

**Note:** This is a CONTENTIOUS pair with contradictory evidence. Some documents support the association while others refute it.

# Document Extracts

{supporting_text}

{refuting_text}

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
        """Build prompt for unidirectional pairs (no supporting+refuting conflict)."""
        padding = getattr(ctx.deps.config.tools.extraction, "region_padding_chunks", 1)

        # Combine all assessments (one polarity category will dominate)
        all_assessments = spread.supporting + spread.refuting + spread.neutral
        document_sections = []
        for assessment in all_assessments:
            resource = ctx.deps.resource_pool.get(assessment.resource_id)
            if not resource:
                continue
            text = collect_relevant_text_for_quotes(
                resource, assessment.quotes, padding
            )
            adjusted_text = adjust_heading_levels(text, target_min_level=3)
            doc_id = assessment.resource_id.id
            section = f"""## Document extract [{doc_id}]: {resource.title}
**Assessment:** {assessment.confidence} confidence - {assessment.relationship}
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

Provide: accepted (true/false), relationship (selected label), confidence (high/medium/low), and detailed reasoning. Cite documents using their IDs in square brackets (e.g., [1_abc12345]) when referencing specific evidence."""

    async def _judge_pair(
        self,
        pair_key: EntityPairKey,
        assessments: list[PairAssessment],
        ctx: GraphRunContext[State, Deps],
    ) -> tuple[EntityPairKey, PairJudgment]:
        """Make final judgment on a single pair."""
        # Increment in-progress counter when work starts (tight scoping)
        if ctx.deps.progress:
            ctx.deps.progress["Unique pairs"].work()

        try:
            from interaction_finder.extraction.utils import build_pair_spread

            # Build PairSpread by polarity
            spread = build_pair_spread(assessments, ctx.state.relationship_polarities)

            # Try deterministic accept
            can_accept, relationship, reasoning = self._can_accept_deterministically(
                assessments
            )
            if can_accept:
                # Create judgment without LLM call
                first_assessment = assessments[0]
                judgment = PairJudgment(
                    entity1=SimpleEntity(
                        name=first_assessment.entity1.canonical,
                        kind=first_assessment.entity1.kind,
                        aliases=first_assessment.entity1.aliases(),
                    ),
                    entity2=SimpleEntity(
                        name=first_assessment.entity2.canonical,
                        kind=first_assessment.entity2.kind,
                        aliases=first_assessment.entity2.aliases(),
                    ),
                    relationship=relationship,
                    spread=spread,
                    accepted=True,
                    confidence="high",
                    reasoning=reasoning,
                )
                return (pair_key, judgment)
            # Need LLM investigation
            # Collect valid document IDs from all assessments for citation validation
            valid_doc_ids = {a.resource_id.id for a in assessments}
            # Detect contentious pairs (supporting + refuting evidence)
            is_contentious = bool(spread.supporting and spread.refuting)

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
                            name=first_assessment.entity1.canonical,
                            kind=first_assessment.entity1.kind,
                            aliases=first_assessment.entity1.aliases(),
                        ),
                        entity2=SimpleEntity(
                            name=first_assessment.entity2.canonical,
                            kind=first_assessment.entity2.kind,
                            aliases=first_assessment.entity2.aliases(),
                        ),
                        relationship=first_assessment.relationship,
                        spread=spread,
                        accepted=False,
                        confidence="low",
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
                    aliases=first_assessment.entity1.aliases(),
                ),
                entity2=SimpleEntity(
                    name=first_assessment.entity2.canonical,
                    kind=first_assessment.entity2.kind,
                    aliases=first_assessment.entity2.aliases(),
                ),
                relationship=result.output.relationship,
                spread=spread,
                accepted=result.output.accepted,
                confidence=result.output.confidence,
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

            # Step 7: Assess regions concurrently
            async def assess_region(region):
                """Assess all pairs in a region with a single LLM call."""
                # Mark region as in-progress
                ctx.deps.progress["Regions"].work()
                try:
                    resource = ctx.deps.resource_pool.get(region.resource_id)
                    if resource is None:
                        return (region, [])
                    validated_entities = ctx.state.validated_entities_by_resource.get(
                        region.resource_id, {}
                    )
                    assessments = await assess_co_mention_region(
                        region,
                        resource,
                        topic=ctx.state.topic,
                        known_relationships=known_relationships,
                        config=ctx.deps.config,
                        deps=ctx.deps,
                        validated_entities=validated_entities,
                    )
                    return (region, assessments)
                finally:
                    # Mark region as done (moves from in-progress to completed)
                    ctx.deps.progress["Regions"].done()

            # Run region assessments with as_completed for live progress
            tasks = [asyncio.create_task(assess_region(r)) for r in regions]
            for coro in asyncio.as_completed(tasks):
                region, assessments = await coro
                # Update stats: count pairs assessed, not regions
                stats.assessed += len(region.candidate_pairs)
                stats.relationships_found += len(assessments)
                stats.no_relationship_claim += len(region.candidate_pairs) - len(
                    assessments
                )
                ctx.deps.progress["Pairs added"].completed = stats.relationships_found
                ctx.deps.progress["Pairs added"].total = stats.assessed
                # Add assessments to state
                for assessment in assessments:
                    if region.resource_id not in ctx.state.pair_assessments_by_resource:
                        ctx.state.pair_assessments_by_resource[region.resource_id] = []
                    ctx.state.pair_assessments_by_resource[region.resource_id].append(
                        assessment
                    )
            # Mark sweep phase complete
            ctx.deps.progress["Regions"].complete()
            ctx.deps.progress["Pairs added"].complete()
            ctx.deps.logger.info(
                f"Co-mention sweep assessed {stats.assessed} pairs in {stats.regions_created} regions, "
                f"found {stats.relationships_found} relationships"
            )

            ctx.state.co_mention_sweep_stats = stats
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
            new_labels_str = "\n".join(f"- {r}" for r in new_labels_list)
            entity_types_str = ", ".join(ctx.state.target_entity_types)
            # Include existing labels so agent can consolidate new ones into them
            existing_labels = sorted(ctx.state.relationship_polarities.keys())
            if existing_labels:
                existing_str = "\n".join(
                    f"- {r} ({ctx.state.relationship_polarities[r]})"
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
                with rename_agent(agent, name="ConsolidateNewRelationshipsNode"):
                    async with ctx.deps.agent_semaphore:
                        result = await agent.run(prompt, deps=ctx.deps, usage=usage)

                # Apply consolidations and store polarities
                for cons in result.output.consolidations:
                    ctx.state.relationship_polarities[cons.original] = cons.polarity
                    ctx.state.relationship_polarities[cons.consolidated] = cons.polarity
                    # Apply consolidation if label changed
                    if cons.original != cons.consolidated:
                        norm_orig = normalize_for_comparison(cons.original)
                        for (
                            assessments
                        ) in ctx.state.pair_assessments_by_resource.values():
                            for assessment in assessments:
                                if (
                                    normalize_for_comparison(assessment.relationship)
                                    == norm_orig
                                ):
                                    assessment.relationship = cons.consolidated

            except (TimeoutError, ConnectionError, ValueError) as e:
                ctx.deps.logger.warning(
                    f"New relationship consolidation failed: {type(e).__name__}: {e}; "
                    f"defaulting new labels to neutral polarity"
                )
                # Default new labels to neutral
                for label in new_labels:
                    ctx.state.relationship_polarities[label] = "neutral"

            # Filter pairs with only irrelevant relationships (if enabled)
            if ctx.deps.config.tools.extraction.filter_irrelevant_relationships:
                self._filter_irrelevant_pairs(ctx)

            return JudgeCrossDocumentNode()

    def _filter_irrelevant_pairs(self, ctx: GraphRunContext[State, Deps]) -> None:
        """Filter pairs that now have only irrelevant relationships after sweep.

        Similar to ConsolidateRelationshipsNode._filter_irrelevant_assessments,
        but only processes pairs that had new sweep assessments. Creates
        pre-rejected judgments for pairs where ALL assessments are irrelevant.
        """
        from interaction_finder.extraction.utils import build_pair_spread

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
                        aliases=first.entity1.aliases(),
                    ),
                    entity2=SimpleEntity(
                        name=first.entity2.canonical,
                        kind=first.entity2.kind,
                        aliases=first.entity2.aliases(),
                    ),
                    relationship=first.relationship,
                    spread=spread,
                    accepted=False,
                    confidence="high",
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
                topic=ctx.state.topic,
                target_entity_types=ctx.state.target_entity_types,
                permitted_pairs={
                    k: list(v) for k, v in ctx.state.permitted_pairs.items()
                },
                resources=ctx.deps.resource_pool,
                judgments=all_judgments,
                metadata=metadata,
                consolidation_rules=_convert_consolidation_rules(
                    ctx.state.consolidation_rules
                ),
            )

            return End(result)
