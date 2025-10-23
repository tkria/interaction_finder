"""
Pipeline nodes for the extraction graph V3.

This version delegates entity extraction, assessment, and pair evaluation
directly to the stateless helpers in `extraction_core`, removing the mutable
`deps.current_*` fields that previously carried per-call state.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, TYPE_CHECKING

from pydantic_graph import BaseNode, GraphRunContext, End

from .state import ExtractionStateV3
from .deps import ExtractionDepsV3
from .cache import compute_extraction_cache_key, compute_assessment_cache_key
from .models import PairCandidate
from ..extraction_graph_v2.models import (
    EntityWithQuotes,
    IndividualAssessment,
    EntityPairOut,
)
from ..extraction_graph_v2.parallelism import with_parallelism_control
from ..resources import ResourceId
from ..extraction_core.analysis import analyze_entity
from ..extraction_core.evaluation import evaluate_pair
from ..extraction_core.extraction import extract_from_resource
from ..extraction_core.pairing import find_cooccurring_pairs
from ..extraction_core.models import (
    AnalysisConfig,
    CooccurrenceStrategy,
    EvaluationConfig,
)

if TYPE_CHECKING:  # pragma: no cover - type-checking only
    from .models import PairCandidate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ExtractEntities
# ---------------------------------------------------------------------------


@dataclass
class ExtractEntities(BaseNode[ExtractionStateV3, ExtractionDepsV3]):
    """
    Extract entities from documents using extraction_core.

    The node is responsible for caching, metrics, and deduplication; the actual
    entity detection is handled by `extract_from_resource`.
    """

    async def run(
        self, ctx: GraphRunContext[ExtractionStateV3, ExtractionDepsV3]
    ) -> "AssessIndividually":
        state, deps = ctx.state, ctx.deps

        resources = state.resource_pool.resources
        logger.info(f"Starting entity extraction from {len(resources)} documents")
        logger.info(f"Entity kinds: {deps.get_entity_kinds()}")

        entity_kinds = deps.get_entity_kinds()
        if not entity_kinds:
            raise ValueError(
                "No entity kinds configured. The V3 extraction pipeline requires "
                "entity types to be specified.\n\n"
                "Add a [task.kinds] section to your config.toml file with at least "
                "one entity type.\n\n"
                "Example configuration:\n"
                "[task.kinds]\n"
                "gene = { kind = 'gene', form = ['name', 'symbol'] }\n"
                "disease = { kind = 'disease', form = ['name'] }\n\n"
                "See documentation for more details on configuring entity kinds."
            )

        results = await with_parallelism_control(
            resources,
            lambda resource: self._extract_from_document(resource.id, ctx),
            parallelism=deps.extraction_parallelism,
            description="document extraction",
        )

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

        if hasattr(deps, "checkpoint_callback") and deps.checkpoint_callback:
            await deps.checkpoint_callback("extraction", state)

        return AssessIndividually()

    async def _extract_from_document(
        self,
        resource_id: ResourceId,
        ctx: GraphRunContext[ExtractionStateV3, ExtractionDepsV3],
    ) -> List[EntityWithQuotes]:
        state, deps = ctx.state, ctx.deps
        cache = state.cache

        resource = state.resource_pool.get(resource_id)
        if resource is None:
            logger.warning(f"Resource {resource_id} not found in pool")
            return []

        cache_key = compute_extraction_cache_key(
            full_doc_text=resource.text,
            entity_kinds=deps.get_entity_kinds(),
            model_version=str(deps.model),
            prompt_version="v3_2025-10",  # Increment when prompts/config change
        )

        if cache and deps.semantic_cache_enabled:
            cached = await cache.get_extraction(cache_key)
            if cached is not None:
                state.metrics.record_cache_hit("extraction")
                logger.debug(f"Cache hit for {resource.title} ({len(cached)} entities)")
                return cached

            state.metrics.record_cache_miss("extraction")

        similarity_threshold = (
            deps.fuzzy_auto_correct_threshold if deps.fuzzy_matching_enabled else 0.90
        )

        start_time = time.time()
        try:
            entities = await extract_from_resource(
                resource=resource,
                entity_kinds=deps.get_entity_kinds(),
                model=deps.model,
                task_context=deps.get_task_context(),
                similarity_threshold=similarity_threshold,
            )
            duration = time.time() - start_time
            state.metrics.record_extraction_call(success=True, duration=duration)

            logger.info(
                f"Extracted {len(entities)} entities from {resource.title} "
                f"in {duration:.2f}s"
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            duration = time.time() - start_time
            state.metrics.record_extraction_call(success=False, duration=duration)
            logger.error(f"Extraction failed for {resource.title}: {exc}")
            return []

        if cache and deps.semantic_cache_enabled and entities:
            await cache.set_extraction(cache_key, entities)
            logger.debug(
                f"Saved {len(entities)} entities to cache for {resource.title}"
            )

        return entities

    @staticmethod
    def _merge_entities_into_state(
        doc_entities: List[EntityWithQuotes], state: ExtractionStateV3
    ) -> None:
        for entity in doc_entities:
            if entity.name in state.entities_found:
                existing = state.entities_found[entity.name]
                existing.quotes.extend(entity.quotes)
                for alias in entity.aliases:
                    if alias not in existing.aliases:
                        existing.aliases.append(alias)
                logger.debug(
                    f"Merged entity '{entity.name}' - now has "
                    f"{len(existing.quotes)} quotes"
                )
            else:
                state.entities_found[entity.name] = entity
                logger.debug(
                    f"Added new entity '{entity.name}' with {len(entity.quotes)} quotes"
                )


# ---------------------------------------------------------------------------
# AssessIndividually
# ---------------------------------------------------------------------------


@dataclass
class AssessIndividually(BaseNode[ExtractionStateV3, ExtractionDepsV3]):
    """
    Assess each entity's relationship potential using extraction_core.
    """

    async def run(
        self, ctx: GraphRunContext[ExtractionStateV3, ExtractionDepsV3]
    ) -> "GeneratePairCandidates":
        state, deps = ctx.state, ctx.deps

        logger.info(
            f"Starting individual assessment of {len(state.entities_found)} entities"
        )

        tasks = []
        entity_names = []
        for entity_name, entity in state.entities_found.items():
            tasks.append(self._assess_one_entity(entity, ctx))
            entity_names.append(entity_name)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for name, result in zip(entity_names, results):
            if isinstance(result, Exception):
                logger.error(f"Assessment failed for {name}: {result}")
                continue
            if isinstance(result, IndividualAssessment):
                state.individual_assessments[name] = result

        logger.info(
            f"Assessment complete: {len(state.individual_assessments)} entities assessed"
        )
        logger.info(
            f"Cache stats - hits: {state.metrics.cache_hits_assessment}, "
            f"misses: {state.metrics.cache_misses_assessment}"
        )

        if hasattr(deps, "checkpoint_callback") and deps.checkpoint_callback:
            await deps.checkpoint_callback("assessment", state)

        return GeneratePairCandidates()

    async def _assess_one_entity(
        self,
        entity: EntityWithQuotes,
        ctx: GraphRunContext[ExtractionStateV3, ExtractionDepsV3],
    ) -> IndividualAssessment:
        state, deps = ctx.state, ctx.deps
        cache = state.cache

        cache_key = compute_assessment_cache_key(
            entity_name=entity.name,
            entity_kind=entity.kind,
            task_context=deps.get_task_context(),
            model_version=str(deps.model),
            prompt_version="v3_2025-10",
        )

        if cache and deps.semantic_cache_enabled:
            cached = await cache.get_assessment(cache_key)
            if cached is not None:
                state.metrics.record_cache_hit("assessment")
                logger.debug(f"Cache hit for {entity.name}")
                return cached

            state.metrics.record_cache_miss("assessment")

        analysis_config = AnalysisConfig(
            analysis_type="relationship_potential",
            target_context=deps.get_task_context(),
            max_contexts=10,
            model=deps.model,
        )

        similarity_threshold = (
            deps.fuzzy_auto_correct_threshold if deps.fuzzy_matching_enabled else 0.90
        )

        start_time = time.time()
        try:
            assessment = await analyze_entity(
                entity=entity,
                analysis_config=analysis_config,
                similarity_threshold=similarity_threshold,
            )
            duration = time.time() - start_time
            state.metrics.record_assessment_call(success=True, duration=duration)
            logger.debug(
                f"Assessed {entity.name} in {duration:.2f}s "
                f"(potential={assessment.relationship_potential})"
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            duration = time.time() - start_time
            state.metrics.record_assessment_call(success=False, duration=duration)
            logger.error(f"Assessment failed for {entity.name}: {exc}")
            return IndividualAssessment(
                entity=entity,
                relationship_potential="none",
                related_entities=[],
                evidence_quotes=[],
                reasoning=f"Assessment failed: {exc}",
                confidence=0.0,
            )

        if cache and deps.semantic_cache_enabled:
            await cache.set_assessment(cache_key, assessment)
            logger.debug(f"Saved assessment to cache for {entity.name}")

        return assessment


# ---------------------------------------------------------------------------
# GeneratePairCandidates
# ---------------------------------------------------------------------------


@dataclass
class GeneratePairCandidates(BaseNode[ExtractionStateV3, ExtractionDepsV3]):
    """
    Generate candidate pairs from co-occurrence analysis and assessment suggestions.
    """

    async def run(
        self, ctx: GraphRunContext[ExtractionStateV3, ExtractionDepsV3]
    ) -> "EvaluatePairs":
        state, deps = ctx.state, ctx.deps

        logger.info(
            f"Starting pair candidate generation from {len(state.entities_found)} entities"
        )

        candidates: Dict[tuple[str, str], PairCandidate] = {}

        for candidate in self._find_cooccurrence_pairs(state, deps):
            key = self._make_pair_key(candidate.entity_a.name, candidate.entity_b.name)
            candidates[key] = candidate
            state.metrics.candidates_from_cooccurrence += 1

        for candidate in self._find_assessment_suggested_pairs(state, deps):
            key = self._make_pair_key(candidate.entity_a.name, candidate.entity_b.name)
            if key in candidates:
                candidates[key].generation_strategy = "both"
            else:
                candidates[key] = candidate
            state.metrics.candidates_from_assessment += 1

        if not deps.include_same_kind_pairs:
            before = len(candidates)
            candidates = {
                k: v
                for k, v in candidates.items()
                if v.entity_a.kind != v.entity_b.kind
            }
            filtered = before - len(candidates)
            if filtered:
                logger.info(f"Filtered {filtered} same-kind candidates")

        state.pair_candidates = candidates
        state.metrics.candidates_generated = len(candidates)
        logger.info(f"Total candidates generated: {len(candidates)}")

        if hasattr(deps, "checkpoint_callback") and deps.checkpoint_callback:
            await deps.checkpoint_callback("candidates", state)

        return EvaluatePairs()

    def _find_cooccurrence_pairs(
        self, state: ExtractionStateV3, deps: ExtractionDepsV3
    ) -> List["PairCandidate"]:
        allowed_levels: Set[str] = set()
        if deps.enable_same_chunk:
            allowed_levels.add("same_chunk")
        if deps.enable_adjacent_chunks:
            allowed_levels.add("adjacent_chunks")
        if deps.enable_document_level:
            allowed_levels.add("document_level")

        if not allowed_levels:
            return []

        if "document_level" in allowed_levels:
            strategy = CooccurrenceStrategy.TIERED
        elif "adjacent_chunks" in allowed_levels:
            strategy = CooccurrenceStrategy.ADJACENT
        else:
            strategy = CooccurrenceStrategy.SAME_CHUNK

        return find_cooccurring_pairs(
            entities=state.entities_found,
            resources=list(state.resource_pool.resources),
            strategy=strategy,
            allowed_levels=allowed_levels,
            include_same_kind_pairs=deps.include_same_kind_pairs,
        )

    def _find_assessment_suggested_pairs(
        self, state: ExtractionStateV3, deps: ExtractionDepsV3
    ) -> List["PairCandidate"]:
        from .models import PairCandidate

        candidates: List[PairCandidate] = []

        for entity_name, assessment in state.individual_assessments.items():
            if assessment.relationship_potential in {"low", "none"}:
                continue

            source_entity = state.entities_found.get(entity_name)
            if source_entity is None:
                continue

            for related in assessment.related_entities:
                target_entity = self._fuzzy_match_entity(related, state.entities_found)
                if target_entity is None:
                    logger.debug(
                        f"Could not match related entity '{related}' suggested by "
                        f"{entity_name}"
                    )
                    continue

                if (
                    source_entity.kind == target_entity.kind
                    and not deps.include_same_kind_pairs
                ):
                    continue

                key = self._make_pair_key(source_entity.name, target_entity.name)
                if key in state.pair_candidates:
                    continue

                candidates.append(
                    PairCandidate(
                        entity_a=source_entity,
                        entity_b=target_entity,
                        co_occurrence_count=0,
                        shared_resources=[],
                        generation_strategy="assessment_suggested",
                    )
                )

        return candidates

    @staticmethod
    def _make_pair_key(name_a: str, name_b: str) -> tuple[str, str]:
        a, b = sorted([name_a, name_b])
        return a, b

    @staticmethod
    def _fuzzy_match_entity(
        target_name: str, entities: Dict[str, EntityWithQuotes]
    ) -> Optional[EntityWithQuotes]:
        target_lower = target_name.lower()
        if target_lower in entities:
            return entities[target_lower]

        for entity in entities.values():
            if entity.name.lower() == target_lower or target_lower in [
                alias.lower() for alias in entity.aliases
            ]:
                return entity

        for entity in entities.values():
            if target_lower in entity.name.lower():
                return entity

        return None


# ---------------------------------------------------------------------------
# EvaluatePairs
# ---------------------------------------------------------------------------


@dataclass
class EvaluatePairs(BaseNode[ExtractionStateV3, ExtractionDepsV3, ExtractionStateV3]):
    """
    Evaluate candidate pairs using extraction_core.
    """

    async def run(
        self, ctx: GraphRunContext[ExtractionStateV3, ExtractionDepsV3]
    ) -> "BaseNode[ExtractionStateV3, ExtractionDepsV3, ExtractionStateV3] | End[ExtractionStateV3]":
        state, deps = ctx.state, ctx.deps

        logger.info(
            f"Starting pair evaluation of {len(state.pair_candidates)} candidates"
        )

        results = await with_parallelism_control(
            list(state.pair_candidates.values()),
            lambda candidate: self._evaluate_one_pair(candidate, ctx),
            parallelism=deps.pair_evaluation_parallelism,
            description="pair evaluation",
        )

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Pair evaluation failed: {result}")
                continue

            if result:
                state.final_pairs.append(result)

        logger.info(
            f"Pair evaluation complete: {state.metrics.pairs_accepted} accepted, "
            f"{state.metrics.pairs_rejected} rejected"
        )
        logger.info(
            f"Acceptance rate: {(state.metrics.pairs_accepted / state.metrics.pair_evaluation_calls * 100):.1f}%"
            if state.metrics.pair_evaluation_calls > 0
            else "Acceptance rate: N/A"
        )

        if hasattr(deps, "checkpoint_callback") and deps.checkpoint_callback:
            await deps.checkpoint_callback("final", state)

        return End(state)

    async def _evaluate_one_pair(
        self,
        candidate: PairCandidate,
        ctx: GraphRunContext[ExtractionStateV3, ExtractionDepsV3],
    ) -> Optional[EntityPairOut]:
        state, deps = ctx.state, ctx.deps

        evaluation_config = EvaluationConfig(
            relationship_type=deps.get_relation_type(),
            task_context=deps.get_task_context(),
            model=deps.model,
            require_shared_resources=False,
        )

        similarity_threshold = (
            deps.fuzzy_auto_correct_threshold if deps.fuzzy_matching_enabled else 0.90
        )

        start_time = time.time()
        try:
            pair = await evaluate_pair(
                entity_a=candidate.entity_a,
                entity_b=candidate.entity_b,
                config=evaluation_config,
                similarity_threshold=similarity_threshold,
            )
            duration = time.time() - start_time
            state.metrics.record_pair_evaluation(
                success=True, accepted=pair is not None, duration=duration
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            duration = time.time() - start_time
            state.metrics.record_pair_evaluation(
                success=False, accepted=False, duration=duration
            )
            logger.error(
                f"Evaluation failed for {candidate.entity_a.name} - "
                f"{candidate.entity_b.name}: {exc}"
            )
            return None

        if pair is None:
            logger.debug(
                f"Rejected pair {candidate.entity_a.name} - {candidate.entity_b.name}: "
                f"no relationship found"
            )
            return None

        if not pair.validate_provenance():
            logger.warning(
                f"Pair provenance validation failed: "
                f"{pair.entity_a.name} - {pair.entity_b.name}"
            )
            return None

        logger.debug(
            f"Accepted pair {pair.entity_a.name} - {pair.entity_b.name} "
            f"with {len(pair.evidence_quotes)} evidence quotes"
        )
        return pair
