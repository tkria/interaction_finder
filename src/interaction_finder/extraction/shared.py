"""Shared helpers for extraction pipeline stages.

Provides utilities used across multiple stages: checkpointing, evidence
aggregation, entity lookups, and statistics.
"""

from pathlib import Path
from statistics import median_low
from collections import Counter

from interaction_finder.checkpoint import ExtractionStageData, PipelineCheckpoint
from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.models import (
    ExtractionMetadata,
    ExtractionResult,
    EvidenceQuality,
    PairAssessment,
)
from interaction_finder.extraction.state import State
from interaction_finder.version import get_version_string


# --- Shared helpers used by multiple stages ---


def get_entity_aliases(canonical: str, state: State) -> list[str]:
    """Get aliases for an entity from the global index."""
    if ref := state.global_entities.get(canonical):
        return ref.aliases()
    return []


def aggregate_evidence(assessments: list[PairAssessment]) -> EvidenceQuality:
    """Aggregate evidence from multiple assessments into a single EvidenceQuality.

    Uses median for overall level (robust to outliers) and mode for factor values.
    For even counts, takes the lower of the two middle values (conservative).
    """
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


def aggregate_cache_stats(state: State) -> tuple[int, int]:
    """Aggregate cache hits and misses across all kinds."""
    hits = sum(c.hits for c in state.merge_cache_by_kind.values())
    misses = sum(c.misses for c in state.merge_cache_by_kind.values())
    return hits, misses


def snapshot_entity_counts(state: State) -> dict[str, dict[str, int]]:
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


async def save_checkpoint(state: State, deps: Deps, stage: str) -> None:
    """Save partial checkpoint with current state for resumption.

    Parameters:
        state: Current pipeline state
        deps: Pipeline dependencies
        stage: Name of the stage just completed
    """
    if not deps.checkpoint_path:
        return

    # Helper to count items in nested dicts
    def count_nested(d):
        return sum(len(v) for v in d.values())

    # Build partial checkpoint with resume state
    checkpoint = PipelineCheckpoint(
        topic=state.topic,
        resources=deps.resource_pool,
        created_by=get_version_string(),
        keywords=deps.input_checkpoint.keywords,
        search=deps.input_checkpoint.search,
        extraction=ExtractionStageData(
            target_entity_types=state.target_entity_types,
            permitted_pairs={k: list(v) for k, v in state.permitted_pairs.items()},
            judgments=[],
            consolidated=state.consolidated,
            paper_quality={
                rid.url: assessment for rid, assessment in state.paper_quality.items()
            },
            metadata=ExtractionMetadata(
                topic=state.topic,
                resource_count=len(deps.resource_pool.resources),
                total_entities_found=count_nested(state.entities_by_resource),
                entities_after_validation=count_nested(
                    state.validated_entities_by_resource
                ),
                entities_merged=state.entities_merged,
                merge_cache_hits=sum(
                    c.hits for c in state.merge_cache_by_kind.values()
                ),
                merge_cache_misses=sum(
                    c.misses for c in state.merge_cache_by_kind.values()
                ),
                proximal_sets_found=count_nested(state.proximal_sets_by_resource),
                total_pairs_found=0,
                pairs_accepted=0,
                pairs_rejected=0,
                quotes_validated=state.quotes_validated,
                quotes_failed=state.quotes_failed,
                resume_from=stage,
                resume_state=state.to_dict(),
            ),
        ),
    )
    Path(deps.checkpoint_path).write_text(checkpoint.model_dump_json(indent=2))
    deps.logger.info(f"Saved partial checkpoint after {stage}")


def empty_result(state: State, deps: Deps) -> ExtractionResult:
    """Create empty result for early termination."""
    return ExtractionResult(
        topic=state.topic,
        target_entity_types=state.target_entity_types,
        permitted_pairs={k: list(v) for k, v in state.permitted_pairs.items()},
        resources=deps.resource_pool,
        judgments=[],
        metadata=ExtractionMetadata(
            topic=state.topic,
            resource_count=len(deps.resource_pool.resources),
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
