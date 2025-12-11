"""Build final output with all judgments and metadata."""

from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.models import ExtractionMetadata, ExtractionResult
from interaction_finder.extraction.shared import (
    aggregate_cache_stats,
    snapshot_entity_counts,
)
from interaction_finder.extraction.state import State
from interaction_finder.logging import logfire


def finalize(state: State, deps: Deps) -> ExtractionResult:
    """Gather all judgments and build final result."""
    with logfire.span("finalize"):
        all_judgments = list(state.pair_judgments.values())
        # Calculate metadata
        total_entities_found = sum(
            len(entities) for entities in state.entities_by_resource.values()
        )
        entities_after_validation = sum(
            len(entities) for entities in state.validated_entities_by_resource.values()
        )
        proximal_sets_found = sum(
            len(sets) for sets in state.proximal_sets_by_resource.values()
        )
        total_pairs_found = len(all_judgments)
        pairs_accepted = sum(1 for j in all_judgments if j.accepted)
        pairs_rejected = total_pairs_found - pairs_accepted
        cache_hits, cache_misses = aggregate_cache_stats(state)
        metadata = ExtractionMetadata(
            topic=state.topic,
            resource_count=len(deps.resource_pool.resources),
            total_entities_found=total_entities_found,
            entities_after_validation=entities_after_validation,
            entities_merged=state.entities_merged,
            merge_cache_hits=cache_hits,
            merge_cache_misses=cache_misses,
            proximal_sets_found=proximal_sets_found,
            total_pairs_found=total_pairs_found,
            pairs_accepted=pairs_accepted,
            pairs_rejected=pairs_rejected,
            quotes_validated=state.quotes_validated,
            quotes_failed=state.quotes_failed,
        )
        # Finalize consolidated data
        state.consolidated.entities.final = snapshot_entity_counts(state)
        # Convert paper_quality keys from ResourceId to URL strings
        paper_quality_by_url = {
            rid.url: assessment for rid, assessment in state.paper_quality.items()
        }
        return ExtractionResult(
            topic=state.topic,
            target_entity_types=state.target_entity_types,
            permitted_pairs={k: list(v) for k, v in state.permitted_pairs.items()},
            resources=deps.resource_pool,
            judgments=all_judgments,
            metadata=metadata,
            consolidated=state.consolidated,
            entities=state.global_entities,
            paper_quality=paper_quality_by_url,
        )
