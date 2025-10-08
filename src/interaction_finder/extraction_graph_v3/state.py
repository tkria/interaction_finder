"""
State management for extraction graph V3.

Extends V2's minimal state pattern with:
- Semantic cache integration (optional)
- Pair candidate tracking (Dict for O(1) lookup)
- Enhanced metrics for caching and pair evaluation
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

# Import V2 models for reuse
from ..extraction_graph_v2.models import (
    EntityWithQuotes,
    IndividualAssessment,
    EntityPairOut,
)
from ..extraction_graph_v2.state import ExtractionMetrics
from ..resources import ResourcePool
from .models import PairCandidate

if TYPE_CHECKING:
    from .cache import SemanticCacheManager


@dataclass
class ExtractionMetricsV3(ExtractionMetrics):
    """
    Extended metrics for V3 with cache tracking and pair evaluation.

    Extends V2 metrics with:
    - Cache hit/miss tracking for extraction and assessment
    - Pair candidate generation metrics
    - Pair evaluation metrics
    """

    # Cache performance (extraction stage)
    cache_hits_extraction: int = 0
    cache_misses_extraction: int = 0

    # Cache performance (assessment stage)
    cache_hits_assessment: int = 0
    cache_misses_assessment: int = 0

    # Pair candidate generation
    candidates_generated: int = 0
    candidates_from_cooccurrence: int = 0
    candidates_from_assessment: int = 0

    # Pair evaluation
    pair_evaluation_calls: int = 0
    pair_evaluation_successes: int = 0
    pairs_accepted: int = 0
    pairs_rejected: int = 0

    # Timing (additional to V2 extraction_time and assessment_time)
    candidate_generation_time: float = 0.0
    pair_evaluation_time: float = 0.0

    def record_cache_hit(self, stage: str) -> None:
        """Record a cache hit for the given stage."""
        if stage == "extraction":
            self.cache_hits_extraction += 1
        elif stage == "assessment":
            self.cache_hits_assessment += 1

    def record_cache_miss(self, stage: str) -> None:
        """Record a cache miss for the given stage."""
        if stage == "extraction":
            self.cache_misses_extraction += 1
        elif stage == "assessment":
            self.cache_misses_assessment += 1

    def record_candidate_generated(self, strategy: str) -> None:
        """Record a pair candidate generation."""
        self.candidates_generated += 1
        if strategy in ["same_chunk", "adjacent_chunks", "document_level"]:
            self.candidates_from_cooccurrence += 1
        elif strategy == "assessment_suggested":
            self.candidates_from_assessment += 1

    def record_pair_evaluation(
        self, success: bool, accepted: bool, duration: float = 0.0
    ) -> None:
        """Record a pair evaluation call."""
        self.pair_evaluation_calls += 1
        if success:
            self.pair_evaluation_successes += 1
        if accepted:
            self.pairs_accepted += 1
        else:
            self.pairs_rejected += 1
        self.pair_evaluation_time += duration

    def get_cache_hit_rate(self) -> Dict[str, float]:
        """
        Calculate cache hit rates as percentages.

        Returns:
            Dictionary with hit rates for each stage
        """
        extraction_total = self.cache_hits_extraction + self.cache_misses_extraction
        assessment_total = self.cache_hits_assessment + self.cache_misses_assessment

        extraction_rate = (
            (self.cache_hits_extraction / extraction_total * 100)
            if extraction_total > 0
            else 0.0
        )
        assessment_rate = (
            (self.cache_hits_assessment / assessment_total * 100)
            if assessment_total > 0
            else 0.0
        )

        # Overall rate across all stages
        total_hits = self.cache_hits_extraction + self.cache_hits_assessment
        total_requests = extraction_total + assessment_total
        overall_rate = (
            (total_hits / total_requests * 100) if total_requests > 0 else 0.0
        )

        return {
            "extraction_hit_rate": extraction_rate,
            "assessment_hit_rate": assessment_rate,
            "overall_hit_rate": overall_rate,
        }

    def get_pair_metrics(self) -> Dict[str, Any]:
        """Get pair-related metrics."""
        evaluation_rate = (
            (self.pair_evaluation_successes / self.pair_evaluation_calls * 100)
            if self.pair_evaluation_calls > 0
            else 0.0
        )
        acceptance_rate = (
            (self.pairs_accepted / self.pair_evaluation_calls * 100)
            if self.pair_evaluation_calls > 0
            else 0.0
        )

        return {
            "candidates_generated": self.candidates_generated,
            "candidates_from_cooccurrence": self.candidates_from_cooccurrence,
            "candidates_from_assessment": self.candidates_from_assessment,
            "evaluation_calls": self.pair_evaluation_calls,
            "evaluation_success_rate": evaluation_rate,
            "pairs_accepted": self.pairs_accepted,
            "pairs_rejected": self.pairs_rejected,
            "acceptance_rate": acceptance_rate,
        }


@dataclass
class ExtractionStateV3:
    """
    State for V3 extraction pipeline with semantic caching and pair tracking.

    Exactly 7 fields (design constraint from spec):
    1. resource_pool: Document management
    2. entities_found: Dict for O(1) lookup during pairing
    3. individual_assessments: Dict for O(1) lookup by entity name
    4. pair_candidates: Dict to prevent duplicate candidate generation
    5. final_pairs: List for final output
    6. cache: Optional semantic cache manager (None until task 02)
    7. metrics: Enhanced metrics with cache and pair tracking
    """

    # Document management
    resource_pool: ResourcePool = field(default_factory=ResourcePool)

    # Extracted entities (Dict for O(1) lookup)
    entities_found: Dict[str, EntityWithQuotes] = field(default_factory=dict)

    # Individual assessments (Dict for O(1) lookup by entity name)
    individual_assessments: Dict[str, IndividualAssessment] = field(
        default_factory=dict
    )

    # Pair candidates (Dict with tuple key for deduplication)
    pair_candidates: Dict[tuple[str, str], PairCandidate] = field(default_factory=dict)

    # Final output pairs
    final_pairs: List[EntityPairOut] = field(default_factory=list)

    # Semantic cache manager (optional, implemented in task 02)
    cache: Optional["SemanticCacheManager"] = None

    # Enhanced metrics
    metrics: ExtractionMetricsV3 = field(default_factory=ExtractionMetricsV3)

    def get_entity_count(self) -> int:
        """Get total number of entities found."""
        return len(self.entities_found)

    def get_assessment_count(self) -> int:
        """Get number of completed assessments."""
        return len(self.individual_assessments)

    def get_candidate_count(self) -> int:
        """Get number of pair candidates generated."""
        return len(self.pair_candidates)

    def get_pairs_count(self) -> int:
        """Get number of final pairs generated."""
        return len(self.final_pairs)

    def validate_provenance_chain(self) -> bool:
        """
        Validate complete provenance chain exists.

        Returns:
            True if all entities have quotes and pairs have evidence
        """
        # All entities must have valid quotes
        for entity in self.entities_found.values():
            if not entity.validate():
                return False

        # All pairs must have evidence quotes
        for pair in self.final_pairs:
            if not pair.validate_provenance():
                return False

        return True

    def get_summary(self) -> Dict[str, Any]:
        """Get processing summary with V3 metrics."""
        base_summary = {
            "documents": len(self.resource_pool.resources),
            "entities": self.get_entity_count(),
            "assessments": self.get_assessment_count(),
            "candidates": self.get_candidate_count(),
            "pairs": self.get_pairs_count(),
        }

        # Add V3-specific metrics
        success_rates = self.metrics.get_success_rates()
        cache_rates = self.metrics.get_cache_hit_rate()
        pair_metrics = self.metrics.get_pair_metrics()

        base_summary.update(
            {
                "agent_calls": {
                    "extraction_calls": self.metrics.entities_extraction_calls,
                    "assessment_calls": self.metrics.assessment_calls,
                    "pair_evaluation_calls": self.metrics.pair_evaluation_calls,
                },
                "success_rates": success_rates,
                "cache_performance": cache_rates,
                "pair_metrics": pair_metrics,
                "quality_metrics": {
                    "entities_with_valid_quotes": self.metrics.entities_with_valid_quotes,
                    "evidence_retrieval": {
                        "requested": self.metrics.evidence_quotes_requested,
                        "found": self.metrics.evidence_quotes_found,
                    },
                },
                "timing": {
                    "total_duration": self.metrics.get_total_duration(),
                    "extraction_time": self.metrics.extraction_time,
                    "assessment_time": self.metrics.assessment_time,
                    "candidate_generation_time": self.metrics.candidate_generation_time,
                    "pair_evaluation_time": self.metrics.pair_evaluation_time,
                },
            }
        )

        return base_summary

    def get_provenance_metrics(self) -> Dict[str, float]:
        """
        Calculate provenance quality metrics.

        Returns:
            Dictionary with various provenance rates
        """
        if not self.entities_found:
            return {"entity_quote_rate": 0.0, "pair_evidence_rate": 0.0}

        # Entity quote rate
        entities_with_quotes = sum(
            1 for entity in self.entities_found.values() if entity.validate()
        )
        entity_quote_rate = entities_with_quotes / len(self.entities_found) * 100

        # Pair evidence rate
        pair_evidence_rate = 0.0
        if self.final_pairs:
            pairs_with_evidence = sum(
                1 for pair in self.final_pairs if pair.validate_provenance()
            )
            pair_evidence_rate = pairs_with_evidence / len(self.final_pairs) * 100

        return {
            "entity_quote_rate": entity_quote_rate,
            "pair_evidence_rate": pair_evidence_rate,
        }
