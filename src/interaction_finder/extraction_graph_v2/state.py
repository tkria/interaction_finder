"""
Minimal state management for extraction graph V2.

Dramatically simplified from the original 15+ fields to just 4 essential fields,
following the pydantic-graph pattern of centralized run-time memory.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time

from .models import EntityWithQuotes, IndividualAssessment, EntityPairOut
from ..resources import ResourcePool, ResourceId


@dataclass
class ExtractionMetrics:
    """
    Essential metrics for Phase 3 LLM-powered extraction.

    Tracks performance and quality metrics for real agent usage.
    """

    # Agent performance
    entities_extraction_calls: int = 0
    entities_extraction_successes: int = 0
    assessment_calls: int = 0
    assessment_successes: int = 0

    # Quality metrics
    entities_with_valid_quotes: int = 0
    evidence_quotes_found: int = 0
    evidence_quotes_requested: int = 0

    # Timing (seconds)
    start_time: Optional[float] = None
    extraction_time: float = 0.0
    assessment_time: float = 0.0

    def record_extraction_call(self, success: bool, duration: float = 0.0) -> None:
        """Record entity extraction agent call."""
        self.entities_extraction_calls += 1
        if success:
            self.entities_extraction_successes += 1
        self.extraction_time += duration

    def record_assessment_call(self, success: bool, duration: float = 0.0) -> None:
        """Record assessment agent call."""
        self.assessment_calls += 1
        if success:
            self.assessment_successes += 1
        self.assessment_time += duration

    def record_entity_validation(self, entity: EntityWithQuotes) -> None:
        """Record entity quote validation results."""
        if entity.validate():
            self.entities_with_valid_quotes += 1

    def record_evidence_search(self, requested: int, found: int) -> None:
        """Record evidence quote search results."""
        self.evidence_quotes_requested += requested
        self.evidence_quotes_found += found

    def get_success_rates(self) -> Dict[str, float]:
        """Calculate success rates as percentages."""
        extraction_rate = (
            (self.entities_extraction_successes / self.entities_extraction_calls * 100)
            if self.entities_extraction_calls > 0
            else 0.0
        )
        assessment_rate = (
            (self.assessment_successes / self.assessment_calls * 100)
            if self.assessment_calls > 0
            else 0.0
        )
        evidence_rate = (
            (self.evidence_quotes_found / self.evidence_quotes_requested * 100)
            if self.evidence_quotes_requested > 0
            else 0.0
        )

        return {
            "extraction_success_rate": extraction_rate,
            "assessment_success_rate": assessment_rate,
            "evidence_retrieval_rate": evidence_rate,
        }

    def start_timing(self) -> None:
        """Start timing the extraction process."""
        self.start_time = time.time()

    def get_total_duration(self) -> float:
        """Get total duration since start (in seconds)."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time


@dataclass
class ExtractionState:
    """
    Minimal shared state for the extraction pipeline.

    Only tracks essential information needed across nodes,
    dramatically reduced from the original bloated implementation.
    """

    # Document management
    resource_pool: ResourcePool = field(default_factory=ResourcePool)

    # Document grouping support - groups as ResourceId references
    document_groups: List[List[ResourceId]] = field(default_factory=list)
    current_group_index: int = 0

    # Extracted entities with their quotes (key insight: preserve provenance)
    entities_found: Dict[str, EntityWithQuotes] = field(default_factory=dict)

    # Individual assessments (key pattern: assess each entity separately)
    individual_assessments: List[IndividualAssessment] = field(default_factory=list)

    # Final output pairs with complete provenance
    final_pairs: List[EntityPairOut] = field(default_factory=list)

    # Phase 3 metrics tracking (essential for LLM performance monitoring)
    metrics: ExtractionMetrics = field(default_factory=ExtractionMetrics)

    def get_entity_count(self) -> int:
        """Get total number of entities found."""
        return len(self.entities_found)

    def get_assessment_count(self) -> int:
        """Get number of completed assessments."""
        return len(self.individual_assessments)

    def get_pairs_count(self) -> int:
        """Get number of final pairs generated."""
        return len(self.final_pairs)

    def get_current_group_resources(self) -> List:
        """Get Resource objects for the current group."""
        if not self.document_groups or self.current_group_index >= len(
            self.document_groups
        ):
            # Fallback: return all resources if no grouping
            return list(self.resource_pool.resources)

        current_group = self.document_groups[self.current_group_index]
        return [self.resource_pool.get(resource_id) for resource_id in current_group]

    def has_more_groups(self) -> bool:
        """Check if there are more groups to process."""
        return self.current_group_index < len(self.document_groups) - 1

    def advance_to_next_group(self) -> bool:
        """
        Advance to the next group.

        Returns:
            True if advanced to next group, False if no more groups
        """
        if self.has_more_groups():
            self.current_group_index += 1
            return True
        return False

    def get_summary(self) -> Dict[str, any]:
        """Get processing summary with Phase 3 metrics."""
        base_summary = {
            "documents": len(self.resource_pool.resources),
            "groups": len(self.document_groups),
            "current_group": self.current_group_index,
            "entities": self.get_entity_count(),
            "assessments": self.get_assessment_count(),
            "pairs": self.get_pairs_count(),
        }

        # Add Phase 3 metrics
        success_rates = self.metrics.get_success_rates()
        base_summary.update(
            {
                "agent_calls": {
                    "extraction_calls": self.metrics.entities_extraction_calls,
                    "assessment_calls": self.metrics.assessment_calls,
                },
                "success_rates": success_rates,
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
                },
            }
        )

        return base_summary

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
