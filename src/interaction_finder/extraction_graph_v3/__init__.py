"""
Extraction graph V3: Enhanced pipeline with semantic caching and pair candidates.

V3 improvements over V2:
- Semantic caching for extraction and assessment results (task 02)
- Fuzzy quote matching for better quote validation (task 03)
- Explicit pair candidate tracking before evaluation (task 07)
- Agent factory pattern for cleaner agent construction (task 04)
- Checkpoint-based resumability (task 09)
- Enhanced metrics and provenance tracking

Key design principles:
- 7-field state (vs V2's 4 fields): adds pair_candidates, cache, enhanced metrics
- Dict-based lookups for O(1) access during candidate generation
- Separation of candidate generation from evaluation for caching
- Progressive disclosure: advanced features opt-in with sensible defaults
"""

# Core infrastructure (Task 01)
from .state import ExtractionStateV3, ExtractionMetricsV3
from .deps import ExtractionDepsV3
from .models import (
    PairCandidate,
    PairEvaluationOut,
    ExtractionMetadata,
    BatchExtractionResultV3,
)

# Agent factories (Task 04)
from .agents import (
    create_entity_extractor_v3,
    create_assessment_agent_v3,
    create_pair_evaluator_v3,
)

# Placeholder for main pipeline (implemented in later tasks)
# from .run import run_extraction_v3

__all__ = [
    # State management
    "ExtractionStateV3",
    "ExtractionMetricsV3",
    # Dependencies
    "ExtractionDepsV3",
    # Models
    "PairCandidate",
    "PairEvaluationOut",
    "ExtractionMetadata",
    "BatchExtractionResultV3",
    # Agent factories
    "create_entity_extractor_v3",
    "create_assessment_agent_v3",
    "create_pair_evaluator_v3",
    # Pipeline (placeholder)
    # "run_extraction_v3",
]
