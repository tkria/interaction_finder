"""
Unified extraction core for gene-disease and protein-protein relationship extraction.

This module provides a pure functional API for extracting, analyzing, and evaluating
biological entities and their relationships from scientific literature. It supports
two primary workflows:

1. Gene-Disease Association (Single-Entity Focus):
   - Extract entities from documents
   - Analyze individual entities for disease/condition associations
   - Generate relationship descriptions from analysis

2. Protein-Protein Relationships (Pairwise Focus):
   - Extract entities from documents
   - Form entity pairs via co-occurrence analysis
   - Evaluate pairs for specific relationship types (interaction, regulation, etc.)

Architecture:
- Pure functions with explicit configuration
- Pydantic models for type safety and validation
- ResourceQuote integration for complete provenance tracking
- Designed for testability, reusability, and clarity

Core Functions:
- extract_from_resource: Extract entities from a single document
- merge_entities: Deduplicate and combine entities across documents
- analyze_entity: Assess entity relevance/associations (gene-disease workflow)
- find_cooccurring_pairs: Generate entity pairs via co-occurrence (relationship workflow)
- evaluate_pair: Assess pair relationships with LLM (relationship workflow)
"""

from .models import (
    AnalysisConfig,
    EvaluationConfig,
    CooccurrenceStrategy,
)

from .extraction import extract_from_resource
from .merging import merge_entities
from .analysis import analyze_entity
from .pairing import find_cooccurring_pairs
from .evaluation import evaluate_pair

__all__ = [
    # Configuration models
    "AnalysisConfig",
    "EvaluationConfig",
    "CooccurrenceStrategy",
    # Core functions
    "extract_from_resource",
    "merge_entities",
    "analyze_entity",
    "find_cooccurring_pairs",
    "evaluate_pair",
]
