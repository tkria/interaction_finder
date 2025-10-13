"""
Reverse search functionality for finding known resources via iterative queries.

This package provides models and algorithms for reverse search: given a set of
known resources (papers/documents), generate queries that will retrieve them
through standard search backends. Used for query evaluation and coverage analysis.

Main exports:
    ReverseSearcher: Main orchestrator for complete reverse search workflow
    KnownResource: Target resource representation with canonical URL
    ResourceMatch: Records how a resource matched to a search result
    ReverseSearchResult: Result of a single query execution
    ReverseSearchSession: Complete session with all queries and metrics
    ReverseSearchConfig: Configuration with validated constraints
    ReverseSearchError: Base error class and subclasses
    KeywordExtractor: Abstract interface for keyword extraction algorithms
    normalize_url: URL normalization for consistent matching
    ResourceMatcher: Multi-strategy resource matching
    QueryGenerator: Query generation from target resources
    KeywordExtractionResult: Stage 1 output (keyword extraction)
    QueryConstructionContext: Stage 2 input (query construction)
"""

from interaction_finder.search.reverse.models import (
    ConfigurationError,
    KeywordExtractionResult,
    KnownResource,
    MatchingError,
    QueryConstructionContext,
    QueryGenerationError,
    ResourceMatch,
    ResourceParseError,
    ReverseSearchConfig,
    ReverseSearchError,
    ReverseSearchResult,
    ReverseSearchSession,
)
from interaction_finder.search.reverse.keyword_extractors import (
    KeywordExtractor,
    YAKEExtractor,
    RAKEExtractor,
    TFIDFExtractor,
    create_extractor,
)
from interaction_finder.search.reverse.utils import normalize_url, TRACKING_PARAMS
from interaction_finder.search.reverse.matchers import ResourceMatcher
from interaction_finder.search.reverse.query_generator import QueryGenerator
from interaction_finder.search.reverse.searcher import ReverseSearcher

__all__ = [
    "ReverseSearcher",
    "KnownResource",
    "ResourceMatch",
    "ReverseSearchResult",
    "ReverseSearchSession",
    "ReverseSearchConfig",
    "ReverseSearchError",
    "ResourceParseError",
    "QueryGenerationError",
    "MatchingError",
    "ConfigurationError",
    "KeywordExtractor",
    "YAKEExtractor",
    "RAKEExtractor",
    "TFIDFExtractor",
    "create_extractor",
    "normalize_url",
    "TRACKING_PARAMS",
    "ResourceMatcher",
    "QueryGenerator",
    "KeywordExtractionResult",
    "QueryConstructionContext",
]
