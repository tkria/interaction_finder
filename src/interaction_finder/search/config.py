"""
Configuration models for search functionality.

This module defines Pydantic models for configuring different search backends
and query expansion strategies.
"""

from typing import List, Dict, Any, Optional, Literal, TYPE_CHECKING
from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from interaction_finder.search.reverse.models import ReverseSearchConfig


class PubMedConfig(BaseModel):
    """Configuration for PubMed/NCBI E-utilities search backend."""

    email: Optional[str] = Field(
        None,
        description="Email address for NCBI E-utilities (recommended for higher rate limits)",
    )
    api_key: Optional[str] = Field(
        None, description="NCBI API key for higher rate limits"
    )
    rate_limit: float = Field(
        3.0,
        description="Requests per second (3 without API key, 10 with API key)",
        ge=0.1,
        le=10.0,
    )
    timeout: int = Field(30, description="Request timeout in seconds", ge=5, le=120)
    retmode: str = Field("xml", description="Return format for search results")
    use_mesh: bool = Field(True, description="Enable MeSH term automatic mapping")


class EuropePMCConfig(BaseModel):
    """Configuration for Europe PMC search backend."""

    rate_limit: float = Field(
        10.0, description="Requests per second for Europe PMC", ge=0.1, le=20.0
    )
    timeout: int = Field(30, description="Request timeout in seconds", ge=5, le=120)
    include_citations: bool = Field(
        True, description="Include citation count in results"
    )
    include_fulltext_urls: bool = Field(
        True, description="Include full text URLs when available"
    )


class SemanticScholarConfig(BaseModel):
    """Configuration for Semantic Scholar API backend."""

    api_key: Optional[str] = Field(
        None, description="Semantic Scholar API key for higher rate limits"
    )
    rate_limit: float = Field(
        1.0,
        description="Requests per second (1 without API key, 100 with API key)",
        ge=0.1,
        le=100.0,
    )
    timeout: int = Field(30, description="Request timeout in seconds", ge=5, le=120)
    include_citations: bool = Field(True, description="Include citation information")
    include_references: bool = Field(
        False, description="Include reference lists (expensive)"
    )


class PerplexicaConfig(BaseModel):
    """Configuration for Perplexica local search backend."""

    base_url: str = Field(
        "http://localhost:3000", description="Base URL for Perplexica API"
    )
    timeout: int = Field(
        60,
        description="Request timeout in seconds (Perplexica can be slow)",
        ge=10,
        le=300,
    )
    search_mode: str = Field(
        "webSearch",
        description="Search mode for Perplexica (webSearch, academicSearch, etc.)",
    )
    max_sources: int = Field(
        20, description="Maximum sources to retrieve per query", ge=5, le=50
    )


class OpenAISearchConfig(BaseModel):
    """Configuration for OpenAI web search backend."""

    api_key: Optional[str] = Field(None, description="OpenAI API key for web search")
    base_url: str = Field(
        "https://api.openai.com/v1", description="OpenAI API base URL"
    )
    timeout: int = Field(60, description="Request timeout in seconds", ge=10, le=300)
    model: str = Field("gpt-4o-mini", description="Model to use for web search")
    reasoning_effort: str = Field(
        "low", description="Reasoning effort: low, medium, high"
    )
    allowed_domains: List[str] = Field(
        default_factory=list, description="List of allowed domains (max 20)"
    )
    user_location: Dict[str, Any] = Field(
        default_factory=dict, description="User location for geo-specific results"
    )


class HybridConfig(BaseModel):
    """Configuration for hybrid multi-backend search."""

    backends: List[str] = Field(
        default_factory=lambda: ["pubmed", "europepmc"],
        description="List of backend names to use in hybrid search",
    )
    backend_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Relative weights for different backends (for relevance scoring)",
    )
    deduplicate: bool = Field(
        True, description="Remove duplicate results across backends"
    )
    merge_strategy: str = Field(
        "weighted_round_robin",
        description="Strategy for merging results: 'weighted_round_robin', 'relevance_based', 'backend_priority'",
    )


class ReviewInformedConfig(BaseModel):
    """Configuration for review-informed search mode."""

    target_reviews: int = Field(
        3, description="Target number of review papers to analyze", ge=1, le=10
    )
    include_reviews_in_results: bool = Field(
        True, description="Include review papers in final results"
    )


class PubMedMeshExpansionConfig(BaseModel):
    """Configuration for PubMed MeSH co-occurrence expansion."""

    max_seed_results: int = Field(
        100,
        description="Maximum PMIDs to analyze for MeSH co-occurrence",
        ge=10,
        le=500,
    )
    top_terms: int = Field(
        5, description="Number of top MeSH terms to include in expansion", ge=1, le=20
    )
    tree_filters: List[str] = Field(
        default_factory=lambda: ["C", "D"],
        description="MeSH tree number prefixes to filter (e.g., C=Diseases, D=Drugs)",
    )
    min_frequency: int = Field(
        3, description="Minimum co-occurrence count to include a term", ge=1, le=50
    )


class QueryExpansionConfig(BaseModel):
    """Configuration for query expansion strategies."""

    enabled: bool = Field(False, description="Enable query expansion")
    min_expansion_terms: int = Field(
        1, description="Minimum number of expansion terms to generate", ge=0, le=20
    )
    max_expansion_terms: int = Field(
        10, description="Maximum number of expansion terms to add", ge=0, le=50
    )
    min_confidence: float = Field(
        0.5,
        description="Minimum confidence threshold for expansion terms",
        ge=0.0,
        le=1.0,
    )
    search_strategy: Literal[
        "single_query", "multi_query", "review_informed", "pubmed_mesh"
    ] = Field(
        "single_query",
        description="Search strategy: 'single_query' (combine all terms), 'multi_query' (separate focused searches), 'review_informed' (use review papers to inform queries), or 'pubmed_mesh' (LLM decomposition + MeSH expansion per component)",
    )
    use_mesh_expansion: bool = Field(
        False,
        description="Apply PubMed MeSH co-occurrence expansion to queries in other strategies (only works with PubMed backend)",
    )
    target_searches: int = Field(
        8,
        description="Number of diverse searches to perform in multi_query mode",
        ge=2,
        le=20,
    )

    # Advanced expansion parameters (consolidated from removed AdvancedExpansionConfig)
    model_name: str = Field(
        "openai:gpt-4o-mini", description="LLM model to use for expansion"
    )
    temperature: float = Field(
        0.3, description="LLM temperature for expansion", ge=0.0, le=1.0
    )

    # Review-informed search configuration
    review_informed: ReviewInformedConfig = Field(default_factory=ReviewInformedConfig)

    # PubMed MeSH expansion configuration
    pubmed_mesh: PubMedMeshExpansionConfig = Field(
        default_factory=PubMedMeshExpansionConfig
    )


class SearchCacheConfig(BaseModel):
    """Configuration for search result caching."""

    enabled: bool = Field(True, description="Enable search result caching")
    ttl_hours: int = Field(
        24,
        description="Time-to-live for cached search results in hours",
        ge=1,
        le=168,  # 1 week
    )
    max_cache_size_mb: int = Field(
        100, description="Maximum cache size in megabytes", ge=10, le=1000
    )
    cache_key_strategy: str = Field(
        "query_hash",
        description="Strategy for generating cache keys: 'query_hash', 'normalized_query'",
    )


class SearchConfig(BaseModel):
    """Main configuration for search functionality."""

    backend: str = Field("pubmed", description="Search backend to use")
    max_results: int = Field(
        100, description="Default maximum results per search", ge=1, le=1000
    )

    # Backend configurations
    pubmed: PubMedConfig = Field(default_factory=PubMedConfig)
    europepmc: EuropePMCConfig = Field(default_factory=EuropePMCConfig)
    semantic_scholar: SemanticScholarConfig = Field(
        default_factory=SemanticScholarConfig
    )
    perplexica: PerplexicaConfig = Field(default_factory=PerplexicaConfig)
    openai_search: OpenAISearchConfig = Field(default_factory=OpenAISearchConfig)
    hybrid: HybridConfig = Field(default_factory=HybridConfig)

    # Query expansion and caching
    expansion: QueryExpansionConfig = Field(default_factory=QueryExpansionConfig)
    cache: SearchCacheConfig = Field(default_factory=SearchCacheConfig)

    # Reverse search
    reverse: Optional[Any] = Field(
        default=None,
        description="Reverse search configuration (query generation for known resources)",
    )

    # Global settings
    concurrent_backends: int = Field(
        2, description="Maximum number of backends to query concurrently", ge=1, le=5
    )
    global_timeout: int = Field(
        120,
        description="Global timeout for all search operations in seconds",
        ge=30,
        le=600,
    )

    @model_validator(mode="after")
    def ensure_reverse_config(self) -> "SearchConfig":
        """Ensure reverse config is initialized with defaults if None."""
        if self.reverse is None:
            # Import here to avoid circular import
            try:
                from interaction_finder.search.reverse.models import ReverseSearchConfig

                self.reverse = ReverseSearchConfig()
            except ImportError:
                # Reverse search module not available, leave as None
                pass
        return self

    def get_backend_config(self, backend_name: str) -> Dict[str, Any]:
        """Get configuration dictionary for a specific backend."""
        backend_configs = {
            "pubmed": self.pubmed.model_dump(),
            "europepmc": self.europepmc.model_dump(),
            "semantic_scholar": self.semantic_scholar.model_dump(),
            "perplexica": self.perplexica.model_dump(),
            "openai_search": self.openai_search.model_dump(),
            "hybrid": self.hybrid.model_dump(),
        }

        if backend_name not in backend_configs:
            raise ValueError(f"Unknown backend: {backend_name}")

        return backend_configs[backend_name]

    def get_available_backends(self) -> List[str]:
        """Get list of all configured backends."""
        return ["pubmed", "perplexica", "openai_search"]

    def is_backend_available(self, backend_name: str) -> bool:
        """Check if a specific backend is available."""
        if backend_name == "pubmed":
            # PubMed is always available (no API key required)
            return True
        elif backend_name == "perplexica":
            # Perplexica requires local installation - will be checked at runtime
            return True  # Assume available, let connection attempt determine
        elif backend_name == "openai_search":
            # OpenAI search requires API key - will be checked at runtime
            return True  # Assume available, let API call determine
        else:
            return backend_name in ["pubmed", "perplexica", "openai_search"]
