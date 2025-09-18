"""
Configuration models for search functionality.

This module defines Pydantic models for configuring different search backends
and query expansion strategies.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


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
        "http://localhost:3001", description="Base URL for Perplexica API"
    )
    timeout: int = Field(
        60,
        description="Request timeout in seconds (Perplexica can be slow)",
        ge=10,
        le=300,
    )
    search_mode: str = Field(
        "academic", description="Search mode for Perplexica (academic, web, etc.)"
    )
    max_sources: int = Field(
        20, description="Maximum sources to retrieve per query", ge=5, le=50
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


class LLMExpansionConfig(BaseModel):
    """Configuration for LLM-based query expansion."""

    model_name: str = Field(
        "openai:gpt-4o-mini", description="LLM model to use for expansion"
    )
    max_terms: int = Field(
        15, description="Maximum expansion terms to generate", ge=1, le=30
    )
    min_confidence: float = Field(
        0.5, description="Minimum confidence threshold", ge=0.0, le=1.0
    )
    expansion_types: List[str] = Field(
        default_factory=lambda: ["synonyms", "related", "abbreviations"],
        description="Types of expansion to include: synonyms, related, abbreviations, broader, narrower",
    )
    domain_context: str = Field(
        "academic research", description="Domain context to guide expansion"
    )
    timeout_seconds: int = Field(
        30, description="LLM request timeout in seconds", ge=5, le=120
    )


class QueryExpansionConfig(BaseModel):
    """Configuration for query expansion strategies."""

    enabled: bool = Field(True, description="Enable query expansion")
    max_expansion_terms: int = Field(
        10, description="Maximum number of expansion terms to add", ge=0, le=50
    )
    min_confidence: float = Field(
        0.5,
        description="Minimum confidence threshold for expansion terms",
        ge=0.0,
        le=1.0,
    )

    # LLM configuration
    llm: LLMExpansionConfig = Field(default_factory=LLMExpansionConfig)

    def get_llm_config(self) -> Dict[str, Any]:
        """Get configuration dictionary for LLM expansion."""
        return self.llm.model_dump()


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

    enabled_backends: List[str] = Field(
        default_factory=lambda: ["pubmed"],
        description="List of enabled search backends",
    )
    default_backend: str = Field(
        "pubmed", description="Default backend to use for searches"
    )
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
    hybrid: HybridConfig = Field(default_factory=HybridConfig)

    # Query expansion and caching
    expansion: QueryExpansionConfig = Field(default_factory=QueryExpansionConfig)
    cache: SearchCacheConfig = Field(default_factory=SearchCacheConfig)

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

    def get_backend_config(self, backend_name: str) -> Dict[str, Any]:
        """Get configuration dictionary for a specific backend."""
        backend_configs = {
            "pubmed": self.pubmed.model_dump(),
            "europepmc": self.europepmc.model_dump(),
            "semantic_scholar": self.semantic_scholar.model_dump(),
            "perplexica": self.perplexica.model_dump(),
            "hybrid": self.hybrid.model_dump(),
        }

        if backend_name not in backend_configs:
            raise ValueError(f"Unknown backend: {backend_name}")

        return backend_configs[backend_name]

    def validate_backend_availability(self) -> Dict[str, bool]:
        """Check which backends are properly configured and available."""
        availability = {}

        for backend in self.enabled_backends:
            if backend == "pubmed":
                # PubMed is always available (no API key required)
                availability[backend] = True
            elif backend == "europepmc":
                # Europe PMC is always available (no API key required)
                availability[backend] = True
            elif backend == "semantic_scholar":
                # Semantic Scholar works without API key but with lower limits
                availability[backend] = True
            elif backend == "perplexica":
                # Perplexica requires local installation
                availability[backend] = False  # Will be checked at runtime
            elif backend == "hybrid":
                # Hybrid depends on other backends being available
                availability[backend] = (
                    len([b for b in self.hybrid.backends if b in self.enabled_backends])
                    > 0
                )
            else:
                availability[backend] = False

        return availability
