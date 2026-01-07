from __future__ import annotations

import re
from pathlib import Path
from typing import Any
import tomli
from copy import deepcopy
from pydantic import BaseModel, Field, field_validator


class StrictModel(BaseModel):
    """Base model that rejects unknown fields.

    All config models inherit from this to catch typos and invalid config keys.
    """

    model_config = {"extra": "forbid"}


def sanitize_topic_for_filename(topic: str, max_length: int = 80) -> str:
    """Convert a topic string to a safe filename component.

    Args:
        topic: Research topic string (e.g., "pulmonary arterial hypertension")
        max_length: Maximum length for the sanitized string

    Returns:
        Lowercase, hyphen-separated string safe for filenames
    """
    # Lowercase and replace whitespace/underscores with hyphens
    result = re.sub(r"[\s_]+", "-", topic.lower().strip())
    # Remove any characters that aren't alphanumeric or hyphens
    result = re.sub(r"[^a-z0-9-]", "", result)
    # Collapse multiple hyphens
    result = re.sub(r"-+", "-", result)
    # Strip leading/trailing hyphens
    result = result.strip("-")
    # Truncate if needed
    if len(result) > max_length:
        result = result[:max_length].rstrip("-")
    return result or "output"


class IfetcherConfig(StrictModel):
    """Configuration for interaction-finder with validation and path resolution.

    Load from TOML file with IfetcherConfig.from_path("config.toml").
    Override values via CLI: -O stage.search.max_rounds=5
    """

    _dir: Path | None = None  # Directory of the config file

    class AgentSpec(StrictModel):
        """LLM agent settings. Inheritance: agents._ → agents.<module>._ → agents.<module>.<agent>

        Modules and their agents:
        - keywords: query_expander, result_selector, keyword_evaluator, document_summarizer, reflector
        - search: goal_planner, query_generator, result_selector, reflector
        - extraction: document_analysis, proximal_pair, entity_consolidator, relationship_consolidator,
                      pair_judge, cross_judge, co_mention_region, entity_group_consolidation
        """

        model_config = {"extra": "forbid"}

        llm: str | None = Field(
            None,
            description="Model identifier as 'provider:model' (e.g., 'openai:gpt-4o', 'anthropic:claude-3-sonnet')",
        )
        expertise: str | None = Field(
            None,
            description="Domain context injected into system prompt (e.g., 'molecular biology')",
        )
        instruction: str | None = Field(
            None,
            description="Additional instructions appended to agent's system prompt",
        )
        retries: int | None = Field(
            None,
            description="Retry attempts on transient failures (rate limits, timeouts)",
            ge=0,
            le=10,
        )
        instrument: bool = Field(
            True,
            description="Log agent calls to Logfire for debugging and cost tracking",
        )
        system_prompt: str | None = Field(
            None,
            description="Complete system prompt override (replaces default prompt entirely)",
        )
        model_settings: dict[str, Any] | None = Field(
            None,
            description="Provider-specific options (e.g., {temperature: 0.7, parallel_tool_calls: false})",
        )

        def merge_with_parent(
            self, parent: "IfetcherConfig.AgentSpec | None"
        ) -> "IfetcherConfig.AgentSpec":
            """Merge this spec with a parent spec, preferring non-None values from self.

            Parameters:
                parent: Parent AgentSpec to inherit from (or None)

            Returns:
                New AgentSpec with merged values
            """
            if parent is None:
                return self
            # Create dict with parent values, override with self's non-None values
            merged_data = {}
            for field_name in IfetcherConfig.AgentSpec.model_fields:
                self_value = getattr(self, field_name)
                parent_value = getattr(parent, field_name)
                # Use self's value if not None, otherwise inherit from parent
                merged_data[field_name] = (
                    self_value if self_value is not None else parent_value
                )
            return IfetcherConfig.AgentSpec.model_validate(merged_data)

    agents: dict[str, dict[str, AgentSpec] | AgentSpec] | AgentSpec = Field(
        default_factory=dict,
        description="LLM agent config with inheritance: [agents._] for global defaults, [agents.extraction._] for module defaults, [agents.extraction.pair_judge] for specific agents",
    )

    class Tools(StrictModel):
        """Tool-specific configuration (algorithms, backends)."""

        class Crawl4AI(StrictModel):
            """Crawl4AI web scraper for fetching and converting web pages to markdown."""

            timeout: int = Field(
                30,
                description="HTTP request timeout for page fetching (seconds)",
                ge=1,
                le=600,
            )

        crawl4ai: Crawl4AI = Field(
            default_factory=Crawl4AI,
            description="Web scraper settings for HTML-to-markdown conversion",
        )

        class Fetcher(StrictModel):
            """Document fetching and full-text resolution."""

            class PubMed(StrictModel):
                """PubMed abstract pages often link to full-text. These settings control link-following."""

                follow_fulltext_links: bool = Field(
                    True,
                    description="Follow links from PubMed abstracts to full-text sources (PMC, publisher sites)",
                )
                max_concurrent_links: int = Field(
                    3,
                    ge=1,
                    le=10,
                    description="Parallel full-text link fetches per document",
                )
                content_improvement_threshold: float = Field(
                    0.5,
                    ge=0.0,
                    le=1.0,
                    description="Only use full-text if it has this much more content than abstract (0.5 = 50% longer)",
                )

            pubmed: PubMed = Field(
                default_factory=PubMed,
                description="Full-text link following for PubMed results",
            )

        fetcher: Fetcher = Field(
            default_factory=Fetcher,
            description="Document fetching behavior",
        )

        class Search(StrictModel):
            """Search backend configuration (PubMed, Perplexica, OpenAI)."""

            timeout: int = Field(
                60,
                description="HTTP timeout for search API calls (seconds)",
                ge=1,
                le=600,
            )

            class PerplexicaConfig(StrictModel):
                """Perplexica AI-powered search engine configuration."""

                class ModelSpec(StrictModel):
                    """LLM or embedding model specification for Perplexica."""

                    provider_id: str = Field(
                        description="Provider UUID from Perplexica's config",
                    )
                    key: str = Field(
                        description="Model key (e.g., 'gpt-4o-mini', 'text-embedding-3-large')",
                    )

                base_url: str = Field(
                    "http://127.0.0.1:3000",
                    description="Perplexica API base URL",
                )
                sources: list[str] = Field(
                    ["web"],
                    description="Search sources: 'web', 'discussions', 'academic'",
                )
                optimization_mode: str = Field(
                    "balanced",
                    description="Search depth: 'speed' (2 iterations), 'balanced' (6), 'quality' (25)",
                )
                # Model configs are optional - if not set, backend auto-discovers from /api/providers
                chat_model: ModelSpec | None = Field(
                    None,
                    description="LLM for search (auto-discovered if not set)",
                )
                embedding_model: ModelSpec | None = Field(
                    None,
                    description="Embedding model (auto-discovered if not set)",
                )

            perplexica: PerplexicaConfig = Field(
                default_factory=PerplexicaConfig,
                description="Perplexica AI search engine settings",
            )

            class PubMedConfig(StrictModel):
                """PubMed/NCBI E-utilities search configuration."""

                email: str | None = Field(
                    None,
                    description="Contact email for NCBI (recommended for better rate limits)",
                )
                api_key: str | None = Field(
                    None,
                    description="NCBI API key for 10 req/sec instead of 3 req/sec (or set NCBI_API_KEY env var)",
                )
                rate_limit: float = Field(
                    3.0,
                    ge=0.1,
                    le=10.0,
                    description="Max requests per second (3 without API key, 10 with)",
                )
                use_mesh: bool = Field(
                    True,
                    description="Enable MeSH term expansion for broader search results",
                )

            pubmed: PubMedConfig = Field(
                default_factory=PubMedConfig,
                description="PubMed/NCBI search settings",
            )

            class OpenAIConfig(StrictModel):
                """OpenAI web search configuration."""

                api_key: str | None = Field(
                    None,
                    description="OpenAI API key (or set OPENAI_API_KEY env var)",
                )
                base_url: str = Field(
                    "https://api.openai.com/v1",
                    description="OpenAI API base URL (for proxies or compatible APIs)",
                )
                model: str = Field(
                    "gpt-4o-mini",
                    description="Model to use for web search queries",
                )

            openai: OpenAIConfig = Field(
                default_factory=OpenAIConfig,
                description="OpenAI web search settings",
            )

        search: Search = Field(
            default_factory=Search,
            description="Search backend configuration",
        )

        class Keywords(StrictModel):
            """Keyword extraction algorithm configuration."""

            class RAKEConfig(StrictModel):
                """RAKE (Rapid Automatic Keyword Extraction) - fast, statistical phrase extraction."""

                min_length: int = Field(
                    1, ge=1, description="Minimum words per keyphrase"
                )
                max_length: int = Field(
                    4, ge=1, description="Maximum words per keyphrase"
                )

            class YAKEConfig(StrictModel):
                """YAKE (Yet Another Keyword Extractor) - unsupervised, position-aware extraction."""

                n_grams: int = Field(
                    3, ge=1, le=5, description="Maximum words per keyphrase"
                )
                deduplication_threshold: float = Field(
                    0.9,
                    ge=0.0,
                    le=1.0,
                    description="Similarity threshold for removing near-duplicate phrases",
                )
                window_size: int = Field(
                    1, ge=1, description="Co-occurrence window for word scoring"
                )

            class TFIDFConfig(StrictModel):
                """TF-IDF - term frequency weighting to find distinctive terms."""

                max_features: int = Field(
                    50, ge=1, description="Maximum unique terms to extract"
                )
                ngram_range: tuple[int, int] = Field(
                    (1, 3), description="(min, max) words per term"
                )
                min_df: int = Field(
                    1, ge=1, description="Minimum documents a term must appear in"
                )

            class KeyBERTConfig(StrictModel):
                """KeyBERT - BERT embeddings + MMR for diverse, semantically-relevant keywords."""

                model_name: str = Field(
                    "all-MiniLM-L6-v2",
                    description="Sentence-transformer model for embeddings",
                )
                diversity: float = Field(
                    0.5,
                    ge=0.0,
                    le=1.0,
                    description="MMR diversity: 0=most relevant, 1=most diverse keywords",
                )
                top_n: int = Field(
                    20, ge=1, description="Candidate keywords before MMR selection"
                )
                device: str | None = Field(
                    None,
                    description="Device for embeddings: 'cpu', 'cuda', or omit for auto-detect",
                )

            rake: RAKEConfig = Field(
                default_factory=RAKEConfig,
                description="RAKE: fast statistical phrase extraction",
            )
            yake: YAKEConfig = Field(
                default_factory=YAKEConfig,
                description="YAKE: position-aware keyword scoring",
            )
            tfidf: TFIDFConfig = Field(
                default_factory=TFIDFConfig,
                description="TF-IDF: distinctive term extraction",
            )
            keybert: KeyBERTConfig = Field(
                default_factory=KeyBERTConfig,
                description="KeyBERT: semantic keyword extraction with diversity",
            )

        keywords: Keywords = Field(
            default_factory=Keywords,
            description="Keyword extraction algorithm settings",
        )

    tools: Tools = Field(
        default_factory=Tools,
        description="Tool-specific configuration (algorithms, backends)",
    )

    class Stage(StrictModel):
        """Pipeline stage configuration: keywords → search → extraction."""

        class Keywords(StrictModel):
            """Bridging term extraction from review articles (Stage 1)."""

            max_rounds: int = Field(
                5,
                ge=1,
                le=10,
                description="Search iterations before stopping (each round fetches new documents)",
            )
            search_backend: str = Field(
                "perplexica",
                description="Backend for finding reviews: 'pubmed', 'perplexica', or 'openai'",
            )
            max_results_per_query: int = Field(
                20,
                ge=1,
                le=100,
                description="Search results to retrieve per query",
            )
            max_documents_to_fetch: int = Field(
                10,
                ge=1,
                le=50,
                description="Documents to fetch full-text for per round",
            )
            max_keywords_per_method: int = Field(
                30,
                ge=5,
                le=100,
                description="Keywords extracted per algorithm (RAKE, YAKE, etc.) before merging",
            )
            max_keywords_for_llm: int = Field(
                50,
                ge=10,
                le=100,
                description="Keywords shown to LLM for evaluation after deduplication",
            )
            rerank_top_k: int = Field(
                0,
                ge=0,
                le=200,
                description="Use semantic reranking to select top-k results (0 = skip reranking, send all to LLM)",
            )
            reranker_model: str = Field(
                "zeroentropy/zerank-1-small",
                description="HuggingFace model for semantic reranking of search results",
            )
            reranker_device: str | None = Field(
                None,
                description="Device for reranker: 'cpu', 'cuda', or omit for auto-detect",
            )
            document_context_chars: int = Field(
                12000,
                ge=1000,
                le=50000,
                description="Characters of document text to include in LLM prompt",
            )

        keywords: Keywords = Field(
            default_factory=Keywords,
            description="Bridging term extraction from review articles (Stage 1)",
        )

        class Search(StrictModel):
            """Query expansion and comprehensive literature discovery (Stage 2)."""

            enabled: bool = Field(
                True, description="Run search stage (disable to skip to extraction)"
            )
            max_rounds: int = Field(
                8,
                ge=1,
                le=15,
                description="Search iterations; LLM decides when coverage is sufficient",
            )
            rerank_top_k: int = Field(
                0,
                ge=0,
                le=200,
                description="Use semantic reranking to select top-k results (0 = skip reranking, send all to LLM)",
            )
            batch_size: int = Field(
                0,
                ge=0,
                description="Results per LLM selection call (0 = all at once; use batching for large result sets)",
            )
            results_per_query: int = Field(
                100,
                ge=1,
                le=300,
                description="Maximum results to fetch from search backend per query",
            )
            reranker_model: str = Field(
                "zeroentropy/zerank-1-small",
                description="HuggingFace model for semantic reranking of search results",
            )
            reranker_device: str | None = Field(
                None,
                description="Device for reranker: 'cpu', 'cuda', or omit for auto-detect",
            )
            search_backend: str = Field(
                "pubmed",
                description="Backend for literature search: 'pubmed', 'perplexica', or 'openai'",
            )

        search: Search = Field(
            default_factory=Search,
            description="Query expansion and literature discovery (Stage 2)",
        )

        class Extraction(StrictModel):
            """Entity-relationship extraction from documents (Stage 3)."""

            proximal_window_chunks: int = Field(
                2,
                ge=0,
                le=10,
                description="Max text chunks between entities to consider them related (chunks ~500 chars)",
            )
            region_padding_chunks: int = Field(
                1,
                ge=0,
                le=5,
                description="Extra chunks to include around entity mentions for context",
            )
            merge_batch_size: int = Field(
                20,
                ge=1,
                le=200,
                description="Entity pairs to evaluate per LLM call during consolidation",
            )
            max_rename_iterations: int = Field(
                5,
                ge=1,
                le=10,
                description="Rounds of LLM review when standardizing entity names",
            )
            cluster_token_overlap_threshold: float = Field(
                0.30,
                ge=0.0,
                le=1.0,
                description="Word overlap required to group entity mentions (e.g., 'BMPR2' and 'BMPR2 gene')",
            )
            cluster_refinement_max_rounds: int = Field(
                10,
                ge=1,
                le=20,
                description="LLM review rounds for merging entity clusters",
            )
            filter_irrelevant_relationships: bool = Field(
                True,
                description="Use LLM to filter out generic relationships (e.g., 'is related to')",
            )
            enable_entity_kind_validation: bool = Field(
                True,
                description="Filter entities that don't match target kinds (-e gene, -e disease)",
            )
            agent_concurrency_limit: int = Field(
                10,
                ge=1,
                le=100,
                description="Parallel LLM calls (higher = faster but may hit rate limits)",
            )
            sweep_co_mentions: bool = Field(
                True,
                description="Second pass to find entity pairs missed in initial extraction",
            )

        extraction: Extraction = Field(
            default_factory=Extraction,
            description="Entity-relationship extraction from documents (Stage 3)",
        )

    stage: Stage = Field(
        default_factory=Stage,
        description="Pipeline stages: keywords → search → extraction",
    )

    class Output(StrictModel):
        """File output paths (relative to config file location)."""

        path: str = Field(
            "{topic}.json",
            description="Default output filename template; {topic} is replaced with sanitized topic",
        )
        cache: str = Field(
            "cache",
            description="Directory for cached web content and API responses",
        )

        @field_validator("path", "cache")
        @classmethod
        def rel_path(cls, v: str) -> str:
            if Path(v).is_absolute():
                raise ValueError("output paths must be relative")
            return v

    output: Output = Field(
        default_factory=Output,
        description="Output paths (relative to config file)",
    )

    modes: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Named config presets. Define as [modes.NAME] with overrides, activate with -m NAME. Example: [modes.fast] with stage.extraction.agent_concurrency_limit=50",
    )

    def abspath(self, path: str | Path, **kwargs: Any) -> Path:
        """
        Resolve a path relative to the config file directory.

        Args:
            path: Path or string that may be relative or absolute
            **kwargs: Format parameters for path template strings

        Returns:
            Absolute Path object
        """
        path = Path(str(path).format(**kwargs))
        if path.is_absolute():
            return path
        if self._dir:
            return (self._dir / path).resolve()
        return Path.cwd() / path

    def resolve_agent_config(self, module: str, agent: str | None = None) -> AgentSpec:
        """Recursively resolve agent configuration with fallback chain.

        Resolution order (most specific to least specific):
        1. agents.module.agent (specific agent override)
        2. agents.module._ (module-level default)
        3. agents._ (global default)
        4. Empty AgentSpec (all None values)

        Parameters:
            module: Module name (e.g., "keywords", "search", "extraction")
            agent: Optional agent name (e.g., "query_expander", "judge")

        Returns:
            Merged AgentSpec with inheritance from parent levels

        Example:
            >>> config.resolve_agent_config("extraction", "judge")
            # Returns merged spec: agent-specific → module-level → global
        """
        # Get global default
        global_spec = IfetcherConfig.AgentSpec()
        if isinstance(self.agents, IfetcherConfig.AgentSpec):
            global_spec = self.agents
        elif isinstance(self.agents, dict) and "_" in self.agents:
            global_spec = self.agents["_"]

        # Get module-level default
        module_spec = None
        if isinstance(self.agents, dict) and module in self.agents:
            module_val = self.agents[module]
            if isinstance(module_val, IfetcherConfig.AgentSpec):
                # Module has a single default spec
                module_spec = module_val
            elif "_" in module_val:
                # Module has explicit _ default
                module_spec = module_val["_"]

        # Merge module spec with global spec
        current_spec = (
            module_spec.merge_with_parent(global_spec) if module_spec else global_spec
        )

        # Get agent-specific config if requested
        if agent is not None:
            if isinstance(self.agents, dict) and module in self.agents:
                module_val = self.agents[module]
                if isinstance(module_val, dict) and agent in module_val:
                    agent_spec = module_val[agent]
                    current_spec = agent_spec.merge_with_parent(current_spec)

        return current_spec

    @staticmethod
    def apply_overrides(
        data: dict[str, Any], overrides: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Apply overrides to a configuration dictionary.

        Args:
            data: Base configuration dictionary
            overrides: Dictionary with dotted key paths and values to override

        Returns:
            Updated configuration dictionary
        """
        result = deepcopy(data)
        # Apply each override
        for key_path, value in overrides.items():
            keys = key_path.split(".")
            # Navigate to the nested dict location
            current = result
            for key in keys[:-1]:
                # Handle array indexing with [n] syntax
                if "[" in key and key.endswith("]"):
                    base_key, index_str = key.split("[", 1)
                    index = int(index_str[:-1])
                    # Ensure the key exists and is a list
                    if base_key not in current or not isinstance(
                        current[base_key], list
                    ):
                        current[base_key] = []
                    # Ensure the list has enough elements
                    while len(current[base_key]) <= index:
                        current[base_key].append({})
                    current = current[base_key][index]
                else:
                    # Regular key navigation
                    if key not in current:
                        current[key] = {}
                    current = current[key]
            # Set the value at the final location
            last_key = keys[-1]
            # Handle array indexing in the last key
            if "[" in last_key and last_key.endswith("]"):
                base_key, index_str = last_key.split("[", 1)
                index = int(index_str[:-1])
                # Ensure the key exists and is a list
                if base_key not in current or not isinstance(current[base_key], list):
                    current[base_key] = []
                # Ensure the list has enough elements
                while len(current[base_key]) <= index:
                    current[base_key].append(None)
                current[base_key][index] = value
            else:
                # Handle comma-separated list values
                if isinstance(value, str) and "," in value:
                    current[last_key] = [item.strip() for item in value.split(",")]
                else:
                    current[last_key] = value
        return result

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        overrides: dict[str, Any] | None = None,
        mode: str | None = None,
    ) -> IfetcherConfig:
        """
        Load configuration from a file path with optional overrides and mode.

        Args:
            path: Path to the configuration file
            overrides: Optional dictionary of configuration overrides
            mode: Optional mode to apply mode-specific overrides from config

        Returns:
            A new IfetcherConfig instance
        """
        config_path = Path(path).expanduser().resolve()
        config_dir = config_path.parent
        data = tomli.loads(config_path.read_text("utf-8"))
        # Apply mode-specific overrides from config if mode is specified
        if mode and "modes" in data and mode in data["modes"]:
            mode_overrides = data["modes"][mode]
            # Handle both flat dotted key format and nested format
            if isinstance(mode_overrides, dict):
                # Check if this is nested format by looking for non-dotted keys that are dicts
                has_nested = any(
                    isinstance(v, dict) and not k.count(".")
                    for k, v in mode_overrides.items()
                )
                if has_nested:
                    # Convert nested format to flat dotted format
                    flat_overrides = cls._flatten_nested_overrides(mode_overrides)
                    data = cls.apply_overrides(data, flat_overrides)
                else:
                    # Already in flat dotted format
                    data = cls.apply_overrides(data, mode_overrides)
        # Apply additional overrides if provided
        if overrides:
            data = cls.apply_overrides(data, overrides)
        config = cls.model_validate(data)
        config._dir = config_dir
        return config

    @staticmethod
    def _flatten_nested_overrides(
        nested_dict: dict[str, Any], parent_key: str = ""
    ) -> dict[str, Any]:
        """
        Convert nested dictionary to flat dictionary with dotted keys.

        Args:
            nested_dict: Nested dictionary to flatten
            parent_key: Parent key prefix for recursion

        Returns:
            Flattened dictionary with dotted keys
        """
        items = []
        for k, v in nested_dict.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            # If value is a dict and key doesn't contain dots (indicating it's not already flattened)
            if isinstance(v, dict) and "." not in k:
                items.extend(
                    IfetcherConfig._flatten_nested_overrides(v, new_key).items()
                )
            else:
                items.append((new_key, v))
        return dict(items)
