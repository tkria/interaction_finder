from __future__ import annotations

from pathlib import Path
from typing import Any
import tomli
from copy import deepcopy
from pydantic import BaseModel, Field, field_validator


class IfetcherConfig(BaseModel):
    """Simplified configuration for interaction finder with validation and path resolution."""

    _dir: Path | None = None  # Directory of the config file

    class AgentSpec(BaseModel):
        """Configuration for AI agent behavior and settings."""

        llm: str | None = Field(
            None, description="Language model to use for this agent"
        )
        expertise: str | None = Field(
            None, description="Domain expertise specification for the agent"
        )
        instruction: str | None = Field(
            None, description="Custom instructions for the agent"
        )
        retries: int | None = Field(
            None,
            description="Number of retry attempts for failed requests",
            ge=0,
            le=10,
        )
        instrument: bool = Field(
            True, description="Enable instrumentation and logging for this agent"
        )
        system_prompt: str | None = Field(
            None, description="Custom system prompt override for the agent"
        )
        model_settings: dict[str, Any] | None = Field(
            None, description="Model-specific settings (e.g., parallel_tool_calls)"
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
            for field_name in self.model_fields:
                self_value = getattr(self, field_name)
                parent_value = getattr(parent, field_name)
                # Use self's value if not None, otherwise inherit from parent
                merged_data[field_name] = (
                    self_value if self_value is not None else parent_value
                )
            return IfetcherConfig.AgentSpec.model_validate(merged_data)

    agents: dict[str, Any] | AgentSpec = Field(
        default_factory=dict,
        description="Configuration for AI agents (supports multi-tier: agents._, agents.module._, agents.module.agent)",
    )

    class Tools(BaseModel):
        """Configuration for external tools."""

        class Crawl4AI(BaseModel):
            """Configuration for Crawl4AI web scraping."""

            timeout: int = Field(
                30, description="Request timeout in seconds", ge=1, le=600
            )

        crawl4ai: Crawl4AI = Field(
            default_factory=Crawl4AI, description="Crawl4AI configuration"
        )

        class Keywords(BaseModel):
            """Configuration for keyword research module."""

            max_rounds: int = Field(5, ge=1, le=10, description="Maximum search rounds")
            search_backend: str = Field(
                "perplexica", description="Search backend to use"
            )
            max_results_per_query: int = Field(
                20, ge=1, le=100, description="Maximum results per search query"
            )
            max_documents_to_fetch: int = Field(
                10, ge=1, le=50, description="Maximum documents to fetch per round"
            )
            max_keywords_per_method: int = Field(
                30, ge=5, le=100, description="Maximum keywords per extraction method"
            )
            max_keywords_for_llm: int = Field(
                50,
                ge=10,
                le=100,
                description="Maximum keywords to show LLM after deduplication and reranking",
            )
            reranker_model: str = Field(
                "zeroentropy/zerank-1-small",
                description="Reranking model name",
            )
            reranker_device: str | None = Field(
                None,
                description="Device for reranker model ('cpu', 'cuda', or None for auto)",
            )
            llm_model: str = Field(
                "openai:gpt-4o-mini", description="LLM model for agents"
            )
            document_context_chars: int = Field(
                12000,
                ge=1000,
                le=50000,
                description="Number of characters from document to send to LLM for evaluation",
            )

            class RAKEConfig(BaseModel):
                min_length: int = Field(1, ge=1, description="Minimum phrase length")
                max_length: int = Field(4, ge=1, description="Maximum phrase length")

            class YAKEConfig(BaseModel):
                n_grams: int = Field(3, ge=1, le=5, description="Maximum n-gram size")
                deduplication_threshold: float = Field(
                    0.9, ge=0.0, le=1.0, description="Deduplication threshold"
                )
                window_size: int = Field(1, ge=1, description="Context window size")

            class TFIDFConfig(BaseModel):
                max_features: int = Field(50, ge=1, description="Maximum features")
                ngram_range: tuple[int, int] = Field(
                    (1, 3), description="N-gram range (min, max)"
                )
                min_df: int = Field(1, ge=1, description="Minimum document frequency")

            class KeyBERTConfig(BaseModel):
                model_name: str = Field(
                    "all-MiniLM-L6-v2", description="Sentence-transformers model"
                )
                diversity: float = Field(
                    0.5, ge=0.0, le=1.0, description="MMR diversity parameter"
                )
                top_n: int = Field(20, ge=1, description="Number of candidates")
                device: str | None = Field(
                    None,
                    description="Device for KeyBERT model ('cpu', 'cuda', or None for auto)",
                )

            rake: RAKEConfig = Field(
                default_factory=RAKEConfig, description="RAKE extractor configuration"
            )
            yake: YAKEConfig = Field(
                default_factory=YAKEConfig, description="YAKE extractor configuration"
            )
            tfidf: TFIDFConfig = Field(
                default_factory=TFIDFConfig,
                description="TF-IDF extractor configuration",
            )
            keybert: KeyBERTConfig = Field(
                default_factory=KeyBERTConfig,
                description="KeyBERT extractor configuration",
            )

        keywords: Keywords = Field(
            default_factory=Keywords, description="Keyword research configuration"
        )

        class Widesearch(BaseModel):
            """Configuration for widesearch query expansion module."""

            enabled: bool = Field(True, description="Enable widesearch functionality")
            max_rounds: int = Field(
                8, ge=1, le=15, description="Maximum search rounds before stopping"
            )
            enable_reranking: bool = Field(
                True,
                description="Enable semantic reranking of search results (if False, pass all results to LLM)",
            )
            rerank_top_k: int = Field(
                50,
                ge=1,
                le=200,
                description="Number of top results to keep after reranking (only used if enable_reranking=True)",
            )
            results_per_query: int = Field(
                100, ge=1, le=300, description="Maximum results to fetch per query"
            )
            reranker_model: str = Field(
                "zeroentropy/zerank-1-small",
                description="Reranking model name",
            )
            reranker_device: str | None = Field(
                None,
                description="Device for reranker model ('cpu', 'cuda', or None for auto)",
            )
            llm_model: str = Field(
                "openai:gpt-4o-mini", description="LLM model for agents"
            )
            search_backend: str = Field("pubmed", description="Search backend to use")

        widesearch: Widesearch = Field(
            default_factory=Widesearch, description="Widesearch configuration"
        )

    tools: Tools = Field(
        default_factory=Tools, description="External tools configuration"
    )

    class Output(BaseModel):
        """Configuration for output file paths and caching."""

        path: str = Field(
            "runs/{mode}/{model}/{repeat}/{term}",
            description="Output path template (supports {mode}, {model}, {repeat}, {term})",
        )
        cache: str = Field("cache", description="Cache directory path")

        @field_validator("path", "cache")
        @classmethod
        def rel_path(cls, v: str) -> str:
            if Path(v).is_absolute():
                raise ValueError("output paths must be relative")
            return v

    output: Output = Field(
        default_factory=Output,
        description="Configuration for output paths and caching",
    )

    training_data: str = Field(
        "training_data/{term}.jsonl",
        description="Training data path template (supports {term})",
    )

    modes: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Mode-specific configuration overrides"
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
        2. agents.module._ (module-level default with underscore)
        3. agents.module (module-level default without underscore, if AgentSpec)
        4. agents._ (global default with underscore)
        5. agents (global default if AgentSpec)
        6. Empty AgentSpec (all None values)

        Note: Both underscore and non-underscore versions are checked for defaults
        to provide flexibility in config style. agents._ and agents are equivalent
        for global defaults, as are agents.module._ and agents.module for module defaults.

        Parameters:
            module: Module name (e.g., "keywords", "widesearch", "extraction")
            agent: Optional agent name (e.g., "query_expander", "judge")

        Returns:
            Merged AgentSpec with inheritance from parent levels

        Example:
            >>> config.resolve_agent_config("extraction", "judge")
            # Returns merged spec: agent-specific → module-level → global
        """
        # Start with empty spec (all None values)
        global_spec = IfetcherConfig.AgentSpec()
        # Get global defaults (check both _ and root level)
        if isinstance(self.agents, IfetcherConfig.AgentSpec):
            global_spec = self.agents
        elif isinstance(self.agents, dict):
            if "_" in self.agents:
                global_val = self.agents["_"]
                if isinstance(global_val, dict):
                    global_spec = IfetcherConfig.AgentSpec.model_validate(global_val)
                elif isinstance(global_val, IfetcherConfig.AgentSpec):
                    global_spec = global_val
        # Get module-level config
        module_spec = None
        if isinstance(self.agents, dict) and module in self.agents:
            module_val = self.agents[module]
            # Check if module value is an AgentSpec (used as module default)
            if isinstance(module_val, dict):
                # Check for _ key for explicit module default
                if "_" in module_val:
                    underscore_val = module_val["_"]
                    if isinstance(underscore_val, dict):
                        module_spec = IfetcherConfig.AgentSpec.model_validate(
                            underscore_val
                        )
                    elif isinstance(underscore_val, IfetcherConfig.AgentSpec):
                        module_spec = underscore_val
                # If no _ key and no agent specified, treat module value as implicit default
                elif agent is None:
                    # Try to parse entire module dict as AgentSpec
                    try:
                        module_spec = IfetcherConfig.AgentSpec.model_validate(
                            module_val
                        )
                    except:
                        pass  # Not a valid AgentSpec, continue
            elif isinstance(module_val, IfetcherConfig.AgentSpec):
                module_spec = module_val
        # Merge module spec with global spec
        current_spec = (
            module_spec.merge_with_parent(global_spec) if module_spec else global_spec
        )
        # If agent is specified, get agent-specific config
        if agent is not None:
            agent_spec = None
            if isinstance(self.agents, dict) and module in self.agents:
                module_val = self.agents[module]
                if isinstance(module_val, dict) and agent in module_val:
                    agent_val = module_val[agent]
                    if isinstance(agent_val, dict):
                        agent_spec = IfetcherConfig.AgentSpec.model_validate(agent_val)
                    elif isinstance(agent_val, IfetcherConfig.AgentSpec):
                        agent_spec = agent_val
            # Merge agent spec with module/global spec
            if agent_spec:
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
                    try:
                        current[last_key] = [item.strip() for item in value.split(",")]
                    except:
                        current[last_key] = value
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
