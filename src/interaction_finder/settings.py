from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
import tomli
import shlex
import os
from copy import deepcopy
from pydantic import BaseModel, Field, field_validator, model_validator

if TYPE_CHECKING:
    from .search.config import SearchConfig


def configure_logfire(verbose: bool = False) -> None:
    """Configure logfire if LOGFIRE_WRITE_TOKEN is available."""
    token = os.environ.get("LOGFIRE_WRITE_TOKEN")
    import logfire
    from logfire import ConsoleOptions

    if verbose:
        coptions = ConsoleOptions()
    else:
        coptions = ConsoleOptions(min_log_level="warn", show_project_link=False)
    _ = logfire.configure(
        send_to_logfire="if-token-present",
        token=token,
        scrubbing=False,
        console=coptions,
    )
    _ = logfire.instrument_pydantic_ai()


def _create_search_config():
    """Create SearchConfig instance - helper to avoid circular imports."""
    from .search.config import SearchConfig

    return SearchConfig(
        backend="pubmed", max_results=100, concurrent_backends=2, global_timeout=120
    )


class IfetcherConfig(BaseModel):
    """Configuration for interaction finder with validation and path resolution."""

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
        prompt: str | None = Field(
            None, description="Custom prompt template for the agent"
        )

    agents: dict[str, AgentSpec] = Field(
        default_factory=dict,
        description="Configuration for different AI agents by name",
    )

    researcher: str = Field(
        "", description="Research mode selection (currently only default supported)"
    )

    @field_validator("researcher")
    @classmethod
    def validate_researcher(cls, v: str) -> str:
        """Validate that researcher is a valid research mode"""
        # NOTE: Currently no researcher modes are implemented
        # This list is prepared for future implementation of different research strategies
        # For now, only empty string is valid (default mode)
        valid_modes = [""]
        if v not in valid_modes:
            raise ValueError(f"researcher must be one of {valid_modes}, got '{v}'")
        return v

    class Tools(BaseModel):
        """Configuration for external tools and services."""

        enabled: list[str] = Field(
            default_factory=list,
            description="List of enabled tools (e.g., crawl4ai, searxng, search)",
        )

        class SearXNG(BaseModel):
            """Configuration for SearXNG search engine integration."""

            categories: str = Field(
                "general", description="Search categories for SearXNG"
            )
            blocked_sites: list[str] = Field(
                default_factory=list,
                description="List of sites to block in search results",
            )

        class Crawl4AI(BaseModel):
            """Configuration for Crawl4AI web scraping tool."""

            timeout: int = Field(
                30, description="Request timeout in seconds", ge=1, le=600
            )
            max_retries: int = Field(
                3, description="Maximum number of retry attempts", ge=0, le=10
            )
            user_agent: str = Field(
                "InteractionFinder/1.0",
                description="HTTP User-Agent string for requests",
            )
            delay_between_requests: float = Field(
                1.0, description="Delay between requests in seconds", ge=0.0, le=10.0
            )
            max_concurrent: int = Field(
                5, description="Maximum concurrent requests", ge=1, le=50
            )

        class Ontologies(BaseModel):
            """Configuration for biological ontology file paths."""

            hpo_path: str | None = Field(
                None, description="Path to HPO (Human Phenotype Ontology) file"
            )
            cl_path: str | None = Field(
                None, description="Path to CL (Cell Ontology) file"
            )

        class ExternalResearcher(BaseModel):
            """Configuration for external research tools and commands."""

            cmd: str | list[str] = Field(
                default_factory=list,
                description="External researcher command and arguments",
            )
            dir: str | None = Field(
                None, description="Working directory for external researcher"
            )
            env: dict[str, str] = Field(
                default_factory=dict,
                description="Environment variables for external researcher",
            )
            strip_prefix_re: str = Field(
                "", description="Regex pattern to strip from output prefix"
            )
            strip_suffix_re: str = Field(
                "", description="Regex pattern to strip from output suffix"
            )
            strip_re: str | list[str] = Field(
                default_factory=list,
                description="List of regex patterns to strip from external researcher output",
            )

            @field_validator("cmd", mode="before")
            @classmethod
            def normalize_cmd(cls, v: Any) -> list[str]:
                """Convert string command to list using shell parsing"""
                if isinstance(v, str):
                    if not v.strip():
                        return []
                    try:
                        return shlex.split(v)
                    except ValueError as e:
                        raise ValueError(f"Invalid command string: {e}")
                return v if v is not None else []

            @field_validator("strip_re", mode="before")
            @classmethod
            def normalize_strip_re(cls, v: Any) -> list[str]:
                """Convert string values to single-item lists for consistency"""
                if isinstance(v, str):
                    return [v] if v else []
                return v if v is not None else []

        crawl4ai: "Crawl4AI" = Field(default_factory=Crawl4AI)
        searxng: "SearXNG" = Field(default_factory=SearXNG)
        ontologies: "Ontologies" = Field(default_factory=Ontologies)
        external_researcher: "ExternalResearcher" = Field(
            default_factory=ExternalResearcher
        )

        search: "SearchConfig" = Field(
            default_factory=lambda: _create_search_config(),
            description="Configuration for document search functionality",
        )

    tools: "Tools" = Field(
        default_factory=Tools,
        description="Configuration for external tools and services",
    )

    class Task(BaseModel):
        """Configuration for extraction task definition and parameters."""

        class Kind(BaseModel):
            """Configuration for entity kind definitions in extraction tasks."""

            kind: list[str] = Field(
                default_factory=list,
                serialization_alias="is",
                description="Entity types for this kind",
            )
            form: list[str] = Field(
                default_factory=lambda: ["name"],
                description="Form variants for entity recognition",
            )
            example: list[str] = Field(
                default_factory=list, description="Example entities of this kind"
            )
            normalise: dict[str, str] = Field(
                default_factory=dict, description="Entity name normalization mappings"
            )

            @field_validator("kind", "example", "form", mode="before")
            @classmethod
            def normalise_to_list(cls, v: Any) -> list[str]:
                """Convert string values to single-item lists for consistency"""
                if isinstance(v, str):
                    return [v]
                return v

        relation: str = Field(
            "Interaction",
            description="Type of relation to extract (e.g., 'Interaction')",
        )
        pairs: str = Field("pairs", description="Pair extraction mode")
        context: str = Field("", description="Context information for task")
        kinds: dict[str, "Kind"] = Field(
            default_factory=dict, description="Entity kind definitions for extraction"
        )
        synonyms: dict[str, str] = Field(
            default_factory=dict, description="Entity synonym mappings"
        )
        example: list[str] = Field(
            default_factory=list, description="Example interactions for the task"
        )

        @field_validator("kinds", mode="before")
        @classmethod
        def normalise_kinds(cls, v: Any) -> dict[str, "Task.Kind"]:
            """
            Normalizes the kinds field from various input formats:

            1. Direct Kind object → {"default": Kind}
            2. Array format like ["celltype", "biomarker"] → {"celltype": Kind(kind=["celltype"]), "biomarker": Kind(kind=["biomarker"])}
            3. Raw dict with Kind properties → {"default": Kind}
            4. Dict of named kinds → unchanged
            """
            # Handle direct Kind object
            if isinstance(v, cls.Kind):
                return {"default": v}

            # Handle array format like task.kinds = ["celltype", "biomarker", ...]
            if isinstance(v, list):
                result = {}
                for item in v:
                    if isinstance(item, str):
                        # Create a Kind object for each string in the list with the string as both key and kind value
                        kind_obj = cls.Kind(kind=[item])
                        result[item] = kind_obj
                return result

            # Handle raw dict that should be a Kind
            if isinstance(v, dict):
                # Check if it's a raw Kind definition (has keys like 'kind', 'form', 'is')
                if any(k in ["kind", "is", "form", "example"] for k in v.keys()):
                    # Detect if this is a flat structure with kind properties directly
                    if any(
                        isinstance(v.get(k), (str, list))
                        for k in ["kind", "is", "form", "example"]
                    ):
                        try:
                            kind_obj = cls.Kind.model_validate(v)
                            return {"default": kind_obj}
                        except Exception:
                            pass
            return v if isinstance(v, dict) else {}

        @field_validator("kinds")
        @classmethod
        def validate_kinds_not_empty(
            cls, v: dict[str, "Task.Kind"]
        ) -> dict[str, "Task.Kind"]:
            if not v:
                raise ValueError(
                    "At least one kind must be defined in [task.kinds] or [task.kinds.*]"
                )
            return v

        def get_kind_names(self) -> list[str]:
            """Get the list of kind names defined in the task."""
            return list(self.kinds.keys())

        def get_complementary_kind(self, kind_name: str) -> str:
            """Get the complementary kind name if there are exactly two kinds defined."""
            if len(self.kinds) == 1:
                return self.get_kind_names()[0]
            elif len(self.kinds) == 2:
                kinds = self.get_kind_names()
                return kinds[1] if kinds[0] == kind_name else kinds[0]
            else:
                raise ValueError(
                    "Cannot get complementary kind: more than two kinds defined."
                )

    task: "Task" = Field(
        default_factory=Task,
        description="Configuration for extraction task definition",
    )

    class Workflow(BaseModel):
        """Configuration for processing workflow and document handling."""

        class Summarisation(BaseModel):
            """Configuration for document summarization parameters."""

            chunksize: int = Field(
                10, description="Chunk size for summarization", ge=1, le=100
            )
            maxchars: int = Field(
                80000,
                description="Maximum characters for summarization",
                ge=1000,
                le=200000,
            )

        class Grouping(BaseModel):
            """Configuration for document grouping within workflow."""

            enabled: bool = Field(
                True, description="Enable document grouping by semantic similarity"
            )
            constraint_type: str = Field(
                "count", description="Group by document count or total word count"
            )  # "count" or "words"
            min_size: int = Field(3, description="Minimum group size", ge=1, le=100)
            max_size: int = Field(8, description="Maximum group size", ge=1, le=100)
            linkage_method: str = Field(
                "average", description="Clustering linkage method for grouping"
            )  # "average", "complete", "single"
            clustering_method: str = Field(
                "agglomerative", description="Clustering algorithm to use"
            )  # "agglomerative", "spectral", "hybrid", "random", "size_annealed_agglomerative"
            embedding_weights: str = Field(
                "idf", description="Document embedding weights"
            )  # "uniform", "idf"
            dual_evaluation: bool = Field(
                False,
                description="Evaluate groups using both embedding weight strategies",
            )
            seeding_method: str = Field(
                "kmeans", description="Spectral seeding method for hybrid clustering"
            )  # "kmeans", "fiedler"
            refinement_method: str = Field(
                "hierarchical", description="Refinement method for hybrid clustering"
            )  # "hierarchical", "agglomerative"

            @field_validator("constraint_type")
            @classmethod
            def validate_constraint_type(cls, v: str) -> str:
                if v not in ("count", "words"):
                    raise ValueError("constraint_type must be 'count' or 'words'")
                return v

            @field_validator("linkage_method")
            @classmethod
            def validate_linkage_method(cls, v: str) -> str:
                if v not in ("average", "complete", "single"):
                    raise ValueError(
                        "linkage_method must be 'average', 'complete', or 'single'"
                    )
                return v

            @field_validator("clustering_method")
            @classmethod
            def validate_clustering_method(cls, v: str) -> str:
                if v not in (
                    "agglomerative",
                    "spectral",
                    "hybrid",
                    "random",
                    "size_annealed_agglomerative",
                ):
                    raise ValueError(
                        "clustering_method must be 'agglomerative', 'spectral', 'hybrid', 'random', or 'size_annealed_agglomerative'"
                    )
                return v

            @field_validator("embedding_weights")
            @classmethod
            def validate_embedding_weights(cls, v: str) -> str:
                if v not in ("uniform", "idf"):
                    raise ValueError("embedding_weights must be 'uniform' or 'idf'")
                return v

            @field_validator("seeding_method")
            @classmethod
            def validate_seeding_method(cls, v: str) -> str:
                if v not in ("kmeans", "fiedler"):
                    raise ValueError("seeding_method must be 'kmeans' or 'fiedler'")
                return v

            @field_validator("refinement_method")
            @classmethod
            def validate_refinement_method(cls, v: str) -> str:
                if v not in ("hierarchical", "agglomerative"):
                    raise ValueError(
                        "refinement_method must be 'hierarchical' or 'agglomerative'"
                    )
                return v

            @model_validator(mode="after")
            def validate_size_range(self) -> "Grouping":
                if self.min_size > self.max_size:
                    raise ValueError("min_size must be <= max_size")
                if self.min_size < 1:
                    raise ValueError("min_size must be >= 1")
                return self

        envfile: str | None = Field(
            None, description="Environment file for workflow configuration"
        )
        max_loops: int = Field(5, description="Maximum processing loops", ge=1, le=50)
        max_tokens: int = Field(
            500000, description="Maximum tokens for processing", ge=1000, le=2000000
        )
        max_requests: int | None = Field(
            None,
            description="Maximum number of requests (null for unlimited)",
            ge=1,
            le=10000,
        )
        summary: "Summarisation" = Field(
            default_factory=Summarisation,
            description="Document summarization settings",
        )
        grouping: "Grouping" = Field(
            default_factory=Grouping,
            description="Document grouping configuration",
        )
        unstructured_comparison: bool = Field(
            False, description="Enable unstructured comparison mode"
        )

    workflow: "Workflow" = Field(
        default_factory=Workflow,
        description="Configuration for processing workflow",
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

    output: "Output" = Field(
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
            path: A path or string that could be relative or absolute

        Returns:
            Path object with the resolved absolute path
        """
        path = Path(str(path).format(**kwargs))
        if path.is_absolute():
            return path
        if self._dir:
            return (self._dir / path).resolve()
        return Path.cwd() / path

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
        # Create a deep copy of the data to avoid modifying the original
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
                    index = int(index_str[:-1])  # Remove the closing ']'

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
                        # Check if this looks like it should be a list of strings
                        current[last_key] = [item.strip() for item in value.split(",")]
                    except:
                        # If conversion fails, use the original string
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
    ) -> "IfetcherConfig":
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


# Rebuild models to resolve forward references
def _rebuild_config_models():
    """Rebuild models after imports are available."""
    from .search.config import SearchConfig

    IfetcherConfig.model_rebuild()


# Import and rebuild on module load
_rebuild_config_models()
