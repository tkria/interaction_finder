from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Union, Any, Optional
import tomli
import shlex
from copy import deepcopy
from pydantic import BaseModel, Field, field_validator, model_validator


class IfetcherConfig(BaseModel):
    _dir: Path | None = None  # Directory of the config file

    class AgentSpec(BaseModel):
        llm: str | None = None
        expertise: str | None = None
        instruction: str | None = None
        retries: int | None = None
        instrument: bool = True
        prompt: str | None = None

    agents: Dict[str, AgentSpec] = Field(default_factory=dict)

    researcher: str = ""

    @field_validator("researcher")
    @classmethod
    def validate_researcher(cls, v):
        """Validate that researcher is a valid research mode"""
        # NOTE: Currently no researcher modes are implemented
        # This list is prepared for future implementation of different research strategies
        # For now, only empty string is valid (default mode)
        valid_modes = [""]
        if v not in valid_modes:
            raise ValueError(f"researcher must be one of {valid_modes}, got '{v}'")
        return v

    class Tools(BaseModel):
        enabled: List[str] = Field(default_factory=list)

        class SearXNG(BaseModel):
            categories: str = "general"
            blocked_sites: List[str] = Field(default_factory=list)

        class Crawl4AI(BaseModel):
            timeout: int = 30
            max_retries: int = 3
            user_agent: str = "InteractionFinder/1.0"
            delay_between_requests: float = 1.0
            max_concurrent: int = 5

        class Ontologies(BaseModel):
            hpo_path: str | None = None
            cl_path: str | None = None

        class ExternalResearcher(BaseModel):
            cmd: Union[str, List[str]] = Field(default_factory=list)
            dir: str | None = None
            env: Dict[str, str] = Field(default_factory=dict)
            strip_prefix_re: str = ""
            strip_suffix_re: str = ""
            strip_re: Union[str, List[str]] = Field(default_factory=list)

            @field_validator("cmd", mode="before")
            @classmethod
            def normalize_cmd(cls, v):
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
            def normalize_strip_re(cls, v):
                """Convert string values to single-item lists for consistency"""
                if isinstance(v, str):
                    return [v] if v else []
                return v if v is not None else []

        crawl4ai: Crawl4AI = Field(default_factory=Crawl4AI)
        searxng: SearXNG = Field(default_factory=SearXNG)
        ontologies: Ontologies = Field(default_factory=Ontologies)
        external_researcher: ExternalResearcher = Field(
            default_factory=ExternalResearcher
        )

    tools: Tools = Field(default_factory=Tools)

    class Task(BaseModel):
        class Kind(BaseModel):
            kind: List[str] = Field(default_factory=list, serialization_alias="is")
            form: List[str] = Field(default_factory=lambda: ["name"])
            example: List[str] = Field(default_factory=list)
            normalise: Dict[str, str] = Field(default_factory=dict)

            @field_validator("kind", "example", "form", mode="before")
            @classmethod
            def normalise_to_list(cls, v):
                """Convert string values to single-item lists for consistency"""
                if isinstance(v, str):
                    return [v]
                return v

        relation: str = "Interaction"
        pairs: str = "pairs"
        context: str = ""
        kinds: Dict[str, Kind] = Field(default_factory=dict)
        synonyms: Dict[str, str] = Field(default_factory=dict)
        example: List[str] = Field(default_factory=list)

        @field_validator("kinds", mode="before")
        @classmethod
        def normalise_kinds(cls, v):
            """
            Normalizes the kinds field from various input formats:

            1. Direct Kind object → {"default": Kind}
            2. Array format like ["celltype", "biomarker"] → {"celltype": Kind(kind="celltype"), "biomarker": Kind(kind="biomarker")}
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
                        kind_obj = cls.Kind(kind=item)
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
        def validate_kinds_not_empty(cls, v):
            if not v:
                raise ValueError(
                    "At least one kind must be defined in [task.kinds] or [task.kinds.*]"
                )
            return v

        def get_kind_names(self) -> List[str]:
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

    task: Task = Field(default_factory=Task)

    class Workflow(BaseModel):
        class Summarisation(BaseModel):
            chunksize: int = 10
            maxchars: int = 80000

        class Grouping(BaseModel):
            """Configuration for document grouping within workflow."""

            enabled: bool = True
            constraint_type: str = "count"  # "count" or "words"
            min_size: int = 3
            max_size: int = 8
            linkage_method: str = "average"  # "average", "complete", "single"

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

            @model_validator(mode="after")
            def validate_size_range(self) -> "Grouping":
                if self.min_size > self.max_size:
                    raise ValueError("min_size must be <= max_size")
                if self.min_size < 1:
                    raise ValueError("min_size must be >= 1")
                return self

        envfile: str | None = None
        max_loops: int = 5
        max_tokens: int = 500000
        max_requests: int | None = None
        summary: Summarisation = Field(default_factory=Summarisation)
        grouping: Grouping = Field(default_factory=Grouping)
        unstructured_comparison: bool = False

    workflow: Workflow = Field(default_factory=Workflow)

    class Output(BaseModel):
        path: str = "runs/{mode}/{model}/{repeat}/{term}"
        cache: str = "cache"

        @field_validator("path", "cache")
        @classmethod
        def rel_path(cls, v: str) -> str:
            if Path(v).is_absolute():
                raise ValueError("output paths must be relative")
            return v

    output: Output = Field(default_factory=Output)

    training_data: str = "training_data/{term}.jsonl"

    modes: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    def abspath(self, path: str | Path, **kwargs) -> Path:
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
        data: Dict[str, Any], overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
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
            for i, key in enumerate(keys[:-1]):
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
        overrides: Optional[Dict[str, Any]] = None,
        mode: Optional[str] = None,
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
        nested_dict: Dict[str, Any], parent_key: str = ""
    ) -> Dict[str, Any]:
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
