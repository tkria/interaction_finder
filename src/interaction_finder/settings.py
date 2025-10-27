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
        prompt: str | None = Field(
            None, description="Custom prompt template for the agent"
        )

    agents: dict[str, AgentSpec] = Field(
        default_factory=dict,
        description="Configuration for different AI agents by name",
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
