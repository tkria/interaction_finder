"""Unified agent configuration and caching system.

Provides a consistent interface for creating and caching LLM agents across
all pipeline stages (keywords, widesearch, extraction) with multi-tier
configuration resolution.
"""

from typing import Any, Callable, Type

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import (
    OpenAIModelProfile,
    OpenAIResponsesModel,
    OpenAIResponsesModelSettings,
)
from pydantic_ai.profiles.openai import openai_model_profile
from pydantic_ai.settings import ModelSettings

from interaction_finder.settings import IfetcherConfig


# Global cache for agent instances
# Cache key: (module, agent, model, frozen_config_params)
_agent_cache: dict[tuple[str, str, str, frozenset[tuple[str, Any]]], Agent] = {}


def _resolve_gpt5_model(
    model_string: str,
) -> tuple[OpenAIResponsesModel | str, OpenAIResponsesModelSettings | None]:
    """Resolve GPT-5 model strings with optional reasoning effort.

    Handles model strings like:
    - "openai:gpt-5" -> OpenAIResponsesModel with default profile
    - "openai:gpt-5-mini" -> OpenAIResponsesModel with default profile
    - "openai:gpt-5/low" -> OpenAIResponsesModel with low reasoning effort
    - "openai:gpt-5-mini/medium" -> OpenAIResponsesModel with medium reasoning effort
    - "openai:gpt-5-nano/high" -> OpenAIResponsesModel with high reasoning effort

    Parameters:
        model_string: Model string in format "openai:gpt-5[-variant][/effort]"

    Returns:
        Tuple of (model, model_settings) where model_settings is OpenAIResponsesModelSettings if reasoning effort specified
    """
    # Check if this is a GPT-5 model string
    if not model_string.startswith("openai:gpt-5"):
        return model_string, None

    # Extract the model name and optional reasoning effort
    parts = model_string.split("/")
    base_model = parts[0].replace("openai:", "")  # e.g., "gpt-5-mini"

    # Create the base GPT-5 model with default profile
    model = OpenAIResponsesModel(
        base_model,
        profile=OpenAIModelProfile.from_profile(openai_model_profile("gpt-5")),
    )

    # Handle reasoning effort if specified
    model_settings = None
    if len(parts) > 1:
        reasoning_effort = parts[1]
        if reasoning_effort in ("low", "medium", "high"):
            model_settings = OpenAIResponsesModelSettings(
                openai_reasoning_effort=reasoning_effort,
            )

    return model, model_settings


def agent_getter(
    module: str,
    agent: str,
    output_type: Type[BaseModel],
    deps_type: Type,
    system_prompt: str,
    *,
    default_model: str = "openai:gpt-4o-mini",
    default_model_settings: ModelSettings | None = None,
) -> Callable[[IfetcherConfig], Agent]:
    """Create a getter function for an agent with baked-in configuration.

    This factory function creates a clean getter that only needs the config object.
    Use this to define agents elegantly in their respective modules.

    Parameters:
        module: Module name (e.g., "extraction")
        agent: Agent name (e.g., "entity")
        output_type: Pydantic output model
        deps_type: Pydantic deps type
        system_prompt: Default system prompt
        default_model: Fallback model
        default_model_settings: Fallback model settings

    Returns:
        Function that takes config and returns configured Agent

    Example:
        >>> get_entity_extractor_agent = agent_getter(
        ...     "extraction", "entity",
        ...     EntityExtractionOut, Deps,
        ...     "You are an expert...",
        ...     default_model_settings=ModelSettings(parallel_tool_calls=False)
        ... )
        >>> # Later, in nodes:
        >>> agent = get_entity_extractor_agent(ctx.deps.config)
    """

    def getter(config: IfetcherConfig) -> Agent:
        return get_agent(
            config,
            module,
            agent,
            output_type,
            deps_type,
            system_prompt,
            default_model=default_model,
            default_model_settings=default_model_settings,
        )

    return getter


def get_agent(
    config: IfetcherConfig,
    module: str,
    agent: str,
    output_type: Type[BaseModel],
    deps_type: Type,
    system_prompt: str,
    *,
    default_model: str = "openai:gpt-4o-mini",
    default_retries: int = 2,
    default_instrument: bool = True,
    default_model_settings: ModelSettings | None = None,
) -> Agent:
    """Get or create agent with multi-tier configuration.

    Resolves agent configuration using multi-tier fallback:
    1. agents.module.agent.param (agent-specific override)
    2. agents.module._.param or agents.module.param (module default)
    3. agents._.param or agents.param (global default)
    4. default_param (code-level fallback)

    Agents are cached by full configuration signature to avoid recreation.

    Parameters:
        config: IfetcherConfig — configuration object with agents settings
        module: str — module name (e.g., "keywords", "widesearch", "extraction")
        agent: str — agent name (e.g., "query_expander", "judge")
        output_type: Type[BaseModel] — Pydantic model for structured output
        deps_type: Type — Pydantic type for dependencies
        system_prompt: str — default system prompt (can be overridden in config)
        default_model: str — fallback model if not in config
        default_retries: int — fallback retry count if not in config
        default_instrument: bool — fallback instrumentation flag if not in config
        default_model_settings: ModelSettings | None — fallback model settings

    Returns:
        Cached or newly created Agent instance

    Example:
        >>> config = IfetcherConfig.from_path("config.toml")
        >>> agent = get_agent(
        ...     config,
        ...     "keywords",
        ...     "query_expander",
        ...     QueryExpansionOut,
        ...     Deps,
        ...     "You are an expert at generating queries...",
        ... )
    """
    # Resolve configuration for this specific agent
    agent_spec = config.resolve_agent_config(module, agent)
    # Extract effective values (config overrides code defaults)
    model_string = agent_spec.llm or default_model
    retries = agent_spec.retries if agent_spec.retries is not None else default_retries
    instrument = agent_spec.instrument
    effective_system_prompt = agent_spec.system_prompt or system_prompt

    # Resolve GPT-5 models with optional reasoning effort
    model, gpt5_settings = _resolve_gpt5_model(model_string)

    # Handle ModelSettings (merge GPT-5 settings with config/default settings)
    model_settings = default_model_settings
    if agent_spec.model_settings:
        # Convert dict to ModelSettings
        model_settings = ModelSettings(**agent_spec.model_settings)

    # Merge GPT-5 settings if present
    if gpt5_settings:
        if model_settings:
            # Merge: GPT-5 settings take precedence for reasoning_effort
            # Both ModelSettings and OpenAIResponsesModelSettings are dict subclasses, not Pydantic models
            model_settings_dict = dict(model_settings)
            gpt5_dict = dict(gpt5_settings)
            model_settings = ModelSettings(**(model_settings_dict | gpt5_dict))
        else:
            model_settings = gpt5_settings
    # Build cache key from all configuration parameters
    parallel_calls = None
    reasoning_effort = None
    if model_settings:
        if isinstance(model_settings, dict):
            parallel_calls = model_settings.get("parallel_tool_calls")
            reasoning_effort = model_settings.get("openai_reasoning_effort")
        else:
            parallel_calls = model_settings.parallel_tool_calls
            # Handle OpenAIResponsesModelSettings
            if hasattr(model_settings, "openai_reasoning_effort"):
                reasoning_effort = model_settings.openai_reasoning_effort

    # Create cache key - use model_string for consistency
    config_params = frozenset(
        [
            ("model", model_string),
            ("retries", retries),
            ("instrument", instrument),
            ("system_prompt", effective_system_prompt),
            ("parallel_tool_calls", parallel_calls),
            ("reasoning_effort", reasoning_effort),
        ]
    )
    cache_key = (module, agent, model_string, config_params)
    # Return cached agent if available
    if cache_key in _agent_cache:
        return _agent_cache[cache_key]
    # Create new agent
    new_agent = Agent(
        model=model,
        deps_type=deps_type,
        output_type=output_type,
        retries=retries,
        system_prompt=effective_system_prompt,
        model_settings=model_settings,
    )
    # Cache and return
    _agent_cache[cache_key] = new_agent
    return new_agent


def clear_agent_cache() -> None:
    """Clear the agent cache.

    Useful for testing to ensure clean state between test cases.
    """
    _agent_cache.clear()
