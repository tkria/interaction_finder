"""Unified agent configuration and caching system.

Provides a consistent interface for creating and caching LLM agents across
all pipeline stages (keywords, widesearch, extraction) with multi-tier
configuration resolution.
"""

from typing import Any, Callable, Type

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from interaction_finder.settings import IfetcherConfig


# Global cache for agent instances
# Cache key: (module, agent, model, frozen_config_params)
_agent_cache: dict[tuple[str, str, str, frozenset[tuple[str, Any]]], Agent] = {}


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
    model = agent_spec.llm or default_model
    retries = agent_spec.retries if agent_spec.retries is not None else default_retries
    instrument = agent_spec.instrument
    effective_system_prompt = agent_spec.system_prompt or system_prompt
    # Handle ModelSettings
    model_settings = default_model_settings
    if agent_spec.model_settings:
        # Convert dict to ModelSettings
        model_settings = ModelSettings(**agent_spec.model_settings)
    # Build cache key from all configuration parameters
    parallel_calls = None
    if model_settings:
        if isinstance(model_settings, dict):
            parallel_calls = model_settings.get("parallel_tool_calls")
        else:
            parallel_calls = model_settings.parallel_tool_calls

    config_params = frozenset(
        [
            ("model", model),
            ("retries", retries),
            ("instrument", instrument),
            ("system_prompt", effective_system_prompt),
            ("parallel_tool_calls", parallel_calls),
        ]
    )
    cache_key = (module, agent, model, config_params)
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
