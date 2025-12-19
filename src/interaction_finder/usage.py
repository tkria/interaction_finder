"""Usage tracking for LLM token consumption across pipeline stages.

Tracks token usage per agent within each pipeline stage, enabling:
- Per-agent breakdown of token consumption
- Per-model aggregation for cost calculation
- Persistence across checkpoint save/resume cycles
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field
from pydantic_ai.usage import RunUsage

if TYPE_CHECKING:
    from pydantic_ai import Agent
    from pydantic_ai.result import AgentRunResult


class AgentUsage(BaseModel):
    """Usage for an agent, including which model was used."""

    model: str
    usage: RunUsage = Field(default_factory=RunUsage)

    def copy_deep(self) -> "AgentUsage":
        """Create a deep copy with independent RunUsage instance."""
        return AgentUsage(
            model=self.model,
            usage=RunUsage(
                requests=self.usage.requests,
                tool_calls=self.usage.tool_calls,
                input_tokens=self.usage.input_tokens,
                output_tokens=self.usage.output_tokens,
                cache_write_tokens=self.usage.cache_write_tokens,
                cache_read_tokens=self.usage.cache_read_tokens,
                input_audio_tokens=self.usage.input_audio_tokens,
                cache_audio_read_tokens=self.usage.cache_audio_read_tokens,
                output_audio_tokens=self.usage.output_audio_tokens,
                details=dict(self.usage.details) if self.usage.details else {},
            ),
        )


# Type alias for stage usage (agent name -> usage data)
StageUsage = dict[str, AgentUsage]


class PipelineUsage(BaseModel):
    """Token usage across pipeline stages, per agent."""

    keywords: dict[str, AgentUsage] = Field(default_factory=dict)
    search: dict[str, AgentUsage] = Field(default_factory=dict)
    extraction: dict[str, AgentUsage] = Field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed across all stages and agents."""
        return sum(
            au.usage.total_tokens
            for stage in [self.keywords, self.search, self.extraction]
            for au in stage.values()
        )

    def total_by_model(self) -> dict[str, int]:
        """Aggregate tokens by model for cost calculation."""
        totals: dict[str, int] = {}
        for stage in [self.keywords, self.search, self.extraction]:
            for au in stage.values():
                totals[au.model] = totals.get(au.model, 0) + au.usage.total_tokens
        return totals


def _get_model_id(agent: Agent[Any, Any]) -> str:
    """Get provider:model_name identifier from an agent."""
    model = agent.model
    if model is None:
        return "unknown"
    if isinstance(model, str):
        return model
    return f"{model.system}:{model.model_name}"


def record_usage(
    usage_dict: StageUsage,
    agent_name: str,
    agent: Agent[Any, Any],
    result_or_usage: AgentRunResult[Any] | RunUsage,
) -> None:
    """Record usage for an agent, accumulating if already present.

    If the same agent is called with different models, usage is tracked
    separately using "{agent_name}:{model}" as the key.

    Parameters:
        usage_dict: Stage-specific usage dictionary to update
        agent_name: Name of the agent (e.g., "query_expander", "entity")
        agent: Pydantic AI agent instance (used to extract model identifier)
        result_or_usage: AgentRunResult from agent.run() or RunUsage object
            (for cases where usage is accumulated across multiple calls)
    """
    model = _get_model_id(agent)
    # Extract usage from result or use directly
    if isinstance(result_or_usage, RunUsage):
        usage = result_or_usage
    else:
        usage = result_or_usage.usage()
    if agent_name not in usage_dict:
        usage_dict[agent_name] = AgentUsage(model=model, usage=usage)
    elif usage_dict[agent_name].model == model:
        usage_dict[agent_name].usage += usage
    else:
        # Different model for same agent - track separately
        key = f"{agent_name}:{model}"
        if key not in usage_dict:
            usage_dict[key] = AgentUsage(model=model, usage=usage)
        else:
            usage_dict[key].usage += usage
