"""Utilities for working with pydantic-ai agents."""

from contextlib import contextmanager


@contextmanager
def rename_agent(agent, name: str):
    """Temporarily rename an agent for span naming purposes.

    This context manager temporarily changes the agent's _name attribute,
    which affects the span name created by pydantic-ai's instrumentation.
    The original name is restored when the context exits.

    Parameters:
        agent: The pydantic-ai Agent to rename
        name: str — temporary name for the agent (affects logfire span name)

    Example:
        >>> with rename_agent(agent, "MyCustomSpanName"):
        ...     result = await agent.run("prompt")
    """
    original_name = agent._name
    try:
        agent._name = name
        yield agent
    finally:
        agent._name = original_name
