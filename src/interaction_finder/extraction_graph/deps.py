"""
Dependency injection for the extraction graph pipeline.

This module defines the Deps dataclass that holds all external services
and configuration needed by the graph nodes and agents.
"""

from dataclasses import dataclass
from typing import List
from pydantic_ai.models import Model

from ..fetcher import PageFetcher, URLCache
from ..settings import IfetcherConfig
from ..resources import ResourcePool


@dataclass
class Deps:
    """
    External dependencies for the extraction graph pipeline.

    This dataclass follows the pydantic-graph pattern of injecting
    external services rather than using globals.
    """

    # Core services
    page_fetcher: PageFetcher
    url_cache: URLCache
    config: IfetcherConfig
    resource_pool: ResourcePool

    # AI model for agents
    model: Model | str

    # Entity configuration derived from task settings
    entity_kinds: List[str]
    relation_type: str
    task_context: str

    # Display settings
    verbose: bool = False

    # Processing settings
    batch_size: int = 5  # Default batch size for parallel LLM operations

    @classmethod
    def from_config(
        cls,
        config: IfetcherConfig,
        page_fetcher: PageFetcher,
        resource_pool: ResourcePool | None = None,
        model: Model | str | None = None,
        verbose: bool = False,
        batch_size: int | None = None,
    ) -> "Deps":
        """
        Create Deps from configuration with sensible defaults.

        Args:
            config: The interaction finder configuration
            page_fetcher: PageFetcher instance for document retrieval
            resource_pool: ResourcePool for document tracking (creates new if None)
            model: AI model for agents (defaults to config or gpt-4o)
            verbose: Enable verbose progress reporting
            batch_size: Batch size for parallel operations

        Returns:
            Configured Deps instance
        """
        # Determine model from config hierarchy
        if model is None:
            default_agent = config.agents.get("_", config.AgentSpec())
            model = default_agent.llm or "openai:gpt-4o"

        # Extract entity configuration from task settings
        entity_kinds = config.task.get_kind_names()
        relation_type = config.task.relation
        task_context = config.task.context

        # Create ResourcePool if not provided
        if resource_pool is None:
            resource_pool = ResourcePool()

        return cls(
            page_fetcher=page_fetcher,
            url_cache=page_fetcher.cache,
            config=config,
            resource_pool=resource_pool,
            model=model,
            entity_kinds=entity_kinds,
            relation_type=relation_type,
            task_context=task_context,
            verbose=verbose,
            batch_size=batch_size or 5,  # Use provided batch_size or default to 5
        )
