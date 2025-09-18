"""
Clean dependency injection for extraction graph V2.

Following pydantic-graph pattern: only external services,
no derived configuration data (that belongs in State).
"""

from dataclasses import dataclass
from typing import Union, Optional, List

from pydantic_ai.models import Model

from ..settings import IfetcherConfig
from ..fetcher import PageFetcher


@dataclass
class ExtractionDeps:
    """
    External dependencies for the extraction pipeline.

    Clean separation: only services and connections,
    no derived config (moved to State).
    """

    # LLM for agents
    model: Union[Model, str]

    # Configuration
    config: IfetcherConfig

    # Document fetching service
    page_fetcher: PageFetcher

    # Target term for extraction context
    target_term: Optional[str] = None

    # Current resources for validation (set by nodes before agent runs)
    current_resources: Optional[List] = None

    @classmethod
    def from_config(
        cls,
        config: IfetcherConfig,
        page_fetcher: PageFetcher,
        model: Union[Model, str, None] = None,
        target_term: Optional[str] = None,
        current_resources: Optional[List] = None,
    ) -> "ExtractionDeps":
        """
        Create deps from configuration with sensible defaults.

        Args:
            config: Main configuration
            page_fetcher: PageFetcher instance
            model: Optional model override
            target_term: Optional target term for extraction context
            current_resources: Optional list of resources for validation

        Returns:
            ExtractionDeps instance
        """
        # Determine model from config hierarchy
        if model is None:
            default_agent = config.agents.get("_", config.AgentSpec())
            model = default_agent.llm or "openai:gpt-4o"

        return cls(
            model=model,
            config=config,
            page_fetcher=page_fetcher,
            target_term=target_term,
            current_resources=current_resources,
        )

    def get_entity_kinds(self) -> list[str]:
        """Get entity kinds from task configuration."""
        return self.config.task.get_kind_names()

    def get_relation_type(self) -> str:
        """Get relation type from task configuration."""
        return self.config.task.relation

    def get_task_context(self) -> str:
        """Get task context from configuration."""
        return self.config.task.context
