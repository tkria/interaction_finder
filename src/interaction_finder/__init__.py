"""Interaction finder - automated extraction of biological interactions from literature."""

from .fetcher import PageFetcher, URLCache
from .settings import IfetcherConfig
from .models import Term
from .term_parser import parse_term_line
from .resources import (
    ResourcePool,
    ResourceId,
    ResourceQuote,
)
from .agent_config import agent_getter, get_agent, clear_agent_cache
from .checkpoint import (
    PipelineCheckpoint,
    KeywordsStageData,
    SearchStageData,
    ExtractionStageData,
    _rebuild_models,
)

# Rebuild checkpoint models to resolve forward references after imports
_rebuild_models()

__all__ = [
    "PageFetcher",
    "URLCache",
    "IfetcherConfig",
    "Term",
    "parse_term_line",
    "ResourcePool",
    "ResourceId",
    "ResourceQuote",
    "agent_getter",
    "get_agent",
    "clear_agent_cache",
    "PipelineCheckpoint",
    "KeywordsStageData",
    "SearchStageData",
    "ExtractionStageData",
]
