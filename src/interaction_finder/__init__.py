"""Interaction finder - automated extraction of biological interactions from literature.

Public API uses lazy imports to keep CLI startup fast (~0.2s vs ~2s).
All exports remain available as `from interaction_finder import X`.
"""

from typing import TYPE_CHECKING

# Lazy import mapping: module -> list of attribute names
_MODULES = {
    ".logging": [
        "configure_logging",
        "dump_log",
        "error_log_path",
        "get_logger",
        "setup_log_output",
    ],
    ".fetcher": ["PageFetcher", "URLCache"],
    ".settings": ["IfetcherConfig"],
    ".models": ["Term"],
    ".term_parser": ["parse_term_line"],
    ".resources": ["ResourcePool", "ResourceId", "ResourceQuote"],
    ".agent_config": ["agent_getter", "get_agent", "clear_agent_cache"],
    ".checkpoint": [
        "PipelineCheckpoint",
        "KeywordsStageData",
        "SearchStageData",
        "ExtractionStageData",
    ],
    ".upgrade": [
        "checkpoint_stage",
        "create_empty_checkpoint",
        "ensure_keywords",
        "ensure_search",
        "ensure_extraction",
    ],
}

# Build reverse mapping and __all__
_IMPORTS = {name: mod for mod, names in _MODULES.items() for name in names}
__all__ = list(_IMPORTS.keys())


def __getattr__(name: str):
    if name not in _IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(_IMPORTS[name], __package__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return __all__


# Type stubs for IDE support (not executed at runtime)
if TYPE_CHECKING:
    from .logging import (
        configure_logging,
        dump_log,
        error_log_path,
        get_logger,
        setup_log_output,
    )
    from .fetcher import PageFetcher, URLCache
    from .settings import IfetcherConfig
    from .models import Term
    from .term_parser import parse_term_line
    from .resources import ResourcePool, ResourceId, ResourceQuote
    from .agent_config import agent_getter, get_agent, clear_agent_cache
    from .checkpoint import (
        PipelineCheckpoint,
        KeywordsStageData,
        SearchStageData,
        ExtractionStageData,
    )
    from .upgrade import (
        checkpoint_stage,
        create_empty_checkpoint,
        ensure_keywords,
        ensure_search,
        ensure_extraction,
    )
