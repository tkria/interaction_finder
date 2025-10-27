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

__all__ = [
    "PageFetcher",
    "URLCache",
    "IfetcherConfig",
    "Term",
    "parse_term_line",
    "ResourcePool",
    "ResourceId",
    "ResourceQuote",
]
