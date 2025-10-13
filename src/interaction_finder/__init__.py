"""
Interaction Finder - A tool for fetching and processing web content and extracting entity pairs.

This package provides functionality for:
- Fetching web pages and converting them to markdown
- Chunking and grouping documents semantically
- Extracting typed entity pairs using AI-driven graph workflows
"""

from .fetcher import PageFetcher, URLCache
from .settings import IfetcherConfig
from . import extraction_graph
# from .task_planner import TaskPlanner, TaskList, TaskRule, TaskNode, TaskStatus

__all__ = ["PageFetcher", "URLCache", "IfetcherConfig", "extraction_graph"]
