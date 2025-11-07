"""Wide search module for query expansion and literature discovery.

This module implements a multi-round query expansion system that generates
diverse queries, executes searches, and uses LLM-driven reflection to determine
when sufficient coverage has been achieved.

Public API:
    - run_widesearch: High-level entry point function (recommended)
    - State: Per-run mutable state
    - Deps: External service dependencies
    - graph: Assembled Pydantic Graph for execution
    - Reranker: Semantic reranker for search results
"""

from interaction_finder.widesearch.deps import Deps
from interaction_finder.widesearch.graph import graph
from interaction_finder.widesearch.reranker import Reranker
from interaction_finder.widesearch.run import run_widesearch
from interaction_finder.widesearch.state import State

__all__ = ["run_widesearch", "State", "Deps", "graph", "Reranker"]
