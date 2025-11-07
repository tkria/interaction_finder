"""Wide search module for query expansion and literature discovery.

This module implements a multi-round query expansion system that generates
diverse queries, executes searches, and uses LLM-driven reflection to determine
when sufficient coverage has been achieved.

Public API:
    - run_widesearch: High-level entry point function (recommended)
    - run_widesearch_with_checkpoint: Entry point returning complete checkpoint
    - WidesearchCheckpoint: Checkpoint model with results, queries, and resource pool
    - State: Per-run mutable state
    - Deps: External service dependencies
    - graph: Assembled Pydantic Graph for execution
    - Reranker: Semantic reranker for search results
    - WidesearchProgress: Live progress display for search operations
    - DummyProgress: No-op progress counter for disabled display
"""

from interaction_finder.widesearch.deps import Deps
from interaction_finder.widesearch.graph import graph
from interaction_finder.widesearch.models import WidesearchCheckpoint
from interaction_finder.widesearch.progress import DummyProgress, WidesearchProgress
from interaction_finder.widesearch.reranker import Reranker
from interaction_finder.widesearch.run import (
    fetch_and_populate_results,
    run_widesearch,
    run_widesearch_with_checkpoint,
)
from interaction_finder.widesearch.state import State

__all__ = [
    "run_widesearch",
    "run_widesearch_with_checkpoint",
    "fetch_and_populate_results",
    "WidesearchCheckpoint",
    "State",
    "Deps",
    "graph",
    "Reranker",
    "WidesearchProgress",
    "DummyProgress",
]
