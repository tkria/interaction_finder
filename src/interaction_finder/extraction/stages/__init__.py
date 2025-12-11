"""Pipeline stages for association extraction.

Each stage is an async function with signature:
    async def stage_name(state: State, deps: Deps) -> bool

Stages modify state in-place. Return False to abort pipeline early.
"""

from interaction_finder.extraction.stages.process_documents import process_documents
from interaction_finder.extraction.stages.consolidate_entities import (
    consolidate_entities,
)
from interaction_finder.extraction.stages.consolidate_relationships import (
    consolidate_relationships,
)
from interaction_finder.extraction.stages.sweep_co_mentions import sweep_co_mentions
from interaction_finder.extraction.stages.consolidate_new_relationships import (
    consolidate_new_relationships,
)
from interaction_finder.extraction.stages.judge_cross_document import (
    judge_cross_document,
)
from interaction_finder.extraction.stages.finalize import finalize

__all__ = [
    "process_documents",
    "consolidate_entities",
    "consolidate_relationships",
    "sweep_co_mentions",
    "consolidate_new_relationships",
    "judge_cross_document",
    "finalize",
]
