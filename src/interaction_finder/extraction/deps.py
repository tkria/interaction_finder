"""External dependencies for the extraction pipeline.

The Deps dataclass holds external services instantiated once per pipeline run
and passed to agents via ctx.deps.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from interaction_finder.checkpoint import PipelineCheckpoint

from interaction_finder.resources import ResourcePool
from interaction_finder.settings import IfetcherConfig
from interaction_finder.usage import StageUsage


class ProgressProtocol(Protocol):
    """Protocol for progress counters."""

    documents_processed: int
    documents_in_progress: int
    documents_total: int
    entities_found: int
    pairs_found: int
    pairs_assessed: int
    pairs_in_progress: int
    pairs_total: int
    quotes_validated: int
    quotes_failed: int
    unique_pairs: int
    judgments_in_progress: int
    accepted: int
    rejected: int

    def set_phase_extracting(self) -> None: ...
    def set_phase_validating(self) -> None: ...
    def set_phase_proximal(self) -> None: ...
    def set_phase_extracting_pairs(self) -> None: ...
    def set_phase_assessing(self) -> None: ...
    def set_phase_judging(self) -> None: ...
    def set_phase_finalizing(self) -> None: ...
    def set_phase_idle(self) -> None: ...
    def update(self) -> None: ...


@dataclass
class Deps:
    """External services for the extraction pipeline.

    Attributes:
        resource_pool: Document storage with provenance tracking
        config: Full configuration object
        logger: Logger for warnings and debugging
        progress: Optional progress counter for live display
        agent_semaphore: Semaphore to limit concurrent agent calls
        checkpoint_path: Optional path for saving partial checkpoints
        input_checkpoint: Input checkpoint to preserve prior stage data
    """

    resource_pool: ResourcePool
    config: IfetcherConfig
    logger: logging.Logger
    progress: ProgressProtocol | None = None
    agent_semaphore: asyncio.Semaphore = None  # type: ignore
    checkpoint_path: str | None = None
    input_checkpoint: "PipelineCheckpoint | None" = None
    usage: StageUsage = field(default_factory=dict)  # LLM usage tracking
