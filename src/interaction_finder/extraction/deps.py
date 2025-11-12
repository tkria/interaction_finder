"""External dependencies for the extraction pipeline.

The Deps dataclass holds external services instantiated once per pipeline run
and passed to agents via ctx.deps.
"""

import logging
from dataclasses import dataclass
from typing import Protocol


from interaction_finder.resources import ResourcePool
from interaction_finder.settings import IfetcherConfig


class ProgressProtocol(Protocol):
    """Protocol for progress counters."""

    documents_processed: int
    documents_total: int
    entities_found: int
    pairs_found: int
    pairs_assessed: int
    pairs_total: int
    quotes_validated: int
    quotes_failed: int
    unique_pairs: int
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
    """

    resource_pool: ResourcePool
    config: IfetcherConfig
    logger: logging.Logger
    progress: ProgressProtocol | None = None
