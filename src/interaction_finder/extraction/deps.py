"""External dependencies for the extraction pipeline.

The Deps dataclass holds external services instantiated once per pipeline run
and passed to agents via ctx.deps.
"""

import logging
from dataclasses import dataclass

from interaction_finder.resources import ResourcePool


@dataclass
class Deps:
    """External services for the extraction pipeline.

    Attributes:
        resource_pool: Document storage with provenance tracking
        extraction_model: LLM model for entity and pair extraction/assessment
        judge_model: LLM model for final judgment on pairs
        logger: Logger for warnings and debugging
    """

    resource_pool: ResourcePool
    extraction_model: str
    judge_model: str
    logger: logging.Logger
