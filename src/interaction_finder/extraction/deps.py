"""External dependencies for the extraction pipeline.

The Deps dataclass holds external services instantiated once per pipeline run
and passed to agents via ctx.deps.
"""

import logging
from dataclasses import dataclass
from typing import Any

from interaction_finder.resources import ResourcePool


@dataclass
class Deps:
    """External services for the extraction pipeline.

    Attributes:
        resource_pool: Document storage with provenance tracking
        config: Configuration dict with model names and thresholds
        logger: Logger for warnings and debugging
    """

    resource_pool: ResourcePool
    config: dict[str, Any]
    logger: logging.Logger
