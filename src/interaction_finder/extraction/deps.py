"""External dependencies for the extraction pipeline.

The Deps dataclass holds external services instantiated once per pipeline run
and passed to agents via ctx.deps.
"""

import logging
from dataclasses import dataclass

from interaction_finder.resources import ResourcePool
from interaction_finder.settings import IfetcherConfig


@dataclass
class Deps:
    """External services for the extraction pipeline.

    Attributes:
        resource_pool: Document storage with provenance tracking
        config: Full configuration object
        logger: Logger for warnings and debugging
    """

    resource_pool: ResourcePool
    config: IfetcherConfig
    logger: logging.Logger
