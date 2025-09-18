"""
Parallelism control utilities for extraction graph V2.

Provides higher-order functions for managing concurrent operations
with configurable limits.
"""

import asyncio
import logging
from typing import List, Callable, Awaitable, TypeVar, Any, Optional

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


async def with_parallelism_limit(
    tasks: List[Awaitable[T]], parallelism: int = 0, description: str = "operations"
) -> List[T]:
    """
    Execute async tasks with optional parallelism limiting.

    Args:
        tasks: List of awaitable tasks to execute
        parallelism: Max concurrent operations. 0 = unlimited (default)
        description: Human-readable description for logging

    Returns:
        List of results in same order as input tasks

    Raises:
        Exception: Re-raises any exceptions from tasks (use return_exceptions=True for gather)
    """
    if not tasks:
        return []

    if parallelism == 0:
        # Unlimited parallelism - current behavior
        logger.debug(f"Executing {len(tasks)} {description} with unlimited parallelism")
        return await asyncio.gather(*tasks, return_exceptions=True)
    else:
        # Limited parallelism with semaphore
        logger.debug(
            f"Executing {len(tasks)} {description} with parallelism limit of {parallelism}"
        )
        semaphore = asyncio.Semaphore(parallelism)

        async def execute_with_limit(task: Awaitable[T]) -> T:
            async with semaphore:
                return await task

        limited_tasks = [execute_with_limit(task) for task in tasks]
        return await asyncio.gather(*limited_tasks, return_exceptions=True)


def create_parallelism_limiter(parallelism: int = 0, description: str = "operations"):
    """
    Create a reusable parallelism limiter function.

    Args:
        parallelism: Max concurrent operations. 0 = unlimited
        description: Human-readable description for logging

    Returns:
        Async function that takes a list of tasks and executes them with the configured limit

    Example:
        limiter = create_parallelism_limiter(parallelism=3, description="document processing")
        results = await limiter(tasks)
    """

    async def limiter(tasks: List[Awaitable[T]]) -> List[T]:
        return await with_parallelism_limit(tasks, parallelism, description)

    return limiter


async def with_parallelism_control(
    items: List[T],
    async_func: Callable[[T], Awaitable[R]],
    parallelism: int = 0,
    description: str = "items",
) -> List[R]:
    """
    Apply an async function to a list of items with parallelism control.

    Args:
        items: List of items to process
        async_func: Async function to apply to each item
        parallelism: Max concurrent operations. 0 = unlimited
        description: Human-readable description for logging

    Returns:
        List of results in same order as input items

    Example:
        # Process documents with parallelism limit
        results = await with_parallelism_control(
            documents,
            extract_entities_from_document,
            parallelism=3,
            description="document extraction"
        )
    """
    if not items:
        return []

    # Create tasks by applying the function to each item
    tasks = [async_func(item) for item in items]

    # Execute with parallelism control
    return await with_parallelism_limit(tasks, parallelism, description)


class ParallelismController:
    """
    Reusable parallelism controller for consistent behavior across operations.

    Example:
        controller = ParallelismController(parallelism=3)

        # Use for different operations
        doc_results = await controller.execute(doc_tasks, "document processing")
        entity_results = await controller.execute(entity_tasks, "entity assessment")
    """

    def __init__(self, parallelism: int = 0, default_description: str = "operations"):
        """
        Initialize parallelism controller.

        Args:
            parallelism: Max concurrent operations. 0 = unlimited
            default_description: Default description for logging
        """
        self.parallelism = parallelism
        self.default_description = default_description

    async def execute(
        self, tasks: List[Awaitable[T]], description: Optional[str] = None
    ) -> List[T]:
        """Execute tasks with configured parallelism limit."""
        desc = description or self.default_description
        return await with_parallelism_limit(tasks, self.parallelism, desc)

    async def map(
        self,
        items: List[T],
        async_func: Callable[[T], Awaitable[R]],
        description: Optional[str] = None,
    ) -> List[R]:
        """Apply function to items with configured parallelism limit."""
        desc = description or self.default_description
        return await with_parallelism_control(items, async_func, self.parallelism, desc)

    def create_limiter(self, description: Optional[str] = None):
        """Create a limiter function with this controller's settings."""
        desc = description or self.default_description
        return create_parallelism_limiter(self.parallelism, desc)

    def __str__(self) -> str:
        if self.parallelism == 0:
            return "ParallelismController(unlimited)"
        else:
            return f"ParallelismController(limit={self.parallelism})"


# Convenience functions for common patterns
async def process_documents_parallel(
    documents: List[T],
    processor_func: Callable[[T], Awaitable[R]],
    parallelism: int = 0,
) -> List[R]:
    """Process documents with parallelism control."""
    return await with_parallelism_control(
        documents, processor_func, parallelism, "document processing"
    )


async def assess_entities_parallel(
    entities: List[T],
    assessment_func: Callable[[T], Awaitable[R]],
    parallelism: int = 0,
) -> List[R]:
    """Assess entities with parallelism control."""
    return await with_parallelism_control(
        entities, assessment_func, parallelism, "entity assessment"
    )
