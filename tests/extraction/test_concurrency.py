"""Tests for agent concurrency limiting with semaphore."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from interaction_finder.extraction.deps import Deps
from interaction_finder.resources import ResourcePool
from interaction_finder.settings import IfetcherConfig


@pytest.fixture
def mock_deps():
    """Create mock Deps with configured semaphore."""
    config = IfetcherConfig()
    resource_pool = ResourcePool()
    logger = MagicMock()
    # Create semaphore with limit of 3 for testing
    agent_semaphore = asyncio.Semaphore(3)

    return Deps(
        resource_pool=resource_pool,
        config=config,
        logger=logger,
        progress=None,
        agent_semaphore=agent_semaphore,
    )


@pytest.mark.asyncio
async def test_semaphore_limits_concurrent_calls(mock_deps):
    """Verify semaphore limits concurrent agent calls to configured max."""
    concurrent_calls = []
    max_concurrent = 0

    async def mock_agent_call(call_id: int):
        """Simulate an agent call that tracks concurrency."""
        async with mock_deps.agent_semaphore:
            concurrent_calls.append(call_id)
            nonlocal max_concurrent
            max_concurrent = max(max_concurrent, len(concurrent_calls))
            await asyncio.sleep(0.01)  # Simulate work
            concurrent_calls.remove(call_id)

    # Launch 10 calls, but semaphore limit is 3
    tasks = [mock_agent_call(i) for i in range(10)]
    await asyncio.gather(*tasks)

    # Verify max concurrent never exceeded semaphore limit
    assert max_concurrent <= 3
    assert max_concurrent > 0  # Ensure some concurrency happened


@pytest.mark.asyncio
async def test_semaphore_with_limit_one():
    """Verify semaphore with limit=1 forces sequential execution."""
    semaphore = asyncio.Semaphore(1)
    execution_order = []

    async def mock_call(call_id: int):
        async with semaphore:
            execution_order.append(f"start_{call_id}")
            await asyncio.sleep(0.001)
            execution_order.append(f"end_{call_id}")

    tasks = [mock_call(i) for i in range(3)]
    await asyncio.gather(*tasks)

    # With limit=1, each call must complete before next starts
    # So we should see: start_X, end_X, start_Y, end_Y, start_Z, end_Z
    for i in range(3):
        start_idx = execution_order.index(f"start_{i}")
        end_idx = execution_order.index(f"end_{i}")
        # End must come after start
        assert end_idx > start_idx
        # No other starts between this start and end (sequential)
        between = execution_order[start_idx + 1 : end_idx]
        assert not any(item.startswith("start_") for item in between)


@pytest.mark.asyncio
async def test_config_default_value():
    """Verify default agent_concurrency_limit is 10."""
    config = IfetcherConfig()
    assert config.stage.extraction.agent_concurrency_limit == 10


@pytest.mark.asyncio
async def test_config_validation_min():
    """Verify agent_concurrency_limit cannot be less than 1."""
    with pytest.raises(ValueError):
        IfetcherConfig.model_validate(
            {"stage": {"extraction": {"agent_concurrency_limit": 0}}}
        )


@pytest.mark.asyncio
async def test_config_validation_max():
    """Verify agent_concurrency_limit cannot exceed 100."""
    with pytest.raises(ValueError):
        IfetcherConfig.model_validate(
            {"stage": {"extraction": {"agent_concurrency_limit": 101}}}
        )


@pytest.mark.asyncio
async def test_config_custom_value():
    """Verify custom agent_concurrency_limit can be set."""
    config = IfetcherConfig.model_validate(
        {"stage": {"extraction": {"agent_concurrency_limit": 25}}}
    )
    assert config.stage.extraction.agent_concurrency_limit == 25


@pytest.mark.asyncio
async def test_semaphore_all_calls_complete(mock_deps):
    """Verify all agent calls complete even with limiting."""
    call_count = 0

    async def mock_agent_call():
        async with mock_deps.agent_semaphore:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.001)

    # Launch 20 calls with limit of 3
    num_calls = 20
    tasks = [mock_agent_call() for _ in range(num_calls)]
    await asyncio.gather(*tasks)

    # All calls should complete
    assert call_count == num_calls
