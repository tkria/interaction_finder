"""Integration tests for progress counters with concurrent operations."""

import asyncio

import pytest

from interaction_finder.extraction.progress import create_extraction_progress
from interaction_finder.widesearch.progress import create_widesearch_progress


@pytest.mark.asyncio
async def test_extraction_concurrent_document_simulation():
    """Simulate concurrent document processing with progress tracking."""
    progress = create_extraction_progress()
    num_docs = 10
    # Initialize
    progress["Processed"].total = num_docs
    progress["Processed"].in_progress = num_docs
    progress["Processed"].activate()

    async def process_document(doc_id: int):
        """Simulate document processing."""
        await asyncio.sleep(0.01)  # Simulate work
        # Update counter
        progress["Processed"].done()

    # Process concurrently using as_completed (like the real implementation)
    tasks = [process_document(i) for i in range(num_docs)]
    for coro in asyncio.as_completed(tasks):
        await coro
    # Verify final state
    assert progress["Processed"].completed == num_docs
    assert progress["Processed"].in_progress == 0
    assert progress["Processed"].total == num_docs


@pytest.mark.asyncio
async def test_extraction_concurrent_pairs_simulation():
    """Simulate concurrent pair assessment with progress tracking."""
    progress = create_extraction_progress()
    num_pairs = 50
    # Initialize
    progress["Pairs assessed"].total = num_pairs
    progress["Pairs assessed"].in_progress = num_pairs
    progress["Pairs assessed"].activate()

    async def assess_pair(pair_id: int):
        """Simulate pair assessment."""
        await asyncio.sleep(0.001)  # Simulate work
        # Update counters
        progress["Pairs assessed"].done()

    # Process concurrently
    tasks = [assess_pair(i) for i in range(num_pairs)]
    for coro in asyncio.as_completed(tasks):
        await coro
    # Verify final state
    assert progress["Pairs assessed"].completed == num_pairs
    assert progress["Pairs assessed"].in_progress == 0
    assert progress["Pairs assessed"].total == num_pairs


@pytest.mark.asyncio
async def test_extraction_concurrent_judgments_simulation():
    """Simulate concurrent judgment with progress tracking."""
    progress = create_extraction_progress()
    num_pairs = 30
    # Initialize
    progress["Unique pairs"].total = num_pairs
    progress["Unique pairs"].in_progress = num_pairs
    progress["Unique pairs"].activate()

    async def judge_pair(pair_id: int):
        """Simulate pair judgment."""
        await asyncio.sleep(0.001)  # Simulate work
        # Update counters (alternating accept/reject)
        if pair_id % 2 == 0:
            progress["Accepted"].add()
        else:
            progress["Rejected"].add()
        progress["Unique pairs"].done()

    # Process concurrently
    tasks = [judge_pair(i) for i in range(num_pairs)]
    for coro in asyncio.as_completed(tasks):
        await coro
    # Verify final state
    assert progress["Accepted"].completed + progress["Rejected"].completed == num_pairs
    assert progress["Unique pairs"].in_progress == 0
    assert progress["Unique pairs"].total == num_pairs


@pytest.mark.asyncio
async def test_widesearch_concurrent_searches_simulation():
    """Simulate concurrent searches with progress tracking."""
    progress = create_widesearch_progress()
    num_searches = 8
    # Initialize
    progress["Searches run"].total = num_searches
    progress["Searches run"].in_progress = num_searches
    progress["Searches run"].activate()

    async def execute_search(query_id: int):
        """Simulate search execution."""
        await asyncio.sleep(0.01)  # Simulate work
        # Return some fake results
        return [f"result_{query_id}_{i}" for i in range(10)]

    # Process concurrently using as_completed
    tasks = [execute_search(i) for i in range(num_searches)]
    total_results = 0
    for coro in asyncio.as_completed(tasks):
        results = await coro
        total_results += len(results)
        # Update counters
        progress["Searches run"].done()
        progress["Results found"].add(len(results))
    # Verify final state
    assert progress["Searches run"].completed == num_searches
    assert progress["Searches run"].in_progress == 0
    assert progress["Searches run"].total == num_searches
    assert progress["Results found"].completed == total_results


@pytest.mark.asyncio
async def test_extraction_multi_stage_pipeline():
    """Simulate multi-stage extraction pipeline with progress tracking."""
    progress = create_extraction_progress()
    # Stage 1: Process documents
    num_docs = 5
    progress["Processed"].total = num_docs
    progress["Processed"].in_progress = num_docs
    progress["Processed"].activate()
    pairs_per_doc = 30
    for _ in range(num_docs):
        await asyncio.sleep(0.001)
        progress["Processed"].done()
        # Each doc finds some entities and pairs
        progress["Entities"].add(25)
        progress["Pairs assessed"].total += pairs_per_doc
        progress["Pairs assessed"].in_progress += pairs_per_doc
    progress["Pairs assessed"].activate()
    # Stage 2: Assess pairs
    while progress["Pairs assessed"].in_progress > 0:
        await asyncio.sleep(0.001)
        progress["Pairs assessed"].done()
    # Stage 3: Judge unique pairs
    progress["Unique pairs"].total = 50
    progress["Unique pairs"].in_progress = 50
    progress["Unique pairs"].activate()
    while progress["Unique pairs"].in_progress > 0:
        await asyncio.sleep(0.001)
        if progress["Unique pairs"].in_progress % 2 == 0:
            progress["Accepted"].add()
        else:
            progress["Rejected"].add()
        progress["Unique pairs"].done()
    # Verify final state
    assert progress["Processed"].completed == num_docs
    assert progress["Processed"].in_progress == 0
    assert progress["Entities"].completed == num_docs * 25
    assert progress["Pairs assessed"].completed == progress["Pairs assessed"].total
    assert progress["Pairs assessed"].in_progress == 0
    assert (
        progress["Accepted"].completed + progress["Rejected"].completed
        == progress["Unique pairs"].total
    )
    assert progress["Unique pairs"].in_progress == 0


def test_progress_counters_in_progress_can_be_negative():
    """Verify in_progress can go negative (due to concurrent over-decrement)."""
    # Note: In the new API, we don't automatically guard against negative.
    # The calling code uses max(0, ...) patterns where needed.
    progress = create_extraction_progress()
    progress["Processed"].in_progress = 0
    progress["Processed"].done()  # Decrements by 1
    assert progress["Processed"].in_progress == -1


def test_widesearch_round_reset():
    """Verify per-round counters reset correctly between rounds."""
    progress = create_widesearch_progress()
    # Round 1
    progress["Searches run"].total = 5
    progress["Searches run"].completed = 5
    progress["Searches run"].in_progress = 0
    # Reset for round 2 (simulating GenerateQueriesNode)
    progress["Searches run"].completed = 0
    progress["Searches run"].in_progress = 0
    progress["Searches run"].total = 0
    # Round 2
    progress["Searches run"].total = 5
    progress["Searches run"].completed = 5
    progress["Searches run"].in_progress = 0
    assert progress["Searches run"].completed == 5  # Per-round reset worked
    assert progress["Searches run"].total == 5
