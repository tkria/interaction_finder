"""Integration tests for progress counters with concurrent operations."""

import asyncio

import pytest

from interaction_finder.extraction.progress import ExtractionProgress
from interaction_finder.widesearch.progress import WidesearchProgress


@pytest.mark.asyncio
async def test_extraction_concurrent_document_simulation():
    """Simulate concurrent document processing with progress tracking."""
    progress = ExtractionProgress()
    num_docs = 10

    # Initialize
    progress.documents_total = num_docs
    progress.documents_in_progress = num_docs

    async def process_document(doc_id: int):
        """Simulate document processing."""
        await asyncio.sleep(0.01)  # Simulate work
        # Update counter
        progress.documents_processed += 1
        progress.documents_in_progress = max(0, progress.documents_in_progress - 1)

    # Process concurrently using as_completed (like the real implementation)
    tasks = [process_document(i) for i in range(num_docs)]
    for coro in asyncio.as_completed(tasks):
        await coro

    # Verify final state
    assert progress.documents_processed == num_docs
    assert progress.documents_in_progress == 0
    assert progress.documents_total == num_docs


@pytest.mark.asyncio
async def test_extraction_concurrent_pairs_simulation():
    """Simulate concurrent pair assessment with progress tracking."""
    progress = ExtractionProgress()
    num_pairs = 50

    # Initialize
    progress.pairs_found = num_pairs
    progress.pairs_in_progress = num_pairs

    async def assess_pair(pair_id: int):
        """Simulate pair assessment."""
        await asyncio.sleep(0.001)  # Simulate work
        # Update counters
        progress.pairs_assessed += 1
        progress.pairs_in_progress = max(0, progress.pairs_in_progress - 1)

    # Process concurrently
    tasks = [assess_pair(i) for i in range(num_pairs)]
    for coro in asyncio.as_completed(tasks):
        await coro

    # Verify final state
    assert progress.pairs_assessed == num_pairs
    assert progress.pairs_in_progress == 0
    assert progress.pairs_found == num_pairs


@pytest.mark.asyncio
async def test_extraction_concurrent_judgments_simulation():
    """Simulate concurrent judgment with progress tracking."""
    progress = ExtractionProgress()
    num_pairs = 30

    # Initialize
    progress.unique_pairs = num_pairs
    progress.judgments_in_progress = num_pairs

    async def judge_pair(pair_id: int):
        """Simulate pair judgment."""
        await asyncio.sleep(0.001)  # Simulate work
        # Update counters (alternating accept/reject)
        if pair_id % 2 == 0:
            progress.accepted += 1
        else:
            progress.rejected += 1
        progress.judgments_in_progress = max(0, progress.judgments_in_progress - 1)

    # Process concurrently
    tasks = [judge_pair(i) for i in range(num_pairs)]
    for coro in asyncio.as_completed(tasks):
        await coro

    # Verify final state
    assert progress.accepted + progress.rejected == num_pairs
    assert progress.judgments_in_progress == 0
    assert progress.unique_pairs == num_pairs


@pytest.mark.asyncio
async def test_widesearch_concurrent_searches_simulation():
    """Simulate concurrent searches with progress tracking."""
    progress = WidesearchProgress()
    num_searches = 8

    # Initialize
    progress.searches_total_this_round = num_searches
    progress.searches_in_progress = num_searches
    progress.searches_run_this_round = 0

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
        progress.searches_run += 1
        progress.searches_run_this_round += 1
        progress.searches_in_progress = max(0, progress.searches_in_progress - 1)
        progress.results_found += len(results)

    # Verify final state
    assert progress.searches_run_this_round == num_searches
    assert progress.searches_in_progress == 0
    assert progress.searches_total_this_round == num_searches
    assert progress.results_found == total_results


@pytest.mark.asyncio
async def test_extraction_multi_stage_pipeline():
    """Simulate multi-stage extraction pipeline with progress tracking."""
    progress = ExtractionProgress()

    # Stage 1: Process documents
    num_docs = 5
    progress.documents_total = num_docs
    progress.documents_in_progress = num_docs

    for _ in range(num_docs):
        await asyncio.sleep(0.001)
        progress.documents_processed += 1
        progress.documents_in_progress -= 1
        # Each doc finds some entities and pairs
        progress.entities_found += 25
        progress.pairs_found += 30
        progress.pairs_in_progress += 30

    # Stage 2: Assess pairs
    while progress.pairs_in_progress > 0:
        await asyncio.sleep(0.001)
        progress.pairs_assessed += 1
        progress.pairs_in_progress -= 1

    # Stage 3: Judge unique pairs
    progress.unique_pairs = 50
    progress.judgments_in_progress = 50

    while progress.judgments_in_progress > 0:
        await asyncio.sleep(0.001)
        if progress.judgments_in_progress % 2 == 0:
            progress.accepted += 1
        else:
            progress.rejected += 1
        progress.judgments_in_progress -= 1

    # Verify final state
    assert progress.documents_processed == num_docs
    assert progress.documents_in_progress == 0
    assert progress.entities_found == num_docs * 25
    assert progress.pairs_assessed == progress.pairs_found
    assert progress.pairs_in_progress == 0
    assert progress.accepted + progress.rejected == progress.unique_pairs
    assert progress.judgments_in_progress == 0


def test_progress_counters_never_negative():
    """Verify max(0, ...) guards prevent negative counters."""
    progress = ExtractionProgress()

    # Try to decrement below zero (simulating over-decrement bug)
    progress.documents_in_progress = max(0, progress.documents_in_progress - 1)
    assert progress.documents_in_progress == 0

    progress.pairs_in_progress = max(0, progress.pairs_in_progress - 5)
    assert progress.pairs_in_progress == 0

    progress.judgments_in_progress = max(0, progress.judgments_in_progress - 1)
    assert progress.judgments_in_progress == 0


def test_widesearch_round_reset():
    """Verify per-round counters reset correctly between rounds."""
    progress = WidesearchProgress()

    # Round 1
    progress.searches_run = 5
    progress.searches_run_this_round = 5
    progress.searches_total_this_round = 5

    # Reset for round 2 (simulating GenerateQueriesNode)
    progress.searches_run_this_round = 0
    progress.searches_in_progress = 0
    progress.searches_total_this_round = 0

    # Round 2
    progress.searches_run = 10  # Cumulative
    progress.searches_run_this_round = 5
    progress.searches_total_this_round = 5

    assert progress.searches_run == 10  # Total preserved
    assert progress.searches_run_this_round == 5  # Per-round reset worked
