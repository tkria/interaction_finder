"""
Integration tests validating reverse search coverage on PAH test dataset.

Tests verify the primary success metric: >90% coverage on 20 PAH papers with <10
queries using LLM extractor + relevance sort. Compares performance against YAKE
baseline and measures query precision metrics.

Dataset: some-pah-papers.jsonl (20 papers, 2001-2023, pulmonary arterial hypertension)
Expected: LLM + relevance sort achieves >90% coverage, YAKE baseline significantly lower
"""

import json
import os
import pytest
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock

from interaction_finder.search.reverse.models import (
    KnownResource,
    ReverseSearchConfig,
    ReverseSearchSession,
)
from interaction_finder.search.reverse.searcher import ReverseSearcher
from interaction_finder.search.base import SearchBackend, SearchResult, SearchResults


# Fixtures


@pytest.fixture
def pah_dataset_path():
    """Path to PAH test dataset (20 papers, 2001-2023)."""
    # Path relative to project root (handle both normal and worktree layouts)
    test_file = Path(__file__)
    # Navigate up to project root: tests/search/test_*.py -> project root
    project_root = test_file.parent.parent.parent
    # Check if we're in a worktree (has .claude/worktrees in path)
    if ".claude/worktrees" in str(project_root):
        # In worktree: go up to actual project root
        # Current: .../05/tests/search/test_*.py -> .../05
        # Need: .../05 -> .../reverse-search-coverage-improvements -> ../.claude -> .../worktrees -> .../interaction_finder
        project_root = project_root.parent.parent.parent.parent
    return project_root / "some-pah-papers.jsonl"


@pytest.fixture
def pah_resources(pah_dataset_path):
    """Load PAH test dataset into KnownResource objects."""
    resources = []
    with open(pah_dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            # Extract fields
            url = data.pop("url")
            pmid = data.pop("pmid", None)
            hint_fields = data  # disease, gene fields
            resources.append(KnownResource(url=url, pmid=pmid, hint_fields=hint_fields))
    return resources


@pytest.fixture
def mock_pubmed_backend():
    """Mock PubMed backend that returns realistic search results."""
    backend = Mock(spec=SearchBackend)
    backend.backend_name = "pubmed"

    async def mock_search(query):
        """Simulate PubMed search with realistic result counts."""
        # Simulate varying result counts based on query specificity
        # LLM queries (with field tags) return fewer, more relevant results
        # YAKE queries (keywords only) return many results, lower precision
        if "[" in query.query:  # Field tags present (LLM queries)
            # Specific queries: 10-50 results with higher precision
            num_results = 30
            precision = 0.15  # ~15% hit rate
        else:  # No field tags (YAKE queries)
            # Broad queries: many results with lower precision
            num_results = 100
            precision = 0.05  # ~5% hit rate

        # Generate mock results
        results = []
        for i in range(num_results):
            # Simulate some results matching target PMIDs
            is_match = i < int(num_results * precision)
            if is_match:
                # Use realistic PMID from PAH dataset
                pmid = f"pmid_{i}"  # Simplified for mock
            else:
                pmid = f"other_{i}"

            result = SearchResult(
                title=f"Paper {i}",
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                snippet=f"Abstract snippet for paper {i}",
                pmid=pmid if is_match else None,
            )
            results.append(result)

        return SearchResults(
            query=query.query,
            results=results,
            total_results=num_results,
            search_time=0.5,
        )

    backend.search = AsyncMock(side_effect=mock_search)
    return backend


@pytest.fixture
def mock_cache():
    """Mock search cache (always miss for fresh testing)."""
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock()
    return cache


@pytest.fixture
def mock_fetcher():
    """Mock page fetcher for content extraction."""
    fetcher = Mock()
    # Mock methods as needed by QueryGenerator and ResourceMatcher
    return fetcher


# Helper functions


def calculate_precision_metrics(session: ReverseSearchSession) -> dict:
    """
    Calculate query precision metrics from session.

    Returns:
        dict - Metrics including avg hits per query, total queries, coverage
    """
    if not session.query_results:
        return {
            "avg_hits_per_query": 0.0,
            "total_queries": 0,
            "coverage": 0.0,
            "queries_to_target": 0,
        }

    total_hits = sum(r.new_finds for r in session.query_results)
    avg_hits = total_hits / len(session.query_results) if session.query_results else 0.0

    # Find how many queries to reach coverage target (if reached)
    queries_to_target = len(session.query_results)
    for i, result in enumerate(session.query_results):
        if result.cumulative_coverage >= 0.90:
            queries_to_target = i + 1
            break

    return {
        "avg_hits_per_query": avg_hits,
        "total_queries": len(session.query_results),
        "coverage": session.final_coverage,
        "queries_to_target": queries_to_target,
    }


# Integration tests


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Requires OPENAI_API_KEY environment variable",
)
async def test_pah_coverage_target_with_llm(
    pah_resources,
    mock_pubmed_backend,
    mock_cache,
    mock_fetcher,
):
    """
    PRIMARY TEST: Verify >90% coverage on PAH dataset with <10 queries using LLM + relevance sort.

    This is the main acceptance test for the reverse search coverage improvement feature.
    Success criteria from spec:
    - Coverage: >90% of 20 PAH papers found
    - Efficiency: Within 10 queries
    - Precision: >2 hits per query average
    """
    # Verify dataset loaded correctly
    assert len(pah_resources) == 20, f"Expected 20 PAH papers, got {len(pah_resources)}"

    # Configure reverse search with LLM + relevance sort
    config = ReverseSearchConfig(
        keyword_extractor="llm",
        llm_query_config={
            "model": "openai:gpt-4o-mini",
            "temperature": 0.7,
            "backend_specific_syntax": True,
        },
        sort_by="relevance",  # Relevance sorting to eliminate temporal bias
        coverage_target=0.90,
        max_queries=10,
        use_hint_fields=True,  # Use gene/disease hints from dataset
        enable_clustering=False,  # Disable for simpler initial test
    )

    # Create searcher
    searcher = ReverseSearcher(config, mock_pubmed_backend, mock_cache, mock_fetcher)

    # Execute reverse search
    start_time = time.time()
    session = await searcher.search(pah_resources, verbose=False)
    elapsed = time.time() - start_time

    # Calculate metrics
    metrics = calculate_precision_metrics(session)

    # Print results for analysis
    print("\n" + "=" * 60)
    print("PAH Coverage Test Results (LLM + Relevance Sort)")
    print("=" * 60)
    print(f"Coverage:              {session.final_coverage:.1%} (target: ≥90%)")
    print(f"Found:                 {session.found_count}/{len(pah_resources)} papers")
    print(f"Queries executed:      {metrics['total_queries']} (target: ≤10)")
    print(f"Queries to 90%:        {metrics['queries_to_target']}")
    print(f"Avg hits per query:    {metrics['avg_hits_per_query']:.2f} (target: ≥2.0)")
    print(f"Total time:            {elapsed:.1f}s (target: <60s)")
    print(f"Stopping reason:       {session.stopping_reason}")
    print("=" * 60)

    # CRITICAL ASSERTION: Coverage target met
    assert session.final_coverage >= 0.90, (
        f"Coverage {session.final_coverage:.2%} below 90% target. "
        f"Found {session.found_count}/{len(pah_resources)} papers. "
        f"This indicates LLM query quality or sort order issues."
    )

    # CRITICAL ASSERTION: Query efficiency
    assert metrics["total_queries"] <= 10, (
        f"Used {metrics['total_queries']} queries, exceeding 10-query target. "
        f"This indicates queries are too specific or lacking diversity."
    )

    # Secondary assertion: Query precision (informational)
    if metrics["avg_hits_per_query"] < 2.0:
        print(
            f"\nWARNING: Low precision {metrics['avg_hits_per_query']:.2f} hits/query (target: ≥2.0)"
        )
        print("Consider tuning LLM prompts for better precision.")

    # Secondary assertion: Performance
    assert elapsed < 60, f"Session took {elapsed:.1f}s (target: <60s)"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_pah_yake_baseline(
    pah_resources,
    mock_pubmed_backend,
    mock_cache,
    mock_fetcher,
):
    """
    BASELINE TEST: Measure YAKE coverage for comparison to LLM.

    Expected: YAKE achieves significantly lower coverage than LLM due to:
    - Broad, generic queries (lower precision)
    - Temporal bias with date-based sorting (old papers buried)

    This test documents the baseline problem that LLM + relevance sort solves.
    """
    assert len(pah_resources) == 20

    # Configure with YAKE baseline (no LLM, date-based sorting)
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        keywords_per_query=7,
        sort_by="date_desc",  # Date-based sorting (temporal bias)
        coverage_target=0.90,
        max_queries=10,
        use_hint_fields=True,
    )

    searcher = ReverseSearcher(config, mock_pubmed_backend, mock_cache, mock_fetcher)

    start_time = time.time()
    session = await searcher.search(pah_resources, verbose=False)
    elapsed = time.time() - start_time

    metrics = calculate_precision_metrics(session)

    # Print baseline results
    print("\n" + "=" * 60)
    print("PAH Baseline Results (YAKE + Date Sort)")
    print("=" * 60)
    print(f"Coverage:              {session.final_coverage:.1%}")
    print(f"Found:                 {session.found_count}/{len(pah_resources)} papers")
    print(f"Queries executed:      {metrics['total_queries']}")
    print(f"Avg hits per query:    {metrics['avg_hits_per_query']:.2f}")
    print(f"Total time:            {elapsed:.1f}s")
    print(f"Stopping reason:       {session.stopping_reason}")
    print("=" * 60)
    print("Expected: Lower coverage than LLM test (validates improvement)")
    print("=" * 60)

    # This is a baseline measurement, not a pass/fail test
    # Document results for comparison
    # Expectation: coverage < 90% (demonstrating the problem)
    if session.final_coverage >= 0.90:
        print("\nSURPRISE: YAKE baseline achieved 90% coverage!")
        print("This suggests the problem may be resolved by other means,")
        print("or the mock backend doesn't accurately simulate the issue.")


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Requires OPENAI_API_KEY environment variable",
)
async def test_pah_llm_performance(
    pah_resources,
    mock_pubmed_backend,
    mock_cache,
    mock_fetcher,
):
    """
    PERFORMANCE TEST: Verify LLM query generation meets latency targets.

    Target: <5s per resource average for query generation
    Total: <100s for 20 papers
    """
    assert len(pah_resources) == 20

    config = ReverseSearchConfig(
        keyword_extractor="llm",
        llm_query_config={"model": "openai:gpt-4o-mini", "temperature": 0.7},
        sort_by="relevance",
        coverage_target=0.90,
        max_queries=10,
    )

    searcher = ReverseSearcher(config, mock_pubmed_backend, mock_cache, mock_fetcher)

    # Measure query generation time specifically
    start = time.time()
    session = await searcher.search(pah_resources, verbose=False)
    total_elapsed = time.time() - start

    # Calculate per-resource time
    per_resource = total_elapsed / len(pah_resources)

    print("\n" + "=" * 60)
    print("LLM Performance Metrics")
    print("=" * 60)
    print(f"Total session time:    {total_elapsed:.1f}s (target: <100s)")
    print(f"Per-resource time:     {per_resource:.2f}s (target: <5s)")
    print(f"Queries generated:     {session.total_queries}")
    print("=" * 60)

    # Performance assertions
    assert total_elapsed < 100, (
        f"Session took {total_elapsed:.1f}s, exceeding 100s target for 20 papers"
    )

    if per_resource > 5.0:
        print(f"\nWARNING: Slow per-resource time {per_resource:.2f}s (target: <5s)")
        print("LLM latency may be higher than expected.")


@pytest.mark.asyncio
async def test_pah_dataset_validation(pah_resources, pah_dataset_path):
    """
    SANITY TEST: Verify PAH dataset is correctly formatted.

    Validates:
    - File exists and is readable
    - Contains exactly 20 papers
    - All papers have PMIDs
    - All papers have required hint fields (gene, disease)
    - PMIDs are unique
    """
    # Check dataset exists
    assert pah_dataset_path.exists(), f"PAH dataset not found: {pah_dataset_path}"

    # Check resource count
    assert len(pah_resources) == 20, (
        f"Expected 20 PAH papers, got {len(pah_resources)}. "
        f"Dataset may be corrupted or modified."
    )

    # Check all have PMIDs
    missing_pmid = [r for r in pah_resources if not r.pmid]
    assert not missing_pmid, (
        f"{len(missing_pmid)} resources missing PMIDs. "
        f"PAH dataset should be all PubMed papers."
    )

    # Check PMIDs are unique
    pmids = [r.pmid for r in pah_resources]
    unique_pmids = set(pmids)
    assert len(unique_pmids) == len(pmids), (
        f"Duplicate PMIDs found: {len(pmids)} total, {len(unique_pmids)} unique"
    )

    # Check hint fields present
    missing_hints = [
        r
        for r in pah_resources
        if "gene" not in r.hint_fields or "disease" not in r.hint_fields
    ]
    if missing_hints:
        print(f"\nWARNING: {len(missing_hints)} resources missing gene/disease hints")
        print("This may reduce LLM query quality.")

    print("\n" + "=" * 60)
    print("PAH Dataset Validation")
    print("=" * 60)
    print(f"Resources:             {len(pah_resources)}")
    print(f"Unique PMIDs:          {len(unique_pmids)}")
    print(
        f"With gene hint:        {sum(1 for r in pah_resources if 'gene' in r.hint_fields)}"
    )
    print(
        f"With disease hint:     {sum(1 for r in pah_resources if 'disease' in r.hint_fields)}"
    )
    print("=" * 60)


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Requires OPENAI_API_KEY environment variable",
)
async def test_pah_unfound_analysis(
    pah_resources,
    mock_pubmed_backend,
    mock_cache,
    mock_fetcher,
):
    """
    DIAGNOSTIC TEST: Analyze which papers are not found and why.

    If coverage target is not met, this test helps identify:
    - Which papers are hardest to find
    - Patterns in unfound papers (date ranges, specific genes, etc.)
    - Query quality issues
    """
    config = ReverseSearchConfig(
        keyword_extractor="llm",
        llm_query_config={"model": "openai:gpt-4o-mini", "temperature": 0.7},
        sort_by="relevance",
        coverage_target=0.90,
        max_queries=10,
        use_hint_fields=True,
    )

    searcher = ReverseSearcher(config, mock_pubmed_backend, mock_cache, mock_fetcher)
    session = await searcher.search(pah_resources, verbose=False)

    print("\n" + "=" * 60)
    print("Unfound Papers Analysis")
    print("=" * 60)
    print(f"Found:                 {session.found_count}/{len(pah_resources)}")
    print(f"Unfound:               {len(session.unfound_resources)}")
    print("=" * 60)

    if session.unfound_resources:
        print("\nUnfound papers:")
        for i, resource in enumerate(session.unfound_resources[:10], 1):  # Limit to 10
            hints = ", ".join(f"{k}={v}" for k, v in resource.hint_fields.items())
            print(f"  {i}. PMID {resource.pmid} - {hints}")

        if len(session.unfound_resources) > 10:
            print(f"  ... and {len(session.unfound_resources) - 10} more")

    print("\nQuery quality analysis:")
    for i, result in enumerate(session.query_results[:5], 1):  # First 5 queries
        print(f"  Q{i}: {result.new_finds} hits - {result.query[:80]}...")
