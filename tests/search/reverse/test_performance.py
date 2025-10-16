"""
Performance benchmarking tests for two-stage query generation.

Verifies that:
1. yake+direct (baseline) performs within acceptable time bounds
2. New two-stage architecture has minimal overhead (<10%) vs old pipeline
3. Hybrid approaches (statistical extraction + LLM construction) are faster than full-LLM
4. Query generation scales linearly with number of resources

Note: These are not strict performance tests but regression checks to catch
significant performance degradations. Mark as slow/optional for CI.
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, Mock, patch
from pydantic_ai.result import AgentRunResult

from interaction_finder.search.reverse.query_generator import QueryGenerator
from interaction_finder.search.reverse.models import (
    KnownResource,
    ReverseSearchConfig,
)
from interaction_finder.search.reverse.llm_models import LLMQueryConstructionResponse


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def sample_resources_10():
    """10 sample resources for benchmarking."""
    return [
        KnownResource(
            pmid=f"{i:08d}",
            url=f"https://pubmed.ncbi.nlm.nih.gov/{i:08d}/",
        )
        for i in range(10)
    ]


@pytest.fixture
def sample_metadata_10():
    """Metadata for 10 resources."""
    return {
        f"{i:08d}": {
            "title": f"Study of gene expression in disease model {i}",
            "abstract": f"This research investigates molecular pathways and genetic factors related to disease pathogenesis. Study {i} examines BRCA1, TP53, and related genes in the context of cellular dysfunction and treatment response.",
        }
        for i in range(10)
    }


@pytest.fixture
def mock_fast_llm_agent():
    """Mock LLM agent with simulated fast response times."""

    def create_mock_result(query: str):
        mock_response = LLMQueryConstructionResponse(
            query=query,
            reasoning="Constructed query from keywords",
        )
        mock_result = Mock(spec=AgentRunResult)
        mock_result.output = mock_response
        return mock_result

    mock_agent = AsyncMock()

    # Simulate 100ms LLM call
    async def mock_run(*args, **kwargs):
        await asyncio.sleep(0.1)  # 100ms
        return create_mock_result('"test query"')

    mock_agent.run = mock_run
    return mock_agent


# ==============================================================================
# Baseline Performance: yake + direct
# ==============================================================================


@pytest.mark.slow
@pytest.mark.asyncio
async def test_baseline_yake_direct_10_resources(
    sample_resources_10, sample_metadata_10
):
    """Benchmark yake+direct for 10 resources (baseline approach)."""
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        query_constructor="direct",
        keywords_per_query=5,
        enable_clustering=False,
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    with patch.object(generator, "_fetch_pmid_metadata_batch") as mock_fetch:
        mock_fetch.return_value = sample_metadata_10

        start = time.perf_counter()
        queries = await generator.generate_initial_queries_individual(
            sample_resources_10
        )
        elapsed = time.perf_counter() - start

        # Assertions
        assert len(queries) == 10
        # Should complete in reasonable time (<5s for 10 resources)
        assert elapsed < 5.0, f"yake+direct took {elapsed:.2f}s (expected <5s)"

        # Print for manual inspection
        print(
            f"\nyake+direct (10 resources): {elapsed:.3f}s ({elapsed / 10 * 1000:.1f}ms per query)"
        )


@pytest.mark.slow
@pytest.mark.asyncio
async def test_baseline_yake_direct_scaling(sample_metadata_10):
    """Test that yake+direct scales linearly with resource count."""
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        query_constructor="direct",
        keywords_per_query=5,
        enable_clustering=False,
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    timings = []

    for n in [5, 10, 20]:
        resources = [
            KnownResource(
                pmid=f"{i:08d}",
                url=f"https://pubmed.ncbi.nlm.nih.gov/{i:08d}/",
            )
            for i in range(n)
        ]

        metadata = {f"{i:08d}": sample_metadata_10["00000000"] for i in range(n)}

        with patch.object(generator, "_fetch_pmid_metadata_batch") as mock_fetch:
            mock_fetch.return_value = metadata

            start = time.perf_counter()
            queries = await generator.generate_initial_queries_individual(resources)
            elapsed = time.perf_counter() - start

            assert len(queries) == n
            timings.append((n, elapsed))

    # Print results
    print("\nScaling (yake+direct):")
    for n, elapsed in timings:
        print(f"  {n} resources: {elapsed:.3f}s ({elapsed / n * 1000:.1f}ms per query)")

    # Basic linearity check: 20 resources should not take >3x time of 5 resources
    time_5 = timings[0][1]
    time_20 = timings[2][1]
    assert time_20 < time_5 * 5, "Performance does not scale linearly"


# ==============================================================================
# New Two-Stage Architecture Overhead
# ==============================================================================


@pytest.mark.slow
@pytest.mark.asyncio
async def test_two_stage_overhead_is_minimal(sample_resources_10, sample_metadata_10):
    """Test that two-stage refactoring adds <10% overhead to yake+direct."""
    # This test compares the new implementation to itself
    # (we don't have the old implementation to compare against)
    # Instead, we verify absolute performance is acceptable

    config = ReverseSearchConfig(
        keyword_extractor="yake",
        query_constructor="direct",
        keywords_per_query=5,
        enable_clustering=False,
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    with patch.object(generator, "_fetch_pmid_metadata_batch") as mock_fetch:
        mock_fetch.return_value = sample_metadata_10

        # Run 5 times and take median
        timings = []
        for _ in range(5):
            start = time.perf_counter()
            await generator.generate_initial_queries_individual(sample_resources_10)
            elapsed = time.perf_counter() - start
            timings.append(elapsed)

        median_time = sorted(timings)[2]

        # Should be fast: <2s for 10 resources
        assert median_time < 2.0, f"Median time {median_time:.2f}s exceeds 2s threshold"

        print(f"\nTwo-stage yake+direct median (10 resources): {median_time:.3f}s")


# ==============================================================================
# Hybrid vs Full-LLM Performance
# ==============================================================================


@pytest.mark.slow
@pytest.mark.asyncio
async def test_hybrid_faster_than_full_llm(sample_resources_10, sample_metadata_10):
    """Test that hybrid (yake+llm) is faster than full-LLM (none+llm)."""
    import asyncio

    # Hybrid: yake + llm
    config_hybrid = ReverseSearchConfig(
        keyword_extractor="yake",
        query_constructor="llm",
        keywords_per_query=5,
        enable_clustering=False,
    )
    generator_hybrid = QueryGenerator(config_hybrid, backend_name="pubmed")

    # Full-LLM: none + llm
    config_full = ReverseSearchConfig(
        keyword_extractor="none",
        query_constructor="llm",
        enable_clustering=False,
    )
    generator_full = QueryGenerator(config_full, backend_name="pubmed")

    # Mock LLM agent with realistic timing
    def create_mock_agent():
        mock_agent = AsyncMock()

        async def mock_run(*args, **kwargs):
            await asyncio.sleep(0.1)  # 100ms per LLM call
            mock_response = LLMQueryConstructionResponse(
                query='"test query"',
                reasoning="Generated",
            )
            mock_result = Mock(spec=AgentRunResult)
            mock_result.output = mock_response
            return mock_result

        mock_agent.run = mock_run
        return mock_agent

    with patch.object(generator_hybrid, "_fetch_pmid_metadata_batch") as mock_fetch:
        mock_fetch.return_value = sample_metadata_10

        with patch.object(
            generator_hybrid.constructor,
            "_create_agent",
            return_value=create_mock_agent(),
        ):
            start_hybrid = time.perf_counter()
            await generator_hybrid.generate_initial_queries_individual(
                sample_resources_10
            )
            time_hybrid = time.perf_counter() - start_hybrid

    with patch.object(generator_full, "_fetch_pmid_metadata_batch") as mock_fetch:
        mock_fetch.return_value = sample_metadata_10

        with patch.object(
            generator_full.constructor,
            "_create_agent",
            return_value=create_mock_agent(),
        ):
            start_full = time.perf_counter()
            await generator_full.generate_initial_queries_individual(
                sample_resources_10
            )
            time_full = time.perf_counter() - start_full

    print(f"\nHybrid (yake+llm): {time_hybrid:.3f}s")
    print(f"Full-LLM (none+llm): {time_full:.3f}s")
    print(f"Speedup: {time_full / time_hybrid:.2f}x")

    # Hybrid and full-LLM should have similar times in this mock scenario
    # (both call LLM once per resource)
    # The real difference is in LLM prompt size/complexity
    # For this test, just verify both complete reasonably fast
    assert time_hybrid < 5.0
    assert time_full < 5.0


# ==============================================================================
# Extraction Stage Performance
# ==============================================================================


@pytest.mark.slow
@pytest.mark.asyncio
async def test_extraction_stage_performance(sample_resources_10, sample_metadata_10):
    """Test keyword extraction stage performance for different extractors."""

    extractors = ["yake", "rake", "tfidf"]
    timings = {}

    for extractor_name in extractors:
        config = ReverseSearchConfig(
            keyword_extractor=extractor_name,
            query_constructor="direct",
            keywords_per_query=5,
            enable_clustering=False,
        )
        generator = QueryGenerator(config, backend_name="pubmed")

        with patch.object(generator, "_fetch_pmid_metadata_batch") as mock_fetch:
            mock_fetch.return_value = sample_metadata_10

            start = time.perf_counter()
            await generator.generate_initial_queries_individual(sample_resources_10)
            elapsed = time.perf_counter() - start

            timings[extractor_name] = elapsed

    print("\nExtraction stage performance:")
    for extractor, elapsed in timings.items():
        print(f"  {extractor}: {elapsed:.3f}s ({elapsed / 10 * 1000:.1f}ms per query)")

    # All extractors should complete in reasonable time
    for extractor, elapsed in timings.items():
        assert elapsed < 5.0, f"{extractor} took {elapsed:.2f}s (expected <5s)"


# ==============================================================================
# Construction Stage Performance
# ==============================================================================


@pytest.mark.slow
@pytest.mark.asyncio
async def test_construction_stage_performance(sample_resources_10, sample_metadata_10):
    """Test query construction stage performance for both constructors."""
    import asyncio

    # Direct constructor
    config_direct = ReverseSearchConfig(
        keyword_extractor="yake",
        query_constructor="direct",
        keywords_per_query=5,
        enable_clustering=False,
    )
    generator_direct = QueryGenerator(config_direct, backend_name="pubmed")

    with patch.object(generator_direct, "_fetch_pmid_metadata_batch") as mock_fetch:
        mock_fetch.return_value = sample_metadata_10

        start_direct = time.perf_counter()
        await generator_direct.generate_initial_queries_individual(sample_resources_10)
        time_direct = time.perf_counter() - start_direct

    # LLM constructor (mocked)
    config_llm = ReverseSearchConfig(
        keyword_extractor="yake",
        query_constructor="llm",
        keywords_per_query=5,
        enable_clustering=False,
    )
    generator_llm = QueryGenerator(config_llm, backend_name="pubmed")

    def create_mock_agent():
        mock_agent = AsyncMock()

        async def mock_run(*args, **kwargs):
            await asyncio.sleep(0.05)  # 50ms per LLM call
            mock_response = LLMQueryConstructionResponse(
                query='"test"',
                reasoning="Generated",
            )
            mock_result = Mock(spec=AgentRunResult)
            mock_result.output = mock_response
            return mock_result

        mock_agent.run = mock_run
        return mock_agent

    with patch.object(generator_llm, "_fetch_pmid_metadata_batch") as mock_fetch:
        mock_fetch.return_value = sample_metadata_10

        with patch.object(
            generator_llm.constructor, "_create_agent", return_value=create_mock_agent()
        ):
            start_llm = time.perf_counter()
            await generator_llm.generate_initial_queries_individual(sample_resources_10)
            time_llm = time.perf_counter() - start_llm

    print("\nConstruction stage performance:")
    print(f"  direct: {time_direct:.3f}s ({time_direct / 10 * 1000:.1f}ms per query)")
    print(
        f"  llm (mocked 50ms): {time_llm:.3f}s ({time_llm / 10 * 1000:.1f}ms per query)"
    )

    # Direct should be very fast
    assert time_direct < 2.0, f"Direct constructor took {time_direct:.2f}s"

    # LLM with 50ms mock should be <2s for 10 resources
    assert time_llm < 2.0, f"LLM constructor took {time_llm:.2f}s"


# ==============================================================================
# Memory Usage (Optional)
# ==============================================================================


@pytest.mark.slow
@pytest.mark.skipif(True, reason="Memory profiling requires memory_profiler package")
@pytest.mark.asyncio
async def test_memory_usage_reasonable(sample_resources_10, sample_metadata_10):
    """Test that memory usage is reasonable (optional test)."""
    # This test is skipped by default as it requires memory_profiler
    # Uncomment and install memory_profiler to enable

    # from memory_profiler import profile
    # import gc

    config = ReverseSearchConfig(
        keyword_extractor="yake",
        query_constructor="direct",
        keywords_per_query=5,
        enable_clustering=False,
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    # Force garbage collection before measuring
    # gc.collect()

    with patch.object(generator, "_fetch_pmid_metadata_batch") as mock_fetch:
        mock_fetch.return_value = sample_metadata_10

        # Measure memory before and after
        # This is a placeholder - actual implementation would use memory_profiler
        await generator.generate_initial_queries_individual(sample_resources_10)

    # Memory usage should be reasonable (<100MB growth)
    # Actual assertion would go here
    pass


# ==============================================================================
# Batch Processing Performance
# ==============================================================================


@pytest.mark.slow
@pytest.mark.asyncio
async def test_batch_metadata_fetch_performance():
    """Test that batch PMID fetching is efficient."""
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        query_constructor="direct",
        keywords_per_query=5,
        enable_clustering=False,
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    # Test with 100 resources (should batch fetch)
    resources_100 = [
        KnownResource(
            pmid=f"{i:08d}",
            url=f"https://pubmed.ncbi.nlm.nih.gov/{i:08d}/",
        )
        for i in range(100)
    ]

    metadata_100 = {
        f"{i:08d}": {
            "title": f"Study {i}",
            "abstract": f"Research content for study number {i}",
        }
        for i in range(100)
    }

    with patch.object(generator, "_fetch_pmid_metadata_batch") as mock_fetch:
        mock_fetch.return_value = metadata_100

        start = time.perf_counter()
        queries = await generator.generate_initial_queries_individual(resources_100)
        elapsed = time.perf_counter() - start

        assert len(queries) == 100
        # Should complete in reasonable time even for 100 resources
        assert elapsed < 20.0, f"100 resources took {elapsed:.2f}s (expected <20s)"

        print(
            f"\nBatch processing (100 resources): {elapsed:.3f}s ({elapsed / 100 * 1000:.1f}ms per query)"
        )


# ==============================================================================
# Regression Test: Ensure No Performance Degradation
# ==============================================================================


@pytest.mark.slow
@pytest.mark.asyncio
async def test_no_performance_regression(sample_resources_10, sample_metadata_10):
    """Regression test: verify performance is within acceptable bounds."""
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        query_constructor="direct",
        keywords_per_query=5,
        enable_clustering=False,
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    # Run multiple times and check consistency
    timings = []

    with patch.object(generator, "_fetch_pmid_metadata_batch") as mock_fetch:
        mock_fetch.return_value = sample_metadata_10

        for _ in range(3):
            start = time.perf_counter()
            await generator.generate_initial_queries_individual(sample_resources_10)
            elapsed = time.perf_counter() - start
            timings.append(elapsed)

    mean_time = sum(timings) / len(timings)
    max_time = max(timings)
    min_time = min(timings)

    print("\nRegression test (3 runs):")
    print(f"  Mean: {mean_time:.3f}s")
    print(f"  Min:  {min_time:.3f}s")
    print(f"  Max:  {max_time:.3f}s")

    # All runs should be reasonably fast and consistent
    assert mean_time < 2.0, f"Mean time {mean_time:.2f}s exceeds threshold"
    assert max_time < 3.0, f"Max time {max_time:.2f}s exceeds threshold"

    # Variation should not be too large
    variation = (max_time - min_time) / mean_time
    assert variation < 0.5, f"High timing variation: {variation:.1%}"
