"""
Comprehensive tests for QueryGenerator.

Tests cover:
- Individual query generation from PMIDs (mock ESummary)
- Individual query generation from URLs (mock PageFetcher)
- Clustered query generation (verify fewer queries than resources)
- Batch PMID fetching (up to 200)
- Metadata fetch failure → content fallback
- Hint fields included when use_hint_fields=True
- Query construction with OR logic
- Refinement queries exclude duplicates
- Empty resources list (return empty queries)
- Single resource (no clustering, individual query)
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from interaction_finder.search.reverse.query_generator import (
    QueryGenerator,
    ResourceContent,
)
from interaction_finder.search.reverse.models import KnownResource, ReverseSearchConfig


# ==============================================================================
# Test Individual Query Generation from PMIDs
# ==============================================================================


@pytest.mark.asyncio
async def test_generate_queries_from_pmids():
    """Test query generation from PMIDs using metadata-first strategy."""
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        keywords_per_query=5,
        use_hint_fields=False,
        enable_clustering=False,  # Force individual query generation
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    # Mock PMID metadata fetching
    with patch.object(generator, "_fetch_pmid_metadata_batch") as mock_fetch:
        mock_fetch.return_value = {
            "123": {
                "title": "Diabetes mellitus and insulin resistance",
                "abstract": "Type 2 diabetes is characterized by insulin resistance and beta cell dysfunction...",
            },
            "456": {
                "title": "BRCA1 mutations in breast cancer",
                "abstract": "BRCA1 gene mutations increase breast cancer risk through DNA repair defects...",
            },
        }

        resources = [
            KnownResource(pmid="123", url="https://pubmed.ncbi.nlm.nih.gov/123/"),
            KnownResource(pmid="456", url="https://pubmed.ncbi.nlm.nih.gov/456/"),
        ]

        queries = await generator.generate_initial_queries_individual(resources)

        assert len(queries) == 2
        # Verify queries contain relevant keywords
        assert any("diabetes" in q.lower() or "insulin" in q.lower() for q in queries)
        assert any("brca1" in q.lower() or "breast" in q.lower() for q in queries)
        # Verify OR logic is used
        assert all(" OR " in q for q in queries)


@pytest.mark.asyncio
async def test_generate_queries_from_urls():
    """Test query generation from URLs using content fallback strategy."""
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        keywords_per_query=5,
        use_hint_fields=False,
        enable_clustering=False,
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    # Mock PageFetcher
    mock_fetcher = AsyncMock()
    mock_doc1 = MagicMock()
    mock_doc1.content_markdown = (
        "Machine learning in genomics. Deep learning models predict gene expression."
    )
    mock_doc2 = MagicMock()
    mock_doc2.content_markdown = (
        "Climate change impacts biodiversity. Species extinction rates accelerate."
    )

    mock_fetcher.fetch_documents = AsyncMock(
        return_value={
            "https://example.com/paper1": mock_doc1,
            "https://example.com/paper2": mock_doc2,
        }
    )

    generator.fetcher = mock_fetcher

    resources = [
        KnownResource(url="https://example.com/paper1"),
        KnownResource(url="https://example.com/paper2"),
    ]

    queries = await generator.generate_initial_queries_individual(resources)

    assert len(queries) == 2
    # Verify PageFetcher was called
    mock_fetcher.fetch_documents.assert_called_once()
    # Verify queries contain relevant keywords
    assert any(
        "machine learning" in q.lower() or "genomics" in q.lower() for q in queries
    )
    assert any("climate" in q.lower() or "biodiversity" in q.lower() for q in queries)


# ==============================================================================
# Test Clustering
# ==============================================================================


@pytest.mark.asyncio
async def test_clustering_reduces_query_count():
    """Test that clustering generates fewer queries than resources."""
    config = ReverseSearchConfig(
        enable_clustering=True,
        min_cluster_size=3,
        target_clusters=2,
        keyword_extractor="yake",
        keywords_per_query=5,
        use_hint_fields=False,
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    # Create 6 resources with similar content (should cluster into 2 groups)
    resources = [
        KnownResource(pmid=str(i), url=f"https://example.com/{i}") for i in range(6)
    ]

    # Mock content fetching to return similar text
    with patch.object(generator, "_fetch_resource_contents") as mock_fetch:
        contents = [
            ResourceContent(
                resource=r,
                text="Diabetes mellitus insulin resistance glucose metabolism homeostasis",
                source="metadata",
            )
            for r in resources
        ]
        contents_dict = {
            r.url: {
                "source": "metadata",
                "title": "Test",
                "content_length": 100,
                "success": True,
                "error": None,
            }
            for r in resources
        }
        mock_fetch.return_value = (contents, contents_dict)

        queries = await generator.generate_initial_queries_clustered(resources)

        # Should generate 2 queries (one per cluster) instead of 6
        assert len(queries) <= 3  # At most target_clusters or slightly more
        assert len(queries) < len(resources)


@pytest.mark.asyncio
async def test_clustering_fallback_for_small_batch():
    """Test that clustering falls back to individual for small batches."""
    config = ReverseSearchConfig(
        enable_clustering=True,
        min_cluster_size=3,
        target_clusters=2,
        keyword_extractor="yake",
        keywords_per_query=5,
        use_hint_fields=False,
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    # Only 2 resources - below min_cluster_size
    resources = [
        KnownResource(pmid="1", url="https://example.com/1"),
        KnownResource(pmid="2", url="https://example.com/2"),
    ]

    # Mock to verify individual method is called
    with patch.object(
        generator, "generate_initial_queries_individual"
    ) as mock_individual:
        mock_individual.return_value = ["query1", "query2"]

        queries = await generator.generate_initial_queries(resources)

        # Should use individual generation for small batch
        mock_individual.assert_called_once()
        assert len(queries) == 2


# ==============================================================================
# Test Batch PMID Fetching
# ==============================================================================


@pytest.mark.asyncio
async def test_batch_pmid_fetching():
    """Test batch PMID fetching with rate limiting."""
    config = ReverseSearchConfig(
        batch_pmid_fetch_size=200,
        keyword_extractor="yake",
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    # Mock HTTP client
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "result": {
            "123": {"title": "Title 1", "abstracttext": "Abstract 1"},
            "456": {"title": "Title 2", "abstracttext": "Abstract 2"},
        }
    }
    mock_client.get = AsyncMock(return_value=mock_response)
    generator.http_client = mock_client

    pmids = ["123", "456"]
    metadata_map = await generator._fetch_pmid_metadata_batch(pmids)

    assert len(metadata_map) == 2
    assert metadata_map["123"]["title"] == "Title 1"
    assert metadata_map["123"]["abstract"] == "Abstract 1"
    assert metadata_map["456"]["title"] == "Title 2"
    assert metadata_map["456"]["abstract"] == "Abstract 2"

    # Verify correct API call
    mock_client.get.assert_called_once()
    call_args = mock_client.get.call_args
    assert (
        call_args[0][0] == "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    )
    assert call_args[1]["params"]["db"] == "pubmed"
    assert call_args[1]["params"]["id"] == "123,456"


@pytest.mark.asyncio
async def test_batch_pmid_fetching_handles_errors():
    """Test that batch PMID fetching handles errors gracefully."""
    config = ReverseSearchConfig(
        batch_pmid_fetch_size=200,
        keyword_extractor="yake",
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    # Mock HTTP client to raise error
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=Exception("Network error"))
    generator.http_client = mock_client

    pmids = ["123", "456"]
    metadata_map = await generator._fetch_pmid_metadata_batch(pmids)

    # Should return empty dict on error
    assert len(metadata_map) == 0


# ==============================================================================
# Test Fallback Behavior
# ==============================================================================


@pytest.mark.asyncio
async def test_metadata_fetch_failure_falls_back_to_content():
    """Test that failed metadata fetch falls back to content fetching."""
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        keywords_per_query=5,
        use_hint_fields=False,
        enable_clustering=False,
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    # Mock PMID fetch to return empty (simulating failure)
    with patch.object(generator, "_fetch_pmid_metadata_batch") as mock_pmid:
        mock_pmid.return_value = {}  # No metadata

        # Mock PageFetcher for fallback
        mock_fetcher = AsyncMock()
        mock_doc = MagicMock()
        mock_doc.content_markdown = "Fallback content for PMID resource"
        mock_fetcher.fetch_documents = AsyncMock(
            return_value={"https://pubmed.ncbi.nlm.nih.gov/123/": mock_doc}
        )
        generator.fetcher = mock_fetcher

        resources = [
            KnownResource(pmid="123", url="https://pubmed.ncbi.nlm.nih.gov/123/")
        ]

        contents, contents_dict = await generator._fetch_resource_contents(resources)

        # Should have fallen back to content
        assert len(contents) == 1
        assert contents[0].source == "content"
        assert "Fallback content" in contents[0].text


@pytest.mark.asyncio
async def test_content_fetch_failure_uses_hint_fields():
    """Test that failed content fetch uses hint fields as last resort."""
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        keywords_per_query=5,
        use_hint_fields=True,
        enable_clustering=False,
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    # Mock PageFetcher to return no document
    mock_fetcher = AsyncMock()
    mock_fetcher.fetch_documents = AsyncMock(return_value={})
    generator.fetcher = mock_fetcher

    resources = [
        KnownResource(
            url="https://example.com/paper",
            hint_fields={"celltype": "CD8+ T cell", "marker": "CD8A"},
        )
    ]

    contents, contents_dict = await generator._fetch_resource_contents(resources)

    # Should have used hint fields
    assert len(contents) == 1
    assert contents[0].source == "hint_fields"
    assert "CD8+ T cell" in contents[0].text
    assert "CD8A" in contents[0].text


# ==============================================================================
# Test Hint Fields
# ==============================================================================


@pytest.mark.asyncio
async def test_hint_fields_included_when_enabled():
    """Test that hint fields are included in queries when enabled."""
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        keywords_per_query=3,
        use_hint_fields=True,
        enable_clustering=False,
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    # Mock content fetching
    with patch.object(generator, "_fetch_resource_contents") as mock_fetch:
        resources = [
            KnownResource(
                pmid="123",
                url="https://pubmed.ncbi.nlm.nih.gov/123/",
                hint_fields={"celltype": "CD8+ T cell", "marker": "CD8A"},
            )
        ]
        contents = [
            ResourceContent(
                resource=resources[0],
                text="T cell biology and immunology research",
                source="metadata",
            )
        ]
        contents_dict = {}
        mock_fetch.return_value = (contents, contents_dict)

        queries = await generator.generate_initial_queries_individual(resources)

        assert len(queries) == 1
        # Hint fields should be in the query
        assert "CD8+ T cell" in queries[0] or "CD8A" in queries[0]


@pytest.mark.asyncio
async def test_hint_fields_excluded_when_disabled():
    """Test that hint fields are excluded when disabled."""
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        keywords_per_query=3,
        use_hint_fields=False,
        enable_clustering=False,
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    # Mock content fetching
    with patch.object(generator, "_fetch_resource_contents") as mock_fetch:
        resources = [
            KnownResource(
                pmid="123",
                url="https://pubmed.ncbi.nlm.nih.gov/123/",
                hint_fields={"celltype": "CD8+ T cell"},
            )
        ]
        contents = [
            ResourceContent(
                resource=resources[0],
                text="T cell biology and immunology research",
                source="metadata",
            )
        ]
        contents_dict = {}
        mock_fetch.return_value = (contents, contents_dict)

        queries = await generator.generate_initial_queries_individual(resources)

        assert len(queries) == 1
        # Hint fields should NOT be in the query
        assert "CD8+ T cell" not in queries[0]


# ==============================================================================
# Test Query Construction
# ==============================================================================


@pytest.mark.asyncio
async def test_construct_query_with_or_logic():
    """Test that DirectQueryConstructor uses OR logic."""
    from interaction_finder.search.reverse.query_constructors import (
        DirectQueryConstructor,
    )
    from interaction_finder.search.reverse.models import QueryConstructionContext

    constructor = DirectQueryConstructor()

    context = QueryConstructionContext(
        keywords=["diabetes", "insulin resistance", "glucose"],
        keyword_scores=None,
        hint_terms=[],
        backend="pubmed",
        resource_content=None,
        extractor_used="yake",
    )
    query = await constructor.construct(context)

    assert query == '"diabetes" OR "insulin resistance" OR "glucose"'


@pytest.mark.asyncio
async def test_construct_query_empty_keywords():
    """Test that DirectQueryConstructor handles empty keywords."""
    from interaction_finder.search.reverse.query_constructors import (
        DirectQueryConstructor,
    )
    from interaction_finder.search.reverse.models import QueryConstructionContext

    constructor = DirectQueryConstructor()

    context = QueryConstructionContext(
        keywords=[],
        keyword_scores=None,
        hint_terms=[],
        backend="pubmed",
        resource_content=None,
        extractor_used="yake",
    )
    query = await constructor.construct(context)
    assert query == ""


def test_extract_hint_terms():
    """Test extraction of unique hint terms from resources."""
    config = ReverseSearchConfig()
    generator = QueryGenerator(config, backend_name="pubmed")

    resources = [
        KnownResource(
            url="https://example.com/1",
            hint_fields={"celltype": "CD8+ T cell", "marker": "CD8A"},
        ),
        KnownResource(
            url="https://example.com/2",
            hint_fields={"celltype": "CD8+ T cell", "marker": "CD4"},
        ),
    ]

    hint_terms = generator._extract_hint_terms(resources)

    # Should have 3 unique terms (CD8+ T cell appears twice, so deduplicated)
    assert len(hint_terms) == 3
    assert "CD8+ T cell" in hint_terms
    assert "CD8A" in hint_terms
    assert "CD4" in hint_terms


# ==============================================================================
# Test Refinement Queries
# ==============================================================================


@pytest.mark.asyncio
async def test_refinement_queries_exclude_duplicates():
    """Test that refinement queries exclude duplicates of previous queries."""
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        keywords_per_query=3,
        use_hint_fields=False,
        enable_clustering=False,
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    # Mock content fetching to return predictable keywords
    with patch.object(generator, "_fetch_resource_contents") as mock_fetch:
        resources = [
            KnownResource(pmid="123", url="https://pubmed.ncbi.nlm.nih.gov/123/"),
            KnownResource(pmid="456", url="https://pubmed.ncbi.nlm.nih.gov/456/"),
        ]
        contents = [
            ResourceContent(
                resource=resources[0],
                text="Diabetes mellitus and insulin resistance",
                source="metadata",
            ),
            ResourceContent(
                resource=resources[1],
                text="Type 2 diabetes and glucose metabolism",
                source="metadata",
            ),
        ]
        contents_dict = {}
        mock_fetch.return_value = (contents, contents_dict)

        # Generate initial queries
        initial_queries = await generator.generate_initial_queries(resources)

        # Now generate refinement queries with same resources
        refinement_queries = await generator.generate_refinement_queries(
            resources, initial_queries
        )

        # Should have filtered out duplicates
        assert len(refinement_queries) == 0  # All queries were duplicates


@pytest.mark.asyncio
async def test_refinement_queries_with_new_resources():
    """Test that refinement queries work with new unfound resources."""
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        keywords_per_query=3,
        use_hint_fields=False,
        enable_clustering=False,
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    # Mock content fetching
    with patch.object(generator, "_fetch_resource_contents") as mock_fetch:
        old_resources = [
            KnownResource(pmid="123", url="https://pubmed.ncbi.nlm.nih.gov/123/")
        ]
        new_resources = [
            KnownResource(pmid="456", url="https://pubmed.ncbi.nlm.nih.gov/456/")
        ]

        # First call for old resources
        contents = [
            ResourceContent(
                resource=old_resources[0],
                text="Diabetes mellitus and insulin resistance",
                source="metadata",
            )
        ]
        contents_dict = {}
        mock_fetch.return_value = (contents, contents_dict)
        initial_queries = await generator.generate_initial_queries(old_resources)

        # Second call for new resources
        contents = [
            ResourceContent(
                resource=new_resources[0],
                text="Cancer biology and tumor suppressor genes",
                source="metadata",
            )
        ]
        contents_dict = {}
        mock_fetch.return_value = (contents, contents_dict)
        refinement_queries = await generator.generate_refinement_queries(
            new_resources, initial_queries
        )

        # Should have generated new queries (different content)
        assert len(refinement_queries) > 0


# ==============================================================================
# Test Edge Cases
# ==============================================================================


@pytest.mark.asyncio
async def test_empty_resources_list():
    """Test that empty resources list returns empty queries."""
    config = ReverseSearchConfig()
    generator = QueryGenerator(config, backend_name="pubmed")

    queries = await generator.generate_initial_queries([])
    assert queries == []


@pytest.mark.asyncio
async def test_single_resource_no_clustering():
    """Test that single resource uses individual generation."""
    config = ReverseSearchConfig(
        enable_clustering=True,
        min_cluster_size=2,
        keyword_extractor="yake",
        keywords_per_query=3,
        use_hint_fields=False,
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    # Mock content fetching
    with patch.object(generator, "_fetch_resource_contents") as mock_fetch:
        resources = [
            KnownResource(pmid="123", url="https://pubmed.ncbi.nlm.nih.gov/123/")
        ]
        contents = [
            ResourceContent(
                resource=resources[0],
                text="Machine learning in genomics",
                source="metadata",
            )
        ]
        contents_dict = {}
        mock_fetch.return_value = (contents, contents_dict)

        queries = await generator.generate_initial_queries(resources)

        # Should generate individual query (not enough for clustering)
        assert len(queries) == 1


@pytest.mark.asyncio
async def test_resources_with_no_extractable_keywords():
    """Test handling of resources with no extractable keywords."""
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        keywords_per_query=5,
        use_hint_fields=False,
        enable_clustering=False,
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    # Mock content fetching with minimal text
    with patch.object(generator, "_fetch_resource_contents") as mock_fetch:
        resources = [
            KnownResource(pmid="123", url="https://pubmed.ncbi.nlm.nih.gov/123/")
        ]
        contents = [
            ResourceContent(
                resource=resources[0],
                text="a b c",  # Very short, unlikely to extract meaningful keywords
                source="metadata",
            )
        ]
        contents_dict = {}
        mock_fetch.return_value = (contents, contents_dict)

        queries = await generator.generate_initial_queries_individual(resources)

        # Should handle gracefully (may return empty or very short query)
        assert isinstance(queries, list)
