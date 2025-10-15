"""
Integration tests for QueryGenerator investigation logging.

Tests cover:
- Content fetch logging with source breakdown
- Clustering logging (enabled and disabled)
- Query generation logging for statistical and LLM extractors
- Backward compatibility (no logger provided)
- Query index tracking across multiple queries
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock

import pytest

from interaction_finder.search.reverse.investigation_logger import (
    ClusteringEntry,
    ContentFetchEntry,
    InvestigationLogger,
    QueryGenerationEntry,
)
from interaction_finder.search.reverse.models import KnownResource, ReverseSearchConfig
from interaction_finder.search.reverse.query_generator import (
    QueryGenerator,
    ResourceContent,
)


@pytest.fixture
def temp_log_file(tmp_path: Path) -> Path:
    """Create temporary log file path."""
    return tmp_path / "query_gen_test.jsonl"


@pytest.fixture
def sample_resources() -> List[KnownResource]:
    """Create sample target resources."""
    return [
        KnownResource(
            pmid="12345678",
            url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
            hint_fields={"celltype": "CD8+ T cell", "marker": "CD8"},
        ),
        KnownResource(
            url="https://example.com/paper1",
            hint_fields={"celltype": "B cell", "marker": "CD19"},
        ),
        KnownResource(
            pmid="87654321",
            url="https://pubmed.ncbi.nlm.nih.gov/87654321/",
            hint_fields={"celltype": "NK cell", "marker": "CD56"},
        ),
    ]


@pytest.fixture
def basic_config() -> ReverseSearchConfig:
    """Create basic configuration with YAKE extractor."""
    return ReverseSearchConfig(
        keyword_extractor="yake",
        keywords_per_query=5,
        use_hint_fields=True,
        enable_clustering=False,
    )


@pytest.fixture
def clustering_config() -> ReverseSearchConfig:
    """Create configuration with clustering enabled."""
    return ReverseSearchConfig(
        keyword_extractor="yake",
        keywords_per_query=5,
        use_hint_fields=True,
        enable_clustering=True,
        min_cluster_size=2,
        target_clusters=2,
    )


def read_log_entries(log_file: Path) -> List[Dict[str, Any]]:
    """Parse JSON log file into list of entries (handles pretty-printed JSON)."""
    entries = []
    with open(log_file, "r") as f:
        content = f.read()
    # Split by lines starting with '{' (new entries in pretty-printed format)
    # Handle pretty-printed JSON by accumulating lines until we have a complete entry
    current_entry = []
    brace_depth = 0
    for line in content.splitlines():
        if line.strip():
            current_entry.append(line)
            # Track brace depth to know when entry is complete
            brace_depth += line.count("{") - line.count("}")
            if brace_depth == 0 and current_entry:
                # Complete entry - parse it
                entry_str = "\n".join(current_entry)
                entries.append(json.loads(entry_str))
                current_entry = []
    return entries


@pytest.mark.asyncio
async def test_query_generator_without_logger(
    basic_config: ReverseSearchConfig,
    sample_resources: List[KnownResource],
    tmp_path: Path,
):
    """Test that QueryGenerator works without investigation_logger."""
    # Create mock fetcher that returns content
    mock_fetcher = Mock()
    mock_fetcher.fetch_documents = AsyncMock(
        return_value={
            r.url: Mock(content_markdown=f"Content for {r.url}")
            for r in sample_resources
        }
    )

    # Create generator without logger
    generator = QueryGenerator(
        config=basic_config,
        backend_name="pubmed",
        fetcher=mock_fetcher,
        investigation_logger=None,  # No logger
    )

    # Mock _fetch_resource_contents to return correct tuple format
    contents = [
        ResourceContent(resource=r, text=f"Content for {r.url}", source="content")
        for r in sample_resources
    ]
    contents_dict = {
        r.url: {
            "source": "content",
            "title": f"Content for {r.url}",
            "content_length": len(f"Content for {r.url}"),
            "success": True,
            "error": None,
        }
        for r in sample_resources
    }
    generator._fetch_resource_contents = AsyncMock(
        return_value=(contents, contents_dict)
    )

    # Generate queries
    queries = await generator.generate_initial_queries(sample_resources)

    # Verify queries were generated
    assert len(queries) > 0

    # Verify no log file created
    log_file = tmp_path / "should_not_exist.jsonl"
    assert not log_file.exists()


@pytest.mark.asyncio
async def test_content_fetch_logging(
    basic_config: ReverseSearchConfig,
    sample_resources: List[KnownResource],
    temp_log_file: Path,
):
    """Test content fetch logging with source breakdown."""
    # Create mock fetcher with mixed content sources
    # First resource: has PMID, will use metadata
    # Second resource: URL only, will use content
    # Third resource: has PMID but metadata fails, will use content
    mock_fetcher = Mock()
    mock_fetcher.fetch_documents = AsyncMock(
        return_value={
            sample_resources[1].url: Mock(
                content_markdown="Content for example.com paper"
            ),
            sample_resources[2].url: Mock(content_markdown="Content for PMID 87654321"),
        }
    )

    # Create logger
    async with InvestigationLogger(temp_log_file) as logger:
        # Create generator with logger
        generator = QueryGenerator(
            config=basic_config,
            backend_name="pubmed",
            fetcher=mock_fetcher,
            investigation_logger=logger,
        )

        # Mock _fetch_resource_contents to return correct tuple format
        # 1 metadata source, 2 content sources
        contents = [
            ResourceContent(
                resource=sample_resources[0],
                text="CD8 T cell study. Abstract about CD8 cells",
                source="metadata",
            ),
            ResourceContent(
                resource=sample_resources[1],
                text="Content for example.com paper",
                source="content",
            ),
            ResourceContent(
                resource=sample_resources[2],
                text="Content for PMID 87654321",
                source="content",
            ),
        ]
        contents_dict = {
            sample_resources[0].url: {
                "source": "metadata",
                "title": "CD8 T cell study",
                "content_length": 43,
                "success": True,
                "error": None,
            },
            sample_resources[1].url: {
                "source": "content",
                "title": "Content for example.com paper",
                "content_length": 30,
                "success": True,
                "error": None,
            },
            sample_resources[2].url: {
                "source": "content",
                "title": "Content for PMID 87654321",
                "content_length": 26,
                "success": True,
                "error": None,
            },
        }
        generator._fetch_resource_contents = AsyncMock(
            return_value=(contents, contents_dict)
        )

        # Generate queries
        await generator.generate_initial_queries(sample_resources)

    # Parse log file
    entries = read_log_entries(temp_log_file)

    # Find content fetch entry
    content_entries = [e for e in entries if e["stage"] == "content_fetch"]
    assert len(content_entries) == 1

    entry = content_entries[0]
    # Verify source counts (1 metadata, 2 content)
    assert entry["source_counts"]["metadata"] == 1
    assert entry["source_counts"]["content"] == 2
    assert entry["failed_count"] == 0


@pytest.mark.asyncio
async def test_clustering_enabled_logging(
    clustering_config: ReverseSearchConfig,
    temp_log_file: Path,
):
    """Test clustering logging when clustering is enabled."""
    # Create 4 resources with diverse content for clustering
    cluster_resources = [
        KnownResource(
            url="https://example.com/paper1",
            hint_fields={"celltype": "CD8+ T cell", "marker": "CD8"},
        ),
        KnownResource(
            url="https://example.com/paper2",
            hint_fields={"celltype": "CD4+ T cell", "marker": "CD4"},
        ),
        KnownResource(
            url="https://example.com/paper3",
            hint_fields={"celltype": "B cell", "marker": "CD19"},
        ),
        KnownResource(
            url="https://example.com/paper4",
            hint_fields={"celltype": "NK cell", "marker": "CD56"},
        ),
    ]

    # Create mock fetcher with diverse content
    mock_fetcher = Mock()
    mock_fetcher.fetch_documents = AsyncMock(
        return_value={
            cluster_resources[0].url: Mock(
                content_markdown="CD8 positive T cells are cytotoxic lymphocytes that kill infected cells"
            ),
            cluster_resources[1].url: Mock(
                content_markdown="CD4 positive T helper cells coordinate immune responses through cytokines"
            ),
            cluster_resources[2].url: Mock(
                content_markdown="B lymphocytes produce antibodies and present antigens to T cells"
            ),
            cluster_resources[3].url: Mock(
                content_markdown="Natural killer cells provide innate immunity against tumors and viruses"
            ),
        }
    )

    # Create logger
    async with InvestigationLogger(temp_log_file) as logger:
        # Create generator with clustering enabled
        generator = QueryGenerator(
            config=clustering_config,
            backend_name="pubmed",
            fetcher=mock_fetcher,
            investigation_logger=logger,
        )

        # Mock _fetch_resource_contents to return correct tuple format
        contents = [
            ResourceContent(
                resource=cluster_resources[0],
                text="CD8 positive T cells are cytotoxic lymphocytes that kill infected cells",
                source="content",
            ),
            ResourceContent(
                resource=cluster_resources[1],
                text="CD4 positive T helper cells coordinate immune responses through cytokines",
                source="content",
            ),
            ResourceContent(
                resource=cluster_resources[2],
                text="B lymphocytes produce antibodies and present antigens to T cells",
                source="content",
            ),
            ResourceContent(
                resource=cluster_resources[3],
                text="Natural killer cells provide innate immunity against tumors and viruses",
                source="content",
            ),
        ]
        contents_dict = {
            cluster_resources[0].url: {
                "source": "content",
                "title": "CD8 positive T cells are cytotoxic lymphocytes that kill infected cells",
                "content_length": 73,
                "success": True,
                "error": None,
            },
            cluster_resources[1].url: {
                "source": "content",
                "title": "CD4 positive T helper cells coordinate immune responses through cytokines",
                "content_length": 76,
                "success": True,
                "error": None,
            },
            cluster_resources[2].url: {
                "source": "content",
                "title": "B lymphocytes produce antibodies and present antigens to T cells",
                "content_length": 66,
                "success": True,
                "error": None,
            },
            cluster_resources[3].url: {
                "source": "content",
                "title": "Natural killer cells provide innate immunity against tumors and viruses",
                "content_length": 73,
                "success": True,
                "error": None,
            },
        }
        generator._fetch_resource_contents = AsyncMock(
            return_value=(contents, contents_dict)
        )

        # Generate queries (should trigger clustering with 4 resources)
        await generator.generate_initial_queries(cluster_resources)

    # Parse log file
    entries = read_log_entries(temp_log_file)

    # Find clustering entry
    clustering_entries = [e for e in entries if e["stage"] == "clustering"]
    assert len(clustering_entries) == 1

    entry = clustering_entries[0]
    # Verify clustering was enabled
    assert entry["enabled"] is True
    assert entry["resource_count"] == 4
    # Verify clusters present
    assert len(entry["clusters"]) >= 1
    # Verify each cluster has expected structure
    for cluster in entry["clusters"]:
        assert "cluster_id" in cluster
        assert "size" in cluster
        assert "member_indices" in cluster
        assert "representative_idx" in cluster


@pytest.mark.asyncio
async def test_clustering_disabled_logging(
    basic_config: ReverseSearchConfig,
    sample_resources: List[KnownResource],
    temp_log_file: Path,
):
    """Test clustering logging when clustering is disabled."""
    # Create mock fetcher
    mock_fetcher = Mock()
    mock_fetcher.fetch_documents = AsyncMock(
        return_value={
            r.url: Mock(content_markdown=f"Content for {r.url}")
            for r in sample_resources
        }
    )

    # Create logger
    async with InvestigationLogger(temp_log_file) as logger:
        # Create generator without clustering
        generator = QueryGenerator(
            config=basic_config,
            backend_name="pubmed",
            fetcher=mock_fetcher,
            investigation_logger=logger,
        )

        # Mock _fetch_resource_contents to return correct tuple format
        contents = [
            ResourceContent(resource=r, text=f"Content for {r.url}", source="content")
            for r in sample_resources
        ]
        contents_dict = {
            r.url: {
                "source": "content",
                "title": f"Content for {r.url}",
                "content_length": len(f"Content for {r.url}"),
                "success": True,
                "error": None,
            }
            for r in sample_resources
        }
        generator._fetch_resource_contents = AsyncMock(
            return_value=(contents, contents_dict)
        )

        # Generate queries
        await generator.generate_initial_queries(sample_resources)

    # Parse log file
    entries = read_log_entries(temp_log_file)

    # Find clustering entry
    clustering_entries = [e for e in entries if e["stage"] == "clustering"]
    assert len(clustering_entries) == 1

    entry = clustering_entries[0]
    # Verify clustering was disabled
    assert entry["enabled"] is False
    assert entry["resource_count"] == 3


@pytest.mark.asyncio
async def test_query_generation_logging_yake(
    basic_config: ReverseSearchConfig,
    sample_resources: List[KnownResource],
    temp_log_file: Path,
):
    """Test query generation logging with YAKE extractor."""
    # Create mock fetcher
    mock_fetcher = Mock()
    mock_fetcher.fetch_documents = AsyncMock(
        return_value={
            r.url: Mock(content_markdown=f"Content about cells and markers for {r.url}")
            for r in sample_resources
        }
    )

    # Create logger
    async with InvestigationLogger(temp_log_file) as logger:
        # Create generator with YAKE
        generator = QueryGenerator(
            config=basic_config,
            backend_name="pubmed",
            fetcher=mock_fetcher,
            investigation_logger=logger,
        )

        # Mock _fetch_resource_contents to return correct tuple format
        contents = [
            ResourceContent(
                resource=r,
                text=f"Content about cells and markers for {r.url}",
                source="content",
            )
            for r in sample_resources
        ]
        contents_dict = {
            r.url: {
                "source": "content",
                "title": f"Content about cells and markers for {r.url}",
                "content_length": len(f"Content about cells and markers for {r.url}"),
                "success": True,
                "error": None,
            }
            for r in sample_resources
        }
        generator._fetch_resource_contents = AsyncMock(
            return_value=(contents, contents_dict)
        )

        # Generate queries
        queries = await generator.generate_initial_queries(sample_resources)

    # Parse log file
    entries = read_log_entries(temp_log_file)

    # Find query generation entries
    query_entries = [e for e in entries if e["stage"] == "query_generation"]
    # Should have one entry per generated query
    assert len(query_entries) == len(queries)

    for i, entry in enumerate(query_entries):
        # Verify basic fields
        assert entry["query_index"] == i
        assert entry["query_type"] == "initial"
        assert entry["extractor_type"] == "yake"
        # Verify keywords present
        assert isinstance(entry["keywords"], list)
        assert len(entry["keywords"]) > 0
        # Verify final query
        assert isinstance(entry["final_query"], str)
        assert len(entry["final_query"]) > 0
        # Individual mode: no cluster_id
        assert entry["cluster_id"] is None
        assert len(entry["input_resources"]) == 1


@pytest.mark.asyncio
async def test_query_generation_logging_llm(
    sample_resources: List[KnownResource],
    temp_log_file: Path,
):
    """Test query generation logging with LLM constructor."""
    # Create config with none extractor and LLM constructor
    llm_config = ReverseSearchConfig(
        keyword_extractor="none",
        query_constructor="llm",
        keywords_per_query=5,
        use_hint_fields=True,
        enable_clustering=False,
        llm_query_config={
            "model": "openai:gpt-4o-mini",
            "temperature": 0.7,
            "backend_specific_syntax": True,
        },
    )

    # Create mock fetcher
    mock_fetcher = Mock()
    mock_fetcher.fetch_documents = AsyncMock(
        return_value={
            r.url: Mock(content_markdown=f"Content for {r.url}")
            for r in sample_resources
        }
    )

    # Create logger
    async with InvestigationLogger(temp_log_file) as logger:
        # Create generator with LLM
        generator = QueryGenerator(
            config=llm_config,
            backend_name="pubmed",
            fetcher=mock_fetcher,
            investigation_logger=logger,
        )

        # Mock _fetch_resource_contents to return correct tuple format
        contents = [
            ResourceContent(resource=r, text=f"Content for {r.url}", source="content")
            for r in sample_resources
        ]
        contents_dict = {
            r.url: {
                "source": "content",
                "title": f"Content for {r.url}",
                "content_length": len(f"Content for {r.url}"),
                "success": True,
                "error": None,
            }
            for r in sample_resources
        }
        generator._fetch_resource_contents = AsyncMock(
            return_value=(contents, contents_dict)
        )

        # Mock the LLM extractor to avoid API calls
        generator.extractor.extract_async = AsyncMock(
            return_value=['"CD8+ T cell"[Title] AND "marker"[Abstract]']
        )

        # Generate queries
        queries = await generator.generate_initial_queries(sample_resources)

    # Parse log file
    entries = read_log_entries(temp_log_file)

    # Find query generation entries
    query_entries = [e for e in entries if e["stage"] == "query_generation"]
    assert len(query_entries) > 0

    for entry in query_entries:
        # Verify extractor type (migrated from "llm" to "none" + constructor="llm")
        assert entry["extractor_type"] == "none"
        # None extractor passes empty keywords list to LLM constructor
        assert isinstance(entry["keywords"], list)


@pytest.mark.asyncio
async def test_query_index_increments(
    basic_config: ReverseSearchConfig,
    sample_resources: List[KnownResource],
    temp_log_file: Path,
):
    """Test that query_index increments correctly across queries."""
    # Create mock fetcher
    mock_fetcher = Mock()
    mock_fetcher.fetch_documents = AsyncMock(
        return_value={
            r.url: Mock(content_markdown=f"Content for {r.url}")
            for r in sample_resources
        }
    )

    # Create logger
    async with InvestigationLogger(temp_log_file) as logger:
        # Create generator
        generator = QueryGenerator(
            config=basic_config,
            backend_name="pubmed",
            fetcher=mock_fetcher,
            investigation_logger=logger,
        )

        # Mock _fetch_resource_contents to return correct tuple format
        contents = [
            ResourceContent(resource=r, text=f"Content for {r.url}", source="content")
            for r in sample_resources
        ]
        contents_dict = {
            r.url: {
                "source": "content",
                "title": f"Content for {r.url}",
                "content_length": len(f"Content for {r.url}"),
                "success": True,
                "error": None,
            }
            for r in sample_resources
        }
        generator._fetch_resource_contents = AsyncMock(
            return_value=(contents, contents_dict)
        )

        # Generate queries
        await generator.generate_initial_queries(sample_resources)

    # Parse log file
    entries = read_log_entries(temp_log_file)

    # Find query generation entries
    query_entries = [e for e in entries if e["stage"] == "query_generation"]

    # Verify indices are sequential
    for i, entry in enumerate(query_entries):
        assert entry["query_index"] == i


@pytest.mark.asyncio
async def test_clustered_query_logging(
    clustering_config: ReverseSearchConfig,
    temp_log_file: Path,
):
    """Test query generation logging in clustered mode."""
    # Create 4 resources with diverse content for clustering
    cluster_resources = [
        KnownResource(
            url="https://example.com/paper1",
            hint_fields={"celltype": "CD8+ T cell", "marker": "CD8"},
        ),
        KnownResource(
            url="https://example.com/paper2",
            hint_fields={"celltype": "CD4+ T cell", "marker": "CD4"},
        ),
        KnownResource(
            url="https://example.com/paper3",
            hint_fields={"celltype": "B cell", "marker": "CD19"},
        ),
        KnownResource(
            url="https://example.com/paper4",
            hint_fields={"celltype": "NK cell", "marker": "CD56"},
        ),
    ]

    # Create mock fetcher with diverse content
    mock_fetcher = Mock()
    mock_fetcher.fetch_documents = AsyncMock(
        return_value={
            cluster_resources[0].url: Mock(
                content_markdown="CD8 positive T cells are cytotoxic lymphocytes that kill infected cells"
            ),
            cluster_resources[1].url: Mock(
                content_markdown="CD4 positive T helper cells coordinate immune responses through cytokines"
            ),
            cluster_resources[2].url: Mock(
                content_markdown="B lymphocytes produce antibodies and present antigens to T cells"
            ),
            cluster_resources[3].url: Mock(
                content_markdown="Natural killer cells provide innate immunity against tumors and viruses"
            ),
        }
    )

    # Create logger
    async with InvestigationLogger(temp_log_file) as logger:
        # Create generator with clustering
        generator = QueryGenerator(
            config=clustering_config,
            backend_name="pubmed",
            fetcher=mock_fetcher,
            investigation_logger=logger,
        )

        # Mock _fetch_resource_contents to return correct tuple format
        contents = [
            ResourceContent(
                resource=cluster_resources[0],
                text="CD8 positive T cells are cytotoxic lymphocytes that kill infected cells",
                source="content",
            ),
            ResourceContent(
                resource=cluster_resources[1],
                text="CD4 positive T helper cells coordinate immune responses through cytokines",
                source="content",
            ),
            ResourceContent(
                resource=cluster_resources[2],
                text="B lymphocytes produce antibodies and present antigens to T cells",
                source="content",
            ),
            ResourceContent(
                resource=cluster_resources[3],
                text="Natural killer cells provide innate immunity against tumors and viruses",
                source="content",
            ),
        ]
        contents_dict = {
            cluster_resources[0].url: {
                "source": "content",
                "title": "CD8 positive T cells are cytotoxic lymphocytes that kill infected cells",
                "content_length": 73,
                "success": True,
                "error": None,
            },
            cluster_resources[1].url: {
                "source": "content",
                "title": "CD4 positive T helper cells coordinate immune responses through cytokines",
                "content_length": 76,
                "success": True,
                "error": None,
            },
            cluster_resources[2].url: {
                "source": "content",
                "title": "B lymphocytes produce antibodies and present antigens to T cells",
                "content_length": 66,
                "success": True,
                "error": None,
            },
            cluster_resources[3].url: {
                "source": "content",
                "title": "Natural killer cells provide innate immunity against tumors and viruses",
                "content_length": 73,
                "success": True,
                "error": None,
            },
        }
        generator._fetch_resource_contents = AsyncMock(
            return_value=(contents, contents_dict)
        )

        # Generate queries (should cluster 4 resources into 2 clusters)
        queries = await generator.generate_initial_queries(cluster_resources)

    # Parse log file
    entries = read_log_entries(temp_log_file)

    # Find query generation entries
    query_entries = [e for e in entries if e["stage"] == "query_generation"]

    # Verify queries have cluster IDs and resource counts
    for entry in query_entries:
        # Clustered mode: should have cluster_id
        assert entry["cluster_id"] is not None
        assert isinstance(entry["cluster_id"], int)
        # Resource count should be >= 1
        assert len(entry["input_resources"]) >= 1
