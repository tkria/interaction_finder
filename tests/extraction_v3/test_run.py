"""
Tests for extraction graph V3 run functions.

Tests checkpoint save/resume, document loading, and main entry point.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from interaction_finder.extraction_graph_v3.run import (
    _load_documents,
    save_checkpoint,
    resume_from_checkpoint,
    run_extraction_v3,
)
from interaction_finder.extraction_graph_v3.state import ExtractionStateV3
from interaction_finder.extraction_graph_v2.models import (
    EntityWithQuotes,
    IndividualAssessment,
)
from interaction_finder.resources import ResourcePool


@pytest.mark.asyncio
async def test_load_documents_success(tmp_path):
    """Should load documents into resource pool."""
    # Mock page fetcher
    page_fetcher = AsyncMock()
    page_fetcher.get_chunks.return_value = [
        "Chunk 1 text here.",
        "Chunk 2 text here.",
    ]
    page_fetcher.get_markdown.return_value = (
        "Chunk 1 text here.\n\nChunk 2 text here.",
        "Test Document",
    )

    urls = ["http://example.com/doc1"]

    # Load documents
    resource_pool = await _load_documents(urls, page_fetcher)

    # Verify
    assert len(resource_pool.resources) == 1
    resource = resource_pool.resources[0]
    assert resource.title == "Test Document"
    assert "Chunk 1 text here" in resource.text
    assert resource.chunks is not None
    assert len(resource.chunks) == 2


@pytest.mark.asyncio
async def test_load_documents_handles_failures(tmp_path):
    """Should handle document loading failures gracefully."""
    # Mock page fetcher with failures
    page_fetcher = AsyncMock()
    page_fetcher.get_chunks.side_effect = [
        Exception("Network error"),
        ["Chunk text"],
    ]
    page_fetcher.get_markdown.return_value = ("Chunk text", "Test Document")

    urls = ["http://example.com/fail", "http://example.com/success"]

    # Load documents
    resource_pool = await _load_documents(urls, page_fetcher)

    # Should have loaded only successful document
    assert len(resource_pool.resources) == 1
    assert resource_pool.resources[0].title == "Test Document"


@pytest.mark.asyncio
async def test_save_checkpoint(tmp_path):
    """Should save checkpoint to JSON file."""
    # Create state with sample data
    resource_pool = ResourcePool()
    resource_pool.add("http://example.com/doc1", "Test Doc", "Sample content")

    state = ExtractionStateV3(resource_pool=resource_pool)

    # Add sample entities (quotes list must have at least 1 item)

    resource = resource_pool.resources[0]
    quote = resource.quote("Sample")
    entity = EntityWithQuotes(
        name="BMPR2",
        kind="gene",
        quotes=[quote],
    )
    state.entities_found["BMPR2"] = entity

    # Add sample assessment
    assessment = IndividualAssessment(
        entity=entity,
        relationship_potential="high",
        related_entities=["PAH"],
        evidence_quotes=[],
        reasoning="Test reasoning",
    )
    state.individual_assessments["BMPR2"] = assessment

    # Save checkpoint
    checkpoint_path = await save_checkpoint("extraction", state, tmp_path)

    # Verify file exists
    assert checkpoint_path.exists()
    assert checkpoint_path.parent == tmp_path

    # Verify content
    data = json.loads(checkpoint_path.read_text())
    assert data["stage"] == "extraction"
    assert "BMPR2" in data["entities_found"]
    assert "BMPR2" in data["assessments"]
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_resume_from_checkpoint(tmp_path):
    """Should restore state from checkpoint."""
    # Create checkpoint file with valid entity quotes
    checkpoint_data = {
        "stage": "assessment",
        "timestamp": "2025-10-08T12:00:00",
        "entities_found": {
            "BMPR2": {
                "name": "BMPR2",
                "kind": "gene",
                "quotes": [
                    {
                        "resource_id": "test_id",
                        "phrase": "BMPR2",
                        "spans": [[0, 5]],
                        "count": 1,
                    }
                ],
            }
        },
        "assessments": {
            "BMPR2": {
                "entity": {
                    "name": "BMPR2",
                    "kind": "gene",
                    "quotes": [
                        {
                            "resource_id": "test_id",
                            "phrase": "BMPR2",
                            "spans": [[0, 5]],
                            "count": 1,
                        }
                    ],
                },
                "relationship_potential": "high",
                "related_entities": ["PAH"],
                "evidence_quotes": [],
                "reasoning": "Test reasoning",
            }
        },
        "candidates": {},
        "pairs": [],
        "metrics": {
            "cache_hits_extraction": 5,
            "cache_misses_extraction": 3,
            "cache_hits_assessment": 2,
            "cache_misses_assessment": 1,
            "candidates_generated": 0,
            "pairs_accepted": 0,
            "pairs_rejected": 0,
        },
    }

    checkpoint_path = tmp_path / "checkpoint_test.json"
    checkpoint_path.write_text(json.dumps(checkpoint_data))

    # Create resource pool with matching content
    resource_pool = ResourcePool()
    resource = resource_pool.add(
        "http://example.com/doc1", "Test Doc", "BMPR2 gene content here"
    )

    # Update checkpoint to use actual resource ID
    checkpoint_data["entities_found"]["BMPR2"]["quotes"][0]["resource_id"] = (
        resource.id.id
    )
    checkpoint_data["assessments"]["BMPR2"]["entity"]["quotes"][0]["resource_id"] = (
        resource.id.id
    )
    checkpoint_path.write_text(json.dumps(checkpoint_data))

    # Resume from checkpoint
    state, start_stage = await resume_from_checkpoint(checkpoint_path, resource_pool)

    # Verify state restored
    assert start_stage == "assessment"
    assert len(state.entities_found) == 1
    assert "BMPR2" in state.entities_found
    assert state.entities_found["BMPR2"].name == "BMPR2"
    assert len(state.individual_assessments) == 1
    assert "BMPR2" in state.individual_assessments
    assert state.metrics.cache_hits_extraction == 5
    assert state.metrics.cache_misses_extraction == 3


@pytest.mark.asyncio
async def test_run_extraction_v3_basic(tmp_path):
    """Should run complete pipeline and return result."""
    # Mock page fetcher
    page_fetcher = AsyncMock()
    page_fetcher.get_chunks.return_value = ["Sample content"]
    page_fetcher.get_markdown.return_value = ("Sample content", "Test Doc")

    # Mock config
    config = MagicMock()
    config.task.entity_kinds = [
        MagicMock(kind_name="gene"),
        MagicMock(kind_name="disease"),
    ]
    config.task.relation = "interaction"
    config.task.context = "Test context"
    config.task.get_kind_names.return_value = ["gene", "disease"]
    config.agents = {"_": MagicMock(llm="openai:gpt-4o")}
    config.paths = {}

    urls = ["http://example.com/doc1"]

    # Mock the graph run to avoid actual LLM calls
    with patch(
        "interaction_finder.extraction_graph_v3.run.extraction_graph_v3"
    ) as mock_graph:
        mock_graph.run = AsyncMock()
        mock_graph.nodes = [
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        ]

        # Run extraction
        result = await run_extraction_v3(urls, config, page_fetcher)

        # Verify result structure
        assert result is not None
        assert hasattr(result, "total_pairs")
        assert hasattr(result, "entity_pairs")
        assert hasattr(result, "metadata")
        assert hasattr(result, "cache_stats")
        assert hasattr(result, "stage_metrics")
        assert result.metadata.pipeline_version == "v3"


@pytest.mark.asyncio
async def test_run_extraction_v3_with_checkpoint_resume(tmp_path):
    """Should resume from checkpoint if provided."""
    # Create checkpoint with valid entity quotes
    checkpoint_data = {
        "stage": "GeneratePairCandidates",
        "timestamp": "2025-10-08T12:00:00",
        "entities_found": {
            "BMPR2": {
                "name": "BMPR2",
                "kind": "gene",
                "quotes": [
                    {
                        "resource_id": "test_id",
                        "phrase": "BMPR2",
                        "spans": [[0, 5]],
                        "count": 1,
                    }
                ],
            }
        },
        "assessments": {
            "BMPR2": {
                "entity": {
                    "name": "BMPR2",
                    "kind": "gene",
                    "quotes": [
                        {
                            "resource_id": "test_id",
                            "phrase": "BMPR2",
                            "spans": [[0, 5]],
                            "count": 1,
                        }
                    ],
                },
                "relationship_potential": "high",
                "related_entities": [],
                "evidence_quotes": [],
                "reasoning": "Test",
            }
        },
        "candidates": {},
        "pairs": [],
        "metrics": {},
    }

    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(json.dumps(checkpoint_data))

    # Mock page fetcher with content containing BMPR2
    page_fetcher = AsyncMock()
    page_fetcher.get_chunks.return_value = ["BMPR2 gene content"]
    page_fetcher.get_markdown.return_value = ("BMPR2 gene content", "Test Doc")

    # Mock config
    config = MagicMock()
    config.task.entity_kinds = [MagicMock(kind_name="gene")]
    config.task.get_kind_names.return_value = ["gene"]
    config.agents = {"_": MagicMock(llm="openai:gpt-4o")}
    config.paths = {}

    urls = ["http://example.com/doc1"]

    # Create resource pool with matching content and update checkpoint
    test_resource_pool = ResourcePool()
    test_resource = test_resource_pool.add(
        "http://example.com/doc1", "Test Doc", "BMPR2 gene content"
    )

    # Update checkpoint to use actual resource ID
    checkpoint_data["entities_found"]["BMPR2"]["quotes"][0]["resource_id"] = (
        test_resource.id.id
    )
    checkpoint_data["assessments"]["BMPR2"]["entity"]["quotes"][0]["resource_id"] = (
        test_resource.id.id
    )
    checkpoint_path.write_text(json.dumps(checkpoint_data))

    # Mock document loading to return our pre-constructed pool
    async def mock_load_documents(urls, page_fetcher):
        return test_resource_pool

    # Mock the graph run and document loading
    with (
        patch(
            "interaction_finder.extraction_graph_v3.run.extraction_graph_v3"
        ) as mock_graph,
        patch(
            "interaction_finder.extraction_graph_v3.run._load_documents",
            side_effect=mock_load_documents,
        ),
    ):
        mock_graph.run = AsyncMock()
        mock_graph.nodes = [
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        ]

        # Run extraction with checkpoint
        result = await run_extraction_v3(
            urls, config, page_fetcher, checkpoint_path=checkpoint_path
        )

        # Verify it resumed (checkpoint was read)
        assert result is not None
        # Graph run should have been called with resumed state
        assert mock_graph.run.called


@pytest.mark.asyncio
async def test_run_extraction_v3_no_documents_loaded():
    """Should return error result if no documents load successfully."""
    # Mock page fetcher that always fails
    page_fetcher = AsyncMock()
    page_fetcher.get_chunks.side_effect = Exception("Network error")

    # Mock config
    config = MagicMock()
    config.task.get_kind_names.return_value = ["gene"]
    config.agents = {"_": MagicMock(llm="openai:gpt-4o")}

    urls = ["http://example.com/fail"]

    # Run extraction
    result = await run_extraction_v3(urls, config, page_fetcher)

    # Should return error result
    assert result.total_pairs == 0
    assert len(result.errors) > 0
    assert "No documents loaded" in result.errors[0]["error"]
