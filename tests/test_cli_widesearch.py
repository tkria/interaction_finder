"""Tests for widesearch CLI command."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from interaction_finder.cli import app
from interaction_finder.keywords.models import BridgingTermsOut
from interaction_finder.resources import ResourcePool
from interaction_finder.search.models import SearchResult
from interaction_finder.widesearch import WidesearchCheckpoint

runner = CliRunner()


@pytest.fixture
def mock_keywords_file(tmp_path):
    """Create a mock keywords JSON file."""
    keywords_file = tmp_path / "keywords.json"
    bridging_terms = BridgingTermsOut(
        terms=["keyword1", "keyword2", "keyword3"],
        scores=[0.95, 0.87, 0.76],
        total_documents_processed=10,
        rounds_completed=2,
        coverage_assessment="Good coverage achieved across multiple relevant research areas and domains.",
        resources=ResourcePool(),
    )
    keywords_file.write_text(json.dumps(bridging_terms.model_dump(mode="json")))
    return keywords_file


@pytest.fixture
def mock_checkpoint():
    """Create a mock checkpoint for testing."""
    return WidesearchCheckpoint(
        results=[
            SearchResult(
                title="Paper 1",
                url="https://example.com/1",
                snippet="snippet 1",
            ),
            SearchResult(
                title="Paper 2",
                url="https://example.com/2",
                snippet="snippet 2",
            ),
        ],
        queries=["query1", "query2", "query3"],
        query_results={
            "query1": ["https://example.com/1"],
            "query2": ["https://example.com/2"],
            "query3": [],
        },
        resources=ResourcePool(),
        topic="test topic",
        keyphrases=["keyword1", "keyword2"],
        rounds_completed=2,
    )


def test_widesearch_command_basic(tmp_path, mock_keywords_file, mock_checkpoint):
    """Test basic widesearch command execution."""
    output_file = tmp_path / "output.json"

    def mock_asyncio_run(coro):
        """Mock asyncio.run that properly closes the coroutine."""
        coro.close()  # Close coroutine to avoid "never awaited" warning
        return mock_checkpoint

    with patch("interaction_finder.cli.asyncio.run", side_effect=mock_asyncio_run):
        result = runner.invoke(
            app,
            [
                "widesearch",
                str(mock_keywords_file),
                "test topic",
                "-o",
                str(output_file),
            ],
        )

        # Command should succeed
        assert result.exit_code == 0
        # Should contain success message
        assert "Widesearch completed" in result.stdout
        # Output file should be created
        assert output_file.exists()
        # Output should be valid JSON
        checkpoint_data = json.loads(output_file.read_text())
        assert "results" in checkpoint_data
        assert "queries" in checkpoint_data
        assert "resources" in checkpoint_data


def test_widesearch_missing_keywords_file(tmp_path):
    """Test error handling when keywords file doesn't exist."""
    missing_file = tmp_path / "missing.json"

    result = runner.invoke(
        app,
        ["widesearch", str(missing_file), "test topic", "-o", "output.json"],
    )

    # Should fail with appropriate error
    assert result.exit_code == 1
    assert "Keywords file not found" in result.stdout


def test_widesearch_invalid_json(tmp_path):
    """Test error handling with invalid JSON file."""
    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text("not valid json{")

    result = runner.invoke(
        app,
        ["widesearch", str(invalid_file), "test topic", "-o", "output.json"],
    )

    # Should fail with JSON error
    assert result.exit_code == 1
    assert "Invalid keywords file" in result.stdout


def test_widesearch_invalid_schema(tmp_path):
    """Test error handling with wrong schema in JSON file."""
    invalid_schema_file = tmp_path / "wrong_schema.json"
    invalid_schema_file.write_text(json.dumps({"wrong": "schema"}))

    result = runner.invoke(
        app,
        ["widesearch", str(invalid_schema_file), "test topic", "-o", "output.json"],
    )

    # Should fail with validation error
    assert result.exit_code == 1
    assert "Invalid keywords file" in result.stdout


def test_widesearch_with_backend_option(tmp_path, mock_keywords_file, mock_checkpoint):
    """Test widesearch with backend option."""
    output_file = tmp_path / "output.json"

    def mock_asyncio_run(coro):
        """Mock asyncio.run that properly closes the coroutine."""
        coro.close()  # Close coroutine to avoid "never awaited" warning
        return mock_checkpoint

    with patch("interaction_finder.cli.asyncio.run", side_effect=mock_asyncio_run):
        result = runner.invoke(
            app,
            [
                "widesearch",
                str(mock_keywords_file),
                "test topic",
                "-b",
                "pubmed",
                "-o",
                str(output_file),
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()


def test_widesearch_with_max_rounds(tmp_path, mock_keywords_file, mock_checkpoint):
    """Test widesearch with max_rounds override."""
    output_file = tmp_path / "output.json"

    def mock_asyncio_run(coro):
        """Mock asyncio.run that properly closes the coroutine."""
        coro.close()  # Close coroutine to avoid "never awaited" warning
        return mock_checkpoint

    with patch("interaction_finder.cli.asyncio.run", side_effect=mock_asyncio_run):
        result = runner.invoke(
            app,
            [
                "widesearch",
                str(mock_keywords_file),
                "test topic",
                "--max-rounds",
                "3",
                "-o",
                str(output_file),
            ],
        )

        assert result.exit_code == 0


def test_widesearch_with_reranking_config(
    tmp_path, mock_keywords_file, mock_checkpoint
):
    """Test widesearch with reranking configuration via overrides."""
    output_file = tmp_path / "output.json"

    def mock_asyncio_run(coro):
        """Mock asyncio.run that properly closes the coroutine."""
        coro.close()  # Close coroutine to avoid "never awaited" warning
        return mock_checkpoint

    with patch("interaction_finder.cli.asyncio.run", side_effect=mock_asyncio_run):
        # Test with rerank_top_k=0 (disabled) via config override
        with patch(
            "interaction_finder.widesearch.reranker.Reranker"
        ) as mock_reranker_cls:
            result = runner.invoke(
                app,
                [
                    "widesearch",
                    str(mock_keywords_file),
                    "test topic",
                    "-O",
                    "tools.widesearch.rerank_top_k=0",
                    "-o",
                    str(output_file),
                ],
            )

            assert result.exit_code == 0
            # Verify Reranker was NOT instantiated when rerank_top_k=0
            mock_reranker_cls.assert_not_called()

        # Test with rerank_top_k=50 (enabled) via config override
        with patch(
            "interaction_finder.widesearch.reranker.Reranker"
        ) as mock_reranker_cls:
            # Mock the reranker instance and its methods
            mock_reranker = MagicMock()
            mock_reranker._get_model.return_value = None
            mock_reranker_cls.return_value = mock_reranker

            result = runner.invoke(
                app,
                [
                    "widesearch",
                    str(mock_keywords_file),
                    "test topic",
                    "-O",
                    "tools.widesearch.rerank_top_k=50",
                    "-o",
                    str(output_file),
                ],
            )

            assert result.exit_code == 0
            # Verify Reranker WAS instantiated when rerank_top_k=50
            mock_reranker_cls.assert_called_once()
            # Verify the model was pre-loaded
            mock_reranker._get_model.assert_called_once()


def test_widesearch_unknown_backend(tmp_path, mock_keywords_file):
    """Test error handling with unknown backend."""
    result = runner.invoke(
        app,
        [
            "widesearch",
            str(mock_keywords_file),
            "test topic",
            "-b",
            "invalid_backend",
            "-o",
            "output.json",
        ],
    )

    # Should fail with backend error
    assert result.exit_code == 1
    assert "Unknown backend" in result.stdout or "invalid_backend" in result.stdout


def test_widesearch_without_output_file(mock_keywords_file, mock_checkpoint):
    """Test widesearch without output file (should still run)."""

    def mock_asyncio_run(coro):
        """Mock asyncio.run that properly closes the coroutine."""
        coro.close()  # Close coroutine to avoid "never awaited" warning
        return mock_checkpoint

    with patch("interaction_finder.cli.asyncio.run", side_effect=mock_asyncio_run):
        result = runner.invoke(
            app,
            ["widesearch", str(mock_keywords_file), "test topic"],
        )

        # Should succeed without output file
        assert result.exit_code == 0
        assert "Widesearch completed" in result.stdout


def test_widesearch_checkpoint_structure(tmp_path, mock_keywords_file, mock_checkpoint):
    """Test that checkpoint JSON has all required fields."""
    output_file = tmp_path / "checkpoint.json"

    def mock_asyncio_run(coro):
        """Mock asyncio.run that properly closes the coroutine."""
        coro.close()  # Close coroutine to avoid "never awaited" warning
        return mock_checkpoint

    with patch("interaction_finder.cli.asyncio.run", side_effect=mock_asyncio_run):
        result = runner.invoke(
            app,
            [
                "widesearch",
                str(mock_keywords_file),
                "test topic",
                "-o",
                str(output_file),
            ],
        )

        assert result.exit_code == 0

        # Load and validate checkpoint structure
        checkpoint_data = json.loads(output_file.read_text())

        # All required fields should be present
        assert "results" in checkpoint_data
        assert "queries" in checkpoint_data
        assert "query_results" in checkpoint_data
        assert "resources" in checkpoint_data
        assert "topic" in checkpoint_data
        assert "keyphrases" in checkpoint_data
        assert "rounds_completed" in checkpoint_data

        # Check types
        assert isinstance(checkpoint_data["results"], list)
        assert isinstance(checkpoint_data["queries"], list)
        assert isinstance(checkpoint_data["query_results"], dict)
        assert isinstance(
            checkpoint_data["resources"], list
        )  # New format: list of resources
        assert isinstance(checkpoint_data["topic"], str)
        assert isinstance(checkpoint_data["keyphrases"], list)
        assert isinstance(checkpoint_data["rounds_completed"], int)


def test_widesearch_with_config_overrides(
    tmp_path, mock_keywords_file, mock_checkpoint
):
    """Test widesearch with config overrides."""
    output_file = tmp_path / "output.json"

    def mock_asyncio_run(coro):
        """Mock asyncio.run that properly closes the coroutine."""
        coro.close()  # Close coroutine to avoid "never awaited" warning
        return mock_checkpoint

    with patch("interaction_finder.cli.asyncio.run", side_effect=mock_asyncio_run):
        result = runner.invoke(
            app,
            [
                "widesearch",
                str(mock_keywords_file),
                "test topic",
                "-O",
                "tools.widesearch.results_per_query=100",
                "-o",
                str(output_file),
            ],
        )

        assert result.exit_code == 0


def test_widesearch_verbose_mode(tmp_path, mock_keywords_file, mock_checkpoint):
    """Test widesearch with verbose flag."""
    output_file = tmp_path / "output.json"

    def mock_asyncio_run(coro):
        """Mock asyncio.run that properly closes the coroutine."""
        coro.close()  # Close coroutine to avoid "never awaited" warning
        return mock_checkpoint

    with patch("interaction_finder.cli.asyncio.run", side_effect=mock_asyncio_run):
        result = runner.invoke(
            app,
            [
                "widesearch",
                str(mock_keywords_file),
                "test topic",
                "-v",
                "-o",
                str(output_file),
            ],
        )

        assert result.exit_code == 0
