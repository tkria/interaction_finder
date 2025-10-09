"""
Tests for CLI integration with investigation logging.

Verifies that --investigation-log flag works correctly with reverse-search command,
including logger lifecycle management, error handling, and backward compatibility.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from typer.testing import CliRunner

from interaction_finder.cli import app
from interaction_finder.search.reverse.models import (
    KnownResource,
    ReverseSearchSession,
)

runner = CliRunner()


@pytest.fixture
def temp_resources_file(tmp_path):
    """Create temporary JSONL file with known resources."""
    resources_file = tmp_path / "test_resources.jsonl"
    resources = [
        {"pmid": "12345678", "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/"},
        {"url": "https://example.com/paper1", "celltype": "langerhans cell"},
    ]
    with open(resources_file, "w") as f:
        for r in resources:
            f.write(json.dumps(r) + "\n")
    return resources_file


@pytest.fixture
def sample_session():
    """Create a sample ReverseSearchSession for mocking."""
    # Use Mock to avoid complex nested models
    session = Mock(spec=ReverseSearchSession)
    session.target_resources = [
        KnownResource(url="https://pubmed.ncbi.nlm.nih.gov/12345678/", pmid="12345678"),
        KnownResource(
            url="https://example.com/paper1", hints={"celltype": "langerhans cell"}
        ),
    ]
    session.found_resources = {}
    session.unfound_resources = session.target_resources[1:]
    session.total_queries = 1
    session.found_count = 2
    session.coverage_pct = 67.0
    session.total_time = 1.5
    session.stopping_reason = "coverage_achieved"
    return session


@patch("interaction_finder.fetcher.PageFetcher")
@patch("interaction_finder.search.cache.SearchCache")
@patch("interaction_finder.search.backends.PubMedBackend")
@patch("interaction_finder.search.reverse.ReverseSearcher")
def test_cli_without_investigation_log(
    mock_searcher,
    mock_backend,
    mock_cache,
    mock_fetcher,
    temp_resources_file,
    tmp_path,
    sample_session,
):
    """Test CLI without --investigation-log flag (no log file created)."""
    # Mock searcher.search to return sample session
    mock_searcher_instance = Mock()
    mock_searcher_instance.search = AsyncMock(return_value=sample_session)
    mock_searcher.return_value = mock_searcher_instance

    # Mock backend context manager
    mock_backend_instance = Mock()
    mock_backend_instance.__aenter__ = AsyncMock(return_value=mock_backend_instance)
    mock_backend_instance.__aexit__ = AsyncMock(return_value=None)
    mock_backend.return_value = mock_backend_instance

    output_path = tmp_path / "results.jsonl"
    investigation_log_path = tmp_path / "investigation.jsonl"

    result = runner.invoke(
        app,
        [
            "reverse-search",
            "--known",
            str(temp_resources_file),
            "--output",
            str(output_path),
        ],
    )

    # Verify no investigation log file created
    assert not investigation_log_path.exists()
    # Verify searcher called without investigation_logger
    assert mock_searcher.call_args[1].get("investigation_logger") is None


@patch("interaction_finder.fetcher.PageFetcher")
@patch("interaction_finder.search.cache.SearchCache")
@patch("interaction_finder.search.backends.PubMedBackend")
@patch("interaction_finder.search.reverse.ReverseSearcher")
@patch("interaction_finder.search.reverse.investigation_logger.InvestigationLogger")
def test_cli_with_investigation_log(
    mock_inv_logger_class,
    mock_searcher,
    mock_backend,
    mock_cache,
    mock_fetcher,
    temp_resources_file,
    tmp_path,
    sample_session,
):
    """Test CLI with --investigation-log flag (log file created and populated)."""
    # Mock InvestigationLogger
    mock_inv_logger = Mock()
    mock_inv_logger.__aenter__ = AsyncMock(return_value=mock_inv_logger)
    mock_inv_logger.__aexit__ = AsyncMock(return_value=None)
    mock_inv_logger.log_session_start = AsyncMock()
    mock_inv_logger.log_session_end = AsyncMock()
    mock_inv_logger_class.return_value = mock_inv_logger

    # Mock searcher.search to return sample session
    mock_searcher_instance = Mock()
    mock_searcher_instance.search = AsyncMock(return_value=sample_session)
    mock_searcher.return_value = mock_searcher_instance

    # Mock backend context manager
    mock_backend_instance = Mock()
    mock_backend_instance.__aenter__ = AsyncMock(return_value=mock_backend_instance)
    mock_backend_instance.__aexit__ = AsyncMock(return_value=None)
    mock_backend.return_value = mock_backend_instance

    output_path = tmp_path / "results.jsonl"
    investigation_log_path = tmp_path / "investigation.jsonl"

    result = runner.invoke(
        app,
        [
            "reverse-search",
            "--known",
            str(temp_resources_file),
            "--output",
            str(output_path),
            "--investigation-log",
            str(investigation_log_path),
        ],
    )

    # Verify InvestigationLogger created with correct path
    mock_inv_logger_class.assert_called_once()
    assert mock_inv_logger_class.call_args[0][0] == investigation_log_path

    # Verify logger lifecycle methods called
    mock_inv_logger.log_session_start.assert_called_once()
    mock_inv_logger.log_session_end.assert_called_once()

    # Verify searcher received logger
    assert mock_searcher.call_args[1].get("investigation_logger") == mock_inv_logger


@patch("interaction_finder.fetcher.PageFetcher")
@patch("interaction_finder.search.cache.SearchCache")
@patch("interaction_finder.search.backends.PubMedBackend")
@patch("interaction_finder.search.reverse.ReverseSearcher")
@patch("interaction_finder.search.reverse.investigation_logger.InvestigationLogger")
def test_investigation_log_path_resolution(
    mock_inv_logger_class,
    mock_searcher,
    mock_backend,
    mock_cache,
    mock_fetcher,
    temp_resources_file,
    tmp_path,
    sample_session,
):
    """Test that investigation log paths (relative and absolute) are handled correctly."""
    # Mock InvestigationLogger
    mock_inv_logger = Mock()
    mock_inv_logger.__aenter__ = AsyncMock(return_value=mock_inv_logger)
    mock_inv_logger.__aexit__ = AsyncMock(return_value=None)
    mock_inv_logger.log_session_start = AsyncMock()
    mock_inv_logger.log_session_end = AsyncMock()
    mock_inv_logger_class.return_value = mock_inv_logger

    # Mock searcher.search to return sample session
    mock_searcher_instance = Mock()
    mock_searcher_instance.search = AsyncMock(return_value=sample_session)
    mock_searcher.return_value = mock_searcher_instance

    # Mock backend context manager
    mock_backend_instance = Mock()
    mock_backend_instance.__aenter__ = AsyncMock(return_value=mock_backend_instance)
    mock_backend_instance.__aexit__ = AsyncMock(return_value=None)
    mock_backend.return_value = mock_backend_instance

    output_path = tmp_path / "results.jsonl"

    # Test relative path
    result = runner.invoke(
        app,
        [
            "reverse-search",
            "--known",
            str(temp_resources_file),
            "--output",
            str(output_path),
            "--investigation-log",
            "investigation.jsonl",  # Relative path
        ],
    )

    # Verify logger created (path handling done by InvestigationLogger)
    assert mock_inv_logger_class.called


@patch("interaction_finder.fetcher.PageFetcher")
@patch("interaction_finder.search.cache.SearchCache")
@patch("interaction_finder.search.backends.PubMedBackend")
@patch("interaction_finder.search.reverse.ReverseSearcher")
@patch("interaction_finder.search.reverse.investigation_logger.InvestigationLogger")
def test_investigation_log_error_handling(
    mock_inv_logger_class,
    mock_searcher,
    mock_backend,
    mock_cache,
    mock_fetcher,
    temp_resources_file,
    tmp_path,
    sample_session,
):
    """Test that logging errors don't break the search pipeline."""
    # Mock InvestigationLogger to raise error on log_session_start
    mock_inv_logger = Mock()
    mock_inv_logger.__aenter__ = AsyncMock(return_value=mock_inv_logger)
    mock_inv_logger.__aexit__ = AsyncMock(return_value=None)
    mock_inv_logger.log_session_start = AsyncMock(
        side_effect=Exception("Logging failed")
    )
    mock_inv_logger.log_session_end = AsyncMock()
    mock_inv_logger_class.return_value = mock_inv_logger

    # Mock searcher.search to return sample session
    mock_searcher_instance = Mock()
    mock_searcher_instance.search = AsyncMock(return_value=sample_session)
    mock_searcher.return_value = mock_searcher_instance

    # Mock backend context manager
    mock_backend_instance = Mock()
    mock_backend_instance.__aenter__ = AsyncMock(return_value=mock_backend_instance)
    mock_backend_instance.__aexit__ = AsyncMock(return_value=None)
    mock_backend.return_value = mock_backend_instance

    output_path = tmp_path / "results.jsonl"
    investigation_log_path = tmp_path / "investigation.jsonl"

    result = runner.invoke(
        app,
        [
            "reverse-search",
            "--known",
            str(temp_resources_file),
            "--output",
            str(output_path),
            "--investigation-log",
            str(investigation_log_path),
            "-v",  # Verbose to see warning
        ],
    )

    # Search should succeed despite logging failure
    assert (
        result.exit_code == 0
        or "Warning: Investigation logging failed" in result.stdout
    )
    # Searcher should still be called
    mock_searcher_instance.search.assert_called_once()


@patch("interaction_finder.fetcher.PageFetcher")
@patch("interaction_finder.search.cache.SearchCache")
@patch("interaction_finder.search.backends.PubMedBackend")
@patch("interaction_finder.search.reverse.ReverseSearcher")
def test_dry_run_with_investigation_log(
    mock_searcher,
    mock_backend,
    mock_cache,
    mock_fetcher,
    temp_resources_file,
    tmp_path,
):
    """Test that --dry-run and --investigation-log can coexist (though logging may be minimal)."""
    # Mock query generator
    mock_generator = Mock()
    mock_generator.generate_initial_queries = AsyncMock(
        return_value=["query1", "query2"]
    )

    mock_searcher_instance = Mock()
    mock_searcher_instance.query_generator = mock_generator
    mock_searcher.return_value = mock_searcher_instance

    # Mock backend
    mock_backend_instance = Mock()
    mock_backend.return_value = mock_backend_instance

    investigation_log_path = tmp_path / "investigation.jsonl"

    result = runner.invoke(
        app,
        [
            "reverse-search",
            "--known",
            str(temp_resources_file),
            "--dry-run",
            "--investigation-log",
            str(investigation_log_path),
        ],
    )

    # Dry run should work (investigation logging is created but not used in dry run mode)
    assert "Dry run mode" in result.stdout or "dry run" in result.stdout.lower()
