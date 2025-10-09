"""
Tests for ReverseSearcher integration with InvestigationLogger.

This module tests that ReverseSearcher correctly integrates with InvestigationLogger
for session start/end logging, error logging, and maintaining backward compatibility.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from interaction_finder.search.base import SearchBackend
from interaction_finder.search.cache import SearchCache
from interaction_finder.search.reverse.investigation_logger import InvestigationLogger
from interaction_finder.search.reverse.models import (
    KnownResource,
    ReverseSearchConfig,
    ReverseSearchError,
)
from interaction_finder.search.reverse.searcher import ReverseSearcher


@pytest.fixture
def mock_backend():
    """Create mock search backend."""
    backend = AsyncMock(spec=SearchBackend)
    backend.backend_name = "pubmed"
    backend.search = AsyncMock()
    return backend


@pytest.fixture
def mock_cache():
    """Create mock search cache."""
    cache = AsyncMock(spec=SearchCache)
    cache.get = AsyncMock(return_value=None)  # Always miss
    cache.set = AsyncMock()
    return cache


@pytest.fixture
def mock_fetcher():
    """Create mock page fetcher."""
    fetcher = MagicMock()
    return fetcher


@pytest.fixture
def sample_resources():
    """Create sample known resources."""
    return [
        KnownResource(
            pmid="12345",
            url="https://example.com/paper1",
            title="Test Paper 1",
            hint_fields={"abstract": "Test abstract 1"},
        ),
        KnownResource(
            pmid="67890",
            url="https://example.com/paper2",
            title="Test Paper 2",
            hint_fields={"abstract": "Test abstract 2"},
        ),
    ]


@pytest.mark.asyncio
async def test_logger_parameter_accepted(mock_backend, mock_cache, mock_fetcher):
    """Test that ReverseSearcher accepts investigation_logger parameter."""
    config = ReverseSearchConfig()
    mock_logger = MagicMock(spec=InvestigationLogger)
    # Should not raise
    searcher = ReverseSearcher(
        config, mock_backend, mock_cache, mock_fetcher, investigation_logger=mock_logger
    )
    assert searcher.inv_logger is mock_logger


@pytest.mark.asyncio
async def test_logger_parameter_optional(mock_backend, mock_cache, mock_fetcher):
    """Test that investigation_logger parameter is optional."""
    config = ReverseSearchConfig()
    # Should not raise without logger
    searcher = ReverseSearcher(config, mock_backend, mock_cache, mock_fetcher)
    assert searcher.inv_logger is None


@pytest.mark.asyncio
async def test_session_start_logged(
    mock_backend, mock_cache, mock_fetcher, sample_resources
):
    """Test that session start is logged when logger provided."""
    config = ReverseSearchConfig()
    mock_logger = AsyncMock(spec=InvestigationLogger)

    searcher = ReverseSearcher(
        config, mock_backend, mock_cache, mock_fetcher, investigation_logger=mock_logger
    )

    # Mock query generator to avoid complex setup
    with patch.object(
        searcher.query_generator,
        "generate_initial_queries",
        new_callable=AsyncMock,
        return_value=["test"],
    ):
        # Mock search to return empty results
        mock_backend.search.return_value = AsyncMock()
        try:
            await searcher.search(sample_resources)
        except Exception:
            pass  # We don't care if search fails, just that session_start was logged

    # Verify session_start was called
    mock_logger.log_session_start.assert_called_once()
    args = mock_logger.log_session_start.call_args[0]
    assert args[0] == sample_resources
    assert args[1] == config
    assert args[2] == "pubmed"


@pytest.mark.asyncio
async def test_session_end_logged(
    mock_backend, mock_cache, mock_fetcher, sample_resources
):
    """Test that session end is logged when logger provided."""
    config = ReverseSearchConfig()
    mock_logger = AsyncMock(spec=InvestigationLogger)

    searcher = ReverseSearcher(
        config, mock_backend, mock_cache, mock_fetcher, investigation_logger=mock_logger
    )

    # Mock query generator to return empty list (no queries)
    with patch.object(
        searcher.query_generator,
        "generate_initial_queries",
        new_callable=AsyncMock,
        return_value=[],
    ):
        await searcher.search(sample_resources)

    # Verify session_end was called
    mock_logger.log_session_end.assert_called_once()
    # Verify it was called with a ReverseSearchSession object
    session_arg = mock_logger.log_session_end.call_args[0][0]
    assert hasattr(session_arg, "total_queries")
    assert hasattr(session_arg, "final_coverage")


@pytest.mark.asyncio
async def test_error_logged_on_query_generation_failure(
    mock_backend, mock_cache, mock_fetcher, sample_resources
):
    """Test that errors during query generation are logged."""
    config = ReverseSearchConfig()
    mock_logger = AsyncMock(spec=InvestigationLogger)

    searcher = ReverseSearcher(
        config, mock_backend, mock_cache, mock_fetcher, investigation_logger=mock_logger
    )

    # Mock query generator to raise exception
    error_msg = "Mock query generation failure"
    with patch.object(
        searcher.query_generator,
        "generate_initial_queries",
        new_callable=AsyncMock,
        side_effect=ValueError(error_msg),
    ):
        # Run search and expect error
        with pytest.raises(ReverseSearchError):
            await searcher.search(sample_resources)

    # Verify error was logged
    mock_logger.log_error.assert_called_once()
    args = mock_logger.log_error.call_args[0]
    assert args[0] == "query_generation"
    assert isinstance(args[1], ValueError)
    assert str(args[1]) == error_msg
    assert args[2] == sample_resources
    assert args[3] == "abort"


@pytest.mark.asyncio
async def test_no_logging_without_logger(
    mock_backend, mock_cache, mock_fetcher, sample_resources
):
    """Test that no errors occur when logger is not provided."""
    config = ReverseSearchConfig()

    # Create searcher WITHOUT logger
    searcher = ReverseSearcher(config, mock_backend, mock_cache, mock_fetcher)

    # Mock query generator to return empty list
    with patch.object(
        searcher.query_generator,
        "generate_initial_queries",
        new_callable=AsyncMock,
        return_value=[],
    ):
        # Should complete without errors
        session = await searcher.search(sample_resources)
        assert session is not None
