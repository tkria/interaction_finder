"""
Test backend validation for ReverseSearcher.

Verifies that ReverseSearcher validates backend match during initialization,
providing clear error messages when config.search_backend doesn't match
backend.backend_name. Tests cover all backend types and validation scenarios.
"""

import pytest
from unittest.mock import Mock

from interaction_finder.search.reverse.models import (
    ReverseSearchConfig,
    BackendMismatchError,
)
from interaction_finder.search.reverse.searcher import ReverseSearcher


# Mock backend fixtures


@pytest.fixture
def mock_pubmed_backend():
    """Mock PubMed backend with backend_name attribute."""
    backend = Mock()
    backend.backend_name = "pubmed"
    return backend


@pytest.fixture
def mock_perplexica_backend():
    """Mock Perplexica backend with backend_name attribute."""
    backend = Mock()
    backend.backend_name = "perplexica"
    return backend


@pytest.fixture
def mock_openai_backend():
    """Mock OpenAI backend with backend_name attribute."""
    backend = Mock()
    backend.backend_name = "openai"
    return backend


@pytest.fixture
def mock_cache():
    """Mock search cache."""
    return Mock()


@pytest.fixture
def mock_fetcher():
    """Mock page fetcher."""
    return Mock()


# Test validation method directly


def test_validate_backend_match_passes_when_matching():
    """Validation passes when config.search_backend matches backend.backend_name."""
    config = ReverseSearchConfig(search_backend="pubmed")
    backend = Mock()
    backend.backend_name = "pubmed"

    # Should not raise
    ReverseSearcher._validate_backend_match(config, backend)


def test_validate_backend_match_raises_on_mismatch():
    """Validation raises BackendMismatchError when backends don't match."""
    config = ReverseSearchConfig(search_backend="pubmed")
    backend = Mock()
    backend.backend_name = "perplexica"

    with pytest.raises(BackendMismatchError) as exc_info:
        ReverseSearcher._validate_backend_match(config, backend)

    # Verify error message includes both backends
    error_msg = str(exc_info.value)
    assert "pubmed" in error_msg
    assert "perplexica" in error_msg


def test_validate_backend_match_error_includes_context():
    """BackendMismatchError includes expected/actual/suggestion in context."""
    config = ReverseSearchConfig(search_backend="pubmed")
    backend = Mock()
    backend.backend_name = "openai"

    with pytest.raises(BackendMismatchError) as exc_info:
        ReverseSearcher._validate_backend_match(config, backend)

    # Verify context fields
    error = exc_info.value
    assert error.context["expected_backend"] == "pubmed"
    assert error.context["actual_backend"] == "openai"
    assert "suggestion" in error.context
    # Suggestion should mention both config update and CLI flag
    assert "config" in error.context["suggestion"].lower()
    assert "--backend" in error.context["suggestion"]


# Test validation in ReverseSearcher.__init__


def test_searcher_init_succeeds_with_matching_backend(
    mock_pubmed_backend, mock_cache, mock_fetcher
):
    """ReverseSearcher initializes successfully when backends match."""
    config = ReverseSearchConfig(search_backend="pubmed")

    # Should not raise
    searcher = ReverseSearcher(
        config=config,
        search_backend=mock_pubmed_backend,
        cache=mock_cache,
        fetcher=mock_fetcher,
    )

    # Verify searcher was initialized
    assert searcher.config == config
    assert searcher.backend == mock_pubmed_backend


def test_searcher_init_fails_with_mismatched_backend(
    mock_perplexica_backend, mock_cache, mock_fetcher
):
    """ReverseSearcher initialization fails when backends don't match."""
    config = ReverseSearchConfig(search_backend="pubmed")

    with pytest.raises(BackendMismatchError) as exc_info:
        ReverseSearcher(
            config=config,
            search_backend=mock_perplexica_backend,
            cache=mock_cache,
            fetcher=mock_fetcher,
        )

    # Verify error details
    error = exc_info.value
    assert error.context["expected_backend"] == "pubmed"
    assert error.context["actual_backend"] == "perplexica"


def test_searcher_init_validation_happens_before_component_creation(
    mock_perplexica_backend, mock_cache, mock_fetcher
):
    """Validation occurs before QueryGenerator/ResourceMatcher creation."""
    config = ReverseSearchConfig(search_backend="pubmed")

    # If validation happens first, expensive component creation should be skipped
    with pytest.raises(BackendMismatchError):
        ReverseSearcher(
            config=config,
            search_backend=mock_perplexica_backend,
            cache=mock_cache,
            fetcher=mock_fetcher,
        )

    # Note: We can't directly verify component creation was skipped without
    # instrumentation, but this test documents the expected behavior


# Test all backend combinations


@pytest.mark.parametrize(
    "config_backend,actual_backend_name,should_pass",
    [
        # Matching backends (should pass)
        ("pubmed", "pubmed", True),
        ("perplexica", "perplexica", True),
        ("openai", "openai", True),
        # Mismatched backends (should fail)
        ("pubmed", "perplexica", False),
        ("pubmed", "openai", False),
        ("perplexica", "pubmed", False),
        ("perplexica", "openai", False),
        ("openai", "pubmed", False),
        ("openai", "perplexica", False),
    ],
)
def test_all_backend_combinations(
    config_backend, actual_backend_name, should_pass, mock_cache, mock_fetcher
):
    """Test validation for all backend type combinations."""
    config = ReverseSearchConfig(search_backend=config_backend)
    backend = Mock()
    backend.backend_name = actual_backend_name

    if should_pass:
        # Should initialize successfully
        searcher = ReverseSearcher(
            config=config,
            search_backend=backend,
            cache=mock_cache,
            fetcher=mock_fetcher,
        )
        assert searcher.backend == backend
    else:
        # Should raise BackendMismatchError
        with pytest.raises(BackendMismatchError) as exc_info:
            ReverseSearcher(
                config=config,
                search_backend=backend,
                cache=mock_cache,
                fetcher=mock_fetcher,
            )

        # Verify error contains correct backend names
        error = exc_info.value
        assert error.context["expected_backend"] == config_backend
        assert error.context["actual_backend"] == actual_backend_name


# Test error message quality


def test_error_message_suggests_config_update():
    """Error message suggests updating config to match actual backend."""
    config = ReverseSearchConfig(search_backend="pubmed")
    backend = Mock()
    backend.backend_name = "perplexica"

    with pytest.raises(BackendMismatchError) as exc_info:
        ReverseSearcher._validate_backend_match(config, backend)

    suggestion = exc_info.value.context["suggestion"]
    # Should suggest updating config to actual backend name
    assert "perplexica" in suggestion
    assert "config" in suggestion.lower()


def test_error_message_suggests_cli_flag():
    """Error message suggests using CLI flag to match config."""
    config = ReverseSearchConfig(search_backend="pubmed")
    backend = Mock()
    backend.backend_name = "perplexica"

    with pytest.raises(BackendMismatchError) as exc_info:
        ReverseSearcher._validate_backend_match(config, backend)

    suggestion = exc_info.value.context["suggestion"]
    # Should suggest CLI flag with expected backend
    assert "--backend pubmed" in suggestion


# Test validation timing (performance constraint)


def test_validation_is_fast():
    """Validation adds minimal overhead (<10ms as per constraint)."""
    import time

    config = ReverseSearchConfig(search_backend="pubmed")
    backend = Mock()
    backend.backend_name = "pubmed"

    # Measure validation time
    start = time.time()
    ReverseSearcher._validate_backend_match(config, backend)
    elapsed = time.time() - start

    # Should be nearly instantaneous (well under 10ms)
    assert elapsed < 0.01  # 10ms
