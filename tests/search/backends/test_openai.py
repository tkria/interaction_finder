"""
Comprehensive test suite for OpenAI search backend.

Tests cover:
- Backend initialization and configuration
- API key handling (config and environment)
- Request building and response parsing
- URL cleaning (tracking parameter removal)
- Network error handling
- Session management and cleanup
- SearchBackend ABC compliance
- Edge cases (empty results, malformed responses)

Mock API Structure Validation:
Mock response structures validated against OpenAI Responses API documentation
(Microsoft Learn: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/responses)
Verified on: 2025-10-30

Response structure:
- output: array of objects with type, content, etc.
- content: array with type="output_text", text, annotations
- annotations: array with type="url_citation", url, title
- web_search_call: object with action.sources array

Alternative formats supported:
- outputs (plural) or output (singular) as response key
- Direct list format for outputs array
"""

import asyncio
import pytest
import os
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from interaction_finder.search.backends.openai import OpenAIBackend
from interaction_finder.search.models import SearchQuery, SearchResult


@pytest.fixture
def mock_env_api_key(monkeypatch):
    """Mock OPENAI_API_KEY environment variable."""
    monkeypatch.setenv("OPENAI_API_KEY", "test_key_from_env")


@pytest.fixture
def openai_backend():
    """Create an OpenAIBackend instance with test configuration."""
    config = {
        "api_key": "test_api_key",
        "model": "gpt-4o-mini",
        "timeout": 30,
    }
    return OpenAIBackend(config)


class TestOpenAIBackendInitialization:
    """Test OpenAIBackend initialization and configuration."""

    def test_backend_creation_with_config_key(self):
        """Test backend creation with API key in config."""
        config = {"api_key": "test_key"}
        backend = OpenAIBackend(config)

        assert backend.name == "openai"
        assert backend.api_key == "test_key"
        assert backend.base_url == "https://api.openai.com/v1"
        assert backend.model == "gpt-4o-mini"
        assert backend.timeout == 60
        assert backend._session is None

    def test_backend_creation_with_env_key(self, mock_env_api_key):
        """Test backend creation with API key from environment."""
        backend = OpenAIBackend({})

        assert backend.api_key == "test_key_from_env"

    def test_backend_creation_no_api_key(self, monkeypatch):
        """Test backend creation fails without API key."""
        # Clear environment variable
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(ValueError, match="OpenAI API key is required"):
            OpenAIBackend({})

    def test_backend_creation_with_custom_config(self):
        """Test backend creation with custom configuration."""
        config = {
            "api_key": "custom_key",
            "base_url": "https://custom.openai.com",
            "model": "gpt-4o",
            "timeout": 120,
        }
        backend = OpenAIBackend(config)

        assert backend.api_key == "custom_key"
        assert backend.base_url == "https://custom.openai.com"
        assert backend.model == "gpt-4o"
        assert backend.timeout == 120

    def test_backend_name_property(self, openai_backend):
        """Test that name property returns correct identifier."""
        assert openai_backend.name == "openai"

    def test_backend_httpx_not_available(self):
        """Test error when httpx is not installed."""
        with patch("interaction_finder.search.backends.openai.HTTPX_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="httpx is required"):
                OpenAIBackend({"api_key": "test"})


class TestSessionManagement:
    """Test HTTP session management and cleanup."""

    @pytest.mark.asyncio
    async def test_get_session_creates_session(self, openai_backend):
        """Test that _get_session creates a new session if none exists."""
        with patch(
            "interaction_finder.search.backends.openai.httpx.AsyncClient"
        ) as mock_client:
            mock_instance = AsyncMock()
            mock_instance.is_closed = False
            mock_client.return_value = mock_instance

            session = await openai_backend._get_session()

            assert session is mock_instance
            assert openai_backend._session is mock_instance
            # Verify headers were set correctly
            call_kwargs = mock_client.call_args[1]
            assert "headers" in call_kwargs
            assert call_kwargs["headers"]["Authorization"] == "Bearer test_api_key"
            assert call_kwargs["headers"]["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_get_session_reuses_existing(self, openai_backend):
        """Test that _get_session reuses existing open session."""
        existing_session = AsyncMock()
        existing_session.is_closed = False
        openai_backend._session = existing_session

        with patch(
            "interaction_finder.search.backends.openai.httpx.AsyncClient"
        ) as mock_client:
            session = await openai_backend._get_session()

            assert session is existing_session
            mock_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_session_recreates_if_closed(self, openai_backend):
        """Test that _get_session creates new session if existing is closed."""
        closed_session = AsyncMock()
        closed_session.is_closed = True
        openai_backend._session = closed_session

        with patch(
            "interaction_finder.search.backends.openai.httpx.AsyncClient"
        ) as mock_client:
            mock_instance = AsyncMock()
            mock_instance.is_closed = False
            mock_client.return_value = mock_instance

            session = await openai_backend._get_session()

            assert session is mock_instance
            assert openai_backend._session is mock_instance

    @pytest.mark.asyncio
    async def test_close_closes_session(self, openai_backend):
        """Test that close() closes the HTTP session."""
        mock_session = AsyncMock()
        mock_session.is_closed = False
        openai_backend._session = mock_session

        await openai_backend.close()

        mock_session.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_handles_no_session(self, openai_backend):
        """Test that close() handles case when no session exists."""
        openai_backend._session = None

        # Should not raise
        await openai_backend.close()

    @pytest.mark.asyncio
    async def test_context_manager_entry(self, openai_backend):
        """Test async context manager entry."""
        result = await openai_backend.__aenter__()
        assert result is openai_backend

    @pytest.mark.asyncio
    async def test_context_manager_exit(self, openai_backend):
        """Test async context manager exit closes session."""
        mock_session = AsyncMock()
        mock_session.is_closed = False
        openai_backend._session = mock_session

        await openai_backend.__aexit__(None, None, None)

        mock_session.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_full_lifecycle(self, mock_env_api_key):
        """Test full async context manager lifecycle."""
        config = {"timeout": 30}

        async with OpenAIBackend(config) as backend:
            assert backend.name == "openai"


class TestRequestBuilding:
    """Test search request construction."""

    def test_build_search_request_minimal(self, openai_backend):
        """Test building search request with minimal query."""
        query = SearchQuery(query="test search", max_results=10)
        request = openai_backend._build_search_request(query)

        assert request["model"] == "gpt-4o-mini"
        assert request["tools"] == [{"type": "web_search"}]
        assert request["tool_choice"] == "required"
        assert "test search" in request["input"]
        assert "include" in request

    def test_build_search_request_custom_model(self):
        """Test building request with custom model."""
        config = {"api_key": "test", "model": "gpt-4o"}
        backend = OpenAIBackend(config)
        query = SearchQuery(query="test", max_results=5)

        request = backend._build_search_request(query)

        assert request["model"] == "gpt-4o"


class TestURLCleaning:
    """Test URL cleaning functionality."""

    def test_clean_url_removes_utm_params(self, openai_backend):
        """Test that UTM parameters are removed from URLs."""
        dirty_url = "https://example.com/page?utm_source=test&utm_medium=email&id=123"
        clean_url = openai_backend._clean_url(dirty_url)

        assert "utm_source" not in clean_url
        assert "utm_medium" not in clean_url
        assert "id=123" in clean_url

    def test_clean_url_removes_tracking_params(self, openai_backend):
        """Test that common tracking parameters are removed."""
        dirty_url = "https://example.com?fbclid=abc&gclid=xyz&normal=param"
        clean_url = openai_backend._clean_url(dirty_url)

        assert "fbclid" not in clean_url
        assert "gclid" not in clean_url
        assert "normal=param" in clean_url

    def test_clean_url_handles_no_params(self, openai_backend):
        """Test URL cleaning with no query parameters."""
        url = "https://example.com/page"
        clean_url = openai_backend._clean_url(url)

        assert clean_url == url

    def test_clean_url_handles_empty_url(self, openai_backend):
        """Test URL cleaning with empty URL."""
        assert openai_backend._clean_url("") == ""
        assert openai_backend._clean_url(None) == None


class TestResponseParsing:
    """Test OpenAI response parsing."""

    def test_parse_response_with_citations(self, openai_backend):
        """Test parsing response with URL citations."""
        query = SearchQuery(query="test", max_results=10)
        response_data = {
            "outputs": [
                {"type": "web_search_call", "action": {"sources": []}},
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url": "https://example.com/article1",
                                    "title": "Test Article 1",
                                },
                                {
                                    "type": "url_citation",
                                    "url": "https://example.com/article2",
                                    "title": "Test Article 2",
                                },
                            ],
                        }
                    ],
                },
            ]
        }

        results = openai_backend._parse_openai_response(response_data, query)

        assert len(results) == 2
        assert results[0].title == "Test Article 1"
        assert results[0].url == "https://example.com/article1"
        assert results[1].title == "Test Article 2"
        assert results[1].url == "https://example.com/article2"

    def test_parse_response_with_sources_fallback(self, openai_backend):
        """Test parsing falls back to sources when no citations."""
        query = SearchQuery(query="test", max_results=10)
        response_data = {
            "outputs": [
                {
                    "type": "web_search_call",
                    "action": {
                        "sources": [
                            {"url": "https://example.com/page1"},
                            "https://example.com/page2",
                        ]
                    },
                },
                {"type": "message", "content": []},
            ]
        }

        results = openai_backend._parse_openai_response(response_data, query)

        assert len(results) == 2
        assert results[0].url == "https://example.com/page1"
        assert results[1].url == "https://example.com/page2"
        # Titles should be domain names
        assert "example.com" in results[0].title

    def test_parse_response_empty(self, openai_backend):
        """Test parsing response with no results."""
        query = SearchQuery(query="test", max_results=10)
        response_data = {"outputs": []}

        results = openai_backend._parse_openai_response(response_data, query)

        assert results == []

    def test_parse_response_respects_max_results(self, openai_backend):
        """Test that parsing respects max_results limit."""
        query = SearchQuery(query="test", max_results=2)
        response_data = {
            "outputs": [
                {"type": "web_search_call", "action": {"sources": []}},
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url": f"https://example.com/page{i}",
                                    "title": f"Page {i}",
                                }
                                for i in range(10)
                            ],
                        }
                    ],
                },
            ]
        }

        results = openai_backend._parse_openai_response(response_data, query)

        assert len(results) <= 2

    def test_parse_response_deduplicates_urls(self, openai_backend):
        """Test that duplicate URLs are filtered out."""
        query = SearchQuery(query="test", max_results=10)
        response_data = {
            "outputs": [
                {"type": "web_search_call", "action": {"sources": []}},
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url": "https://example.com/page",
                                    "title": "Page 1",
                                },
                                {
                                    "type": "url_citation",
                                    "url": "https://example.com/page",
                                    "title": "Page 2",
                                },
                            ],
                        }
                    ],
                },
            ]
        }

        results = openai_backend._parse_openai_response(response_data, query)

        assert len(results) == 1

    def test_parse_response_list_format(self, openai_backend):
        """Test parsing when response_data is a list directly."""
        query = SearchQuery(query="test", max_results=10)
        # List format response - outputs provided as list directly
        response_data = [
            {"type": "web_search_call", "action": {"sources": []}},
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": "https://example.com/list-result",
                                "title": "List Format Result",
                            }
                        ],
                    }
                ],
            },
        ]

        results = openai_backend._parse_openai_response(response_data, query)

        assert len(results) == 1
        assert results[0].title == "List Format Result"
        assert results[0].url == "https://example.com/list-result"


class TestSearchMethod:
    """Test the main search() method."""

    @pytest.mark.asyncio
    async def test_search_success(self, openai_backend):
        """Test successful end-to-end search."""
        query = SearchQuery(query="test query", max_results=10)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [
                {"type": "web_search_call", "action": {"sources": []}},
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url": "https://example.com/result",
                                    "title": "Test Result",
                                }
                            ],
                        }
                    ],
                },
            ]
        }

        with patch.object(openai_backend, "_get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_session.post = AsyncMock(return_value=mock_response)
            mock_get_session.return_value = mock_session

            results = await openai_backend.search(query)

            assert len(results) == 1
            assert isinstance(results[0], SearchResult)
            assert results[0].title == "Test Result"

    @pytest.mark.asyncio
    async def test_search_authentication_error(self, openai_backend):
        """Test search handles 401 authentication errors."""
        query = SearchQuery(query="test", max_results=10)

        mock_response = Mock()
        mock_response.status_code = 401

        with patch.object(openai_backend, "_get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_session.post = AsyncMock(return_value=mock_response)
            mock_get_session.return_value = mock_session

            with pytest.raises(RuntimeError, match="authentication failed"):
                await openai_backend.search(query)

    @pytest.mark.asyncio
    async def test_search_rate_limit_error(self, openai_backend):
        """Test search handles 429 rate limit errors."""
        query = SearchQuery(query="test", max_results=10)

        mock_response = Mock()
        mock_response.status_code = 429

        with patch.object(openai_backend, "_get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_session.post = AsyncMock(return_value=mock_response)
            mock_get_session.return_value = mock_session

            with pytest.raises(RuntimeError, match="rate limit exceeded"):
                await openai_backend.search(query)

    @pytest.mark.asyncio
    async def test_search_timeout(self, openai_backend):
        """Test search handles timeout errors."""
        query = SearchQuery(query="test", max_results=10)

        with patch.object(openai_backend, "_get_session") as mock_get_session:
            import httpx

            mock_session = AsyncMock()
            mock_session.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_get_session.return_value = mock_session

            with pytest.raises(RuntimeError, match="timed out"):
                await openai_backend.search(query)

    @pytest.mark.asyncio
    async def test_search_network_error(self, openai_backend):
        """Test search handles network errors."""
        query = SearchQuery(query="test", max_results=10)

        with patch.object(openai_backend, "_get_session") as mock_get_session:
            import httpx

            mock_session = AsyncMock()
            mock_session.post = AsyncMock(
                side_effect=httpx.RequestError("network error")
            )
            mock_get_session.return_value = mock_session

            with pytest.raises(RuntimeError, match="network error"):
                await openai_backend.search(query)

    @pytest.mark.asyncio
    async def test_search_generic_http_error(self, openai_backend):
        """Test search handles generic HTTP errors (502, 503, etc)."""
        query = SearchQuery(query="test", max_results=10)

        mock_response = Mock()
        mock_response.status_code = 502
        mock_response.text = "Bad Gateway"

        with patch.object(openai_backend, "_get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_session.post = AsyncMock(return_value=mock_response)
            mock_get_session.return_value = mock_session

            with pytest.raises(RuntimeError, match="HTTP 502"):
                await openai_backend.search(query)

    @pytest.mark.asyncio
    async def test_search_wraps_unexpected_errors(self, openai_backend):
        """Test that unexpected errors are wrapped in RuntimeError."""
        query = SearchQuery(query="test", max_results=10)

        with patch.object(openai_backend, "_get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_session.post = AsyncMock(side_effect=ValueError("Unexpected error"))
            mock_get_session.return_value = mock_session

            with pytest.raises(RuntimeError, match="Unexpected error"):
                await openai_backend.search(query)


class TestHealthCheck:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_async_health_check_success(self, openai_backend):
        """Test async health check succeeds."""
        mock_response = Mock()
        mock_response.status_code = 200

        with patch.object(openai_backend, "_get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_session.post = AsyncMock(return_value=mock_response)
            mock_get_session.return_value = mock_session

            result = await openai_backend._async_health_check()

            assert result is True

    @pytest.mark.asyncio
    async def test_async_health_check_accepts_400(self, openai_backend):
        """Test async health check accepts 400 as healthy."""
        mock_response = Mock()
        mock_response.status_code = 400

        with patch.object(openai_backend, "_get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_session.post = AsyncMock(return_value=mock_response)
            mock_get_session.return_value = mock_session

            result = await openai_backend._async_health_check()

            assert result is True

    @pytest.mark.asyncio
    async def test_async_health_check_failure(self, openai_backend):
        """Test async health check fails on error."""
        with patch.object(openai_backend, "_get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_session.post = AsyncMock(side_effect=Exception("API down"))
            mock_get_session.return_value = mock_session

            result = await openai_backend._async_health_check()

            assert result is False

    def test_healthy_check_running_loop(self, openai_backend):
        """Test healthy() when event loop is already running."""
        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = Mock()
            mock_loop.is_running.return_value = True
            mock_get_loop.return_value = mock_loop

            # Should return True without calling async health check
            result = openai_backend.healthy()

            assert result is True


class TestABCCompliance:
    """Test SearchBackend ABC compliance."""

    def test_implements_name_property(self, openai_backend):
        """Test that name property is implemented."""
        assert hasattr(openai_backend, "name")
        assert isinstance(openai_backend.name, str)

    def test_implements_search_method(self, openai_backend):
        """Test that search method is implemented."""
        assert hasattr(openai_backend, "search")
        assert callable(openai_backend.search)

    def test_implements_healthy_method(self, openai_backend):
        """Test that healthy method is implemented."""
        assert hasattr(openai_backend, "healthy")
        assert callable(openai_backend.healthy)

    def test_inherits_from_searchbackend(self, openai_backend):
        """Test that OpenAIBackend inherits from SearchBackend."""
        from interaction_finder.search.models import SearchBackend

        assert isinstance(openai_backend, SearchBackend)
