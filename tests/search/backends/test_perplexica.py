"""
Comprehensive test suite for Perplexica search backend.

Tests cover:
- Backend initialization and configuration
- Request building and response parsing
- Network error handling
- Session management and cleanup
- SearchBackend ABC compliance
- Edge cases (empty results, malformed responses, missing fields)

Mock API Structure Validation:
Mock response structures validated against Perplexica API documentation
(GitHub: https://github.com/ItzCrazyKns/Perplexica/blob/master/docs/API/SEARCH.md)
Verified on: 2025-10-30

Response structure:
- message: string (search result text)
- sources: array of objects with:
  - pageContent: string (content snippet)
  - metadata: object with title (string) and url (string)
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from interaction_finder.search.backends.perplexica import PerplexicaBackend
from interaction_finder.search.models import SearchQuery, SearchResult


@pytest.fixture
def perplexica_backend():
    """Create a PerplexicaBackend instance with test configuration.

    Pre-initializes models to avoid needing to mock /api/providers in every test.
    """
    config = {
        "base_url": "http://localhost:3000",
        "timeout": 30,
        "sources": ["web"],
        "chat_model": {"providerId": "test-provider-uuid", "key": "test-chat-model"},
        "embedding_model": {
            "providerId": "test-provider-uuid",
            "key": "test-embed-model",
        },
    }
    return PerplexicaBackend(config)


@pytest.fixture
def perplexica_backend_no_models():
    """Create a PerplexicaBackend without pre-configured models (for auto-discovery tests)."""
    config = {
        "base_url": "http://localhost:3000",
        "timeout": 30,
        "sources": ["web"],
    }
    return PerplexicaBackend(config)


class TestPerplexicaBackendInitialization:
    """Test PerplexicaBackend initialization and configuration."""

    def test_backend_creation_basic(self):
        """Test basic backend creation with default config."""
        backend = PerplexicaBackend()

        assert backend.name == "perplexica"
        assert backend.base_url == "http://localhost:3000"
        assert backend.timeout == 60
        assert backend.sources == ["web"]
        assert backend.optimization_mode == "balanced"
        assert backend._session is None

    def test_backend_creation_with_config(self, perplexica_backend):
        """Test backend creation with custom configuration."""
        assert perplexica_backend.base_url == "http://localhost:3000"
        assert perplexica_backend.timeout == 30
        assert perplexica_backend.sources == ["web"]
        assert perplexica_backend.optimization_mode == "balanced"

    def test_backend_creation_custom_models(self):
        """Test backend creation with custom model configuration."""
        config = {
            "chat_model": {"providerId": "anthropic-uuid-123", "key": "claude-3"},
            "embedding_model": {"providerId": "cohere-uuid-456", "key": "embed-v3"},
        }
        backend = PerplexicaBackend(config)
        assert backend._chat_model["providerId"] == "anthropic-uuid-123"
        assert backend._embedding_model["providerId"] == "cohere-uuid-456"
        assert backend._models_initialized is True

    def test_backend_creation_invalid_source(self):
        """Test that invalid source type raises error."""
        config = {"sources": ["invalid_source"]}
        with pytest.raises(ValueError, match="Invalid source"):
            PerplexicaBackend(config)

    def test_backend_creation_multiple_sources(self):
        """Test backend creation with multiple sources."""
        config = {"sources": ["web", "academic"]}
        backend = PerplexicaBackend(config)
        assert backend.sources == ["web", "academic"]

    def test_backend_creation_custom_optimization_mode(self):
        """Test backend creation with custom optimization mode."""
        config = {"optimization_mode": "speed"}
        backend = PerplexicaBackend(config)
        assert backend.optimization_mode == "speed"

    def test_backend_name_property(self, perplexica_backend):
        """Test that name property returns correct identifier."""
        assert perplexica_backend.name == "perplexica"

    def test_backend_httpx_not_available(self):
        """Test error when httpx is not installed."""
        with patch(
            "interaction_finder.search.backends.perplexica.HTTPX_AVAILABLE", False
        ):
            with pytest.raises(RuntimeError, match="httpx is required"):
                PerplexicaBackend()


class TestSessionManagement:
    """Test HTTP session management and cleanup."""

    @pytest.mark.asyncio
    async def test_get_session_creates_session(self, perplexica_backend):
        """Test that _get_session creates a new session if none exists."""
        with patch(
            "interaction_finder.search.backends.perplexica.httpx.AsyncClient"
        ) as mock_client:
            mock_instance = AsyncMock()
            mock_instance.is_closed = False
            mock_client.return_value = mock_instance

            session = await perplexica_backend._get_session()

            assert session is mock_instance
            assert perplexica_backend._session is mock_instance
            mock_client.assert_called_once_with(timeout=30)

    @pytest.mark.asyncio
    async def test_get_session_reuses_existing(self, perplexica_backend):
        """Test that _get_session reuses existing open session."""
        existing_session = AsyncMock()
        existing_session.is_closed = False
        perplexica_backend._session = existing_session

        with patch(
            "interaction_finder.search.backends.perplexica.httpx.AsyncClient"
        ) as mock_client:
            session = await perplexica_backend._get_session()

            assert session is existing_session
            mock_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_session_recreates_if_closed(self, perplexica_backend):
        """Test that _get_session creates new session if existing is closed."""
        closed_session = AsyncMock()
        closed_session.is_closed = True
        perplexica_backend._session = closed_session

        with patch(
            "interaction_finder.search.backends.perplexica.httpx.AsyncClient"
        ) as mock_client:
            mock_instance = AsyncMock()
            mock_instance.is_closed = False
            mock_client.return_value = mock_instance

            session = await perplexica_backend._get_session()

            assert session is mock_instance
            assert perplexica_backend._session is mock_instance

    @pytest.mark.asyncio
    async def test_close_closes_session(self, perplexica_backend):
        """Test that close() closes the HTTP session."""
        mock_session = AsyncMock()
        mock_session.is_closed = False
        perplexica_backend._session = mock_session

        await perplexica_backend.close()

        mock_session.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_handles_no_session(self, perplexica_backend):
        """Test that close() handles case when no session exists."""
        perplexica_backend._session = None

        # Should not raise
        await perplexica_backend.close()

    @pytest.mark.asyncio
    async def test_context_manager_entry(self, perplexica_backend):
        """Test async context manager entry."""
        result = await perplexica_backend.__aenter__()
        assert result is perplexica_backend

    @pytest.mark.asyncio
    async def test_context_manager_exit(self, perplexica_backend):
        """Test async context manager exit closes session."""
        mock_session = AsyncMock()
        mock_session.is_closed = False
        perplexica_backend._session = mock_session

        await perplexica_backend.__aexit__(None, None, None)

        mock_session.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_full_lifecycle(self):
        """Test full async context manager lifecycle."""
        config = {"timeout": 30}

        async with PerplexicaBackend(config) as backend:
            assert backend.name == "perplexica"


class TestRequestBuilding:
    """Test search request construction."""

    def test_build_search_request_web_sources(self, perplexica_backend):
        """Test building search request for web search with pre-configured models."""
        query = SearchQuery(query="test search", max_results=10)
        request = perplexica_backend._build_search_request(query)
        assert request["query"] == "test search"
        assert request["sources"] == ["web"]
        assert request["stream"] is False
        assert request["history"] == []
        assert request["optimizationMode"] == "balanced"
        # Pre-configured models from fixture
        assert request["chatModel"]["providerId"] == "test-provider-uuid"
        assert request["embeddingModel"]["providerId"] == "test-provider-uuid"

    def test_build_search_request_speed_mode(self):
        """Test building search request with speed optimization mode."""
        config = {"optimization_mode": "speed"}
        backend = PerplexicaBackend(config)
        query = SearchQuery(query="test", max_results=5)
        request = backend._build_search_request(query)
        assert request["optimizationMode"] == "speed"

    def test_build_search_request_no_models(self, perplexica_backend_no_models):
        """Test building search request before auto-discovery (models None)."""
        query = SearchQuery(query="test search", max_results=10)
        request = perplexica_backend_no_models._build_search_request(query)
        assert request["query"] == "test search"
        # Models are None before _fetch_default_models() is called
        assert request["chatModel"] is None
        assert request["embeddingModel"] is None

    def test_build_search_request_academic_sources(self):
        """Test building search request for academic search."""
        config = {"sources": ["academic"]}
        backend = PerplexicaBackend(config)
        query = SearchQuery(query="research paper", max_results=10)
        request = backend._build_search_request(query)
        assert request["sources"] == ["academic"]
        assert request["history"] == []

    def test_build_search_request_multiple_sources(self):
        """Test building search request with multiple sources."""
        config = {"sources": ["web", "academic", "discussions"]}
        backend = PerplexicaBackend(config)
        query = SearchQuery(query="test", max_results=5)
        request = backend._build_search_request(query)
        assert request["sources"] == ["web", "academic", "discussions"]

    def test_build_search_request_custom_models(self):
        """Test building request with custom model configuration."""
        config = {
            "chat_model": {"providerId": "anthropic-uuid", "key": "custom-model"},
            "embedding_model": {"providerId": "cohere-uuid", "key": "custom-embed"},
        }
        backend = PerplexicaBackend(config)
        query = SearchQuery(query="test", max_results=5)
        request = backend._build_search_request(query)
        assert request["chatModel"]["providerId"] == "anthropic-uuid"
        assert request["chatModel"]["key"] == "custom-model"
        assert request["embeddingModel"]["providerId"] == "cohere-uuid"
        assert request["embeddingModel"]["key"] == "custom-embed"


class TestResponseParsing:
    """Test Perplexica response parsing."""

    def test_parse_response_success(self, perplexica_backend):
        """Test parsing successful response with sources."""
        query = SearchQuery(query="test", max_results=10)
        response_data = {
            "sources": [
                {
                    "metadata": {
                        "title": "Article 1",
                        "url": "https://example.com/article1",
                    },
                    "pageContent": "This is the content of article 1",
                },
                {
                    "metadata": {
                        "title": "Article 2",
                        "url": "https://example.com/article2",
                    },
                    "pageContent": "This is the content of article 2",
                },
            ]
        }

        results = perplexica_backend._parse_perplexica_response(response_data, query)

        assert len(results) == 2
        assert results[0].title == "Article 1"
        assert results[0].url == "https://example.com/article1"
        assert results[0].snippet == "This is the content of article 1"
        assert results[1].title == "Article 2"

    def test_parse_response_empty(self, perplexica_backend):
        """Test parsing response with no sources."""
        query = SearchQuery(query="test", max_results=10)
        response_data = {"sources": []}

        results = perplexica_backend._parse_perplexica_response(response_data, query)

        assert results == []

    def test_parse_response_respects_max_results(self, perplexica_backend):
        """Test that parsing respects max_results limit."""
        query = SearchQuery(query="test", max_results=2)
        response_data = {
            "sources": [
                {
                    "metadata": {
                        "title": f"Article {i}",
                        "url": f"https://example.com/article{i}",
                    },
                    "pageContent": f"Content {i}",
                }
                for i in range(10)
            ]
        }

        results = perplexica_backend._parse_perplexica_response(response_data, query)

        assert len(results) == 2

    def test_parse_response_handles_missing_page_content(self, perplexica_backend):
        """Test parsing handles sources without page content."""
        query = SearchQuery(query="test", max_results=10)
        response_data = {
            "sources": [
                {
                    "metadata": {
                        "title": "Article",
                        "url": "https://example.com/article",
                    },
                    "pageContent": "",
                }
            ]
        }

        results = perplexica_backend._parse_perplexica_response(response_data, query)

        assert len(results) == 1
        assert results[0].snippet is None

    def test_convert_source_missing_title(self, perplexica_backend):
        """Test converting source without title returns None."""
        source = {"metadata": {"url": "https://example.com"}, "pageContent": "content"}

        result = perplexica_backend._convert_source_to_result(source, 0)

        assert result is None

    def test_convert_source_missing_url(self, perplexica_backend):
        """Test converting source without URL returns None."""
        source = {"metadata": {"title": "Test"}, "pageContent": "content"}

        result = perplexica_backend._convert_source_to_result(source, 0)

        assert result is None

    def test_convert_source_calculates_relevance(self, perplexica_backend):
        """Test relevance score calculation based on position."""
        source = {
            "metadata": {"title": "Test", "url": "https://example.com"},
            "pageContent": "content",
        }

        result1 = perplexica_backend._convert_source_to_result(source, 0)
        result2 = perplexica_backend._convert_source_to_result(source, 5)

        assert result1.relevance > result2.relevance
        assert result1.relevance <= 1.0
        assert result2.relevance >= 0.1


class TestSearchMethod:
    """Test the main search() method."""

    @pytest.mark.asyncio
    async def test_search_success(self, perplexica_backend):
        """Test successful end-to-end search."""
        query = SearchQuery(query="test query", max_results=10)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "sources": [
                {
                    "metadata": {
                        "title": "Test Result",
                        "url": "https://example.com/result",
                    },
                    "pageContent": "Test content",
                }
            ]
        }

        with patch.object(perplexica_backend, "_get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_session.post = AsyncMock(return_value=mock_response)
            mock_get_session.return_value = mock_session

            results = await perplexica_backend.search(query)

            assert len(results) == 1
            assert isinstance(results[0], SearchResult)
            assert results[0].title == "Test Result"

    @pytest.mark.asyncio
    async def test_search_server_error(self, perplexica_backend):
        """Test search handles 500 server errors."""
        query = SearchQuery(query="test", max_results=10)

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch.object(perplexica_backend, "_get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_session.post = AsyncMock(return_value=mock_response)
            mock_get_session.return_value = mock_session

            with pytest.raises(RuntimeError, match="internal server error"):
                await perplexica_backend.search(query)

    @pytest.mark.asyncio
    async def test_search_http_error(self, perplexica_backend):
        """Test search handles HTTP error responses."""
        query = SearchQuery(query="test", max_results=10)

        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"

        with patch.object(perplexica_backend, "_get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_session.post = AsyncMock(return_value=mock_response)
            mock_get_session.return_value = mock_session

            with pytest.raises(RuntimeError, match="HTTP 404"):
                await perplexica_backend.search(query)

    @pytest.mark.asyncio
    async def test_search_timeout(self, perplexica_backend):
        """Test search handles timeout errors."""
        query = SearchQuery(query="test", max_results=10)

        with patch.object(perplexica_backend, "_get_session") as mock_get_session:
            import httpx

            mock_session = AsyncMock()
            mock_session.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_get_session.return_value = mock_session

            with pytest.raises(RuntimeError, match="timed out"):
                await perplexica_backend.search(query)

    @pytest.mark.asyncio
    async def test_search_connection_error(self, perplexica_backend):
        """Test search handles connection errors."""
        query = SearchQuery(query="test", max_results=10)

        with patch.object(perplexica_backend, "_get_session") as mock_get_session:
            import httpx

            mock_session = AsyncMock()
            mock_session.post = AsyncMock(
                side_effect=httpx.ConnectError("connection failed")
            )
            mock_get_session.return_value = mock_session

            with pytest.raises(RuntimeError, match="Could not connect"):
                await perplexica_backend.search(query)

    @pytest.mark.asyncio
    async def test_search_network_error(self, perplexica_backend):
        """Test search handles network errors."""
        query = SearchQuery(query="test", max_results=10)

        with patch.object(perplexica_backend, "_get_session") as mock_get_session:
            import httpx

            mock_session = AsyncMock()
            mock_session.post = AsyncMock(
                side_effect=httpx.RequestError("network error")
            )
            mock_get_session.return_value = mock_session

            with pytest.raises(RuntimeError, match="network error"):
                await perplexica_backend.search(query)

    @pytest.mark.asyncio
    async def test_search_wraps_unexpected_errors(self, perplexica_backend):
        """Test that unexpected errors are wrapped in RuntimeError."""
        query = SearchQuery(query="test", max_results=10)

        with patch.object(perplexica_backend, "_get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_session.post = AsyncMock(side_effect=ValueError("Unexpected error"))
            mock_get_session.return_value = mock_session

            with pytest.raises(RuntimeError, match="Unexpected error"):
                await perplexica_backend.search(query)

    @pytest.mark.asyncio
    async def test_search_handles_conversion_errors(self, perplexica_backend):
        """Test that conversion errors for individual results don't fail entire search."""
        query = SearchQuery(query="test", max_results=10)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "sources": [
                {
                    "metadata": {"title": "Good", "url": "https://example.com/good"},
                    "pageContent": "content",
                },
                {
                    "metadata": {"url": "https://example.com/bad"},  # Missing title
                    "pageContent": "content",
                },
            ]
        }

        with patch.object(perplexica_backend, "_get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_session.post = AsyncMock(return_value=mock_response)
            mock_get_session.return_value = mock_session

            results = await perplexica_backend.search(query)

            # Should get only the valid result
            assert len(results) == 1
            assert results[0].title == "Good"


class TestHealthCheck:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_async_health_check_success(self, perplexica_backend):
        """Test async health check succeeds."""
        mock_response = AsyncMock()
        mock_response.status_code = 200

        with patch.object(perplexica_backend, "_get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_session.get = AsyncMock(return_value=mock_response)
            mock_get_session.return_value = mock_session

            result = await perplexica_backend._async_health_check()

            assert result is True

    @pytest.mark.asyncio
    async def test_async_health_check_failure(self, perplexica_backend):
        """Test async health check fails on error."""
        with patch.object(perplexica_backend, "_get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_session.get = AsyncMock(side_effect=Exception("API down"))
            mock_get_session.return_value = mock_session

            result = await perplexica_backend._async_health_check()

            assert result is False

    def test_healthy_check_running_loop(self, perplexica_backend):
        """Test healthy() when event loop is already running."""
        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = Mock()
            mock_loop.is_running.return_value = True
            mock_get_loop.return_value = mock_loop

            # Should return True without calling async health check
            result = perplexica_backend.healthy()

            assert result is True


class TestABCCompliance:
    """Test SearchBackend ABC compliance."""

    def test_implements_name_property(self, perplexica_backend):
        """Test that name property is implemented."""
        assert hasattr(perplexica_backend, "name")
        assert isinstance(perplexica_backend.name, str)

    def test_implements_search_method(self, perplexica_backend):
        """Test that search method is implemented."""
        assert hasattr(perplexica_backend, "search")
        assert callable(perplexica_backend.search)

    def test_implements_healthy_method(self, perplexica_backend):
        """Test that healthy method is implemented."""
        assert hasattr(perplexica_backend, "healthy")
        assert callable(perplexica_backend.healthy)

    def test_inherits_from_searchbackend(self, perplexica_backend):
        """Test that PerplexicaBackend inherits from SearchBackend."""
        from interaction_finder.search.models import SearchBackend

        assert isinstance(perplexica_backend, SearchBackend)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_search_with_unicode_query(self, perplexica_backend):
        """Test search handles unicode characters in query."""
        query = SearchQuery(query="α-synuclein β-amyloid", max_results=10)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"sources": []}

        with patch.object(perplexica_backend, "_get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_session.post = AsyncMock(return_value=mock_response)
            mock_get_session.return_value = mock_session

            results = await perplexica_backend.search(query)

            assert results == []
            # Verify query was passed correctly
            call_args = mock_session.post.call_args
            request_data = call_args[1]["json"]
            assert request_data["query"] == "α-synuclein β-amyloid"

    def test_parse_response_handles_missing_metadata(self, perplexica_backend):
        """Test parsing handles sources with missing metadata."""
        query = SearchQuery(query="test", max_results=10)
        response_data = {"sources": [{"pageContent": "content"}]}

        results = perplexica_backend._parse_perplexica_response(response_data, query)

        assert results == []

    def test_parse_response_handles_empty_strings(self, perplexica_backend):
        """Test parsing handles sources with empty strings."""
        query = SearchQuery(query="test", max_results=10)
        response_data = {
            "sources": [
                {"metadata": {"title": "", "url": ""}, "pageContent": "content"}
            ]
        }

        results = perplexica_backend._parse_perplexica_response(response_data, query)

        assert results == []


class TestModelAutoDiscovery:
    """Test automatic model discovery from /api/providers."""

    @pytest.mark.asyncio
    async def test_fetch_default_models_success(self, perplexica_backend_no_models):
        """Test successful auto-discovery of models from providers endpoint."""
        providers_response = {
            "providers": [
                {
                    "id": "openai-uuid-123",
                    "chatModels": [{"name": "GPT-4", "key": "gpt-4"}],
                    "embeddingModels": [
                        {"name": "Embedding", "key": "text-embedding-3-large"}
                    ],
                }
            ]
        }
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = providers_response
        with patch.object(
            perplexica_backend_no_models, "_get_session"
        ) as mock_get_session:
            mock_session = AsyncMock()
            mock_session.get = AsyncMock(return_value=mock_response)
            mock_get_session.return_value = mock_session
            await perplexica_backend_no_models._fetch_default_models()
        assert perplexica_backend_no_models._chat_model == {
            "providerId": "openai-uuid-123",
            "key": "gpt-4",
        }
        assert perplexica_backend_no_models._embedding_model == {
            "providerId": "openai-uuid-123",
            "key": "text-embedding-3-large",
        }
        assert perplexica_backend_no_models._models_initialized is True

    @pytest.mark.asyncio
    async def test_fetch_default_models_different_providers(
        self, perplexica_backend_no_models
    ):
        """Test auto-discovery when chat and embedding are from different providers."""
        providers_response = {
            "providers": [
                {
                    "id": "openai-uuid",
                    "chatModels": [{"name": "GPT-4", "key": "gpt-4"}],
                    "embeddingModels": [],
                },
                {
                    "id": "transformers-uuid",
                    "chatModels": [],
                    "embeddingModels": [{"name": "MiniLM", "key": "all-MiniLM-L6-v2"}],
                },
            ]
        }
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = providers_response
        with patch.object(
            perplexica_backend_no_models, "_get_session"
        ) as mock_get_session:
            mock_session = AsyncMock()
            mock_session.get = AsyncMock(return_value=mock_response)
            mock_get_session.return_value = mock_session
            await perplexica_backend_no_models._fetch_default_models()
        assert perplexica_backend_no_models._chat_model["providerId"] == "openai-uuid"
        assert (
            perplexica_backend_no_models._embedding_model["providerId"]
            == "transformers-uuid"
        )

    @pytest.mark.asyncio
    async def test_fetch_default_models_no_chat_models(
        self, perplexica_backend_no_models
    ):
        """Test error when no chat models available."""
        providers_response = {
            "providers": [
                {
                    "id": "transformers-uuid",
                    "chatModels": [],
                    "embeddingModels": [{"name": "MiniLM", "key": "all-MiniLM-L6-v2"}],
                }
            ]
        }
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = providers_response
        with patch.object(
            perplexica_backend_no_models, "_get_session"
        ) as mock_get_session:
            mock_session = AsyncMock()
            mock_session.get = AsyncMock(return_value=mock_response)
            mock_get_session.return_value = mock_session
            with pytest.raises(RuntimeError, match="No chat models available"):
                await perplexica_backend_no_models._fetch_default_models()

    @pytest.mark.asyncio
    async def test_fetch_default_models_no_embedding_models(
        self, perplexica_backend_no_models
    ):
        """Test error when no embedding models available."""
        providers_response = {
            "providers": [
                {
                    "id": "openai-uuid",
                    "chatModels": [{"name": "GPT-4", "key": "gpt-4"}],
                    "embeddingModels": [],
                }
            ]
        }
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = providers_response
        with patch.object(
            perplexica_backend_no_models, "_get_session"
        ) as mock_get_session:
            mock_session = AsyncMock()
            mock_session.get = AsyncMock(return_value=mock_response)
            mock_get_session.return_value = mock_session
            with pytest.raises(RuntimeError, match="No embedding models available"):
                await perplexica_backend_no_models._fetch_default_models()

    @pytest.mark.asyncio
    async def test_fetch_default_models_skips_if_initialized(self, perplexica_backend):
        """Test that fetch is skipped if models already initialized."""
        # perplexica_backend fixture has models pre-configured
        with patch.object(perplexica_backend, "_get_session") as mock_get_session:
            await perplexica_backend._fetch_default_models()
            # Should not call _get_session since models are already set
            mock_get_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_fetch_default_models_prefers_gpt4o_mini(
        self, perplexica_backend_no_models
    ):
        """Test that gpt-4o-mini is preferred over other models."""
        providers_response = {
            "providers": [
                {
                    "id": "openai-uuid",
                    "chatModels": [
                        {"name": "GPT-3.5", "key": "gpt-3.5-turbo"},
                        {"name": "GPT-4o mini", "key": "gpt-4o-mini"},
                        {"name": "GPT-4", "key": "gpt-4"},
                    ],
                    "embeddingModels": [
                        {"name": "Small", "key": "text-embedding-3-small"},
                        {"name": "Large", "key": "text-embedding-3-large"},
                    ],
                }
            ]
        }
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = providers_response
        with patch.object(
            perplexica_backend_no_models, "_get_session"
        ) as mock_get_session:
            mock_session = AsyncMock()
            mock_session.get = AsyncMock(return_value=mock_response)
            mock_get_session.return_value = mock_session
            await perplexica_backend_no_models._fetch_default_models()
        # Should select gpt-4o-mini (preferred) not gpt-3.5-turbo (first)
        assert perplexica_backend_no_models._chat_model["key"] == "gpt-4o-mini"
        # Should select text-embedding-3-large (preferred) not small (first)
        assert (
            perplexica_backend_no_models._embedding_model["key"]
            == "text-embedding-3-large"
        )
