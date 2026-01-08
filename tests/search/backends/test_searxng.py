"""Tests for SearXNG search backend."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from interaction_finder.search.backends.searxng import SearXNGBackend
from interaction_finder.search.models import SearchQuery


@pytest.fixture
def searxng_backend():
    """Create a SearXNGBackend with default config."""
    return SearXNGBackend(
        {
            "base_url": "http://localhost:8080",
            "timeout": 30,
            "categories": ["general"],
        }
    )


class TestSearXNGBackendInitialization:
    """Test SearXNGBackend initialization and configuration."""

    def test_backend_creation_basic(self):
        """Test basic backend creation with default config."""
        backend = SearXNGBackend()
        assert backend.name == "searxng"
        assert backend.base_url == "http://127.0.0.1:8080"
        assert backend.timeout == 30
        assert backend.categories == ["general"]
        assert backend.engines is None
        assert backend.language == "en"

    def test_backend_creation_with_config(self, searxng_backend):
        """Test backend creation with custom configuration."""
        assert searxng_backend.base_url == "http://localhost:8080"
        assert searxng_backend.timeout == 30
        assert searxng_backend.categories == ["general"]

    def test_backend_creation_custom_categories(self):
        """Test backend creation with custom categories."""
        backend = SearXNGBackend({"categories": ["science", "news"]})
        assert backend.categories == ["science", "news"]

    def test_backend_creation_with_engines(self):
        """Test backend creation with specific engines."""
        backend = SearXNGBackend({"engines": ["google", "bing"]})
        assert backend.engines == ["google", "bing"]

    def test_backend_creation_custom_language(self):
        """Test backend creation with custom language."""
        backend = SearXNGBackend({"language": "de"})
        assert backend.language == "de"

    def test_backend_name_property(self, searxng_backend):
        """Test that name property returns correct identifier."""
        assert searxng_backend.name == "searxng"


class TestRequestBuilding:
    """Test search request construction."""

    def test_build_search_params_basic(self, searxng_backend):
        """Test building search params with basic query."""
        query = SearchQuery(query="test search", max_results=10)
        params = searxng_backend._build_search_params(query)
        assert params["q"] == "test search"
        assert params["format"] == "json"
        assert params["language"] == "en"
        assert params["categories"] == "general"

    def test_build_search_params_multiple_categories(self):
        """Test building search params with multiple categories."""
        backend = SearXNGBackend({"categories": ["science", "news", "it"]})
        query = SearchQuery(query="test", max_results=5)
        params = backend._build_search_params(query)
        assert params["categories"] == "science,news,it"

    def test_build_search_params_with_engines(self):
        """Test building search params with specific engines."""
        backend = SearXNGBackend({"engines": ["google", "bing"]})
        query = SearchQuery(query="test", max_results=5)
        params = backend._build_search_params(query)
        assert params["engines"] == "google,bing"

    def test_build_search_params_no_engines(self, searxng_backend):
        """Test that engines param is omitted when not specified."""
        query = SearchQuery(query="test", max_results=5)
        params = searxng_backend._build_search_params(query)
        assert "engines" not in params


class TestResponseParsing:
    """Test SearXNG response parsing."""

    def test_parse_response_basic(self, searxng_backend):
        """Test parsing basic SearXNG response."""
        response_data = {
            "results": [
                {
                    "title": "Test Result 1",
                    "url": "https://example.com/1",
                    "content": "This is a snippet",
                    "score": 4.0,
                },
                {
                    "title": "Test Result 2",
                    "url": "https://example.com/2",
                    "content": "Another snippet",
                    "score": 2.0,
                },
            ]
        }
        query = SearchQuery(query="test", max_results=10)
        results = searxng_backend._parse_response(response_data, query)
        assert len(results) == 2
        assert results[0].title == "Test Result 1"
        assert results[0].url == "https://example.com/1"
        assert results[0].snippet == "This is a snippet"
        # Scores normalized: 4.0/4.0=1.0, 2.0/4.0=0.5
        assert results[0].relevance == 1.0
        assert results[1].relevance == 0.5

    def test_parse_response_respects_max_results(self, searxng_backend):
        """Test that parsing respects max_results limit."""
        response_data = {
            "results": [
                {"title": f"Result {i}", "url": f"https://example.com/{i}"}
                for i in range(20)
            ]
        }
        query = SearchQuery(query="test", max_results=5)
        results = searxng_backend._parse_response(response_data, query)
        assert len(results) == 5

    def test_parse_response_handles_missing_content(self, searxng_backend):
        """Test parsing response with missing content field."""
        response_data = {
            "results": [{"title": "No Snippet", "url": "https://example.com/1"}]
        }
        query = SearchQuery(query="test", max_results=10)
        results = searxng_backend._parse_response(response_data, query)
        assert len(results) == 1
        assert results[0].snippet is None

    def test_parse_response_skips_results_without_url(self, searxng_backend):
        """Test that results without URL are skipped."""
        response_data = {
            "results": [
                {"title": "Has URL", "url": "https://example.com/1"},
                {"title": "No URL"},
                {"title": "Also Has URL", "url": "https://example.com/2"},
            ]
        }
        query = SearchQuery(query="test", max_results=10)
        results = searxng_backend._parse_response(response_data, query)
        assert len(results) == 2

    def test_parse_response_handles_empty_results(self, searxng_backend):
        """Test parsing empty results."""
        query = SearchQuery(query="test", max_results=10)
        results = searxng_backend._parse_response({"results": []}, query)
        assert len(results) == 0

    def test_parse_response_handles_missing_title(self, searxng_backend):
        """Test parsing response with missing title."""
        response_data = {
            "results": [{"url": "https://example.com/1", "content": "Some content"}]
        }
        query = SearchQuery(query="test", max_results=10)
        results = searxng_backend._parse_response(response_data, query)
        assert len(results) == 1
        assert results[0].title == "(No title)"

    def test_parse_response_raises_on_error_field(self, searxng_backend):
        """Test that error field in response raises RuntimeError."""
        response_data = {"error": "Engine timeout"}
        query = SearchQuery(query="test", max_results=10)
        with pytest.raises(RuntimeError, match="SearXNG error: Engine timeout"):
            searxng_backend._parse_response(response_data, query)


class TestSearchMethod:
    """Test the search method."""

    @pytest.mark.asyncio
    async def test_search_success(self, searxng_backend):
        """Test successful search execution."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "Test Result",
                    "url": "https://example.com/test",
                    "content": "Test content",
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_response)
        mock_session.is_closed = False
        with patch.object(searxng_backend, "_get_session", return_value=mock_session):
            query = SearchQuery(query="test query", max_results=10)
            results = await searxng_backend.search(query)
            assert len(results) == 1
            assert results[0].title == "Test Result"
            mock_session.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_timeout_error(self, searxng_backend):
        """Test search handles timeout error."""
        import httpx

        mock_session = AsyncMock()
        mock_session.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
        mock_session.is_closed = False
        with patch.object(searxng_backend, "_get_session", return_value=mock_session):
            query = SearchQuery(query="test", max_results=10)
            with pytest.raises(RuntimeError, match="timed out"):
                await searxng_backend.search(query)

    @pytest.mark.asyncio
    async def test_search_http_error(self, searxng_backend):
        """Test search handles HTTP error."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Server error", request=MagicMock(), response=mock_response
            )
        )
        mock_session.is_closed = False
        with patch.object(searxng_backend, "_get_session", return_value=mock_session):
            query = SearchQuery(query="test", max_results=10)
            with pytest.raises(RuntimeError, match="HTTP error"):
                await searxng_backend.search(query)

    @pytest.mark.asyncio
    async def test_search_connection_error(self, searxng_backend):
        """Test search handles connection error."""
        import httpx

        mock_session = AsyncMock()
        mock_session.get = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        mock_session.is_closed = False
        with patch.object(searxng_backend, "_get_session", return_value=mock_session):
            query = SearchQuery(query="test", max_results=10)
            with pytest.raises(RuntimeError, match="connection error"):
                await searxng_backend.search(query)


class TestSessionManagement:
    """Test HTTP session management."""

    @pytest.mark.asyncio
    async def test_get_session_creates_new(self, searxng_backend):
        """Test _get_session creates new session when none exists."""
        assert searxng_backend._session is None
        session = await searxng_backend._get_session()
        assert session is not None
        assert searxng_backend._session is session
        await searxng_backend.close()

    @pytest.mark.asyncio
    async def test_get_session_reuses_existing(self, searxng_backend):
        """Test _get_session reuses existing session."""
        session1 = await searxng_backend._get_session()
        session2 = await searxng_backend._get_session()
        assert session1 is session2
        await searxng_backend.close()

    @pytest.mark.asyncio
    async def test_close_session(self, searxng_backend):
        """Test close() properly closes session."""
        await searxng_backend._get_session()
        assert searxng_backend._session is not None
        await searxng_backend.close()
        # Session object remains but is closed
        assert searxng_backend._session.is_closed


class TestABCCompliance:
    """Test SearchBackend ABC compliance."""

    def test_implements_name_property(self, searxng_backend):
        """Test name property is implemented."""
        assert hasattr(searxng_backend, "name")
        assert isinstance(searxng_backend.name, str)

    def test_implements_search_method(self, searxng_backend):
        """Test search method is implemented."""
        assert hasattr(searxng_backend, "search")
        assert callable(searxng_backend.search)

    def test_implements_healthy_method(self, searxng_backend):
        """Test healthy method is implemented."""
        assert hasattr(searxng_backend, "healthy")
        assert callable(searxng_backend.healthy)

    def test_inherits_from_searchbackend(self, searxng_backend):
        """Test backend inherits from SearchBackend."""
        from interaction_finder.search.models import SearchBackend

        assert isinstance(searxng_backend, SearchBackend)
