"""
Comprehensive test suite for PubMed search backend.

Tests cover:
- Backend initialization and configuration
- ESearch and ESummary API interactions
- XML response parsing (happy path and errors)
- Rate limiting enforcement
- Network error handling
- Session management and cleanup
- SearchBackend ABC compliance
- Edge cases (empty results, malformed XML, missing fields)
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from interaction_finder.search.backends.pubmed import PubMedBackend
from interaction_finder.search.models import SearchQuery, SearchResult


# XML fixtures for realistic PubMed responses
ESEARCH_SUCCESS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<eSearchResult>
    <Count>2</Count>
    <RetMax>2</RetMax>
    <RetStart>0</RetStart>
    <IdList>
        <Id>12345678</Id>
        <Id>87654321</Id>
    </IdList>
    <WebEnv>test_web_env</WebEnv>
    <QueryKey>1</QueryKey>
</eSearchResult>
"""

ESEARCH_ERROR_XML = """<?xml version="1.0" encoding="UTF-8"?>
<eSearchResult>
    <Count>0</Count>
    <ErrorList>
        <PhraseNotFound>invalid term</PhraseNotFound>
        <PhraseNotFound>another error</PhraseNotFound>
    </ErrorList>
</eSearchResult>
"""

ESEARCH_EMPTY_XML = """<?xml version="1.0" encoding="UTF-8"?>
<eSearchResult>
    <Count>0</Count>
    <RetMax>0</RetMax>
    <RetStart>0</RetStart>
    <IdList/>
</eSearchResult>
"""

ESUMMARY_SUCCESS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<eSummaryResult>
    <DocSum>
        <Id>12345678</Id>
        <Item Name="Title" Type="String">COVID-19 and inflammation</Item>
        <Item Name="Source" Type="String">Nature</Item>
        <Item Name="PubDate" Type="String">2023</Item>
        <Item Name="AuthorList" Type="List">
            <Item Type="String">Smith J</Item>
            <Item Type="String">Doe A</Item>
        </Item>
        <Item Name="Volume" Type="Integer">123</Item>
    </DocSum>
    <DocSum>
        <Id>87654321</Id>
        <Item Name="Title" Type="String">SARS-CoV-2 mechanisms</Item>
        <Item Name="Source" Type="String">Science</Item>
        <Item Name="PubDate" Type="String">2023</Item>
    </DocSum>
</eSummaryResult>
"""

ESUMMARY_MISSING_FIELDS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<eSummaryResult>
    <DocSum>
        <Id>11111111</Id>
    </DocSum>
</eSummaryResult>
"""


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx.AsyncClient for testing."""
    client = AsyncMock()
    client.is_closed = False
    return client


@pytest.fixture
def pubmed_backend():
    """Create a PubMedBackend instance with test configuration."""
    config = {
        "email": "test@example.com",
        "api_key": "test_key",
        "rate_limit": 10.0,  # High rate limit for fast tests
        "timeout": 30,
    }
    return PubMedBackend(config)


@pytest.fixture
def pubmed_backend_no_httpx():
    """Test backend initialization when httpx is not available."""
    with patch("interaction_finder.search.backends.pubmed.HTTPX_AVAILABLE", False):
        with pytest.raises(RuntimeError, match="httpx is required"):
            PubMedBackend()


class TestPubMedBackendInitialization:
    """Test PubMedBackend initialization and configuration."""

    def test_backend_creation_basic(self):
        """Test basic backend creation with default config."""
        backend = PubMedBackend()

        assert backend.name == "pubmed"
        assert backend.base_url == "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        assert backend.email is None
        assert backend.api_key is None
        assert backend.rate_limit == 3.0
        assert backend.timeout == 30
        assert backend.use_mesh is True
        assert backend._session is None

    def test_backend_creation_with_config(self, pubmed_backend):
        """Test backend creation with custom configuration."""
        assert pubmed_backend.email == "test@example.com"
        assert pubmed_backend.api_key == "test_key"
        assert pubmed_backend.rate_limit == 10.0
        assert pubmed_backend.timeout == 30

    def test_backend_name_property(self, pubmed_backend):
        """Test that name property returns correct identifier."""
        assert pubmed_backend.name == "pubmed"

    def test_backend_httpx_not_available(self):
        """Test error when httpx is not installed."""
        with patch("interaction_finder.search.backends.pubmed.HTTPX_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="httpx is required"):
                PubMedBackend()

    def test_api_key_warning_check_when_not_configured(self):
        """Test that warning check returns True when API key is missing."""
        backend = PubMedBackend()
        assert backend.should_show_api_key_warning() is True

    def test_api_key_warning_check_when_configured(self):
        """Test that warning check returns False when API key is present."""
        backend = PubMedBackend({"api_key": "test_key"})
        assert backend.should_show_api_key_warning() is False


class TestSessionManagement:
    """Test HTTP session management and cleanup."""

    @pytest.mark.asyncio
    async def test_get_session_creates_session(self, pubmed_backend):
        """Test that _get_session creates a new session if none exists."""
        with patch(
            "interaction_finder.search.backends.pubmed.httpx.AsyncClient"
        ) as mock_client:
            mock_instance = AsyncMock()
            mock_instance.is_closed = False
            mock_client.return_value = mock_instance

            session = await pubmed_backend._get_session()

            assert session is mock_instance
            assert pubmed_backend._session is mock_instance
            mock_client.assert_called_once_with(timeout=30)

    @pytest.mark.asyncio
    async def test_get_session_reuses_existing(self, pubmed_backend):
        """Test that _get_session reuses existing open session."""
        existing_session = AsyncMock()
        existing_session.is_closed = False
        pubmed_backend._session = existing_session

        with patch(
            "interaction_finder.search.backends.pubmed.httpx.AsyncClient"
        ) as mock_client:
            session = await pubmed_backend._get_session()

            assert session is existing_session
            mock_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_session_recreates_if_closed(self, pubmed_backend):
        """Test that _get_session creates new session if existing is closed."""
        closed_session = AsyncMock()
        closed_session.is_closed = True
        pubmed_backend._session = closed_session

        with patch(
            "interaction_finder.search.backends.pubmed.httpx.AsyncClient"
        ) as mock_client:
            mock_instance = AsyncMock()
            mock_instance.is_closed = False
            mock_client.return_value = mock_instance

            session = await pubmed_backend._get_session()

            assert session is mock_instance
            assert pubmed_backend._session is mock_instance

    @pytest.mark.asyncio
    async def test_close_closes_session(self, pubmed_backend):
        """Test that close() closes the HTTP session."""
        mock_session = AsyncMock()
        mock_session.is_closed = False
        pubmed_backend._session = mock_session

        await pubmed_backend.close()

        mock_session.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_handles_no_session(self, pubmed_backend):
        """Test that close() handles case when no session exists."""
        pubmed_backend._session = None

        # Should not raise
        await pubmed_backend.close()

    @pytest.mark.asyncio
    async def test_close_handles_closed_session(self, pubmed_backend):
        """Test that close() handles already closed session."""
        mock_session = AsyncMock()
        mock_session.is_closed = True
        pubmed_backend._session = mock_session

        await pubmed_backend.close()

        mock_session.aclose.assert_not_called()

    @pytest.mark.asyncio
    async def test_context_manager_entry(self, pubmed_backend):
        """Test async context manager entry."""
        result = await pubmed_backend.__aenter__()
        assert result is pubmed_backend

    @pytest.mark.asyncio
    async def test_context_manager_exit(self, pubmed_backend):
        """Test async context manager exit closes session."""
        mock_session = AsyncMock()
        mock_session.is_closed = False
        pubmed_backend._session = mock_session

        await pubmed_backend.__aexit__(None, None, None)

        mock_session.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_full_lifecycle(self):
        """Test full async context manager lifecycle."""
        config = {"rate_limit": 10.0}

        async with PubMedBackend(config) as backend:
            assert backend.name == "pubmed"
            # Session created on demand, not during __aenter__


class TestRateLimiting:
    """Test rate limiting enforcement."""

    @pytest.mark.asyncio
    async def test_rate_limit_enforced(self, pubmed_backend):
        """Test that rate limiting delays requests appropriately."""
        pubmed_backend.rate_limit = 2.0  # 2 requests per second

        # First request should be immediate
        await pubmed_backend._enforce_rate_limit()
        first_time = asyncio.get_event_loop().time()

        # Second request should be delayed
        await pubmed_backend._enforce_rate_limit()
        second_time = asyncio.get_event_loop().time()

        # Time between requests should be at least 1/rate_limit seconds
        min_interval = 1.0 / pubmed_backend.rate_limit
        actual_interval = second_time - first_time

        assert actual_interval >= min_interval * 0.9  # Allow 10% tolerance

    @pytest.mark.asyncio
    async def test_rate_limit_concurrent_requests(self, pubmed_backend):
        """Test that rate limiting works correctly with concurrent requests."""
        pubmed_backend.rate_limit = 5.0  # 5 requests per second

        # Launch multiple concurrent requests
        tasks = [pubmed_backend._enforce_rate_limit() for _ in range(3)]

        start_time = asyncio.get_event_loop().time()
        await asyncio.gather(*tasks)
        end_time = asyncio.get_event_loop().time()

        # Should take at least 2 intervals (3 requests = 0 + 1 + 2 intervals)
        min_time = 2 * (1.0 / pubmed_backend.rate_limit)
        actual_time = end_time - start_time

        assert actual_time >= min_time * 0.9  # Allow 10% tolerance


class TestRetryBehavior:
    """Test retry logic with exponential backoff."""

    @pytest.mark.asyncio
    async def test_esearch_succeeds_after_one_retry(self, pubmed_backend):
        """Test that ESearch succeeds on second attempt after 429."""
        query = SearchQuery(query="covid", max_results=10)

        # First response: 429, second response: success
        mock_429_response = AsyncMock()
        mock_429_response.status_code = 429

        mock_success_response = AsyncMock()
        mock_success_response.status_code = 200
        mock_success_response.text = ESEARCH_SUCCESS_XML

        with patch.object(pubmed_backend, "_get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_session.get = AsyncMock(
                side_effect=[mock_429_response, mock_success_response]
            )
            mock_get_session.return_value = mock_session

            # Should succeed after one retry
            result = await pubmed_backend._esearch(query)

            # Verify success
            assert result["count"] == 2
            assert len(result["pmids"]) == 2
            # Verify it made 2 attempts
            assert mock_session.get.call_count == 2

    @pytest.mark.asyncio
    async def test_esummary_succeeds_after_one_retry(self, pubmed_backend):
        """Test that ESummary succeeds on second attempt after 429."""
        pmids = ["12345678", "87654321"]

        # First response: 429, second response: success
        mock_429_response = AsyncMock()
        mock_429_response.status_code = 429

        mock_success_response = AsyncMock()
        mock_success_response.status_code = 200
        mock_success_response.text = ESUMMARY_SUCCESS_XML

        with patch.object(pubmed_backend, "_get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_session.get = AsyncMock(
                side_effect=[mock_429_response, mock_success_response]
            )
            mock_get_session.return_value = mock_session

            # Should succeed after one retry
            result = await pubmed_backend._esummary(pmids)

            # Verify success
            assert len(result) == 2
            assert result[0]["pmid"] == "12345678"
            # Verify it made 2 attempts
            assert mock_session.get.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_uses_backoff(self, pubmed_backend):
        """Test that retries use exponential backoff with proper timing."""
        from interaction_finder.search.backends.pubmed import INITIAL_BACKOFF_SECONDS

        query = SearchQuery(query="covid", max_results=10)

        # Always return 429 to trigger all retries
        mock_response = AsyncMock()
        mock_response.status_code = 429

        with patch.object(pubmed_backend, "_get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_session.get = AsyncMock(return_value=mock_response)
            mock_get_session.return_value = mock_session

            start_time = asyncio.get_event_loop().time()

            # Should fail after all retries
            with pytest.raises(RuntimeError, match="rate limit exceeded"):
                await pubmed_backend._esearch(query)

            end_time = asyncio.get_event_loop().time()
            elapsed = end_time - start_time

            # Should have slept for approximately: 1s + 2s + 4s = 7s (with jitter)
            # Allow generous tolerance due to jitter and test timing
            min_expected_backoff = INITIAL_BACKOFF_SECONDS * (1 + 2 + 4) * 0.7
            assert elapsed >= min_expected_backoff

    @pytest.mark.asyncio
    async def test_backoff_calculation(self, pubmed_backend):
        """Test backoff calculation increases exponentially."""
        from interaction_finder.search.backends.pubmed import (
            MAX_BACKOFF_SECONDS,
        )

        # First attempt (attempt 0)
        backoff_0 = pubmed_backend._calculate_backoff(0)
        # Should be ~1s with jitter
        assert 0.8 <= backoff_0 <= 1.5

        # Second attempt (attempt 1)
        backoff_1 = pubmed_backend._calculate_backoff(1)
        # Should be ~2s with jitter
        assert 1.6 <= backoff_1 <= 2.5

        # Third attempt (attempt 2)
        backoff_2 = pubmed_backend._calculate_backoff(2)
        # Should be ~4s with jitter
        assert 3.2 <= backoff_2 <= 5.0

        # Verify exponential growth (accounting for jitter)
        assert backoff_1 > backoff_0
        assert backoff_2 > backoff_1

        # Test that very large attempts are capped at MAX_BACKOFF_SECONDS
        backoff_large = pubmed_backend._calculate_backoff(10)
        # Should be capped at 60s with jitter
        assert backoff_large <= MAX_BACKOFF_SECONDS * 1.3

    @pytest.mark.asyncio
    async def test_non_429_errors_do_not_retry(self, pubmed_backend):
        """Test that non-429 HTTP errors fail immediately without retry."""
        query = SearchQuery(query="covid", max_results=10)

        # Return 500 error
        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch.object(pubmed_backend, "_get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_session.get = AsyncMock(return_value=mock_response)
            mock_get_session.return_value = mock_session

            # Should fail immediately without retry
            with pytest.raises(RuntimeError, match="HTTP 500"):
                await pubmed_backend._esearch(query)

            # Verify it only attempted once
            assert mock_session.get.call_count == 1


class TestBuildSearchParams:
    """Test search parameter construction."""

    def test_build_search_params_minimal(self, pubmed_backend):
        """Test building search params with minimal query."""
        query = SearchQuery(query="covid", max_results=10)
        params = pubmed_backend._build_search_params(query)

        assert params["db"] == "pubmed"
        assert params["term"] == "covid"
        assert params["retmax"] == "10"
        assert params["retmode"] == "xml"
        assert params["usehistory"] == "y"
        assert params["email"] == "test@example.com"
        assert params["api_key"] == "test_key"

    def test_build_search_params_no_credentials(self):
        """Test building params without email/api_key."""
        backend = PubMedBackend({})
        query = SearchQuery(query="covid", max_results=10)
        params = backend._build_search_params(query)

        assert "email" not in params
        assert "api_key" not in params


class TestXMLParsing:
    """Test XML response parsing."""

    def test_parse_esearch_success(self, pubmed_backend):
        """Test parsing successful ESearch response."""
        result = pubmed_backend._parse_esearch_response(ESEARCH_SUCCESS_XML)

        assert result["count"] == 2
        assert result["pmids"] == ["12345678", "87654321"]
        assert result["web_env"] == "test_web_env"
        assert result["query_key"] == "1"

    def test_parse_esearch_empty_results(self, pubmed_backend):
        """Test parsing ESearch response with no results."""
        result = pubmed_backend._parse_esearch_response(ESEARCH_EMPTY_XML)

        assert result["count"] == 0
        assert result["pmids"] == []

    def test_parse_esearch_error(self, pubmed_backend):
        """Test parsing ESearch response with errors."""
        with pytest.raises(RuntimeError, match="PubMed search errors"):
            pubmed_backend._parse_esearch_response(ESEARCH_ERROR_XML)

    def test_parse_esearch_malformed_xml(self, pubmed_backend):
        """Test parsing malformed XML raises appropriate error."""
        malformed_xml = "<eSearchResult><Count>broken</eSearchResult"

        with pytest.raises(
            RuntimeError, match="Failed to parse PubMed search response"
        ):
            pubmed_backend._parse_esearch_response(malformed_xml)

    def test_parse_esearch_error_with_none_text(self, pubmed_backend):
        """Test parsing ESearch errors filters out None text values."""
        xml_with_none = """<?xml version="1.0" encoding="UTF-8"?>
        <eSearchResult>
            <Count>0</Count>
            <ErrorList>
                <PhraseNotFound>valid error</PhraseNotFound>
                <PhraseNotFound></PhraseNotFound>
            </ErrorList>
        </eSearchResult>
        """

        with pytest.raises(RuntimeError, match="valid error"):
            pubmed_backend._parse_esearch_response(xml_with_none)

    def test_parse_esummary_success(self, pubmed_backend):
        """Test parsing successful ESummary response."""
        summaries = pubmed_backend._parse_esummary_response(ESUMMARY_SUCCESS_XML)

        assert len(summaries) == 2

        # First summary
        assert summaries[0]["pmid"] == "12345678"
        assert summaries[0]["Title"] == "COVID-19 and inflammation"
        assert summaries[0]["Source"] == "Nature"
        assert summaries[0]["PubDate"] == "2023"
        assert summaries[0]["AuthorList"] == ["Smith J", "Doe A"]
        assert summaries[0]["Volume"] == 123

        # Second summary
        assert summaries[1]["pmid"] == "87654321"
        assert summaries[1]["Title"] == "SARS-CoV-2 mechanisms"

    def test_parse_esummary_missing_fields(self, pubmed_backend):
        """Test parsing ESummary with missing fields."""
        summaries = pubmed_backend._parse_esummary_response(ESUMMARY_MISSING_FIELDS_XML)

        assert len(summaries) == 1
        assert summaries[0]["pmid"] == "11111111"

    def test_parse_esummary_malformed_xml(self, pubmed_backend):
        """Test parsing malformed ESummary XML."""
        malformed_xml = "<eSummaryResult><DocSum>broken</eSummaryResult"

        with pytest.raises(
            RuntimeError, match="Failed to parse PubMed summary response"
        ):
            pubmed_backend._parse_esummary_response(malformed_xml)

    def test_parse_esummary_empty_response(self, pubmed_backend):
        """Test parsing empty ESummary response."""
        empty_xml = (
            '<?xml version="1.0" encoding="UTF-8"?><eSummaryResult></eSummaryResult>'
        )
        summaries = pubmed_backend._parse_esummary_response(empty_xml)

        assert summaries == []

    def test_parse_esummary_integer_conversion_error(self, pubmed_backend):
        """Test ESummary integer parsing handles non-numeric values."""
        xml_with_bad_int = """<?xml version="1.0" encoding="UTF-8"?>
        <eSummaryResult>
            <DocSum>
                <Id>12345</Id>
                <Item Name="Volume" Type="Integer">not_a_number</Item>
            </DocSum>
        </eSummaryResult>
        """

        summaries = pubmed_backend._parse_esummary_response(xml_with_bad_int)
        assert summaries[0]["Volume"] == 0  # Defaults to 0 on error


class TestResultConversion:
    """Test conversion of PubMed summaries to SearchResult objects."""

    def test_convert_summary_to_result(self, pubmed_backend):
        """Test converting PubMed summary to SearchResult."""
        summary = {
            "pmid": "12345678",
            "Title": "Test Article",
            "Source": "Nature",
        }

        result = pubmed_backend._convert_pubmed_summary_to_result(summary)

        assert isinstance(result, SearchResult)
        assert result.title == "Test Article"
        assert result.url == "https://pubmed.ncbi.nlm.nih.gov/12345678/"
        assert result.snippet == "Nature"
        assert result.relevance is None

    def test_convert_summary_minimal_fields(self, pubmed_backend):
        """Test converting summary with minimal fields."""
        summary = {"pmid": "12345"}

        result = pubmed_backend._convert_pubmed_summary_to_result(summary)

        assert result.title == ""
        assert result.url == "https://pubmed.ncbi.nlm.nih.gov/12345/"
        assert result.snippet is None

    def test_convert_summary_no_source(self, pubmed_backend):
        """Test converting summary without Source field."""
        summary = {"pmid": "12345", "Title": "Test"}

        result = pubmed_backend._convert_pubmed_summary_to_result(summary)

        assert result.snippet is None


class TestESearchAPI:
    """Test ESearch API interactions."""

    @pytest.mark.asyncio
    async def test_esearch_success_get(self, pubmed_backend):
        """Test successful ESearch with GET request."""
        query = SearchQuery(query="covid", max_results=10)

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = ESEARCH_SUCCESS_XML

        with patch.object(pubmed_backend, "_get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_session.get = AsyncMock(return_value=mock_response)
            mock_get_session.return_value = mock_session

            result = await pubmed_backend._esearch(query)

            assert result["count"] == 2
            assert len(result["pmids"]) == 2
            mock_session.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_esearch_success_post(self, pubmed_backend):
        """Test successful ESearch with POST request for long query."""
        # Create a very long query that exceeds 2000 characters
        long_query = "covid " * 500
        query = SearchQuery(query=long_query, max_results=10)

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = ESEARCH_SUCCESS_XML

        with patch.object(pubmed_backend, "_get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_session.post = AsyncMock(return_value=mock_response)
            mock_get_session.return_value = mock_session

            result = await pubmed_backend._esearch(query)

            assert result["count"] == 2
            mock_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_esearch_rate_limit_retries_then_fails(self, pubmed_backend):
        """Test ESearch retries on 429 and fails after MAX_RETRIES attempts."""
        from interaction_finder.search.backends.pubmed import MAX_RETRIES

        query = SearchQuery(query="covid", max_results=10)

        mock_response = AsyncMock()
        mock_response.status_code = 429

        with patch.object(pubmed_backend, "_get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_session.get = AsyncMock(return_value=mock_response)
            mock_get_session.return_value = mock_session

            # Should retry MAX_RETRIES times then fail
            with pytest.raises(
                RuntimeError, match=f"rate limit exceeded after {MAX_RETRIES} retries"
            ):
                await pubmed_backend._esearch(query)

            # Verify it attempted MAX_RETRIES + 1 times (initial + retries)
            assert mock_session.get.call_count == MAX_RETRIES + 1

    @pytest.mark.asyncio
    async def test_esearch_http_error(self, pubmed_backend):
        """Test ESearch handles HTTP error responses."""
        query = SearchQuery(query="covid", max_results=10)

        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch.object(pubmed_backend, "_get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_session.get = AsyncMock(return_value=mock_response)
            mock_get_session.return_value = mock_session

            with pytest.raises(RuntimeError, match="HTTP 500"):
                await pubmed_backend._esearch(query)

    @pytest.mark.asyncio
    async def test_esearch_timeout(self, pubmed_backend):
        """Test ESearch handles timeout errors."""
        query = SearchQuery(query="covid", max_results=10)

        with patch.object(pubmed_backend, "_get_session") as mock_get_session:
            import httpx

            mock_session = AsyncMock()
            mock_session.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_get_session.return_value = mock_session

            with pytest.raises(RuntimeError, match="timed out"):
                await pubmed_backend._esearch(query)

    @pytest.mark.asyncio
    async def test_esearch_network_error(self, pubmed_backend):
        """Test ESearch handles network errors."""
        query = SearchQuery(query="covid", max_results=10)

        with patch.object(pubmed_backend, "_get_session") as mock_get_session:
            import httpx

            mock_session = AsyncMock()
            mock_session.get = AsyncMock(
                side_effect=httpx.RequestError("network error")
            )
            mock_get_session.return_value = mock_session

            with pytest.raises(RuntimeError, match="network error"):
                await pubmed_backend._esearch(query)


class TestESummaryAPI:
    """Test ESummary API interactions."""

    @pytest.mark.asyncio
    async def test_esummary_success_get(self, pubmed_backend):
        """Test successful ESummary with GET request."""
        pmids = ["12345678", "87654321"]

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = ESUMMARY_SUCCESS_XML

        with patch.object(pubmed_backend, "_get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_session.get = AsyncMock(return_value=mock_response)
            mock_get_session.return_value = mock_session

            summaries = await pubmed_backend._esummary(pmids)

            assert len(summaries) == 2
            mock_session.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_esummary_success_post(self, pubmed_backend):
        """Test successful ESummary with POST for many PMIDs."""
        # Create 250 PMIDs to trigger POST
        pmids = [str(i) for i in range(250)]

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = ESUMMARY_SUCCESS_XML

        with patch.object(pubmed_backend, "_get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_session.post = AsyncMock(return_value=mock_response)
            mock_get_session.return_value = mock_session

            await pubmed_backend._esummary(pmids)

            mock_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_esummary_empty_list(self, pubmed_backend):
        """Test ESummary with empty PMID list."""
        summaries = await pubmed_backend._esummary([])
        assert summaries == []

    @pytest.mark.asyncio
    async def test_esummary_rate_limit_retries_then_fails(self, pubmed_backend):
        """Test ESummary retries on 429 and fails after MAX_RETRIES attempts."""
        from interaction_finder.search.backends.pubmed import MAX_RETRIES

        pmids = ["12345678"]

        mock_response = AsyncMock()
        mock_response.status_code = 429

        with patch.object(pubmed_backend, "_get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_session.get = AsyncMock(return_value=mock_response)
            mock_get_session.return_value = mock_session

            # Should retry MAX_RETRIES times then fail
            with pytest.raises(
                RuntimeError, match=f"rate limit exceeded after {MAX_RETRIES} retries"
            ):
                await pubmed_backend._esummary(pmids)

            # Verify it attempted MAX_RETRIES + 1 times (initial + retries)
            assert mock_session.get.call_count == MAX_RETRIES + 1

    @pytest.mark.asyncio
    async def test_esummary_timeout(self, pubmed_backend):
        """Test ESummary handles timeout errors."""
        pmids = ["12345678"]

        with patch.object(pubmed_backend, "_get_session") as mock_get_session:
            import httpx

            mock_session = AsyncMock()
            mock_session.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_get_session.return_value = mock_session

            with pytest.raises(RuntimeError, match="timed out"):
                await pubmed_backend._esummary(pmids)


class TestSearchMethod:
    """Test the main search() method."""

    @pytest.mark.asyncio
    async def test_search_success(self, pubmed_backend):
        """Test successful end-to-end search."""
        query = SearchQuery(query="covid", max_results=10)

        # Mock ESearch
        with patch.object(pubmed_backend, "_esearch") as mock_esearch:
            mock_esearch.return_value = {
                "pmids": ["12345678", "87654321"],
                "count": 2,
                "web_env": "test",
                "query_key": "1",
            }

            # Mock ESummary
            with patch.object(pubmed_backend, "_esummary") as mock_esummary:
                mock_esummary.return_value = [
                    {"pmid": "12345678", "Title": "Article 1", "Source": "Nature"},
                    {"pmid": "87654321", "Title": "Article 2", "Source": "Science"},
                ]

                results = await pubmed_backend.search(query)

                assert len(results) == 2
                assert all(isinstance(r, SearchResult) for r in results)
                assert results[0].title == "Article 1"
                assert results[1].title == "Article 2"

    @pytest.mark.asyncio
    async def test_search_empty_results(self, pubmed_backend):
        """Test search with no results."""
        query = SearchQuery(query="nonexistent", max_results=10)

        with patch.object(pubmed_backend, "_esearch") as mock_esearch:
            mock_esearch.return_value = {
                "pmids": [],
                "count": 0,
                "web_env": None,
                "query_key": None,
            }

            results = await pubmed_backend.search(query)

            assert results == []

    @pytest.mark.asyncio
    async def test_search_conversion_error_skipped(self, pubmed_backend):
        """Test that conversion errors for individual results don't fail entire search."""
        query = SearchQuery(query="covid", max_results=10)

        with patch.object(pubmed_backend, "_esearch") as mock_esearch:
            mock_esearch.return_value = {
                "pmids": ["12345678", "87654321"],
                "count": 2,
                "web_env": "test",
                "query_key": "1",
            }

            with patch.object(pubmed_backend, "_esummary") as mock_esummary:
                mock_esummary.return_value = [
                    {"pmid": "12345678", "Title": "Article 1"},
                    {"pmid": "87654321", "Title": "Article 2"},
                ]

                # Make first conversion fail
                original_convert = pubmed_backend._convert_pubmed_summary_to_result

                def mock_convert(summary):
                    if summary["pmid"] == "12345678":
                        raise ValueError("Conversion error")
                    return original_convert(summary)

                with patch.object(
                    pubmed_backend,
                    "_convert_pubmed_summary_to_result",
                    side_effect=mock_convert,
                ):
                    results = await pubmed_backend.search(query)

                    # Should get only the second result
                    assert len(results) == 1
                    assert results[0].title == "Article 2"

    @pytest.mark.asyncio
    async def test_search_propagates_runtime_errors(self, pubmed_backend):
        """Test that RuntimeErrors are propagated from search."""
        query = SearchQuery(query="covid", max_results=10)

        with patch.object(pubmed_backend, "_esearch") as mock_esearch:
            mock_esearch.side_effect = RuntimeError("API error")

            with pytest.raises(RuntimeError, match="API error"):
                await pubmed_backend.search(query)

    @pytest.mark.asyncio
    async def test_search_wraps_unexpected_errors(self, pubmed_backend):
        """Test that unexpected errors are wrapped in RuntimeError."""
        query = SearchQuery(query="covid", max_results=10)

        with patch.object(pubmed_backend, "_esearch") as mock_esearch:
            mock_esearch.side_effect = ValueError("Unexpected error")

            with pytest.raises(
                RuntimeError, match="Unexpected error during PubMed search"
            ):
                await pubmed_backend.search(query)


class TestHealthCheck:
    """Test health check functionality."""

    def test_healthy_check_sync(self, pubmed_backend):
        """Test synchronous healthy() method."""
        with patch.object(pubmed_backend, "_async_health_check") as mock_async_check:
            mock_async_check.return_value = True

            # Mock event loop
            with patch("asyncio.get_event_loop") as mock_get_loop:
                mock_loop = Mock()
                mock_loop.is_running.return_value = False
                mock_loop.run_until_complete.return_value = True
                mock_get_loop.return_value = mock_loop

                result = pubmed_backend.healthy()

                assert result is True
                mock_loop.run_until_complete.assert_called_once()

    def test_healthy_check_running_loop(self, pubmed_backend):
        """Test healthy() when event loop is already running."""
        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = Mock()
            mock_loop.is_running.return_value = True
            mock_get_loop.return_value = mock_loop

            # Should return True without calling async health check
            result = pubmed_backend.healthy()

            assert result is True

    def test_healthy_check_no_loop(self, pubmed_backend):
        """Test healthy() when no event loop exists."""
        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_get_loop.side_effect = RuntimeError("No event loop")

            with patch("asyncio.run") as mock_run:
                mock_run.return_value = True

                result = pubmed_backend.healthy()

                assert result is True
                mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_health_check_success(self, pubmed_backend):
        """Test async health check succeeds."""
        with patch.object(pubmed_backend, "_esearch") as mock_esearch:
            mock_esearch.return_value = {"pmids": [], "count": 0}

            result = await pubmed_backend._async_health_check()

            assert result is True

    @pytest.mark.asyncio
    async def test_async_health_check_failure(self, pubmed_backend):
        """Test async health check fails on error."""
        with patch.object(pubmed_backend, "_esearch") as mock_esearch:
            mock_esearch.side_effect = RuntimeError("API down")

            result = await pubmed_backend._async_health_check()

            assert result is False


class TestABCCompliance:
    """Test SearchBackend ABC compliance."""

    def test_implements_name_property(self, pubmed_backend):
        """Test that name property is implemented."""
        assert hasattr(pubmed_backend, "name")
        assert isinstance(pubmed_backend.name, str)

    def test_implements_search_method(self, pubmed_backend):
        """Test that search method is implemented."""
        assert hasattr(pubmed_backend, "search")
        assert callable(pubmed_backend.search)

    def test_implements_healthy_method(self, pubmed_backend):
        """Test that healthy method is implemented."""
        assert hasattr(pubmed_backend, "healthy")
        assert callable(pubmed_backend.healthy)

    def test_inherits_from_searchbackend(self, pubmed_backend):
        """Test that PubMedBackend inherits from SearchBackend."""
        from interaction_finder.search.models import SearchBackend

        assert isinstance(pubmed_backend, SearchBackend)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_search_with_unicode_query(self, pubmed_backend):
        """Test search handles unicode characters in query."""
        query = SearchQuery(query="α-synuclein β-amyloid", max_results=10)

        with patch.object(pubmed_backend, "_esearch") as mock_esearch:
            mock_esearch.return_value = {
                "pmids": [],
                "count": 0,
                "web_env": None,
                "query_key": None,
            }

            results = await pubmed_backend.search(query)

            assert results == []
            # Verify query was passed correctly
            call_args = mock_esearch.call_args[0][0]
            assert call_args.query == "α-synuclein β-amyloid"

    @pytest.mark.asyncio
    async def test_search_with_max_results(self, pubmed_backend):
        """Test search respects max_results parameter."""
        query = SearchQuery(query="covid", max_results=5)

        with patch.object(pubmed_backend, "_esearch") as mock_esearch:
            mock_esearch.return_value = {
                "pmids": [],
                "count": 0,
                "web_env": None,
                "query_key": None,
            }

            await pubmed_backend.search(query)

            # Check that params were built with correct max_results
            call_args = mock_esearch.call_args[0][0]
            assert call_args.max_results == 5

    def test_parse_esummary_handles_empty_string_items(self, pubmed_backend):
        """Test ESummary parsing handles empty string items correctly."""
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <eSummaryResult>
            <DocSum>
                <Id>12345</Id>
                <Item Name="Title" Type="String"></Item>
                <Item Name="Source" Type="String"></Item>
            </DocSum>
        </eSummaryResult>
        """

        summaries = pubmed_backend._parse_esummary_response(xml)

        assert len(summaries) == 1
        assert summaries[0]["Title"] == ""
        assert summaries[0]["Source"] == ""

    def test_parse_esummary_handles_empty_list_items(self, pubmed_backend):
        """Test ESummary parsing handles empty list items."""
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <eSummaryResult>
            <DocSum>
                <Id>12345</Id>
                <Item Name="AuthorList" Type="List"></Item>
            </DocSum>
        </eSummaryResult>
        """

        summaries = pubmed_backend._parse_esummary_response(xml)

        assert len(summaries) == 1
        assert summaries[0]["AuthorList"] == []
