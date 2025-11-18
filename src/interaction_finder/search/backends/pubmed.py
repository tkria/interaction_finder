"""
PubMed search backend using NCBI E-utilities API.

This module provides a SearchBackend implementation for querying PubMed/MEDLINE
using the NCBI E-utilities web API. It handles rate limiting, query formatting,
result parsing, and automatic retry with exponential backoff for rate limit errors.

Rate Limit Handling:
- Proactive rate limiting: Enforces configurable requests/second before each API call
- Reactive retry logic: On HTTP 429 errors, retries with exponential backoff
- Backoff strategy: Initial 1s delay, doubles each retry (1s, 2s, 4s), capped at 60s
- Max retries: 3 attempts (configurable via MAX_RETRIES constant)
- Jitter: ±20% randomness to avoid thundering herd
"""

import asyncio
import random
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore

from interaction_finder.logging import logfire, get_logger

logger = get_logger(__name__)
from ..models import (
    SearchBackend,
    SearchQuery,
    SearchResult,
)

# Retry configuration constants
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 60.0
BACKOFF_JITTER_FRACTION = 0.2


class PubMedBackend(SearchBackend):
    """PubMed search backend using NCBI E-utilities API."""

    def __init__(self, config: Dict[str, Any] = {}):
        """Initialize PubMed backend with configuration."""
        super().__init__(config)

        if not HTTPX_AVAILABLE:
            raise RuntimeError(
                "httpx is required for PubMed backend but not installed. "
                "Install it with: pip install httpx"
            )

        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.email = config.get("email")
        self.api_key = config.get("api_key")
        self.rate_limit = config.get("rate_limit", 3.0)
        self.timeout = config.get("timeout", 30)
        self.retmode = config.get("retmode", "xml")
        self.use_mesh = config.get("use_mesh", True)

        # Rate limiting
        self._last_request_time = 0.0
        self._request_lock = asyncio.Lock()

        # Session for connection reuse
        self._session: Optional[httpx.AsyncClient] = None  # type: ignore[valid-type]

    @property
    def name(self) -> str:
        """Name identifier for this backend."""
        return "pubmed"

    def should_show_api_key_warning(self) -> bool:
        """Check if API key warning should be displayed."""
        return not self.api_key

    async def _get_session(self) -> httpx.AsyncClient:  # type: ignore[valid-type]
        """Get or create HTTP session."""
        if self._session is None or self._session.is_closed:
            self._session = httpx.AsyncClient(timeout=self.timeout)  # type: ignore[misc]
        return self._session

    async def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting for API requests."""
        async with self._request_lock:
            elapsed = asyncio.get_event_loop().time() - self._last_request_time
            if sleep_time := max(0, 1.0 / self.rate_limit - elapsed):
                await asyncio.sleep(sleep_time)
            self._last_request_time = asyncio.get_event_loop().time()

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter: 1s, 2s, 4s (±20%)."""
        backoff = min(INITIAL_BACKOFF_SECONDS * (2**attempt), MAX_BACKOFF_SECONDS)
        jitter = backoff * BACKOFF_JITTER_FRACTION * (2 * random.random() - 1)
        return backoff + jitter

    async def _request_with_retry(
        self, url: str, params: Dict[str, Any], use_post: bool
    ) -> str:
        """Execute HTTP request with rate limiting and retry on 429 errors."""
        for attempt in range(MAX_RETRIES + 1):
            await self._enforce_rate_limit()
            session = await self._get_session()
            response = (
                await session.post(url, data=params)
                if use_post
                else await session.get(url, params=params)
            )

            if response.status_code == 200:
                return response.text

            if response.status_code == 429 and attempt < MAX_RETRIES:
                backoff = self._calculate_backoff(attempt)
                logger.warning(
                    f"PubMed rate limit (429), retry {attempt + 1}/{MAX_RETRIES} "
                    f"after {backoff:.1f}s"
                )
                await asyncio.sleep(backoff)
                continue

            # Rate limit exhausted or other HTTP error
            error_msg = (
                f"rate limit exceeded after {MAX_RETRIES} retries"
                if response.status_code == 429
                else f"HTTP {response.status_code}: {response.text}"
            )
            raise RuntimeError(f"PubMed API error: {error_msg}")

    def _build_search_params(self, query: SearchQuery) -> Dict[str, str]:
        """Build parameters for ESearch request."""
        params = {
            "db": "pubmed",
            "term": query.query,
            "retmax": str(query.max_results),
            "retmode": "xml",
            "usehistory": "y",
        }
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    async def _esearch(self, query: SearchQuery) -> Dict[str, Any]:
        """Perform ESearch to get PMIDs with retry on rate limit errors."""
        params = self._build_search_params(query)
        url = f"{self.base_url}/esearch.fcgi"
        use_post = len(params.get("term", "")) > 2000

        try:
            response_text = await self._request_with_retry(url, params, use_post)
            return self._parse_esearch_response(response_text)
        except httpx.TimeoutException:  # type: ignore[misc]
            raise RuntimeError(
                f"PubMed search request timed out for query: {query.query}"
            )
        except httpx.RequestError as e:  # type: ignore[misc]
            raise RuntimeError(f"PubMed network error: {str(e)}")

    def _parse_esearch_response(self, xml_content: str) -> Dict[str, Any]:
        """Parse ESearch XML response."""
        try:
            root = ET.fromstring(xml_content)

            # Check for errors - log but don't fail on PhraseNotFound
            error_list = root.find("ErrorList")
            if error_list is not None:
                errors = [
                    error.text
                    for error in error_list.findall("PhraseNotFound")
                    if error.text
                ]
                if errors:
                    # Log the problematic phrases but return empty results instead of failing
                    logger.warning(
                        f"PubMed rejected search phrases: {', '.join(errors)}",
                        extra={
                            "rejected_phrases": errors,
                        },
                    )
                    # Return empty result set - the query contained invalid syntax
                    return {
                        "pmids": [],
                        "count": 0,
                        "web_env": None,
                        "query_key": None,
                    }

            # Extract search results
            id_list = root.find("IdList")
            pmids = (
                [id_elem.text for id_elem in id_list.findall("Id")]
                if id_list is not None
                else []
            )

            count = root.findtext("Count", "0")
            web_env = root.findtext("WebEnv")
            query_key = root.findtext("QueryKey")

            return {
                "pmids": pmids,
                "count": int(count),
                "web_env": web_env,
                "query_key": query_key,
            }

        except ET.ParseError as e:
            raise RuntimeError(f"Failed to parse PubMed search response: {str(e)}")

    async def _esummary(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """Fetch summaries for PMIDs using ESummary with retry on rate limit errors."""
        if not pmids:
            return []

        params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key

        url = f"{self.base_url}/esummary.fcgi"
        use_post = len(params["id"]) > 2000 or len(pmids) > 200

        try:
            response_text = await self._request_with_retry(url, params, use_post)
            return self._parse_esummary_response(response_text)
        except httpx.TimeoutException:  # type: ignore[misc]
            raise RuntimeError("PubMed summary request timed out")
        except httpx.RequestError as e:  # type: ignore[misc]
            raise RuntimeError(f"PubMed network error: {str(e)}")

    def _parse_esummary_response(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse ESummary XML response."""
        try:
            root = ET.fromstring(xml_content)
            summaries: List[Dict[str, Any]] = []

            for doc_sum in root.findall("DocSum"):
                pmid = doc_sum.findtext("Id")
                if not pmid:
                    continue

                summary: Dict[str, Any] = {"pmid": pmid}

                # Parse item elements
                for item in doc_sum.findall("Item"):
                    name = item.get("Name")
                    item_type = item.get("Type")

                    # Skip items without name
                    if not name:
                        continue

                    if item_type == "String":
                        summary[name] = item.text or ""
                    elif item_type == "List":
                        # Handle list items (e.g., authors)
                        list_items = []
                        for list_item in item.findall("Item"):
                            list_items.append(list_item.text or "")
                        summary[name] = list_items
                    elif item_type == "Integer":
                        try:
                            summary[name] = int(item.text or "0")
                        except ValueError:
                            summary[name] = 0

                summaries.append(summary)

            return summaries

        except ET.ParseError as e:
            raise RuntimeError(f"Failed to parse PubMed summary response: {str(e)}")

    def _convert_pubmed_summary_to_result(
        self, summary: Dict[str, Any]
    ) -> SearchResult:
        """Convert PubMed summary to SearchResult."""
        pmid = summary.get("pmid", "")
        title = summary.get("Title", "")

        # Build URL
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

        # Extract snippet from summary if available
        snippet = None
        if "Source" in summary:
            # Use journal/source as snippet
            snippet = summary["Source"]

        return SearchResult(
            title=title,
            url=url,
            snippet=snippet,
            relevance=None,  # PubMed doesn't provide relevance scores
        )

    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform a search with the given query."""
        query_text = query.query
        with logfire.span(
            f"PubMed: {query_text}",
            query=query_text,
            max_results=query.max_results,
        ):
            try:
                # Step 1: Search for PMIDs
                search_result = await self._esearch(query)
                pmids = search_result["pmids"]
                total_count = search_result["count"]

                if not pmids:
                    logger.info(
                        f"PubMed search returned 0 results",
                        extra={
                            "query": query_text,
                            "total_matches": total_count,
                            "pmids": [],
                            "results": [],
                        },
                    )
                    return []

                # Step 2: Fetch summaries for PMIDs
                summaries = await self._esummary(pmids)

                # Step 3: Convert to SearchResult objects
                results = []
                conversion_errors = 0
                for summary in summaries:
                    try:
                        result = self._convert_pubmed_summary_to_result(summary)
                        results.append(result)
                    except Exception as e:
                        # Log conversion error but don't fail the whole search
                        conversion_errors += 1
                        logger.warning(
                            f"Failed to convert PubMed summary to result: {e}"
                        )
                        continue

                if conversion_errors > 0:
                    logger.warning(
                        f"Failed to convert {conversion_errors}/{len(summaries)} summaries"
                    )

                logger.info(
                    f"PubMed search returned {len(results)} results (total matches: {total_count})",
                    extra={
                        "query": query_text,
                        "total_matches": total_count,
                        "pmids": pmids,
                        "results": results,
                    },
                )
                return results

            except RuntimeError:
                # Re-raise runtime errors
                logger.error("PubMed search failed with RuntimeError")
                raise
            except Exception as e:
                # Wrap unexpected errors
                logger.error(f"Unexpected error during PubMed search: {str(e)}")
                raise RuntimeError(f"Unexpected error during PubMed search: {str(e)}")

    async def _async_health_check(self) -> bool:
        """Check if PubMed API is available and working (async)."""
        try:
            # Simple test query
            test_query = SearchQuery(query="covid", max_results=1)
            await self._esearch(test_query)
            return True
        except Exception as exc:
            logger.warning(
                "PubMed health check failed",
                extra={
                    "error": str(exc),
                    "base_url": self.base_url,
                },
            )
            return False

    def healthy(self) -> bool:
        """Check if PubMed API is available and working."""
        # Run async health check in a new event loop if needed
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we can't use run_until_complete
                # Just return True and let the actual search fail if unhealthy
                return True
            else:
                return loop.run_until_complete(self._async_health_check())
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self._async_health_check())

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.is_closed:
            await self._session.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
