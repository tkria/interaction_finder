"""
PubMed search backend using NCBI E-utilities API.

This module provides a SearchBackend implementation for querying PubMed/MEDLINE
using the NCBI E-utilities web API. It handles rate limiting, query formatting,
and result parsing.
"""

import asyncio
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore

from interaction_finder.logging import logfire
from ..models import (
    SearchBackend,
    SearchQuery,
    SearchResult,
)


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

    async def _get_session(self) -> httpx.AsyncClient:  # type: ignore[valid-type]
        """Get or create HTTP session."""
        if self._session is None or self._session.is_closed:
            self._session = httpx.AsyncClient(timeout=self.timeout)  # type: ignore[misc]
        return self._session

    async def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting for API requests."""
        async with self._request_lock:
            now = asyncio.get_event_loop().time()
            time_since_last = now - self._last_request_time
            min_interval = 1.0 / self.rate_limit

            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                await asyncio.sleep(sleep_time)

            self._last_request_time = asyncio.get_event_loop().time()

    def _build_search_params(self, query: SearchQuery) -> Dict[str, str]:
        """Build parameters for ESearch request."""
        params = {
            "db": "pubmed",
            "term": query.query,
            "retmax": str(query.max_results),
            "retmode": "xml",
            "usehistory": "y",  # Use history for large result sets
        }

        # Add email and API key if provided
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key

        return params

    async def _esearch(self, query: SearchQuery) -> Dict[str, Any]:
        """Perform ESearch to get PMIDs."""
        await self._enforce_rate_limit()

        params = self._build_search_params(query)
        url = f"{self.base_url}/esearch.fcgi"

        session = await self._get_session()

        # Use POST for long queries to avoid URI length limits
        query_length = len(params.get("term", ""))
        use_post = query_length > 2000

        try:
            if use_post:
                # Use POST with form data for long queries
                response = await session.post(url, data=params)
            else:
                # Use GET for short queries
                response = await session.get(url, params=params)

            if response.status_code == 429:
                raise RuntimeError("PubMed rate limit exceeded")
            elif response.status_code != 200:
                raise RuntimeError(
                    f"PubMed API returned HTTP {response.status_code}: {response.text}"
                )

            content = response.text
            return self._parse_esearch_response(content)

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

            # Check for errors
            error_list = root.find("ErrorList")
            if error_list is not None:
                errors = [
                    error.text
                    for error in error_list.findall("PhraseNotFound")
                    if error.text
                ]
                if errors:
                    raise RuntimeError(f"PubMed search errors: {', '.join(errors)}")

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
        """Fetch summaries for PMIDs using ESummary."""
        if not pmids:
            return []

        await self._enforce_rate_limit()

        params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}

        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key

        url = f"{self.base_url}/esummary.fcgi"
        session = await self._get_session()

        # Use POST for many PMIDs to avoid URI length limits
        id_list_length = len(params["id"])
        use_post = id_list_length > 2000 or len(pmids) > 200

        try:
            if use_post:
                # Use POST with form data for long PMID lists
                response = await session.post(url, data=params)
            else:
                # Use GET for short PMID lists
                response = await session.get(url, params=params)

            if response.status_code == 429:
                raise RuntimeError("PubMed rate limit exceeded")
            elif response.status_code != 200:
                raise RuntimeError(
                    f"PubMed API returned HTTP {response.status_code}: {response.text}"
                )

            content = response.text
            return self._parse_esummary_response(content)

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
        with logfire.span(
            "PubMedBackend.search",
            query=query.query[:100],
            max_results=query.max_results,
        ):
            logfire.info(f"Searching PubMed: {query.query[:100]}...")
            try:
                # Step 1: Search for PMIDs
                search_result = await self._esearch(query)
                pmids = search_result["pmids"]
                total_count = search_result["count"]
                logfire.info(f"Found {len(pmids)} PMIDs (total matches: {total_count})")

                if not pmids:
                    return []

                # Step 2: Fetch summaries for PMIDs
                summaries = await self._esummary(pmids)
                logfire.info(f"Fetched {len(summaries)} summaries")

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
                        logfire.warning(
                            f"Failed to convert PubMed summary to result: {e}"
                        )
                        continue

                if conversion_errors > 0:
                    logfire.warning(
                        f"Failed to convert {conversion_errors}/{len(summaries)} summaries"
                    )

                logfire.info(f"PubMed search complete: {len(results)} results")
                return results

            except RuntimeError:
                # Re-raise runtime errors
                logfire.error("PubMed search failed with RuntimeError")
                raise
            except Exception as e:
                # Wrap unexpected errors
                logfire.error(f"Unexpected error during PubMed search: {str(e)}")
                raise RuntimeError(f"Unexpected error during PubMed search: {str(e)}")

    async def _async_health_check(self) -> bool:
        """Check if PubMed API is available and working (async)."""
        try:
            # Simple test query
            test_query = SearchQuery(query="covid", max_results=1)
            await self._esearch(test_query)
            return True
        except Exception:
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
