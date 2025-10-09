"""
PubMed search backend using NCBI E-utilities API.

This module provides a SearchBackend implementation for querying PubMed/MEDLINE
using the NCBI E-utilities web API. It handles rate limiting, query formatting,
and result parsing.
"""

import asyncio
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional

import httpx

from ..base import (
    SearchBackend,
    SearchQuery,
    SearchResults,
    SearchResult,
    SearchError,
    SearchTimeoutError,
    SearchRateLimitError,
)


class PubMedBackend(SearchBackend):
    """PubMed search backend using NCBI E-utilities API."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize PubMed backend with configuration."""
        super().__init__(config)

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
        self._session: Optional[httpx.AsyncClient] = None

    @property
    def backend_name(self) -> str:
        """Name identifier for this backend."""
        return "pubmed"

    async def _get_session(self) -> httpx.AsyncClient:
        """Get or create HTTP session."""
        if self._session is None or self._session.is_closed:
            self._session = httpx.AsyncClient(timeout=self.timeout)
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
            "term": query.full_query,
            "retmax": str(query.max_results),
            "retmode": "xml",
            "usehistory": "y",  # Use history for large result sets
        }

        # Add email and API key if provided
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key

        # Add sort parameter if specified
        if query.filters and "sort_by" in query.filters:
            sort_by = query.filters["sort_by"]
            # Map sort_by values to PubMed API parameters
            sort_map = {
                "relevance": "relevance",
                "date": "pub_date",
                "date_desc": "pub_date",
            }
            params["sort"] = sort_map.get(sort_by, "relevance")
            # Add descending order for date_desc
            if sort_by == "date_desc":
                params["sort_order"] = "desc"

        # Add filters
        if query.filters:
            filter_terms = []

            # Date range filter
            if "date_range" in query.filters:
                date_range = query.filters["date_range"]
                if isinstance(date_range, dict):
                    start = date_range.get("start")
                    end = date_range.get("end")
                    if start and end:
                        filter_terms.append(
                            f'("{start}"[Date - Publication] : "{end}"[Date - Publication])'
                        )

            # Publication type filter
            if "publication_type" in query.filters:
                pub_types = query.filters["publication_type"]
                if isinstance(pub_types, list):
                    type_terms = [f'"{pt}"[Publication Type]' for pt in pub_types]
                    filter_terms.append(f"({' OR '.join(type_terms)})")

            # Journal filter
            if "journal" in query.filters:
                journal = query.filters["journal"]
                filter_terms.append(f'"{journal}"[Journal]')

            # Add filters to the main query
            if filter_terms:
                params["term"] = (
                    f"({params['term']}) AND ({' AND '.join(filter_terms)})"
                )

        return params

    async def _esearch(self, query: SearchQuery) -> Dict[str, Any]:
        """Perform ESearch to get PMIDs."""
        await self._enforce_rate_limit()

        params = self._build_search_params(query)
        url = f"{self.base_url}/esearch.fcgi"

        session = await self._get_session()

        # Use POST for long queries to avoid URI length limits
        query_length = len(params.get("term", ""))
        use_post = query_length > 2000  # Use POST if query is long

        try:
            if use_post:
                # Use POST with form data for long queries
                response = await session.post(url, data=params)
            else:
                # Use GET for short queries
                response = await session.get(url, params=params)

            if response.status_code == 429:
                raise SearchRateLimitError(
                    "Rate limit exceeded", backend=self.backend_name
                )
            elif response.status_code != 200:
                raise SearchError(
                    f"HTTP {response.status_code}: {response.text}",
                    backend=self.backend_name,
                )

            content = response.text
            return self._parse_esearch_response(content)

        except httpx.TimeoutException:
            raise SearchTimeoutError(
                "Search request timed out", backend=self.backend_name, query=query.query
            )
        except httpx.RequestError as e:
            raise SearchError(
                f"Network error: {str(e)}", backend=self.backend_name, query=query.query
            )

    def _parse_esearch_response(self, xml_content: str) -> Dict[str, Any]:
        """Parse ESearch XML response."""
        try:
            root = ET.fromstring(xml_content)

            # Check for errors
            error_list = root.find("ErrorList")
            if error_list is not None:
                errors = [error.text for error in error_list.findall("PhraseNotFound")]
                if errors:
                    raise SearchError(
                        f"Search errors: {', '.join(errors)}", backend=self.backend_name
                    )

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
            raise SearchError(
                f"Failed to parse search response: {str(e)}", backend=self.backend_name
            )

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
                raise SearchRateLimitError(
                    "Rate limit exceeded", backend=self.backend_name
                )
            elif response.status_code != 200:
                raise SearchError(
                    f"HTTP {response.status_code}: {response.text}",
                    backend=self.backend_name,
                )

            content = response.text
            return self._parse_esummary_response(content)

        except httpx.TimeoutException:
            raise SearchTimeoutError(
                "Summary request timed out", backend=self.backend_name
            )
        except httpx.RequestError as e:
            raise SearchError(f"Network error: {str(e)}", backend=self.backend_name)

    def _parse_esummary_response(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse ESummary XML response."""
        try:
            root = ET.fromstring(xml_content)
            summaries = []

            for doc_sum in root.findall("DocSum"):
                pmid = doc_sum.findtext("Id")
                if not pmid:
                    continue

                summary = {"pmid": pmid}

                # Parse item elements
                for item in doc_sum.findall("Item"):
                    name = item.get("Name")
                    item_type = item.get("Type")

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
            raise SearchError(
                f"Failed to parse summary response: {str(e)}", backend=self.backend_name
            )

    def _convert_pubmed_summary_to_result(
        self, summary: Dict[str, Any]
    ) -> SearchResult:
        """Convert PubMed summary to SearchResult."""
        pmid = summary.get("pmid", "")
        title = summary.get("Title", "")

        # Build URL
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

        # Note: Additional metadata like authors, DOI, journal, etc. is available
        # in the summary but moved to metadata since it's not reliably extractable
        # from all search backends

        return SearchResult(
            title=title,
            url=url,
            snippet=None,  # Could extract from abstract if available
            relevance_score=None,  # PubMed doesn't provide relevance scores
            backend=self.backend_name,
            metadata={"pubmed_summary": summary},
        )

    async def search(self, query: SearchQuery) -> SearchResults:
        """Perform a search with the given query."""
        start_time = asyncio.get_event_loop().time()

        try:
            # Step 1: Search for PMIDs
            search_result = await self._esearch(query)
            pmids = search_result["pmids"]

            if not pmids:
                return SearchResults(
                    query=query,
                    results=[],
                    total_found=0,
                    search_time=asyncio.get_event_loop().time() - start_time,
                    backend=self.backend_name,
                )

            # Step 2: Fetch summaries for PMIDs
            summaries = await self._esummary(pmids)

            # Step 3: Convert to SearchResult objects
            results = []
            for summary in summaries:
                try:
                    result = self._convert_pubmed_summary_to_result(summary)
                    results.append(result)
                except Exception as e:
                    # Log conversion error but don't fail the whole search
                    print(f"Warning: Failed to convert PubMed summary to result: {e}")
                    continue

            search_time = asyncio.get_event_loop().time() - start_time

            return SearchResults(
                query=query,
                results=results,
                total_found=search_result["count"],
                search_time=search_time,
                backend=self.backend_name,
            )

        except SearchError:
            # Re-raise search errors
            raise
        except Exception as e:
            # Wrap unexpected errors
            raise SearchError(
                f"Unexpected error during search: {str(e)}",
                backend=self.backend_name,
                query=query.query,
            )

    async def health_check(self) -> bool:
        """Check if PubMed API is available and working."""
        try:
            # Simple test query
            test_query = SearchQuery(query="covid", max_results=1)
            await self._esearch(test_query)
            return True
        except Exception:
            return False

    def supports_filters(self) -> List[str]:
        """Return list of supported filter types for PubMed."""
        return [
            "date_range",
            "publication_type",
            "journal",
            "author",
            "language",
            "has_fulltext",
        ]

    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Return rate limiting information for PubMed."""
        return {
            "requests_per_second": self.rate_limit,
            "requests_per_minute": None,
            "requests_per_hour": None,
            "burst_limit": None,
            "notes": "3 req/s without API key, 10 req/s with API key",
        }

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
