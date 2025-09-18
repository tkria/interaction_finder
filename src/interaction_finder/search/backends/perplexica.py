"""
Perplexica search backend for AI-powered academic search.

This module provides a SearchBackend implementation for querying Perplexica,
an AI-powered search engine that can perform academic searches with context
understanding and real-time information retrieval.
"""

import asyncio
import re
from datetime import datetime
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse

import httpx

from ..base import (
    SearchBackend,
    SearchQuery,
    SearchResults,
    SearchResult,
    SearchError,
    SearchTimeoutError,
    SearchUnavailableError,
)


class PerplexicaBackend(SearchBackend):
    """Perplexica search backend using local API instance."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Perplexica backend with configuration."""
        super().__init__(config)

        self.base_url = config.get("base_url", "http://localhost:3000")
        self.timeout = config.get("timeout", 60)
        self.search_mode = config.get("search_mode", "academicSearch")
        self.max_sources = config.get("max_sources", 20)

        # Model configurations
        self.chat_model = config.get(
            "chat_model", {"provider": "openai", "name": "gpt-4o-mini"}
        )
        self.embedding_model = config.get(
            "embedding_model", {"provider": "openai", "name": "text-embedding-3-large"}
        )

        # Session for connection reuse
        self._session: Optional[httpx.AsyncClient] = None

    @property
    def backend_name(self) -> str:
        """Name identifier for this backend."""
        return "perplexica"

    async def _get_session(self) -> httpx.AsyncClient:
        """Get or create HTTP session."""
        if self._session is None or self._session.is_closed:
            self._session = httpx.AsyncClient(timeout=self.timeout)
        return self._session

    def _build_search_request(self, query: SearchQuery) -> Dict[str, Any]:
        """Build the request payload for Perplexica API."""
        request_data = {
            "chatModel": self.chat_model,
            "embeddingModel": self.embedding_model,
            "optimizationMode": "balanced",  # Balance speed and quality
            "focusMode": self.search_mode,
            "query": query.full_query,
            "stream": False,  # Use non-streaming for simpler processing
        }

        # Add academic-focused system instructions
        system_instructions = []

        if self.search_mode == "academicSearch":
            system_instructions.append(
                "Focus on peer-reviewed scientific literature, research papers, and academic sources. "
                "Prioritize recent publications and high-impact journals."
            )

        # Add context based on query content or filters
        if query.filters:
            if "publication_type" in query.filters:
                pub_types = query.filters["publication_type"]
                if isinstance(pub_types, list):
                    types_str = ", ".join(pub_types)
                    system_instructions.append(f"Focus on {types_str} publications.")

            if "date_range" in query.filters:
                date_range = query.filters["date_range"]
                if isinstance(date_range, dict):
                    start = date_range.get("start")
                    end = date_range.get("end")
                    if start and end:
                        system_instructions.append(
                            f"Prioritize publications from {start} to {end}."
                        )

        if system_instructions:
            request_data["systemInstructions"] = " ".join(system_instructions)

        return request_data

    def _parse_perplexica_response(
        self, response_data: Dict[str, Any], query: SearchQuery
    ) -> List[SearchResult]:
        """Parse Perplexica API response into SearchResult objects."""
        results = []
        sources = response_data.get("sources", [])

        # Limit sources based on max_results
        limited_sources = sources[: query.max_results]

        for i, source in enumerate(limited_sources):
            try:
                result = self._convert_source_to_result(source, i)
                if result:
                    results.append(result)
            except Exception as e:
                # Log conversion error but don't fail the whole search
                print(f"Warning: Failed to convert Perplexica source to result: {e}")
                continue

        return results

    def _convert_source_to_result(
        self, source: Dict[str, Any], index: int
    ) -> Optional[SearchResult]:
        """Convert a Perplexica source to SearchResult."""
        metadata = source.get("metadata", {})
        page_content = source.get("pageContent", "")

        title = metadata.get("title", "")
        url = metadata.get("url", "")

        if not url or not title:
            return None

        # Try to extract additional metadata from content and URL
        authors = self._extract_authors_from_content(page_content, url)
        journal = self._extract_journal_from_url_or_content(url, page_content, title)
        pub_date = self._extract_publication_date(page_content, url)
        doi = self._extract_doi_from_content(page_content, url)
        pmid = self._extract_pmid_from_url(url)

        # Use page content as abstract if it looks academic
        abstract = (
            page_content
            if len(page_content) > 50 and len(page_content) < 2000
            else None
        )

        # Calculate a simple relevance score based on position (earlier = more relevant)
        relevance_score = max(0.1, 1.0 - (index * 0.05))

        return SearchResult(
            title=title,
            url=url,
            abstract=abstract,
            authors=authors,
            publication_date=pub_date,
            journal=journal,
            doi=doi,
            pmid=pmid,
            relevance_score=relevance_score,
            citation_count=None,  # Not available from Perplexica
            backend=self.backend_name,
            metadata={
                "perplexica_source": source,
                "page_content_length": len(page_content),
            },
        )

    def _extract_authors_from_content(self, content: str, url: str) -> List[str]:
        """Extract author names from page content."""
        authors = []

        # Look for common author patterns in academic content
        author_patterns = [
            r"(?i)(?:authors?|by):\s*([^.]+)",
            r"(?i)([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)*[A-Z][a-z]+)(?:\s*,\s*([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)*[A-Z][a-z]+))*",
            r"(?i)([A-Z][a-z]+,\s*[A-Z]\.(?:\s*[A-Z]\.)*)",
        ]

        for pattern in author_patterns:
            matches = re.findall(pattern, content)
            if matches:
                for match in matches[:5]:  # Limit to first 5 matches
                    if isinstance(match, tuple):
                        author = match[0].strip()
                    else:
                        author = match.strip()

                    if (
                        len(author) > 3 and len(author) < 50
                    ):  # Reasonable author name length
                        authors.append(author)

        # Clean up and deduplicate
        cleaned_authors = []
        seen = set()
        for author in authors:
            clean_author = re.sub(r"[^\w\s\.-]", "", author).strip()
            if clean_author and clean_author.lower() not in seen:
                seen.add(clean_author.lower())
                cleaned_authors.append(clean_author)

        return cleaned_authors[:10]  # Limit to reasonable number

    def _extract_journal_from_url_or_content(
        self, url: str, content: str, title: str
    ) -> Optional[str]:
        """Extract journal name from URL or content."""
        # Try to extract from URL domain
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()

        # Common journal domains
        journal_domains = {
            "pubmed.ncbi.nlm.nih.gov": "PubMed",
            "www.ncbi.nlm.nih.gov": "NCBI",
            "www.nature.com": "Nature",
            "science.org": "Science",
            "www.cell.com": "Cell",
            "www.nejm.org": "New England Journal of Medicine",
            "jamanetwork.com": "JAMA",
            "www.bmj.com": "BMJ",
            "www.thelancet.com": "The Lancet",
            "journals.plos.org": "PLOS",
            "www.frontiersin.org": "Frontiers",
            "link.springer.com": "Springer",
            "onlinelibrary.wiley.com": "Wiley",
            "www.sciencedirect.com": "ScienceDirect",
            "academic.oup.com": "Oxford Academic",
            "www.tandfonline.com": "Taylor & Francis",
        }

        for domain_pattern, journal_name in journal_domains.items():
            if domain_pattern in domain:
                return journal_name

        # Try to extract journal name from content
        journal_patterns = [
            r"(?i)(?:published in|journal|in)\s+([A-Z][^.,]{5,50})",
            r"(?i)([A-Z][a-z\s]+(?:Journal|Review|Letters|Science|Medicine|Biology))",
        ]

        for pattern in journal_patterns:
            matches = re.findall(pattern, content)
            if matches:
                journal = matches[0].strip()
                if 5 < len(journal) < 50:  # Reasonable journal name length
                    return journal

        return None

    def _extract_publication_date(self, content: str, url: str) -> Optional[datetime]:
        """Extract publication date from content."""
        # Look for various date patterns
        date_patterns = [
            r"(?i)(?:published|date|year):\s*(\d{4})",
            r"(?i)(\d{1,2}/\d{1,2}/\d{4})",
            r"(?i)(\d{4}-\d{2}-\d{2})",
            r"(?i)((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4})",
            r"(\d{4})",  # Just year as fallback
        ]

        for pattern in date_patterns:
            matches = re.findall(pattern, content)
            if matches:
                date_str = matches[0].strip()
                try:
                    # Try different date formats
                    if len(date_str) == 4 and date_str.isdigit():  # Just year
                        return datetime(int(date_str), 1, 1)
                    elif "/" in date_str:  # MM/DD/YYYY
                        return datetime.strptime(date_str, "%m/%d/%Y")
                    elif "-" in date_str:  # YYYY-MM-DD
                        return datetime.strptime(date_str, "%Y-%m-%d")
                    else:  # Try natural language date
                        import dateutil.parser

                        return dateutil.parser.parse(date_str)
                except (ValueError, ImportError):
                    continue

        return None

    def _extract_doi_from_content(self, content: str, url: str) -> Optional[str]:
        """Extract DOI from content or URL."""
        # Check if URL is a DOI link
        if "doi.org" in url:
            doi_match = re.search(r"doi\.org/(.+)", url)
            if doi_match:
                return doi_match.group(1)

        # Look for DOI in content
        doi_patterns = [
            r"(?i)DOI:\s*(10\.\d+/[^\s]+)",
            r"(?i)(10\.\d+/[^\s]+)",
        ]

        for pattern in doi_patterns:
            matches = re.findall(pattern, content)
            if matches:
                doi = matches[0].strip()
                if doi.startswith("10."):
                    return doi

        return None

    def _extract_pmid_from_url(self, url: str) -> Optional[str]:
        """Extract PubMed ID from URL if it's a PubMed link."""
        if "pubmed.ncbi.nlm.nih.gov" in url:
            pmid_match = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", url)
            if pmid_match:
                return pmid_match.group(1)

        return None

    async def search(self, query: SearchQuery) -> SearchResults:
        """Perform a search with the given query."""
        start_time = asyncio.get_event_loop().time()

        try:
            # Build request payload
            request_data = self._build_search_request(query)

            # Make API request
            session = await self._get_session()
            url = f"{self.base_url}/api/search"

            response = await session.post(url, json=request_data)

            if response.status_code == 400:
                raise SearchError(
                    f"Bad request to Perplexica: {response.text}",
                    backend=self.backend_name,
                    query=query.query,
                )
            elif response.status_code == 500:
                raise SearchError(
                    "Perplexica internal server error",
                    backend=self.backend_name,
                    query=query.query,
                )
            elif response.status_code != 200:
                raise SearchError(
                    f"HTTP {response.status_code}: {response.text}",
                    backend=self.backend_name,
                    query=query.query,
                )

            response_data = response.json()

            # Parse results
            results = self._parse_perplexica_response(response_data, query)
            search_time = asyncio.get_event_loop().time() - start_time

            # Perplexica doesn't provide total count, use results count
            total_found = len(response_data.get("sources", []))

            return SearchResults(
                query=query,
                results=results,
                total_found=total_found,
                search_time=search_time,
                backend=self.backend_name,
            )

        except httpx.TimeoutException:
            raise SearchTimeoutError(
                f"Perplexica search timed out after {self.timeout}s",
                backend=self.backend_name,
                query=query.query,
            )
        except httpx.ConnectError:
            raise SearchUnavailableError(
                f"Could not connect to Perplexica at {self.base_url}",
                backend=self.backend_name,
                query=query.query,
            )
        except httpx.RequestError as e:
            raise SearchError(
                f"Network error: {str(e)}", backend=self.backend_name, query=query.query
            )
        except SearchError:
            # Re-raise search errors
            raise
        except Exception as e:
            # Wrap unexpected errors
            raise SearchError(
                f"Unexpected error during Perplexica search: {str(e)}",
                backend=self.backend_name,
                query=query.query,
            )

    async def health_check(self) -> bool:
        """Check if Perplexica API is available and working."""
        try:
            session = await self._get_session()

            # Try to get available models as a health check
            response = await session.get(f"{self.base_url}/api/models", timeout=10)
            return response.status_code == 200

        except Exception:
            return False

    def supports_filters(self) -> List[str]:
        """Return list of supported filter types for Perplexica."""
        return [
            "date_range",
            "publication_type",
        ]  # Limited filtering compared to PubMed

    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Return rate limiting information for Perplexica."""
        return {
            "requests_per_second": None,  # Depends on local setup
            "requests_per_minute": None,
            "requests_per_hour": None,
            "burst_limit": None,
            "notes": "Rate limits depend on local Perplexica instance and underlying LLM providers",
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
