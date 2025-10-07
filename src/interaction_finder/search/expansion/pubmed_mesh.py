"""
PubMed MeSH-based query expansion using co-occurrence analysis.

This module provides query expansion by analyzing MeSH terms co-indexed with
seed search results. It uses NCBI E-utilities to discover related concepts
through the curated MeSH vocabulary.
"""

import asyncio
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Dict, List, Optional, Any, Tuple

import httpx

from ..base import QueryExpander, ExpandedQuery, ExpansionTerm, SearchQuery


class PubMedMeshExpander(QueryExpander):
    """Query expander using PubMed MeSH co-occurrence analysis."""

    def __init__(
        self,
        max_seed_results: int = 100,
        top_terms: int = 5,
        tree_filters: Optional[List[str]] = None,
        min_frequency: int = 3,
        email: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        """
        Initialize PubMed MeSH expander.

        Args:
            max_seed_results: Maximum PMIDs to analyze for MeSH co-occurrence
            top_terms: Number of top MeSH terms to include in expansion
            tree_filters: MeSH tree number prefixes to filter (e.g., ["C", "D"] for Diseases/Drugs)
            min_frequency: Minimum co-occurrence count to include a term
            email: Email for NCBI E-utilities (optional but recommended)
            api_key: NCBI API key (optional, increases rate limits)
            timeout: HTTP request timeout in seconds
        """
        self.max_seed_results = max_seed_results
        self.top_terms = top_terms
        self.tree_filters = tree_filters or ["C", "D"]  # Diseases and Drugs by default
        self.min_frequency = min_frequency
        self.email = email
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        # Rate limiting
        self._last_request_time = 0.0
        self._request_lock = asyncio.Lock()
        self.rate_limit = 10.0 if api_key else 3.0  # req/sec
        # Session for connection reuse
        self._session: Optional[httpx.AsyncClient] = None

    @property
    def expansion_method(self) -> str:
        """Name of the expansion method."""
        return "pubmed_mesh"

    def supports_context(self) -> List[str]:
        """Return list of supported context keys."""
        return [
            "max_seed_results",
            "top_terms",
            "tree_filters",
            "min_frequency",
        ]

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

    async def _esearch(self, query: str, max_results: int) -> List[str]:
        """
        Perform ESearch to get seed PMIDs.

        Args:
            query: Search query string
            max_results: Maximum number of PMIDs to retrieve

        Returns:
            List of PMIDs
        """
        await self._enforce_rate_limit()

        params = {
            "db": "pubmed",
            "term": query,
            "retmax": str(max_results),
            "retmode": "xml",
        }

        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key

        url = f"{self.base_url}/esearch.fcgi"
        session = await self._get_session()

        try:
            response = await session.get(url, params=params)
            if response.status_code != 200:
                return []

            root = ET.fromstring(response.text)
            id_list = root.find("IdList")
            if id_list is None:
                return []

            return [id_elem.text for id_elem in id_list.findall("Id") if id_elem.text]

        except Exception as e:
            print(f"Warning: ESearch failed: {e}")
            return []

    async def _efetch_mesh_terms(self, pmids: List[str]) -> List[str]:
        """
        Use EFetch to get MeSH terms from PubMed records.

        Args:
            pmids: List of PubMed IDs

        Returns:
            List of MeSH descriptor names
        """
        if not pmids:
            return []

        await self._enforce_rate_limit()

        params = {
            "db": "pubmed",
            "id": ",".join(pmids[:200]),  # Limit to 200 IDs per request
            "retmode": "xml",
            "rettype": "medline",
        }

        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key

        url = f"{self.base_url}/efetch.fcgi"
        session = await self._get_session()

        try:
            response = await session.get(url, params=params)
            if response.status_code != 200:
                print(f"Debug: EFetch HTTP error: {response.status_code}")
                return []

            root = ET.fromstring(response.text)
            mesh_terms = []

            # Parse MeSH headings from PubmedArticle records
            for article in root.findall(".//PubmedArticle"):
                mesh_heading_list = article.find(".//MeshHeadingList")
                if mesh_heading_list is not None:
                    for mesh_heading in mesh_heading_list.findall("MeshHeading"):
                        descriptor = mesh_heading.find("DescriptorName")
                        if descriptor is not None and descriptor.text:
                            mesh_terms.append(descriptor.text)

            return mesh_terms

        except Exception as e:
            print(f"Warning: EFetch for MeSH terms failed: {e}")
            return []

    async def expand_query(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> ExpandedQuery:
        """
        Expand query using PubMed MeSH co-occurrence analysis.

        Args:
            query: Original search query
            context: Optional context for expansion parameters

        Returns:
            ExpandedQuery with MeSH-derived expansion terms
        """
        if not query or not query.strip():
            return ExpandedQuery(
                original_query=query,
                expanded_terms=[],
                expansion_method=self.expansion_method,
                total_confidence=0.0,
            )

        try:
            # Apply context overrides
            max_seed = (
                context.get("max_seed_results", self.max_seed_results)
                if context
                else self.max_seed_results
            )
            top_n = (
                context.get("top_terms", self.top_terms) if context else self.top_terms
            )
            min_freq = (
                context.get("min_frequency", self.min_frequency)
                if context
                else self.min_frequency
            )

            # Step 1: Get seed PMIDs
            pmids = await self._esearch(query, max_seed)
            if not pmids:
                return ExpandedQuery(
                    original_query=query,
                    expanded_terms=[],
                    expansion_method=self.expansion_method,
                    total_confidence=0.0,
                )

            # Step 2: Fetch MeSH terms from PMIDs
            mesh_terms = await self._efetch_mesh_terms(pmids)
            if not mesh_terms:
                return ExpandedQuery(
                    original_query=query,
                    expanded_terms=[],
                    expansion_method=self.expansion_method,
                    total_confidence=0.0,
                )

            # Step 3: Count MeSH term frequency
            mesh_counts = Counter(mesh_terms)

            # Step 4: Filter out common demographic/organism terms
            excluded_terms = {
                "Humans",
                "Male",
                "Female",
                "Animals",
                "Adult",
                "Middle Aged",
                "Aged",
                "Young Adult",
                "Adolescent",
                "Child",
                "Infant",
                "Child, Preschool",
                "Aged, 80 and over",
                "Infant, Newborn",
                "Mice",
                "Rats",
                "Dogs",
                "Cats",
                "Rabbits",
                "Swine",
                "Cattle",
            }
            mesh_counts = {
                term: count
                for term, count in mesh_counts.items()
                if term not in excluded_terms
            }

            # Step 5: Build expansion terms with frequency counts
            expansion_terms = []

            # Sort by frequency
            sorted_terms = sorted(mesh_counts.items(), key=lambda x: -x[1])

            # Take top N terms with minimum frequency
            for term, count in sorted_terms[:top_n]:
                if count < min_freq:
                    continue

                # Calculate confidence based on frequency
                # Normalize by number of seed PMIDs
                confidence = min(1.0, count / len(pmids))

                expansion_terms.append(
                    ExpansionTerm(
                        term=term,
                        confidence=confidence,
                        source="pubmed_mesh",
                        category="mesh_coindex",
                    )
                )

            # Calculate total confidence as average
            total_confidence = (
                sum(t.confidence for t in expansion_terms) / len(expansion_terms)
                if expansion_terms
                else 0.0
            )

            return ExpandedQuery(
                original_query=query,
                expanded_terms=expansion_terms,
                expansion_method=self.expansion_method,
                total_confidence=total_confidence,
            )

        except Exception as e:
            print(f"PubMed MeSH expansion failed: {e}")
            return ExpandedQuery(
                original_query=query,
                expanded_terms=[],
                expansion_method=self.expansion_method,
                total_confidence=0.0,
            )

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


def create_pubmed_mesh_expander(
    max_seed_results: int = 100,
    top_terms: int = 5,
    tree_filters: Optional[List[str]] = None,
    min_frequency: int = 3,
    email: Optional[str] = None,
    api_key: Optional[str] = None,
) -> PubMedMeshExpander:
    """Factory function to create a PubMed MeSH query expander."""
    return PubMedMeshExpander(
        max_seed_results=max_seed_results,
        top_terms=top_terms,
        tree_filters=tree_filters,
        min_frequency=min_frequency,
        email=email,
        api_key=api_key,
    )
