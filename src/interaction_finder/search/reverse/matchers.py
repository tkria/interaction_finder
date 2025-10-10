"""
Resource matching logic for reverse search.

This module implements multi-strategy resource matching to identify when search
results correspond to known target resources. Strategies are tried in priority
order: PMID exact match → URL normalized match → DOI match (conditional) →
title similarity fallback.
"""

import logging
from typing import Any, Dict, List, Set, Optional, Tuple, TYPE_CHECKING

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import KnownResource, ResourceMatch, ReverseSearchConfig
from .utils import normalize_url
from interaction_finder.search.base import SearchResult, SearchResults
from .investigation_logger import _format_resource_ref

if TYPE_CHECKING:
    from .investigation_logger import InvestigationLogger

logger = logging.getLogger(__name__)


class ResourceMatcher:
    """
    Matches search results to known resources using multiple strategies.

    Strategies tried in order:
    1. PMID exact match (confidence 1.0)
    2. URL normalized match (confidence 1.0)
    3. DOI match - conditional on title similarity (confidence 1.0)
    4. Title similarity (confidence = similarity score)

    First successful match is returned for each result.
    """

    def __init__(
        self,
        config: ReverseSearchConfig,
        fetcher: Any,
        investigation_logger: Optional["InvestigationLogger"] = None,
    ):
        """
        Initialize resource matcher.

        Parameters:
            config: ReverseSearchConfig - Configuration including title similarity threshold
            fetcher: PageFetcher - For fetching DOIs when needed
            investigation_logger: Optional[InvestigationLogger] - Logger for investigation tracking
        """
        self.config = config
        self.fetcher = fetcher
        self.title_threshold = config.title_similarity_threshold
        self.inv_logger = investigation_logger
        self.last_matching_details: List[Dict[str, Any]] = []

    async def match_results(
        self,
        search_results: SearchResults,
        target_resources: Set[KnownResource],
        query_index: int,
    ) -> List[ResourceMatch]:
        """
        Match search results against target resources (async).

        Parameters:
            search_results: SearchResults - Results from search backend
            target_resources: Set[KnownResource] - Target resources to match against
            query_index: int - Index of query that produced these results

        Returns:
            List[ResourceMatch] - Successfully matched resources with match metadata

        Note:
            Each search result matches at most one target resource (first match wins).
            Strategies are tried in priority order: PMID → URL → DOI (conditional) → title.
            DOI fetching only occurs when title similarity ≥ threshold.
            Detailed matching information is stored in self.last_matching_details.
        """
        matches = []
        # Reset matching details for this batch
        self.last_matching_details = []
        # Build enhanced match_details for investigation logging
        enhanced_match_details = []

        for result_index, result in enumerate(search_results.results):
            # Extract compact result ID
            result_pmid = self._extract_pmid_from_metadata(result.metadata)
            result_id = f"PMID:{result_pmid}" if result_pmid else result.url
            # Initialize match detail for this result (existing detailed tracking)
            match_detail: Dict[str, Any] = {
                "result_url": result.url,
                "result_title": result.title or "",
                "strategies_attempted": {},
                "final_match": {},
            }
            matched_resource = None
            match_method = None
            confidence = 0.0
            # Strategy 1: PMID exact match
            pmid_match, pmid_method, pmid_confidence, pmid_detail = (
                self._try_pmid_match_with_details(result, target_resources)
            )
            match_detail["strategies_attempted"]["pmid"] = pmid_detail
            if pmid_match:
                matched_resource = pmid_match
                match_method = pmid_method
                confidence = pmid_confidence
                # PMID match found - skip other strategies
                match_detail["strategies_attempted"]["url"] = {"attempted": False}
                match_detail["strategies_attempted"]["doi"] = {"attempted": False}
                match_detail["strategies_attempted"]["title"] = {"attempted": False}
            else:
                # Strategy 2: URL exact match (normalized)
                url_match, url_method, url_confidence, url_detail = (
                    self._try_url_match_with_details(result, target_resources)
                )
                match_detail["strategies_attempted"]["url"] = url_detail
                if url_match:
                    matched_resource = url_match
                    match_method = url_method
                    confidence = url_confidence
                    # URL match found - skip remaining strategies
                    match_detail["strategies_attempted"]["doi"] = {"attempted": False}
                    match_detail["strategies_attempted"]["title"] = {"attempted": False}
                else:
                    # Strategy 3: DOI match (conditional)
                    (
                        doi_match,
                        doi_method,
                        doi_confidence,
                        doi_detail,
                    ) = await self._try_doi_match_with_details(result, target_resources)
                    match_detail["strategies_attempted"]["doi"] = doi_detail
                    if doi_match:
                        matched_resource = doi_match
                        match_method = doi_method
                        confidence = doi_confidence
                        # DOI match found - skip title strategy
                        match_detail["strategies_attempted"]["title"] = {
                            "attempted": False
                        }
                    else:
                        # Strategy 4: Title similarity (fallback)
                        title_match, title_method, title_confidence, title_detail = (
                            self._try_title_match_with_details(result, target_resources)
                        )
                        match_detail["strategies_attempted"]["title"] = title_detail
                        if title_match:
                            matched_resource = title_match
                            match_method = title_method
                            confidence = title_confidence
            # Record final match result
            match_detail["final_match"] = {
                "matched": matched_resource is not None,
                "resource_url": matched_resource.url if matched_resource else None,
                "method": match_method,
                "confidence": confidence,
            }
            # Store match detail (existing detailed tracking)
            self.last_matching_details.append(match_detail)
            # Build enhanced match detail for investigation logging
            enhanced_match_detail = {
                "result_index": result_index,
                "result_id": result_id,
                "matched": matched_resource is not None,
                "matched_resource": (
                    _format_resource_ref(matched_resource) if matched_resource else None
                ),
                "match_method": match_method,
                "confidence": confidence,
            }
            enhanced_match_details.append(enhanced_match_detail)
            # Record match if found
            if matched_resource:
                matches.append(
                    ResourceMatch(
                        resource=matched_resource,
                        search_result=result,
                        match_method=match_method,
                        confidence=confidence,
                        query_index=query_index,
                    )
                )
        # Calculate cumulative coverage
        cumulative_coverage = (
            len(matches) / len(target_resources) if target_resources else 0.0
        )
        # Log matching with enhanced details
        if self.inv_logger:
            await self.inv_logger.log_matching(
                query_index=query_index,
                results=search_results,
                targets=list(target_resources),
                matches=matches,
                details={"match_details": enhanced_match_details},
                coverage=cumulative_coverage,
            )

        return matches

    def _extract_pmid_from_metadata(self, metadata: Dict[str, Any]) -> Optional[str]:
        """
        Extract PMID from search result metadata (backend-agnostic).

        Tries multiple locations in priority order:
        1. metadata["pmid"] - direct (OpenAI, Perplexica)
        2. metadata["pubmed_summary"]["pmid"] - nested (PubMed)

        Parameters:
            metadata: Dict[str, Any] - Search result metadata

        Returns:
            Optional[str] - PMID if found, None otherwise
        """
        # Try direct location first (most common)
        pmid = metadata.get("pmid")
        if pmid:
            return str(pmid).strip()
        # Try nested PubMed location
        pubmed_summary = metadata.get("pubmed_summary")
        if pubmed_summary and isinstance(pubmed_summary, dict):
            pmid = pubmed_summary.get("pmid")
            if pmid:
                return str(pmid).strip()

        return None

    def _normalize_doi(self, doi: str) -> str:
        """
        Normalize DOI for comparison.

        Strips common prefixes and converts to lowercase.

        Parameters:
            doi: str - Raw DOI string

        Returns:
            str - Normalized DOI (lowercase, no prefix)

        Example:
            >>> _normalize_doi("doi:10.1234/ABC")
            "10.1234/abc"
            >>> _normalize_doi("https://doi.org/10.1234/ABC")
            "10.1234/abc"
        """
        normalized = doi.strip()
        # Remove prefixes (case-insensitive for "doi:")
        if normalized.lower().startswith("doi:"):
            normalized = normalized[4:]
        elif normalized.startswith("https://doi.org/"):
            normalized = normalized[16:]
        elif normalized.startswith("http://dx.doi.org/"):
            normalized = normalized[18:]
        # Lowercase and trim
        return normalized.strip().lower()

    def _try_pmid_match(
        self,
        result: SearchResult,
        target_resources: Set[KnownResource],
    ) -> Tuple[Optional[KnownResource], Optional[str], float]:
        """
        Try matching by PMID exact match.

        Returns:
            (matched_resource, match_method, confidence) or (None, None, 0.0)
        """
        # Extract PMID from result metadata using backend-agnostic helper
        result_pmid = self._extract_pmid_from_metadata(result.metadata)
        if not result_pmid:
            return None, None, 0.0
        # Search for matching resource
        for resource in target_resources:
            if resource.pmid and resource.pmid.strip() == result_pmid:
                return resource, "pmid", 1.0

        return None, None, 0.0

    def _try_pmid_match_with_details(
        self,
        result: SearchResult,
        target_resources: Set[KnownResource],
    ) -> Tuple[Optional[KnownResource], Optional[str], float, Dict[str, Any]]:
        """
        Try matching by PMID exact match with detailed tracking.

        Returns:
            (matched_resource, match_method, confidence, detail_dict)
        """
        # Extract PMID from result metadata
        result_pmid = self._extract_pmid_from_metadata(result.metadata)
        if not result_pmid:
            return None, None, 0.0, {"attempted": False}
        # Search for matching resource
        for resource in target_resources:
            if resource.pmid and resource.pmid.strip() == result_pmid:
                detail = {
                    "attempted": True,
                    "result_pmid": result_pmid,
                    "matched_resource": resource.url,
                    "confidence": 1.0,
                }
                return resource, "pmid", 1.0, detail
        # PMID present but no match
        detail = {
            "attempted": True,
            "result_pmid": result_pmid,
            "matched_resource": None,
            "confidence": 0.0,
        }
        return None, None, 0.0, detail

    def _try_url_match(
        self,
        result: SearchResult,
        target_resources: Set[KnownResource],
    ) -> Tuple[Optional[KnownResource], Optional[str], float]:
        """
        Try matching by URL (normalized).

        Returns:
            (matched_resource, match_method, confidence) or (None, None, 0.0)
        """
        # Normalize search result URL
        normalized_result_url = normalize_url(result.url)
        if not normalized_result_url:
            return None, None, 0.0
        # Search for matching resource
        for resource in target_resources:
            if resource.canonical_url == normalized_result_url:
                return resource, "url", 1.0

        return None, None, 0.0

    def _try_url_match_with_details(
        self,
        result: SearchResult,
        target_resources: Set[KnownResource],
    ) -> Tuple[Optional[KnownResource], Optional[str], float, Dict[str, Any]]:
        """
        Try matching by URL (normalized) with detailed tracking.

        Returns:
            (matched_resource, match_method, confidence, detail_dict)
        """
        # Normalize search result URL
        normalized_result_url = normalize_url(result.url)
        if not normalized_result_url:
            return None, None, 0.0, {"attempted": False}
        # Get sample of normalized target URLs (limit to 10 for log size)
        normalized_target_urls = [r.canonical_url for r in target_resources][:10]
        # Search for matching resource
        for resource in target_resources:
            if resource.canonical_url == normalized_result_url:
                detail = {
                    "attempted": True,
                    "normalized_result_url": normalized_result_url,
                    "normalized_target_urls": normalized_target_urls,
                    "matched_resource": resource.url,
                    "confidence": 1.0,
                }
                return resource, "url", 1.0, detail
        # URL present but no match
        detail = {
            "attempted": True,
            "normalized_result_url": normalized_result_url,
            "normalized_target_urls": normalized_target_urls,
            "matched_resource": None,
            "confidence": 0.0,
        }
        return None, None, 0.0, detail

    async def _try_doi_match(
        self,
        result: SearchResult,
        target_resources: Set[KnownResource],
    ) -> Tuple[Optional[KnownResource], Optional[str], float]:
        """
        Try matching by DOI (conditional on title similarity).

        Only fetches DOI if title similarity with best resource ≥ threshold.
        This avoids unnecessary network fetches for clearly unrelated results.

        Parameters:
            result: SearchResult - Search result to match
            target_resources: Set[KnownResource] - Resources to match against

        Returns:
            (matched_resource, match_method, confidence) or (None, None, 0.0)

        Note:
            Requires async because DOI fetching may need network request.
            Returns confidence 1.0 on successful DOI match.
        """
        # Phase 1: Check title similarity for all resources
        # Only fetch DOI if title suggests potential relevance
        if not result.title or not result.title.strip():
            return None, None, 0.0
        best_similarity = 0.0
        for resource in target_resources:
            resource_title = self._get_resource_title(resource)
            if resource_title:
                similarity = self._compute_title_similarity(
                    result.title, resource_title
                )
                best_similarity = max(best_similarity, similarity)
        # Skip DOI fetch if title similarity too low
        if best_similarity < self.title_threshold:
            return None, None, 0.0
        # Phase 2: Fetch DOI conditionally
        try:
            result_doi = await self.fetcher.get_doi(result.url)
        except Exception as e:
            logger.warning(
                "DOI fetch failed for %s: %s (will try title similarity)",
                result.url,
                str(e),
            )
            return None, None, 0.0
        # If no DOI available, return None
        if result_doi is None:
            return None, None, 0.0
        # Normalize fetched DOI
        normalized_result_doi = self._normalize_doi(result_doi)
        # Phase 3: Compare DOI with target resources
        for resource in target_resources:
            # Extract DOI from resource URL (if it's a DOI-based URL)
            if resource.url.startswith("https://doi.org/") or resource.url.startswith(
                "http://dx.doi.org/"
            ):
                resource_doi = self._normalize_doi(resource.url)
                if resource_doi == normalized_result_doi:
                    return resource, "doi", 1.0

        return None, None, 0.0

    async def _try_doi_match_with_details(
        self,
        result: SearchResult,
        target_resources: Set[KnownResource],
    ) -> Tuple[Optional[KnownResource], Optional[str], float, Dict[str, Any]]:
        """
        Try matching by DOI (conditional on title similarity) with detailed tracking.

        Returns:
            (matched_resource, match_method, confidence, detail_dict)
        """
        # Phase 1: Check title similarity threshold
        if not result.title or not result.title.strip():
            return None, None, 0.0, {"attempted": False}
        best_similarity = 0.0
        for resource in target_resources:
            resource_title = self._get_resource_title(resource)
            if resource_title:
                similarity = self._compute_title_similarity(
                    result.title, resource_title
                )
                best_similarity = max(best_similarity, similarity)
        # Check if title similarity meets threshold
        title_threshold_met = best_similarity >= self.title_threshold
        if not title_threshold_met:
            detail = {
                "attempted": False,
                "title_similarity_threshold_met": False,
            }
            return None, None, 0.0, detail
        # Phase 2: Fetch DOI
        try:
            result_doi = await self.fetcher.get_doi(result.url)
        except Exception as e:
            logger.warning(
                "DOI fetch failed for %s: %s (will try title similarity)",
                result.url,
                str(e),
            )
            detail = {
                "attempted": True,
                "fetched_doi": None,
                "title_similarity_threshold_met": True,
                "matched_resource": None,
                "confidence": 0.0,
            }
            return None, None, 0.0, detail
        # If no DOI available
        if result_doi is None:
            detail = {
                "attempted": True,
                "fetched_doi": None,
                "title_similarity_threshold_met": True,
                "matched_resource": None,
                "confidence": 0.0,
            }
            return None, None, 0.0, detail
        # Normalize fetched DOI
        normalized_result_doi = self._normalize_doi(result_doi)
        # Phase 3: Compare DOI with target resources
        for resource in target_resources:
            if resource.url.startswith("https://doi.org/") or resource.url.startswith(
                "http://dx.doi.org/"
            ):
                resource_doi = self._normalize_doi(resource.url)
                if resource_doi == normalized_result_doi:
                    detail = {
                        "attempted": True,
                        "fetched_doi": result_doi,
                        "title_similarity_threshold_met": True,
                        "matched_resource": resource.url,
                        "confidence": 1.0,
                    }
                    return resource, "doi", 1.0, detail
        # DOI fetched but no match
        detail = {
            "attempted": True,
            "fetched_doi": result_doi,
            "title_similarity_threshold_met": True,
            "matched_resource": None,
            "confidence": 0.0,
        }
        return None, None, 0.0, detail

    def _try_title_match(
        self,
        result: SearchResult,
        target_resources: Set[KnownResource],
    ) -> Tuple[Optional[KnownResource], Optional[str], float]:
        """
        Try matching by title similarity (fallback).

        Uses TF-IDF vectorization and cosine similarity.
        Returns match if similarity >= threshold.

        Returns:
            (matched_resource, match_method, confidence) or (None, None, 0.0)
        """
        if not result.title or not result.title.strip():
            return None, None, 0.0

        result_title = result.title.strip()
        best_match = None
        best_similarity = 0.0
        # Try matching against each resource
        for resource in target_resources:
            # Extract title from resource
            resource_title = self._get_resource_title(resource)
            if not resource_title:
                continue
            # Compute similarity
            similarity = self._compute_title_similarity(result_title, resource_title)
            # Update best match if above threshold
            if similarity >= self.title_threshold and similarity > best_similarity:
                best_match = resource
                best_similarity = similarity

        if best_match:
            return best_match, "title_similarity", best_similarity

        return None, None, 0.0

    def _try_title_match_with_details(
        self,
        result: SearchResult,
        target_resources: Set[KnownResource],
    ) -> Tuple[Optional[KnownResource], Optional[str], float, Dict[str, Any]]:
        """
        Try matching by title similarity (fallback) with detailed tracking.

        Returns:
            (matched_resource, match_method, confidence, detail_dict)
        """
        if not result.title or not result.title.strip():
            return None, None, 0.0, {"attempted": False}

        result_title = result.title.strip()
        best_match = None
        best_similarity = 0.0
        # Compute similarities for all resources with titles
        similarities = []
        for resource in target_resources:
            resource_title = self._get_resource_title(resource)
            if not resource_title:
                continue
            similarity = self._compute_title_similarity(result_title, resource_title)
            similarities.append({"resource_url": resource.url, "score": similarity})
            # Update best match if above threshold
            if similarity >= self.title_threshold and similarity > best_similarity:
                best_match = resource
                best_similarity = similarity
        # Sort by score descending and take top 5
        similarities.sort(key=lambda x: x["score"], reverse=True)
        top_similarities = similarities[:5]
        # Build detail
        detail = {
            "attempted": True,
            "similarities": top_similarities,
            "best_score": best_similarity,
            "threshold": self.title_threshold,
            "matched_resource": best_match.url if best_match else None,
            "confidence": best_similarity,
        }
        if best_match:
            return best_match, "title_similarity", best_similarity, detail

        return None, None, 0.0, detail

    def _get_resource_title(self, resource: KnownResource) -> Optional[str]:
        """
        Get title for resource from metadata or hint fields.

        Parameters:
            resource: KnownResource - Resource to get title for

        Returns:
            Optional[str] - Title if available, None otherwise

        Note:
            Currently checks hint_fields['title']. In future, may fetch
            metadata from PMID or URL if title not present.
        """
        # Check hint fields for title
        title = resource.hint_fields.get("title")
        if title:
            return str(title).strip()
        # TODO: In future, could fetch title from PMID or URL
        # For now, return None if not in hint fields
        return None

    def _compute_title_similarity(self, title1: str, title2: str) -> float:
        """
        Compute cosine similarity between two titles using TF-IDF.

        Parameters:
            title1: str - First title
            title2: str - Second title

        Returns:
            float - Cosine similarity score [0.0, 1.0]

        Note:
            Returns 0.0 if vectorization fails (e.g., no common words).
            Clips result to [0.0, 1.0] to handle floating point precision issues.
        """
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([title1, title2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            # Clip to [0.0, 1.0] to handle floating point precision issues
            return float(max(0.0, min(1.0, similarity)))
        except (ValueError, IndexError):
            # Vectorization failed (no features extracted)
            return 0.0
