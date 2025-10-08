"""
Query generation from target resources using metadata-first strategy.

This module implements QueryGenerator, which transforms KnownResource objects
into search queries using keyword extraction from PubMed metadata (PMID → title/abstract)
or full content (URL → PageFetcher → Markdown). Supports clustering-based
refinement for generating shared queries across similar resources.

Order: helper dataclasses → QueryGenerator class → helper methods
"""

from typing import Any, Dict, List, Optional
import asyncio
import httpx
from dataclasses import dataclass

from .models import KnownResource, ReverseSearchConfig
from .keyword_extractors import create_extractor


# ==============================================================================
# Helper Dataclasses
# ==============================================================================


@dataclass
class ResourceContent:
    """
    Content extracted from a resource for keyword extraction.

    Fields:
        resource: KnownResource - The resource this content belongs to
        text: str - Title + abstract or full content for keyword extraction
        source: Literal - Content source ("metadata", "content", "hint_fields")
    """

    resource: KnownResource
    text: str
    source: str  # "metadata", "content", or "hint_fields"


# ==============================================================================
# QueryGenerator Class
# ==============================================================================


class QueryGenerator:
    """
    Generates search queries from target resources.

    Strategies:
    1. Metadata-first: PMID → PubMed ESummary → title/abstract → keywords
    2. Content fallback: URL → PageFetcher → Markdown → keywords
    3. Clustering: Group similar unfound resources, generate shared queries

    All strategies use domain-agnostic keyword extraction (YAKE, RAKE, TF-IDF).

    Example:
        >>> config = ReverseSearchConfig(
        ...     keyword_extractor="yake",
        ...     keywords_per_query=7,
        ...     use_hint_fields=True
        ... )
        >>> generator = QueryGenerator(config)
        >>> resources = [
        ...     KnownResource(pmid="123", url="https://pubmed.ncbi.nlm.nih.gov/123/"),
        ...     KnownResource(pmid="456", url="https://pubmed.ncbi.nlm.nih.gov/456/"),
        ... ]
        >>> queries = await generator.generate_initial_queries(resources)
    """

    def __init__(
        self,
        config: ReverseSearchConfig,
        fetcher: Optional[Any] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        """
        Initialize query generator.

        Parameters:
            config: ReverseSearchConfig - Configuration
            fetcher: Optional[PageFetcher] - For content fetching (created if None)
            http_client: Optional[httpx.AsyncClient] - For PubMed API (created if None)
        """
        self.config = config
        self.fetcher = fetcher
        self.http_client = http_client
        # Create keyword extractor
        self.extractor = create_extractor(config.keyword_extractor)

    async def generate_initial_queries(
        self,
        resources: List[KnownResource],
    ) -> List[str]:
        """
        Generate initial queries for target resources.

        Uses clustering if enabled and sufficient resources, otherwise
        generates one query per resource.

        Parameters:
            resources: List[KnownResource] - Target resources

        Returns:
            List[str] - Generated queries

        Raises:
            QueryGenerationError: If query generation fails
        """
        if not resources:
            return []

        # Decide strategy based on config and resource count
        if (
            self.config.enable_clustering
            and len(resources) >= self.config.min_cluster_size
        ):
            return await self.generate_initial_queries_clustered(resources)
        else:
            return await self.generate_initial_queries_individual(resources)

    async def generate_initial_queries_individual(
        self,
        resources: List[KnownResource],
    ) -> List[str]:
        """
        Generate one query per resource (no clustering).

        Parameters:
            resources: List[KnownResource] - Target resources

        Returns:
            List[str] - One query per resource
        """
        # Fetch content for all resources
        contents = await self._fetch_resource_contents(resources)

        queries = []
        for content in contents:
            # Extract keywords
            keywords = self.extractor.extract(
                content.text, self.config.keywords_per_query
            )

            # Add hint fields if enabled
            if self.config.use_hint_fields:
                hint_terms = self._extract_hint_terms([content.resource])
                keywords = hint_terms + keywords

            # Construct query
            query = self._construct_query(keywords)
            if query:  # Only add non-empty queries
                queries.append(query)

        return queries

    async def generate_initial_queries_clustered(
        self,
        resources: List[KnownResource],
    ) -> List[str]:
        """
        Generate queries by clustering resources (more efficient).

        Parameters:
            resources: List[KnownResource] - Target resources

        Returns:
            List[str] - One query per cluster (fewer than resources)
        """
        import numpy as np
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans

        # Fetch content for all resources
        contents = await self._fetch_resource_contents(resources)

        # Compute embeddings
        model = SentenceTransformer("all-MiniLM-L6-v2")  # Fast, general-purpose
        texts = [c.text for c in contents]
        embeddings = model.encode(texts)

        # Cluster resources
        n_clusters = min(self.config.target_clusters, len(resources) // 2)
        if n_clusters < 2:
            # Not enough resources to cluster effectively
            return await self.generate_initial_queries_individual(resources)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)

        # Generate query for each cluster
        queries = []
        # Use actual number of unique clusters (may be less than n_clusters if data is identical)
        unique_labels = set(labels)
        for cluster_id in unique_labels:
            # Get resources in this cluster
            cluster_indices = [
                i for i, label in enumerate(labels) if label == cluster_id
            ]
            # Skip empty clusters
            if not cluster_indices:
                continue

            cluster_resources = [contents[i].resource for i in cluster_indices]

            # Find representative (closest to centroid)
            cluster_embeddings = embeddings[labels == cluster_id]
            centroid = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            representative_idx = cluster_indices[np.argmin(distances)]
            representative_content = contents[representative_idx]

            # Extract keywords from representative
            keywords = self.extractor.extract(
                representative_content.text, self.config.keywords_per_query
            )

            # Add hint fields from cluster resources
            if self.config.use_hint_fields:
                hint_terms = self._extract_hint_terms(cluster_resources)
                keywords = hint_terms + keywords

            # Construct query
            query = self._construct_query(keywords)
            if query:  # Only add non-empty queries
                queries.append(query)

        return queries

    async def generate_refinement_queries(
        self,
        unfound_resources: List[KnownResource],
        previous_queries: List[str],
    ) -> List[str]:
        """
        Generate refinement queries for remaining unfound resources.

        Parameters:
            unfound_resources: List[KnownResource] - Resources not yet found
            previous_queries: List[str] - Queries already executed

        Returns:
            List[str] - New queries (excluding duplicates of previous)
        """
        if not unfound_resources:
            return []

        # Generate queries using same logic as initial
        new_queries = await self.generate_initial_queries(unfound_resources)

        # Filter out duplicates of previous queries
        previous_set = set(previous_queries)
        unique_queries = [q for q in new_queries if q not in previous_set]

        return unique_queries

    # ==============================================================================
    # Helper Methods
    # ==============================================================================

    async def _fetch_resource_contents(
        self,
        resources: List[KnownResource],
    ) -> List[ResourceContent]:
        """
        Fetch content for resources (metadata-first with content fallback).

        Parameters:
            resources: List[KnownResource] - Resources to fetch content for

        Returns:
            List[ResourceContent] - Content for each resource
        """
        contents = []

        # Separate resources by type
        pmid_resources = [r for r in resources if r.pmid]
        url_only_resources = [r for r in resources if not r.pmid]

        # Fetch PMID metadata in batches
        if pmid_resources:
            metadata_map = await self._fetch_pmid_metadata_batch(
                [r.pmid for r in pmid_resources]
            )

            for resource in pmid_resources:
                metadata = metadata_map.get(resource.pmid)
                if metadata:
                    # Successfully fetched metadata
                    text = (
                        f"{metadata.get('title', '')}. {metadata.get('abstract', '')}"
                    )
                    contents.append(
                        ResourceContent(
                            resource=resource,
                            text=text,
                            source="metadata",
                        )
                    )
                else:
                    # Metadata fetch failed, fall back to content
                    url_only_resources.append(resource)

        # Fetch content for URL-only resources (or fallback)
        if url_only_resources:
            # Ensure we have a PageFetcher
            if not self.fetcher:
                from interaction_finder.fetcher.web_client import PageFetcher
                from interaction_finder.settings import IfetcherConfig

                cfg = IfetcherConfig.from_path()
                self.fetcher = PageFetcher(cfg, show_status=False)

            urls = [r.url for r in url_only_resources]
            documents = await self.fetcher.fetch_documents(urls)

            for resource in url_only_resources:
                doc = documents.get(resource.url)
                if doc and doc.content_markdown:
                    contents.append(
                        ResourceContent(
                            resource=resource,
                            text=doc.content_markdown,
                            source="content",
                        )
                    )
                else:
                    # Content fetch also failed, use hint fields as last resort
                    text = " ".join(str(v) for v in resource.hint_fields.values())
                    contents.append(
                        ResourceContent(
                            resource=resource,
                            text=text or "unknown",
                            source="hint_fields",
                        )
                    )

        return contents

    async def _fetch_pmid_metadata_batch(
        self,
        pmids: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Batch fetch PMID metadata from PubMed ESummary API.

        Parameters:
            pmids: List[str] - PMIDs to fetch (up to 200 per batch)

        Returns:
            Dict[pmid, metadata] - Metadata for each PMID (empty dict if fetch failed)
        """
        if not pmids:
            return {}

        # Ensure we have an HTTP client
        if not self.http_client:
            self.http_client = httpx.AsyncClient(timeout=30.0)

        # Batch into chunks of batch_pmid_fetch_size (ESummary limit)
        metadata_map = {}
        for i in range(0, len(pmids), self.config.batch_pmid_fetch_size):
            batch = pmids[i : i + self.config.batch_pmid_fetch_size]

            try:
                # Construct ESummary request
                url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                params = {
                    "db": "pubmed",
                    "id": ",".join(batch),
                    "retmode": "json",
                }

                response = await self.http_client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                # Extract metadata for each PMID
                result = data.get("result", {})
                for pmid in batch:
                    doc = result.get(pmid)
                    if doc and isinstance(doc, dict):
                        # Extract title and abstract (if available)
                        title = doc.get("title", "")
                        # Abstract is in 'abstracttext' or may not exist
                        abstract = ""
                        if "abstracttext" in doc:
                            abstract = doc["abstracttext"]

                        metadata_map[pmid] = {
                            "title": title,
                            "abstract": abstract,
                        }

            except Exception as e:
                # Log error but continue (will fall back to content fetching)
                print(f"Warning: PMID batch fetch failed for {len(batch)} PMIDs: {e}")

            # Rate limiting: sleep between batches (safe default ~3 req/s)
            await asyncio.sleep(0.34)

        return metadata_map

    def _extract_hint_terms(self, resources: List[KnownResource]) -> List[str]:
        """
        Extract unique hint field values from resources.

        Parameters:
            resources: List[KnownResource] - Resources to extract hints from

        Returns:
            List[str] - Unique hint terms
        """
        hint_terms = set()
        for resource in resources:
            for value in resource.hint_fields.values():
                if value:
                    hint_terms.add(str(value).strip())
        return list(hint_terms)

    def _construct_query(self, keywords: List[str]) -> str:
        """
        Construct boolean query from keywords.

        Uses OR logic for broad coverage.

        Parameters:
            keywords: List[str] - Keywords to include

        Returns:
            str - Query string
        """
        if not keywords:
            return ""

        # Quote keywords to handle multi-word terms
        quoted_keywords = [f'"{kw}"' for kw in keywords if kw]
        return " OR ".join(quoted_keywords)
