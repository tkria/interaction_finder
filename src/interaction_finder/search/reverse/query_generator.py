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

from .models import (
    KnownResource,
    ReverseSearchConfig,
    QueryConstructionContext,
)
from .keyword_extractors import create_extractor
from .query_constructors import create_constructor
from .investigation_logger import _format_resource_ref


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
        source: Literal - Content source ("metadata", "content")
    """

    resource: KnownResource
    text: str
    source: str  # "metadata" or "content"


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
        ...     keywords_per_query=7
        ... )
        >>> generator = QueryGenerator(config, backend_name="pubmed")
        >>> resources = [
        ...     KnownResource(pmid="123", url="https://pubmed.ncbi.nlm.nih.gov/123/"),
        ...     KnownResource(pmid="456", url="https://pubmed.ncbi.nlm.nih.gov/456/"),
        ... ]
        >>> queries = await generator.generate_initial_queries(resources)
    """

    def __init__(
        self,
        config: ReverseSearchConfig,
        backend_name: str,
        fetcher: Optional[Any] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        console: Optional[Any] = None,
        investigation_logger: Optional[Any] = None,
    ):
        """
        Initialize query generator.

        Parameters:
            config: ReverseSearchConfig - Configuration
            backend_name: str - Search backend name for query adaptation
            fetcher: Optional[PageFetcher] - For content fetching (created if None)
            http_client: Optional[httpx.AsyncClient] - For PubMed API (created if None)
            console: Optional[Console] - Rich console for verbose output
            investigation_logger: Optional[InvestigationLogger] - For logging query generation stages
        """
        self.config = config
        self.backend_name = backend_name
        self.fetcher = fetcher
        self.http_client = http_client
        self.console = console
        self.inv_logger = investigation_logger
        # Create keyword extractor
        self.extractor = create_extractor(config.keyword_extractor)

        # Create query constructor based on config.query_constructor
        if config.query_constructor == "llm":
            # LLM constructor with fallback to direct on errors
            llm_config = config.llm_query_config
            enable_fallback = config.query_construction_config.get(
                "enable_fallback", True
            )
            self.constructor = create_constructor(
                "llm",
                model=llm_config.get("model", "openai:gpt-4o-mini"),
                temperature=llm_config.get("temperature", 0.7),
                backend_specific=llm_config.get("backend_specific_syntax", True),
                enable_fallback=enable_fallback,
                console=console,
            )
        else:
            # Direct constructor (simple keyword concatenation)
            self.constructor = create_constructor("direct")

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
        Generate one query per resource using two-stage pipeline (no clustering).

        Two-stage flow:
            Stage 1: Keyword Extraction - Extract keywords from resource content
            Stage 2: Query Construction - Build search queries from extracted keywords

        Parameters:
            resources: List[KnownResource] - Target resources

        Returns:
            List[str] - One query per resource
        """
        # Fetch content for all resources with enhanced details
        contents, contents_dict = await self._fetch_resource_contents(resources)

        # Log content fetch if investigation logger is available
        if self.inv_logger:
            await self.inv_logger.log_content_fetch(resources, contents_dict)

        # Log clustering disabled (individual mode)
        if self.inv_logger:
            await self.inv_logger.log_clustering(
                enabled=False,
                resources=resources,
                clusters=None,
                representatives=None,
                embeddings=None,
                centroids=None,
            )

        # Log content sources if verbose logging is available
        self._log_content_sources(contents)

        queries = []
        query_index = 0
        for content in contents:
            # Stage 1: Keyword Extraction
            keywords_with_scores = []
            keywords = []

            # Statistical extractors (sync) - returns list or tuples with scores
            keywords_raw = self.extractor.extract(
                content.text, self.config.keywords_per_query
            )
            # Convert to keyword/score dicts
            for kw in keywords_raw:
                if isinstance(kw, tuple):
                    # (keyword, score) or (score, keyword) depending on extractor
                    # YAKE: (keyword, score), RAKE: (score, keyword)
                    if self.config.keyword_extractor == "rake":
                        keywords_with_scores.append({"keyword": kw[1], "score": kw[0]})
                    else:
                        keywords_with_scores.append({"keyword": kw[0], "score": kw[1]})
                else:
                    # Plain string
                    keywords_with_scores.append({"keyword": kw, "score": None})
            # Extract keywords list
            keywords = [kw["keyword"] for kw in keywords_with_scores]

            # Stage 2: Query Construction
            # Build QueryConstructionContext with ALL available info
            keyword_scores = [kw["score"] for kw in keywords_with_scores]
            context = QueryConstructionContext(
                keywords=keywords,
                keyword_scores=keyword_scores
                if any(s is not None for s in keyword_scores)
                else None,
                backend=self.backend_name,
                resource_content=content.text,  # FULL CONTENT for LLM constructors
                extractor_used=self.config.keyword_extractor,
            )

            # Construct query using constructor
            query = await self.constructor.construct(context)

            # Log query generation if investigation logger is available
            if self.inv_logger and query:
                # Format input resource ID
                input_resources = [_format_resource_ref(content.resource)]
                await self.inv_logger.log_query_generation(
                    query_index=query_index,
                    query_type="initial",
                    extractor_type=self.config.keyword_extractor,
                    keywords=keywords_with_scores,
                    final_query=query,
                    cluster_id=None,
                    resource_count=1,
                    input_resources=input_resources,
                    cumulative_coverage=0.0,  # Will be updated by searcher
                )
                query_index += 1

            if query:  # Only add non-empty queries
                queries.append(query)

        return queries

    async def generate_initial_queries_clustered(
        self,
        resources: List[KnownResource],
    ) -> List[str]:
        """
        Generate queries by clustering resources using two-stage pipeline (more efficient).

        Two-stage flow:
            Stage 1: Keyword Extraction - Extract keywords from cluster representative
            Stage 2: Query Construction - Build search queries from extracted keywords

        Parameters:
            resources: List[KnownResource] - Target resources

        Returns:
            List[str] - One query per cluster (fewer than resources)
        """
        import numpy as np
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans

        # Fetch content for all resources with enhanced details
        contents, contents_dict = await self._fetch_resource_contents(resources)

        # Log content fetch if investigation logger is available
        if self.inv_logger:
            await self.inv_logger.log_content_fetch(resources, contents_dict)

        # Log content sources if verbose logging is available
        self._log_content_sources(contents)

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

        # Build cluster assignments for logging
        clusters = [
            [i for i, label in enumerate(labels) if label == cid]
            for cid in range(n_clusters)
        ]
        # Find representative for each cluster
        representatives = []
        for cluster_id in range(n_clusters):
            cluster_indices = clusters[cluster_id]
            if cluster_indices:
                cluster_embeddings = embeddings[labels == cluster_id]
                centroid = kmeans.cluster_centers_[cluster_id]
                distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
                representatives.append(cluster_indices[np.argmin(distances)])
            else:
                representatives.append(None)

        # Log clustering if investigation logger is available
        if self.inv_logger:
            await self.inv_logger.log_clustering(
                enabled=True,
                resources=resources,
                clusters=clusters,
                representatives=representatives,
                embeddings=embeddings,
                centroids=kmeans.cluster_centers_,
            )

        # Generate query for each cluster
        queries = []
        query_index = 0
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

            # Stage 1: Keyword Extraction from representative
            keywords_with_scores = []
            keywords = []

            # Statistical extractors (sync) - returns list or tuples with scores
            keywords_raw = self.extractor.extract(
                representative_content.text, self.config.keywords_per_query
            )
            # Convert to keyword/score dicts
            for kw in keywords_raw:
                if isinstance(kw, tuple):
                    # (keyword, score) or (score, keyword) depending on extractor
                    # YAKE: (keyword, score), RAKE: (score, keyword)
                    if self.config.keyword_extractor == "rake":
                        keywords_with_scores.append({"keyword": kw[1], "score": kw[0]})
                    else:
                        keywords_with_scores.append({"keyword": kw[0], "score": kw[1]})
                else:
                    # Plain string
                    keywords_with_scores.append({"keyword": kw, "score": None})
            # Extract keywords list
            keywords = [kw["keyword"] for kw in keywords_with_scores]

            # Stage 2: Query Construction
            # Build QueryConstructionContext with ALL available info
            keyword_scores = [kw["score"] for kw in keywords_with_scores]
            context = QueryConstructionContext(
                keywords=keywords,
                keyword_scores=keyword_scores
                if any(s is not None for s in keyword_scores)
                else None,
                backend=self.backend_name,
                resource_content=representative_content.text,  # FULL CONTENT for LLM constructors
                extractor_used=self.config.keyword_extractor,
            )

            # Construct query using constructor
            query = await self.constructor.construct(context)

            # Log query generation if investigation logger is available
            if self.inv_logger and query:
                # Format input resource IDs
                input_resources = [_format_resource_ref(r) for r in cluster_resources]
                await self.inv_logger.log_query_generation(
                    query_index=query_index,
                    query_type="initial",
                    extractor_type=self.config.keyword_extractor,
                    keywords=keywords_with_scores,
                    final_query=query,
                    cluster_id=cluster_id,
                    resource_count=len(cluster_resources),
                    input_resources=input_resources,
                    cumulative_coverage=0.0,  # Will be updated by searcher
                )
                query_index += 1

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
    ) -> tuple[List[ResourceContent], Dict[str, Any]]:
        """
        Fetch content for resources (metadata-first with content fallback).

        Parameters:
            resources: List[KnownResource] - Resources to fetch content for

        Returns:
            Tuple of (List[ResourceContent], Dict[url, details]) - Content for each resource
            and detailed fetch information for logging
        """
        contents = []
        # Track per-resource details for enhanced logging
        contents_dict: Dict[str, Any] = {}

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
                    title = metadata.get("title", "")[:100]
                    contents.append(
                        ResourceContent(
                            resource=resource,
                            text=text,
                            source="metadata",
                        )
                    )
                    # Log enhanced details
                    contents_dict[resource.url] = {
                        "source": "metadata",
                        "title": title,
                        "content_length": len(text),
                        "success": True,
                        "error": None,
                    }
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
            # Enable progress if fetcher supports it (during query generation phase)
            documents = await self.fetcher.fetch_documents(urls, progress=True)

            for resource in url_only_resources:
                doc = documents.get(resource.url)
                if doc and doc.content_markdown:
                    # Extract first line as title approximation
                    title = doc.content_markdown.split("\n")[0].strip()[:100]
                    contents.append(
                        ResourceContent(
                            resource=resource,
                            text=doc.content_markdown,
                            source="content",
                        )
                    )
                    # Log enhanced details
                    contents_dict[resource.url] = {
                        "source": "content",
                        "title": title,
                        "content_length": len(doc.content_markdown),
                        "success": True,
                        "error": None,
                    }
                else:
                    # Content fetch failed, skip this resource (no content available)
                    contents_dict[resource.url] = {
                        "source": None,
                        "title": None,
                        "content_length": 0,
                        "success": False,
                        "error": "No content sources available (metadata and content fetch both failed)",
                    }

        return contents, contents_dict

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

    def _log_content_sources(self, contents: List[ResourceContent]) -> None:
        """
        Log content source breakdown for debugging/transparency.

        Parameters:
            contents: List[ResourceContent] - Fetched contents with sources
        """
        if not self.console:
            return

        # Count sources
        source_counts = {"metadata": 0, "content": 0}
        for content in contents:
            source_counts[content.source] = source_counts.get(content.source, 0) + 1

        # Display summary
        parts = []
        if source_counts["metadata"] > 0:
            parts.append(f"{source_counts['metadata']} from PMID metadata")
        if source_counts["content"] > 0:
            parts.append(f"{source_counts['content']} from URL content")

        if parts:
            self.console.print(f"  [dim]Content sources: {', '.join(parts)}[/dim]")
