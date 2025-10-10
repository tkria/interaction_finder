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
        console: Optional[Any] = None,
        investigation_logger: Optional[Any] = None,
    ):
        """
        Initialize query generator.

        Parameters:
            config: ReverseSearchConfig - Configuration
            fetcher: Optional[PageFetcher] - For content fetching (created if None)
            http_client: Optional[httpx.AsyncClient] - For PubMed API (created if None)
            console: Optional[Console] - Rich console for verbose output
            investigation_logger: Optional[InvestigationLogger] - For logging query generation stages
        """
        self.config = config
        self.fetcher = fetcher
        self.http_client = http_client
        self.console = console
        self.inv_logger = investigation_logger
        # Create keyword extractor with LLM-specific config if needed
        if config.keyword_extractor == "llm":
            llm_config = config.llm_query_config
            self.extractor = create_extractor(
                "llm",
                model=llm_config.get("model", "openai:gpt-4o-mini"),
                temperature=llm_config.get("temperature", 0.7),
                backend_specific=llm_config.get("backend_specific_syntax", True),
                console=console,
            )
        else:
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

        # Log LLM query generation if console available
        if self.console and self.config.keyword_extractor == "llm":
            llm_config = self.config.llm_query_config
            self.console.print(
                f"  [dim]Generating LLM queries (model: {llm_config.get('model', 'openai:gpt-4o-mini')})[/dim]"
            )

        queries = []
        query_index = 0
        for content in contents:
            # Extract hint terms separately first (for statistical extractors)
            hint_terms = []
            if self.config.use_hint_fields and self.config.keyword_extractor != "llm":
                hint_terms = self._extract_hint_terms([content.resource])
            # Extract keywords/queries (use async for LLM, sync for others)
            keywords_with_scores = []
            if self.config.keyword_extractor == "llm" and hasattr(
                self.extractor, "extract_async"
            ):
                # LLM extraction with hint fields
                hint_fields = (
                    content.resource.hint_fields
                    if self.config.use_hint_fields
                    else None
                )
                keywords = await self.extractor.extract_async(
                    content.text,
                    self.config.keywords_per_query,
                    hint_fields=hint_fields,
                )
                # LLM extractor returns strings, wrap with None scores for logging
                keywords_with_scores = [
                    {"keyword": kw, "score": None} for kw in keywords
                ]
            else:
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
                            keywords_with_scores.append(
                                {"keyword": kw[1], "score": kw[0]}
                            )
                        else:
                            keywords_with_scores.append(
                                {"keyword": kw[0], "score": kw[1]}
                            )
                    else:
                        # Plain string
                        keywords_with_scores.append({"keyword": kw, "score": None})
                # Build final keywords list for query construction (just strings)
                keywords = [kw["keyword"] for kw in keywords_with_scores] + hint_terms
            # For LLM, keywords list already constructed
            if self.config.keyword_extractor == "llm":
                keywords = [kw["keyword"] for kw in keywords_with_scores]
            # Construct query
            query = self._construct_query(keywords)

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
                    hint_terms=hint_terms,
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
        Generate queries by clustering resources (more efficient).

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

            # Extract hint terms separately first (for statistical extractors)
            hint_terms = []
            if self.config.use_hint_fields and self.config.keyword_extractor != "llm":
                hint_terms = self._extract_hint_terms(cluster_resources)
            # Extract keywords/queries from representative (use async for LLM, sync for others)
            keywords_with_scores = []
            if self.config.keyword_extractor == "llm" and hasattr(
                self.extractor, "extract_async"
            ):
                # LLM extraction with hint fields from entire cluster
                hint_fields = None
                if self.config.use_hint_fields:
                    # Aggregate hint fields from all cluster resources
                    hint_fields = {}
                    for resource in cluster_resources:
                        for key, value in resource.hint_fields.items():
                            if key not in hint_fields:
                                hint_fields[key] = []
                            if value and value not in hint_fields[key]:
                                hint_fields[key].append(value)
                    # Flatten lists to strings (first value for simplicity)
                    hint_fields = {k: v[0] if v else "" for k, v in hint_fields.items()}

                keywords = await self.extractor.extract_async(
                    representative_content.text,
                    self.config.keywords_per_query,
                    hint_fields=hint_fields,
                )
                # LLM extractor returns strings, wrap with None scores for logging
                keywords_with_scores = [
                    {"keyword": kw, "score": None} for kw in keywords
                ]
            else:
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
                            keywords_with_scores.append(
                                {"keyword": kw[1], "score": kw[0]}
                            )
                        else:
                            keywords_with_scores.append(
                                {"keyword": kw[0], "score": kw[1]}
                            )
                    else:
                        # Plain string
                        keywords_with_scores.append({"keyword": kw, "score": None})
                # Build final keywords list for query construction (just strings)
                keywords = [kw["keyword"] for kw in keywords_with_scores] + hint_terms
            # For LLM, keywords list already constructed
            if self.config.keyword_extractor == "llm":
                keywords = [kw["keyword"] for kw in keywords_with_scores]
            # Cluster information logged to investigation log only (not console)
            # Console remains focused on progress and status
            # Construct query
            query = self._construct_query(keywords)

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
                    hint_terms=hint_terms,
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
                    # Content fetch also failed, use hint fields as last resort
                    text = " ".join(str(v) for v in resource.hint_fields.values())
                    # Use first hint field value as title approximation
                    title = (
                        list(resource.hint_fields.values())[0]
                        if resource.hint_fields
                        else ""
                    )[:100]
                    contents.append(
                        ResourceContent(
                            resource=resource,
                            text=text or "unknown",
                            source="hint_fields",
                        )
                    )
                    # Log enhanced details
                    contents_dict[resource.url] = {
                        "source": "hint_fields",
                        "title": title,
                        "content_length": len(text or "unknown"),
                        "success": bool(text),
                        "error": "No content sources available" if not text else None,
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

        Uses OR logic for broad coverage. Handles both LLM complete queries
        (with field tags or boolean operators) and keyword lists from
        statistical extractors.

        Parameters:
            keywords: List[str] - Keywords to include (or complete queries from LLM)

        Returns:
            str - Query string
        """
        if not keywords:
            return ""

        # If keywords look like complete queries (contain field tags or boolean ops), use first one
        # LLM may return queries like: '"BRCA1"[Title] AND "breast cancer"[Abstract]'
        if any("[" in kw or " AND " in kw or " OR " in kw for kw in keywords):
            return keywords[0]  # LLM-generated complete query

        # Otherwise, construct boolean OR query from keywords (statistical extractors)
        quoted_keywords = [f'"{kw}"' for kw in keywords if kw]
        return " OR ".join(quoted_keywords)

    def _log_content_sources(self, contents: List[ResourceContent]) -> None:
        """
        Log content source breakdown for debugging/transparency.

        Parameters:
            contents: List[ResourceContent] - Fetched contents with sources
        """
        if not self.console:
            return

        # Count sources
        source_counts = {"metadata": 0, "content": 0, "hint_fields": 0}
        for content in contents:
            source_counts[content.source] = source_counts.get(content.source, 0) + 1

        # Display summary
        parts = []
        if source_counts["metadata"] > 0:
            parts.append(f"{source_counts['metadata']} from PMID metadata")
        if source_counts["content"] > 0:
            parts.append(f"{source_counts['content']} from URL content")
        if source_counts["hint_fields"] > 0:
            parts.append(f"{source_counts['hint_fields']} from hint fields")

        if parts:
            self.console.print(f"  [dim]Content sources: {', '.join(parts)}[/dim]")
