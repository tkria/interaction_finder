"""
Runner for extraction graph V2 pipeline.

Simple entry point for testing the complete pipeline.
"""

import logging
from typing import List

from typing import Optional, Callable
from pathlib import Path

from ..resources import ResourcePool, ResourceId
from ..fetcher import PageFetcher
from ..settings import IfetcherConfig
from .state import ExtractionState
from .deps import ExtractionDeps
from .graph import extraction_graph_v2
from .nodes import ExtractEntities
from .models import EntityPairOut, BatchExtractionResultV2

logger = logging.getLogger(__name__)


async def run_extraction_v2(
    resource_pool: ResourcePool,
    deps: ExtractionDeps,
    document_groups: Optional[List[List[ResourceId]]] = None,
) -> List[EntityPairOut]:
    """
    Run the complete extraction pipeline.

    Args:
        resource_pool: ResourcePool with loaded documents
        deps: External dependencies (model, config, etc.)
        document_groups: Optional list of groups as ResourceId lists

    Returns:
        List of EntityPairOut objects with complete provenance
    """
    logger.info("Starting extraction graph V2 pipeline")

    # Create initial state with document groups
    state = ExtractionState(
        resource_pool=resource_pool, document_groups=document_groups or []
    )

    # Log initial state
    summary = state.get_summary()
    logger.info(f"Initial state: {summary}")

    # Process all groups sequentially
    all_pairs = []

    try:
        # If no groups defined, run once on all documents (backward compatibility)
        if not document_groups:
            logger.info("No document groups defined, processing all documents together")
            result = await extraction_graph_v2.run(
                start_node=ExtractEntities(), state=state, deps=deps
            )
            # Extract pairs from result
            if hasattr(result, "output") and isinstance(result.output, list):
                all_pairs = result.output
            else:
                all_pairs = state.final_pairs
        else:
            # Process each group sequentially
            logger.info(f"Processing {len(document_groups)} document groups")

            for group_index in range(len(document_groups)):
                state.current_group_index = group_index
                current_group = document_groups[group_index]

                logger.info(
                    f"Processing group {group_index + 1}/{len(document_groups)} ({len(current_group)} documents)"
                )

                # Clear previous group's results but preserve metrics
                state.entities_found.clear()
                state.individual_assessments.clear()
                state.final_pairs.clear()

                # Run extraction for this group
                result = await extraction_graph_v2.run(
                    start_node=ExtractEntities(), state=state, deps=deps
                )

                # Collect pairs from this group
                group_pairs = []
                if hasattr(result, "output") and isinstance(result.output, list):
                    group_pairs = result.output
                else:
                    group_pairs = state.final_pairs

                all_pairs.extend(group_pairs)
                logger.info(
                    f"Group {group_index + 1} completed: {len(group_pairs)} pairs found"
                )

        # Update state with all collected pairs
        state.final_pairs = all_pairs

        # Log final metrics
        final_summary = state.get_summary()
        provenance_metrics = state.get_provenance_metrics()

        logger.info(f"Pipeline complete: {final_summary}")
        logger.info(f"Provenance metrics: {provenance_metrics}")

        # Validate provenance chain
        if state.validate_provenance_chain():
            logger.info("✓ Complete provenance chain validated")
        else:
            logger.warning("⚠ Provenance chain validation failed")

        return all_pairs

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        # Return partial results if available
        return all_pairs


def create_test_resource_pool(documents: List[tuple[str, str, str]]) -> ResourcePool:
    """
    Create test resource pool from document tuples.

    Args:
        documents: List of (url, title, content) tuples

    Returns:
        ResourcePool with loaded documents
    """
    pool = ResourcePool()

    for url, title, content in documents:
        resource = pool.add(url, title, content)
        logger.info(f"Added test document: {resource.id.id} - {title}")

    return pool


async def extract_from_urls_v2(
    urls: List[str],
    config: IfetcherConfig,
    page_fetcher: Optional[PageFetcher] = None,
    model: Optional[str] = None,
    verbose: bool = False,
    save_callback: Optional[Callable[[BatchExtractionResultV2], None]] = None,
    parallelism: Optional[int] = None,
    target_term: Optional[str] = None,
) -> BatchExtractionResultV2:
    """
    Complete V2 pipeline from URLs to entity pairs.

    Compatible interface with extraction_graph.extract_from_urls for CLI integration.

    Args:
        urls: List of URLs to process
        config: Configuration object
        page_fetcher: Optional PageFetcher instance
        model: Optional model override
        verbose: Enable verbose logging
        save_callback: Optional callback for incremental saves
        parallelism: Optional parallelism override (unused in v2)
        target_term: Optional target term for entity extraction context

    Returns:
        BatchExtractionResultV2 with extracted pairs and metadata
    """
    logger.info(f"Starting extraction_v2 pipeline with {len(urls)} URLs")

    try:
        # Create or use provided page_fetcher
        if page_fetcher is None:
            page_fetcher = PageFetcher(config, show_status=True, verbose=verbose)

        # Use PageFetcher grouping to cluster related documents
        try:
            # Get document groups using PageFetcher's clustering
            if verbose:
                logger.info(f"Clustering {len(urls)} URLs into document groups...")

            groups_with_chunks, clustering_metadata = await page_fetcher.get_groups(
                urls=urls,
                constraint_type=config.workflow.grouping.constraint_type,
                min_size=config.workflow.grouping.min_size,
                max_size=config.workflow.grouping.max_size,
                linkage_method=config.workflow.grouping.linkage_method,
                clustering_method=config.workflow.grouping.clustering_method,
                embedding_weights=config.workflow.grouping.embedding_weights,
                include_chunks=True,
                progress=verbose,
                retry=False,
                fail_fast=False,
                return_metadata=True,
            )
        except Exception as e:
            logger.warning(f"Grouping failed, using single group: {e}")
            # Fallback to single group
            groups_with_chunks = [{"documents": urls, "chunks": {}}]

        # Create single ResourcePool and populate with all documents
        resource_pool = ResourcePool()
        url_to_resource_id = {}
        successful_urls = 0
        errors = []

        # Add all documents to the resource pool
        for group_data in groups_with_chunks:
            for url in group_data["documents"]:
                if url not in url_to_resource_id:
                    try:
                        # Fetch content - get_chunks returns List[str] (chunk texts)
                        chunk_texts = await page_fetcher.get_chunks(url)
                        if chunk_texts:
                            # Combine chunks into single content
                            content = "\n\n".join(chunk_texts)

                            # Try to extract title
                            title = f"Document from {url}"
                            try:
                                # Try to get cached markdown to extract title
                                if await page_fetcher.cache.has_path(url, "markdown"):
                                    md_content = await page_fetcher.cache.get_content(
                                        url, "markdown"
                                    )
                                    # Simple title extraction from markdown
                                    lines = md_content.split("\n")
                                    for line in lines[:10]:  # Check first 10 lines
                                        line = line.strip()
                                        if line.startswith("# "):
                                            title = line[2:].strip()
                                            break
                            except Exception:
                                pass  # Use default title

                            resource = resource_pool.add(url, title, content)
                            url_to_resource_id[url] = resource.id
                            successful_urls += 1
                            if verbose:
                                logger.info(
                                    f"Added document: {resource.id.id} - {title}"
                                )
                        else:
                            logger.warning(f"No content retrieved from {url}")
                            errors.append({"url": url, "error": "No content retrieved"})
                    except Exception as e:
                        logger.error(f"Failed to fetch {url}: {e}")
                        errors.append({"url": url, "error": str(e)})

        # Convert groups to ResourceId lists
        document_groups = []
        for group_data in groups_with_chunks:
            group_ids = []
            for url in group_data["documents"]:
                if url in url_to_resource_id:
                    group_ids.append(url_to_resource_id[url])
            if group_ids:  # Only add non-empty groups
                document_groups.append(group_ids)

        if verbose and document_groups:
            logger.info(f"Created {len(document_groups)} document groups:")
            for i, group in enumerate(document_groups):
                logger.info(f"  Group {i + 1}: {len(group)} documents")

        if resource_pool.resources:
            # Create extraction dependencies
            deps = ExtractionDeps.from_config(config, page_fetcher, model, target_term)

            # Run extraction pipeline with document groups
            pairs = await run_extraction_v2(resource_pool, deps, document_groups)

            # Create result object
            result = BatchExtractionResultV2.from_pairs(pairs)
            result.errors = errors
            result.total_groups = 1  # V2 doesn't use grouping yet
            result.successful_groups = 1 if pairs else 0

            # Call save callback if provided
            if save_callback and pairs:
                try:
                    save_callback(result)
                except Exception as e:
                    logger.warning(f"Save callback failed: {e}")

            logger.info(
                f"V2 pipeline complete: {len(pairs)} pairs from {successful_urls} URLs"
            )
            return result
        else:
            # No successful documents
            result = BatchExtractionResultV2()
            result.errors = errors
            result.total_groups = 1
            result.successful_groups = 0
            return result

    except Exception as e:
        logger.error(f"V2 pipeline failed: {e}")
        result = BatchExtractionResultV2()
        result.errors = [{"error": str(e)}]
        result.total_groups = 1
        result.successful_groups = 0
        return result


def save_results_v2(
    result: BatchExtractionResultV2,
    output_path: Path,
    term: str,
    mode: str = "basic",
    repeat: int = 1,
    entity_kinds: Optional[List[str]] = None,
) -> None:
    """
    Save V2 extraction results in compatible format with v1.

    Args:
        result: V2 extraction result
        output_path: Path to save results (.jsonl file)
        term: Research term for directory structure
        mode: Research mode for directory structure
        repeat: Repeat number for directory structure
        entity_kinds: List of entity kinds for normalization
    """
    import json

    # Create output directory using same logic as v1
    output_dir = output_path.with_suffix("")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get entity pairs
    all_pairs = result.entity_pairs

    # Use provided entity kinds or extract from pairs
    if entity_kinds is None:
        kinds_set = set()
        for pair in all_pairs:
            kinds_set.add(pair.entity_a.kind)
            kinds_set.add(pair.entity_b.kind)
        entity_kinds = list(kinds_set)

    # Save pairs.json - convert to output format
    pairs_file = output_dir / "pairs.json"
    pairs_data = [pair.to_output_format() for pair in all_pairs]

    with open(pairs_file, "w", encoding="utf-8") as f:
        json.dump(pairs_data, f, indent=2, ensure_ascii=False)

    # Save resources.json - extract resource information
    resources_file = output_dir / "resources.json"
    resources_data = {}

    for pair in all_pairs:
        # Add resources from entity quotes
        for entity in [pair.entity_a, pair.entity_b]:
            for quote in entity.quotes:
                resource_id = quote.resource.id.id
                if resource_id not in resources_data:
                    resources_data[resource_id] = {
                        "url": quote.resource.id.url,
                        "title": quote.resource.title,
                        "content_length": len(quote.resource.text),
                    }

    with open(resources_file, "w", encoding="utf-8") as f:
        json.dump(resources_data, f, indent=2, ensure_ascii=False)

    # Save summary.json
    summary_file = output_dir / "summary.json"
    summary_data = {
        "extraction": {
            "term": term,
            "mode": mode,
            "repeat": repeat,
            "entity_kinds": entity_kinds,
            "extraction_version": "v2",
        },
        "results": {
            "total_pairs": result.total_pairs,
            "total_groups": result.total_groups,
            "successful_groups": result.successful_groups,
            "total_entities": result.total_entities,
            "error_count": len(result.errors),
        },
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    logger.info(f"V2 results saved to: {output_dir}")
    logger.info(
        f"Files: pairs.json ({len(all_pairs)} pairs), resources.json ({len(resources_data)} resources)"
    )
