"""
Runner for extraction graph V2 pipeline.

Simple entry point for testing the complete pipeline.
"""

import logging
from typing import List
import json

from typing import Optional, Callable
from pathlib import Path

from pydantic_graph import GraphRunContext
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ..resources import ResourcePool, ResourceId, compute_chunk_spans
from ..fetcher import PageFetcher
from ..settings import IfetcherConfig
from .state import ExtractionState
from .deps import ExtractionDeps
from .nodes import ExtractEntities
from .models import EntityPairOut, BatchExtractionResultV2
from .quote_logging import save_quote_errors_incremental

logger = logging.getLogger(__name__)
console = Console()


async def run_extraction_v2(
    resource_pool: ResourcePool,
    deps: ExtractionDeps,
    verbose: bool = False,
) -> List[EntityPairOut]:
    """
    Run the complete extraction pipeline.

    Args:
        resource_pool: ResourcePool with loaded documents
        deps: External dependencies (model, config, etc.)
        verbose: Enable verbose Rich logging

    Returns:
        List of EntityPairOut objects with complete provenance
    """
    logger.info("Starting extraction graph V2 pipeline")

    # Create initial state (single-group processing in v2)
    state = ExtractionState(resource_pool=resource_pool)

    # Log initial state
    summary = state.get_summary()
    logger.info(f"Initial state: {summary}")

    # Process all groups sequentially
    all_pairs = []

    try:
        # Process all documents together (no grouping in v2)
        logger.info("Processing all documents together (no document groups in v2)")

        # Phase 1: Extract entities from all documents as single group
        logger.info("Phase 1: Extracting entities from all documents")
        from .nodes import ExtractEntities

        extract_node = ExtractEntities()
        await extract_node.run(GraphRunContext(state=state, deps=deps))
        all_extracted_entities = list(state.entities_found.values())
        logger.info(
            f"Phase 1 complete: {len(all_extracted_entities)} entities extracted"
        )

        # Phase 2: No deduplication needed for single group
        logger.info("Phase 2: No deduplication needed (single group)")
        merged_entities = all_extracted_entities

        # Phase 3: Global assessment
        logger.info("Phase 3: Assessing all entities")
        state.entities_found = {entity.name: entity for entity in merged_entities}
        state.individual_assessments.clear()

        from .nodes import AssessIndividually

        assess_node = AssessIndividually()
        await assess_node.run(GraphRunContext(state=state, deps=deps))
        logger.info(
            f"Phase 3 complete: {len(state.individual_assessments)} assessments"
        )

        # Phase 4: Form pairs
        logger.info("Phase 4: Forming pairs")
        state.final_pairs.clear()

        from .nodes import AggregateIntoPairs

        pair_node = AggregateIntoPairs()
        await pair_node.run(GraphRunContext(state=state, deps=deps))

        all_pairs = state.final_pairs
        logger.info(f"Phase 4 complete: {len(all_pairs)} pairs formed")

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


def _merge_duplicate_entities(entities):
    """
    Merge entities with same name+kind, combining ALL evidence.

    The primary goal is to aggregate all quotes, mentions, and evidence
    for each unique entity across all document batches.

    Args:
        entities: List of EntityWithQuotes objects

    Returns:
        List of merged EntityWithQuotes objects with combined evidence
    """
    from .models import EntityWithQuotes

    entity_map = {}
    for entity in entities:
        # Create key based on normalized name and kind
        key = (entity.name.lower().strip(), entity.kind.lower().strip())

        if key in entity_map:
            # Combine ALL evidence for this entity
            existing_entity = entity_map[key]

            # Merge all quotes (preserving resource provenance)
            existing_entity.quotes.extend(entity.quotes)

            # Update confidence (take maximum - best assessment seen)
            existing_entity.confidence = max(
                existing_entity.confidence, entity.confidence
            )

            # Use the most complete name variant (longest one)
            if len(entity.name) > len(existing_entity.name):
                existing_entity.name = entity.name

            logger.debug(f"Combined evidence for entity: {entity.name} ({entity.kind})")
        else:
            entity_map[key] = entity

    merged_entities = list(entity_map.values())

    # Log evidence aggregation statistics
    total_quotes_before = sum(len(e.quotes) for e in entities)
    total_quotes_after = sum(len(e.quotes) for e in merged_entities)

    if len(merged_entities) < len(entities):
        duplicates_merged = len(entities) - len(merged_entities)
        logger.info(f"Entity deduplication: {duplicates_merged} duplicates merged")
        logger.info(
            f"Evidence aggregation: {total_quotes_before} → {total_quotes_after} total quotes"
        )

    return merged_entities


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
    output_dir: Optional[Path] = None,
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
                        # Fetch both full document and chunks
                        chunk_texts = await page_fetcher.get_chunks(url)
                        if chunk_texts:
                            # Try to get full document for better chunk mapping
                            full_document = None
                            chunk_spans = None

                            try:
                                full_document = await page_fetcher.get_markdown(url)
                                if full_document:
                                    # Compute chunk spans in full document
                                    chunk_spans = compute_chunk_spans(
                                        full_document, chunk_texts
                                    )
                                    if verbose and chunk_spans:
                                        logger.info(
                                            f"Computed {len(chunk_spans)} chunk spans for {url}"
                                        )
                            except Exception as e:
                                if verbose:
                                    logger.warning(
                                        f"Could not get full document for {url}: {e}"
                                    )

                            # Use full document if available, otherwise fall back to joined chunks
                            if full_document:
                                content = full_document
                            else:
                                content = "\n\n".join(chunk_texts)
                                chunk_spans = (
                                    None  # No chunks if we don't have full document
                                )

                            # Try to extract title
                            title = f"Document from {url}"
                            try:
                                # Simple title extraction from content
                                lines = content.split("\n")
                                for line in lines[:10]:  # Check first 10 lines
                                    line = line.strip()
                                    if line.startswith("# "):
                                        title = line[2:].strip()
                                        break
                            except Exception:
                                pass  # Use default title

                            resource = resource_pool.add(
                                url, title, content, chunks=chunk_spans
                            )
                            url_to_resource_id[url] = resource.id
                            successful_urls += 1
                            if verbose:
                                chunk_info = (
                                    f" with {len(chunk_spans)} chunks"
                                    if chunk_spans
                                    else ""
                                )
                                logger.info(
                                    f"Added document: {resource.id.id} - {title}{chunk_info}"
                                )
                        else:
                            logger.warning(f"No content retrieved from {url}")
                            errors.append({"url": url, "error": "No content retrieved"})
                    except Exception as e:
                        logger.error(f"Failed to fetch {url}: {e}")
                        errors.append({"url": url, "error": str(e)})

        if resource_pool.resources:
            # Create extraction dependencies
            deps = ExtractionDeps.from_config(
                config,
                page_fetcher,
                model,
                target_term,
                parallelism or 0,
                output_dir=output_dir,
            )

            # Run extraction pipeline (v2 processes all documents together)
            pairs = await run_extraction_v2(resource_pool, deps, verbose=verbose)

            # Create result object
            result = BatchExtractionResultV2.from_pairs(pairs)
            result.errors = errors
            result.quote_errors = deps.quote_error_log
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

    # Save quote errors log (overwrite with final version)
    save_quote_errors_incremental(
        result.quote_errors, output_dir, append_mode=False, saved_count=0
    )

    logger.info(f"V2 results saved to: {output_dir}")
    logger.info(
        f"Files: pairs.json ({len(all_pairs)} pairs), resources.json ({len(resources_data)} resources), quote_errors.json ({len(result.quote_errors)} errors)"
    )
