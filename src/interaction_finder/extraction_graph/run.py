"""
Entry point for running the entity extraction graph pipeline.

This module provides the high-level API for processing document groups
and extracting entity pairs.
"""

import time
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.spinner import Spinner
from rich.live import Live
from rich import box

try:
    import logfire
except ImportError:
    # Create a no-op logfire if not available
    class _NoOpLogfire:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

    logfire = _NoOpLogfire()

from ..fetcher import PageFetcher
from ..settings import IfetcherConfig
from .deps import Deps
from .state import EntityExtractionState, DocumentGroup
from .graph import extraction_graph
from .nodes import InitialRouter
from .models import BatchExtractionResult, ExtractionSummary


async def prepare_document_groups(
    urls: List[str],
    page_fetcher: PageFetcher,
    config: IfetcherConfig,
) -> List[DocumentGroup]:
    """
    Prepare document groups with chunk tracking from URLs.

    Args:
        urls: List of document URLs to process
        page_fetcher: PageFetcher instance for content retrieval
        config: Configuration for grouping parameters

    Returns:
        List of DocumentGroup objects with chunk attribution
    """
    if not urls:
        return []

    console = Console()

    # Use PageFetcher's grouping functionality instead of creating our own DocumentGrouper
    console.print(f"[dim]Fetching content and grouping {len(urls)} URLs...[/dim]")

    try:
        # Call PageFetcher's get_groups with chunks and metadata included
        groups_with_chunks, clustering_metadata = await page_fetcher.get_groups(
            urls=urls,
            constraint_type=config.workflow.grouping.constraint_type,
            min_size=config.workflow.grouping.min_size,
            max_size=config.workflow.grouping.max_size,
            linkage_method=config.workflow.grouping.linkage_method,
            clustering_method=config.workflow.grouping.clustering_method,
            embedding_weights=config.workflow.grouping.embedding_weights,
            seeding_method=config.workflow.grouping.seeding_method,
            refinement_method=config.workflow.grouping.refinement_method,
            include_chunks=True,
            progress=True,
            retry=False,
            fail_fast=False,
            return_metadata=True,
        )

        # Extract groups - comprehensive metadata already obtained from get_groups
        groups = [group_data["documents"] for group_data in groups_with_chunks]

        # Add basic fields to clustering_metadata if missing
        if "group_cohesions" not in clustering_metadata:
            clustering_metadata["group_cohesions"] = [
                group_data["cohesion_score"] for group_data in groups_with_chunks
            ]
        if "total_word_counts" not in clustering_metadata:
            clustering_metadata["total_word_counts"] = [
                group_data["total_words"] for group_data in groups_with_chunks
            ]

        # Compute metadata for display
        if clustering_metadata.get("group_cohesions"):
            cohesions = clustering_metadata["group_cohesions"]
            clustering_metadata["avg_cohesion"] = sum(cohesions) / len(cohesions)
            import numpy as np

            clustering_metadata["cohesion_std"] = float(np.std(cohesions))

    except Exception as e:
        console.print(f"[red]Grouping failed, using single group: {e}[/red]")
        # Fallback to single group containing all documents
        groups = [urls]
        clustering_metadata = {}
        # Create dummy groups_with_chunks for the conversion below
        groups_with_chunks = [
            {"documents": urls, "chunks": {}, "total_words": 0, "cohesion_score": 1.0}
        ]

    # Create DocumentGroup objects with attribution
    document_groups = []
    for group_data in groups_with_chunks:
        group_urls = group_data["documents"]
        group_chunks = group_data.get("chunks", {})

        # Gather metadata (could be enhanced with DOI, title, etc.)
        metadata = {}
        for url in group_urls:
            metadata[url] = {
                "chunk_count": len(group_chunks.get(url, [])),
                "word_count": sum(
                    chunk.get("wordcount", 0) for chunk in group_chunks.get(url, [])
                ),
            }

        document_group = DocumentGroup(
            urls=group_urls, chunks=group_chunks, metadata=metadata
        )
        document_groups.append(document_group)

    # Create clustering summary table
    if clustering_metadata:
        table = Table(show_header=False, box=None, pad_edge=False)
        table.add_column("Key", style="dim")
        table.add_column("Value", style="bright_white")

        # Build algorithm description
        algorithm_desc = clustering_metadata["algorithm"]

        table.add_row("Algorithm", algorithm_desc)
        if clustering_metadata.get("embedding_weights"):
            embedding_weights = clustering_metadata["embedding_weights"]
            if embedding_weights == "uniform":
                display_text = "uniform weighting"
            elif embedding_weights == "idf":
                display_text = "idf weighting"
            else:
                display_text = f"{embedding_weights} weighting"
            table.add_row("Embedding", display_text)
        table.add_row(
            "Constraints",
            f"{clustering_metadata['constraint_type']} {clustering_metadata['constraints']['min_size']}-{clustering_metadata['constraints']['max_size']}",
        )
        table.add_row(
            "Documents",
            f"{clustering_metadata['documents_clustered']}/{clustering_metadata['total_documents']} clustered ({clustering_metadata['documents_with_embeddings']} with embeddings)",
        )
        if clustering_metadata.get("modularity") is not None:
            modularity_value = clustering_metadata["modularity"]
            table.add_row("Modularity", f"{modularity_value:.3f}")

        # Display comprehensive clustering metrics instead of just avg cohesion
        if clustering_metadata.get("pair_weighted_cohesion") is not None:
            table.add_row(
                "Cohesion (pw) [dim]↑[/dim]",
                f"{clustering_metadata['pair_weighted_cohesion']:.3f}",
            )

        if clustering_metadata.get("contrast") is not None:
            table.add_row(
                "Contrast [dim]↑[/dim]",
                f"{clustering_metadata['contrast']:.3f}",
            )

        if clustering_metadata.get("robust_cohesion") is not None:
            robust = clustering_metadata["robust_cohesion"]
            table.add_row(
                "Robust (p10) [dim]↑[/dim]",
                f"{robust['median']:.3f} (min: {robust['min']:.3f})",
            )

        if clustering_metadata.get("leakage") is not None:
            leakage = clustering_metadata["leakage"]
            table.add_row(
                "Max leakage [dim]↓[/dim]",
                f"{leakage['max']:.3f}",
            )

        if clustering_metadata.get("silhouette") is not None:
            silhouette = clustering_metadata["silhouette"]
            table.add_row(
                "Silhouette [dim]↑[/dim]",
                f"{silhouette['mean']:.3f}",
            )

        console.print(
            Panel(
                table,
                title=f"[bright_white]{len(document_groups)} Document Groups Created[/bright_white]",
                border_style="blue",
            )
        )
    else:
        console.print(
            f"[bright_white]Created {len(document_groups)} document groups[/bright_white]"
        )

    # Display groups in a table
    if clustering_metadata and document_groups:
        groups_table = Table(
            show_header=True,
            header_style="bold blue",
            box=box.HORIZONTALS,
            border_style="blue",
            show_lines=False,
            padding=(0, 2),
        )
        groups_table.add_column("Group", style="cyan", justify="center")
        groups_table.add_column("Docs", style="bright_white", justify="right")
        groups_table.add_column("Chunks", style="bright_white", justify="right")
        groups_table.add_column("Words", style="green", justify="right")
        groups_table.add_column("Cohesion", style="yellow", justify="center")
        groups_table.add_column("Nearby", style="dim", justify="center")

        for i, group in enumerate(document_groups):
            cohesion = (
                clustering_metadata["group_cohesions"][i]
                if i < len(clustering_metadata.get("group_cohesions", []))
                else 0.0
            )
            words = (
                clustering_metadata["total_word_counts"][i]
                if i < len(clustering_metadata.get("total_word_counts", []))
                else 0
            )
            nearby = (
                clustering_metadata["nearby_groups"][i]
                if i < len(clustering_metadata.get("nearby_groups", []))
                else "—"
            )

            # Format cohesion with color coding
            if cohesion >= 0.8:
                cohesion_display = f"[green]{cohesion:.3f}[/green]"
            elif cohesion >= 0.5:
                cohesion_display = f"[yellow]{cohesion:.3f}[/yellow]"
            else:
                cohesion_display = f"[red]{cohesion:.3f}[/red]"

            groups_table.add_row(
                str(i + 1),
                str(len(group.urls)),
                str(group.get_total_chunks()),
                f"{words:,}",
                cohesion_display,
                nearby,
            )

        console.print()
        console.print(groups_table)
    else:
        # Fallback simple display if no metadata
        for i, group in enumerate(document_groups):
            console.print(
                f"  [cyan]Group {i + 1}[/cyan]: [bright_white]{len(group.urls)}[/bright_white] documents, [bright_white]{group.get_total_chunks()}[/bright_white] chunks"
            )

    return document_groups


def _save_incremental_results(
    all_entity_pairs: List,
    group_summaries: List,
    errors: List,
    total_entities: Dict[str, int],
    deps: Deps,
    save_callback: Optional[Callable[[BatchExtractionResult], None]] = None,
) -> None:
    """Save incremental results after each group processing."""
    if not save_callback:
        return

    # Create intermediate batch result
    intermediate_result = BatchExtractionResult(
        total_groups=len(group_summaries) + len(errors),  # Approximate
        successful_groups=len([s for s in group_summaries if s.pairs_validated > 0]),
        total_entities=total_entities.copy(),
        total_pairs=len(all_entity_pairs),
        entity_pairs=all_entity_pairs.copy(),
        group_summaries=group_summaries.copy(),
        errors=errors.copy(),
        processing_config={
            "entity_kinds": deps.entity_kinds,
            "relation_type": deps.relation_type,
            "task_context": deps.task_context,
            "model": str(deps.model),
        },
    )

    # Call the save callback
    save_callback(intermediate_result)


async def extract_entity_pairs(
    document_groups: List[DocumentGroup],
    config: IfetcherConfig,
    page_fetcher: PageFetcher,
    model: Optional[str] = None,
    verbose: bool = False,
    save_callback: Optional[Callable[[BatchExtractionResult], None]] = None,
    parallelism: Optional[int] = None,
) -> BatchExtractionResult:
    """
    Extract entity pairs from document groups using the graph pipeline.

    Args:
        document_groups: Prepared document groups to process
        config: Configuration for extraction
        page_fetcher: PageFetcher instance for any additional content needs
        model: Optional model override
        verbose: Enable verbose progress reporting
        save_callback: Optional function to call after each group with intermediate results

    Returns:
        BatchExtractionResult with all extracted pairs and metadata
    """
    if not document_groups:
        return BatchExtractionResult(
            total_groups=0,
            successful_groups=0,
            total_entities={},
            total_pairs=0,
            group_summaries=[],
            errors=[],
            processing_config=config.model_dump(),
        )

    # Create dependencies
    deps = Deps.from_config(config, page_fetcher, model, verbose, parallelism)

    # Initialize batch results
    all_pairs = []
    all_entity_pairs = []  # Store actual EntityPairOut objects
    group_summaries = []
    errors = []
    successful_groups = 0
    total_entities = {kind: 0 for kind in deps.entity_kinds}

    # Initialize progress tracker for verbose mode
    progress_tracker = None
    if verbose:
        from .progress_tracker import create_progress_tracker

        progress_tracker = create_progress_tracker()
        if progress_tracker:
            progress_tracker.start(len(document_groups))

    if not verbose:
        print(f"Processing {len(document_groups)} document groups...")

    # Process each document group
    for group_idx, document_group in enumerate(document_groups):
        # Update progress tracker
        if progress_tracker:
            progress_tracker.update_group(
                group_idx, document_group.urls, document_group.get_total_chunks()
            )
        else:
            print(f"\nProcessing group {group_idx + 1}/{len(document_groups)}")
            print(f"  URLs: {len(document_group.urls)}")
            print(f"  Chunks: {document_group.get_total_chunks()}")

        try:
            start_time = time.time()

            # Create state for this group
            state = EntityExtractionState(
                document_groups=[document_group],
                current_group_index=0,
                entity_kinds=deps.entity_kinds,
                relation_type=deps.relation_type,
            )

            # Store progress tracker in state for access by nodes
            if progress_tracker:
                state.add_processing_note("progress_tracker", progress_tracker)

            # Run the graph
            result = await extraction_graph.run(InitialRouter(), state=state, deps=deps)

            processing_time = time.time() - start_time

            # Extract results
            if hasattr(result, "output") and isinstance(result.output, list):
                group_pairs = result.output
                all_pairs.extend(group_pairs)
            else:
                group_pairs = []

            # Extract EntityPairOut objects from state for output formatting
            from .models import EntityPairOut

            for pair_data in state.entity_pairs:
                if isinstance(pair_data, dict):
                    try:
                        # Convert dict back to EntityPairOut object
                        entity_pair = EntityPairOut.model_validate(pair_data)
                        all_entity_pairs.append(entity_pair)
                    except Exception:
                        # Skip invalid pairs
                        pass

            # Update totals
            for kind in deps.entity_kinds:
                total_entities[kind] += state.get_entity_count(kind)

            if group_pairs or state.entity_pairs:
                successful_groups += 1

            # Update progress tracker with entity counts
            if progress_tracker:
                entities_by_kind = {
                    kind: state.get_entity_count(kind) for kind in deps.entity_kinds
                }
                progress_tracker.update_entities(entities_by_kind)

            # Show intermediate results for debugging
            if not verbose:
                entity_counts = {
                    kind: state.get_entity_count(kind) for kind in deps.entity_kinds
                }
                total_entities_found = sum(entity_counts.values())
                print(
                    f"    Entities found: {total_entities_found} ({dict(entity_counts)})"
                )
                print(
                    f"    Individual assessments: {len(state.individual_entity_assessments)}"
                )
                print(f"    Pairs generated: {len(state.entity_pairs)}")

                # Show routing decision
                routing_decision = state.processing_notes.get("routing_decision", {})
                if routing_decision:
                    classification = routing_decision.get("classification", "unknown")
                    confidence = routing_decision.get("confidence", 0)
                    print(
                        f"    Content classification: {classification} (confidence: {confidence:.2f})"
                    )

            # Create summary
            summary = ExtractionSummary(
                document_group_urls=document_group.urls,
                total_chunks=document_group.get_total_chunks(),
                entity_counts={
                    kind: state.get_entity_count(kind) for kind in deps.entity_kinds
                },
                pairs_extracted=len(state.entity_pairs),
                pairs_validated=len(group_pairs),
                processing_time=processing_time,
                node_path=state.node_history,
                quality_metrics={
                    "routing_confidence": state.processing_notes.get(
                        "routing_decision", {}
                    ).get("confidence", 0),
                    "validation_success_rate": len(group_pairs)
                    / max(1, len(state.entity_pairs)),
                },
            )
            group_summaries.append(summary)

            print(
                f"  Result: {len(group_pairs)} validated pairs in {processing_time:.1f}s"
            )

            logfire.info(
                f"Group {group_idx + 1}/{len(document_groups)} completed: {len(group_pairs)} pairs in {processing_time:.1f}s"
            )

            # Save incremental results after successful processing
            _save_incremental_results(
                all_entity_pairs,
                group_summaries,
                errors,
                total_entities,
                deps,
                save_callback,
            )

        except Exception as e:
            error_info = {
                "group_index": group_idx,
                "urls": document_group.urls,
                "error": str(e),
                "error_type": type(e).__name__,
            }
            errors.append(error_info)

            # Update progress tracker
            if progress_tracker:
                progress_tracker.increment_errors()
            else:
                # Show concise error message in non-verbose mode
                error_type = type(e).__name__
                if "api_key" in str(e).lower():
                    print(f"  Failed: API key not configured")
                elif len(str(e)) > 50:
                    print(f"  Failed: {error_type}")
                else:
                    print(f"  Failed: {e}")

            # Create minimal summary for failed group
            summary = ExtractionSummary(
                document_group_urls=document_group.urls,
                total_chunks=document_group.get_total_chunks(),
                entity_counts={kind: 0 for kind in deps.entity_kinds},
                pairs_extracted=0,
                pairs_validated=0,
                processing_time=0.0,
                node_path=["error"],
                quality_metrics={},
            )
            group_summaries.append(summary)

            # Save incremental results after failed processing too
            _save_incremental_results(
                all_entity_pairs,
                group_summaries,
                errors,
                total_entities,
                deps,
                save_callback,
            )

    # Cleanup progress tracker
    if progress_tracker:
        progress_tracker.stop()

    # Create final result
    result = BatchExtractionResult(
        total_groups=len(document_groups),
        successful_groups=successful_groups,
        total_entities=total_entities,
        total_pairs=len(all_entity_pairs),
        entity_pairs=all_entity_pairs,
        group_summaries=group_summaries,
        errors=errors,
        processing_config={
            "entity_kinds": deps.entity_kinds,
            "relation_type": deps.relation_type,
            "task_context": deps.task_context,
            "model": str(deps.model),
        },
    )

    if not verbose:
        print(f"\nBatch processing complete:")
        print(f"  Groups processed: {len(document_groups)}")
        print(f"  Successful groups: {successful_groups}")
        print(f"  Total entity pairs: {len(all_entity_pairs)}")
        print(f"  Entities by kind: {total_entities}")
        if errors:
            print(f"  Errors: {len(errors)}")

    return result


async def extract_from_urls(
    urls: List[str],
    config: IfetcherConfig,
    page_fetcher: Optional[PageFetcher] = None,
    model: Optional[str] = None,
    verbose: bool = False,
    save_callback: Optional[Callable[[BatchExtractionResult], None]] = None,
    parallelism: Optional[int] = None,
) -> BatchExtractionResult:
    """
    Complete pipeline from URLs to entity pairs.

    Convenience function that handles document preparation and extraction.

    Args:
        urls: List of document URLs
        config: Configuration
        page_fetcher: Optional PageFetcher instance
        model: Optional model override
        verbose: Enable verbose progress reporting
        save_callback: Optional function to call after each group with intermediate results

    Returns:
        BatchExtractionResult with extracted pairs
    """
    # Create PageFetcher if not provided
    if page_fetcher is None:
        page_fetcher = PageFetcher(config)

    # Prepare document groups
    document_groups = await prepare_document_groups(urls, page_fetcher, config)

    if not document_groups:
        return BatchExtractionResult(
            total_groups=0,
            successful_groups=0,
            total_entities={},
            total_pairs=0,
            group_summaries=[],
            errors=[
                {
                    "error": "No valid document groups could be created from provided URLs"
                }
            ],
            processing_config=config.model_dump(),
        )

    # Extract entity pairs
    return await extract_entity_pairs(
        document_groups,
        config,
        page_fetcher,
        model,
        verbose,
        save_callback,
        parallelism,
    )


def save_results(
    result: BatchExtractionResult,
    output_path: Path,
    term: str,
    mode: str = "basic",
    repeat: int = 1,
    entity_kinds: Optional[List[str]] = None,
) -> None:
    """
    Save extraction results using OutputFormatter for compliance.

    Args:
        result: BatchExtractionResult to save
        output_path: Path for output file (legacy compatibility)
        term: Research term for directory structure
        mode: Research mode for directory structure
        repeat: Repeat number for directory structure
        entity_kinds: List of entity kinds for normalization
    """
    from .output_formatter import OutputFormatter
    from .usage_tracker import get_usage_summary
    from .models import EntityPairOut, convert_pairs_to_final_format

    # Use the CLI-generated path directly instead of OutputFormatter structure
    # The output_path from CLI is a .jsonl file, we want to save to a directory with the same name
    output_dir = output_path.with_suffix("")

    # Ensure directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save files directly to the CLI-generated path structure
    import json

    # Get entity pairs from the batch result
    all_pairs = result.entity_pairs

    # Use entity kinds from config if not provided
    if entity_kinds is None:
        entity_kinds = list(
            result.processing_config.get("entity_kinds", ["gene", "disease"])
        )

    # Save pairs.json - convert to final aggregated format
    pairs_file = output_dir / "pairs.json"
    # Convert EntityPairOut objects to final format with aggregation
    aggregated_pairs = convert_pairs_to_final_format(all_pairs)

    with open(pairs_file, "w", encoding="utf-8") as f:
        json.dump(aggregated_pairs, f, indent=2, ensure_ascii=False)

    # Get and save usage data
    usage_data = get_usage_summary()
    usage_file = output_dir / "usage.json"
    with open(usage_file, "w", encoding="utf-8") as f:
        json.dump(usage_data, f, indent=2)

    # Save metadata.json
    metadata = {
        "total_groups": result.total_groups,
        "successful_groups": result.successful_groups,
        "total_entities": result.total_entities,
        "total_pairs": result.total_pairs,
        "processing_config": result.processing_config,
        "errors": result.errors,
        "group_summaries": [
            summary.model_dump() if hasattr(summary, "model_dump") else summary
            for summary in result.group_summaries
        ],
    }
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_dir}")
    print(f"  - pairs.json: {len(aggregated_pairs)} aggregated pairs")
    print(f"  - usage.json: {len(usage_data)} model(s) tracked")
    print(f"  - metadata.json: batch processing metadata")
