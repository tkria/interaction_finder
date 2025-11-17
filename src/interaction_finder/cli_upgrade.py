"""CLI upgrade command implementation for migrating old checkpoint formats.

This module provides self-contained upgrade functionality to migrate legacy
checkpoint files (BridgingTermsOut, WidesearchCheckpoint, ExtractionResult)
to the unified PipelineCheckpoint format.

Can be safely deleted once all checkpoints are upgraded.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Literal, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from interaction_finder.checkpoint import (
    ExtractionStageData,
    KeywordsStageData,
    PipelineCheckpoint,
    SearchStageData,
)
from interaction_finder.fetcher import PageFetcher
from interaction_finder.resources import Resource, ResourceId, ResourcePool
from interaction_finder.settings import IfetcherConfig


def detect_format(data: dict) -> Literal["keywords", "searches", "extraction", "new"]:
    """Detect checkpoint format from JSON structure.

    Parameters:
        data: dict — loaded JSON data

    Returns:
        str — "keywords", "searches", "extraction", or "new"

    Raises:
        ValueError: If format cannot be determined
    """
    # Check if already new format
    if "keywords" in data or "search" in data or "extraction" in data:
        return "new"

    # Old extraction format: has judgments + metadata
    if "judgments" in data and "metadata" in data:
        return "extraction"

    # Old search format: has results + query_results
    if "results" in data and "query_results" in data:
        return "searches"

    # Old keywords format: has terms + scores + coverage_assessment
    if "terms" in data and "scores" in data and "coverage_assessment" in data:
        return "keywords"

    raise ValueError(
        "Unknown checkpoint format: could not detect stage from JSON structure. "
        "Expected one of: keywords (terms/scores), searches (results/query_results), "
        "extraction (judgments/metadata), or new format (keywords/search/extraction)."
    )


def parse_keywords(data: dict) -> tuple[str, KeywordsStageData, list[dict]]:
    """Parse old keywords format to stage data.

    Parameters:
        data: dict — old BridgingTermsOut JSON

    Returns:
        tuple — (topic, KeywordsStageData, resources_list)
    """
    # Extract topic (may be missing in very old files)
    topic = data.get("topic", "Unknown Topic")

    # Extract resource URLs from resources list
    resources_list = data.get("resources", [])
    if isinstance(resources_list, dict):
        # Handle resource_map format from very old keywords files
        resources_list = list(resources_list.get("resource_map", {}).values())
    resource_urls = [r["url"] for r in resources_list if "url" in r]

    # Build KeywordsStageData
    keywords_data = KeywordsStageData(
        terms=data["terms"],
        scores=data["scores"],
        total_documents_processed=data["total_documents_processed"],
        rounds_completed=data["rounds_completed"],
        coverage_assessment=data["coverage_assessment"],
        resource_urls=resource_urls,
    )

    return topic, keywords_data, resources_list


def parse_searches(data: dict) -> tuple[str, SearchStageData, list[dict]]:
    """Parse old searches format to stage data.

    Parameters:
        data: dict — old WidesearchCheckpoint JSON

    Returns:
        tuple — (topic, SearchStageData, resources_list)
    """
    topic = data["topic"]
    resources_list = data.get("resources", [])

    # Build SearchStageData
    search_data = SearchStageData(
        results=data["results"],
        queries=data["queries"],
        query_results=data["query_results"],
        keyphrases=data["keyphrases"],
        rounds_completed=data["rounds_completed"],
    )

    return topic, search_data, resources_list


def infer_entity_types(judgments: list[dict]) -> list[str]:
    """Infer target entity types from judgments.

    Parameters:
        judgments: list[dict] — pair judgments

    Returns:
        list[str] — unique entity kinds found
    """
    kinds = set()
    for judgment in judgments:
        if "entity1" in judgment:
            kinds.add(judgment["entity1"]["kind"])
        if "entity2" in judgment:
            kinds.add(judgment["entity2"]["kind"])
    return sorted(kinds)


def infer_permitted_pairs(judgments: list[dict]) -> dict[str, list[str]]:
    """Infer permitted entity kind pairs from judgments.

    Parameters:
        judgments: list[dict] — pair judgments

    Returns:
        dict[str, list[str]] — mapping from kind to permitted partner kinds
    """
    pairs: dict[str, set[str]] = {}
    for judgment in judgments:
        if "entity1" in judgment and "entity2" in judgment:
            kind1 = judgment["entity1"]["kind"]
            kind2 = judgment["entity2"]["kind"]

            if kind1 not in pairs:
                pairs[kind1] = set()
            if kind2 not in pairs:
                pairs[kind2] = set()

            pairs[kind1].add(kind2)
            pairs[kind2].add(kind1)

    # Convert sets to sorted lists
    return {k: sorted(v) for k, v in sorted(pairs.items())}


def parse_extraction(data: dict) -> tuple[str, dict, list[dict]]:
    """Parse old extraction format to stage data dict.

    Parameters:
        data: dict — old ExtractionResult JSON

    Returns:
        tuple — (topic, extraction_stage_dict, resources_list)

    Note:
        Returns dict instead of ExtractionStageData to defer validation until
        after resource rehydration in PipelineCheckpoint.
    """
    topic = data["topic"]
    resources_list = data.get("resources", [])
    judgments = data["judgments"]

    # Infer entity configuration if not present
    target_entity_types = data.get("target_entity_types") or infer_entity_types(
        judgments
    )
    permitted_pairs = data.get("permitted_pairs") or infer_permitted_pairs(judgments)

    # Build extraction stage dict (defer validation)
    extraction_dict = {
        "target_entity_types": target_entity_types,
        "permitted_pairs": permitted_pairs,
        "judgments": judgments,
        "metadata": data["metadata"],
    }

    return topic, extraction_dict, resources_list


async def enrich_resources(
    resources_list: list[dict], config: IfetcherConfig, console: Console
) -> list[Resource]:
    """Enrich resources with DOI and publication date metadata.

    Fetches DOI and publication dates from cache or OpenAlex API.
    Shows progress using Rich console.

    Parameters:
        resources_list: list[dict] — old resource dicts
        config: IfetcherConfig — config with cache path
        console: Console — Rich console for output

    Returns:
        list[Resource] — enriched Resource objects
    """
    cache_dir = config.abspath(config.output.cache)
    fetcher = PageFetcher(cache_dir=cache_dir)

    enriched = []
    stats = {"doi_cached": 0, "doi_fetched": 0, "date_cached": 0, "date_fetched": 0}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Enriching {len(resources_list)} resources with DOI/dates...",
            total=len(resources_list),
        )

        for idx, resource_dict in enumerate(resources_list, start=1):
            url = resource_dict["url"]
            title = resource_dict.get("title", "")
            text = resource_dict.get("text", "")
            chunks = resource_dict.get("chunks", [(0, len(text))])

            # Try to get DOI from cache first, otherwise None
            doi = resource_dict.get("doi")
            if doi is None:
                cached_doi = await fetcher.get_doi(url)
                if cached_doi:
                    doi = cached_doi
                    stats["doi_cached"] += 1

            # Try to get publication date (requires DOI)
            publication_date = resource_dict.get("publication_date")
            if publication_date is None and doi:
                cached_date = await fetcher.get_publication_date(url, doi=doi)
                if cached_date:
                    publication_date = cached_date
                    stats["date_cached"] += 1

            # Create ResourceId and Resource
            resource_id = ResourceId(url=url, counter=idx)
            resource = Resource(
                id=resource_id,
                title=title,
                text=text,
                chunks=chunks,
                doi=doi,
                publication_date=publication_date,
            )
            enriched.append(resource)

            progress.update(task, advance=1)

    # Show enrichment stats
    console.print(
        f"[dim]Enriched {stats['doi_cached']} DOIs from cache, "
        f"{stats['date_cached']} dates from cache[/dim]"
    )

    return enriched


def build_resource_pool(enriched_resources: list[Resource]) -> ResourcePool:
    """Build ResourcePool from enriched resources.

    Deduplicates by URL and creates proper ResourcePool structure.

    Parameters:
        enriched_resources: list[Resource] — enriched resources

    Returns:
        ResourcePool — deduplicated pool
    """
    pool = ResourcePool()
    seen_urls = set()

    for resource in enriched_resources:
        if resource.id.url not in seen_urls:
            pool.resource_map[resource.id] = resource
            seen_urls.add(resource.id.url)

    return pool


async def run_upgrade(
    keywords_path: Optional[Path],
    searches_path: Optional[Path],
    extraction_path: Optional[Path],
    output_path: Path,
    config: IfetcherConfig,
    console: Console,
    verbose: bool,
) -> None:
    """Execute upgrade process.

    Parameters:
        keywords_path: Optional[Path] — path to old keywords file
        searches_path: Optional[Path] — path to old searches file
        extraction_path: Optional[Path] — path to old extraction file
        output_path: Path — where to write upgraded checkpoint
        config: IfetcherConfig — configuration
        console: Console — Rich console
        verbose: bool — show detailed output
    """
    # Validate at least one input provided
    if not any([keywords_path, searches_path, extraction_path]):
        console.print(
            "[red]Error:[/red] At least one input file must be provided "
            "(--keywords, --searches, or --extraction)"
        )
        raise SystemExit(1)

    # Load and detect formats
    loaded_data: dict[str, tuple[str, Any, list[dict]]] = {}

    for label, path in [
        ("keywords", keywords_path),
        ("searches", searches_path),
        ("extraction", extraction_path),
    ]:
        if path is None:
            continue

        if not path.exists():
            console.print(f"[red]Error:[/red] File not found: {path}")
            raise SystemExit(1)

        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            console.print(f"[red]Error:[/red] Invalid JSON in {path}: {e}")
            raise SystemExit(1)

        # Detect format
        detected = detect_format(data)
        if detected == "new":
            console.print(f"[yellow]Skipped:[/yellow] {path} is already in new format")
            continue

        if detected != label:
            console.print(
                f"[yellow]Warning:[/yellow] {path} appears to be {detected} format, "
                f"but was provided as --{label}"
            )

        # Parse based on detected format
        if detected == "keywords":
            loaded_data["keywords"] = parse_keywords(data)
        elif detected == "searches":
            loaded_data["searches"] = parse_searches(data)
        elif detected == "extraction":
            loaded_data["extraction"] = parse_extraction(data)

    # Check if any files were actually loaded
    if not loaded_data:
        console.print(
            "[yellow]No files to upgrade:[/yellow] All inputs are already in new format"
        )
        return

    # Determine topic (prefer extraction > searches > keywords)
    topic = (
        loaded_data.get("extraction", (None, None, None))[0]
        or loaded_data.get("searches", (None, None, None))[0]
        or loaded_data.get("keywords", (None, None, None))[0]
        or "Unknown Topic"
    )

    console.print(f"[bold]Upgrading checkpoint for topic:[/bold] {topic}")

    # Collect all resources (deduplicate later)
    all_resources: list[dict] = []
    for _, _, resources_list in loaded_data.values():
        all_resources.extend(resources_list)

    # Deduplicate by URL
    seen_urls = set()
    unique_resources = []
    for r in all_resources:
        if r["url"] not in seen_urls:
            unique_resources.append(r)
            seen_urls.add(r["url"])

    console.print(f"Found {len(unique_resources)} unique resources")

    # Enrich resources with DOI/dates
    enriched_resources = await enrich_resources(unique_resources, config, console)

    # Build ResourcePool
    resource_pool = build_resource_pool(enriched_resources)

    # Build checkpoint data dict (for proper rehydration)
    checkpoint_data = {
        "topic": topic,
        "resources": resource_pool.model_dump(mode="json"),
    }

    # Add stage data if present
    if "keywords" in loaded_data:
        keywords_stage = loaded_data["keywords"][1]
        checkpoint_data["keywords"] = keywords_stage.model_dump(mode="json")

    if "searches" in loaded_data:
        search_stage = loaded_data["searches"][1]
        checkpoint_data["search"] = search_stage.model_dump(mode="json")

    if "extraction" in loaded_data:
        extraction_dict = loaded_data["extraction"][1]
        # extraction_dict is already a dict, no need to dump
        checkpoint_data["extraction"] = extraction_dict

    # Build PipelineCheckpoint (this will trigger rehydration validators)
    checkpoint = PipelineCheckpoint.model_validate(checkpoint_data)

    # Validate checkpoint
    try:
        stage = checkpoint.stage
        console.print(f"[green]✓[/green] Created checkpoint at stage: {stage}")
    except ValueError as e:
        console.print(f"[red]Error:[/red] Invalid checkpoint: {e}")
        raise SystemExit(1)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_json = checkpoint.model_dump(mode="json")
    output_path.write_text(json.dumps(checkpoint_json, indent=2))

    console.print(f"[green]✓[/green] Wrote upgraded checkpoint to: {output_path}")

    # Show summary
    stages = []
    if checkpoint.keywords:
        stages.append(f"keywords ({len(checkpoint.keywords.terms)} terms)")
    if checkpoint.search:
        stages.append(f"search ({len(checkpoint.search.results)} results)")
    if checkpoint.extraction:
        stages.append(
            f"extraction ({checkpoint.extraction.metadata.pairs_accepted} accepted pairs)"
        )

    console.print(f"[bold]Stages included:[/bold] {', '.join(stages)}")
    console.print(
        f"[bold]Resources:[/bold] {len(resource_pool.resources)} with full content"
    )
