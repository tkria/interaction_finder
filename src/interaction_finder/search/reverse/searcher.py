"""
Orchestrates complete reverse search workflow with query generation and matching.

This module implements ReverseSearcher, which coordinates the iterative process of
finding known resources through search queries. It generates initial queries,
executes them, matches results to targets, tracks coverage, evaluates stopping
criteria, and generates refinement queries when needed.

Order: imports → ReverseSearcher class → helper methods
"""

import time
from typing import List, Set, Any, Optional, Tuple, TYPE_CHECKING

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)

from .models import (
    KnownResource,
    ReverseSearchResult,
    ReverseSearchSession,
    ReverseSearchConfig,
    ReverseSearchError,
)
from .query_generator import QueryGenerator
from .matchers import ResourceMatcher
from interaction_finder.search.base import SearchBackend, SearchQuery, SearchResults
from interaction_finder.search.cache import SearchCache

if TYPE_CHECKING:
    from .investigation_logger import InvestigationLogger, QueryConstructionDetails


class ReverseSearcher:
    """
    Orchestrates reverse search to find known resources via query generation.

    Workflow:
    1. Generate initial queries from target resources
    2. Execute queries via search backend
    3. Match results to target resources
    4. Update coverage tracking
    5. Check stopping criteria (coverage/consecutive zeros/max queries)
    6. If not stopped: generate refinement queries and repeat from step 2

    Stopping criteria:
    - Coverage ≥ target (default 95%)
    - Consecutive queries with zero new finds (default 3)
    - Maximum queries reached (default 100)

    Example:
        >>> config = ReverseSearchConfig(coverage_target=0.95)
        >>> searcher = ReverseSearcher(config, backend, cache, fetcher)
        >>> resources = [KnownResource(pmid="123", url="https://..."), ...]
        >>> session = await searcher.search(resources, verbose=True)
        >>> print(f"Found {session.found_count}/{len(resources)} resources")
    """

    def __init__(
        self,
        config: ReverseSearchConfig,
        search_backend: SearchBackend,
        cache: SearchCache,
        fetcher: Any,
        investigation_logger: Optional["InvestigationLogger"] = None,
    ):
        """
        Initialize reverse searcher.

        Parameters:
            config: ReverseSearchConfig - Configuration
            search_backend: SearchBackend - Search backend for query execution
            cache: SearchCache - Cache for search results
            fetcher: PageFetcher - For content fetching in query generation
            investigation_logger: Optional[InvestigationLogger] - Logger for investigation tracking
        """
        self.config = config
        self.backend = search_backend
        self.cache = cache
        self.fetcher = fetcher
        self.inv_logger = investigation_logger
        # Console for progress display
        self.console = Console()
        # Initialize components (pass console and investigation logger)
        self.query_generator = QueryGenerator(
            config,
            backend_name=self.backend.backend_name,
            fetcher=fetcher,
            console=self.console,
            investigation_logger=investigation_logger,
        )
        self.matcher = ResourceMatcher(
            config, fetcher, investigation_logger=investigation_logger
        )

    async def search(
        self,
        target_resources: List[KnownResource],
        verbose: bool = False,
    ) -> ReverseSearchSession:
        """
        Execute reverse search to find target resources.

        Parameters:
            target_resources: List[KnownResource] - Resources to find
            verbose: bool - Show progress with Rich console

        Returns:
            ReverseSearchSession - Complete session with queries, matches, metrics

        Raises:
            ValueError: If target_resources is empty
            ReverseSearchError: If query generation or execution fails critically
        """
        if not target_resources:
            raise ValueError("target_resources cannot be empty")

        start_time = time.time()
        query_results = []
        matches = []
        unfound = set(target_resources)
        consecutive_zero_finds = 0
        # Log session start
        if self.inv_logger:
            await self.inv_logger.log_session_start(
                target_resources,
                self.config,
                self.backend.backend_name,
            )
        # Phase 1: Generate initial queries (with content fetching if verbose)
        if verbose:
            self.console.print(
                f"[blue]Phase 1:[/blue] Fetching content for {len(target_resources)} resources..."
            )

        try:
            initial_queries = await self.query_generator.generate_initial_queries(
                target_resources
            )
        except Exception as e:
            # Log error if logger available
            if self.inv_logger:
                await self.inv_logger.log_error(
                    "query_generation",
                    e,
                    target_resources,
                    "abort",
                )
            raise ReverseSearchError(
                f"Failed to generate initial queries: {e}",
                context={"resource_count": len(target_resources)},
            )

        if verbose:
            self.console.print(
                f"[green]✓[/green] Generated {len(initial_queries)} initial queries"
            )
            self.console.print(
                "\n[blue]Phase 2:[/blue] Executing queries to find resources..."
            )

        # Setup progress display for query execution phase
        progress = None
        progress_task = None
        if verbose:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console,
            )
            progress.start()
            progress_task = progress.add_task(
                "Finding resources via queries...",
                total=len(target_resources),
            )

        query_index = 0
        all_queries: List[Tuple[str, "QueryConstructionDetails"]] = (
            initial_queries.copy()
        )
        # Phase 2: Execute queries until stopping criteria met
        while query_index < len(all_queries):
            # Check stopping criteria before executing next query
            if self._should_stop(
                unfound, target_resources, consecutive_zero_finds, query_index
            ):
                break

            # Unpack query tuple
            query_text, construction_details = all_queries[query_index]
            # Execute query with caching
            query_start_time = time.time()
            try:
                search_results = await self._execute_query_cached(query_text)
            except Exception as e:
                # Log error but continue (may succeed with next query)
                if verbose:
                    self.console.print(
                        f"[yellow]Query {query_index} failed: {e}[/yellow]"
                    )
                query_index += 1
                continue
            # Calculate search time
            search_time = time.time() - query_start_time
            cache_hit = getattr(search_results, "from_cache", False)
            # Match results to target resources
            new_matches = await self.matcher.match_results(
                search_results, unfound, query_index
            )
            matches.extend(new_matches)
            # Update tracking
            newly_found = {m.resource for m in new_matches}
            unfound -= newly_found
            coverage = 1.0 - (len(unfound) / len(target_resources))
            # Log consolidated query entry if investigation logger available
            if self.inv_logger:
                try:
                    await self.inv_logger.log_query(
                        query_index=query_index,
                        construction_details=construction_details,
                        query_text=query_text,
                        backend=self.backend.backend_name,
                        cache_hit=cache_hit,
                        search_time=search_time,
                        results=search_results,
                        matches=new_matches,
                        cumulative_coverage=coverage,
                    )
                except Exception as log_error:
                    # Non-fatal: log warning but continue search
                    if verbose:
                        self.console.print(
                            f"[yellow]Warning: Failed to write investigation log for query {query_index}: {log_error}[/yellow]"
                        )
            # Record result
            result = ReverseSearchResult(
                query=query_text,
                query_index=query_index,
                search_results=search_results,
                resources_found=list(newly_found),
                new_finds=len(newly_found),
                cumulative_coverage=coverage,
                search_time=search_results.search_time or 0.0,
                backend=self.backend.backend_name,
            )
            query_results.append(result)
            # Update consecutive zero finds tracking
            if len(newly_found) == 0:
                consecutive_zero_finds += 1
            else:
                consecutive_zero_finds = 0
            # Update progress display and log query info
            if verbose and progress and progress_task is not None:
                found_count = len(target_resources) - len(unfound)
                num_results = (
                    len(search_results.results) if search_results.results else 0
                )
                progress.update(
                    progress_task,
                    completed=found_count,
                    description=f"Query {query_index + 1}: {num_results} results, {len(newly_found)} new ({coverage:.1%} coverage)",
                )

            query_index += 1
            # Phase 3: Generate refinement queries if needed
            if query_index >= len(all_queries) and len(unfound) > 0:
                # Check if we should continue refining
                if self._should_stop(
                    unfound, target_resources, consecutive_zero_finds, query_index
                ):
                    break

                try:
                    # Extract query strings for duplicate checking
                    previous_query_strings = [q[0] for q in all_queries]
                    refinement_queries = (
                        await self.query_generator.generate_refinement_queries(
                            list(unfound),
                            previous_query_strings,
                        )
                    )
                    if refinement_queries:
                        all_queries.extend(refinement_queries)
                    else:
                        # No more queries to generate
                        break
                except Exception as e:
                    if verbose:
                        self.console.print(
                            f"[yellow]Refinement generation failed: {e}[/yellow]"
                        )
                    break
        # Determine stopping reason
        coverage = 1.0 - (len(unfound) / len(target_resources))
        if coverage >= self.config.coverage_target:
            stopping_reason = "coverage_achieved"
        elif consecutive_zero_finds >= self.config.consecutive_zero_limit:
            stopping_reason = "consecutive_zero_finds"
        else:
            stopping_reason = "max_queries"
        # Stop progress display
        if verbose and progress:
            progress.stop()
            # Print final summary
            self.console.print("\n[green]Reverse search complete![/green]")
            self.console.print(
                f"Found {len(target_resources) - len(unfound)} / {len(target_resources)} resources ({coverage:.1%} coverage)"
            )
            self.console.print(f"Stopping reason: {stopping_reason}")
            if unfound:
                self.console.print(
                    f"[yellow]{len(unfound)} resources not found[/yellow]"
                )
        # Build session result
        session = ReverseSearchSession(
            target_resources=target_resources,
            query_results=query_results,
            matches=matches,
            total_queries=len(query_results),
            final_coverage=coverage,
            found_count=len(target_resources) - len(unfound),
            unfound_resources=list(unfound),
            total_time=time.time() - start_time,
            stopping_reason=stopping_reason,
        )
        # Log session end
        if self.inv_logger:
            await self.inv_logger.log_session_end(session)

        return session

    def _should_stop(
        self,
        unfound: Set[KnownResource],
        target_resources: List[KnownResource],
        consecutive_zero_finds: int,
        current_query_index: int,
    ) -> bool:
        """
        Check if stopping criteria are met.

        Criteria:
        1. Coverage ≥ target (e.g., 95%)
        2. Consecutive zero finds ≥ limit (e.g., 3)
        3. Current query index ≥ max queries (e.g., 100)

        Parameters:
            unfound: Set[KnownResource] - Resources not yet found
            target_resources: List[KnownResource] - All target resources
            consecutive_zero_finds: int - Count of consecutive queries with no finds
            current_query_index: int - Index of next query to execute

        Returns:
            bool - True if should stop, False otherwise
        """
        # Calculate coverage
        coverage = 1.0 - (len(unfound) / len(target_resources))
        # Check coverage target
        if coverage >= self.config.coverage_target:
            return True
        # Check consecutive zero finds
        if consecutive_zero_finds >= self.config.consecutive_zero_limit:
            return True
        # Check max queries
        if current_query_index >= self.config.max_queries:
            return True

        return False

    async def _execute_query_cached(self, query_text: str) -> SearchResults:
        """
        Execute query with caching.

        Parameters:
            query_text: str - Query to execute

        Returns:
            SearchResults - Results from backend or cache

        Raises:
            Exception - If backend execution fails
        """
        # Create SearchQuery
        search_query = SearchQuery(
            query=query_text,
            max_results=self.config.max_results_per_query,
            filters={"sort_by": self.config.sort_by},
        )
        # Check cache first
        cache_context = {"reverse_search": True}
        cached_results = await self.cache.get(
            search_query,
            self.backend.backend_name,
            cache_context,
        )

        if cached_results:
            return cached_results
        # Cache miss: execute query via backend
        results = await self.backend.search(search_query)
        # Cache results
        await self.cache.set(
            search_query,
            self.backend.backend_name,
            results,
            cache_context,
        )

        return results
