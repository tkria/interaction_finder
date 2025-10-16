"""
Integration tests for query_results in investigation logger orchestration.

Tests verify that query_results is properly populated when log_query_construction
is called with both search_results and matches, simulating the orchestration flow.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from interaction_finder.search.base import SearchQuery, SearchResult, SearchResults
from interaction_finder.search.reverse.investigation_logger import InvestigationLogger
from interaction_finder.search.reverse.models import KnownResource, ResourceMatch


def parse_pretty_json_entries(content: str) -> List[Dict[str, Any]]:
    """
    Parse pretty-printed JSON entries from log content.

    Each entry starts with { at the beginning of a line and ends with } at
    the beginning of a line, with proper brace matching for nested structures.
    """
    entries = []
    lines = content.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == "{":
            # Start of new object
            obj_lines = [line]
            brace_count = 1
            i += 1
            while i < len(lines) and brace_count > 0:
                line = lines[i]
                obj_lines.append(line)
                # Count braces carefully
                for char in line:
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                i += 1
            # Parse complete object
            obj_text = "\n".join(obj_lines)
            entry = json.loads(obj_text)
            entries.append(entry)
        else:
            i += 1
    return entries


@pytest.fixture
def sample_resources() -> list[KnownResource]:
    """Create sample target resources."""
    return [
        KnownResource(
            pmid="12345678",
            url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
            hint_fields={"celltype": "CD8+ T cell", "marker": "CD8"},
        ),
        KnownResource(
            url="https://example.com/paper1",
            hint_fields={"celltype": "B cell", "marker": "CD19"},
        ),
        KnownResource(
            pmid="87654321",
            url="https://pubmed.ncbi.nlm.nih.gov/87654321/",
            hint_fields={"celltype": "NK cell", "marker": "CD56"},
        ),
    ]


@pytest.mark.asyncio
async def test_orchestration_flow_with_multiple_queries(
    tmp_path: Path,
    sample_resources: list[KnownResource],
):
    """
    Test orchestration flow with multiple queries simulating real usage.

    This integration test simulates how the ReverseSearcher would call
    log_query_construction after search and matching, verifying:
    1. Query construction entries contain query_results field
    2. query_results.results matches search results
    3. query_results.found_resources correlates with matches
    4. Indices are properly tracked across multiple queries
    """
    log_file = tmp_path / "investigation.jsonl"

    async with InvestigationLogger(log_file) as logger:
        # Simulate first query (finds first resource)
        query1 = SearchQuery(query="CD8 T cell marker", max_results=10)
        results1 = [
            SearchResult(
                title="CD8+ T cell markers",
                url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
                snippet="CD8 marker study",
                relevance_score=0.95,
                backend="pubmed",
                metadata={"pmid": "12345678"},
            ),
            SearchResult(
                title="Unrelated paper",
                url="https://example.com/unrelated",
                snippet="Different topic",
                relevance_score=0.60,
                backend="pubmed",
                metadata={},
            ),
        ]
        search_results1 = SearchResults(
            query=query1,
            results=results1,
            total_found=2,
            search_time=1.0,
            backend="pubmed",
        )

        # Match found first resource at index 0
        matches1 = [
            ResourceMatch(
                resource=sample_resources[0],
                search_result=results1[0],
                match_method="pmid",
                confidence=1.0,
                query_index=0,
            )
        ]

        # Log query construction with results and matches
        await logger.log_query_construction(
            query_index=0,
            constructor_type="direct",
            constructor_config={"max_keywords": 7},
            input_keywords=["CD8", "T cell", "marker"],
            keyword_scores=[0.9, 0.85, 0.80],
            final_query="CD8 T cell marker",
            construction_time=0.5,
            fallback_used=False,
            backend="pubmed",
            cumulative_coverage=0.33,
            search_results=search_results1,
            matches=matches1,
        )

        # Simulate second query (finds third resource)
        query2 = SearchQuery(query="NK cell CD56", max_results=10)
        results2 = [
            SearchResult(
                title="NK cell markers",
                url="https://pubmed.ncbi.nlm.nih.gov/87654321/",
                snippet="NK cell study",
                relevance_score=0.90,
                backend="pubmed",
                metadata={"pmid": "87654321"},
            ),
        ]
        search_results2 = SearchResults(
            query=query2,
            results=results2,
            total_found=1,
            search_time=1.0,
            backend="pubmed",
        )

        # Match found third resource at index 0 of this query's results
        matches2 = [
            ResourceMatch(
                resource=sample_resources[2],
                search_result=results2[0],
                match_method="pmid",
                confidence=1.0,
                query_index=1,
            )
        ]

        # Log second query construction
        await logger.log_query_construction(
            query_index=1,
            constructor_type="direct",
            constructor_config={"max_keywords": 7},
            input_keywords=["NK", "cell", "CD56"],
            keyword_scores=[0.88, 0.82, 0.78],
            final_query="NK cell CD56",
            construction_time=0.5,
            fallback_used=False,
            backend="pubmed",
            cumulative_coverage=0.67,
            search_results=search_results2,
            matches=matches2,
        )

        # Simulate third query (no matches)
        query3 = SearchQuery(query="B cell CD19", max_results=10)
        results3 = [
            SearchResult(
                title="Unrelated B cell paper",
                url="https://example.com/bcell",
                snippet="Different study",
                relevance_score=0.70,
                backend="pubmed",
                metadata={},
            ),
        ]
        search_results3 = SearchResults(
            query=query3,
            results=results3,
            total_found=1,
            search_time=1.0,
            backend="pubmed",
        )

        # No matches for this query
        matches3 = []

        # Log third query construction
        await logger.log_query_construction(
            query_index=2,
            constructor_type="direct",
            constructor_config={"max_keywords": 7},
            input_keywords=["B", "cell", "CD19"],
            keyword_scores=[0.87, 0.81, 0.75],
            final_query="B cell CD19",
            construction_time=0.5,
            fallback_used=False,
            backend="pubmed",
            cumulative_coverage=0.67,  # No change
            search_results=search_results3,
            matches=matches3,
        )

    # Parse log and verify orchestration flow
    content = log_file.read_text()
    entries = parse_pretty_json_entries(content)

    query_construction_entries = [
        e for e in entries if e["stage"] == "query_construction"
    ]
    assert len(query_construction_entries) == 3

    # Verify first query entry
    qc1 = query_construction_entries[0]
    assert qc1["query_index"] == 0
    assert qc1["cumulative_coverage"] == 0.33
    qr1 = qc1["query_results"]
    assert qr1 is not None
    # Two results: one with PMID, one without
    assert len(qr1["results"]) == 2
    assert qr1["results"][0] == "PMID:12345678"
    assert qr1["results"][1] == "https://example.com/unrelated"
    # One match at index 0
    assert len(qr1["found_resources"]) == 1
    assert qr1["found_resources"][0]["resource"] == "PMID:12345678"
    assert qr1["found_resources"][0]["index"] == 0

    # Verify second query entry
    qc2 = query_construction_entries[1]
    assert qc2["query_index"] == 1
    assert qc2["cumulative_coverage"] == 0.67
    qr2 = qc2["query_results"]
    assert qr2 is not None
    # One result with PMID
    assert len(qr2["results"]) == 1
    assert qr2["results"][0] == "PMID:87654321"
    # One match at index 0
    assert len(qr2["found_resources"]) == 1
    assert qr2["found_resources"][0]["resource"] == "PMID:87654321"
    assert qr2["found_resources"][0]["index"] == 0

    # Verify third query entry (no matches)
    qc3 = query_construction_entries[2]
    assert qc3["query_index"] == 2
    assert qc3["cumulative_coverage"] == 0.67  # No change
    qr3 = qc3["query_results"]
    assert qr3 is not None
    # One result but no PMID
    assert len(qr3["results"]) == 1
    assert qr3["results"][0] == "https://example.com/bcell"
    # No matches
    assert len(qr3["found_resources"]) == 0


@pytest.mark.asyncio
async def test_match_referencing_nonexistent_result(
    tmp_path: Path,
    sample_resources: list[KnownResource],
    caplog,
):
    """
    Test edge case: match references result URL not in search_results.

    This indicates a matcher bug. The logger should log a warning and skip
    the match rather than failing.
    """
    log_file = tmp_path / "investigation.jsonl"

    async with InvestigationLogger(log_file) as logger:
        # Create search results
        query = SearchQuery(query="test", max_results=10)
        results = [
            SearchResult(
                title="Valid result",
                url="https://example.com/valid",
                snippet="Test",
                relevance_score=0.9,
                backend="pubmed",
                metadata={},
            ),
        ]
        search_results = SearchResults(
            query=query,
            results=results,
            total_found=1,
            search_time=1.0,
            backend="pubmed",
        )

        # Create match with URL not in results (simulates matcher bug)
        bogus_result = SearchResult(
            title="Bogus result",
            url="https://example.com/bogus",  # NOT in results list
            snippet="Bogus",
            relevance_score=0.8,
            backend="pubmed",
            metadata={},
        )
        matches = [
            ResourceMatch(
                resource=sample_resources[0],
                search_result=bogus_result,  # References non-existent result
                match_method="url",
                confidence=0.9,
                query_index=0,
            )
        ]

        # Log query construction (should handle gracefully)
        await logger.log_query_construction(
            query_index=0,
            constructor_type="direct",
            constructor_config={},
            input_keywords=["test"],
            keyword_scores=[0.9],
            final_query="test",
            construction_time=0.5,
            fallback_used=False,
            backend="pubmed",
            cumulative_coverage=0.0,
            search_results=search_results,
            matches=matches,
        )

    # Verify warning was logged
    assert any(
        "Match for resource" in record.message
        and "not in results list" in record.message
        for record in caplog.records
    )

    # Verify log entry was created but match was skipped
    content = log_file.read_text()
    entries = parse_pretty_json_entries(content)
    qc_entry = [e for e in entries if e["stage"] == "query_construction"][0]

    qr = qc_entry["query_results"]
    assert qr is not None
    assert len(qr["results"]) == 1
    # Match should be skipped (not in found_resources)
    assert len(qr["found_resources"]) == 0
