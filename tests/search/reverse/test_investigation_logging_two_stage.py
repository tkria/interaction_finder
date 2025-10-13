"""
Tests for two-stage investigation logging (keyword extraction + query construction).

Tests cover:
- KeywordExtractionEntry structure and validation
- QueryConstructionEntry structure and validation
- log_keyword_extraction() method
- log_query_construction() method
- Session ID and query_index consistency across stages
- JSON serialization of new entry types
- Migration utility (synthesize_legacy_format)
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from pydantic import ValidationError

from interaction_finder.search.reverse.investigation_logger import (
    InvestigationLogger,
    KeywordDetail,
    KeywordExtractionEntry,
    QueryConstructionEntry,
)
from interaction_finder.search.reverse.utils import synthesize_legacy_format


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
def temp_log_file(tmp_path: Path) -> Path:
    """Create temporary log file path."""
    return tmp_path / "test_two_stage.jsonl"


@pytest.mark.asyncio
async def test_keyword_extraction_entry_structure(temp_log_file: Path):
    """Test KeywordExtractionEntry structure and logging."""
    async with InvestigationLogger(temp_log_file) as logger:
        await logger.log_keyword_extraction(
            query_index=0,
            extractor_type="yake",
            extractor_config={"top_n": 10, "max_ngram": 3},
            keywords=[
                {"keyword": "CD8", "score": 0.85},
                {"keyword": "T cell", "score": 0.72},
                {"keyword": "marker", "score": 0.68},
            ],
            extraction_time=0.15,
            input_resources=["PMID:12345678"],
            content_source="metadata",
        )

    # Parse and verify
    content = temp_log_file.read_text()
    entries = parse_pretty_json_entries(content)
    assert len(entries) == 1

    entry = entries[0]
    assert entry["stage"] == "keyword_extraction"
    assert entry["query_index"] == 0
    assert entry["extractor_type"] == "yake"
    assert entry["extractor_config"] == {"top_n": 10, "max_ngram": 3}
    assert len(entry["keywords"]) == 3
    assert entry["keywords"][0]["keyword"] == "CD8"
    assert entry["keywords"][0]["score"] == 0.85
    assert entry["extraction_time"] == 0.15
    assert entry["input_resources"] == ["PMID:12345678"]
    assert entry["content_source"] == "metadata"
    assert "session_id" in entry
    assert "timestamp" in entry


@pytest.mark.asyncio
async def test_query_construction_entry_structure(temp_log_file: Path):
    """Test QueryConstructionEntry structure and logging."""
    async with InvestigationLogger(temp_log_file) as logger:
        await logger.log_query_construction(
            query_index=0,
            constructor_type="direct",
            constructor_config={"operator": "OR", "quote_terms": True},
            input_keywords=["CD8", "T cell", "marker"],
            keyword_scores=[0.85, 0.72, 0.68],
            hint_terms=["immune", "response"],
            final_query='"CD8" OR "T cell" OR "marker" OR "immune" OR "response"',
            construction_time=0.02,
            fallback_used=False,
            backend="pubmed",
            cumulative_coverage=0.33,
        )

    # Parse and verify
    content = temp_log_file.read_text()
    entries = parse_pretty_json_entries(content)
    assert len(entries) == 1

    entry = entries[0]
    assert entry["stage"] == "query_construction"
    assert entry["query_index"] == 0
    assert entry["constructor_type"] == "direct"
    assert entry["constructor_config"] == {"operator": "OR", "quote_terms": True}
    assert entry["input_keywords"] == ["CD8", "T cell", "marker"]
    assert entry["keyword_scores"] == [0.85, 0.72, 0.68]
    assert entry["hint_terms"] == ["immune", "response"]
    assert (
        entry["final_query"]
        == '"CD8" OR "T cell" OR "marker" OR "immune" OR "response"'
    )
    assert entry["construction_time"] == 0.02
    assert entry["fallback_used"] is False
    assert entry["backend"] == "pubmed"
    assert entry["cumulative_coverage"] == 0.33
    assert "session_id" in entry
    assert "timestamp" in entry


@pytest.mark.asyncio
async def test_two_stage_logging_consistency(temp_log_file: Path):
    """Test that both stages share same session_id and query_index."""
    async with InvestigationLogger(temp_log_file) as logger:
        # Log extraction stage
        await logger.log_keyword_extraction(
            query_index=0,
            extractor_type="yake",
            extractor_config={},
            keywords=[{"keyword": "CD8", "score": 0.85}],
            extraction_time=0.15,
            input_resources=["PMID:12345678"],
            content_source="metadata",
        )

        # Log construction stage with same query_index
        await logger.log_query_construction(
            query_index=0,
            constructor_type="direct",
            constructor_config={},
            input_keywords=["CD8"],
            keyword_scores=[0.85],
            hint_terms=[],
            final_query='"CD8"',
            construction_time=0.02,
            fallback_used=False,
            backend="pubmed",
            cumulative_coverage=0.0,
        )

    # Parse and verify
    content = temp_log_file.read_text()
    entries = parse_pretty_json_entries(content)
    assert len(entries) == 2

    extraction_entry = entries[0]
    construction_entry = entries[1]

    # Verify stages
    assert extraction_entry["stage"] == "keyword_extraction"
    assert construction_entry["stage"] == "query_construction"

    # Verify consistency
    assert extraction_entry["session_id"] == construction_entry["session_id"]
    assert extraction_entry["query_index"] == construction_entry["query_index"]


@pytest.mark.asyncio
async def test_multiple_queries_two_stage(temp_log_file: Path):
    """Test logging multiple queries with two-stage format."""
    async with InvestigationLogger(temp_log_file) as logger:
        # Query 0
        await logger.log_keyword_extraction(
            query_index=0,
            extractor_type="yake",
            extractor_config={},
            keywords=[{"keyword": "CD8", "score": 0.85}],
            extraction_time=0.15,
            input_resources=["PMID:12345678"],
            content_source="metadata",
        )
        await logger.log_query_construction(
            query_index=0,
            constructor_type="direct",
            constructor_config={},
            input_keywords=["CD8"],
            keyword_scores=[0.85],
            hint_terms=[],
            final_query='"CD8"',
            construction_time=0.02,
            fallback_used=False,
            backend="pubmed",
            cumulative_coverage=0.33,
        )

        # Query 1
        await logger.log_keyword_extraction(
            query_index=1,
            extractor_type="yake",
            extractor_config={},
            keywords=[{"keyword": "B cell", "score": 0.90}],
            extraction_time=0.12,
            input_resources=["PMID:87654321"],
            content_source="metadata",
        )
        await logger.log_query_construction(
            query_index=1,
            constructor_type="direct",
            constructor_config={},
            input_keywords=["B cell"],
            keyword_scores=[0.90],
            hint_terms=[],
            final_query='"B cell"',
            construction_time=0.03,
            fallback_used=False,
            backend="pubmed",
            cumulative_coverage=0.67,
        )

    # Parse and verify
    content = temp_log_file.read_text()
    entries = parse_pretty_json_entries(content)
    assert len(entries) == 4

    # Verify ordering: extraction → construction → extraction → construction
    assert entries[0]["stage"] == "keyword_extraction"
    assert entries[0]["query_index"] == 0
    assert entries[1]["stage"] == "query_construction"
    assert entries[1]["query_index"] == 0

    assert entries[2]["stage"] == "keyword_extraction"
    assert entries[2]["query_index"] == 1
    assert entries[3]["stage"] == "query_construction"
    assert entries[3]["query_index"] == 1


@pytest.mark.asyncio
async def test_keyword_extraction_with_null_scores(temp_log_file: Path):
    """Test keyword extraction with None scores (LLM extractor)."""
    async with InvestigationLogger(temp_log_file) as logger:
        await logger.log_keyword_extraction(
            query_index=0,
            extractor_type="llm",
            extractor_config={"model": "openai:gpt-4o-mini"},
            keywords=[
                {"keyword": "CD8", "score": None},
                {"keyword": "T cell", "score": None},
            ],
            extraction_time=1.25,
            input_resources=["PMID:12345678"],
            content_source="metadata",
        )

    # Parse and verify
    content = temp_log_file.read_text()
    entries = parse_pretty_json_entries(content)
    assert len(entries) == 1

    entry = entries[0]
    assert entry["extractor_type"] == "llm"
    assert entry["keywords"][0]["score"] is None
    assert entry["keywords"][1]["score"] is None


@pytest.mark.asyncio
async def test_query_construction_with_fallback(temp_log_file: Path):
    """Test query construction with fallback flag set."""
    async with InvestigationLogger(temp_log_file) as logger:
        await logger.log_query_construction(
            query_index=0,
            constructor_type="direct",
            constructor_config={},
            input_keywords=["CD8"],
            keyword_scores=[0.85],
            hint_terms=[],
            final_query='"CD8"',
            construction_time=0.02,
            fallback_used=True,
            backend="pubmed",
            cumulative_coverage=0.0,
        )

    # Parse and verify
    content = temp_log_file.read_text()
    entries = parse_pretty_json_entries(content)
    assert len(entries) == 1

    entry = entries[0]
    assert entry["fallback_used"] is True


def test_keyword_extraction_entry_validation():
    """Test Pydantic validation for KeywordExtractionEntry."""
    # Valid entry
    valid_entry = KeywordExtractionEntry(
        timestamp="2025-10-10T12:00:00+00:00",
        session_id="test-session",
        query_index=0,
        extractor_type="yake",
        extractor_config={"top_n": 10},
        keywords=[KeywordDetail(keyword="CD8", score=0.85)],
        extraction_time=0.15,
        input_resources=["PMID:12345678"],
        content_source="metadata",
    )
    assert valid_entry.stage == "keyword_extraction"
    assert valid_entry.extractor_type == "yake"
    assert len(valid_entry.keywords) == 1


def test_query_construction_entry_validation():
    """Test Pydantic validation for QueryConstructionEntry."""
    # Valid entry
    valid_entry = QueryConstructionEntry(
        timestamp="2025-10-10T12:00:01+00:00",
        session_id="test-session",
        query_index=0,
        constructor_type="direct",
        constructor_config={"operator": "OR"},
        input_keywords=["CD8"],
        keyword_scores=[0.85],
        hint_terms=[],
        final_query='"CD8"',
        construction_time=0.02,
        fallback_used=False,
        backend="pubmed",
        cumulative_coverage=0.33,
    )
    assert valid_entry.stage == "query_construction"
    assert valid_entry.constructor_type == "direct"
    assert valid_entry.cumulative_coverage == 0.33

    # Invalid entry - coverage out of range
    with pytest.raises(ValidationError):
        QueryConstructionEntry(
            timestamp="2025-10-10T12:00:01+00:00",
            session_id="test-session",
            query_index=0,
            constructor_type="direct",
            constructor_config={},
            input_keywords=["CD8"],
            keyword_scores=[0.85],
            hint_terms=[],
            final_query='"CD8"',
            construction_time=0.02,
            fallback_used=False,
            backend="pubmed",
            cumulative_coverage=1.5,  # Invalid: > 1.0
        )


def test_synthesize_legacy_format_basic():
    """Test synthesize_legacy_format with basic entries."""
    extraction = {
        "stage": "keyword_extraction",
        "session_id": "abc123",
        "timestamp": "2025-10-10T12:00:00Z",
        "query_index": 0,
        "extractor_type": "yake",
        "extractor_config": {"top_n": 10},
        "keywords": [{"keyword": "CD8", "score": 0.85}],
        "extraction_time": 0.15,
        "input_resources": ["PMID:12345678"],
        "content_source": "metadata",
    }

    construction = {
        "stage": "query_construction",
        "session_id": "abc123",
        "timestamp": "2025-10-10T12:00:01Z",
        "query_index": 0,
        "constructor_type": "direct",
        "constructor_config": {"operator": "OR"},
        "input_keywords": ["CD8"],
        "keyword_scores": [0.85],
        "hint_terms": ["marker"],
        "final_query": '"CD8" OR "marker"',
        "construction_time": 0.02,
        "fallback_used": False,
        "backend": "pubmed",
        "cumulative_coverage": 0.33,
    }

    legacy = synthesize_legacy_format(extraction, construction)

    # Verify legacy format structure
    assert legacy["stage"] == "query_generation"
    assert legacy["session_id"] == "abc123"
    assert legacy["query_index"] == 0
    assert legacy["query_type"] == "initial"
    assert legacy["extractor_type"] == "yake"
    assert legacy["keywords"] == [{"keyword": "CD8", "score": 0.85}]
    assert legacy["input_resources"] == ["PMID:12345678"]
    assert legacy["hint_terms"] == ["marker"]
    assert legacy["final_query"] == '"CD8" OR "marker"'
    assert legacy["cumulative_coverage"] == 0.33
    assert legacy["timestamp"] == "2025-10-10T12:00:01Z"  # Construction timestamp


def test_synthesize_legacy_format_validation():
    """Test synthesize_legacy_format validates input stages."""
    extraction = {
        "stage": "keyword_extraction",
        "session_id": "abc123",
        "timestamp": "2025-10-10T12:00:00Z",
        "query_index": 0,
        "extractor_type": "yake",
        "keywords": [],
        "input_resources": [],
    }

    construction = {
        "stage": "query_construction",
        "session_id": "abc123",
        "timestamp": "2025-10-10T12:00:01Z",
        "query_index": 0,
        "final_query": "test",
        "hint_terms": [],
        "cumulative_coverage": 0.0,
    }

    # Valid call should work
    legacy = synthesize_legacy_format(extraction, construction)
    assert legacy["stage"] == "query_generation"

    # Wrong stage in extraction
    with pytest.raises(ValueError, match="Expected keyword_extraction stage"):
        bad_extraction = {**extraction, "stage": "wrong_stage"}
        synthesize_legacy_format(bad_extraction, construction)

    # Wrong stage in construction
    with pytest.raises(ValueError, match="Expected query_construction stage"):
        bad_construction = {**construction, "stage": "wrong_stage"}
        synthesize_legacy_format(extraction, bad_construction)

    # Mismatched query_index
    with pytest.raises(ValueError, match="Query indices do not match"):
        bad_construction = {**construction, "query_index": 1}
        synthesize_legacy_format(extraction, bad_construction)


@pytest.mark.asyncio
async def test_json_serialization_round_trip(temp_log_file: Path):
    """Test that entries serialize and deserialize correctly."""
    async with InvestigationLogger(temp_log_file) as logger:
        await logger.log_keyword_extraction(
            query_index=0,
            extractor_type="yake",
            extractor_config={"top_n": 10},
            keywords=[{"keyword": "CD8", "score": 0.85}],
            extraction_time=0.15,
            input_resources=["PMID:12345678"],
            content_source="metadata",
        )

    # Read back and parse
    content = temp_log_file.read_text()
    entries = parse_pretty_json_entries(content)
    assert len(entries) == 1

    # Verify can reconstruct Pydantic model
    entry_dict = entries[0]
    reconstructed = KeywordExtractionEntry(**entry_dict)
    assert reconstructed.stage == "keyword_extraction"
    assert reconstructed.query_index == 0
    assert reconstructed.extractor_type == "yake"


@pytest.mark.asyncio
async def test_empty_keywords_handling(temp_log_file: Path):
    """Test logging with empty keywords list."""
    async with InvestigationLogger(temp_log_file) as logger:
        await logger.log_keyword_extraction(
            query_index=0,
            extractor_type="yake",
            extractor_config={},
            keywords=[],  # Empty keywords
            extraction_time=0.10,
            input_resources=["PMID:12345678"],
            content_source="metadata",
        )

        await logger.log_query_construction(
            query_index=0,
            constructor_type="direct",
            constructor_config={},
            input_keywords=[],  # Empty keywords
            keyword_scores=[],
            hint_terms=[],
            final_query="",  # Empty query
            construction_time=0.01,
            fallback_used=False,
            backend="pubmed",
            cumulative_coverage=0.0,
        )

    # Parse and verify
    content = temp_log_file.read_text()
    entries = parse_pretty_json_entries(content)
    assert len(entries) == 2

    assert entries[0]["keywords"] == []
    assert entries[1]["input_keywords"] == []
    assert entries[1]["keyword_scores"] == []
