"""
Unit tests for two-stage pipeline data models.

Tests cover KeywordExtractionResult and QueryConstructionContext dataclasses,
verifying field validation, edge cases, and JSON serialization for investigation logging.
"""

import json
from dataclasses import asdict

import pytest

from interaction_finder.search.reverse.models import (
    KeywordExtractionResult,
    QueryConstructionContext,
)


# KeywordExtractionResult tests


def test_keyword_extraction_result_creation_with_scores():
    """Test creating KeywordExtractionResult with all fields populated."""
    result = KeywordExtractionResult(
        keywords=["BRCA1", "breast cancer", "mutation"],
        scores=[0.95, 0.87, 0.82],
        extractor="yake",
        extraction_time=0.15,
    )
    assert result.keywords == ["BRCA1", "breast cancer", "mutation"]
    assert result.scores == [0.95, 0.87, 0.82]
    assert result.extractor == "yake"
    assert result.extraction_time == 0.15


def test_keyword_extraction_result_creation_without_scores():
    """Test creating KeywordExtractionResult with None scores."""
    result = KeywordExtractionResult(
        keywords=["gene", "protein", "pathway"],
        scores=None,
        extractor="rake",
        extraction_time=0.08,
    )
    assert result.keywords == ["gene", "protein", "pathway"]
    assert result.scores is None
    assert result.extractor == "rake"
    assert result.extraction_time == 0.08


def test_keyword_extraction_result_empty_keywords():
    """Test creating KeywordExtractionResult with empty keyword list."""
    # Dataclasses don't validate, so this should work
    result = KeywordExtractionResult(
        keywords=[],
        scores=None,
        extractor="yake",
        extraction_time=0.0,
    )
    assert result.keywords == []
    assert result.scores is None


def test_keyword_extraction_result_single_keyword():
    """Test creating KeywordExtractionResult with single keyword."""
    result = KeywordExtractionResult(
        keywords=["BRCA1"],
        scores=[0.99],
        extractor="tfidf",
        extraction_time=0.05,
    )
    assert result.keywords == ["BRCA1"]
    assert result.scores == [0.99]


def test_keyword_extraction_result_mismatched_lengths():
    """Test creating KeywordExtractionResult with mismatched keyword/score lengths."""
    # Dataclasses don't validate, so this should work but is semantically wrong
    result = KeywordExtractionResult(
        keywords=["BRCA1", "TP53", "mutation"],
        scores=[0.95, 0.87],  # Only 2 scores for 3 keywords
        extractor="yake",
        extraction_time=0.10,
    )
    # This is allowed but should be avoided - future validation could catch this
    assert len(result.keywords) == 3
    assert len(result.scores) == 2


def test_keyword_extraction_result_json_serialization():
    """Test KeywordExtractionResult serializes to JSON for investigation logging."""
    result = KeywordExtractionResult(
        keywords=["BRCA1", "breast cancer"],
        scores=[0.95, 0.87],
        extractor="yake",
        extraction_time=0.15,
    )
    # Convert to dict using dataclasses.asdict
    result_dict = asdict(result)
    assert result_dict == {
        "keywords": ["BRCA1", "breast cancer"],
        "scores": [0.95, 0.87],
        "extractor": "yake",
        "extraction_time": 0.15,
    }
    # Verify it can be JSON serialized
    json_str = json.dumps(result_dict)
    assert json_str is not None
    # Verify round-trip
    parsed = json.loads(json_str)
    assert parsed["keywords"] == ["BRCA1", "breast cancer"]
    assert parsed["scores"] == [0.95, 0.87]
    assert parsed["extractor"] == "yake"


def test_keyword_extraction_result_json_serialization_no_scores():
    """Test KeywordExtractionResult with None scores serializes correctly."""
    result = KeywordExtractionResult(
        keywords=["gene", "disease"],
        scores=None,
        extractor="rake",
        extraction_time=0.08,
    )
    result_dict = asdict(result)
    assert result_dict["scores"] is None
    # JSON serialization handles None
    json_str = json.dumps(result_dict)
    parsed = json.loads(json_str)
    assert parsed["scores"] is None


def test_keyword_extraction_result_zero_extraction_time():
    """Test KeywordExtractionResult with zero extraction time."""
    result = KeywordExtractionResult(
        keywords=["test"],
        scores=None,
        extractor="cached",
        extraction_time=0.0,
    )
    assert result.extraction_time == 0.0


def test_keyword_extraction_result_float_scores():
    """Test KeywordExtractionResult handles various float score formats."""
    result = KeywordExtractionResult(
        keywords=["k1", "k2", "k3"],
        scores=[1.0, 0.5, 0.001],
        extractor="yake",
        extraction_time=0.1,
    )
    assert result.scores == [1.0, 0.5, 0.001]


# QueryConstructionContext tests


def test_query_construction_context_full():
    """Test creating QueryConstructionContext with all fields populated."""
    context = QueryConstructionContext(
        keywords=["BRCA1", "breast cancer", "mutation"],
        keyword_scores=[0.95, 0.87, 0.82],
        hint_terms=["mammary epithelial cell", "TP53"],
        backend="pubmed",
        resource_content="Full paper text about BRCA1...",
        extractor_used="yake",
    )
    assert context.keywords == ["BRCA1", "breast cancer", "mutation"]
    assert context.keyword_scores == [0.95, 0.87, 0.82]
    assert context.hint_terms == ["mammary epithelial cell", "TP53"]
    assert context.backend == "pubmed"
    assert context.resource_content == "Full paper text about BRCA1..."
    assert context.extractor_used == "yake"


def test_query_construction_context_minimal():
    """Test creating QueryConstructionContext with optional fields set to None/empty."""
    context = QueryConstructionContext(
        keywords=["gene", "disease"],
        keyword_scores=None,
        hint_terms=[],
        backend="pubmed",
        resource_content=None,
        extractor_used="rake",
    )
    assert context.keywords == ["gene", "disease"]
    assert context.keyword_scores is None
    assert context.hint_terms == []
    assert context.backend == "pubmed"
    assert context.resource_content is None
    assert context.extractor_used == "rake"


def test_query_construction_context_empty_keywords():
    """Test creating QueryConstructionContext with empty keyword list."""
    context = QueryConstructionContext(
        keywords=[],
        keyword_scores=None,
        hint_terms=["cell type"],
        backend="pubmed",
        resource_content=None,
        extractor_used="yake",
    )
    assert context.keywords == []


def test_query_construction_context_empty_hint_terms():
    """Test creating QueryConstructionContext with empty hint_terms."""
    context = QueryConstructionContext(
        keywords=["BRCA1"],
        keyword_scores=[0.95],
        hint_terms=[],
        backend="pubmed",
        resource_content=None,
        extractor_used="yake",
    )
    assert context.hint_terms == []


def test_query_construction_context_different_backends():
    """Test creating QueryConstructionContext with different backend values."""
    for backend in ["pubmed", "perplexica", "openai", "custom_backend"]:
        context = QueryConstructionContext(
            keywords=["test"],
            keyword_scores=None,
            hint_terms=[],
            backend=backend,
            resource_content=None,
            extractor_used="yake",
        )
        assert context.backend == backend


def test_query_construction_context_json_serialization():
    """Test QueryConstructionContext serializes to JSON for investigation logging."""
    context = QueryConstructionContext(
        keywords=["BRCA1", "mutation"],
        keyword_scores=[0.95, 0.80],
        hint_terms=["T cell", "CD8A"],
        backend="pubmed",
        resource_content="Sample content",
        extractor_used="yake",
    )
    context_dict = asdict(context)
    assert context_dict == {
        "keywords": ["BRCA1", "mutation"],
        "keyword_scores": [0.95, 0.80],
        "hint_terms": ["T cell", "CD8A"],
        "backend": "pubmed",
        "resource_content": "Sample content",
        "extractor_used": "yake",
    }
    # Verify JSON serialization
    json_str = json.dumps(context_dict)
    parsed = json.loads(json_str)
    assert parsed["keywords"] == ["BRCA1", "mutation"]
    assert parsed["backend"] == "pubmed"
    assert parsed["extractor_used"] == "yake"


def test_query_construction_context_json_serialization_minimal():
    """Test QueryConstructionContext with None/empty fields serializes correctly."""
    context = QueryConstructionContext(
        keywords=["gene"],
        keyword_scores=None,
        hint_terms=[],
        backend="pubmed",
        resource_content=None,
        extractor_used="rake",
    )
    context_dict = asdict(context)
    assert context_dict["keyword_scores"] is None
    assert context_dict["hint_terms"] == []
    assert context_dict["resource_content"] is None
    # JSON serialization
    json_str = json.dumps(context_dict)
    parsed = json.loads(json_str)
    assert parsed["keyword_scores"] is None
    assert parsed["resource_content"] is None


def test_query_construction_context_large_content():
    """Test QueryConstructionContext handles large resource content."""
    large_content = "A" * 10000  # 10KB string
    context = QueryConstructionContext(
        keywords=["test"],
        keyword_scores=None,
        hint_terms=[],
        backend="pubmed",
        resource_content=large_content,
        extractor_used="yake",
    )
    assert len(context.resource_content) == 10000


def test_query_construction_context_unicode_content():
    """Test QueryConstructionContext handles Unicode characters."""
    context = QueryConstructionContext(
        keywords=["α-synuclein", "β-amyloid", "γ-secretase"],
        keyword_scores=[0.9, 0.8, 0.7],
        hint_terms=["神经退行性疾病", "Alzheimer's"],
        backend="pubmed",
        resource_content="研究内容 with mixed 文字",
        extractor_used="llm",
    )
    assert context.keywords[0] == "α-synuclein"
    assert "神经退行性疾病" in context.hint_terms
    # Verify JSON handles Unicode
    context_dict = asdict(context)
    json_str = json.dumps(context_dict, ensure_ascii=False)
    parsed = json.loads(json_str)
    assert parsed["keywords"][0] == "α-synuclein"


# Integration tests


def test_pipeline_data_flow():
    """Test typical data flow from extraction to construction."""
    # Stage 1: Extraction produces KeywordExtractionResult
    extraction_result = KeywordExtractionResult(
        keywords=["BRCA1", "breast cancer", "mutation", "PARP"],
        scores=[0.95, 0.87, 0.82, 0.75],
        extractor="yake",
        extraction_time=0.15,
    )

    # Stage 2: Create context from extraction result + additional info
    construction_context = QueryConstructionContext(
        keywords=extraction_result.keywords,
        keyword_scores=extraction_result.scores,
        hint_terms=["mammary epithelial cell"],
        backend="pubmed",
        resource_content=None,  # Not available in this case
        extractor_used=extraction_result.extractor,
    )

    # Verify data flows correctly
    assert construction_context.keywords == extraction_result.keywords
    assert construction_context.keyword_scores == extraction_result.scores
    assert construction_context.extractor_used == extraction_result.extractor


def test_pipeline_data_flow_no_scores():
    """Test data flow when extractor doesn't provide scores."""
    # Stage 1: RAKE extraction (no scores)
    extraction_result = KeywordExtractionResult(
        keywords=["gene", "protein", "pathway", "interaction"],
        scores=None,
        extractor="rake",
        extraction_time=0.08,
    )

    # Stage 2: Create context without scores
    construction_context = QueryConstructionContext(
        keywords=extraction_result.keywords,
        keyword_scores=extraction_result.scores,
        hint_terms=["CD8+ T cell", "CD8A"],
        backend="pubmed",
        resource_content="Full text...",
        extractor_used=extraction_result.extractor,
    )

    # Verify None scores are preserved
    assert construction_context.keyword_scores is None
    assert construction_context.extractor_used == "rake"


def test_serialization_round_trip_both_models():
    """Test both models can be serialized and deserialized for logging."""
    # Create instances
    extraction = KeywordExtractionResult(
        keywords=["k1", "k2"],
        scores=[0.9, 0.8],
        extractor="yake",
        extraction_time=0.1,
    )
    context = QueryConstructionContext(
        keywords=["k1", "k2"],
        keyword_scores=[0.9, 0.8],
        hint_terms=["hint"],
        backend="pubmed",
        resource_content="content",
        extractor_used="yake",
    )

    # Serialize both to JSON
    extraction_json = json.dumps(asdict(extraction))
    context_json = json.dumps(asdict(context))

    # Deserialize
    extraction_parsed = json.loads(extraction_json)
    context_parsed = json.loads(context_json)

    # Verify key fields
    assert extraction_parsed["keywords"] == ["k1", "k2"]
    assert extraction_parsed["extractor"] == "yake"
    assert context_parsed["keywords"] == ["k1", "k2"]
    assert context_parsed["backend"] == "pubmed"
    assert context_parsed["extractor_used"] == "yake"
