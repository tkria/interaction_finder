"""Tests for CLI upgrade command."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from interaction_finder.cli_upgrade import (
    build_resource_pool,
    detect_format,
    infer_entity_types,
    infer_permitted_pairs,
    parse_extraction,
    parse_keywords,
    parse_searches,
)
from interaction_finder.resources import Resource, ResourceId


class TestFormatDetection:
    """Test format detection logic."""

    def test_detect_new_format_with_keywords(self):
        """New format has keywords field."""
        data = {"topic": "test", "resources": [], "keywords": {}}
        assert detect_format(data) == "new"

    def test_detect_new_format_with_search(self):
        """New format has search field."""
        data = {"topic": "test", "resources": [], "search": {}}
        assert detect_format(data) == "new"

    def test_detect_new_format_with_extraction(self):
        """New format has extraction field."""
        data = {"topic": "test", "resources": [], "extraction": {}}
        assert detect_format(data) == "new"

    def test_detect_keywords_format(self):
        """Old keywords has terms, scores, coverage_assessment."""
        data = {
            "terms": ["term1"],
            "scores": [0.9],
            "coverage_assessment": "good",
            "resources": [],
        }
        assert detect_format(data) == "keywords"

    def test_detect_searches_format(self):
        """Old searches has results, query_results."""
        data = {
            "results": [],
            "query_results": {},
            "queries": [],
            "resources": [],
        }
        assert detect_format(data) == "searches"

    def test_detect_extraction_format(self):
        """Old extraction has judgments, metadata."""
        data = {
            "judgments": [],
            "metadata": {},
            "resources": [],
        }
        assert detect_format(data) == "extraction"

    def test_detect_unknown_format(self):
        """Unknown format raises ValueError."""
        data = {"foo": "bar"}
        with pytest.raises(ValueError, match="Unknown checkpoint format"):
            detect_format(data)


class TestKeywordsParsing:
    """Test parsing of old keywords format."""

    def test_parse_keywords_minimal(self):
        """Parse minimal keywords data."""
        data = {
            "terms": ["term1", "term2"],
            "scores": [0.9, 0.8],
            "total_documents_processed": 5,
            "rounds_completed": 2,
            "coverage_assessment": "Good coverage achieved with comprehensive review of topic",
            "resources": [
                {"url": "https://example.com/1", "title": "Doc 1", "text": "text1"},
                {"url": "https://example.com/2", "title": "Doc 2", "text": "text2"},
            ],
        }

        topic, keywords_stage, resources_list = parse_keywords(data)

        assert topic == "Unknown Topic"  # No topic in minimal data
        assert keywords_stage.terms == ["term1", "term2"]
        assert keywords_stage.scores == [0.9, 0.8]
        assert keywords_stage.total_documents_processed == 5
        assert keywords_stage.rounds_completed == 2
        assert (
            keywords_stage.coverage_assessment
            == "Good coverage achieved with comprehensive review of topic"
        )
        assert keywords_stage.resource_urls == [
            "https://example.com/1",
            "https://example.com/2",
        ]
        assert len(resources_list) == 2

    def test_parse_keywords_with_topic(self):
        """Parse keywords data with topic field."""
        data = {
            "topic": "My Research Topic",
            "terms": ["term1"],
            "scores": [0.9],
            "total_documents_processed": 1,
            "rounds_completed": 1,
            "coverage_assessment": "Coverage assessment completed successfully with good results",
            "resources": [{"url": "https://example.com/1"}],
        }

        topic, _, _ = parse_keywords(data)
        assert topic == "My Research Topic"

    def test_parse_keywords_with_resource_map(self):
        """Parse keywords data with old resource_map format."""
        data = {
            "terms": ["term1"],
            "scores": [0.9],
            "total_documents_processed": 1,
            "rounds_completed": 1,
            "coverage_assessment": "Coverage assessment completed successfully with good results",
            "resources": {
                "resource_map": {
                    "key1": {"url": "https://example.com/1"},
                    "key2": {"url": "https://example.com/2"},
                }
            },
        }

        _, keywords_stage, resources_list = parse_keywords(data)
        assert len(keywords_stage.resource_urls) == 2
        assert len(resources_list) == 2


class TestSearchesParsing:
    """Test parsing of old searches format."""

    def test_parse_searches(self):
        """Parse searches data."""
        data = {
            "topic": "Test Topic",
            "results": [
                {"url": "https://example.com/1", "title": "Result 1", "snippet": "..."}
            ],
            "queries": ["query1", "query2"],
            "query_results": {
                "query1": ["https://example.com/1"],
                "query2": [],
            },
            "keyphrases": ["phrase1", "phrase2"],
            "rounds_completed": 1,
            "resources": [{"url": "https://example.com/1", "title": "Doc 1"}],
        }

        topic, search_stage, resources_list = parse_searches(data)

        assert topic == "Test Topic"
        assert len(search_stage.results) == 1
        assert search_stage.queries == ["query1", "query2"]
        assert search_stage.query_results == {
            "query1": ["https://example.com/1"],
            "query2": [],
        }
        assert search_stage.keyphrases == ["phrase1", "phrase2"]
        assert search_stage.rounds_completed == 1
        assert search_stage.resource_urls == ["https://example.com/1"]
        assert len(resources_list) == 1


class TestExtractionParsing:
    """Test parsing of old extraction format."""

    def test_infer_entity_types(self):
        """Infer entity types from judgments."""
        judgments = [
            {
                "entity1": {"kind": "gene", "name": "BRCA1"},
                "entity2": {"kind": "disease", "name": "cancer"},
            },
            {
                "entity1": {"kind": "gene", "name": "TP53"},
                "entity2": {"kind": "disease", "name": "tumor"},
            },
            {
                "entity1": {"kind": "protein", "name": "p53"},
                "entity2": {"kind": "gene", "name": "TP53"},
            },
        ]

        entity_types = infer_entity_types(judgments)
        assert entity_types == ["disease", "gene", "protein"]  # Sorted

    def test_infer_permitted_pairs(self):
        """Infer permitted pairs from judgments."""
        judgments = [
            {
                "entity1": {"kind": "gene", "name": "BRCA1"},
                "entity2": {"kind": "disease", "name": "cancer"},
            },
            {
                "entity1": {"kind": "gene", "name": "TP53"},
                "entity2": {"kind": "gene", "name": "MDM2"},
            },
        ]

        permitted_pairs = infer_permitted_pairs(judgments)
        assert permitted_pairs == {
            "disease": ["gene"],
            "gene": ["disease", "gene"],
        }

    def test_parse_extraction_with_inference(self):
        """Parse extraction data, inferring entity config."""
        data = {
            "topic": "Cancer Research",
            "resources": [{"url": "https://example.com/1"}],
            "judgments": [
                {
                    "entity1": {"kind": "gene", "name": "BRCA1", "aliases": []},
                    "entity2": {"kind": "disease", "name": "cancer", "aliases": []},
                    "relationship": "associated_with",
                    "assessments": [],
                    "accepted": True,
                    "confidence": "high",
                    "reasoning": "test",
                }
            ],
            "metadata": {
                "topic": "Cancer Research",
                "resource_count": 1,
                "total_entities_found": 2,
                "entities_after_validation": 2,
                "entities_merged": 0,
                "merge_cache_hits": 0,
                "merge_cache_misses": 0,
                "proximal_sets_found": 1,
                "total_pairs_found": 1,
                "pairs_accepted": 1,
                "pairs_rejected": 0,
                "quotes_validated": 5,
                "quotes_failed": 0,
            },
        }

        topic, extraction_dict, resources_list = parse_extraction(data)

        assert topic == "Cancer Research"
        assert extraction_dict["target_entity_types"] == ["disease", "gene"]
        assert extraction_dict["permitted_pairs"] == {
            "disease": ["gene"],
            "gene": ["disease"],
        }
        assert len(extraction_dict["judgments"]) == 1
        assert len(resources_list) == 1

    def test_parse_extraction_with_existing_config(self):
        """Parse extraction data with existing entity config."""
        data = {
            "topic": "Test",
            "target_entity_types": ["gene", "protein"],
            "permitted_pairs": {"gene": ["protein"], "protein": ["gene"]},
            "resources": [],
            "judgments": [],
            "metadata": {
                "topic": "Test",
                "resource_count": 0,
                "total_entities_found": 0,
                "entities_after_validation": 0,
                "entities_merged": 0,
                "merge_cache_hits": 0,
                "merge_cache_misses": 0,
                "proximal_sets_found": 0,
                "total_pairs_found": 0,
                "pairs_accepted": 0,
                "pairs_rejected": 0,
                "quotes_validated": 0,
                "quotes_failed": 0,
            },
        }

        _, extraction_dict, _ = parse_extraction(data)

        # Should use provided config, not infer
        assert extraction_dict["target_entity_types"] == ["gene", "protein"]
        assert extraction_dict["permitted_pairs"] == {
            "gene": ["protein"],
            "protein": ["gene"],
        }


class TestResourcePoolBuilding:
    """Test resource pool building."""

    def test_build_resource_pool(self):
        """Build resource pool from enriched resources."""
        resources = [
            Resource(
                id=ResourceId(url="https://example.com/1", counter=1),
                title="Doc 1",
                text="text1",
                chunks=[(0, 5)],
                doi="10.1234/test1",
                publication_date="2024-01-01",
            ),
            Resource(
                id=ResourceId(url="https://example.com/2", counter=2),
                title="Doc 2",
                text="text2",
                chunks=[(0, 5)],
                doi=None,
                publication_date=None,
            ),
        ]

        pool = build_resource_pool(resources)

        assert len(pool.resources) == 2
        assert "https://example.com/1" in pool
        assert "https://example.com/2" in pool

    def test_build_resource_pool_deduplicates(self):
        """Resource pool deduplicates by URL."""
        resources = [
            Resource(
                id=ResourceId(url="https://example.com/1", counter=1),
                title="Doc 1",
                text="text1",
            ),
            Resource(
                id=ResourceId(url="https://example.com/1", counter=2),
                title="Doc 1 duplicate",
                text="text1",
            ),
            Resource(
                id=ResourceId(url="https://example.com/2", counter=3),
                title="Doc 2",
                text="text2",
            ),
        ]

        pool = build_resource_pool(resources)

        # Should deduplicate to 2 unique URLs
        assert len(pool.resources) == 2
        assert "https://example.com/1" in pool
        assert "https://example.com/2" in pool
