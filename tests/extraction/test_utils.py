"""Tests for extraction utility functions."""

import pytest

from interaction_finder.extraction.models import EntityMention, ProximalEntitySet
from interaction_finder.extraction.utils import (
    build_text_region,
    collect_relevant_text_for_quotes,
    find_substring_entities,
    identify_proximal_sets,
    make_entity_pair_key,
    normalize_for_comparison,
)
from interaction_finder.resources import ResourcePool


class TestNormalizeForComparison:
    """Tests for normalize_for_comparison function."""

    def test_basic_normalization(self):
        """Test basic lowercase normalization."""
        assert normalize_for_comparison("BRCA1") == "brca1"
        assert normalize_for_comparison("BrCa1") == "brca1"

    def test_handles_unicode(self):
        """Test unicode normalization."""
        # The function uses the resources normalize_text_for_matching
        text = "α-synuclein"
        normalized = normalize_for_comparison(text)
        assert isinstance(normalized, str)
        # Should be lowercase
        assert normalized.islower() or not normalized.isalpha()


class TestFindSubstringEntities:
    """Tests for find_substring_entities function."""

    def test_finds_simple_substring(self):
        """Test finding simple substring entity."""
        entities = {
            "BRCA1": EntityMention(
                kind="gene",
                name="BRCA1",
                aliases=["BRCA1"],
                quotes=[],
                reasoning="test",
            ),
            "BRCA": EntityMention(
                kind="gene", name="BRCA", aliases=["BRCA"], quotes=[], reasoning="test"
            ),
        }

        pairs = find_substring_entities(entities)
        assert len(pairs) == 1
        # BRCA1 is parent (longer), BRCA is child (shorter)
        assert pairs[0] == ("BRCA1", "BRCA")

    def test_finds_multiple_substrings(self):
        """Test finding multiple substring entities."""
        entities = {
            "BRCA1": EntityMention(
                kind="gene",
                name="BRCA1",
                aliases=["BRCA1"],
                quotes=[],
                reasoning="test",
            ),
            "BRCA": EntityMention(
                kind="gene", name="BRCA", aliases=["BRCA"], quotes=[], reasoning="test"
            ),
            "TP53": EntityMention(
                kind="gene", name="TP53", aliases=["TP53"], quotes=[], reasoning="test"
            ),
            "TP": EntityMention(
                kind="gene", name="TP", aliases=["TP"], quotes=[], reasoning="test"
            ),
        }

        pairs = find_substring_entities(entities)
        assert len(pairs) == 2
        # Check both pairs are found
        pair_set = set(pairs)
        assert ("BRCA1", "BRCA") in pair_set
        assert ("TP53", "TP") in pair_set

    def test_no_substrings_found(self):
        """Test when no substring relationships exist."""
        entities = {
            "BRCA1": EntityMention(
                kind="gene",
                name="BRCA1",
                aliases=["BRCA1"],
                quotes=[],
                reasoning="test",
            ),
            "TP53": EntityMention(
                kind="gene", name="TP53", aliases=["TP53"], quotes=[], reasoning="test"
            ),
        }

        pairs = find_substring_entities(entities)
        assert len(pairs) == 0

    def test_empty_entities(self):
        """Test with empty entities dict."""
        pairs = find_substring_entities({})
        assert len(pairs) == 0


class TestIdentifyProximalSets:
    """Tests for identify_proximal_sets function."""

    def setup_method(self):
        """Set up test resources."""
        self.pool = ResourcePool()
        # Create a resource with known chunk structure
        self.resource = self.pool.add(
            url="http://example.com",
            title="Test",
            document_text="Chunk0. Chunk1. Chunk2. Chunk3. Chunk4. Chunk5. Chunk6.",
        )
        # Manually set chunks for testing (each "ChunkN." is a chunk)
        self.resource.chunks = [
            (0, 7),  # "Chunk0."
            (8, 15),  # "Chunk1."
            (16, 23),  # "Chunk2."
            (24, 31),  # "Chunk3."
            (32, 39),  # "Chunk4."
            (40, 47),  # "Chunk5."
            (48, 55),  # "Chunk6."
        ]

    def test_identifies_simple_proximal_set(self):
        """Test identifying entities in adjacent chunks."""
        # Create quotes in chunks 0 and 1
        quote1 = self.resource.quote("Chunk0")
        quote2 = self.resource.quote("Chunk1")

        entities = {
            "Entity1": EntityMention(
                kind="gene",
                name="Entity1",
                aliases=["Entity1"],
                quotes=[quote1],
                reasoning="test",
            ),
            "Entity2": EntityMention(
                kind="gene",
                name="Entity2",
                aliases=["Entity2"],
                quotes=[quote2],
                reasoning="test",
            ),
        }

        sets = identify_proximal_sets(entities, threshold=2, resource=self.resource)

        assert len(sets) == 1
        assert len(sets[0].entities) == 2
        assert "Entity1" in sets[0].entities
        assert "Entity2" in sets[0].entities

    def test_separates_distant_entities(self):
        """Test that distant entities form separate sets."""
        # Create quotes in chunks 0 and 5 (too far apart with threshold=2)
        quote1 = self.resource.quote("Chunk0")
        quote2 = self.resource.quote("Chunk5")

        entities = {
            "Entity1": EntityMention(
                kind="gene",
                name="Entity1",
                aliases=["Entity1"],
                quotes=[quote1],
                reasoning="test",
            ),
            "Entity2": EntityMention(
                kind="gene",
                name="Entity2",
                aliases=["Entity2"],
                quotes=[quote2],
                reasoning="test",
            ),
        }

        sets = identify_proximal_sets(entities, threshold=2, resource=self.resource)

        # Should be no sets (need at least 2 entities per set)
        assert len(sets) == 0

    def test_expands_window_with_threshold(self):
        """Test window expansion with threshold."""
        # Create quotes in chunks 0, 2, and 4
        quote1 = self.resource.quote("Chunk0")
        quote2 = self.resource.quote("Chunk2")
        quote3 = self.resource.quote("Chunk4")

        entities = {
            "Entity1": EntityMention(
                kind="gene",
                name="Entity1",
                aliases=["Entity1"],
                quotes=[quote1],
                reasoning="test",
            ),
            "Entity2": EntityMention(
                kind="gene",
                name="Entity2",
                aliases=["Entity2"],
                quotes=[quote2],
                reasoning="test",
            ),
            "Entity3": EntityMention(
                kind="gene",
                name="Entity3",
                aliases=["Entity3"],
                quotes=[quote3],
                reasoning="test",
            ),
        }

        sets = identify_proximal_sets(entities, threshold=2, resource=self.resource)

        # All three should be in one set (window expands)
        assert len(sets) == 1
        assert len(sets[0].entities) == 3

    def test_empty_entities(self):
        """Test with no entities."""
        sets = identify_proximal_sets({}, threshold=2, resource=self.resource)
        assert len(sets) == 0


class TestBuildTextRegion:
    """Tests for build_text_region function."""

    def setup_method(self):
        """Set up test resource."""
        self.pool = ResourcePool()
        self.resource = self.pool.add(
            url="http://example.com",
            title="Test",
            document_text="Chunk0\nChunk1\nChunk2\nChunk3\nChunk4",
        )
        # Set up chunks
        self.resource.chunks = [
            (0, 6),  # "Chunk0"
            (7, 13),  # "Chunk1"
            (14, 20),  # "Chunk2"
            (21, 27),  # "Chunk3"
            (28, 34),  # "Chunk4"
        ]

    def test_builds_region_without_padding(self):
        """Test building region without padding."""
        text = build_text_region(self.resource, chunk_start=1, chunk_end=2, padding=0)
        assert "Chunk1" in text
        assert "Chunk2" in text
        assert "Chunk0" not in text
        assert "Chunk3" not in text

    def test_builds_region_with_padding(self):
        """Test building region with padding."""
        text = build_text_region(self.resource, chunk_start=1, chunk_end=2, padding=1)
        assert "Chunk0" in text  # Padding before
        assert "Chunk1" in text
        assert "Chunk2" in text
        assert "Chunk3" in text  # Padding after
        assert "Chunk4" not in text

    def test_respects_bounds(self):
        """Test that padding respects resource bounds."""
        text = build_text_region(self.resource, chunk_start=0, chunk_end=1, padding=10)
        # Should not crash, should include all available chunks
        assert "Chunk0" in text
        assert "Chunk4" in text


class TestCollectRelevantTextForQuotes:
    """Tests for collect_relevant_text_for_quotes function."""

    def setup_method(self):
        """Set up test resource."""
        self.pool = ResourcePool()
        self.resource = self.pool.add(
            url="http://example.com",
            title="Test",
            document_text="Chunk0\nChunk1\nChunk2\nChunk3\nChunk4",
        )
        self.resource.chunks = [
            (0, 6),
            (7, 13),
            (14, 20),
            (21, 27),
            (28, 34),
        ]

    def test_collects_text_for_single_quote(self):
        """Test collecting text for a single quote."""
        quote = self.resource.quote("Chunk1")
        text = collect_relevant_text_for_quotes(self.resource, [quote], padding=0)
        assert "Chunk1" in text

    def test_collects_text_for_multiple_quotes(self):
        """Test collecting text for multiple quotes."""
        quote1 = self.resource.quote("Chunk1")
        quote2 = self.resource.quote("Chunk3")
        text = collect_relevant_text_for_quotes(
            self.resource, [quote1, quote2], padding=0
        )
        # Should include both quotes
        assert "Chunk1" in text
        assert "Chunk3" in text

    def test_handles_empty_quotes(self):
        """Test handling empty quote list."""
        text = collect_relevant_text_for_quotes(self.resource, [], padding=0)
        assert text == ""


class TestMakeEntityPairKey:
    """Tests for make_entity_pair_key function."""

    def test_orders_by_kind(self):
        """Test that entities are ordered by kind."""
        entity1 = EntityMention(
            kind="gene",
            name="BRCA1",
            aliases=["BRCA1"],
            quotes=[],
            reasoning="test",
        )
        entity2 = EntityMention(
            kind="disease",
            name="breast cancer",
            aliases=["breast cancer"],
            quotes=[],
            reasoning="test",
        )

        # disease comes before gene alphabetically
        key = make_entity_pair_key(entity1, entity2)
        assert key.entity1_name == "breast cancer"
        assert key.entity2_name == "BRCA1"

        # Reverse order should give same key
        key2 = make_entity_pair_key(entity2, entity1)
        assert key == key2

    def test_orders_by_name_when_same_kind(self):
        """Test ordering by name when kinds are equal."""
        entity1 = EntityMention(
            kind="gene", name="BRCA2", aliases=["BRCA2"], quotes=[], reasoning="test"
        )
        entity2 = EntityMention(
            kind="gene", name="BRCA1", aliases=["BRCA1"], quotes=[], reasoning="test"
        )

        key = make_entity_pair_key(entity1, entity2)
        # BRCA1 comes before BRCA2
        assert key.entity1_name == "BRCA1"
        assert key.entity2_name == "BRCA2"

    def test_consistent_ordering(self):
        """Test that ordering is consistent regardless of input order."""
        entity1 = EntityMention(
            kind="gene", name="TP53", aliases=["TP53"], quotes=[], reasoning="test"
        )
        entity2 = EntityMention(
            kind="gene", name="BRCA1", aliases=["BRCA1"], quotes=[], reasoning="test"
        )

        key1 = make_entity_pair_key(entity1, entity2)
        key2 = make_entity_pair_key(entity2, entity1)

        assert key1 == key2
