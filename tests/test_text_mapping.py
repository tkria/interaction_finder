"""Tests for TextPositionMapper."""

import re

import pytest

from interaction_finder.text_mapping import TextPositionMapper


class TestTextPositionMapper:
    """Tests for TextPositionMapper class."""

    def test_basic_initialization(self):
        """Test basic initialization with valid offsets."""
        source = "hello world"
        target = "HELLO WORLD"
        offsets = [(0, 0), (5, 5), (11, 11)]

        mapper = TextPositionMapper(source, target, offsets)

        assert mapper.source == source
        assert mapper.target == target
        assert mapper._offsets == offsets

    def test_empty_offsets_raises_error(self):
        """Test that empty offsets list raises ValueError."""
        with pytest.raises(ValueError, match="Offsets list cannot be empty"):
            TextPositionMapper("hello", "HELLO", [])

    def test_unsorted_offsets_raises_error(self):
        """Test that unsorted offsets raise ValueError."""
        with pytest.raises(ValueError, match="must be sorted by source position"):
            TextPositionMapper("hello", "HELLO", [(5, 5), (0, 0)])

    def test_targetpos_identity_mapping(self):
        """Test position mapping when source and target are identical."""
        text = "hello world"  # 11 chars
        # Identity mapping: delta = 0 throughout
        offsets = [(0, 0)]
        mapper = TextPositionMapper(text, text, offsets)

        assert mapper.targetpos(0) == 0
        assert mapper.targetpos(5) == 5
        assert mapper.targetpos(11) == 11

    def test_targetpos_simple_expansion(self):
        """Test position mapping with text expansion (e.g., Greek letters)."""
        # "α" → "alpha" expands text
        source = "hello alpha world"  # 17 chars
        target = "hello α world"  # 13 chars
        offsets = [
            (0, 0),  # Delta = 0: 'h' → 'h'
            (6, 0),  # Delta = 0: 'a' in "alpha" → 'α'
            (11, -4),  # Delta = -4: 'w' → 'w' (alpha contracted to α)
        ]
        mapper = TextPositionMapper(source, target, offsets)

        assert mapper.targetpos(0) == 0  # Start
        assert mapper.targetpos(6) == 6  # 'a' in "alpha"
        assert mapper.targetpos(11) == 7  # 'w' in "world"
        assert mapper.targetpos(17) == 13  # End (uses delta = -4)

    def test_targetpos_out_of_bounds(self):
        """Test that out of bounds position raises ValueError."""
        mapper = TextPositionMapper("hello", "HELLO", [(0, 0)])

        with pytest.raises(ValueError, match="out of bounds"):
            mapper.targetpos(-1)

        with pytest.raises(ValueError, match="out of bounds"):
            mapper.targetpos(100)

    def test_targetpos_edge_case_zero(self):
        """Test position 0 handling."""
        mapper = TextPositionMapper("hello", "HELLO", [(0, 0)])
        assert mapper.targetpos(0) == 0

    def test_targetpos_edge_case_end(self):
        """Test position at end of text."""
        source = "hello"
        target = "HELLO"
        offsets = [(0, 0)]
        mapper = TextPositionMapper(source, target, offsets)

        assert mapper.targetpos(5) == 5

    def test_targetspan_basic(self):
        """Test span mapping."""
        source = "hello world"
        target = "HELLO WORLD"
        offsets = [(0, 0)]  # Identity mapping: delta = 0
        mapper = TextPositionMapper(source, target, offsets)

        start, end = mapper.targetspan(0, 5)
        assert start == 0
        assert end == 5

    def test_targetspan_with_expansion(self):
        """Test span mapping with text expansion."""
        source = "hello alpha world"
        target = "hello α world"
        offsets = [(0, 0), (6, 0), (11, -4)]
        mapper = TextPositionMapper(source, target, offsets)

        # Span covering "alpha"
        start, end = mapper.targetspan(6, 11)
        assert start == 6  # 'a'
        assert end == 7  # After 'α'

    def test_find_string_match(self):
        """Test finding string pattern in source."""
        source = "hello world"
        target = "HELLO WORLD"
        # Sparse offsets: delta = 0 throughout (identity mapping)
        offsets = [(0, 0)]
        mapper = TextPositionMapper(source, target, offsets)

        result = mapper.find("world")
        assert result == (6, 11)

    def test_find_string_not_found(self):
        """Test find returns None when pattern not found."""
        mapper = TextPositionMapper("hello", "HELLO", [(0, 0)])
        assert mapper.find("xyz") is None

    def test_find_regex_match(self):
        """Test finding regex pattern in source."""
        source = "hello world"
        target = "HELLO WORLD"
        offsets = [(0, 0)]
        mapper = TextPositionMapper(source, target, offsets)

        result = mapper.find(re.compile(r"\bworld\b"))
        assert result == (6, 11)

    def test_find_regex_not_found(self):
        """Test regex find returns None when not found."""
        mapper = TextPositionMapper("hello", "HELLO", [(0, 0)])
        assert mapper.find(re.compile(r"\d+")) is None

    def test_find_with_expansion(self):
        """Test find with Greek letter expansion."""
        source = "hello alpha world"
        target = "hello α world"
        offsets = [(0, 0), (6, 0), (11, -4)]
        mapper = TextPositionMapper(source, target, offsets)

        # Search for "alpha" in source, get target coordinates
        result = mapper.find("alpha")
        assert result == (6, 7)  # Maps to 'α' position in target

    def test_findall_string_multiple_matches(self):
        """Test finding all occurrences of string pattern."""
        source = "hello hello world"
        target = "HELLO HELLO WORLD"
        offsets = [(0, 0), (17, 17)]
        mapper = TextPositionMapper(source, target, offsets)

        results = mapper.findall("hello")
        assert len(results) == 2
        assert results[0] == (0, 5)
        assert results[1] == (6, 11)

    def test_findall_string_no_matches(self):
        """Test findall returns empty list when no matches."""
        mapper = TextPositionMapper("hello", "HELLO", [(0, 0)])
        assert mapper.findall("xyz") == []

    def test_findall_regex_multiple_matches(self):
        """Test finding all regex matches."""
        source = "cat bat rat"
        target = "CAT BAT RAT"
        offsets = [(0, 0)]  # Identity mapping: delta = 0
        mapper = TextPositionMapper(source, target, offsets)

        results = mapper.findall(re.compile(r"\b\w+at\b"))
        assert len(results) == 3
        assert results[0] == (0, 3)  # cat
        assert results[1] == (4, 7)  # bat
        assert results[2] == (8, 11)  # rat

    def test_findall_with_expansion(self):
        """Test findall with text expansion."""
        source = "alpha beta alpha"  # 16 chars
        target = "α β α"  # 5 chars
        # alpha(0-5)→α(0-1): delta=0 at 0, then -4 at 5
        # " beta"(5-10)→" β"(1-3): delta=-4 at 5
        # " alpha"(10-16)→" α"(3-5): delta=-7 at 11
        offsets = [(0, 0), (5, -4), (11, -7)]
        mapper = TextPositionMapper(source, target, offsets)

        results = mapper.findall("alpha")
        assert len(results) == 2
        assert results[0] == (
            0,
            1,
        )  # First α: pos 0-5 with delta 0 → 0-1 (but 1 uses delta -4, so 0-1)
        assert results[1] == (4, 9)  # Second α: pos 11-16 with delta -7 → 4-9

    def test_findall_overlapping_positions(self):
        """Test findall with overlapping search positions."""
        source = "aaa"
        target = "AAA"
        offsets = [(0, 0)]  # Identity mapping
        mapper = TextPositionMapper(source, target, offsets)

        # Find all 'aa' - should find at position 0 and 1
        results = mapper.findall("aa")
        assert len(results) == 2
        assert results[0] == (0, 2)
        assert results[1] == (1, 3)

    def test_complex_normalization_scenario(self):
        """Test realistic scientific text normalization scenario."""
        # Original: "TGF-α receptor"
        # Normalized: "tgf alpha receptor"
        source = "tgf alpha receptor"  # 18 chars
        target = "TGF-α receptor"  # 14 chars

        # Delta changes when "alpha" contracts to "α"
        # Positions 0-3: "tgf " → "TGF-", delta = 0
        # Positions 4-8: "alpha" → "α", delta = 0 at 4
        # Positions 9-17: " receptor" shifts left by 4, delta = -4 at 9
        offsets = [
            (0, 0),  # Delta = 0
            (4, 0),  # Delta = 0 (start of "alpha")
            (9, -4),  # Delta = -4 (after "alpha" contracted)
        ]

        mapper = TextPositionMapper(source, target, offsets)

        # Search for "alpha" in normalized, get original position
        result = mapper.find("alpha")
        assert result == (4, 5)  # Maps to 'α' in "TGF-α" (position 4, ends at 5)

        # Search for "receptor"
        result = mapper.find("receptor")
        assert result == (6, 14)  # Maps to "receptor" in target

    def test_empty_source_text(self):
        """Test with empty source text."""
        mapper = TextPositionMapper("", "", [(0, 0)])
        assert mapper.targetpos(0) == 0

    def test_single_character_text(self):
        """Test with single character text."""
        mapper = TextPositionMapper("a", "A", [(0, 0)])
        assert mapper.targetpos(0) == 0
        assert mapper.targetpos(1) == 1

    def test_position_between_offsets(self):
        """Test position lookup between explicit offset entries."""
        # Sparse offsets - not every character has an entry
        source = "hello world"
        target = "HELLO WORLD"
        offsets = [(0, 0)]  # Delta = 0 throughout
        mapper = TextPositionMapper(source, target, offsets)

        # Position 3 should use the offset at or before it (position 0 with delta 0)
        # So it returns 3 + 0 = 3
        result = mapper.targetpos(3)
        assert result == 3

    def test_regex_case_insensitive(self):
        """Test regex with case insensitive flag."""
        source = "Hello World"
        target = "HELLO WORLD"
        offsets = [(0, 0), (11, 11)]
        mapper = TextPositionMapper(source, target, offsets)

        result = mapper.find(re.compile(r"hello", re.IGNORECASE))
        assert result == (0, 5)

    def test_regex_multiline(self):
        """Test regex with multiline text."""
        source = "line1\nline2\nline3"
        target = "LINE1\nLINE2\nLINE3"
        offsets = [(0, 0), (17, 17)]
        mapper = TextPositionMapper(source, target, offsets)

        results = mapper.findall(re.compile(r"^line\d", re.MULTILINE))
        assert len(results) == 3
