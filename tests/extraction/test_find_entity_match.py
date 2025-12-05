"""Tests for find_entity_match function.

Tests cover the progressive matching strategy:
- Stage 0: Direct canonical match (handles contested variants)
- Stage 1: Exact variant matches
- Stage 2: Fuzzy matching with validation
"""

import pytest
from interaction_finder.extraction.entity_matching import (
    extract_entity_variants,
    find_entity_match,
)


class TestDirectCanonicalMatch:
    """Test Stage 0: Direct canonical name matching."""

    def test_exact_canonical_match_lowercase_to_uppercase(self):
        """Query matching canonical name exactly (case-insensitive) should match directly."""
        entities = {
            "BRCA1": extract_entity_variants("BRCA1", None),
        }

        match = find_entity_match("brca1", entities, allow_fuzzy=False)

        assert match is not None
        assert match.canonical == "BRCA1"
        assert match.match_penalty == 0  # PENALTY_EXACT
        # Direct canonical match uses original variant with spec=0
        assert match.matched_variant.speculation == 0

    def test_direct_canonical_bypasses_contested_variants(self):
        """Direct canonical match should work even when variant is contested.

        Regression test: When both "Familial PAH" and "Heritable/familial PAH"
        exist as candidates, the variant "familial pah" becomes contested.
        But a query exactly matching "Familial PAH" should still match via
        direct canonical matching.
        """
        entities = {
            "Familial pulmonary arterial hypertension": extract_entity_variants(
                "Familial pulmonary arterial hypertension", None
            ),
            "Heritable/familial pulmonary arterial hypertension": extract_entity_variants(
                "Heritable/familial pulmonary arterial hypertension", None
            ),
        }

        # This would fail without Stage 0 because "familial pulmonary arterial
        # hypertension" is a contested variant (maps to both canonicals)
        match = find_entity_match(
            "Familial pulmonary arterial hypertension", entities, allow_fuzzy=True
        )

        assert match is not None
        assert match.canonical == "Familial pulmonary arterial hypertension"
        assert match.match_penalty == 0  # PENALTY_EXACT


class TestExactVariantMatch:
    """Test Stage 1: Exact variant matching."""

    def test_matches_slash_alternation_variant(self):
        """Query matching a slash-alternation variant should match."""
        entities = {
            "Heritable/familial pulmonary arterial hypertension": extract_entity_variants(
                "Heritable/familial pulmonary arterial hypertension", None
            ),
        }

        # "Heritable pulmonary arterial hypertension" is a slash-alternation variant
        match = find_entity_match(
            "Heritable pulmonary arterial hypertension", entities, allow_fuzzy=False
        )

        assert match is not None
        assert match.canonical == "Heritable/familial pulmonary arterial hypertension"
        assert match.match_penalty == 0  # PENALTY_EXACT

    def test_matches_parenthetical_variant(self):
        """Query matching parenthetical content should match."""
        entities = {
            "ACTB (β-Actin)": extract_entity_variants("ACTB (β-Actin)", None),
        }

        # "β-Actin" is extracted from parenthetical
        match = find_entity_match("β-Actin", entities, allow_fuzzy=False)

        assert match is not None
        assert match.canonical == "ACTB (β-Actin)"
        assert match.match_penalty == 0  # PENALTY_EXACT


class TestFuzzyMatch:
    """Test Stage 2: Fuzzy matching with validation."""

    def test_fuzzy_match_obvious_variant(self):
        """Fuzzy matching with obvious spelling variant should work."""
        entities = {
            "hemorrhagic": extract_entity_variants("hemorrhagic", None),
        }

        match = find_entity_match("haemorrhagic", entities, allow_fuzzy=True)

        assert match is not None
        assert match.canonical == "hemorrhagic"
        assert match.match_penalty == 3  # PENALTY_FUZZY

    def test_fuzzy_match_disabled_returns_none(self):
        """With allow_fuzzy=False, fuzzy matches should not be found."""
        entities = {
            "hemorrhagic": extract_entity_variants("hemorrhagic", None),
        }

        match = find_entity_match("haemorrhagic", entities, allow_fuzzy=False)

        assert match is None

    def test_fuzzy_match_too_short_rejected(self):
        """Fuzzy matching requires minimum length (MIN_LENGTH_FOR_FUZZY=10)."""
        entities = {
            "gene": extract_entity_variants("gene", None),
        }

        # Too short for fuzzy matching (< 10 chars)
        match = find_entity_match("geno", entities, allow_fuzzy=True)

        assert match is None

    def test_fuzzy_match_hyphenation_variant(self):
        """Hyphenation differences should match as obvious variants."""
        entities = {
            "Venoocular disease": extract_entity_variants("Venoocular disease", None),
        }

        # Query with hyphen should match
        match = find_entity_match("Veno-ocular disease", entities, allow_fuzzy=True)

        assert match is not None
        assert match.canonical == "Venoocular disease"
        assert match.match_penalty == 3  # PENALTY_FUZZY

    def test_fuzzy_match_short_hyphenation_variant(self):
        """Short hyphenated terms should match when they're obvious variants."""
        entities = {
            "alphaSMA": extract_entity_variants("alphaSMA", None),
        }

        # Query with hyphen (8 chars, below MIN_LENGTH_FOR_FUZZY=10)
        # Should still match because it's an obvious variant
        match = find_entity_match("alpha-SMA", entities, allow_fuzzy=True)

        assert match is not None
        assert match.canonical == "alphaSMA"
        assert match.match_penalty == 3  # PENALTY_FUZZY

    def test_fuzzy_match_multiple_hyphens(self):
        """Multiple hyphenation differences should match."""
        entities = {
            "alphabetagamma": extract_entity_variants("alphabetagamma", None),
        }

        match = find_entity_match("alpha-beta-gamma", entities, allow_fuzzy=True)

        assert match is not None
        assert match.canonical == "alphabetagamma"

    def test_hyphenation_without_fuzzy_fails(self):
        """Hyphenation differences should not match with fuzzy disabled."""
        entities = {
            "Venoocular disease": extract_entity_variants("Venoocular disease", None),
        }

        # Should not match with fuzzy disabled
        match = find_entity_match("Veno-ocular disease", entities, allow_fuzzy=False)

        assert match is None


class TestNoMatch:
    """Test cases where no match should be found."""

    def test_no_match_completely_different(self):
        """Completely different query should not match."""
        entities = {
            "BRCA1": extract_entity_variants("BRCA1", None),
        }

        match = find_entity_match("TP53", entities, allow_fuzzy=True)

        assert match is None

    def test_no_match_contested_variant_without_direct_canonical(self):
        """Contested variant should not match unless query is exact canonical.

        When "familial pah" is contested (maps to multiple canonicals) and
        query is a variant form (not exact canonical), it should fail.
        """
        entities = {
            "Familial pulmonary arterial hypertension": extract_entity_variants(
                "Familial pulmonary arterial hypertension", None
            ),
            "Heritable/familial pulmonary arterial hypertension": extract_entity_variants(
                "Heritable/familial pulmonary arterial hypertension", None
            ),
        }

        # This variant form exists in the slash-separated entity but should
        # fail because the normalized form is contested
        match = find_entity_match(
            "familial pah",
            entities,
            allow_fuzzy=False,  # Can't match via fuzzy
        )

        # Should not match because:
        # 1. Not a direct canonical match
        # 2. Variant "familial pah" is contested (excluded from variant map)
        # 3. Fuzzy disabled
        assert match is None
