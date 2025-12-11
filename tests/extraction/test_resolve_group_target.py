"""Tests for _resolve_group_target in ConsolidateEntitiesNode.

Tests cover resolution of merge targets from LLM responses in various formats:
- Pure digit references ("1", "2")
- Number + name formats ("2. Entity Name", "2) Entity Name")
- Exact member names
- Fuzzy member name matching
- New canonical names
"""

import logging
import pytest
from interaction_finder.extraction.nodes import ConsolidateEntitiesNode
from interaction_finder.extraction.entity_matching import extract_entity_variants


@pytest.fixture
def node():
    """Create a ConsolidateEntitiesNode instance for testing."""
    return ConsolidateEntitiesNode()


@pytest.fixture
def logger():
    """Create a logger for testing."""
    return logging.getLogger("test")


@pytest.fixture
def sample_members():
    """Sample member list as would be presented to LLM."""
    return [
        "Pulmonary arterial hypertension",
        "Idiopathic PAH",
        "TGF-β signaling",
        "5-HT receptor",
    ]


@pytest.fixture
def sample_entities(sample_members):
    """Build entities dict from sample members."""
    return {name: extract_entity_variants(name, None) for name in sample_members}


class TestPureDigitReference:
    """Test Case 1: Pure digit references."""

    def test_digit_resolves_to_member(
        self, node, sample_members, sample_entities, logger
    ):
        """Pure digit '1' resolves to first member."""
        result = node._resolve_group_target(
            "1", sample_members, sample_entities, logger
        )
        assert result == "Pulmonary arterial hypertension"

    def test_digit_2_resolves_to_second_member(
        self, node, sample_members, sample_entities, logger
    ):
        """Pure digit '2' resolves to second member."""
        result = node._resolve_group_target(
            "2", sample_members, sample_entities, logger
        )
        assert result == "Idiopathic PAH"

    def test_digit_out_of_range_falls_through(
        self, node, sample_members, sample_entities, logger
    ):
        """Digit beyond member count falls through to other cases."""
        # "99" is out of range, falls through - not a member name either
        # so treated as new canonical
        result = node._resolve_group_target(
            "99", sample_members, sample_entities, logger
        )
        assert result == "99"


class TestNumberPlusNameFormat:
    """Test Case 2: Number + name formats from LLM echoing prompt."""

    def test_number_dot_name_resolves_by_number(
        self, node, sample_members, sample_entities, logger
    ):
        """'2. Idiopathic PAH' resolves to member 2."""
        result = node._resolve_group_target(
            "2. Idiopathic PAH", sample_members, sample_entities, logger
        )
        assert result == "Idiopathic PAH"

    def test_number_paren_name_resolves_by_number(
        self, node, sample_members, sample_entities, logger
    ):
        """'2) Idiopathic PAH' resolves to member 2."""
        result = node._resolve_group_target(
            "2) Idiopathic PAH", sample_members, sample_entities, logger
        )
        assert result == "Idiopathic PAH"

    def test_number_space_name_resolves_by_number(
        self, node, sample_members, sample_entities, logger
    ):
        """'2 Idiopathic PAH' resolves to member 2."""
        result = node._resolve_group_target(
            "2 Idiopathic PAH", sample_members, sample_entities, logger
        )
        assert result == "Idiopathic PAH"

    def test_leading_whitespace_handled(
        self, node, sample_members, sample_entities, logger
    ):
        """'  2. Idiopathic PAH  ' handles whitespace."""
        result = node._resolve_group_target(
            "  2. Idiopathic PAH  ", sample_members, sample_entities, logger
        )
        assert result == "Idiopathic PAH"

    def test_number_name_mismatch_uses_number(
        self, node, sample_members, sample_entities, logger, caplog
    ):
        """When number and name disagree, use number and log warning."""
        # LLM says "2. Wrong Name" but member 2 is "Idiopathic PAH"
        result = node._resolve_group_target(
            "2. Wrong Name", sample_members, sample_entities, logger
        )
        assert result == "Idiopathic PAH"
        assert "mismatch" in caplog.text.lower() or len(caplog.records) > 0

    def test_number_with_variant_name_matches(
        self, node, sample_members, sample_entities, logger
    ):
        """Number + slight name variant should match (no warning)."""
        # "TGF-beta signaling" is a variant of "TGF-β signaling"
        result = node._resolve_group_target(
            "3. TGF-beta signaling", sample_members, sample_entities, logger
        )
        assert result == "TGF-β signaling"


class TestExactMemberName:
    """Test Case 3: Exact member name matches."""

    def test_exact_name_resolves(self, node, sample_members, sample_entities, logger):
        """Exact member name resolves to that member."""
        result = node._resolve_group_target(
            "Idiopathic PAH", sample_members, sample_entities, logger
        )
        assert result == "Idiopathic PAH"

    def test_exact_name_case_insensitive(
        self, node, sample_members, sample_entities, logger
    ):
        """Member name with different case still matches."""
        result = node._resolve_group_target(
            "idiopathic pah", sample_members, sample_entities, logger
        )
        assert result == "Idiopathic PAH"


class TestFuzzyMemberNameMatch:
    """Test Case 3 (fuzzy): Name that fuzzy-matches a member."""

    def test_greek_letter_variant_matches(
        self, node, sample_members, sample_entities, logger
    ):
        """'TGF-beta signaling' matches 'TGF-β signaling' via fuzzy."""
        result = node._resolve_group_target(
            "TGF-beta signaling", sample_members, sample_entities, logger
        )
        assert result == "TGF-β signaling"

    def test_hyphenation_variant_matches(
        self, node, sample_members, sample_entities, logger
    ):
        """Hyphenation differences should match."""
        # Create entities with hyphenation variants
        members = ["Veno-ocular disease", "Other entity"]
        entities = {name: extract_entity_variants(name, None) for name in members}
        result = node._resolve_group_target(
            "Venoocular disease", members, entities, logger
        )
        assert result == "Veno-ocular disease"


class TestDigitPrefixedEntityNames:
    """Test edge case: entity names starting with digits (e.g., '5-HT')."""

    def test_digit_prefixed_name_not_confused_with_number_ref(
        self, node, sample_members, sample_entities, logger
    ):
        """'5-HT receptor' should not be parsed as member 5."""
        result = node._resolve_group_target(
            "5-HT receptor", sample_members, sample_entities, logger
        )
        assert result == "5-HT receptor"

    def test_number_ref_to_digit_prefixed_member(
        self, node, sample_members, sample_entities, logger
    ):
        """'4' should resolve to '5-HT receptor' (4th member)."""
        result = node._resolve_group_target(
            "4", sample_members, sample_entities, logger
        )
        assert result == "5-HT receptor"

    def test_number_plus_digit_prefixed_name(
        self, node, sample_members, sample_entities, logger
    ):
        """'4. 5-HT receptor' should resolve to member 4."""
        result = node._resolve_group_target(
            "4. 5-HT receptor", sample_members, sample_entities, logger
        )
        assert result == "5-HT receptor"


class TestNewCanonicalName:
    """Test Case 4: New canonical names that don't match any member."""

    def test_new_name_returned_as_is(
        self, node, sample_members, sample_entities, logger
    ):
        """Completely new name is returned as-is (new canonical)."""
        result = node._resolve_group_target(
            "Completely New Entity", sample_members, sample_entities, logger
        )
        assert result == "Completely New Entity"

    def test_rename_to_standard_form(
        self, node, sample_members, sample_entities, logger
    ):
        """LLM might suggest renaming to a standard form."""
        result = node._resolve_group_target(
            "Pulmonary Arterial Hypertension", sample_members, sample_entities, logger
        )
        # This matches existing member via fuzzy/normalization
        assert result == "Pulmonary arterial hypertension"
