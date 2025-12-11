"""Test group target resolution for entity consolidation.

The _resolve_group_target function handles multiple formats that LLMs may return:
- Pure digit: "1", "2" (member number)
- Number + name: "2. Entity Name", "2) Entity Name"
- Exact name: "Entity Name"
- Fuzzy name: variant spelling
- New name: creates a new canonical
"""

import logging

import pytest

from interaction_finder.extraction.stages.consolidate_entities import (
    _resolve_group_target,
)
from interaction_finder.extraction.entity_matching import SpeculatedVariant


@pytest.fixture
def sample_members():
    return ["BRCA1", "BRCA2", "TP53", "EGFR"]


@pytest.fixture
def sample_entities():
    """Entities dict with variants for fuzzy matching."""
    return {
        "BRCA1": [
            SpeculatedVariant(
                form="BRCA1", speculation=0, source="original", is_from_alias=False
            )
        ],
        "BRCA2": [
            SpeculatedVariant(
                form="BRCA2", speculation=0, source="original", is_from_alias=False
            )
        ],
        "TP53": [
            SpeculatedVariant(
                form="TP53", speculation=0, source="original", is_from_alias=False
            ),
            SpeculatedVariant(
                form="p53", speculation=1, source="alias", is_from_alias=True
            ),
        ],
        "EGFR": [
            SpeculatedVariant(
                form="EGFR", speculation=0, source="original", is_from_alias=False
            )
        ],
    }


@pytest.fixture
def logger():
    return logging.getLogger("test")


class TestResolveGroupTarget:
    """Test target resolution from group decisions."""

    def test_pure_digit(self, sample_members, sample_entities, logger):
        """Pure digit like '1' resolves to member at that index."""
        result = _resolve_group_target("1", sample_members, sample_entities, logger)
        assert result == "BRCA1"
        result = _resolve_group_target("3", sample_members, sample_entities, logger)
        assert result == "TP53"

    def test_digit_out_of_range(self, sample_members, sample_entities, logger):
        """Out of range digit returns as new canonical name."""
        result = _resolve_group_target("99", sample_members, sample_entities, logger)
        assert result == "99"

    def test_number_dot_name_format(self, sample_members, sample_entities, logger):
        """'2. BRCA2' format resolves using number."""
        result = _resolve_group_target(
            "2. BRCA2", sample_members, sample_entities, logger
        )
        assert result == "BRCA2"

    def test_number_paren_name_format(self, sample_members, sample_entities, logger):
        """'2) BRCA2' format resolves using number."""
        result = _resolve_group_target(
            "2) BRCA2", sample_members, sample_entities, logger
        )
        assert result == "BRCA2"

    def test_number_space_name_format(self, sample_members, sample_entities, logger):
        """'2 BRCA2' format resolves using number."""
        result = _resolve_group_target(
            "2 BRCA2", sample_members, sample_entities, logger
        )
        assert result == "BRCA2"

    def test_exact_member_name(self, sample_members, sample_entities, logger):
        """Exact member name matches directly."""
        result = _resolve_group_target("TP53", sample_members, sample_entities, logger)
        assert result == "TP53"

    def test_fuzzy_match_alias(self, sample_members, sample_entities, logger):
        """Fuzzy match via alias (p53 -> TP53)."""
        result = _resolve_group_target("p53", sample_members, sample_entities, logger)
        assert result == "TP53"

    def test_new_canonical_name(self, sample_members, sample_entities, logger):
        """Unrecognized name becomes new canonical."""
        result = _resolve_group_target(
            "Novel Gene", sample_members, sample_entities, logger
        )
        assert result == "Novel Gene"

    def test_whitespace_handling(self, sample_members, sample_entities, logger):
        """Leading/trailing whitespace handled - pure digit still resolves."""
        result = _resolve_group_target("  1  ", sample_members, sample_entities, logger)
        # isdigit() on stripped string matches, so resolves to first member
        assert result == "BRCA1"

    def test_number_name_mismatch_uses_number(
        self, sample_members, sample_entities, logger, caplog
    ):
        """When number and name conflict, uses number and logs warning."""
        # "1. BRCA2" - number says BRCA1, name says BRCA2
        result = _resolve_group_target(
            "1. BRCA2", sample_members, sample_entities, logger
        )
        assert result == "BRCA1"  # Uses member number

    def test_empty_members_list(self, sample_entities, logger):
        """Empty members list returns target as new canonical."""
        result = _resolve_group_target("1", [], sample_entities, logger)
        assert result == "1"

    def test_case_sensitive_exact_match(self, sample_members, sample_entities, logger):
        """Exact match is case-sensitive."""
        result = _resolve_group_target("brca1", sample_members, sample_entities, logger)
        # Should fuzzy match to BRCA1
        assert result == "BRCA1"

    def test_number_name_resolves_to_same_canonical(
        self, sample_members, sample_entities, logger, caplog
    ):
        """Number + name where LLM name resolves to same canonical should not warn."""
        # LLM returns "1. brca1" - find_entity_match resolves to "BRCA1" which matches member
        with caplog.at_level(logging.WARNING):
            result = _resolve_group_target(
                "1. brca1", sample_members, sample_entities, logger
            )
        assert result == "BRCA1"  # Returns canonical member
        # No warning since brca1 resolves to BRCA1 which matches member exactly
        assert "mismatch" not in caplog.text.lower()

    def test_number_name_imprecise_match(self, sample_members, sample_entities, logger):
        """Number + name with alias (p53 for TP53) should resolve correctly."""
        # LLM returns "3. p53" - member 3 is TP53, p53 is an alias
        result = _resolve_group_target(
            "3. p53", sample_members, sample_entities, logger
        )
        assert result == "TP53"

    def test_number_name_imprecise_mismatch_warns(
        self, sample_members, sample_entities, logger, caplog
    ):
        """Number + name where alias resolves to wrong member should warn."""
        # LLM returns "1. p53" - member 1 is BRCA1, but p53 resolves to TP53
        with caplog.at_level(logging.WARNING):
            result = _resolve_group_target(
                "1. p53", sample_members, sample_entities, logger
            )
        assert result == "BRCA1"  # Still uses member number
        assert "mismatch" in caplog.text.lower()

    def test_case4_fuzzy_match_case_sensitive(self, logger):
        """Case 4: Fuzzy match membership check is case-sensitive."""
        # members has "Cataract", entities has "cataract"
        members = ["Cataract", "Glaucoma"]
        entities = {
            "cataract": [
                SpeculatedVariant(
                    form="cataract",
                    speculation=0,
                    source="original",
                    is_from_alias=False,
                )
            ],
            "Glaucoma": [
                SpeculatedVariant(
                    form="Glaucoma",
                    speculation=0,
                    source="original",
                    is_from_alias=False,
                )
            ],
        }
        # "cataract" fuzzy matches to canonical "cataract" but members has "Cataract"
        # Since these should come from same source, case mismatch indicates a bug
        # Case 4 is case-sensitive, so it won't match and falls through to Case 5
        result = _resolve_group_target("cataract", members, entities, logger)
        # Returns as new canonical since "cataract" not in members (case-sensitive)
        assert result == "cataract"

    def test_case4_fuzzy_match_consistent_case(self, logger):
        """Case 4: Fuzzy match works when members and entities are consistent."""
        members = ["cataract", "Glaucoma"]
        entities = {
            "cataract": [
                SpeculatedVariant(
                    form="cataract",
                    speculation=0,
                    source="original",
                    is_from_alias=False,
                )
            ],
            "Glaucoma": [
                SpeculatedVariant(
                    form="Glaucoma",
                    speculation=0,
                    source="original",
                    is_from_alias=False,
                )
            ],
        }
        # "Cataract" (different case) fuzzy matches to canonical "cataract"
        # which is in members, so it should return the canonical
        result = _resolve_group_target("Cataract", members, entities, logger)
        assert result == "cataract"
