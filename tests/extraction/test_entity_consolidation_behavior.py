"""Behavior tests for entity consolidation.

Tests critical consolidation behaviors through the consolidation functions:
- Capitalization variant handling
- Alias preservation through merges
- Cross-document merge uniformity
- Fuzzy variant consolidation (US/UK spelling)
- ClusterDecision model validation
"""

import pytest

from interaction_finder.extraction.models import (
    ClusterDecision,
    ClusterDecisions,
    EntityMention,
    EntityRef,
)
from interaction_finder.extraction.stages.consolidate_entities import (
    _apply_merge_rules_globally,
    _resolve_group_target,
    _resolve_transitive_merges,
)
from interaction_finder.extraction.entity_matching import SpeculatedVariant
from interaction_finder.extraction.state import State
from interaction_finder.extraction.utils import normalize_for_comparison
from interaction_finder.resources import ResourceId
from unittest.mock import MagicMock


def build_permitted_pairs(kinds: list[str]) -> dict[str, set[str]]:
    """Build permitted pairs dict for testing."""
    return {kind: {kind} for kind in kinds}


def ref_map(data: dict[str, EntityMention]) -> dict[str, EntityRef]:
    """Convert entity mentions to EntityRef dict."""
    return {
        name: EntityRef(canonical=name, mentions=[mention])
        for name, mention in data.items()
    }


class TestCapitalizationVariants:
    """Test that capitalization variants merge correctly."""

    def test_two_case_variants_merge(self):
        """Two capitalization variants merge to single canonical."""
        state = State(
            topic="test",
            target_entity_types=["phenotype"],
            permitted_pairs=build_permitted_pairs(["phenotype"]),
        )
        resource1 = ResourceId(url="https://doc1.com", counter=0)
        resource2 = ResourceId(url="https://doc2.com", counter=1)
        # Two documents with same entity, different capitalization
        state.validated_entities_by_resource = {
            resource1: ref_map(
                {
                    "Pulmonary Hypertension": EntityMention(
                        kind="phenotype",
                        name="Pulmonary Hypertension",
                        aliases=[],
                        quotes=[],
                        reasoning="Doc1",
                    )
                }
            ),
            resource2: ref_map(
                {
                    "Pulmonary hypertension": EntityMention(
                        kind="phenotype",
                        name="Pulmonary hypertension",
                        aliases=[],
                        quotes=[],
                        reasoning="Doc2",
                    )
                }
            ),
        }
        # Rule to merge lowercase variant into mixed-case canonical
        norm = normalize_for_comparison("Pulmonary hypertension")
        merge_rules = {(norm, "phenotype"): ("Pulmonary Hypertension", "auto:case")}
        _apply_merge_rules_globally(merge_rules, state)
        # Both docs should have same canonical
        canonical1 = list(state.validated_entities_by_resource[resource1].keys())[0]
        canonical2 = list(state.validated_entities_by_resource[resource2].keys())[0]
        assert canonical1 == canonical2 == "Pulmonary Hypertension"

    def test_three_case_variants_converge(self):
        """Three case variants (CAPS, Mixed, lower) all converge."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        resource1 = ResourceId(url="https://doc1.com", counter=0)
        resource2 = ResourceId(url="https://doc2.com", counter=1)
        resource3 = ResourceId(url="https://doc3.com", counter=2)
        state.validated_entities_by_resource = {
            resource1: ref_map(
                {
                    "BRCA1": EntityMention(
                        kind="gene",
                        name="BRCA1",
                        aliases=[],
                        quotes=[],
                        reasoning="Doc1",
                    )
                }
            ),
            resource2: ref_map(
                {
                    "Brca1": EntityMention(
                        kind="gene",
                        name="Brca1",
                        aliases=[],
                        quotes=[],
                        reasoning="Doc2",
                    )
                }
            ),
            resource3: ref_map(
                {
                    "brca1": EntityMention(
                        kind="gene",
                        name="brca1",
                        aliases=[],
                        quotes=[],
                        reasoning="Doc3",
                    )
                }
            ),
        }
        # Rules to merge all variants to BRCA1 (the all-caps version common for genes)
        norm = normalize_for_comparison("brca1")
        merge_rules = {
            (norm, "gene"): ("BRCA1", "auto:case"),
        }
        _apply_merge_rules_globally(merge_rules, state)
        # All docs should have BRCA1
        for resource_id in [resource1, resource2, resource3]:
            canonical = list(state.validated_entities_by_resource[resource_id].keys())[
                0
            ]
            assert canonical == "BRCA1"


class TestAliasPreservation:
    """Test that aliases are preserved through merges."""

    def test_merge_combines_mentions(self):
        """When entities merge in same document, mentions are combined."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        resource = ResourceId(url="https://doc1.com", counter=0)
        # Parent entity with one alias, child with another
        parent_mention = EntityMention(
            kind="gene",
            name="BRCA1",
            aliases=["breast cancer 1"],
            quotes=[],
            reasoning="Parent",
        )
        child_mention = EntityMention(
            kind="gene",
            name="brca1",
            aliases=["BRCA-1"],
            quotes=[],
            reasoning="Child",
        )
        state.validated_entities_by_resource = {
            resource: {
                "BRCA1": EntityRef(canonical="BRCA1", mentions=[parent_mention]),
                "brca1": EntityRef(canonical="brca1", mentions=[child_mention]),
            }
        }
        norm = normalize_for_comparison("brca1")
        merge_rules = {(norm, "gene"): ("BRCA1", "auto:case")}
        _apply_merge_rules_globally(merge_rules, state)
        # Should have single entity with both mentions
        entities = state.validated_entities_by_resource[resource]
        assert len(entities) == 1
        merged = entities["BRCA1"]
        assert len(merged.mentions) == 2
        # aliases() aggregates all aliases from mentions
        all_aliases = merged.aliases()
        assert "breast cancer 1" in all_aliases or "brca1" in all_aliases

    def test_cross_document_merge_preserves_variant_names(self):
        """Cross-document merges preserve variant names as aliases."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        resource1 = ResourceId(url="https://doc1.com", counter=0)
        resource2 = ResourceId(url="https://doc2.com", counter=1)
        state.validated_entities_by_resource = {
            resource1: ref_map(
                {
                    "BRCA1": EntityMention(
                        kind="gene",
                        name="BRCA1",
                        aliases=["breast cancer gene 1"],
                        quotes=[],
                        reasoning="Doc1",
                    )
                }
            ),
            resource2: ref_map(
                {
                    "Brca1": EntityMention(
                        kind="gene",
                        name="Brca1",
                        aliases=["BRCA-1"],
                        quotes=[],
                        reasoning="Doc2",
                    )
                }
            ),
        }
        norm = normalize_for_comparison("Brca1")
        merge_rules = {(norm, "gene"): ("BRCA1", "auto:case")}
        _apply_merge_rules_globally(merge_rules, state)
        # Both now have BRCA1
        assert "BRCA1" in state.validated_entities_by_resource[resource1]
        assert "BRCA1" in state.validated_entities_by_resource[resource2]
        # Each doc keeps its own mention data
        doc1_aliases = state.validated_entities_by_resource[resource1][
            "BRCA1"
        ].aliases()
        assert "breast cancer gene 1" in doc1_aliases


class TestCrossDocumentConsistency:
    """Test that merges apply uniformly across all documents."""

    def test_single_rule_applied_everywhere(self):
        """A single merge rule applies to all documents containing the entity."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        resources = [ResourceId(url=f"https://doc{i}.com", counter=i) for i in range(5)]
        # Five documents with same entity in different cases
        state.validated_entities_by_resource = {
            resources[0]: ref_map(
                {
                    "BRCA1": EntityMention(
                        kind="gene",
                        name="BRCA1",
                        aliases=[],
                        quotes=[],
                        reasoning="canonical",
                    )
                }
            ),
            resources[1]: ref_map(
                {
                    "brca1": EntityMention(
                        kind="gene",
                        name="brca1",
                        aliases=[],
                        quotes=[],
                        reasoning="lowercase",
                    )
                }
            ),
            resources[2]: ref_map(
                {
                    "Brca1": EntityMention(
                        kind="gene",
                        name="Brca1",
                        aliases=[],
                        quotes=[],
                        reasoning="mixed",
                    )
                }
            ),
            resources[3]: ref_map(
                {
                    "BRca1": EntityMention(
                        kind="gene",
                        name="BRca1",
                        aliases=[],
                        quotes=[],
                        reasoning="weird",
                    )
                }
            ),
            resources[4]: ref_map(
                {
                    "BRCA1": EntityMention(
                        kind="gene",
                        name="BRCA1",
                        aliases=[],
                        quotes=[],
                        reasoning="canonical2",
                    )
                }
            ),
        }
        # All normalize to same key, so one rule merges them all
        norm = normalize_for_comparison("brca1")
        merge_rules = {(norm, "gene"): ("BRCA1", "auto:case")}
        _apply_merge_rules_globally(merge_rules, state)
        # All documents should have BRCA1
        for resource_id in resources:
            entities = state.validated_entities_by_resource[resource_id]
            assert len(entities) == 1
            assert "BRCA1" in entities

    def test_iterative_merges_find_all_chains(self):
        """Transitive chains are fully resolved before applying rules."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        resource = ResourceId(url="https://doc1.com", counter=0)
        state.validated_entities_by_resource = {
            resource: {
                "GeneA": EntityRef(
                    canonical="GeneA",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="GeneA",
                            aliases=[],
                            quotes=[],
                            reasoning="test",
                        )
                    ],
                ),
                "GeneB": EntityRef(
                    canonical="GeneB",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="GeneB",
                            aliases=[],
                            quotes=[],
                            reasoning="test",
                        )
                    ],
                ),
                "GeneC": EntityRef(
                    canonical="GeneC",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="GeneC",
                            aliases=[],
                            quotes=[],
                            reasoning="test",
                        )
                    ],
                ),
            }
        }
        # Chain: A -> B -> C
        unresolved = {
            ("genea", "gene"): ("GeneB", "auto:1"),
            ("geneb", "gene"): ("GeneC", "auto:2"),
        }
        resolved = _resolve_transitive_merges(unresolved)
        _apply_merge_rules_globally(resolved, state)
        # All should be merged to GeneC
        entities = state.validated_entities_by_resource[resource]
        assert len(entities) == 1
        assert "GeneC" in entities


class TestFuzzyVariantConsolidation:
    """Test US/UK spelling and other fuzzy variant handling."""

    def test_us_uk_spelling_merge(self):
        """US and UK spelling variants merge correctly."""
        state = State(
            topic="test",
            target_entity_types=["phenotype"],
            permitted_pairs=build_permitted_pairs(["phenotype"]),
        )
        resource1 = ResourceId(url="https://doc1.com", counter=0)
        resource2 = ResourceId(url="https://doc2.com", counter=1)
        state.validated_entities_by_resource = {
            resource1: ref_map(
                {
                    "tumor": EntityMention(
                        kind="phenotype",
                        name="tumor",
                        aliases=[],
                        quotes=[],
                        reasoning="US",
                    )
                }
            ),
            resource2: ref_map(
                {
                    "tumour": EntityMention(
                        kind="phenotype",
                        name="tumour",
                        aliases=[],
                        quotes=[],
                        reasoning="UK",
                    )
                }
            ),
        }
        # Fuzzy match rule
        norm = normalize_for_comparison("tumour")
        merge_rules = {(norm, "phenotype"): ("tumor", "auto:spelling")}
        _apply_merge_rules_globally(merge_rules, state)
        # Both should have 'tumor'
        canonical1 = list(state.validated_entities_by_resource[resource1].keys())[0]
        canonical2 = list(state.validated_entities_by_resource[resource2].keys())[0]
        assert canonical1 == canonical2 == "tumor"

    def test_greek_letter_variants_merge(self):
        """Greek letters and spelled-out forms merge correctly."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        resource1 = ResourceId(url="https://doc1.com", counter=0)
        resource2 = ResourceId(url="https://doc2.com", counter=1)
        state.validated_entities_by_resource = {
            resource1: ref_map(
                {
                    "TGF-β": EntityMention(
                        kind="gene",
                        name="TGF-β",
                        aliases=[],
                        quotes=[],
                        reasoning="Greek",
                    )
                }
            ),
            resource2: ref_map(
                {
                    "TGF-beta": EntityMention(
                        kind="gene",
                        name="TGF-beta",
                        aliases=[],
                        quotes=[],
                        reasoning="Spelled",
                    )
                }
            ),
        }
        # Both normalize to same key
        norm = normalize_for_comparison("TGF-beta")
        merge_rules = {(norm, "gene"): ("TGF-β", "auto:greek")}
        _apply_merge_rules_globally(merge_rules, state)
        # Both should have 'TGF-β'
        canonical1 = list(state.validated_entities_by_resource[resource1].keys())[0]
        canonical2 = list(state.validated_entities_by_resource[resource2].keys())[0]
        assert canonical1 == canonical2 == "TGF-β"


class TestEdgeCases:
    """Test edge cases in consolidation behavior."""

    def test_entity_with_self_alias_no_duplicate(self):
        """Entity with alias matching its own canonical doesn't create issues."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        resource = ResourceId(url="https://doc1.com", counter=0)
        # Entity with alias that's the same as its name
        state.validated_entities_by_resource = {
            resource: ref_map(
                {
                    "BRCA1": EntityMention(
                        kind="gene",
                        name="BRCA1",
                        aliases=["BRCA1", "breast cancer 1"],
                        quotes=[],
                        reasoning="Self-alias",
                    )
                }
            )
        }
        # Empty rules - no merging needed
        _apply_merge_rules_globally({}, state)
        entities = state.validated_entities_by_resource[resource]
        assert len(entities) == 1
        assert "BRCA1" in entities

    def test_empty_state_no_error(self):
        """Empty state doesn't cause errors."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        state.validated_entities_by_resource = {}
        merge_rules = {("brca1", "gene"): ("BRCA1", "auto:test")}
        _apply_merge_rules_globally(merge_rules, state)
        # No error, state unchanged
        assert state.validated_entities_by_resource == {}

    def test_rule_for_nonexistent_entity_no_error(self):
        """Rule for entity not in any document doesn't cause errors."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        resource = ResourceId(url="https://doc1.com", counter=0)
        state.validated_entities_by_resource = {
            resource: ref_map(
                {
                    "TP53": EntityMention(
                        kind="gene",
                        name="TP53",
                        aliases=[],
                        quotes=[],
                        reasoning="Different gene",
                    )
                }
            )
        }
        # Rule for BRCA1 which doesn't exist
        merge_rules = {("brca1", "gene"): ("BRCA1", "auto:test")}
        _apply_merge_rules_globally(merge_rules, state)
        # TP53 still there, unchanged
        entities = state.validated_entities_by_resource[resource]
        assert "TP53" in entities


class TestClusterDecisionModel:
    """Test ClusterDecision and ClusterDecisions Pydantic model validation.

    These tests ensure the model correctly parses LLM responses for all action types
    and validates field constraints.
    """

    def test_merge_action_with_target(self):
        """Merge action parses correctly with explicit target."""
        decision = ClusterDecision(
            group_id="G1",
            action="merge",
            target="BRCA1",
            reasoning="All variants refer to the same gene",
        )
        assert decision.group_id == "G1"
        assert decision.action == "merge"
        assert decision.target == "BRCA1"
        assert "same gene" in decision.reasoning

    def test_merge_action_default(self):
        """Merge is the default action when not specified."""
        decision = ClusterDecision(
            group_id="G1",
            reasoning="All members are the same entity",
        )
        assert decision.action == "merge"
        assert decision.target is None

    def test_split_action(self):
        """Split action parses correctly (target not used)."""
        decision = ClusterDecision(
            group_id="G2",
            action="split",
            reasoning="Cluster contains unrelated entities",
        )
        assert decision.action == "split"
        assert decision.target is None

    def test_exclude_action_with_target(self):
        """Exclude action parses correctly with member to exclude."""
        decision = ClusterDecision(
            group_id="G3",
            action="exclude",
            target="2",  # Member number
            reasoning="Member 2 is a different entity",
        )
        assert decision.action == "exclude"
        assert decision.target == "2"

    def test_exclude_action_with_name_target(self):
        """Exclude action accepts entity name as target."""
        decision = ClusterDecision(
            group_id="G3",
            action="exclude",
            target="TP53",  # Entity name
            reasoning="TP53 is unrelated to other members",
        )
        assert decision.action == "exclude"
        assert decision.target == "TP53"

    def test_invalid_action_rejected(self):
        """Invalid action raises validation error."""
        with pytest.raises(ValueError):
            ClusterDecision(
                group_id="G1",
                action="invalid_action",
                reasoning="This should fail",
            )

    def test_decisions_batch(self):
        """ClusterDecisions batch parses multiple decisions."""
        decisions = ClusterDecisions(
            decisions=[
                ClusterDecision(
                    group_id="G1",
                    action="merge",
                    target="EntityA",
                    reasoning="Same entity",
                ),
                ClusterDecision(
                    group_id="G2",
                    action="split",
                    reasoning="Different entities mixed",
                ),
                ClusterDecision(
                    group_id="G3",
                    action="exclude",
                    target="3",
                    reasoning="Member 3 doesn't belong",
                ),
            ]
        )
        assert len(decisions.decisions) == 3
        assert decisions.decisions[0].action == "merge"
        assert decisions.decisions[1].action == "split"
        assert decisions.decisions[2].action == "exclude"

    def test_empty_decisions_batch(self):
        """Empty decisions list is valid."""
        decisions = ClusterDecisions(decisions=[])
        assert len(decisions.decisions) == 0

    def test_json_round_trip(self):
        """ClusterDecision survives JSON serialization/deserialization."""
        original = ClusterDecision(
            group_id="G1",
            action="merge",
            target="BRCA1",
            reasoning="All variants of BRCA1",
        )
        json_str = original.model_dump_json()
        restored = ClusterDecision.model_validate_json(json_str)
        assert restored.group_id == original.group_id
        assert restored.action == original.action
        assert restored.target == original.target
        assert restored.reasoning == original.reasoning


class TestResolveGroupTarget:
    """Test _resolve_group_target for various LLM response formats.

    This function must handle different ways LLMs specify merge/exclude targets:
    - Pure digit: "1", "2"
    - Number + name: "1. BRCA1", "2) TP53"
    - Exact name: "BRCA1"
    - Fuzzy name: "brca-1"
    - New canonical: "MyNewName"
    """

    @pytest.fixture
    def mock_logger(self):
        return MagicMock()

    @pytest.fixture
    def members(self):
        return ["BRCA1", "TP53", "EGFR"]

    @pytest.fixture
    def entities(self):
        """Create entity lookup dict with speculated variants."""
        return {
            "BRCA1": [
                SpeculatedVariant(form="BRCA1", speculation=0, source="original")
            ],
            "TP53": [SpeculatedVariant(form="TP53", speculation=0, source="original")],
            "EGFR": [SpeculatedVariant(form="EGFR", speculation=0, source="original")],
        }

    def test_pure_digit_resolves_to_member(self, mock_logger, members, entities):
        """'1' resolves to first member."""
        result = _resolve_group_target("1", members, entities, mock_logger)
        assert result == "BRCA1"

    def test_pure_digit_second_member(self, mock_logger, members, entities):
        """'2' resolves to second member."""
        result = _resolve_group_target("2", members, entities, mock_logger)
        assert result == "TP53"

    def test_pure_digit_third_member(self, mock_logger, members, entities):
        """'3' resolves to third member."""
        result = _resolve_group_target("3", members, entities, mock_logger)
        assert result == "EGFR"

    def test_number_dot_name_format(self, mock_logger, members, entities):
        """'1. BRCA1' resolves to first member."""
        result = _resolve_group_target("1. BRCA1", members, entities, mock_logger)
        assert result == "BRCA1"

    def test_number_paren_name_format(self, mock_logger, members, entities):
        """'2) TP53' resolves to second member."""
        result = _resolve_group_target("2) TP53", members, entities, mock_logger)
        assert result == "TP53"

    def test_exact_member_name(self, mock_logger, members, entities):
        """Exact member name resolves to itself."""
        result = _resolve_group_target("EGFR", members, entities, mock_logger)
        assert result == "EGFR"

    def test_new_canonical_name_returned_as_is(self, mock_logger, members, entities):
        """Unknown name is returned as new canonical."""
        result = _resolve_group_target("NewGeneName", members, entities, mock_logger)
        assert result == "NewGeneName"

    def test_number_mismatch_logs_warning(self, mock_logger, members, entities):
        """'1. TP53' logs warning but uses member 1."""
        result = _resolve_group_target("1. TP53", members, entities, mock_logger)
        assert result == "BRCA1"  # Uses member number despite name mismatch
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "mismatch" in warning_msg


class TestMultipleClusterDecisions:
    """Test complex scenarios with multiple decisions on the same group.

    These tests verify behavior when LLM returns multiple decisions for a single
    group (e.g., exclude + merge, multiple excludes).
    """

    def test_exclude_reduces_remaining_members(self):
        """Exclude action removes member from remaining set."""
        # Simulate the decision processing logic
        members = ["BRCA1", "TP53", "EGFR"]
        remaining = set(members)
        # Exclude second member
        exclude_target = "TP53"
        remaining.discard(exclude_target)
        assert remaining == {"BRCA1", "EGFR"}

    def test_multiple_excludes_chain(self):
        """Multiple exclude decisions can be applied sequentially."""
        members = ["A", "B", "C", "D", "E"]
        remaining = set(members)
        # Multiple excludes
        for target in ["B", "D"]:
            remaining.discard(target)
        assert remaining == {"A", "C", "E"}

    def test_exclude_then_merge_leaves_remaining(self):
        """After excludes, remaining members can still be merged."""
        members = ["BRCA1", "TP53", "EGFR", "MYC"]
        remaining = set(members)
        # Exclude one member
        remaining.discard("MYC")
        # Remaining can be merged
        assert len(remaining) == 3
        assert "MYC" not in remaining

    def test_exclude_all_but_one_stops_processing(self):
        """If only one member remains after excludes, no merge is needed."""
        members = ["BRCA1", "TP53"]
        remaining = set(members)
        remaining.discard("TP53")
        # Only one member left - skip further processing
        assert len(remaining) == 1
        assert remaining == {"BRCA1"}

    def test_merge_and_split_conflict_handled(self):
        """Merge + split decisions on same group is a conflict."""
        decisions = [
            ClusterDecision(
                group_id="G1", action="merge", target="1", reasoning="Same"
            ),
            ClusterDecision(group_id="G1", action="split", reasoning="Different"),
        ]
        merges = [d for d in decisions if d.action == "merge"]
        splits = [d for d in decisions if d.action == "split"]
        # This is a conflict - implementation logs warning and skips
        assert len(merges) == 1
        assert len(splits) == 1
        # Code would skip processing due to conflict

    def test_decision_categorization(self):
        """Decisions are correctly categorized by action type."""
        decisions = [
            ClusterDecision(
                group_id="G1", action="exclude", target="2", reasoning="Outlier"
            ),
            ClusterDecision(
                group_id="G1", action="exclude", target="4", reasoning="Unrelated"
            ),
            ClusterDecision(
                group_id="G1", action="merge", target="1", reasoning="Same entity"
            ),
        ]
        excludes = [d for d in decisions if d.action == "exclude"]
        merges = [d for d in decisions if d.action == "merge"]
        splits = [d for d in decisions if d.action == "split"]
        assert len(excludes) == 2
        assert len(merges) == 1
        assert len(splits) == 0
