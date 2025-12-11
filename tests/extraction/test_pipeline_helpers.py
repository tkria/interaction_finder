"""Tests for shared.py helper functions."""

import pytest

from interaction_finder.extraction.shared import (
    aggregate_cache_stats,
    aggregate_evidence,
    get_entity_aliases,
    snapshot_entity_counts,
)
from interaction_finder.extraction.run import STAGE_NAMES, get_resume_stage_index
from interaction_finder.extraction.models import (
    EntityMention,
    EntityRef,
    EvidenceQuality,
    PairAssessment,
)
from interaction_finder.extraction.state import MergeCacheForKind, State
from interaction_finder.resources import ResourceId


def build_permitted_pairs(kinds: list[str]) -> dict[str, set[str]]:
    """Build permitted pairs dict for testing."""
    return {kind: {kind} for kind in kinds}


def make_assessment(
    *,
    directness: str = "explicit",
    source_type: str = "primary",
    specificity: str = "mechanistic",
    language: str = "definitive",
    overall: int = 5,
) -> PairAssessment:
    """Create a PairAssessment with configurable evidence."""
    resource_id = ResourceId(url="https://test.com", counter=0)
    return PairAssessment(
        topic_relevance=3,
        resource_id=resource_id,
        entity1=EntityRef(
            canonical="BRCA1",
            mentions=[
                EntityMention(
                    kind="gene", name="BRCA1", aliases=[], quotes=[], reasoning="test"
                )
            ],
        ),
        entity2=EntityRef(
            canonical="Cancer",
            mentions=[
                EntityMention(
                    kind="disease",
                    name="Cancer",
                    aliases=[],
                    quotes=[],
                    reasoning="test",
                )
            ],
        ),
        relationship="associated_with",
        evidence=EvidenceQuality(
            directness=directness,
            source_type=source_type,
            specificity=specificity,
            language=language,
            overall=overall,
        ),
        quotes=[],
        reasoning="Test assessment",
    )


class TestGetResumeStageIndex:
    """Tests for get_resume_stage_index function."""

    def test_valid_first_stage_returns_one(self):
        """First stage returns index 1 (next stage)."""
        assert get_resume_stage_index("process_documents") == 1

    def test_valid_middle_stage_returns_next_index(self):
        """Middle stage returns the next index."""
        assert get_resume_stage_index("consolidate_entities") == 2
        assert get_resume_stage_index("consolidate_relationships") == 3

    def test_valid_last_stage_returns_len(self):
        """Last stage returns length of STAGE_NAMES list."""
        last_stage = STAGE_NAMES[-1]
        assert get_resume_stage_index(last_stage) == len(STAGE_NAMES)

    def test_invalid_stage_returns_zero(self):
        """Unknown stage name returns 0 (start from beginning)."""
        assert get_resume_stage_index("nonexistent_stage") == 0
        assert get_resume_stage_index("") == 0

    def test_case_sensitive(self):
        """Stage names are case-sensitive."""
        assert get_resume_stage_index("Process_Documents") == 0
        assert get_resume_stage_index("PROCESS_DOCUMENTS") == 0


class TestGetEntityAliases:
    """Tests for get_entity_aliases function."""

    def test_found_entity_returns_aliases(self):
        """Entity in global index returns its aliases."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        state.global_entities = {
            "BRCA1": EntityRef(
                canonical="BRCA1",
                mentions=[
                    EntityMention(
                        kind="gene",
                        name="brca1",
                        aliases=["breast cancer gene 1"],
                        quotes=[],
                        reasoning="test",
                    )
                ],
            )
        }
        aliases = get_entity_aliases("BRCA1", state)
        assert "brca1" in aliases
        assert "breast cancer gene 1" in aliases

    def test_missing_entity_returns_empty(self):
        """Entity not in global index returns empty list."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        state.global_entities = {}
        assert get_entity_aliases("NonExistent", state) == []

    def test_entity_with_no_aliases(self):
        """Entity with mention name == canonical returns empty aliases list."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        state.global_entities = {
            "BRCA1": EntityRef(
                canonical="BRCA1",
                mentions=[
                    EntityMention(
                        kind="gene",
                        name="BRCA1",  # Same as canonical
                        aliases=[],
                        quotes=[],
                        reasoning="test",
                    )
                ],
            )
        }
        aliases = get_entity_aliases("BRCA1", state)
        # aliases() only returns names that differ from canonical
        assert aliases == []


class TestAggregateEvidence:
    """Tests for aggregate_evidence function."""

    def test_empty_returns_defaults(self):
        """Empty assessment list returns conservative defaults."""
        result = aggregate_evidence([])
        assert result.directness == "tangential"
        assert result.source_type == "other"
        assert result.specificity == "vague"
        assert result.language == "speculative"
        assert result.overall == 1

    def test_single_assessment_returns_its_values(self):
        """Single assessment returns its own values."""
        assessment = make_assessment(
            directness="explicit",
            source_type="primary",
            specificity="mechanistic",
            language="definitive",
            overall=5,
        )
        result = aggregate_evidence([assessment])
        assert result.directness == "explicit"
        assert result.source_type == "primary"
        assert result.specificity == "mechanistic"
        assert result.language == "definitive"
        assert result.overall == 5

    def test_median_with_odd_count(self):
        """Odd count uses middle value for overall."""
        assessments = [
            make_assessment(overall=1),
            make_assessment(overall=3),
            make_assessment(overall=5),
        ]
        result = aggregate_evidence(assessments)
        assert result.overall == 3

    def test_median_with_even_count_conservative(self):
        """Even count uses lower of two middle values (median_low)."""
        assessments = [
            make_assessment(overall=2),
            make_assessment(overall=3),
            make_assessment(overall=4),
            make_assessment(overall=5),
        ]
        result = aggregate_evidence(assessments)
        # median_low([2,3,4,5]) = 3 (lower of 3,4)
        assert result.overall == 3

    def test_mode_selection_for_factors(self):
        """Factors use most common value (mode)."""
        assessments = [
            make_assessment(directness="explicit", source_type="primary"),
            make_assessment(directness="explicit", source_type="review"),
            make_assessment(directness="implied", source_type="review"),
        ]
        result = aggregate_evidence(assessments)
        assert result.directness == "explicit"  # 2 vs 1
        assert result.source_type == "review"  # 2 vs 1

    def test_ties_use_first_encountered(self):
        """When modes are tied, Counter.most_common returns one deterministically."""
        assessments = [
            make_assessment(directness="explicit"),
            make_assessment(directness="implied"),
        ]
        result = aggregate_evidence(assessments)
        # Both have count 1 - Counter returns one of them
        assert result.directness in ("explicit", "implied")

    def test_mixed_evidence_levels(self):
        """Test with diverse evidence levels."""
        assessments = [
            make_assessment(overall=1),
            make_assessment(overall=1),
            make_assessment(overall=5),
            make_assessment(overall=5),
            make_assessment(overall=3),
        ]
        result = aggregate_evidence(assessments)
        # Sorted: [1,1,3,5,5] - median is 3
        assert result.overall == 3


class TestAggregateCacheStats:
    """Tests for aggregate_cache_stats function."""

    def test_empty_cache_returns_zeros(self):
        """Empty cache dict returns (0, 0)."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        state.merge_cache_by_kind = {}
        hits, misses = aggregate_cache_stats(state)
        assert hits == 0
        assert misses == 0

    def test_single_kind_sums_correctly(self):
        """Single kind returns its stats."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        cache = MergeCacheForKind()
        cache.hits = 5
        cache.misses = 3
        state.merge_cache_by_kind = {"gene": cache}
        hits, misses = aggregate_cache_stats(state)
        assert hits == 5
        assert misses == 3

    def test_multiple_kinds_summed(self):
        """Multiple kinds have their stats summed."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        gene_cache = MergeCacheForKind()
        gene_cache.hits = 10
        gene_cache.misses = 2
        disease_cache = MergeCacheForKind()
        disease_cache.hits = 5
        disease_cache.misses = 8
        state.merge_cache_by_kind = {"gene": gene_cache, "disease": disease_cache}
        hits, misses = aggregate_cache_stats(state)
        assert hits == 15
        assert misses == 10


class TestSnapshotEntityCounts:
    """Tests for snapshot_entity_counts function."""

    def test_empty_state_returns_empty(self):
        """Empty validated entities returns empty dict."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        state.validated_entities_by_resource = {}
        result = snapshot_entity_counts(state)
        assert result == {}

    def test_single_resource_single_kind(self):
        """Single resource with one kind counted correctly."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        resource = ResourceId(url="https://doc1.com", counter=0)
        state.validated_entities_by_resource = {
            resource: {
                "BRCA1": EntityRef(
                    canonical="BRCA1",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="BRCA1",
                            aliases=[],
                            quotes=[],
                            reasoning="test",
                        )
                    ],
                )
            }
        }
        result = snapshot_entity_counts(state)
        assert result == {"gene": {"BRCA1": 1}}

    def test_aggregates_across_resources(self):
        """Same entity in multiple resources has counts summed."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        resource1 = ResourceId(url="https://doc1.com", counter=0)
        resource2 = ResourceId(url="https://doc2.com", counter=1)
        state.validated_entities_by_resource = {
            resource1: {
                "BRCA1": EntityRef(
                    canonical="BRCA1",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="BRCA1",
                            aliases=[],
                            quotes=[],
                            reasoning="doc1",
                        )
                    ],
                )
            },
            resource2: {
                "BRCA1": EntityRef(
                    canonical="BRCA1",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="BRCA1",
                            aliases=[],
                            quotes=[],
                            reasoning="doc2-1",
                        ),
                        EntityMention(
                            kind="gene",
                            name="brca1",
                            aliases=[],
                            quotes=[],
                            reasoning="doc2-2",
                        ),
                    ],
                )
            },
        }
        result = snapshot_entity_counts(state)
        # doc1 has 1 mention, doc2 has 2 mentions
        assert result == {"gene": {"BRCA1": 3}}

    def test_groups_by_kind(self):
        """Different kinds are grouped separately."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        resource = ResourceId(url="https://doc1.com", counter=0)
        state.validated_entities_by_resource = {
            resource: {
                "BRCA1": EntityRef(
                    canonical="BRCA1",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="BRCA1",
                            aliases=[],
                            quotes=[],
                            reasoning="gene",
                        )
                    ],
                ),
                "Cancer": EntityRef(
                    canonical="Cancer",
                    mentions=[
                        EntityMention(
                            kind="disease",
                            name="Cancer",
                            aliases=[],
                            quotes=[],
                            reasoning="disease",
                        )
                    ],
                ),
            }
        }
        result = snapshot_entity_counts(state)
        assert "gene" in result
        assert "disease" in result
        assert result["gene"] == {"BRCA1": 1}
        assert result["disease"] == {"Cancer": 1}

    def test_multiple_entities_per_kind(self):
        """Multiple entities of same kind counted separately."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        resource = ResourceId(url="https://doc1.com", counter=0)
        state.validated_entities_by_resource = {
            resource: {
                "BRCA1": EntityRef(
                    canonical="BRCA1",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="BRCA1",
                            aliases=[],
                            quotes=[],
                            reasoning="test",
                        )
                    ],
                ),
                "TP53": EntityRef(
                    canonical="TP53",
                    mentions=[
                        EntityMention(
                            kind="gene",
                            name="TP53",
                            aliases=[],
                            quotes=[],
                            reasoning="test",
                        ),
                        EntityMention(
                            kind="gene",
                            name="p53",
                            aliases=[],
                            quotes=[],
                            reasoning="alias",
                        ),
                    ],
                ),
            }
        }
        result = snapshot_entity_counts(state)
        assert result["gene"]["BRCA1"] == 1
        assert result["gene"]["TP53"] == 2
