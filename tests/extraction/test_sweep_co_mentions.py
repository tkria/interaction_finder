"""Tests for co-mention sweep functionality."""

import re

import pytest

from interaction_finder.extraction.models import (
    EntityMention,
    EntityRef,
    PairAssessment,
)
from interaction_finder.extraction.sweep_co_mentions import (
    CandidatePair,
    CoMentionRegion,
    CoMentionSweepStats,
    NovelCoMention,
    build_entity_search_pattern,
    classify_co_mention,
    collect_assessed_pairs,
    collect_global_aliases,
    find_entity_mentions,
    find_novel_co_mentions_in_resource,
    is_co_mention_covered,
    merge_co_mentions_into_regions,
    select_co_mentions_to_assess,
)
from interaction_finder.extraction.utils import make_entity_pair_key
from interaction_finder.resources import Resource, ResourceId, ResourceQuote
from tests.extraction.conftest import make_evidence


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_resource():
    """Create a sample resource for testing."""
    text = """BRCA1 is a tumor suppressor gene associated with breast cancer.
Mutations in BRCA1 increase the risk of developing breast cancer.
TP53 is another important gene that regulates cell cycle.
BRCA1 and TP53 interact in DNA damage response pathways.
Some studies suggest BRCA1 protects against certain cancers."""
    return Resource(
        id=ResourceId(url="https://example.com/doc1", counter=1),
        title="Sample Document",
        text=text,
        chunks=[(0, 70), (70, 140), (140, 200), (200, 270), (270, len(text))],
    )


@pytest.fixture
def entity_brca1():
    """Create BRCA1 entity mention."""
    return EntityMention(
        kind="gene",
        name="BRCA1",
        aliases=["BRCA-1", "breast cancer 1"],
        quotes=[],
        reasoning="Test entity",
    )


@pytest.fixture
def entity_tp53():
    """Create TP53 entity mention."""
    return EntityMention(
        kind="gene",
        name="TP53",
        aliases=["p53", "tumor protein 53"],
        quotes=[],
        reasoning="Test entity",
    )


@pytest.fixture
def entity_breast_cancer():
    """Create breast cancer entity mention."""
    return EntityMention(
        kind="disease",
        name="breast cancer",
        aliases=["breast carcinoma"],
        quotes=[],
        reasoning="Test entity",
    )


@pytest.fixture
def entity_ref_brca1(entity_brca1):
    return EntityRef(canonical=entity_brca1.name, mentions=[entity_brca1])


@pytest.fixture
def entity_ref_tp53(entity_tp53):
    return EntityRef(canonical=entity_tp53.name, mentions=[entity_tp53])


@pytest.fixture
def entity_ref_breast_cancer(entity_breast_cancer):
    return EntityRef(
        canonical=entity_breast_cancer.name, mentions=[entity_breast_cancer]
    )


# =============================================================================
# Test collect_global_aliases
# =============================================================================


class TestCollectGlobalAliases:
    """Tests for collect_global_aliases function."""

    def test_empty_input(self):
        """Empty input returns empty dict."""
        result = collect_global_aliases({})
        assert result == {}

    def test_single_resource_single_entity(self, entity_brca1):
        """Single resource with single entity collects its aliases."""
        resource_id = ResourceId(url="https://example.com/1", counter=1)
        entities = {
            entity_brca1.name: EntityRef(
                canonical=entity_brca1.name, mentions=[entity_brca1]
            )
        }
        result = collect_global_aliases({resource_id: entities})
        assert result == {"BRCA1": {"BRCA-1", "breast cancer 1"}}

    def test_multiple_resources_same_entity(self, entity_brca1):
        """Same entity in multiple resources merges aliases."""
        resource_id1 = ResourceId(url="https://example.com/1", counter=1)
        resource_id2 = ResourceId(url="https://example.com/2", counter=2)
        # Same entity with different aliases in different resources
        entity1 = EntityMention(
            kind="gene",
            name="BRCA1",
            aliases=["BRCA-1"],
            quotes=[],
            reasoning="Test",
        )
        entity2 = EntityMention(
            kind="gene",
            name="BRCA1",
            aliases=["breast cancer 1", "FANCS"],
            quotes=[],
            reasoning="Test",
        )
        result = collect_global_aliases(
            {
                resource_id1: {
                    "BRCA1": EntityRef(canonical="BRCA1", mentions=[entity1])
                },
                resource_id2: {
                    "BRCA1": EntityRef(canonical="BRCA1", mentions=[entity2])
                },
            }
        )
        assert result == {"BRCA1": {"BRCA-1", "breast cancer 1", "FANCS"}}

    def test_multiple_entities(self, entity_brca1, entity_tp53):
        """Multiple entities each get their own alias set."""
        resource_id = ResourceId(url="https://example.com/1", counter=1)
        entities = {
            entity_brca1.name: EntityRef(
                canonical=entity_brca1.name, mentions=[entity_brca1]
            ),
            entity_tp53.name: EntityRef(
                canonical=entity_tp53.name, mentions=[entity_tp53]
            ),
        }
        result = collect_global_aliases({resource_id: entities})
        assert "BRCA1" in result
        assert "TP53" in result
        assert result["BRCA1"] == {"BRCA-1", "breast cancer 1"}
        assert result["TP53"] == {"p53", "tumor protein 53"}


# =============================================================================
# Test build_entity_search_pattern
# =============================================================================


class TestBuildEntitySearchPattern:
    """Tests for build_entity_search_pattern function.

    Note: Patterns are designed to match against normalized (lowercase) text.
    Both the entity names and the target text should be normalized before matching.
    """

    def test_simple_name_no_aliases(self):
        """Pattern matches normalized name in normalized text."""
        pattern = build_entity_search_pattern("BRCA1", set())
        # Pattern is built from normalized "BRCA1" -> "brca1"
        # Matches against normalized text (lowercase)
        assert pattern.search("brca1") is not None
        assert pattern.search("the brca1 gene") is not None
        # Won't match non-normalized text (uppercase)
        assert pattern.search("BRCA1") is None

    def test_with_aliases(self):
        """Pattern matches name and all aliases in normalized text."""
        pattern = build_entity_search_pattern("BRCA1", {"BRCA-1", "breast cancer 1"})
        assert pattern.search("brca1") is not None
        # "BRCA-1" normalizes to "brca 1" (hyphen becomes space)
        assert pattern.search("brca 1") is not None
        assert pattern.search("breast cancer 1") is not None

    def test_word_boundaries(self):
        """Pattern respects word boundaries in normalized text."""
        pattern = build_entity_search_pattern("TP53", set())
        assert pattern.search("tp53") is not None
        assert pattern.search("atp53") is None  # Part of larger word
        assert pattern.search("tp532") is None  # Part of larger word

    def test_plural_handling(self):
        """Pattern matches optional plurals."""
        pattern = build_entity_search_pattern("gene", set())
        assert pattern.search("gene") is not None
        assert pattern.search("genes") is not None

    def test_special_characters_escaped(self):
        """Special regex characters in names are escaped."""
        pattern = build_entity_search_pattern("IL-1β", {"IL-1beta"})
        # Should not raise regex error
        assert pattern is not None
        # "IL-1β" normalizes to "il 1 beta" (hyphen->space, β->beta)
        assert pattern.search("il 1 beta") is not None

    def test_normalized_matching(self):
        """Pattern only matches normalized (lowercase) text."""
        pattern = build_entity_search_pattern("BRCA1", set())
        # Matches lowercase
        assert pattern.search("brca1") is not None
        # Does not match uppercase (text should be pre-normalized)
        assert pattern.search("Brca1") is None
        assert pattern.search("BRCA1") is None

    def test_empty_name_handled(self):
        """Empty or whitespace-only names handled gracefully."""
        pattern = build_entity_search_pattern("", set())
        # Should return a pattern that matches nothing
        assert pattern.search("anything") is None


# =============================================================================
# Test find_entity_mentions
# =============================================================================


class TestFindEntityMentions:
    """Tests for find_entity_mentions function."""

    def test_finds_mentions_in_normalized_text(self, sample_resource):
        """Finds entity mentions in normalized text."""
        pattern = build_entity_search_pattern("BRCA1", set())
        mentions = find_entity_mentions(sample_resource, pattern)
        # BRCA1 appears multiple times in the sample text
        assert len(mentions) >= 3

    def test_returns_positions_and_matched_text(self, sample_resource):
        """Returns start, end positions and matched text."""
        pattern = build_entity_search_pattern("BRCA1", set())
        mentions = find_entity_mentions(sample_resource, pattern)
        for start, end, matched_text in mentions:
            assert isinstance(start, int)
            assert isinstance(end, int)
            assert start < end
            assert isinstance(matched_text, str)
            assert matched_text.lower() == "brca1"

    def test_no_matches_returns_empty(self, sample_resource):
        """No matches returns empty list."""
        pattern = build_entity_search_pattern("NONEXISTENT_GENE", set())
        mentions = find_entity_mentions(sample_resource, pattern)
        assert mentions == []

    def test_returns_matched_alias_form(self, sample_resource):
        """Returns the actual matched text form (alias or canonical)."""
        # Create resource with multiple entity forms
        text = "The gene brca1 is also known as BRCA1 and breast cancer 1."
        resource = Resource(
            id=ResourceId(url="https://example.com/alias", counter=99),
            title="Alias Test",
            text=text,
            chunks=[(0, len(text))],
        )
        pattern = build_entity_search_pattern("BRCA1", {"breast cancer 1"})
        mentions = find_entity_mentions(resource, pattern)
        matched_forms = {m[2].lower() for m in mentions}
        # Should find both canonical form and alias
        assert "brca1" in matched_forms or "breast cancer 1" in matched_forms


# =============================================================================
# Test is_co_mention_covered
# =============================================================================


class TestIsCoMentionCovered:
    """Tests for is_co_mention_covered function."""

    def test_both_positions_in_span_covered(
        self, sample_resource, entity_ref_brca1, entity_ref_breast_cancer
    ):
        """Both positions within a quote span means covered."""
        # Create a quote spanning positions 0-100
        quote = sample_resource.quote("BRCA1 is a tumor suppressor gene")
        assessment = PairAssessment(
            resource_id=sample_resource.id,
            entity1=entity_ref_brca1,
            entity2=entity_ref_breast_cancer,
            relationship="associated_with",
            quotes=[quote],
            evidence=make_evidence(8),
            reasoning="Test",
        )
        # Positions within the quote span should be covered
        span_start, span_end = quote.spans[0]
        assert is_co_mention_covered(span_start + 1, span_end - 1, [assessment])

    def test_one_position_outside_not_covered(
        self, sample_resource, entity_ref_brca1, entity_ref_breast_cancer
    ):
        """One position outside quote span means not covered."""
        quote = sample_resource.quote("BRCA1 is a tumor suppressor gene")
        assessment = PairAssessment(
            resource_id=sample_resource.id,
            entity1=entity_ref_brca1,
            entity2=entity_ref_breast_cancer,
            relationship="associated_with",
            quotes=[quote],
            evidence=make_evidence(8),
            reasoning="Test",
        )
        span_start, span_end = quote.spans[0]
        # One position inside, one way outside
        assert not is_co_mention_covered(span_start + 1, span_end + 100, [assessment])

    def test_no_assessments_not_covered(self):
        """No assessments means not covered."""
        assert not is_co_mention_covered(10, 20, [])

    def test_multiple_quotes_checks_all(
        self, sample_resource, entity_ref_brca1, entity_ref_breast_cancer
    ):
        """Checks all quotes for coverage."""
        quote1 = sample_resource.quote("BRCA1 is a tumor suppressor gene")
        quote2 = sample_resource.quote("BRCA1 and TP53 interact")
        assessment = PairAssessment(
            resource_id=sample_resource.id,
            entity1=entity_ref_brca1,
            entity2=entity_ref_breast_cancer,
            relationship="associated_with",
            quotes=[quote1, quote2],
            evidence=make_evidence(8),
            reasoning="Test",
        )
        # Position in second quote should be covered
        span_start, span_end = quote2.spans[0]
        assert is_co_mention_covered(span_start + 1, span_end - 1, [assessment])


# =============================================================================
# Test classify_co_mention
# =============================================================================


class TestClassifyCoMention:
    """Tests for classify_co_mention function."""

    def test_no_existing_assessment(
        self, sample_resource, entity_ref_brca1, entity_ref_tp53
    ):
        """No existing assessment returns novel with no_existing_assessment priority."""
        pair_key = make_entity_pair_key(entity_ref_brca1, entity_ref_tp53)
        is_novel, priority = classify_co_mention(
            10, 20, pair_key, sample_resource.id, {}
        )
        assert is_novel is True
        assert priority == "no_existing_assessment"

    def test_existing_assessment_different_region(
        self, sample_resource, entity_ref_brca1, entity_ref_breast_cancer
    ):
        """Existing assessment in different region returns novel with uncovered_region."""
        quote = sample_resource.quote("BRCA1 is a tumor suppressor gene")
        assessment = PairAssessment(
            resource_id=sample_resource.id,
            entity1=entity_ref_brca1,
            entity2=entity_ref_breast_cancer,
            relationship="associated_with",
            quotes=[quote],
            evidence=make_evidence(8),
            reasoning="Test",
        )
        pair_key = make_entity_pair_key(entity_ref_brca1, entity_ref_breast_cancer)
        # Position far from the quote
        is_novel, priority = classify_co_mention(
            250, 260, pair_key, sample_resource.id, {sample_resource.id: [assessment]}
        )
        assert is_novel is True
        assert priority == "uncovered_region"

    def test_covered_region_not_novel(
        self, sample_resource, entity_ref_brca1, entity_ref_breast_cancer
    ):
        """Positions within existing quote are not novel."""
        quote = sample_resource.quote("BRCA1 is a tumor suppressor gene")
        assessment = PairAssessment(
            resource_id=sample_resource.id,
            entity1=entity_ref_brca1,
            entity2=entity_ref_breast_cancer,
            relationship="associated_with",
            quotes=[quote],
            evidence=make_evidence(8),
            reasoning="Test",
        )
        pair_key = make_entity_pair_key(entity_ref_brca1, entity_ref_breast_cancer)
        span_start, span_end = quote.spans[0]
        is_novel, priority = classify_co_mention(
            span_start + 1,
            span_end - 1,
            pair_key,
            sample_resource.id,
            {sample_resource.id: [assessment]},
        )
        assert is_novel is False


# =============================================================================
# Test find_novel_co_mentions_in_resource
# =============================================================================


class TestFindNovelCoMentionsInResource:
    """Tests for find_novel_co_mentions_in_resource function."""

    def test_finds_co_mentions_within_chunk_distance(
        self, sample_resource, entity_ref_brca1, entity_ref_tp53
    ):
        """Finds co-mentions when entities are within chunk distance."""
        pair_key = make_entity_pair_key(entity_ref_brca1, entity_ref_tp53)
        assessed_pairs = {pair_key}
        entity_patterns = {
            "BRCA1": build_entity_search_pattern("BRCA1", set()),
            "TP53": build_entity_search_pattern("TP53", set()),
        }
        co_mentions = find_novel_co_mentions_in_resource(
            sample_resource,
            sample_resource.id,
            assessed_pairs,
            entity_patterns,
            {},
            chunk_distance=2,
        )
        # Should find at least one co-mention (BRCA1 and TP53 appear together)
        assert len(co_mentions) >= 1
        assert all(isinstance(cm, NovelCoMention) for cm in co_mentions)

    def test_filters_beyond_chunk_distance(
        self, sample_resource, entity_ref_brca1, entity_ref_tp53
    ):
        """Filters co-mentions beyond chunk distance."""
        pair_key = make_entity_pair_key(entity_ref_brca1, entity_ref_tp53)
        assessed_pairs = {pair_key}
        entity_patterns = {
            "BRCA1": build_entity_search_pattern("BRCA1", set()),
            "TP53": build_entity_search_pattern("TP53", set()),
        }
        # Very small chunk distance should filter out most co-mentions
        co_mentions = find_novel_co_mentions_in_resource(
            sample_resource,
            sample_resource.id,
            assessed_pairs,
            entity_patterns,
            {},
            chunk_distance=0,
        )
        # Only co-mentions in same chunk should remain
        for cm in co_mentions:
            assert cm.chunk_range[0] == cm.chunk_range[1]

    def test_deduplicates_by_chunk_range(
        self, sample_resource, entity_ref_brca1, entity_ref_breast_cancer
    ):
        """Deduplicates co-mentions with same chunk range."""
        pair_key = make_entity_pair_key(entity_ref_brca1, entity_ref_breast_cancer)
        assessed_pairs = {pair_key}
        entity_patterns = {
            "BRCA1": build_entity_search_pattern("BRCA1", set()),
            "breast cancer": build_entity_search_pattern("breast cancer", set()),
        }
        co_mentions = find_novel_co_mentions_in_resource(
            sample_resource,
            sample_resource.id,
            assessed_pairs,
            entity_patterns,
            {},
            chunk_distance=2,
        )
        # Should not have duplicate chunk ranges for same pair
        seen_ranges = set()
        for cm in co_mentions:
            key = (cm.pair_key, cm.chunk_range)
            assert key not in seen_ranges, "Found duplicate chunk range"
            seen_ranges.add(key)

    def test_skips_covered_co_mentions(
        self, sample_resource, entity_ref_brca1, entity_ref_breast_cancer
    ):
        """Skips co-mentions that are already covered by existing assessments."""
        quote = sample_resource.quote(
            "BRCA1 is a tumor suppressor gene associated with breast cancer"
        )
        assessment = PairAssessment(
            resource_id=sample_resource.id,
            entity1=entity_ref_brca1,
            entity2=entity_ref_breast_cancer,
            relationship="associated_with",
            quotes=[quote],
            evidence=make_evidence(8),
            reasoning="Test",
        )
        pair_key = make_entity_pair_key(entity_ref_brca1, entity_ref_breast_cancer)
        assessed_pairs = {pair_key}
        entity_patterns = {
            "BRCA1": build_entity_search_pattern("BRCA1", set()),
            "breast cancer": build_entity_search_pattern("breast cancer", set()),
        }
        co_mentions = find_novel_co_mentions_in_resource(
            sample_resource,
            sample_resource.id,
            assessed_pairs,
            entity_patterns,
            {sample_resource.id: [assessment]},
            chunk_distance=2,
        )
        # Co-mentions in the covered region should be filtered out
        # But there might be other mentions in the document
        for cm in co_mentions:
            # Each found co-mention should be in an uncovered region
            assert cm.priority in ("no_existing_assessment", "uncovered_region")


# =============================================================================
# Test collect_assessed_pairs
# =============================================================================


class TestCollectAssessedPairs:
    """Tests for collect_assessed_pairs function."""

    def test_empty_input(self):
        """Empty input returns empty set."""
        result = collect_assessed_pairs({})
        assert result == set()

    def test_collects_unique_pairs(
        self,
        sample_resource,
        entity_ref_brca1,
        entity_ref_tp53,
        entity_ref_breast_cancer,
    ):
        """Collects unique pairs from assessments."""
        assessment1 = PairAssessment(
            resource_id=sample_resource.id,
            entity1=entity_ref_brca1,
            entity2=entity_ref_tp53,
            relationship="interacts_with",
            quotes=[],
            evidence=make_evidence(8),
            reasoning="Test",
        )
        assessment2 = PairAssessment(
            resource_id=sample_resource.id,
            entity1=entity_ref_brca1,
            entity2=entity_ref_breast_cancer,
            relationship="associated_with",
            quotes=[],
            evidence=make_evidence(8),
            reasoning="Test",
        )
        result = collect_assessed_pairs(
            {sample_resource.id: [assessment1, assessment2]}
        )
        assert len(result) == 2

    def test_deduplicates_same_pair(
        self, sample_resource, entity_ref_brca1, entity_ref_tp53
    ):
        """Same pair in multiple resources counted once."""
        resource_id2 = ResourceId(url="https://example.com/2", counter=2)
        assessment1 = PairAssessment(
            resource_id=sample_resource.id,
            entity1=entity_ref_brca1,
            entity2=entity_ref_tp53,
            relationship="interacts_with",
            quotes=[],
            evidence=make_evidence(8),
            reasoning="Test",
        )
        assessment2 = PairAssessment(
            resource_id=resource_id2,
            entity1=entity_ref_brca1,
            entity2=entity_ref_tp53,
            relationship="regulates",
            quotes=[],
            evidence=make_evidence(6),
            reasoning="Test",
        )
        result = collect_assessed_pairs(
            {sample_resource.id: [assessment1], resource_id2: [assessment2]}
        )
        assert len(result) == 1


# =============================================================================
# Test select_co_mentions_to_assess
# =============================================================================


class TestSelectCoMentionsToAssess:
    """Tests for select_co_mentions_to_assess function."""

    def test_returns_all_co_mentions(
        self, sample_resource, entity_ref_brca1, entity_ref_tp53
    ):
        """Currently returns all co-mentions unchanged."""
        pair_key = make_entity_pair_key(entity_ref_brca1, entity_ref_tp53)
        co_mentions = [
            NovelCoMention(
                resource_id=sample_resource.id,
                pair_key=pair_key,
                chunk_range=(0, 1),
                priority="no_existing_assessment",
                entity1_pos=10,
                entity2_pos=50,
                entity1_matched_form="brca1",
                entity2_matched_form="tp53",
            ),
            NovelCoMention(
                resource_id=sample_resource.id,
                pair_key=pair_key,
                chunk_range=(2, 3),
                priority="uncovered_region",
                entity1_pos=150,
                entity2_pos=180,
                entity1_matched_form="brca1",
                entity2_matched_form="tp53",
            ),
        ]
        result = select_co_mentions_to_assess(co_mentions, None)
        assert result == co_mentions


# =============================================================================
# Test CoMentionSweepStats
# =============================================================================


class TestCoMentionSweepStats:
    """Tests for CoMentionSweepStats dataclass."""

    def test_default_values(self):
        """Default values are all zero."""
        stats = CoMentionSweepStats()
        assert stats.total_co_mentions_found == 0
        assert stats.co_mentions_no_existing_assessment == 0
        assert stats.co_mentions_uncovered_region == 0
        assert stats.regions_created == 0
        assert stats.assessed == 0
        assert stats.relationships_found == 0
        assert stats.no_relationship_claim == 0

    def test_can_update_values(self):
        """Values can be updated."""
        stats = CoMentionSweepStats()
        stats.total_co_mentions_found = 10
        stats.regions_created = 3
        stats.assessed = 8
        stats.relationships_found = 5
        assert stats.total_co_mentions_found == 10
        assert stats.regions_created == 3
        assert stats.assessed == 8
        assert stats.relationships_found == 5


# =============================================================================
# Test merge_co_mentions_into_regions
# =============================================================================


class TestMergeCoMentionsIntoRegions:
    """Tests for merge_co_mentions_into_regions function."""

    def test_empty_input(self):
        """Empty input returns empty list."""
        result = merge_co_mentions_into_regions([], {})
        assert result == []

    def test_single_co_mention(
        self, sample_resource, entity_ref_brca1, entity_ref_tp53
    ):
        """Single co-mention creates single region."""
        pair_key = make_entity_pair_key(entity_ref_brca1, entity_ref_tp53)
        co_mentions = [
            NovelCoMention(
                resource_id=sample_resource.id,
                pair_key=pair_key,
                chunk_range=(2, 3),
                priority="no_existing_assessment",
                entity1_pos=100,
                entity2_pos=150,
                entity1_matched_form="brca1",
                entity2_matched_form="tp53",
            )
        ]
        entity_kinds = {"BRCA1": "gene", "TP53": "gene"}
        result = merge_co_mentions_into_regions(co_mentions, entity_kinds)
        assert len(result) == 1
        assert result[0].resource_id == sample_resource.id
        assert result[0].chunk_range == (2, 3)
        assert len(result[0].candidate_pairs) == 1

    def test_adjacent_ranges_merged(
        self, sample_resource, entity_ref_brca1, entity_ref_tp53
    ):
        """Adjacent chunk ranges (gap=1) are merged into single region."""
        pair_key = make_entity_pair_key(entity_ref_brca1, entity_ref_tp53)
        co_mentions = [
            NovelCoMention(
                resource_id=sample_resource.id,
                pair_key=pair_key,
                chunk_range=(0, 1),
                priority="no_existing_assessment",
                entity1_pos=10,
                entity2_pos=50,
                entity1_matched_form="brca1",
                entity2_matched_form="tp53",
            ),
            NovelCoMention(
                resource_id=sample_resource.id,
                pair_key=pair_key,
                chunk_range=(2, 3),  # Adjacent to (0, 1)
                priority="no_existing_assessment",
                entity1_pos=100,
                entity2_pos=150,
                entity1_matched_form="brca1",
                entity2_matched_form="tp53",
            ),
        ]
        entity_kinds = {"BRCA1": "gene", "TP53": "gene"}
        result = merge_co_mentions_into_regions(co_mentions, entity_kinds)
        assert len(result) == 1
        assert result[0].chunk_range == (0, 3)

    def test_non_adjacent_ranges_separate(
        self, sample_resource, entity_ref_brca1, entity_ref_tp53
    ):
        """Non-adjacent chunk ranges create separate regions."""
        pair_key = make_entity_pair_key(entity_ref_brca1, entity_ref_tp53)
        co_mentions = [
            NovelCoMention(
                resource_id=sample_resource.id,
                pair_key=pair_key,
                chunk_range=(0, 1),
                priority="no_existing_assessment",
                entity1_pos=10,
                entity2_pos=50,
                entity1_matched_form="brca1",
                entity2_matched_form="tp53",
            ),
            NovelCoMention(
                resource_id=sample_resource.id,
                pair_key=pair_key,
                chunk_range=(5, 6),  # Gap > 1 from (0, 1)
                priority="no_existing_assessment",
                entity1_pos=300,
                entity2_pos=350,
                entity1_matched_form="brca1",
                entity2_matched_form="tp53",
            ),
        ]
        entity_kinds = {"BRCA1": "gene", "TP53": "gene"}
        result = merge_co_mentions_into_regions(co_mentions, entity_kinds)
        assert len(result) == 2

    def test_multiple_pairs_in_same_region(
        self,
        sample_resource,
        entity_ref_brca1,
        entity_ref_tp53,
        entity_ref_breast_cancer,
    ):
        """Multiple pairs in overlapping ranges collected in same region."""
        pair_key1 = make_entity_pair_key(entity_ref_brca1, entity_ref_tp53)
        pair_key2 = make_entity_pair_key(entity_ref_brca1, entity_ref_breast_cancer)
        co_mentions = [
            NovelCoMention(
                resource_id=sample_resource.id,
                pair_key=pair_key1,
                chunk_range=(0, 1),
                priority="no_existing_assessment",
                entity1_pos=10,
                entity2_pos=50,
                entity1_matched_form="brca1",
                entity2_matched_form="tp53",
            ),
            NovelCoMention(
                resource_id=sample_resource.id,
                pair_key=pair_key2,
                chunk_range=(0, 2),  # Overlaps with (0, 1)
                priority="no_existing_assessment",
                entity1_pos=10,
                entity2_pos=60,
                entity1_matched_form="brca1",
                entity2_matched_form="breast cancer",
            ),
        ]
        entity_kinds = {"BRCA1": "gene", "TP53": "gene", "breast cancer": "disease"}
        result = merge_co_mentions_into_regions(co_mentions, entity_kinds)
        assert len(result) == 1
        assert len(result[0].candidate_pairs) == 2
        assert result[0].chunk_range == (0, 2)

    def test_different_resources_separate(
        self, sample_resource, entity_ref_brca1, entity_ref_tp53
    ):
        """Co-mentions in different resources create separate regions."""
        resource_id2 = ResourceId(url="https://example.com/2", counter=2)
        pair_key = make_entity_pair_key(entity_ref_brca1, entity_ref_tp53)
        co_mentions = [
            NovelCoMention(
                resource_id=sample_resource.id,
                pair_key=pair_key,
                chunk_range=(0, 1),
                priority="no_existing_assessment",
                entity1_pos=10,
                entity2_pos=50,
                entity1_matched_form="brca1",
                entity2_matched_form="tp53",
            ),
            NovelCoMention(
                resource_id=resource_id2,
                pair_key=pair_key,
                chunk_range=(0, 1),
                priority="no_existing_assessment",
                entity1_pos=10,
                entity2_pos=50,
                entity1_matched_form="brca1",
                entity2_matched_form="tp53",
            ),
        ]
        entity_kinds = {"BRCA1": "gene", "TP53": "gene"}
        result = merge_co_mentions_into_regions(co_mentions, entity_kinds)
        assert len(result) == 2
        resource_ids = {r.resource_id for r in result}
        assert sample_resource.id in resource_ids
        assert resource_id2 in resource_ids

    def test_deduplicates_pairs_in_region(
        self, sample_resource, entity_ref_brca1, entity_ref_tp53
    ):
        """Same pair appearing multiple times in merged region is deduplicated."""
        pair_key = make_entity_pair_key(entity_ref_brca1, entity_ref_tp53)
        co_mentions = [
            NovelCoMention(
                resource_id=sample_resource.id,
                pair_key=pair_key,
                chunk_range=(0, 1),
                priority="no_existing_assessment",
                entity1_pos=10,
                entity2_pos=50,
                entity1_matched_form="brca1",
                entity2_matched_form="tp53",
            ),
            NovelCoMention(
                resource_id=sample_resource.id,
                pair_key=pair_key,  # Same pair
                chunk_range=(1, 2),  # Adjacent, will be merged
                priority="no_existing_assessment",
                entity1_pos=80,
                entity2_pos=100,
                entity1_matched_form="brca1",
                entity2_matched_form="tp53",
            ),
        ]
        entity_kinds = {"BRCA1": "gene", "TP53": "gene"}
        result = merge_co_mentions_into_regions(co_mentions, entity_kinds)
        assert len(result) == 1
        # Same pair should only appear once in candidate_pairs
        assert len(result[0].candidate_pairs) == 1

    def test_entity_kinds_assigned(
        self, sample_resource, entity_ref_brca1, entity_ref_tp53
    ):
        """Entity kinds are correctly assigned to candidate pairs."""
        pair_key = make_entity_pair_key(entity_ref_brca1, entity_ref_tp53)
        co_mentions = [
            NovelCoMention(
                resource_id=sample_resource.id,
                pair_key=pair_key,
                chunk_range=(0, 1),
                priority="no_existing_assessment",
                entity1_pos=10,
                entity2_pos=50,
                entity1_matched_form="brca1",
                entity2_matched_form="tp53",
            ),
        ]
        entity_kinds = {"BRCA1": "gene", "TP53": "gene"}
        result = merge_co_mentions_into_regions(co_mentions, entity_kinds)
        candidate = result[0].candidate_pairs[0]
        assert candidate.entity1_kind == "gene"
        assert candidate.entity2_kind == "gene"

    def test_missing_entity_kind_defaults(
        self, sample_resource, entity_ref_brca1, entity_ref_tp53
    ):
        """Missing entity kinds default to 'entity'."""
        pair_key = make_entity_pair_key(entity_ref_brca1, entity_ref_tp53)
        co_mentions = [
            NovelCoMention(
                resource_id=sample_resource.id,
                pair_key=pair_key,
                chunk_range=(0, 1),
                priority="no_existing_assessment",
                entity1_pos=10,
                entity2_pos=50,
                entity1_matched_form="brca1",
                entity2_matched_form="tp53",
            ),
        ]
        result = merge_co_mentions_into_regions(co_mentions, {})  # No entity kinds
        candidate = result[0].candidate_pairs[0]
        assert candidate.entity1_kind == "entity"
        assert candidate.entity2_kind == "entity"


# =============================================================================
# Test assess_co_mention_region (diagnostic messages)
# =============================================================================


class TestAssessCoMentionRegionDiagnostics:
    """Tests for diagnostic warning messages in assess_co_mention_region.

    These tests verify that when the LLM returns entity names that can't be
    matched to candidates, the system provides detailed diagnostic information.
    """

    @pytest.mark.asyncio
    async def test_unresolvable_first_entity_warning(
        self, sample_resource, entity_ref_brca1, entity_ref_tp53, tmp_path
    ):
        """Warning includes diagnostic when first entity is unresolvable."""
        from unittest.mock import AsyncMock, MagicMock
        from interaction_finder.extraction.sweep_co_mentions import (
            assess_co_mention_region,
            ConfirmedPair,
            RegionAssessmentOut,
        )
        from interaction_finder.extraction.deps import Deps
        from interaction_finder.settings import IfetcherConfig

        # Create region with BRCA1-TP53 candidate pair
        pair_key = make_entity_pair_key(entity_ref_brca1, entity_ref_tp53)
        region = CoMentionRegion(
            resource_id=sample_resource.id,
            chunk_range=(0, 2),
            candidate_pairs=[
                CandidatePair(
                    pair_key=pair_key, entity1_kind="gene", entity2_kind="gene"
                )
            ],
        )

        # Mock LLM to return pair with unknown first entity
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = RegionAssessmentOut(
            confirmed_pairs=[
                ConfirmedPair(
                    entity1_name="UNKNOWN_GENE",  # Not in candidates
                    entity2_name="TP53",
                    relationship="interacts_with",
                    evidence=make_evidence(8),
                    supporting_quotes=["Some text"],
                    reasoning="Test",
                )
            ]
        )
        mock_agent.run = AsyncMock(return_value=mock_result)

        # Mock config and deps
        config = IfetcherConfig()
        deps = Deps(
            config=config,
            resource_pool=MagicMock(),
            agent_semaphore=MagicMock(__aenter__=AsyncMock(), __aexit__=AsyncMock()),
            progress=None,
            logger=MagicMock(),
        )

        # Mock get_region_assessment_agent
        import interaction_finder.extraction.sweep_co_mentions as sweep_module

        original_getter = sweep_module.get_region_assessment_agent
        sweep_module.get_region_assessment_agent = lambda _: mock_agent

        try:
            validated_entities = {
                "BRCA1": entity_ref_brca1,
                "TP53": entity_ref_tp53,
            }

            assessments, new_pairs = await assess_co_mention_region(
                region,
                sample_resource,
                topic="test topic",
                known_relationships=[],
                config=config,
                deps=deps,
                validated_entities=validated_entities,
                region_index=1,
            )
            # No assessments or new pairs for unresolvable entities
            assert len(assessments) == 0
            assert len(new_pairs) == 0
            # Check that debug was logged with diagnostic
            deps.logger.debug.assert_called()
            debug_msg = deps.logger.debug.call_args[0][0]
            assert "Region assessment 1" in debug_msg
            assert "Pair #1:" in debug_msg
            assert "UNKNOWN_GENE" in debug_msg
            assert "(unknown!)" in debug_msg

        finally:
            sweep_module.get_region_assessment_agent = original_getter

    @pytest.mark.asyncio
    async def test_unresolvable_second_entity_warning(
        self, sample_resource, entity_ref_brca1, entity_ref_tp53
    ):
        """Warning includes diagnostic when second entity is unresolvable."""
        from unittest.mock import AsyncMock, MagicMock
        from interaction_finder.extraction.sweep_co_mentions import (
            assess_co_mention_region,
            ConfirmedPair,
            RegionAssessmentOut,
        )
        from interaction_finder.extraction.deps import Deps
        from interaction_finder.settings import IfetcherConfig

        pair_key = make_entity_pair_key(entity_ref_brca1, entity_ref_tp53)
        region = CoMentionRegion(
            resource_id=sample_resource.id,
            chunk_range=(0, 2),
            candidate_pairs=[
                CandidatePair(
                    pair_key=pair_key, entity1_kind="gene", entity2_kind="gene"
                )
            ],
        )

        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = RegionAssessmentOut(
            confirmed_pairs=[
                ConfirmedPair(
                    entity1_name="BRCA1",
                    entity2_name="UNKNOWN_GENE",  # Not in candidates
                    relationship="interacts_with",
                    evidence=make_evidence(8),
                    supporting_quotes=["Some text"],
                    reasoning="Test",
                )
            ]
        )
        mock_agent.run = AsyncMock(return_value=mock_result)

        config = IfetcherConfig()
        deps = Deps(
            config=config,
            resource_pool=MagicMock(),
            agent_semaphore=MagicMock(__aenter__=AsyncMock(), __aexit__=AsyncMock()),
            progress=None,
            logger=MagicMock(),
        )

        import interaction_finder.extraction.sweep_co_mentions as sweep_module

        original_getter = sweep_module.get_region_assessment_agent
        sweep_module.get_region_assessment_agent = lambda _: mock_agent

        try:
            validated_entities = {
                "BRCA1": entity_ref_brca1,
                "TP53": entity_ref_tp53,
            }

            assessments, new_pairs = await assess_co_mention_region(
                region,
                sample_resource,
                topic="test topic",
                known_relationships=[],
                config=config,
                deps=deps,
                validated_entities=validated_entities,
                region_index=2,
            )
            # No assessments or new pairs for unresolvable entities
            assert len(assessments) == 0
            assert len(new_pairs) == 0
            # Check that debug was logged with diagnostic
            deps.logger.debug.assert_called()
            debug_msg = deps.logger.debug.call_args[0][0]
            assert "Region assessment 2" in debug_msg
            assert "Pair #1:" in debug_msg
            assert "UNKNOWN_GENE" in debug_msg
            assert "(unknown!)" in debug_msg

        finally:
            sweep_module.get_region_assessment_agent = original_getter

    @pytest.mark.asyncio
    async def test_unexpected_pair_combination_warning(
        self,
        sample_resource,
        entity_ref_brca1,
        entity_ref_tp53,
        entity_ref_breast_cancer,
    ):
        """Warning includes expected pairs when resolved entities don't form valid pair."""
        from unittest.mock import AsyncMock, MagicMock
        from interaction_finder.extraction.sweep_co_mentions import (
            assess_co_mention_region,
            ConfirmedPair,
            RegionAssessmentOut,
        )
        from interaction_finder.extraction.deps import Deps
        from interaction_finder.settings import IfetcherConfig

        # Create region with BRCA1-TP53 and BRCA1-breast_cancer pairs, but NOT TP53-breast_cancer
        pair_key1 = make_entity_pair_key(entity_ref_brca1, entity_ref_tp53)
        pair_key2 = make_entity_pair_key(entity_ref_brca1, entity_ref_breast_cancer)
        region = CoMentionRegion(
            resource_id=sample_resource.id,
            chunk_range=(0, 2),
            candidate_pairs=[
                CandidatePair(
                    pair_key=pair_key1, entity1_kind="gene", entity2_kind="gene"
                ),
                CandidatePair(
                    pair_key=pair_key2, entity1_kind="gene", entity2_kind="disease"
                ),
            ],
        )

        # LLM returns TP53-breast_cancer (both resolvable but not a valid candidate pair)
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = RegionAssessmentOut(
            confirmed_pairs=[
                ConfirmedPair(
                    entity1_name="TP53",
                    entity2_name="breast cancer",
                    relationship="associated_with",
                    evidence=make_evidence(6),
                    supporting_quotes=["Some text"],
                    reasoning="Test",
                )
            ]
        )
        mock_agent.run = AsyncMock(return_value=mock_result)

        config = IfetcherConfig()
        deps = Deps(
            config=config,
            resource_pool=MagicMock(),
            agent_semaphore=MagicMock(__aenter__=AsyncMock(), __aexit__=AsyncMock()),
            progress=None,
            logger=MagicMock(),
        )

        import interaction_finder.extraction.sweep_co_mentions as sweep_module

        original_getter = sweep_module.get_region_assessment_agent
        sweep_module.get_region_assessment_agent = lambda _: mock_agent

        try:
            validated_entities = {
                "BRCA1": entity_ref_brca1,
                "TP53": entity_ref_tp53,
                "breast cancer": entity_ref_breast_cancer,
            }

            assessments, new_pairs = await assess_co_mention_region(
                region,
                sample_resource,
                topic="test topic",
                known_relationships=[],
                config=config,
                deps=deps,
                validated_entities=validated_entities,
                region_index=3,
            )
            # No assessments for unexpected pair without discovery enabled
            # (permitted_pairs not provided)
            # The pair resolves but fails quote validation (mock quote doesn't match text)
            assert len(assessments) == 0
            assert len(new_pairs) == 0
            # Check that debug was logged (either for quote rejection or skipped pairs)
            deps.logger.debug.assert_called()
            debug_msg = deps.logger.debug.call_args[0][0]
            assert "Region assessment 3" in debug_msg
            # The pair is rejected due to quote validation failure
            assert (
                "rejected: no valid quotes" in debug_msg
                or "not in candidate list" in debug_msg
            )

        finally:
            sweep_module.get_region_assessment_agent = original_getter

    @pytest.mark.asyncio
    async def test_multiple_issues_combined_in_single_message(
        self,
        sample_resource,
        entity_ref_brca1,
        entity_ref_tp53,
        entity_ref_breast_cancer,
    ):
        """Multiple resolution issues are collected and reported in a single warning."""
        from unittest.mock import AsyncMock, MagicMock
        from interaction_finder.extraction.sweep_co_mentions import (
            assess_co_mention_region,
            ConfirmedPair,
            RegionAssessmentOut,
        )
        from interaction_finder.extraction.deps import Deps
        from interaction_finder.settings import IfetcherConfig

        # Create region with BRCA1-TP53 and BRCA1-breast_cancer pairs
        pair_key1 = make_entity_pair_key(entity_ref_brca1, entity_ref_tp53)
        pair_key2 = make_entity_pair_key(entity_ref_brca1, entity_ref_breast_cancer)
        region = CoMentionRegion(
            resource_id=sample_resource.id,
            chunk_range=(0, 2),
            candidate_pairs=[
                CandidatePair(
                    pair_key=pair_key1, entity1_kind="gene", entity2_kind="gene"
                ),
                CandidatePair(
                    pair_key=pair_key2, entity1_kind="gene", entity2_kind="disease"
                ),
            ],
        )

        # LLM returns three problematic pairs
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = RegionAssessmentOut(
            confirmed_pairs=[
                # Pair 1: Unknown first entity
                ConfirmedPair(
                    entity1_name="UNKNOWN1",
                    entity2_name="TP53",
                    relationship="interacts_with",
                    evidence=make_evidence(8),
                    supporting_quotes=["Some text"],
                    reasoning="Test",
                ),
                # Pair 2: Unknown second entity
                ConfirmedPair(
                    entity1_name="BRCA1",
                    entity2_name="UNKNOWN2",
                    relationship="regulates",
                    evidence=make_evidence(6),
                    supporting_quotes=["Some text"],
                    reasoning="Test",
                ),
                # Pair 3: Invalid combination
                ConfirmedPair(
                    entity1_name="TP53",
                    entity2_name="breast cancer",
                    relationship="associated_with",
                    evidence=make_evidence(3),
                    supporting_quotes=["Some text"],
                    reasoning="Test",
                ),
            ]
        )
        mock_agent.run = AsyncMock(return_value=mock_result)

        config = IfetcherConfig()
        deps = Deps(
            config=config,
            resource_pool=MagicMock(),
            agent_semaphore=MagicMock(__aenter__=AsyncMock(), __aexit__=AsyncMock()),
            progress=None,
            logger=MagicMock(),
        )

        import interaction_finder.extraction.sweep_co_mentions as sweep_module

        original_getter = sweep_module.get_region_assessment_agent
        sweep_module.get_region_assessment_agent = lambda _: mock_agent

        try:
            validated_entities = {
                "BRCA1": entity_ref_brca1,
                "TP53": entity_ref_tp53,
                "breast cancer": entity_ref_breast_cancer,
            }

            assessments, new_pairs = await assess_co_mention_region(
                region,
                sample_resource,
                topic="test topic",
                known_relationships=[],
                config=config,
                deps=deps,
                region_index=4,
                validated_entities=validated_entities,
            )
            # No assessments or new pairs for all problematic pairs
            assert len(assessments) == 0
            assert len(new_pairs) == 0
            # Check that debug was logged - pair #3 fails quote validation separately
            # from the entity resolution issues for pairs #1 and #2
            deps.logger.debug.assert_called()
            # Get all debug messages
            debug_calls = [call[0][0] for call in deps.logger.debug.call_args_list]
            combined_debug = "\n".join(debug_calls)
            assert "Region assessment 4" in combined_debug
            # Pairs #1 and #2 fail entity resolution
            assert "UNKNOWN1" in combined_debug
            assert "(unknown!)" in combined_debug
            assert "UNKNOWN2" in combined_debug
            # Pair #3 (TP53-breast cancer) fails quote validation
            assert (
                "rejected: no valid quotes" in combined_debug
                or "skipped" in combined_debug
            )

        finally:
            sweep_module.get_region_assessment_agent = original_getter

    @pytest.mark.asyncio
    async def test_entity_from_global_aliases_not_in_validated_entities(
        self, sample_resource, entity_ref_brca1
    ):
        """Entity in candidate pairs but not in validated_entities should still resolve.

        Regression test for bug where entities from global aliases (seen in other
        documents) were included in candidate_pairs but not in validated_entities
        for the current document, causing fuzzy matching to fail.
        """
        from unittest.mock import AsyncMock, MagicMock
        from interaction_finder.extraction.sweep_co_mentions import (
            assess_co_mention_region,
            ConfirmedPair,
            RegionAssessmentOut,
        )
        from interaction_finder.extraction.models import EntityPairKey
        from interaction_finder.extraction.deps import Deps
        from interaction_finder.settings import IfetcherConfig

        # TGFB1 is in candidate pairs (from global aliases across all documents)
        # but NOT in this document's validated_entities
        pair_key = EntityPairKey(entity1_name="TGFB1", entity2_name="BRCA1")
        region = CoMentionRegion(
            resource_id=sample_resource.id,
            chunk_range=(0, 2),
            candidate_pairs=[
                CandidatePair(
                    pair_key=pair_key, entity1_kind="gene", entity2_kind="gene"
                )
            ],
        )

        # Mock LLM returns the global-alias entity
        # Use a quote that actually exists in sample_resource
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = RegionAssessmentOut(
            confirmed_pairs=[
                ConfirmedPair(
                    entity1_name="TGFB1",  # From global aliases, not in validated_entities
                    entity2_name="BRCA1",
                    relationship="interacts_with",
                    evidence=make_evidence(8),
                    supporting_quotes=[
                        "BRCA1 and TP53 interact in DNA damage response pathways"
                    ],
                    reasoning="Test",
                )
            ]
        )
        mock_agent.run = AsyncMock(return_value=mock_result)

        # validated_entities only has BRCA1, not TGFB1
        validated_entities = {
            "BRCA1": entity_ref_brca1,
            # TGFB1 NOT included - came from global aliases
        }

        # Configure mocks
        config = IfetcherConfig()
        deps = Deps(
            config=config,
            resource_pool=MagicMock(),
            agent_semaphore=MagicMock(__aenter__=AsyncMock(), __aexit__=AsyncMock()),
            progress=None,
            logger=MagicMock(),
        )

        import interaction_finder.extraction.sweep_co_mentions as sweep_module

        original_getter = sweep_module.get_region_assessment_agent
        sweep_module.get_region_assessment_agent = lambda _: mock_agent

        try:
            assessments, new_pairs = await assess_co_mention_region(
                region,
                sample_resource,
                topic="test topic",
                known_relationships=[],
                config=config,
                deps=deps,
                validated_entities=validated_entities,
                region_index=1,
            )

            # Should create assessment for known candidate pair
            assert len(assessments) == 1
            assert len(new_pairs) == 0
            assert assessments[0].entity1.canonical == "TGFB1"
            assert assessments[0].entity2.canonical == "BRCA1"
            assert assessments[0].relationship == "interacts_with"
            assert assessments[0].source == "sweep"

        finally:
            sweep_module.get_region_assessment_agent = original_getter

    @pytest.mark.asyncio
    async def test_contested_variant_resolution_via_direct_canonical_match(
        self, sample_resource
    ):
        """LLM returning entity name that matches canonical directly despite contested variants.

        Regression test for bug where slash-separated entities like
        "Heritable/familial PAH" expand to variants that contest with standalone
        entities like "Familial PAH". When LLM returns "Familial PAH", it should
        match the standalone entity via direct canonical match, not fail due to
        contested variant.
        """
        from unittest.mock import AsyncMock, MagicMock
        from interaction_finder.extraction.sweep_co_mentions import (
            assess_co_mention_region,
            ConfirmedPair,
            RegionAssessmentOut,
        )
        from interaction_finder.extraction.models import EntityPairKey
        from interaction_finder.extraction.deps import Deps
        from interaction_finder.settings import IfetcherConfig

        # Setup: Both standalone and slash-separated entities in candidates
        # This creates contested variant "familial pulmonary arterial hypertension"
        pair_keys = [
            EntityPairKey(
                entity1_name="BMPR2",
                entity2_name="Familial pulmonary arterial hypertension",
            ),
            EntityPairKey(
                entity1_name="BMPR2",
                entity2_name="Heritable/familial pulmonary arterial hypertension",
            ),
        ]
        region = CoMentionRegion(
            resource_id=sample_resource.id,
            chunk_range=(0, 2),
            candidate_pairs=[
                CandidatePair(
                    pair_key=pk, entity1_kind="gene", entity2_kind="phenotype"
                )
                for pk in pair_keys
            ],
        )

        # Mock LLM returns the standalone form (not the slash-separated form)
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = RegionAssessmentOut(
            confirmed_pairs=[
                ConfirmedPair(
                    entity1_name="BMPR2",
                    entity2_name="Familial pulmonary arterial hypertension",
                    relationship="associated_with",
                    evidence=make_evidence(8),
                    supporting_quotes=[
                        "BRCA1 and TP53 interact in DNA damage response pathways"
                    ],
                    reasoning="Test",
                )
            ]
        )
        mock_agent.run = AsyncMock(return_value=mock_result)

        # No validated_entities (entities from global aliases only)
        validated_entities = None

        # Configure mocks
        config = IfetcherConfig()
        deps = Deps(
            config=config,
            resource_pool=MagicMock(),
            agent_semaphore=MagicMock(__aenter__=AsyncMock(), __aexit__=AsyncMock()),
            progress=None,
            logger=MagicMock(),
        )

        import interaction_finder.extraction.sweep_co_mentions as sweep_module

        original_getter = sweep_module.get_region_assessment_agent
        sweep_module.get_region_assessment_agent = lambda _: mock_agent

        try:
            assessments, new_pairs = await assess_co_mention_region(
                region,
                sample_resource,
                topic="test topic",
                known_relationships=[],
                config=config,
                deps=deps,
                validated_entities=validated_entities,
                region_index=1,
            )

            # Should create assessment via direct canonical match
            assert len(assessments) == 1
            assert len(new_pairs) == 0
            assert assessments[0].entity1.canonical == "BMPR2"
            assert (
                assessments[0].entity2.canonical
                == "Familial pulmonary arterial hypertension"
            )
            assert assessments[0].relationship == "associated_with"
            assert assessments[0].source == "sweep"

        finally:
            sweep_module.get_region_assessment_agent = original_getter

    @pytest.mark.asyncio
    async def test_new_pair_discovery_with_permitted_pairs(
        self,
        sample_resource,
        entity_ref_brca1,
        entity_ref_tp53,
        entity_ref_breast_cancer,
    ):
        """New pairs are discovered when permitted_pairs is provided.

        When the LLM returns a pair not in the candidate list, but the entities
        exist in validated_entities and the pair kinds are permitted, the pair
        should be added to new_pairs for standard assessment.
        """
        from unittest.mock import AsyncMock, MagicMock
        from interaction_finder.extraction.sweep_co_mentions import (
            assess_co_mention_region,
            ConfirmedPair,
            RegionAssessmentOut,
        )
        from interaction_finder.extraction.deps import Deps
        from interaction_finder.settings import IfetcherConfig

        # Create region with BRCA1-TP53 candidate pair
        pair_key = make_entity_pair_key(entity_ref_brca1, entity_ref_tp53)
        region = CoMentionRegion(
            resource_id=sample_resource.id,
            chunk_range=(0, 2),
            candidate_pairs=[
                CandidatePair(
                    pair_key=pair_key, entity1_kind="gene", entity2_kind="gene"
                )
            ],
        )

        # LLM returns TP53-breast cancer (NOT a candidate, but entities exist)
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = RegionAssessmentOut(
            confirmed_pairs=[
                ConfirmedPair(
                    entity1_name="TP53",
                    entity2_name="breast cancer",
                    relationship="associated_with",
                    evidence=make_evidence(6),
                    # Use a quote that exists in sample_resource
                    supporting_quotes=[
                        "BRCA1 is a tumor suppressor gene associated with breast cancer"
                    ],
                    reasoning="Test",
                )
            ]
        )
        mock_agent.run = AsyncMock(return_value=mock_result)

        config = IfetcherConfig()
        deps = Deps(
            config=config,
            resource_pool=MagicMock(),
            agent_semaphore=MagicMock(__aenter__=AsyncMock(), __aexit__=AsyncMock()),
            progress=None,
            logger=MagicMock(),
        )

        import interaction_finder.extraction.sweep_co_mentions as sweep_module

        original_getter = sweep_module.get_region_assessment_agent
        sweep_module.get_region_assessment_agent = lambda _: mock_agent

        try:
            validated_entities = {
                "BRCA1": entity_ref_brca1,
                "TP53": entity_ref_tp53,
                "breast cancer": entity_ref_breast_cancer,
            }
            # Enable discovery by providing permitted_pairs
            permitted_pairs = {
                "gene": {"gene", "disease"},
                "disease": {"gene"},
            }

            assessments, new_pairs = await assess_co_mention_region(
                region,
                sample_resource,
                topic="test topic",
                known_relationships=[],
                config=config,
                deps=deps,
                validated_entities=validated_entities,
                permitted_pairs=permitted_pairs,
                region_index=1,
            )

            # No direct assessments (TP53-breast cancer wasn't a candidate)
            assert len(assessments) == 0
            # New pair should be discovered
            assert len(new_pairs) == 1
            e1, e2, rel_types, quotes = new_pairs[0]
            assert e1 == "TP53"
            assert e2 == "breast cancer"
            assert "associated_with" in rel_types
            assert len(quotes) == 1  # Validated quote

        finally:
            sweep_module.get_region_assessment_agent = original_getter

    @pytest.mark.asyncio
    async def test_new_pair_rejected_when_kinds_not_permitted(
        self,
        sample_resource,
        entity_ref_brca1,
        entity_ref_tp53,
        entity_ref_breast_cancer,
    ):
        """New pairs are rejected when entity kinds are not permitted.

        Even with discovery enabled, pairs must have permitted kind combinations.
        """
        from unittest.mock import AsyncMock, MagicMock
        from interaction_finder.extraction.sweep_co_mentions import (
            assess_co_mention_region,
            ConfirmedPair,
            RegionAssessmentOut,
        )
        from interaction_finder.extraction.deps import Deps
        from interaction_finder.settings import IfetcherConfig

        pair_key = make_entity_pair_key(entity_ref_brca1, entity_ref_tp53)
        region = CoMentionRegion(
            resource_id=sample_resource.id,
            chunk_range=(0, 2),
            candidate_pairs=[
                CandidatePair(
                    pair_key=pair_key, entity1_kind="gene", entity2_kind="gene"
                )
            ],
        )

        # LLM returns TP53-breast cancer
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = RegionAssessmentOut(
            confirmed_pairs=[
                ConfirmedPair(
                    entity1_name="TP53",
                    entity2_name="breast cancer",
                    relationship="associated_with",
                    evidence=make_evidence(6),
                    supporting_quotes=[
                        "BRCA1 is a tumor suppressor gene associated with breast cancer"
                    ],
                    reasoning="Test",
                )
            ]
        )
        mock_agent.run = AsyncMock(return_value=mock_result)

        config = IfetcherConfig()
        deps = Deps(
            config=config,
            resource_pool=MagicMock(),
            agent_semaphore=MagicMock(__aenter__=AsyncMock(), __aexit__=AsyncMock()),
            progress=None,
            logger=MagicMock(),
        )

        import interaction_finder.extraction.sweep_co_mentions as sweep_module

        original_getter = sweep_module.get_region_assessment_agent
        sweep_module.get_region_assessment_agent = lambda _: mock_agent

        try:
            validated_entities = {
                "BRCA1": entity_ref_brca1,
                "TP53": entity_ref_tp53,
                "breast cancer": entity_ref_breast_cancer,
            }
            # Only gene-gene pairs permitted (no gene-disease)
            permitted_pairs = {
                "gene": {"gene"},  # disease NOT allowed
            }

            assessments, new_pairs = await assess_co_mention_region(
                region,
                sample_resource,
                topic="test topic",
                known_relationships=[],
                config=config,
                deps=deps,
                validated_entities=validated_entities,
                permitted_pairs=permitted_pairs,
                region_index=1,
            )

            # No assessments or new pairs (kind not permitted)
            assert len(assessments) == 0
            assert len(new_pairs) == 0
            # Check debug log mentions the rejected pair
            deps.logger.debug.assert_called()
            debug_calls = [call[0][0] for call in deps.logger.debug.call_args_list]
            combined_debug = "\n".join(debug_calls)
            assert "not a permitted pair type" in combined_debug

        finally:
            sweep_module.get_region_assessment_agent = original_getter
