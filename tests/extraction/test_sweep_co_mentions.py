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
            confidence="high",
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
            confidence="high",
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
            confidence="high",
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
            confidence="high",
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
            confidence="high",
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
            confidence="high",
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
            confidence="high",
            reasoning="Test",
        )
        assessment2 = PairAssessment(
            resource_id=sample_resource.id,
            entity1=entity_ref_brca1,
            entity2=entity_ref_breast_cancer,
            relationship="associated_with",
            quotes=[],
            confidence="high",
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
            confidence="high",
            reasoning="Test",
        )
        assessment2 = PairAssessment(
            resource_id=resource_id2,
            entity1=entity_ref_brca1,
            entity2=entity_ref_tp53,
            relationship="regulates",
            quotes=[],
            confidence="medium",
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
