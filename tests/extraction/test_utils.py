"""Tests for extraction utility functions."""

import pytest

from interaction_finder.extraction.models import (
    EntityMention,
    PairAssessment,
    ProximalEntitySet,
)
from interaction_finder.extraction.utils import (
    _is_valid_entity_form,
    build_pair_spread,
    build_permitted_pairs,
    build_text_region,
    collect_relevant_text_for_quotes,
    extract_all_forms,
    find_substring_entities,
    get_relationship_polarity,
    identify_proximal_sets,
    make_entity_pair_key,
    normalize_for_comparison,
    strip_kind_annotation,
)
from interaction_finder.resources import ResourcePool


class TestBuildPermittedPairs:
    """Tests for build_permitted_pairs function."""

    def test_two_different_kinds_no_self_pairs(self):
        """Test that two different kinds only allow cross-pairs."""
        result = build_permitted_pairs(["gene", "disease"])
        assert result == {"gene": {"disease"}, "disease": {"gene"}}

    def test_single_kind_allows_self_pairs(self):
        """Test that a single kind allows self-pairs."""
        result = build_permitted_pairs(["gene"])
        assert result == {"gene": {"gene"}}

    def test_repeated_kind_allows_self_pairs(self):
        """Test that repeating a kind allows self-pairs."""
        result = build_permitted_pairs(["gene", "gene", "disease"])
        assert result == {"gene": {"gene", "disease"}, "disease": {"gene"}}

    def test_multiple_repeated_kinds(self):
        """Test multiple kinds with repetition."""
        result = build_permitted_pairs(["gene", "gene", "disease", "disease"])
        assert result == {"gene": {"gene", "disease"}, "disease": {"gene", "disease"}}

    def test_three_different_kinds(self):
        """Test three different kinds with no self-pairs."""
        result = build_permitted_pairs(["gene", "disease", "protein"])
        assert result == {
            "gene": {"disease", "protein"},
            "disease": {"gene", "protein"},
            "protein": {"gene", "disease"},
        }

    def test_three_kinds_one_repeated(self):
        """Test three kinds where one is repeated."""
        result = build_permitted_pairs(["gene", "gene", "disease", "protein"])
        assert result == {
            "gene": {"gene", "disease", "protein"},
            "disease": {"gene", "protein"},
            "protein": {"gene", "disease"},
        }

    def test_order_independence(self):
        """Test that input order doesn't affect result."""
        result1 = build_permitted_pairs(["gene", "disease", "gene"])
        result2 = build_permitted_pairs(["gene", "gene", "disease"])
        result3 = build_permitted_pairs(["disease", "gene", "gene"])
        assert result1 == result2 == result3

    def test_empty_list(self):
        """Test empty list returns empty dict."""
        result = build_permitted_pairs([])
        assert result == {}

    def test_many_repetitions(self):
        """Test that many repetitions still work (only need 2)."""
        result = build_permitted_pairs(["gene"] * 5 + ["disease"])
        assert result == {"gene": {"gene", "disease"}, "disease": {"gene"}}


class TestStripKindAnnotation:
    """Tests for strip_kind_annotation function."""

    def test_strips_gene_annotation(self):
        """Test stripping (gene) annotation."""
        assert strip_kind_annotation("BRCA1 (gene)") == "BRCA1"
        assert strip_kind_annotation("TP53 (gene)") == "TP53"

    def test_strips_phenotype_annotation(self):
        """Test stripping (phenotype) annotation."""
        assert strip_kind_annotation("Iron deficiency (phenotype)") == "Iron deficiency"
        assert (
            strip_kind_annotation("Right ventricular hypertrophy (phenotype)")
            == "Right ventricular hypertrophy"
        )

    def test_strips_disease_annotation(self):
        """Test stripping (disease) annotation."""
        assert strip_kind_annotation("breast cancer (disease)") == "breast cancer"

    def test_strips_protein_annotation(self):
        """Test stripping (protein) annotation."""
        assert strip_kind_annotation("p53 (protein)") == "p53"

    def test_handles_underscored_kinds(self):
        """Test stripping annotations with underscores."""
        assert strip_kind_annotation("test (some_kind)") == "test"

    def test_preserves_name_without_annotation(self):
        """Test that names without annotations are unchanged."""
        assert strip_kind_annotation("BRCA1") == "BRCA1"
        assert strip_kind_annotation("Iron deficiency") == "Iron deficiency"
        assert strip_kind_annotation("TP53") == "TP53"

    def test_handles_multiple_words(self):
        """Test entities with multiple words."""
        assert (
            strip_kind_annotation("pulmonary arterial hypertension (phenotype)")
            == "pulmonary arterial hypertension"
        )

    def test_handles_extra_whitespace(self):
        """Test handling of extra whitespace."""
        assert strip_kind_annotation("BRCA1  (gene)") == "BRCA1"
        assert strip_kind_annotation("BRCA1 (gene) ") == "BRCA1"

    def test_preserves_parentheses_in_middle(self):
        """Test that parentheses not at end are preserved."""
        # This should NOT match our pattern (not at end of string)
        assert strip_kind_annotation("HIF2α (HIF2α)") == "HIF2α (HIF2α)"
        # But if followed by kind annotation, strip only the kind
        assert strip_kind_annotation("HIF2α (HIF2α) (gene)") == "HIF2α (HIF2α)"

    def test_case_sensitivity(self):
        """Test that all kinds of annotations are stripped regardless of case."""
        # Pattern now matches any alphanumeric string in parentheses
        assert strip_kind_annotation("BRCA1 (Gene)") == "BRCA1"
        assert strip_kind_annotation("BRCA1 (GENE)") == "BRCA1"
        assert strip_kind_annotation("BRCA1 (gene)") == "BRCA1"

    def test_strips_uppercase_abbreviations(self):
        """Test that uppercase abbreviations in parentheses are stripped."""
        assert (
            strip_kind_annotation("Pulmonary arterial hypertension (PAH)")
            == "Pulmonary arterial hypertension"
        )
        assert strip_kind_annotation("Breast cancer (BC)") == "Breast cancer"
        assert strip_kind_annotation("TP53 (P53)") == "TP53"


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
        # BRCA is parent (general), BRCA1 is child (specific)
        assert pairs[0] == ("BRCA", "BRCA1")

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
        # Check both pairs are found (general, specific)
        pair_set = set(pairs)
        assert ("BRCA", "BRCA1") in pair_set
        assert ("TP", "TP53") in pair_set

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

    def test_finds_exact_normalized_match(self):
        """Test finding entities with identical normalized names."""
        entities = {
            "Pulmonary arterial hypertension": EntityMention(
                kind="phenotype",
                name="Pulmonary arterial hypertension",
                aliases=["PAH"],
                quotes=[],
                reasoning="test",
            ),
            "Pulmonary Arterial Hypertension": EntityMention(
                kind="phenotype",
                name="Pulmonary Arterial Hypertension",
                aliases=["PAH"],
                quotes=[],
                reasoning="test",
            ),
        }

        pairs = find_substring_entities(entities)
        assert len(pairs) == 1
        # First entity in dict order is kept as parent
        assert pairs[0] == (
            "Pulmonary arterial hypertension",
            "Pulmonary Arterial Hypertension",
        )

    def test_finds_exact_normalized_match_case_only(self):
        """Test normalized match handles case-only differences."""
        entities = {
            "brca1": EntityMention(
                kind="gene",
                name="brca1",
                aliases=["brca1"],
                quotes=[],
                reasoning="test",
            ),
            "BRCA1": EntityMention(
                kind="gene",
                name="BRCA1",
                aliases=["BRCA1"],
                quotes=[],
                reasoning="test",
            ),
        }

        pairs = find_substring_entities(entities)
        assert len(pairs) == 1
        # First entity in dict order is kept as parent
        assert pairs[0] == ("brca1", "BRCA1")

    def test_combines_exact_match_and_substring(self):
        """Test handling mix of exact matches and substring relationships."""
        entities = {
            "pah": EntityMention(
                kind="phenotype",
                name="pah",
                aliases=["pah"],
                quotes=[],
                reasoning="test",
            ),
            "PAH": EntityMention(
                kind="phenotype",
                name="PAH",
                aliases=["PAH"],
                quotes=[],
                reasoning="test",
            ),
            "BRCA1": EntityMention(
                kind="gene",
                name="BRCA1",
                aliases=["BRCA1"],
                quotes=[],
                reasoning="test",
            ),
            "BRCA": EntityMention(
                kind="gene",
                name="BRCA",
                aliases=["BRCA"],
                quotes=[],
                reasoning="test",
            ),
        }

        pairs = find_substring_entities(entities)
        assert len(pairs) == 2
        pair_set = set(pairs)
        # Exact match: pah/PAH (first in dict order wins)
        assert ("pah", "PAH") in pair_set
        # Substring match: BRCA is general, BRCA1 is specific
        assert ("BRCA", "BRCA1") in pair_set


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


class TestGetRelationshipPolarity:
    """Tests for get_relationship_polarity function."""

    def test_gets_polarity_for_known_relationship(self):
        """Test looking up polarity for a known relationship."""
        polarity_map = {
            "increases_risk_of": "supporting",
            "protects_against": "refuting",
            "regulates": "neutral",
            "spatial_colocalization": "irrelevant",
        }

        assert (
            get_relationship_polarity("increases_risk_of", polarity_map) == "supporting"
        )
        assert get_relationship_polarity("protects_against", polarity_map) == "refuting"
        assert get_relationship_polarity("regulates", polarity_map) == "neutral"
        assert (
            get_relationship_polarity("spatial_colocalization", polarity_map)
            == "irrelevant"
        )

    def test_raises_key_error_for_unknown_relationship(self):
        """Test that KeyError is raised for unknown relationship."""
        polarity_map = {"increases_risk_of": "supporting"}

        with pytest.raises(KeyError):
            get_relationship_polarity("unknown_relationship", polarity_map)

    def test_works_with_empty_map(self):
        """Test behavior with empty polarity map."""
        polarity_map = {}

        with pytest.raises(KeyError):
            get_relationship_polarity("any_relationship", polarity_map)


class TestBuildPairSpread:
    """Tests for build_pair_spread function."""

    def setup_method(self):
        """Set up test data."""
        self.pool = ResourcePool()
        self.resource = self.pool.add(
            url="http://example.com",
            title="Test",
            document_text="Test text with entities.",
        )

        # Create test entities
        self.entity1 = EntityMention(
            kind="gene",
            name="BRCA1",
            aliases=["BRCA1"],
            quotes=[],
            reasoning="test",
        )
        self.entity2 = EntityMention(
            kind="disease",
            name="breast cancer",
            aliases=["breast cancer"],
            quotes=[],
            reasoning="test",
        )

    def test_groups_assessments_by_polarity(self):
        """Test that assessments are correctly grouped by polarity."""
        polarity_map = {
            "increases_risk_of": "supporting",
            "protects_against": "refuting",
            "regulates": "neutral",
            "spatial_colocalization": "irrelevant",
        }

        assessments = [
            PairAssessment(
                resource_id=self.resource.id,
                entity1=self.entity1,
                entity2=self.entity2,
                relationship="increases_risk_of",
                quotes=[],
                confidence="high",
                reasoning="test",
            ),
            PairAssessment(
                resource_id=self.resource.id,
                entity1=self.entity1,
                entity2=self.entity2,
                relationship="protects_against",
                quotes=[],
                confidence="medium",
                reasoning="test",
            ),
            PairAssessment(
                resource_id=self.resource.id,
                entity1=self.entity1,
                entity2=self.entity2,
                relationship="regulates",
                quotes=[],
                confidence="low",
                reasoning="test",
            ),
        ]

        spread = build_pair_spread(assessments, polarity_map)

        assert len(spread.supporting) == 1
        assert spread.supporting[0].relationship == "increases_risk_of"
        assert len(spread.refuting) == 1
        assert spread.refuting[0].relationship == "protects_against"
        assert len(spread.neutral) == 1
        assert spread.neutral[0].relationship == "regulates"
        assert len(spread.irrelevant) == 0

    def test_handles_all_same_polarity(self):
        """Test with all assessments having same polarity."""
        polarity_map = {
            "increases_risk_of": "supporting",
            "causes": "supporting",
            "associated_with": "supporting",
        }

        assessments = [
            PairAssessment(
                resource_id=self.resource.id,
                entity1=self.entity1,
                entity2=self.entity2,
                relationship="increases_risk_of",
                quotes=[],
                confidence="high",
                reasoning="test",
            ),
            PairAssessment(
                resource_id=self.resource.id,
                entity1=self.entity1,
                entity2=self.entity2,
                relationship="causes",
                quotes=[],
                confidence="high",
                reasoning="test",
            ),
            PairAssessment(
                resource_id=self.resource.id,
                entity1=self.entity1,
                entity2=self.entity2,
                relationship="associated_with",
                quotes=[],
                confidence="medium",
                reasoning="test",
            ),
        ]

        spread = build_pair_spread(assessments, polarity_map)

        assert len(spread.supporting) == 3
        assert len(spread.refuting) == 0
        assert len(spread.neutral) == 0
        assert len(spread.irrelevant) == 0

    def test_handles_empty_assessments(self):
        """Test with empty assessments list."""
        polarity_map = {"increases_risk_of": "supporting"}

        spread = build_pair_spread([], polarity_map)

        assert len(spread.supporting) == 0
        assert len(spread.refuting) == 0
        assert len(spread.neutral) == 0
        assert len(spread.irrelevant) == 0

    def test_raises_key_error_for_unmapped_relationship(self):
        """Test that KeyError is raised for unmapped relationship."""
        polarity_map = {"increases_risk_of": "supporting"}

        assessments = [
            PairAssessment(
                resource_id=self.resource.id,
                entity1=self.entity1,
                entity2=self.entity2,
                relationship="unknown_relationship",
                quotes=[],
                confidence="high",
                reasoning="test",
            ),
        ]

        with pytest.raises(KeyError):
            build_pair_spread(assessments, polarity_map)

    def test_contentious_pair_detection(self):
        """Test identifying contentious pairs (supporting + refuting)."""
        polarity_map = {
            "increases_risk_of": "supporting",
            "protects_against": "refuting",
        }

        assessments = [
            PairAssessment(
                resource_id=self.resource.id,
                entity1=self.entity1,
                entity2=self.entity2,
                relationship="increases_risk_of",
                quotes=[],
                confidence="high",
                reasoning="test",
            ),
            PairAssessment(
                resource_id=self.resource.id,
                entity1=self.entity1,
                entity2=self.entity2,
                relationship="protects_against",
                quotes=[],
                confidence="high",
                reasoning="test",
            ),
        ]

        spread = build_pair_spread(assessments, polarity_map)

        # Contentious: has both supporting and refuting
        is_contentious = bool(spread.supporting and spread.refuting)
        assert is_contentious
        assert len(spread.supporting) == 1
        assert len(spread.refuting) == 1


class TestIsValidEntityForm:
    """Tests for _is_valid_entity_form validation function."""

    def test_accepts_valid_gene_names(self):
        """Test that valid gene names are accepted."""
        assert _is_valid_entity_form("BRCA1")
        assert _is_valid_entity_form("TP53")
        assert _is_valid_entity_form("p53")
        assert _is_valid_entity_form("PAH")
        assert _is_valid_entity_form("BMPR2")
        assert _is_valid_entity_form("ACVRL1")

    def test_rejects_pure_numbers(self):
        """Test that pure numbers are rejected."""
        assert not _is_valid_entity_form("1")
        assert not _is_valid_entity_form("5")
        assert not _is_valid_entity_form("123")
        assert not _is_valid_entity_form("1-5")
        assert not _is_valid_entity_form("1 2 3")

    def test_rejects_list_like_content(self):
        """Test that list-like content is rejected."""
        assert not _is_valid_entity_form("1, 5, 8")
        assert not _is_valid_entity_form("1,5,8")
        assert not _is_valid_entity_form("BRCA1, BRCA2")
        assert not _is_valid_entity_form("gene; protein")
        assert not _is_valid_entity_form("1/5/8")  # Multiple slashes

    def test_rejects_single_slash_but_accepts_gene_aliases(self):
        """Test slash handling."""
        # Single slash is OK for gene names like "SMAD1/5/9"
        # But we reject multiple slashes as they indicate lists
        assert _is_valid_entity_form("SMAD1/5")  # Single slash OK
        assert not _is_valid_entity_form("1/5/8")  # Multiple slashes rejected

    def test_rejects_very_short_forms(self):
        """Test that very short forms (< 3 chars) are rejected."""
        assert not _is_valid_entity_form("1")
        assert not _is_valid_entity_form("X")
        assert not _is_valid_entity_form("Y")
        assert not _is_valid_entity_form("ab")
        assert not _is_valid_entity_form("52")

    def test_rejects_empty_strings(self):
        """Test that empty strings are rejected."""
        assert not _is_valid_entity_form("")
        assert not _is_valid_entity_form("   ")

    def test_requires_sufficient_alphabetic_characters(self):
        """Test that sufficient alphabetic characters are required."""
        assert _is_valid_entity_form("p53")  # Short form: 1 letter OK
        assert _is_valid_entity_form("HLA")  # 3 letters = valid
        assert not _is_valid_entity_form(
            "1a2"
        )  # Only 1 letter in short form, but mostly numbers
        assert not _is_valid_entity_form(
            "123a"
        )  # Only 1 letter but > 4 chars total = needs 2+ letters

    def test_accepts_valid_disease_names(self):
        """Test that valid disease names are accepted."""
        assert _is_valid_entity_form("Pulmonary arterial hypertension")
        assert _is_valid_entity_form("breast cancer")
        assert _is_valid_entity_form("hereditary hemorrhagic telangiectasia")

    def test_handles_whitespace(self):
        """Test proper handling of whitespace."""
        assert _is_valid_entity_form("  BRCA1  ")  # Leading/trailing spaces OK
        assert _is_valid_entity_form("bone morphogenetic protein")  # Internal spaces OK


class TestExtractAllForms:
    """Tests for extract_all_forms function."""

    def test_expands_parenthetical_content(self):
        """Test expansion of parenthetical content."""
        forms = extract_all_forms("PAH (Pulmonary arterial hypertension)", [])
        assert "PAH" in forms
        assert "Pulmonary arterial hypertension" in forms
        # Original form may or may not be included depending on validation
        assert len(forms) >= 2

    def test_includes_aliases(self):
        """Test that aliases are included."""
        forms = extract_all_forms("Telangiectasia", ["HHT"])
        assert "Telangiectasia" in forms
        assert "HHT" in forms

    def test_filters_kind_annotations(self):
        """Test that kind annotations are filtered."""
        forms = extract_all_forms("BRCA1 (gene)", [])
        assert "BRCA1" in forms
        assert "gene" not in forms  # Single lowercase word filtered
        assert "BRCA1 (gene)" not in forms  # Invalid form filtered

    def test_filters_list_content_from_parens(self):
        """Test that list-like parenthetical content is filtered."""
        forms = extract_all_forms("R-SMADs (1, 5, 8)", [])
        assert "R-SMADs" in forms
        assert "1, 5, 8" not in forms  # List content filtered
        assert "1" not in forms
        assert "5" not in forms
        assert "8" not in forms

    def test_filters_pure_numbers(self):
        """Test that pure numbers are filtered out."""
        # Even if somehow passed as entity name or alias
        forms = extract_all_forms("ACVRL1", ["1", "52"])
        assert "ACVRL1" in forms
        assert "1" not in forms  # Pure number filtered
        assert "52" not in forms  # Pure number filtered

    def test_handles_gene_names_with_numbers(self):
        """Test proper handling of gene names containing numbers."""
        forms = extract_all_forms("ACVRL1", [])
        assert "ACVRL1" in forms
        # Should not extract "1" separately

        forms = extract_all_forms("HLA-DPA1", [])
        assert "HLA-DPA1" in forms
        # Should not extract "1" separately

    def test_handles_slash_separated_genes(self):
        """Test handling of slash-separated gene names."""
        forms = extract_all_forms("SMAD1/5/9", [])
        assert "SMAD1/5/9" not in forms  # Multiple slashes = invalid
        # Should not extract individual numbers

    def test_complex_parenthetical_case(self):
        """Test complex case with gene name and abbreviation."""
        # When parens are at the end, should expand properly
        forms = extract_all_forms(
            "bone morphogenetic protein receptor type 2 (BMPR2)", []
        )
        # Should keep both forms
        assert "BMPR2" in forms
        assert "bone morphogenetic protein receptor type 2" in forms

        # When entity name has trailing words after parens, nothing expandable
        # (This is not a valid pattern for extraction - should be caught earlier)
        forms = extract_all_forms(
            "bone morphogenetic protein receptor type 2 (BMPR2) gene", []
        )
        # Has parens but not at end, so rejected as a whole
        assert len(forms) == 0

    def test_filters_single_characters_from_complex_names(self):
        """Test that single characters from complex names are filtered."""
        # Simulating entity names that might be extracted
        forms = extract_all_forms("activin receptor like kinase 1", ["ALK1", "1"])
        assert "activin receptor like kinase 1" in forms
        assert "ALK1" in forms
        assert "1" not in forms  # Single character filtered

    def test_handles_multiple_aliases_with_parentheticals(self):
        """Test multiple aliases with parenthetical content."""
        forms = extract_all_forms(
            "Hereditary hemorrhagic telangiectasia (HHT)",
            ["HHT", "Osler-Weber-Rendu syndrome"],
        )
        assert "Hereditary hemorrhagic telangiectasia" in forms
        assert "HHT" in forms
        assert "Osler-Weber-Rendu syndrome" in forms

    def test_deduplicates_forms(self):
        """Test that duplicate forms are removed."""
        forms = extract_all_forms("PAH (PAH)", ["PAH"])
        # Should only have one "PAH" entry
        assert forms.count("PAH") == 1

    def test_returns_sorted_list(self):
        """Test that output is sorted."""
        forms = extract_all_forms("Zebra (AAA)", ["MMM"])
        assert forms == sorted(forms)
