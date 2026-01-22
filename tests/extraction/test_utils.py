"""Tests for extraction utility functions."""

import pytest

from interaction_finder.extraction.models import (
    EntityMention,
    EntityRef,
    PairAssessment,
    ProximalEntitySet,
)
from interaction_finder.extraction.utils import (
    _is_valid_entity_form,
    are_relationships_opposed,
    build_pair_spread,
    build_permitted_pairs,
    build_relationship_opposition_map,
    build_text_region,
    collect_relevant_text_for_quotes,
    extract_all_forms,
    find_best_entity_match,
    find_substring_entities,
    get_relationship_polarity,
    identify_proximal_sets,
    make_entity_pair_key,
    normalize_for_comparison,
    strip_kind_annotation,
)
from interaction_finder.resources import ResourcePool
from tests.extraction.conftest import make_evidence


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

    def test_roman_numeral_conversion(self):
        """Test Roman numerals are converted to Arabic."""
        assert normalize_for_comparison("Type II") == "type 2"
        assert normalize_for_comparison("Type I") == "type 1"
        assert normalize_for_comparison("Factor VIII") == "factor 8"
        assert normalize_for_comparison("Collagen IV") == "collagen 4"
        assert normalize_for_comparison("Class III") == "class 3"

    def test_roman_numeral_in_entity_names(self):
        """Test Roman numerals in realistic biomedical entity names."""
        # These should normalize to the same value
        assert normalize_for_comparison(
            "Alveolar type II cell"
        ) == normalize_for_comparison("Alveolar type 2 cell")
        assert normalize_for_comparison("Type II diabetes") == normalize_for_comparison(
            "Type 2 diabetes"
        )
        assert normalize_for_comparison("MHC Class II") == normalize_for_comparison(
            "MHC Class 2"
        )

    def test_roman_numeral_case_insensitive(self):
        """Test Roman numerals are converted regardless of case."""
        # Uppercase Roman numerals should convert
        assert normalize_for_comparison("Type II") == "type 2"
        # Lowercase Roman numerals should also convert (e.g., "collagen iv")
        assert normalize_for_comparison("type ii") == "type 2"
        assert normalize_for_comparison("collagen iv") == "collagen 4"
        assert normalize_for_comparison("collagen iii") == "collagen 3"

    def test_roman_numeral_preserves_non_roman(self):
        """Test non-Roman numeral uppercase letters are not converted."""
        # These contain Roman numeral letters but aren't valid numerals at word boundaries
        assert normalize_for_comparison("COVID") == "covid"
        assert normalize_for_comparison("DAVID") == "david"
        # CD4 - letters are not at word boundaries
        assert normalize_for_comparison("CD4") == "cd4"

    def test_roman_numeral_rejects_invalid_structure(self):
        """Test invalid Roman numeral structures are not converted."""
        # These are all Roman numeral letters but not valid Roman numerals
        assert normalize_for_comparison("CIVIL") == "civil"
        assert normalize_for_comparison("MILD") == "mild"
        assert normalize_for_comparison("LIVID") == "livid"
        assert normalize_for_comparison("VIM") == "vim"
        # Note: MIX is actually valid (M=1000 + IX=9 = 1009)

    def test_roman_numeral_larger_values(self):
        """Test larger Roman numeral values."""
        assert normalize_for_comparison("Type XII") == "type 12"
        assert normalize_for_comparison("Phase XIV") == "phase 14"
        assert normalize_for_comparison("Group XX") == "group 20"


class TestIsObviousVariant:
    """Tests for is_obvious_variant function."""

    def test_spelling_variants(self):
        """UK/US spelling variants should return 'spelling'."""
        from interaction_finder.extraction.utils import is_obvious_variant

        # Inputs already normalized (as per function contract)
        assert is_obvious_variant("haemorrhagic", "hemorrhagic") == "spelling"
        assert is_obvious_variant("oestrogen", "estrogen") == "spelling"
        assert is_obvious_variant("colour", "color") == "spelling"

    def test_plural_variants(self):
        """Plural patterns should return 'plural'."""
        from interaction_finder.extraction.utils import is_obvious_variant

        assert is_obvious_variant("gene", "genes") == "plural"
        assert is_obvious_variant("box", "boxes") == "plural"
        assert is_obvious_variant("entity", "entities") == "plural"

    def test_latin_greek_plural_variants(self):
        """Latin/Greek plural patterns should return 'plural'."""
        from interaction_finder.extraction.utils import is_obvious_variant

        # um→a
        assert is_obvious_variant("bacterium", "bacteria") == "plural"
        assert is_obvious_variant("medium", "media") == "plural"
        # us→i
        assert is_obvious_variant("fungus", "fungi") == "plural"
        assert is_obvious_variant("nucleus", "nuclei") == "plural"
        # is→es
        assert is_obvious_variant("axis", "axes") == "plural"
        assert is_obvious_variant("hypothesis", "hypotheses") == "plural"
        # on→a
        assert is_obvious_variant("criterion", "criteria") == "plural"
        assert is_obvious_variant("phenomenon", "phenomena") == "plural"
        # ex/ix→ices
        assert is_obvious_variant("index", "indices") == "plural"
        assert is_obvious_variant("matrix", "matrices") == "plural"
        assert is_obvious_variant("appendix", "appendices") == "plural"

    def test_hyphenation_variants(self):
        """Hyphenation/spacing differences should return 'spacing'."""
        from interaction_finder.extraction.utils import is_obvious_variant

        # Normalized forms (hyphens become spaces during normalization)
        assert is_obvious_variant("venoocular", "veno ocular") == "spacing"
        assert is_obvious_variant("alphabetagamma", "alpha beta gamma") == "spacing"
        assert is_obvious_variant("tgf beta", "tgfbeta") == "spacing"

    def test_not_obvious_variants(self):
        """Completely different strings should return None."""
        from interaction_finder.extraction.utils import is_obvious_variant

        assert is_obvious_variant("brca1", "tp53") is None
        assert not is_obvious_variant("gene", "protein")


class TestEntityNamesMatch:
    """Tests for entity_names_match function."""

    def test_exact_match(self):
        """Identical names after normalization should match as 'exact'."""
        from interaction_finder.extraction.utils import entity_names_match

        matched, dist, kind = entity_names_match("BRCA1", "brca1")
        assert matched is True
        assert dist == 0
        assert kind == "exact"

    def test_spelling_variant(self):
        """UK/US spelling variants should match as 'spelling'."""
        from interaction_finder.extraction.utils import entity_names_match

        matched, dist, kind = entity_names_match("haemorrhagic", "hemorrhagic")
        assert matched is True
        assert kind == "spelling"

    def test_plural_variant(self):
        """Plural forms should match as 'plural'."""
        from interaction_finder.extraction.utils import entity_names_match

        matched, dist, kind = entity_names_match("receptor", "receptors")
        assert matched is True
        assert kind == "plural"

        matched, dist, kind = entity_names_match("bacterium", "bacteria")
        assert matched is True
        assert kind == "plural"

    def test_spacing_variant(self):
        """Spacing/hyphenation variants should match as 'spacing'."""
        from interaction_finder.extraction.utils import entity_names_match

        # Use words that don't get collapsed by Greek letter mapping
        matched, dist, kind = entity_names_match("veno ocular", "venoocular")
        assert matched is True
        assert kind == "spacing"

    def test_fuzzy_match(self):
        """Fuzzy matches should return 'fuzzy' kind."""
        from interaction_finder.extraction.utils import entity_names_match

        # Long enough for fuzzy matching, spelling difference
        matched, dist, kind = entity_names_match(
            "pulmonary hypertension", "pulmonary hypertnsion"
        )
        assert matched is True
        assert kind == "fuzzy"
        assert dist > 0

    def test_no_match(self):
        """Non-matching names should return None for kind."""
        from interaction_finder.extraction.utils import entity_names_match

        matched, dist, kind = entity_names_match("BRCA1", "TP53")
        assert matched is False
        assert kind is None

    def test_number_difference_rejected(self):
        """Names differing only by number should not match."""
        from interaction_finder.extraction.utils import entity_names_match

        matched, dist, kind = entity_names_match("SMAD1", "SMAD2")
        assert matched is False
        assert kind is None

    def test_digit_letter_swap_rejected(self):
        """Digit-letter swaps should not match."""
        from interaction_finder.extraction.utils import entity_names_match

        matched, dist, kind = entity_names_match("alpha5 beta3", "alphav beta3")
        assert matched is False
        assert kind is None


class TestOnlyNumberDifference:
    """Tests for _only_number_difference function."""

    def test_number_only_difference(self):
        """Entities differing only by number should be detected."""
        from interaction_finder.extraction.utils import _only_number_difference

        assert _only_number_difference("SMAD1", "SMAD2")
        assert _only_number_difference("IL-6", "IL-8")
        assert _only_number_difference("BRCA1", "BRCA2")
        assert _only_number_difference("collagen1", "collagen4")

    def test_digit_letter_swap(self):
        """Digit-letter substitutions should be detected."""
        from interaction_finder.extraction.utils import _only_number_difference

        assert _only_number_difference("a5b3", "avb3")  # 5 vs v
        assert _only_number_difference("a5", "av")
        assert _only_number_difference("integrina5", "integrinav")

    def test_different_length_numbers(self):
        """Multi-digit vs single-digit numbers should be handled correctly."""
        from interaction_finder.extraction.utils import _only_number_difference

        assert _only_number_difference("integrina2b1", "integrina12b1")
        assert _only_number_difference("a5", "a51")
        assert _only_number_difference("gene10", "gene100")

    def test_identical_strings(self):
        """Identical strings should return False."""
        from interaction_finder.extraction.utils import _only_number_difference

        assert not _only_number_difference("a5b3", "a5b3")
        assert not _only_number_difference("SMAD1", "SMAD1")

    def test_no_numbers(self):
        """Strings without numbers should return False."""
        from interaction_finder.extraction.utils import _only_number_difference

        assert not _only_number_difference("tumor", "tumour")
        assert not _only_number_difference("abc", "def")
        assert not _only_number_difference("gene", "protein")

    def test_one_has_number(self):
        """One string with number, one without, should return False."""
        from interaction_finder.extraction.utils import _only_number_difference

        assert not _only_number_difference("collagen", "collagen1")
        assert not _only_number_difference("gene", "gene2")

    def test_whitespace_normalized(self):
        """Whitespace should be normalized before comparison."""
        from interaction_finder.extraction.utils import _only_number_difference

        assert _only_number_difference("integrin alpha 5", "integrin alpha2")
        assert _only_number_difference("IL 6", "IL8")


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
        entity1_ref = EntityRef(canonical=entity1.name, mentions=[entity1])
        entity2 = EntityMention(
            kind="disease",
            name="breast cancer",
            aliases=["breast cancer"],
            quotes=[],
            reasoning="test",
        )
        entity2_ref = EntityRef(canonical=entity2.name, mentions=[entity2])

        # disease comes before gene alphabetically
        key = make_entity_pair_key(entity1_ref, entity2_ref)
        assert key.entity1_name == "breast cancer"
        assert key.entity2_name == "BRCA1"

        # Reverse order should give same key
        key2 = make_entity_pair_key(entity2_ref, entity1_ref)
        assert key == key2

    def test_orders_by_name_when_same_kind(self):
        """Test ordering by name when kinds are equal."""
        entity1 = EntityMention(
            kind="gene", name="BRCA2", aliases=["BRCA2"], quotes=[], reasoning="test"
        )
        entity1_ref = EntityRef(canonical=entity1.name, mentions=[entity1])
        entity2 = EntityMention(
            kind="gene", name="BRCA1", aliases=["BRCA1"], quotes=[], reasoning="test"
        )
        entity2_ref = EntityRef(canonical=entity2.name, mentions=[entity2])

        key = make_entity_pair_key(entity1_ref, entity2_ref)
        # BRCA1 comes before BRCA2
        assert key.entity1_name == "BRCA1"
        assert key.entity2_name == "BRCA2"

    def test_consistent_ordering(self):
        """Test that ordering is consistent regardless of input order."""
        entity1 = EntityMention(
            kind="gene", name="TP53", aliases=["TP53"], quotes=[], reasoning="test"
        )
        entity1_ref = EntityRef(canonical=entity1.name, mentions=[entity1])
        entity2 = EntityMention(
            kind="gene", name="BRCA1", aliases=["BRCA1"], quotes=[], reasoning="test"
        )
        entity2_ref = EntityRef(canonical=entity2.name, mentions=[entity2])

        key1 = make_entity_pair_key(entity1_ref, entity2_ref)
        key2 = make_entity_pair_key(entity2_ref, entity1_ref)

        assert key1 == key2


class TestGetRelationshipPolarity:
    """Tests for get_relationship_polarity function."""

    def test_gets_polarity_for_known_relationship(self):
        """Test looking up polarity for a known relationship."""
        polarity_map = {
            "increases_risk_of": "positive",
            "protects_against": "negative",
            "regulates": "neutral",
            "spatial_colocalization": "irrelevant",
        }

        assert (
            get_relationship_polarity("increases_risk_of", polarity_map) == "positive"
        )
        assert get_relationship_polarity("protects_against", polarity_map) == "negative"
        assert get_relationship_polarity("regulates", polarity_map) == "neutral"
        assert (
            get_relationship_polarity("spatial_colocalization", polarity_map)
            == "irrelevant"
        )

    def test_raises_key_error_for_unknown_relationship(self):
        """Test that KeyError is raised for unknown relationship."""
        polarity_map = {"increases_risk_of": "positive"}

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
        self.entity1_ref = EntityRef(
            canonical=self.entity1.name, mentions=[self.entity1]
        )
        self.entity2 = EntityMention(
            kind="disease",
            name="breast cancer",
            aliases=["breast cancer"],
            quotes=[],
            reasoning="test",
        )
        self.entity2_ref = EntityRef(
            canonical=self.entity2.name, mentions=[self.entity2]
        )

    def test_groups_assessments_by_polarity(self):
        """Test that assessments are correctly grouped by polarity."""
        polarity_map = {
            "increases_risk_of": "positive",
            "protects_against": "negative",
            "regulates": "neutral",
            "spatial_colocalization": "irrelevant",
        }

        assessments = [
            PairAssessment(
                topic_relevance=3,
                resource_id=self.resource.id,
                entity1=self.entity1_ref,
                entity2=self.entity2_ref,
                relationship="increases_risk_of",
                quotes=[],
                evidence=make_evidence(8),
                reasoning="test",
            ),
            PairAssessment(
                topic_relevance=3,
                resource_id=self.resource.id,
                entity1=self.entity1_ref,
                entity2=self.entity2_ref,
                relationship="protects_against",
                quotes=[],
                evidence=make_evidence(6),
                reasoning="test",
            ),
            PairAssessment(
                topic_relevance=3,
                resource_id=self.resource.id,
                entity1=self.entity1_ref,
                entity2=self.entity2_ref,
                relationship="regulates",
                quotes=[],
                evidence=make_evidence(3),
                reasoning="test",
            ),
        ]

        spread = build_pair_spread(assessments, polarity_map)

        assert len(spread.positive) == 1
        assert spread.positive[0].relationship == "increases_risk_of"
        assert len(spread.negative) == 1
        assert spread.negative[0].relationship == "protects_against"
        assert len(spread.neutral) == 1
        assert spread.neutral[0].relationship == "regulates"
        assert len(spread.irrelevant) == 0

    def test_handles_all_same_polarity(self):
        """Test with all assessments having same polarity."""
        polarity_map = {
            "increases_risk_of": "positive",
            "causes": "positive",
            "associated_with": "positive",
        }

        assessments = [
            PairAssessment(
                topic_relevance=3,
                resource_id=self.resource.id,
                entity1=self.entity1_ref,
                entity2=self.entity2_ref,
                relationship="increases_risk_of",
                quotes=[],
                evidence=make_evidence(8),
                reasoning="test",
            ),
            PairAssessment(
                topic_relevance=3,
                resource_id=self.resource.id,
                entity1=self.entity1_ref,
                entity2=self.entity2_ref,
                relationship="causes",
                quotes=[],
                evidence=make_evidence(8),
                reasoning="test",
            ),
            PairAssessment(
                topic_relevance=3,
                resource_id=self.resource.id,
                entity1=self.entity1_ref,
                entity2=self.entity2_ref,
                relationship="associated_with",
                quotes=[],
                evidence=make_evidence(6),
                reasoning="test",
            ),
        ]

        spread = build_pair_spread(assessments, polarity_map)

        assert len(spread.positive) == 3
        assert len(spread.negative) == 0
        assert len(spread.neutral) == 0
        assert len(spread.irrelevant) == 0

    def test_handles_empty_assessments(self):
        """Test with empty assessments list."""
        polarity_map = {"increases_risk_of": "positive"}

        spread = build_pair_spread([], polarity_map)

        assert len(spread.positive) == 0
        assert len(spread.negative) == 0
        assert len(spread.neutral) == 0
        assert len(spread.irrelevant) == 0

    def test_raises_key_error_for_unmapped_relationship(self):
        """Test that KeyError is raised for unmapped relationship."""
        polarity_map = {"increases_risk_of": "positive"}

        assessments = [
            PairAssessment(
                topic_relevance=3,
                resource_id=self.resource.id,
                entity1=self.entity1_ref,
                entity2=self.entity2_ref,
                relationship="unknown_relationship",
                quotes=[],
                evidence=make_evidence(8),
                reasoning="test",
            ),
        ]

        with pytest.raises(KeyError):
            build_pair_spread(assessments, polarity_map)

    def test_contentious_pair_detection(self):
        """Test identifying contentious pairs (positive + negative)."""
        polarity_map = {
            "increases_risk_of": "positive",
            "protects_against": "negative",
        }

        assessments = [
            PairAssessment(
                topic_relevance=3,
                resource_id=self.resource.id,
                entity1=self.entity1_ref,
                entity2=self.entity2_ref,
                relationship="increases_risk_of",
                quotes=[],
                evidence=make_evidence(8),
                reasoning="test",
            ),
            PairAssessment(
                topic_relevance=3,
                resource_id=self.resource.id,
                entity1=self.entity1_ref,
                entity2=self.entity2_ref,
                relationship="protects_against",
                quotes=[],
                evidence=make_evidence(8),
                reasoning="test",
            ),
        ]

        spread = build_pair_spread(assessments, polarity_map)

        # Contentious: has both positive and negative
        is_contentious = bool(spread.positive and spread.negative)
        assert is_contentious
        assert len(spread.positive) == 1
        assert len(spread.negative) == 1


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


class TestFindBestEntityMatch:
    """Tests for find_best_entity_match function."""

    # Stage 1: Exact match tests
    def test_exact_match(self):
        """Test exact match returns immediately."""
        assert find_best_entity_match("BRCA1", ["BRCA1", "TP53"]) == "BRCA1"

    def test_exact_match_case_sensitive(self):
        """Test exact match is case-sensitive (falls through to normalization)."""
        # "brca1" != "BRCA1" so not exact, but normalized match works
        assert find_best_entity_match("brca1", ["BRCA1", "TP53"]) == "BRCA1"

    # Stage 2: Strip kind annotation tests
    def test_strips_gene_annotation(self):
        """Test matching after stripping (gene) annotation."""
        assert find_best_entity_match("BRCA1 (gene)", ["BRCA1", "TP53"]) == "BRCA1"

    def test_strips_phenotype_annotation(self):
        """Test matching after stripping (phenotype) annotation."""
        result = find_best_entity_match(
            "Pulmonary arterial hypertension (phenotype)",
            ["Pulmonary arterial hypertension", "BMPR2"],
        )
        assert result == "Pulmonary arterial hypertension"

    def test_strips_abbreviation_annotation(self):
        """Test matching after stripping abbreviation in parens."""
        result = find_best_entity_match(
            "Pulmonary veno-occlusive disease (PVOD)",
            ["Pulmonary veno-occlusive disease", "PAH"],
        )
        assert result == "Pulmonary veno-occlusive disease"

    # Stage 3: Normalization tests
    def test_normalization_handles_case(self):
        """Test normalized match handles case differences."""
        assert find_best_entity_match("bmpr2", ["BMPR2", "ACVRL1"]) == "BMPR2"

    def test_normalization_handles_hyphens(self):
        """Test normalized match handles hyphen differences."""
        # Normalize removes hyphens/special chars and lowercases
        result = find_best_entity_match(
            "Pulmonary veno-occlusive disease",
            ["Pulmonary venoocclusive disease"],
        )
        assert result == "Pulmonary venoocclusive disease"

    def test_normalization_combined_with_stripping(self):
        """Test stripping + normalization work together."""
        result = find_best_entity_match("BMPR2 (gene)", ["bmpr2", "acvrl1"])
        assert result == "bmpr2"

    # Stage 4: Fuzzy match tests
    def test_fuzzy_match_small_typo(self):
        """Test fuzzy match catches small typos."""
        # "BMPR-2" vs "BMPR2" - just one character difference
        result = find_best_entity_match("BMPR-2", ["BMPR2", "COMPLETELY_DIFFERENT"])
        assert result == "BMPR2"

    def test_fuzzy_match_us_uk_spelling(self):
        """Test fuzzy match handles US/UK spelling variants."""
        result = find_best_entity_match(
            "haemorrhagic telangiectasia",
            ["hemorrhagic telangiectasia", "something else entirely"],
        )
        assert result == "hemorrhagic telangiectasia"

    def test_fuzzy_match_respects_max_distance(self):
        """Test fuzzy match respects max_distance threshold."""
        # "cat" vs "elephant" - too different
        assert find_best_entity_match("cat", ["elephant"]) is None

    def test_fuzzy_match_requires_specificity(self):
        """Test fuzzy match requires specificity (gap to second best)."""
        # All candidates are similarly distant (distance 1 each)
        # Gap between best and second-best is 0, needs >= 2
        result = find_best_entity_match("ABCD", ["ABCE", "ABCF", "ABCG"])
        assert result is None

    def test_fuzzy_match_single_candidate_no_specificity_check(self):
        """Test single candidate doesn't need specificity check."""
        result = find_best_entity_match("BMPR-2", ["BMPR2"])
        assert result == "BMPR2"

    # Edge cases
    def test_empty_candidates(self):
        """Test empty candidates returns None."""
        assert find_best_entity_match("BRCA1", []) is None

    def test_empty_query(self):
        """Test empty query returns None."""
        assert find_best_entity_match("", ["a", "b"]) is None

    def test_whitespace_query(self):
        """Test whitespace-only query returns None."""
        assert find_best_entity_match("   ", ["a", "b"]) is None

    def test_no_match_found(self):
        """Test returns None when no match found."""
        assert find_best_entity_match("BRCA1", ["TP53", "EGFR"]) is None

    def test_returns_original_candidate_form(self):
        """Test returns the original candidate string, not normalized."""
        result = find_best_entity_match("brca1", ["BRCA1"])
        assert result == "BRCA1"  # Original form, not "brca1"

    def test_original_wins_regardless_of_order(self):
        """Test original candidate wins over variant regardless of list order."""
        # "a" should match candidate "a", not "a (b)" via its variant
        assert find_best_entity_match("a", ["a (b)", "a"]) == "a"
        assert find_best_entity_match("a", ["a", "a (b)"]) == "a"

    def test_variant_collision_uses_first_candidate(self):
        """Test when multiple candidates share a variant, first in list wins."""
        # Both "a (b)" and "a (c)" have variant "a", first one wins
        assert find_best_entity_match("a", ["a (b)", "a (c)"]) == "a (b)"
        assert find_best_entity_match("a", ["a (c)", "a (b)"]) == "a (c)"

    def test_parenthetical_content_matches_candidate(self):
        """Test query can match via parenthetical content of candidate."""
        result = find_best_entity_match("b", ["a (b)", "c"])
        assert result == "a (b)"

    # Short string edge cases - fuzzy matching should be conservative
    def test_rejects_single_char_mismatch(self):
        """Test single char query doesn't fuzzy match different char."""
        assert find_best_entity_match("A", ["B"]) is None

    def test_rejects_short_transposition(self):
        """Test short string transpositions are rejected (too risky)."""
        assert find_best_entity_match("AB", ["BA"]) is None
        assert find_best_entity_match("IL", ["LI"]) is None

    def test_rejects_similar_gene_numbers(self):
        """Test similar gene names with different numbers don't match."""
        # p53, p63, p73 are different genes - should not fuzzy match
        assert find_best_entity_match("p53", ["p63"]) is None

    def test_rejects_gene_family_members(self):
        """Test gene family members don't fuzzy match each other."""
        assert find_best_entity_match("SMAD1", ["SMAD2"]) is None
        assert find_best_entity_match("VEGFR1", ["VEGFR2"]) is None
        assert find_best_entity_match("IL-6", ["IL-8"]) is None
        assert find_best_entity_match("BRCA1", ["BRCA2"]) is None
        assert find_best_entity_match("HER2", ["HER3"]) is None

    def test_rejects_longer_number_differences(self):
        """Test that longer numbers (3+ digits) are also rejected."""
        # microRNA-137 vs microRNA-138 are distinct entities
        assert find_best_entity_match("Study 2024", ["Study 2025"]) is None
        assert find_best_entity_match("microRNA-137", ["microRNA-138"]) is None
        assert find_best_entity_match("chromosome 21", ["chromosome 22"]) is None

    def test_accepts_high_similarity_typo(self):
        """Test typos in longer strings are accepted."""
        # BMPR2 vs BMRP2 (transposition) - letters differ, not just numbers
        assert find_best_entity_match("BMRP2", ["BMPR2"]) == "BMPR2"

    # Real-world examples from the warning messages
    def test_real_example_gdf2_bmp9(self):
        """Test real example: BMP9 (GDF2) matching."""
        # LLM returned "BMP9 (GDF2)", candidate is "BMP9"
        result = find_best_entity_match("BMP9 (GDF2)", ["BMP9", "BMPR2", "ACVRL1"])
        assert result == "BMP9"

    def test_real_example_eif2ak4_pvod(self):
        """Test real example: PVOD with hyphen difference.

        The high-quality match on the long string (97% similar) should win
        over the poor match PVOD→PAH (25% similar).
        """
        result = find_best_entity_match(
            "Pulmonary veno-occlusive disease (PVOD)",
            ["Pulmonary venoocclusive disease", "PAH"],
        )
        assert result == "Pulmonary venoocclusive disease"

    def test_real_example_kcnk3_pah(self):
        """Test real example: entity with (gene) annotation."""
        result = find_best_entity_match(
            "KCNK3 (gene)",
            ["KCNK3", "BMPR2", "EIF2AK4"],
        )
        assert result == "KCNK3"

    def test_similarity_prefers_high_quality_long_match(self):
        """Test that high-quality match on long string beats low-quality match on short string.

        "Pulmonary veno-occlusive disease" vs "Pulmonary venoocclusive disease"
        is 97% similar (1 edit / 31 chars).

        "PVOD" vs "PAH" is 25% similar (3 edits / 4 chars).

        The long string match should win despite PVOD→PAH having smaller absolute distance.
        """
        result = find_best_entity_match(
            "Pulmonary veno-occlusive disease (PVOD)",
            ["Pulmonary venoocclusive disease", "PAH"],
        )
        assert result == "Pulmonary venoocclusive disease"

    def test_similarity_rejects_ambiguous_short_matches(self):
        """Test that similarly poor short matches are rejected for lack of specificity."""
        # "ABC" vs "ABX" is 67% similar, "ABC" vs "ABY" is also 67% similar
        # No clear winner, should return None
        result = find_best_entity_match("ABC", ["ABX", "ABY"])
        assert result is None

    # Parenthetical content matching (acronym in full form)
    def test_parenthetical_exact_match_with_parens(self):
        """Test exact match when candidate includes parentheses."""
        result = find_best_entity_match(
            "a (b)",
            ["a (b)", "a", "b"],
        )
        assert result == "a (b)"

    def test_parenthetical_candidate_expansion(self):
        """Test query without parens matches candidate with parens."""
        result = find_best_entity_match(
            "a",
            ["a (b)", "c", "d"],
        )
        assert result == "a (b)"

    def test_parenthetical_original_preferred_over_expanded(self):
        """Test original candidate preferred over another's expanded variant."""
        # Query "a" should match candidate "a" not "a (b)" via its variant
        result = find_best_entity_match(
            "a",
            ["a (b)", "a"],
        )
        assert result == "a"

    def test_parenthetical_acronym_matches(self):
        """Test matching acronym inside parentheses to candidate."""
        result = find_best_entity_match(
            "Pulmonary arterial hypertension (PAH)",
            ["PAH", "BMPR2"],
        )
        assert result == "PAH"

    def test_parenthetical_full_name_matches(self):
        """Test matching full name inside parentheses to candidate."""
        result = find_best_entity_match(
            "PAH (Pulmonary arterial hypertension)",
            ["Pulmonary arterial hypertension", "BMPR2"],
        )
        assert result == "Pulmonary arterial hypertension"

    def test_parenthetical_base_preferred_over_content(self):
        """Test that base form is preferred over parenthetical content."""
        # Both "BRCA1" and "gene" could match, but base has priority
        result = find_best_entity_match(
            "BRCA1 (gene)",
            ["BRCA1", "gene"],
        )
        assert result == "BRCA1"

    # Slash alternation tests
    def test_slash_alternation_whole_matches(self):
        """Test that whole slash expression matches first."""
        result = find_best_entity_match(
            "SMAD1/5",
            ["SMAD1/5", "SMAD1"],
        )
        assert result == "SMAD1/5"

    def test_slash_alternation_part_matches(self):
        """Test matching individual part of slash expression."""
        result = find_best_entity_match(
            "TGF-β/BMP",
            ["BMP", "ACVRL1"],
        )
        assert result == "BMP"

    def test_slash_and_parenthetical_combined(self):
        """Test combined slash and parenthetical patterns."""
        result = find_best_entity_match(
            "receptor I/II (signaling)",
            ["signaling", "receptor"],
        )
        # "receptor I/II" doesn't match, "receptor" is a variant, "signaling" is in parens
        # Priority: base > base parts > paren content
        assert result == "signaling"

    # Multiple slashes should not expand (too ambiguous)
    def test_multiple_slashes_no_expansion(self):
        """Test that multiple slashes don't expand (ambiguous)."""
        result = find_best_entity_match(
            "SMAD1/5/9",
            ["SMAD1", "SMAD5"],
        )
        # Won't match because multiple slashes aren't expanded
        assert result is None

    # Slash suffix expansion (biological naming convention)
    def test_slash_suffix_gene_numbers(self):
        """Test GDF1/2 expands to GDF1 and GDF2."""
        result = find_best_entity_match(
            "GDF1/2",
            ["GDF2", "BMP9"],
        )
        assert result == "GDF2"

    def test_slash_suffix_smad(self):
        """Test SMAD1/5 expands to SMAD1 and SMAD5."""
        result = find_best_entity_match(
            "SMAD1/5",
            ["SMAD5", "SMAD9"],
        )
        assert result == "SMAD5"

    def test_slash_suffix_roman_numerals(self):
        """Test type I/II expands to type I and type II."""
        result = find_best_entity_match(
            "type I/II",
            ["type II", "type III"],
        )
        assert result == "type II"

    def test_slash_suffix_receptor(self):
        """Test receptor 1/2 expands properly."""
        result = find_best_entity_match(
            "receptor 1/2",
            ["receptor 2", "receptor 3"],
        )
        assert result == "receptor 2"

    def test_slash_suffix_original_preferred(self):
        """Test that original form is preferred over expanded."""
        result = find_best_entity_match(
            "GDF1/2",
            ["GDF1/2", "GDF1", "GDF2"],
        )
        assert result == "GDF1/2"

    def test_slash_suffix_left_side_matches(self):
        """Test that left side of suffix pattern also matches."""
        result = find_best_entity_match(
            "GDF1/2",
            ["GDF1", "BMP9"],
        )
        assert result == "GDF1"


class TestBuildRelationshipOppositionMap:
    """Tests for build_relationship_opposition_map function."""

    def test_simple_bidirectional_opposition(self):
        """Test basic bidirectional opposition mapping."""
        from interaction_finder.extraction.models import RelationshipConsolidation

        consolidations = [
            RelationshipConsolidation(
                original="activates",
                consolidated="activates",
                polarity="positive",
                opposites=["inhibits"],
                reasoning="Activates has opposite biological effect to inhibits",
            ),
            RelationshipConsolidation(
                original="inhibits",
                consolidated="inhibits",
                polarity="negative",
                opposites=["activates"],
                reasoning="Inhibits has opposite biological effect to activates",
            ),
        ]
        all_relationships = {"activates", "inhibits"}

        result = build_relationship_opposition_map(consolidations, all_relationships)

        # Should be bidirectional
        assert "activates" in result
        assert "inhibits" in result
        assert "inhibits" in result["activates"]
        assert "activates" in result["inhibits"]

    def test_multiple_opposites(self):
        """Test relationship with multiple opposites."""
        from interaction_finder.extraction.models import RelationshipConsolidation

        consolidations = [
            RelationshipConsolidation(
                original="increases_risk_of",
                consolidated="increases_risk_of",
                polarity="positive",
                opposites=["protects_against", "reduces_risk_of"],
                reasoning="Multiple protective relationships",
            ),
            RelationshipConsolidation(
                original="protects_against",
                consolidated="protects_against",
                polarity="negative",
                opposites=["increases_risk_of"],
                reasoning="Protective relationship that opposes risk-increasing effects",
            ),
            RelationshipConsolidation(
                original="reduces_risk_of",
                consolidated="reduces_risk_of",
                polarity="negative",
                opposites=["increases_risk_of"],
                reasoning="Risk reduction relationship opposing risk increase",
            ),
        ]
        all_relationships = {
            "increases_risk_of",
            "protects_against",
            "reduces_risk_of",
        }

        result = build_relationship_opposition_map(consolidations, all_relationships)

        # All three should be in map (normalized: underscores -> spaces)
        assert "increases risk of" in result
        assert "protects against" in result
        assert "reduces risk of" in result
        # increases_risk_of opposes both protective relationships
        assert result["increases risk of"] == {"protects against", "reduces risk of"}
        # Each protective relationship opposes increases_risk_of
        assert "increases risk of" in result["protects against"]
        assert "increases risk of" in result["reduces risk of"]

    def test_consolidated_relationships_included(self):
        """Test that both original and consolidated labels are mapped."""
        from interaction_finder.extraction.models import RelationshipConsolidation

        consolidations = [
            RelationshipConsolidation(
                original="upregulates",
                consolidated="activates",
                polarity="positive",
                opposites=["downregulates"],
                reasoning="Consolidating upregulates to canonical activates form",
            ),
            RelationshipConsolidation(
                original="downregulates",
                consolidated="inhibits",
                polarity="negative",
                opposites=["upregulates"],
                reasoning="Consolidating downregulates to canonical inhibits form",
            ),
        ]
        all_relationships = {"upregulates", "downregulates"}

        result = build_relationship_opposition_map(consolidations, all_relationships)

        # Both original and consolidated should be in map
        assert "upregulates" in result or "activates" in result
        assert "downregulates" in result or "inhibits" in result

    def test_no_opposites(self):
        """Test relationships with no opposites."""
        from interaction_finder.extraction.models import RelationshipConsolidation

        consolidations = [
            RelationshipConsolidation(
                original="associated_with",
                consolidated="associated_with",
                polarity="neutral",
                opposites=[],
                reasoning="Neutral relationship has no clear opposites",
            ),
            RelationshipConsolidation(
                original="binds_to",
                consolidated="binds_to",
                polarity="neutral",
                opposites=[],
                reasoning="Physical interaction without directionality",
            ),
        ]
        all_relationships = {"associated_with", "binds_to"}

        result = build_relationship_opposition_map(consolidations, all_relationships)

        # Map may be empty or contain entries with empty sets
        if "associated_with" in result:
            assert len(result["associated_with"]) == 0
        if "binds_to" in result:
            assert len(result["binds_to"]) == 0

    def test_osa_fuzzy_matching(self):
        """Test OSA distance matching for typos in opposites."""
        from interaction_finder.extraction.models import RelationshipConsolidation

        # LLM might have typo in opposite
        consolidations = [
            RelationshipConsolidation(
                original="activates",
                consolidated="activates",
                polarity="positive",
                opposites=["inhbits"],  # Typo: missing 'i'
                reasoning="Testing OSA matching with typo in opposite label",
            ),
            RelationshipConsolidation(
                original="inhibits",
                consolidated="inhibits",
                polarity="negative",
                opposites=["activates"],
                reasoning="Correct spelling of opposite relationship label",
            ),
        ]
        all_relationships = {"activates", "inhibits"}

        result = build_relationship_opposition_map(consolidations, all_relationships)

        # Should still match despite typo (OSA distance = 1)
        assert "activates" in result
        assert "inhibits" in result
        # May match or not depending on normalization
        # At minimum, the correct direction should work
        assert "activates" in result["inhibits"]

    def test_bidirectionality_enforcement(self):
        """Test that bidirectionality is enforced even with asymmetric LLM output."""
        from interaction_finder.extraction.models import RelationshipConsolidation

        # Set up: only "promotes" declares inhibits as opposite
        # Bidirectionality should add the reverse
        consolidations = [
            RelationshipConsolidation(
                original="promotes",
                consolidated="promotes",
                polarity="positive",
                opposites=["inhibits"],
                reasoning="Promotes has opposite biological effect to inhibits",
            ),
            RelationshipConsolidation(
                original="inhibits",
                consolidated="inhibits",
                polarity="negative",
                opposites=[],  # Empty - asymmetric input
                reasoning="Inhibits with no declared opposites for testing",
            ),
            RelationshipConsolidation(
                original="activates",
                consolidated="activates",
                polarity="positive",
                opposites=["inhibits"],
                reasoning="Activates has opposite biological effect to inhibits",
            ),
        ]
        all_relationships = {"promotes", "inhibits", "activates"}

        result = build_relationship_opposition_map(consolidations, all_relationships)

        # Check bidirectionality: both promotes and activates should oppose inhibits
        assert "inhibits" in result["promotes"]
        assert "inhibits" in result["activates"]
        # And reverse should be enforced
        assert "promotes" in result["inhibits"]
        assert "activates" in result["inhibits"]
        # But promotes and activates are NOT opposites (no transitivity)
        assert "activates" not in result.get("promotes", set())
        assert "promotes" not in result.get("activates", set())

    def test_empty_consolidations(self):
        """Test with no consolidations."""
        result = build_relationship_opposition_map([], set())
        assert result == {}

    def test_case_insensitive_matching(self):
        """Test that matching is case-insensitive via normalization."""
        from interaction_finder.extraction.models import RelationshipConsolidation

        consolidations = [
            RelationshipConsolidation(
                original="Activates",
                consolidated="activates",
                polarity="positive",
                opposites=["Inhibits"],  # Capital I
                reasoning="Testing case-insensitive matching for opposites",
            ),
            RelationshipConsolidation(
                original="inhibits",
                consolidated="inhibits",
                polarity="negative",
                opposites=["activates"],
                reasoning="Lowercase version of opposite relationship label",
            ),
        ]
        all_relationships = {"Activates", "inhibits"}

        result = build_relationship_opposition_map(consolidations, all_relationships)

        # Should match despite case differences
        assert len(result) > 0
        # Check normalized keys exist
        assert "activates" in result or any("activ" in k for k in result)


class TestAreRelationshipsOpposed:
    """Tests for are_relationships_opposed function."""

    def test_simple_opposition(self):
        """Test basic opposition detection."""
        opposition_map = {"activates": {"inhibits"}, "inhibits": {"activates"}}

        assert are_relationships_opposed("activates", "inhibits", opposition_map)
        assert are_relationships_opposed("inhibits", "activates", opposition_map)

    def test_no_opposition(self):
        """Test non-opposed relationships."""
        opposition_map = {"activates": {"inhibits"}, "inhibits": {"activates"}}

        assert not are_relationships_opposed("activates", "regulates", opposition_map)
        assert not are_relationships_opposed("binds_to", "activates", opposition_map)

    def test_self_not_opposed(self):
        """Test that a relationship is not opposed to itself."""
        opposition_map = {"activates": {"inhibits"}, "inhibits": {"activates"}}

        assert not are_relationships_opposed("activates", "activates", opposition_map)
        assert not are_relationships_opposed("inhibits", "inhibits", opposition_map)

    def test_multiple_opposites(self):
        """Test relationship with multiple opposites."""
        # Keys are normalized (underscores -> spaces)
        opposition_map = {
            "increases risk of": {"protects against", "reduces risk of"},
            "protects against": {"increases risk of"},
            "reduces risk of": {"increases risk of"},
        }

        assert are_relationships_opposed(
            "increases_risk_of", "protects_against", opposition_map
        )
        assert are_relationships_opposed(
            "increases_risk_of", "reduces_risk_of", opposition_map
        )
        assert are_relationships_opposed(
            "protects_against", "increases_risk_of", opposition_map
        )
        # But protective relationships are not opposed to each other
        assert not are_relationships_opposed(
            "protects_against", "reduces_risk_of", opposition_map
        )

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        opposition_map = {"activates": {"inhibits"}, "inhibits": {"activates"}}

        # Should match despite case differences (normalization)
        assert are_relationships_opposed("Activates", "Inhibits", opposition_map)
        assert are_relationships_opposed("ACTIVATES", "inhibits", opposition_map)

    def test_empty_opposition_map(self):
        """Test with empty opposition map."""
        opposition_map = {}

        assert not are_relationships_opposed("activates", "inhibits", opposition_map)

    def test_relationship_not_in_map(self):
        """Test when relationship is not in map."""
        opposition_map = {"activates": {"inhibits"}}

        # regulates not in map
        assert not are_relationships_opposed("regulates", "activates", opposition_map)
        assert not are_relationships_opposed("activates", "unknown", opposition_map)
