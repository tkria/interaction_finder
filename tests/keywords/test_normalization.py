"""Tests for term normalization in keywords module."""

from interaction_finder.keywords.normalization import (
    normalize_term_for_deduplication,
    strip_parenthetical_abbreviations,
    strip_common_suffixes,
    lemmatize_term,
)


class TestStripParentheticalAbbreviations:
    """Test parenthetical abbreviation stripping."""

    def test_strips_simple_abbreviation(self):
        """Test stripping simple uppercase abbreviation."""
        assert (
            strip_parenthetical_abbreviations("Pulmonary arterial hypertension (PAH)")
            == "Pulmonary arterial hypertension"
        )

    def test_strips_multiple_abbreviations(self):
        """Test stripping multiple abbreviations."""
        result = strip_parenthetical_abbreviations(
            "Single nucleotide polymorphism (SNP) variants (V)"
        )
        assert result == "Single nucleotide polymorphism variants"

    def test_strips_abbreviation_with_numbers(self):
        """Test abbreviation with numbers."""
        assert (
            strip_parenthetical_abbreviations("Interleukin 6 (IL6)") == "Interleukin 6"
        )

    def test_strips_abbreviation_with_greek(self):
        """Test abbreviation with Greek letters."""
        assert (
            strip_parenthetical_abbreviations("Transforming Growth Factor-β (TGF-β)")
            == "Transforming Growth Factor-β"
        )

    def test_preserves_lowercase_parentheses(self):
        """Test that lowercase content in parentheses is preserved."""
        # Our pattern only matches capital letter start
        text = "something (with lowercase)"
        assert strip_parenthetical_abbreviations(text) == text

    def test_handles_no_abbreviations(self):
        """Test term without abbreviations passes through."""
        text = "genetic risk factors"
        assert strip_parenthetical_abbreviations(text) == text

    def test_handles_empty_string(self):
        """Test empty string."""
        assert strip_parenthetical_abbreviations("") == ""


class TestStripCommonSuffixes:
    """Test common suffix stripping."""

    def test_strips_pathway(self):
        """Test stripping ' pathway' suffix."""
        assert strip_common_suffixes("BMP pathway") == "BMP"

    def test_strips_signaling_pathway(self):
        """Test stripping ' signaling pathway' suffix."""
        assert strip_common_suffixes("TGF-β signaling pathway") == "TGF-β"

    def test_strips_signaling(self):
        """Test stripping ' signaling' suffix."""
        assert strip_common_suffixes("BMP signaling") == "BMP"

    def test_strips_longest_match_first(self):
        """Test that longest suffix is matched first."""
        # Should match " signaling pathway" not just " pathway"
        assert strip_common_suffixes("BMP signaling pathway") == "BMP"

    def test_preserves_without_suffix(self):
        """Test term without suffix passes through."""
        text = "genetic mutations"
        assert strip_common_suffixes(text) == text

    def test_case_insensitive(self):
        """Test suffix matching is case-insensitive."""
        assert strip_common_suffixes("BMP PATHWAY") == "BMP"
        assert strip_common_suffixes("BMP Pathway") == "BMP"

    def test_handles_empty_string(self):
        """Test empty string."""
        assert strip_common_suffixes("") == ""


class TestLemmatizeTerm:
    """Test lemmatization."""

    def test_lemmatizes_simple_plural(self):
        """Test simple plural lemmatization."""
        result = lemmatize_term("factors")
        assert result == "factor"

    def test_lemmatizes_phrase(self):
        """Test phrase lemmatization."""
        result = lemmatize_term("genetic risk factors")
        assert result == "genetic risk factor"

    def test_lemmatizes_ies_ending(self):
        """Test -ies ending."""
        result = lemmatize_term("studies")
        assert result == "study"

    def test_lemmatizes_es_ending(self):
        """Test -es ending."""
        result = lemmatize_term("diseases")
        # Could be "disease" depending on lemmatizer
        assert result in ["disease", "diseases"]

    def test_preserves_singular(self):
        """Test singular terms unchanged."""
        result = lemmatize_term("hypertension")
        assert result == "hypertension"

    def test_preserves_short_words(self):
        """Test short words preserved."""
        result = lemmatize_term("is")
        assert result in ["is", "be"]  # spaCy might lemmatize "is" to "be"

    def test_handles_empty_string(self):
        """Test empty string."""
        assert lemmatize_term("") == ""


class TestNormalizeTermForDeduplication:
    """Test full normalization pipeline."""

    def test_normalizes_case(self):
        """Test case normalization."""
        assert normalize_term_for_deduplication(
            "Pulmonary Arterial Hypertension"
        ) == normalize_term_for_deduplication("pulmonary arterial hypertension")

    def test_strips_abbreviation_and_normalizes(self):
        """Test abbreviation stripping integrated."""
        result = normalize_term_for_deduplication(
            "Pulmonary arterial hypertension (PAH)"
        )
        assert result == "pulmonary arterial hypertension"

    def test_handles_plurals(self):
        """Test plural normalization."""
        singular = normalize_term_for_deduplication("genetic risk factor")
        plural = normalize_term_for_deduplication("genetic risk factors")
        assert singular == plural

    def test_strips_pathway_suffix(self):
        """Test pathway suffix stripping."""
        with_suffix = normalize_term_for_deduplication("BMP signaling pathway")
        without_suffix = normalize_term_for_deduplication("BMP")
        assert with_suffix == without_suffix == "bmp"

    def test_handles_greek_letters(self):
        """Test Greek letter normalization."""
        result = normalize_term_for_deduplication("TGF-β pathway")
        # Greek β → "beta", hyphens → spaces, "pathway" stripped
        assert result == "tgf beta"

    def test_handles_hyphens(self):
        """Test hyphen normalization."""
        result = normalize_term_for_deduplication("gene-environment interactions")
        # Hyphens become spaces
        assert "gene" in result and "environment" in result

    def test_complex_example_from_sample(self):
        """Test real examples from sample data."""
        # These should all normalize to same key
        terms = [
            "BMP signaling",
            "BMP signaling pathway",
            "BMP Signaling Pathway",
        ]
        normalized = [normalize_term_for_deduplication(t) for t in terms]
        assert len(set(normalized)) == 1, f"Expected all same, got {normalized}"

    def test_pah_variations(self):
        """Test PAH abbreviation variations."""
        terms = [
            "Pulmonary arterial hypertension",
            "Pulmonary arterial hypertension (PAH)",
            "pulmonary arterial hypertension",
        ]
        normalized = [normalize_term_for_deduplication(t) for t in terms]
        assert len(set(normalized)) == 1

    def test_genotype_phenotype_variations(self):
        """Test genotype-phenotype correlation variations."""
        singular = normalize_term_for_deduplication("Genotype-phenotype correlation")
        plural = normalize_term_for_deduplication("Genotype-phenotype correlations")
        assert singular == plural

    def test_inflammation_case_normalization(self):
        """Test inflammation case normalization."""
        terms = [
            "Inflammation",
            "inflammation",
            "INFLAMMATION",
        ]
        # After normalization, all should be same
        normalized = [normalize_term_for_deduplication(t) for t in terms]
        assert len(set(normalized)) == 1
        assert normalized[0] == "inflammation"

    def test_tgf_beta_variations(self):
        """Test TGF-β pathway variations."""
        terms = [
            "TGF-β pathway",
            "TGF-β signaling pathway",
            "TGF-β/BMP signaling pathway",
        ]
        normalized = [normalize_term_for_deduplication(t) for t in terms]
        # First two should be same, third adds BMP
        assert normalized[0] == normalized[1]
        assert "tgf beta" in normalized[0]
        assert "bmp" in normalized[2]

    def test_preserves_important_distinctions(self):
        """Test that genuinely different terms remain different."""
        term1 = normalize_term_for_deduplication("BMPR2 mutations")
        term2 = normalize_term_for_deduplication("BMPR2 gene")
        assert term1 != term2

    def test_empty_string(self):
        """Test empty string handling."""
        assert normalize_term_for_deduplication("") == ""

    def test_caching_works(self):
        """Test that caching doesn't break functionality."""
        # Call twice with same input
        result1 = normalize_term_for_deduplication("Test Term")
        result2 = normalize_term_for_deduplication("Test Term")
        assert result1 == result2


class TestNormalizationEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_word_term(self):
        """Test single word terms."""
        assert normalize_term_for_deduplication("Hypertension") == "hypertension"

    def test_very_long_term(self):
        """Test very long terms don't break."""
        long_term = " ".join(["word"] * 50)
        result = normalize_term_for_deduplication(long_term)
        assert result  # Should produce something

    def test_special_characters(self):
        """Test terms with special characters."""
        result = normalize_term_for_deduplication("p53α/β")
        # Greek letters normalized, slash becomes space
        assert "alpha" in result or "beta" in result

    def test_numbers_preserved(self):
        """Test that numbers are preserved."""
        result = normalize_term_for_deduplication("IL6 signaling")
        assert "6" in result or "il6" in result

    def test_whitespace_normalization(self):
        """Test excessive whitespace is normalized."""
        result = normalize_term_for_deduplication("genetic   risk    factors")
        # Multiple spaces should collapse
        assert "  " not in result


class TestIntegrationWithSampleData:
    """Integration tests using real sample data patterns."""

    def test_sample_duplicates_found(self):
        """Test that known duplicates from sample are detected."""
        # Test cases from the actual sample data
        test_cases = [
            # Case variations
            (["Precision Medicine", "Precision medicine", "precision medicine"], 1),
            # Plural variations
            (
                [
                    "Genetic risk factor",
                    "Genetic risk factors",
                    "genetic risk factors",
                ],
                1,
            ),
            # Pathway variations
            (["BMP signaling", "BMP signaling pathway", "BMP Signaling"], 1),
            # Abbreviation variations
            (
                [
                    "Pulmonary arterial hypertension",
                    "Pulmonary arterial hypertension (PAH)",
                ],
                1,
            ),
            # SNP variations
            (
                [
                    "Single nucleotide polymorphism (SNP)",
                    "Single nucleotide polymorphisms (SNPs)",
                ],
                1,
            ),
            # Genotype-phenotype
            (
                ["Genotype-phenotype correlation", "Genotype-phenotype correlations"],
                1,
            ),
            # Histone modifications
            (["histone modification", "histone modifications"], 1),
        ]

        for terms, expected_unique in test_cases:
            normalized = [normalize_term_for_deduplication(t) for t in terms]
            unique_count = len(set(normalized))
            assert unique_count == expected_unique, (
                f"Expected {expected_unique} unique for {terms}, "
                f"got {unique_count}: {set(normalized)}"
            )

    def test_sample_distinctions_preserved(self):
        """Test that genuinely different terms remain distinct."""
        # These should NOT be deduplicated
        test_cases = [
            ["BMPR2", "BMPR2 gene", "BMPR2 mutations", "BMPR2 genetic mutations"],
            ["Endothelial cell proliferation", "Endothelial cell function"],
            ["Genetic testing", "Genetic counseling"],
        ]

        for terms in test_cases:
            normalized = [normalize_term_for_deduplication(t) for t in terms]
            unique_count = len(set(normalized))
            # Should have multiple unique keys
            assert unique_count > 1, (
                f"Terms should remain distinct: {terms} → {set(normalized)}"
            )
