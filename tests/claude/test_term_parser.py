"""
Tests for the term_parser module.
"""

import pytest
from interaction_finder.term_parser import (
    parse_term_line,
    _parse_attributes,
    parse_terms_from_lines,
)
from interaction_finder.models import Term


class TestParseTermLine:
    """Test the parse_term_line function."""

    def test_simple_term_name_only(self):
        """Test parsing a simple term name without comments."""
        result = parse_term_line("BRCA1")

        assert result is not None
        assert result.name == "BRCA1"
        assert result.kind is None
        assert result.attributes == {}

    def test_term_with_whitespace(self):
        """Test parsing a term with leading/trailing whitespace."""
        result = parse_term_line("  BRCA1  ")

        assert result is not None
        assert result.name == "BRCA1"
        assert result.kind is None
        assert result.attributes == {}

    def test_term_with_simple_comment(self):
        """Test parsing a term with a simple comment."""
        result = parse_term_line("BRCA1 # breast cancer gene")

        assert result is not None
        assert result.name == "BRCA1"
        assert result.kind is None
        assert result.attributes == {}

    def test_term_with_kind_attribute(self):
        """Test parsing a term with kind attribute."""
        result = parse_term_line("BRCA1 # &kind=gene")

        assert result is not None
        assert result.name == "BRCA1"
        assert result.kind == "gene"
        assert result.attributes == {}

    def test_term_with_multiple_attributes(self):
        """Test parsing a term with multiple attributes."""
        result = parse_term_line("BRCA1 # &kind=gene &disease=cancer &chromosome=17")

        assert result is not None
        assert result.name == "BRCA1"
        assert result.kind == "gene"
        assert result.attributes == {"disease": "cancer", "chromosome": "17"}

    def test_term_with_comma_separated_attributes(self):
        """Test parsing a term with comma-separated attributes."""
        result = parse_term_line("BRCA1 # &kind=gene, &disease=cancer, &chromosome=17")

        assert result is not None
        assert result.name == "BRCA1"
        assert result.kind == "gene"
        assert result.attributes == {"disease": "cancer", "chromosome": "17"}

    def test_term_with_mixed_separators(self):
        """Test parsing a term with mixed separators (commas and spaces)."""
        result = parse_term_line("BRCA1 # &kind=gene, &disease=cancer &chromosome=17")

        assert result is not None
        assert result.name == "BRCA1"
        assert result.kind == "gene"
        assert result.attributes == {"disease": "cancer", "chromosome": "17"}

    def test_term_with_multi_word_values(self):
        """Test parsing a term with multi-word attribute values."""
        result = parse_term_line(
            "BRCA1 # &kind=gene &disease=breast cancer &location=chromosome 17"
        )

        assert result is not None
        assert result.name == "BRCA1"
        assert result.kind == "gene"
        assert result.attributes == {
            "disease": "breast cancer",
            "location": "chromosome 17",
        }

    def test_term_with_comment_and_attributes(self):
        """Test parsing a term with both comment and attributes."""
        result = parse_term_line("BRCA1 # important gene &kind=gene &disease=cancer")

        assert result is not None
        assert result.name == "BRCA1"
        assert result.kind == "gene"
        assert result.attributes == {"disease": "cancer"}

    def test_empty_line(self):
        """Test parsing an empty line."""
        result = parse_term_line("")
        assert result is None

    def test_whitespace_only_line(self):
        """Test parsing a line with only whitespace."""
        result = parse_term_line("   \t   ")
        assert result is None

    def test_comment_only_line(self):
        """Test parsing a line that starts with #."""
        result = parse_term_line("# This is just a comment")
        assert result is None

    def test_line_with_only_comment_after_hash(self):
        """Test parsing a line where term name is empty but there's a comment."""
        result = parse_term_line("# just a comment")
        assert result is None

    def test_term_with_empty_comment(self):
        """Test parsing a term with empty comment section."""
        result = parse_term_line("BRCA1 #")

        assert result is not None
        assert result.name == "BRCA1"
        assert result.kind is None
        assert result.attributes == {}

    def test_term_with_malformed_attributes(self):
        """Test parsing a term with malformed attributes (no equals sign)."""
        result = parse_term_line("BRCA1 # &kind gene &disease")

        assert result is not None
        assert result.name == "BRCA1"
        assert result.kind is None
        assert result.attributes == {}

    def test_term_with_empty_attribute_values(self):
        """Test parsing a term with empty attribute values."""
        result = parse_term_line("BRCA1 # &kind= &disease=cancer")

        assert result is not None
        assert result.name == "BRCA1"
        assert result.kind is None  # Empty kind should not be set
        assert result.attributes == {"disease": "cancer"}

    def test_term_with_special_characters_in_name(self):
        """Test parsing a term with special characters in the name."""
        result = parse_term_line("BRCA1/2 # &kind=gene")

        assert result is not None
        assert result.name == "BRCA1/2"
        assert result.kind == "gene"

    def test_term_with_underscore_in_attribute_name(self):
        """Test parsing a term with underscore in attribute name."""
        result = parse_term_line("BRCA1 # &gene_type=tumor_suppressor")

        assert result is not None
        assert result.name == "BRCA1"
        assert result.kind is None
        assert result.attributes == {"gene_type": "tumor_suppressor"}


class TestParseAttributes:
    """Test the _parse_attributes helper function."""

    def test_single_attribute(self):
        """Test parsing a single attribute."""
        result = _parse_attributes("&kind=gene")
        assert result == {"kind": "gene"}

    def test_multiple_attributes_space_separated(self):
        """Test parsing multiple space-separated attributes."""
        result = _parse_attributes("&kind=gene &disease=cancer")
        assert result == {"kind": "gene", "disease": "cancer"}

    def test_multiple_attributes_comma_separated(self):
        """Test parsing multiple comma-separated attributes."""
        result = _parse_attributes("&kind=gene, &disease=cancer")
        assert result == {"kind": "gene", "disease": "cancer"}

    def test_multiple_attributes_mixed_separators(self):
        """Test parsing multiple attributes with mixed separators."""
        result = _parse_attributes("&kind=gene, &disease=cancer &location=chromosome")
        assert result == {"kind": "gene", "disease": "cancer", "location": "chromosome"}

    def test_multi_word_values(self):
        """Test parsing attributes with multi-word values."""
        result = _parse_attributes("&disease=breast cancer &location=chromosome 17")
        assert result == {"disease": "breast cancer", "location": "chromosome 17"}

    def test_empty_string(self):
        """Test parsing an empty string."""
        result = _parse_attributes("")
        assert result == {}

    def test_no_attributes(self):
        """Test parsing text with no attributes."""
        result = _parse_attributes("just some text")
        assert result == {}

    def test_malformed_attributes(self):
        """Test parsing malformed attributes (no equals sign)."""
        result = _parse_attributes("&kind gene &disease")
        assert result == {}

    def test_empty_attribute_values(self):
        """Test parsing attributes with empty values."""
        result = _parse_attributes("&kind= &disease=cancer")
        assert result == {"disease": "cancer"}

    def test_attribute_with_trailing_comma(self):
        """Test parsing attributes where last attribute has trailing comma."""
        result = _parse_attributes("&kind=gene, &disease=cancer,")
        assert result == {"kind": "gene", "disease": "cancer"}

    def test_attributes_with_extra_whitespace(self):
        """Test parsing attributes with extra whitespace."""
        result = _parse_attributes("  &kind=gene  ,  &disease=cancer  ")
        assert result == {"kind": "gene", "disease": "cancer"}

    def test_mixed_content_with_attributes(self):
        """Test parsing text that has both comments and attributes."""
        result = _parse_attributes(
            "important gene &kind=gene more text &disease=cancer"
        )
        assert result == {"kind": "gene more text", "disease": "cancer"}

    def test_attribute_names_with_underscores(self):
        """Test parsing attribute names with underscores."""
        result = _parse_attributes(
            "&gene_type=tumor_suppressor &protein_function=DNA_repair"
        )
        assert result == {
            "gene_type": "tumor_suppressor",
            "protein_function": "DNA_repair",
        }

    def test_complex_values_with_special_chars(self):
        """Test parsing attribute values with special characters."""
        result = _parse_attributes("&location=17q21.31 &alias=BRCA1/BRCAI")
        assert result == {"location": "17q21.31", "alias": "BRCA1/BRCAI"}


class TestParseTermsFromLines:
    """Test the parse_terms_from_lines function."""

    def test_single_line(self):
        """Test parsing a single line."""
        lines = ["BRCA1 # &kind=gene"]
        result = parse_terms_from_lines(lines)

        assert len(result) == 1
        assert result[0].name == "BRCA1"
        assert result[0].kind == "gene"

    def test_multiple_lines(self):
        """Test parsing multiple lines."""
        lines = [
            "BRCA1 # &kind=gene &disease=cancer",
            "TP53 # &kind=gene &function=tumor_suppressor",
            "EGFR # &kind=gene &location=chromosome_7",
        ]
        result = parse_terms_from_lines(lines)

        assert len(result) == 3
        assert result[0].name == "BRCA1"
        assert result[0].kind == "gene"
        assert result[0].attributes["disease"] == "cancer"

        assert result[1].name == "TP53"
        assert result[1].kind == "gene"
        assert result[1].attributes["function"] == "tumor_suppressor"

        assert result[2].name == "EGFR"
        assert result[2].kind == "gene"
        assert result[2].attributes["location"] == "chromosome_7"

    def test_mixed_valid_and_invalid_lines(self):
        """Test parsing a mix of valid and invalid lines."""
        lines = [
            "BRCA1 # &kind=gene",
            "",  # empty line
            "# just a comment",
            "TP53 # &kind=gene",
            "   ",  # whitespace only
            "EGFR",  # no attributes
        ]
        result = parse_terms_from_lines(lines)

        assert len(result) == 3
        assert result[0].name == "BRCA1"
        assert result[1].name == "TP53"
        assert result[2].name == "EGFR"

    def test_empty_list(self):
        """Test parsing an empty list of lines."""
        result = parse_terms_from_lines([])
        assert result == []

    def test_all_invalid_lines(self):
        """Test parsing lines that are all invalid."""
        lines = ["", "# comment only", "   ", "# another comment"]
        result = parse_terms_from_lines(lines)
        assert result == []

    def test_complex_example(self):
        """Test parsing a complex example with various line types."""
        lines = [
            "# Gene list for breast cancer study",
            "",
            "BRCA1 # tumor suppressor gene &kind=gene &disease=breast cancer &chromosome=17",
            "BRCA2 # &kind=gene, &disease=breast cancer, &chromosome=13",
            "",
            "# Oncogenes",
            "MYC # oncogene &kind=gene &function=transcription factor",
            "ERBB2 # &kind=gene &alias=HER2 &disease=breast cancer",
            "",
            "# Tumor suppressor",
            "TP53 # guardian of the genome &kind=gene &function=tumor suppressor &location=17p13.1",
        ]
        result = parse_terms_from_lines(lines)

        assert len(result) == 5

        # Check BRCA1
        brca1 = result[0]
        assert brca1.name == "BRCA1"
        assert brca1.kind == "gene"
        assert brca1.attributes["disease"] == "breast cancer"
        assert brca1.attributes["chromosome"] == "17"

        # Check BRCA2
        brca2 = result[1]
        assert brca2.name == "BRCA2"
        assert brca2.kind == "gene"
        assert brca2.attributes["disease"] == "breast cancer"
        assert brca2.attributes["chromosome"] == "13"

        # Check MYC
        myc = result[2]
        assert myc.name == "MYC"
        assert myc.kind == "gene"
        assert myc.attributes["function"] == "transcription factor"

        # Check ERBB2
        erbb2 = result[3]
        assert erbb2.name == "ERBB2"
        assert erbb2.kind == "gene"
        assert erbb2.attributes["alias"] == "HER2"
        assert erbb2.attributes["disease"] == "breast cancer"

        # Check TP53
        tp53 = result[4]
        assert tp53.name == "TP53"
        assert tp53.kind == "gene"
        assert tp53.attributes["function"] == "tumor suppressor"
        assert tp53.attributes["location"] == "17p13.1"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_term_name_with_hash_symbol(self):
        """Test term names that contain hash symbols."""
        # The first # splits the line, so "C#_gene # &kind=gene" becomes name="C"
        result = parse_term_line("C#_gene # &kind=gene")

        assert result is not None
        assert result.name == "C"
        assert result.kind == "gene"

    def test_attribute_values_with_equals_signs(self):
        """Test attribute values that contain equals signs."""
        result = parse_term_line("equation # &formula=E=mc2 &kind=physics")

        assert result is not None
        assert result.name == "equation"
        assert result.kind == "physics"
        assert (
            result.attributes["formula"] == "E=mc2"
        )  # Full value after first = is kept

    def test_very_long_line(self):
        """Test parsing a very long line."""
        long_name = "A" * 1000
        long_value = "B" * 1000
        line = f"{long_name} # &kind=gene &description={long_value}"

        result = parse_term_line(line)

        assert result is not None
        assert result.name == long_name
        assert result.kind == "gene"
        assert result.attributes["description"] == long_value

    def test_unicode_characters(self):
        """Test parsing lines with unicode characters."""
        result = parse_term_line("α-gene # &kind=gene &description=Greek letter α")

        assert result is not None
        assert result.name == "α-gene"
        assert result.kind == "gene"
        assert result.attributes["description"] == "Greek letter α"

    def test_multiple_hash_symbols(self):
        """Test lines with multiple hash symbols."""
        result = parse_term_line("gene1 # comment with # hash &kind=gene # more text")

        assert result is not None
        assert result.name == "gene1"
        assert result.kind == "gene # more text"


if __name__ == "__main__":
    pytest.main([__file__])
