"""
Tests for core data models.

Tests cover:
- Term model creation and validation
- Attribute handling and serialization
- String representations and equality
"""

import pytest
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from interaction_finder.models import Term


class TestTermModel:
    """Test the Term data model."""

    def test_term_creation_basic(self):
        """Test basic term creation."""
        term = Term(name="BRCA1", kind="gene")

        assert term.name == "BRCA1"
        assert term.kind == "gene"
        assert term.attributes == {}

    def test_term_creation_with_attributes(self):
        """Test term creation with attributes."""
        attributes = {
            "symbol": "BRCA1",
            "full_name": "BRCA1 DNA Repair Associated",
            "location": "17q21.31",
        }
        term = Term(name="BRCA1", kind="gene", attributes=attributes)

        assert term.name == "BRCA1"
        assert term.kind == "gene"
        assert term.attributes == attributes
        assert term.attributes["symbol"] == "BRCA1"
        assert term.attributes["full_name"] == "BRCA1 DNA Repair Associated"
        assert term.attributes["location"] == "17q21.31"

    def test_term_equality(self):
        """Test term equality comparison."""
        term1 = Term(name="BRCA1", kind="gene")
        term2 = Term(name="BRCA1", kind="gene")
        term3 = Term(name="TP53", kind="gene")
        term4 = Term(name="BRCA1", kind="protein")

        assert term1 == term2
        assert term1 != term3  # Different name
        assert term1 != term4  # Different kind

    def test_term_equality_with_attributes(self):
        """Test term equality with attributes."""
        attr1 = {"symbol": "BRCA1"}
        attr2 = {"symbol": "BRCA1"}
        attr3 = {"symbol": "BRCA2"}

        term1 = Term(name="BRCA1", kind="gene", attributes=attr1)
        term2 = Term(name="BRCA1", kind="gene", attributes=attr2)
        term3 = Term(name="BRCA1", kind="gene", attributes=attr3)
        term4 = Term(name="BRCA1", kind="gene")  # No attributes

        assert term1 == term2  # Same attributes
        assert term1 != term3  # Different attributes
        assert term1 != term4  # One has attributes, one doesn't

    def test_term_string_representation(self):
        """Test term string representation."""
        term = Term(name="BRCA1", kind="gene")

        assert str(term) == "BRCA1"
        assert repr(term) == "Term(name='BRCA1', kind='gene', attributes={})"

    def test_term_string_representation_with_attributes(self):
        """Test term string representation with attributes."""
        attributes = {"symbol": "BRCA1", "type": "tumor_suppressor"}
        term = Term(name="BRCA1", kind="gene", attributes=attributes)

        assert str(term) == "BRCA1"
        # repr should include attributes
        repr_str = repr(term)
        assert "Term(name='BRCA1'" in repr_str
        assert "kind='gene'" in repr_str
        assert "attributes=" in repr_str

    def test_term_hash(self):
        """Test that terms can be used in sets and as dict keys."""
        term1 = Term(name="BRCA1", kind="gene")
        term2 = Term(name="BRCA1", kind="gene")
        term3 = Term(name="TP53", kind="gene")

        # Should be hashable
        term_set = {term1, term2, term3}
        assert len(term_set) == 2  # term1 and term2 are equal

        # Should work as dict keys
        term_dict = {term1: "value1", term3: "value2"}
        assert term_dict[term2] == "value1"  # term2 equals term1

    def test_term_attribute_access(self):
        """Test accessing term attributes."""
        attributes = {
            "symbol": "BRCA1",
            "chromosome": "17",
            "start": 43044295,
            "end": 43170245,
        }
        term = Term(name="BRCA1", kind="gene", attributes=attributes)

        # Attributes should be accessible
        assert term.attributes["symbol"] == "BRCA1"
        assert term.attributes["chromosome"] == "17"
        assert term.attributes["start"] == 43044295
        assert term.attributes["end"] == 43170245

    def test_term_empty_attributes(self):
        """Test term with empty attributes dict."""
        term = Term(name="BRCA1", kind="gene", attributes={})

        assert term.attributes == {}
        assert len(term.attributes) == 0

    def test_term_none_attributes(self):
        """Test term with None attributes (should default to empty dict)."""
        # This tests the default value behavior
        term = Term(name="BRCA1", kind="gene")

        assert term.attributes == {}
        assert isinstance(term.attributes, dict)

    def test_term_mutable_attributes(self):
        """Test that attributes can be modified after creation."""
        term = Term(name="BRCA1", kind="gene")

        # Should be able to add attributes
        term.attributes["symbol"] = "BRCA1"
        term.attributes["full_name"] = "BRCA1 DNA Repair Associated"

        assert term.attributes["symbol"] == "BRCA1"
        assert term.attributes["full_name"] == "BRCA1 DNA Repair Associated"
        assert len(term.attributes) == 2

    def test_term_various_kinds(self):
        """Test terms with different kinds."""
        gene = Term(name="BRCA1", kind="gene")
        disease = Term(name="Breast Cancer", kind="disease")
        protein = Term(name="BRCA1 protein", kind="protein")
        drug = Term(name="Tamoxifen", kind="drug")

        assert gene.kind == "gene"
        assert disease.kind == "disease"
        assert protein.kind == "protein"
        assert drug.kind == "drug"

        # All should be different due to different kinds or names
        all_terms = {gene, disease, protein, drug}
        assert len(all_terms) == 4

    def test_term_various_attribute_types(self):
        """Test terms with different attribute value types."""
        attributes = {
            "string_attr": "text_value",
            "int_attr": 12345,
            "float_attr": 3.14159,
            "bool_attr": True,
            "list_attr": ["item1", "item2", "item3"],
            "dict_attr": {"nested_key": "nested_value"},
        }
        term = Term(name="Test", kind="test", attributes=attributes)

        assert term.attributes["string_attr"] == "text_value"
        assert term.attributes["int_attr"] == 12345
        assert term.attributes["float_attr"] == 3.14159
        assert term.attributes["bool_attr"] is True
        assert term.attributes["list_attr"] == ["item1", "item2", "item3"]
        assert term.attributes["dict_attr"] == {"nested_key": "nested_value"}

    def test_term_biological_examples(self):
        """Test terms representing biological entities."""
        # Gene
        brca1 = Term(
            name="BRCA1",
            kind="gene",
            attributes={
                "symbol": "BRCA1",
                "full_name": "BRCA1 DNA Repair Associated",
                "chromosome": "17",
                "function": "tumor suppressor",
            },
        )

        # Disease
        breast_cancer = Term(
            name="Breast Cancer",
            kind="disease",
            attributes={
                "icd10": "C50",
                "category": "neoplasm",
                "affected_organ": "breast",
            },
        )

        # Protein
        p53 = Term(
            name="p53",
            kind="protein",
            attributes={
                "gene": "TP53",
                "molecular_weight": 53000,
                "cellular_location": "nucleus",
            },
        )

        assert brca1.name == "BRCA1"
        assert brca1.kind == "gene"
        assert brca1.attributes["function"] == "tumor suppressor"

        assert breast_cancer.name == "Breast Cancer"
        assert breast_cancer.kind == "disease"
        assert breast_cancer.attributes["icd10"] == "C50"

        assert p53.name == "p53"
        assert p53.kind == "protein"
        assert p53.attributes["molecular_weight"] == 53000
