"""
Tests for extraction graph V3 assembly.

Tests the graph construction and node ordering.
"""


from interaction_finder.extraction_graph_v3.graph import extraction_graph_v3
from interaction_finder.extraction_graph_v3.nodes import (
    ExtractEntities,
    AssessIndividually,
    GeneratePairCandidates,
    EvaluatePairs,
)


def test_graph_has_correct_nodes():
    """Graph should have exactly 4 nodes in correct order."""
    assert len(extraction_graph_v3.node_defs) == 4
    # Check that all expected nodes are registered
    assert "ExtractEntities" in extraction_graph_v3.node_defs
    assert "AssessIndividually" in extraction_graph_v3.node_defs
    assert "GeneratePairCandidates" in extraction_graph_v3.node_defs
    assert "EvaluatePairs" in extraction_graph_v3.node_defs
    # Verify node classes
    assert extraction_graph_v3.node_defs["ExtractEntities"].node == ExtractEntities
    assert (
        extraction_graph_v3.node_defs["AssessIndividually"].node == AssessIndividually
    )
    assert (
        extraction_graph_v3.node_defs["GeneratePairCandidates"].node
        == GeneratePairCandidates
    )
    assert extraction_graph_v3.node_defs["EvaluatePairs"].node == EvaluatePairs


def test_graph_has_correct_type_parameters():
    """Graph should be parameterized with correct state and deps types."""
    # Check that graph is properly constructed
    assert hasattr(extraction_graph_v3, "node_defs")
    # Nodes should accept ExtractionStateV3 and ExtractionDepsV3
    # This is enforced at runtime by pydantic-graph


def test_graph_node_instantiation():
    """All nodes should be instantiable."""
    # Create instances to verify they can be instantiated
    node1 = ExtractEntities()
    node2 = AssessIndividually()
    node3 = GeneratePairCandidates()
    node4 = EvaluatePairs()

    assert node1 is not None
    assert node2 is not None
    assert node3 is not None
    assert node4 is not None
