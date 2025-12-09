"""Tests to verify that entity names preserve original casing.

The system should:
1. Use normalization only for comparison/grouping
2. Store entities with their original casing from LLM output
3. Keep canonical names readable (e.g., "BRCA1" not "brca1")
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai.usage import RunUsage
from pydantic_graph import GraphRunContext

from interaction_finder.extraction.document_pipeline import analyze_document
from interaction_finder.extraction.models import (
    DocumentAnalysisOut,
    EntityInfo,
    PaperQualityAssessment,
    QualityDimensionScore,
)
from interaction_finder.extraction.utils import build_permitted_pairs
from interaction_finder.resources import Resource, ResourceId, ResourcePool


def _make_quality_assessment() -> PaperQualityAssessment:
    """Create a default paper quality assessment for tests."""
    dim = QualityDimensionScore(score=2, justification="Standard quality for testing.")
    return PaperQualityAssessment(
        method_clarity=dim,
        data_provenance=dim,
        statistical_rigour=dim,
        internal_consistency=dim,
        plausibility=dim,
        reproducibility_signals=dim,
        integrity_indicators=dim,
    )


def create_mock_agent_with_override(run_return_value):
    """Create a mock agent with working rename_agent() support."""
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=run_return_value)
    mock_agent._name = "mock_agent"
    return mock_agent


@pytest.fixture
def mock_deps():
    """Create mock dependencies with a resource pool."""
    deps = MagicMock()
    deps.config = MagicMock()
    deps.logger = MagicMock()
    deps.resource_pool = ResourcePool()
    # Add a test resource
    deps.resource_pool.add(
        url="https://example.com/doc1",
        title="Test Document",
        document_text="BRCA1 is a tumor suppressor gene. The BRCA1 protein plays a role in DNA repair.",
        chunks=[(0, 50), (50, 100)],
    )
    return deps


class TestEntityCasing:
    """Test that entity names preserve original casing."""

    @pytest.mark.asyncio
    async def test_preserves_uppercase_gene_names(self, mock_deps):
        """Entity names should preserve uppercase (e.g., BRCA1, not brca1)."""
        # Mock LLM response with uppercase gene name
        mock_result = MagicMock()
        mock_result.output = DocumentAnalysisOut(
            paper_quality=_make_quality_assessment(),
            entities=[
                EntityInfo(
                    kind="gene",
                    name="BRCA1",
                    aliases=["BRCA1", "BRCA-1"],
                    quotes=["BRCA1 is a tumor suppressor gene"],
                    reasoning="BRCA1 is a well-known breast cancer susceptibility gene",
                )
            ],
        )
        mock_agent = create_mock_agent_with_override(mock_result)

        # Patch the agent getter
        import interaction_finder.extraction.document_pipeline as pipeline_module

        original_getter = pipeline_module.get_document_analysis_agent
        pipeline_module.get_document_analysis_agent = lambda config: mock_agent

        try:
            resource = list(mock_deps.resource_pool.resources)[0]
            entities, quality, _, _ = await analyze_document(
                resource,
                "breast cancer genetics",
                ["gene"],
                mock_deps.config,
                mock_deps,
            )

            # Should have exactly one entity
            assert len(entities) == 1

            # Entity should be accessible by its canonical name "BRCA1"
            assert "BRCA1" in entities
            entity = entities["BRCA1"]

            # Canonical name should be uppercase
            assert entity.name == "BRCA1"
            assert entity.name != "brca1"

            # Quality assessment should be returned
            assert quality is not None

        finally:
            pipeline_module.get_document_analysis_agent = original_getter

    @pytest.mark.asyncio
    async def test_preserves_mixed_case_disease_names(self, mock_deps):
        """Disease names with mixed case should be preserved."""
        # Mock LLM response
        mock_result = MagicMock()
        mock_result.output = DocumentAnalysisOut(
            paper_quality=_make_quality_assessment(),
            entities=[
                EntityInfo(
                    kind="disease",
                    name="Alzheimer's disease",
                    aliases=["Alzheimer's disease", "AD"],
                    quotes=["BRCA1 is a tumor suppressor gene"],  # Use valid quote
                    reasoning="Neurodegenerative disease",
                )
            ],
        )
        mock_agent = create_mock_agent_with_override(mock_result)

        import interaction_finder.extraction.document_pipeline as pipeline_module

        original_getter = pipeline_module.get_document_analysis_agent
        pipeline_module.get_document_analysis_agent = lambda config: mock_agent

        try:
            resource = list(mock_deps.resource_pool.resources)[0]
            entities, _, _, _ = await analyze_document(
                resource,
                "cardiovascular diseases",
                ["disease"],
                mock_deps.config,
                mock_deps,
            )

            # Should preserve mixed case
            assert "Alzheimer's disease" in entities
            entity = entities["Alzheimer's disease"]
            assert entity.name == "Alzheimer's disease"
            assert entity.name != "alzheimer's disease"

        finally:
            pipeline_module.get_document_analysis_agent = original_getter

    @pytest.mark.asyncio
    async def test_groups_case_variants_by_normalization(self, mock_deps):
        """Entities differing only in case should be grouped together."""
        # Mock LLM response
        mock_result = MagicMock()
        mock_result.output = DocumentAnalysisOut(
            paper_quality=_make_quality_assessment(),
            entities=[
                EntityInfo(
                    kind="gene",
                    name="BRCA1",
                    aliases=["BRCA1"],
                    quotes=["BRCA1 is a tumor suppressor gene"],
                    reasoning="First mention of this gene in the document",
                ),
                EntityInfo(
                    kind="gene",
                    name="brca1",  # Different case
                    aliases=["brca1"],
                    quotes=["The BRCA1 protein plays a role"],
                    reasoning="Second mention of this gene with different case",
                ),
                EntityInfo(
                    kind="gene",
                    name="Brca1",  # Yet another case
                    aliases=["Brca1"],
                    quotes=["BRCA1 is a tumor suppressor gene"],
                    reasoning="Third mention with mixed case variant",
                ),
            ],
        )
        mock_agent = create_mock_agent_with_override(mock_result)

        import interaction_finder.extraction.document_pipeline as pipeline_module

        original_getter = pipeline_module.get_document_analysis_agent
        pipeline_module.get_document_analysis_agent = lambda config: mock_agent

        try:
            resource = list(mock_deps.resource_pool.resources)[0]
            entities, _, _, _ = await analyze_document(
                resource,
                "genetics",
                ["gene"],
                mock_deps.config,
                mock_deps,
            )

            # Should have exactly one entity (all variants merged)
            assert len(entities) == 1

            # Should use first variant's casing as canonical
            assert "BRCA1" in entities
            entity = entities["BRCA1"]
            assert entity.name == "BRCA1"

            # Should have merged quotes from all three mentions
            assert len(entity.quotes) == 3

            # Should have merged reasoning from all variants
            assert "First mention" in entity.reasoning
            assert "Second mention" in entity.reasoning
            assert "Third mention" in entity.reasoning

        finally:
            pipeline_module.get_document_analysis_agent = original_getter

    @pytest.mark.asyncio
    async def test_preserves_casing_in_acronyms(self, mock_deps):
        """Acronyms should preserve their specific casing."""
        # Mock LLM response
        mock_result = MagicMock()
        mock_result.output = DocumentAnalysisOut(
            paper_quality=_make_quality_assessment(),
            entities=[
                EntityInfo(
                    kind="disease",
                    name="PAH",
                    aliases=["PAH", "pulmonary arterial hypertension"],
                    quotes=["BRCA1 is a tumor suppressor gene"],  # Use valid quote
                    reasoning="Pulmonary arterial hypertension acronym",
                )
            ],
        )
        mock_agent = create_mock_agent_with_override(mock_result)

        import interaction_finder.extraction.document_pipeline as pipeline_module

        original_getter = pipeline_module.get_document_analysis_agent
        pipeline_module.get_document_analysis_agent = lambda config: mock_agent

        try:
            resource = list(mock_deps.resource_pool.resources)[0]
            entities, _, _, _ = await analyze_document(
                resource,
                "hypertension",
                ["disease"],
                mock_deps.config,
                mock_deps,
            )

            # Should preserve uppercase acronym
            assert "PAH" in entities
            entity = entities["PAH"]
            assert entity.name == "PAH"
            assert entity.name != "pah"

        finally:
            pipeline_module.get_document_analysis_agent = original_getter
