"""
Tests for the agents module.
"""

import pytest
from unittest.mock import Mock, patch

from interaction_finder.agents import (
    GeneDiseaseResult,
    create_gene_disease_agent,
    extract_genes_for_disease,
    create_gene_extraction_terms,
)
from interaction_finder.settings import IfetcherConfig
from interaction_finder.models import Term


class TestGeneDiseaseModels:
    """Test the Pydantic models for gene-disease extraction."""

    def test_gene_disease_result_valid(self):
        """Test creating a valid GeneDiseaseResult."""
        result = GeneDiseaseResult(
            disease="breast cancer", genes=["BRCA1", "BRCA2", "TP53"]
        )

        assert result.disease == "breast cancer"
        assert len(result.genes) == 3
        assert "BRCA1" in result.genes
        assert "BRCA2" in result.genes
        assert "TP53" in result.genes

    def test_gene_disease_result_empty_genes(self):
        """Test creating a result with no genes."""
        result = GeneDiseaseResult(disease="rare disease", genes=[])

        assert result.disease == "rare disease"
        assert len(result.genes) == 0


class TestTermConversion:
    """Test conversion to Term objects."""

    def test_create_gene_extraction_terms(self):
        """Test conversion of extraction results to Term objects."""
        result = GeneDiseaseResult(disease="breast cancer", genes=["BRCA1", "BRCA2"])

        terms = create_gene_extraction_terms(result)

        assert len(terms) == 2

        # Check first term
        term1 = terms[0]
        assert isinstance(term1, Term)
        assert term1.kind == "gene"
        assert term1.name == "BRCA1"
        assert term1.attributes["disease"] == "breast cancer"

        # Check second term
        term2 = terms[1]
        assert isinstance(term2, Term)
        assert term2.kind == "gene"
        assert term2.name == "BRCA2"
        assert term2.attributes["disease"] == "breast cancer"

    def test_create_gene_extraction_terms_empty(self):
        """Test conversion with no genes."""
        result = GeneDiseaseResult(disease="test disease", genes=[])

        terms = create_gene_extraction_terms(result)

        assert len(terms) == 0


class TestAgentCreation:
    """Test agent creation and configuration."""

    @patch("interaction_finder.agents.Agent")
    def test_create_gene_disease_agent(self, mock_agent_class):
        """Test creating a gene-disease extraction agent."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        config_data = {"agents": {"gene_disease_extractor": {"llm": "openai:gpt-4o"}}}

        config = IfetcherConfig.model_validate(config_data)
        agent = create_gene_disease_agent(config)

        # Verify agent was created
        assert agent is mock_agent
        mock_agent_class.assert_called_once()

    @patch("interaction_finder.agents.Agent")
    def test_create_gene_disease_agent_default_config(self, mock_agent_class):
        """Test creating agent with default configuration."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        config_data = {"agents": {"_": {"llm": "openai:gpt-3.5-turbo"}}}

        config = IfetcherConfig.model_validate(config_data)
        agent = create_gene_disease_agent(config)

        assert agent is mock_agent


class TestIntegration:
    """Integration tests that require mocking the AI agent."""

    @patch("interaction_finder.agents.Agent")
    def test_extract_genes_for_disease_success(self, mock_agent_class):
        """Test successful gene extraction with mocked agent."""
        # Mock the agent and its response
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        # Create a mock result
        mock_result = Mock()
        mock_result.output = GeneDiseaseResult(
            disease="breast cancer", genes=["BRCA1", "BRCA2", "TP53"]
        )

        mock_agent.run_sync.return_value = mock_result

        # Test the extraction
        config_data = {"agents": {"_": {"llm": "openai:gpt-4o"}}}

        config = IfetcherConfig.model_validate(config_data)

        markdown_content = """
        # Test Paper
        BRCA1 and BRCA2 mutations are strongly associated with breast cancer.
        TP53 is also frequently mutated in breast cancer patients.
        """

        result = extract_genes_for_disease(
            markdown_content=markdown_content,
            target_disease="breast cancer",
            config=config,
        )

        assert result.disease == "breast cancer"
        assert len(result.genes) == 3
        assert "BRCA1" in result.genes
        assert "BRCA2" in result.genes
        assert "TP53" in result.genes

    @patch("interaction_finder.agents.Agent")
    def test_extract_genes_for_disease_no_genes(self, mock_agent_class):
        """Test extraction when no genes are found."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        mock_result = Mock()
        mock_result.output = GeneDiseaseResult(disease="rare disease", genes=[])

        mock_agent.run_sync.return_value = mock_result

        config_data = {"agents": {"_": {"llm": "openai:gpt-4o"}}}
        config = IfetcherConfig.model_validate(config_data)

        result = extract_genes_for_disease(
            markdown_content="This paper discusses rare disease but mentions no specific genes.",
            target_disease="rare disease",
            config=config,
        )

        assert result.disease == "rare disease"
        assert len(result.genes) == 0

    @patch("interaction_finder.agents.Agent")
    def test_extract_genes_for_disease_agent_failure(self, mock_agent_class):
        """Test handling of agent failures."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        # Simulate agent failure
        mock_agent.run_sync.side_effect = Exception("AI model error")

        config_data = {"agents": {"_": {"llm": "openai:gpt-4o"}}}
        config = IfetcherConfig.model_validate(config_data)

        with pytest.raises(
            RuntimeError, match="Failed to extract gene-disease associations"
        ):
            extract_genes_for_disease(
                markdown_content="test content", target_disease="cancer", config=config
            )

    @patch("interaction_finder.agents.Agent")
    def test_extract_genes_large_content(self, mock_agent_class):
        """Test that large content is truncated properly."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        mock_result = Mock()
        mock_result.output = GeneDiseaseResult(
            disease="diabetes", genes=["INS", "INSR"]
        )

        mock_agent.run_sync.return_value = mock_result

        config_data = {"agents": {"_": {"llm": "openai:gpt-4o"}}}
        config = IfetcherConfig.model_validate(config_data)

        # Create content longer than 50k characters
        large_content = "A" * 60000

        result = extract_genes_for_disease(
            markdown_content=large_content, target_disease="diabetes", config=config
        )

        # Verify the agent was called with truncated content
        mock_agent.run_sync.assert_called_once()
        call_args = mock_agent.run_sync.call_args[0][0]

        # The prompt should contain truncated content (50k chars max)
        assert len(call_args) < len(large_content) + 1000  # Allow for prompt overhead

        assert result.disease == "diabetes"
        assert "INS" in result.genes
        assert "INSR" in result.genes


if __name__ == "__main__":
    pytest.main([__file__])
