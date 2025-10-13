"""
PydanticAI agents for extracting biological interactions from scientific literature.

This module provides AI agents that can analyze paper markdown content and extract
genes mentioned as being associated with specific diseases using structured output.
"""

from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from .models import Term
from .settings import IfetcherConfig


class GeneDiseaseResult(BaseModel):
    """Result of gene-disease extraction from a paper."""

    disease: str = Field(description="The disease that was searched for")
    genes: List[str] = Field(description="List of genes associated with the disease")


def create_gene_disease_agent(config: IfetcherConfig) -> Agent[None, GeneDiseaseResult]:
    """
    Create a PydanticAI agent for extracting gene-disease associations.

    Args:
        config: Configuration object containing agent settings

    Returns:
        Configured PydanticAI agent for gene-disease extraction
    """

    # Get agent configuration with fallbacks
    agent_config = config.agents.get("gene_disease_extractor", config.AgentSpec())
    default_config = config.agents.get("_", config.AgentSpec())

    # Determine LLM model to use
    model = agent_config.llm or default_config.llm or "openai:gpt-4o"

    # Prefer configuration-provided prompt/instruction to avoid hardcoding domain specifics
    if agent_config.prompt:
        system_prompt = agent_config.prompt
    else:
        # Build a generic-but-sane default with optional expertise/instruction
        expertise = agent_config.expertise or "scientific literature analysis"
        extra_instruction = (
            f"\n\nAdditional instruction: {agent_config.instruction}"
            if agent_config.instruction
            else ""
        )

        system_prompt = (
            "You are an expert in "
            + expertise
            + ".\n\nYour task is to analyze scientific paper content and identify ALL genes mentioned as being associated with a provided target disease.\n\n"
            "IMPORTANT GUIDELINES:\n"
            "1. Only extract genes explicitly associated with the target disease\n"
            "2. Return clean gene symbols/names (e.g., BRCA1, TP53, EGFR)\n"
            "3. Focus on the target disease; ignore genes for other diseases unless also relevant\n"
            "4. Consider mutations, expression changes, pathway involvement, therapeutic targets, biomarkers\n"
            "5. Use exact names from the text when clear and unambiguous\n\n"
            "Return a simple list of gene names associated with the target disease."
            + extra_instruction
        )

    agent = Agent(model, output_type=GeneDiseaseResult, system_prompt=system_prompt)

    return agent


def extract_genes_for_disease(
    markdown_content: str, target_disease: str, config: IfetcherConfig
) -> GeneDiseaseResult:
    """
    Extract all genes mentioned as being associated with a given disease from paper markdown content.

    Args:
        markdown_content: The markdown content of the scientific paper
        target_disease: The specific disease to look for gene associations with
        config: Configuration object containing agent settings

    Returns:
        GeneDiseaseResult containing the disease and associated genes

    Raises:
        RuntimeError: If the AI agent fails to process the content
    """

    # Create the specialized agent
    agent = create_gene_disease_agent(config)

    # Prepare the analysis prompt
    user_prompt = f"""
Please analyze the following scientific paper content and extract ALL genes mentioned as being associated with {target_disease}.

TARGET DISEASE: {target_disease}

PAPER CONTENT:
{markdown_content[:50000]}  # Limit to first 50k characters to avoid token limits

Please identify all genes mentioned in association with {target_disease}.
Return clean gene names/symbols only.
Focus specifically on {target_disease} and avoid genes associated with other diseases unless they are also relevant to {target_disease}.
"""

    try:
        # Run the agent
        result = agent.run_sync(user_prompt)
        return result.output

    except Exception as e:
        raise RuntimeError(f"Failed to extract gene-disease associations: {str(e)}")


def create_gene_extraction_terms(result: GeneDiseaseResult) -> List[Term]:
    """
    Convert extraction results to Term objects for compatibility with the existing system.

    Args:
        result: Results from gene-disease extraction

    Returns:
        List of Term objects representing the extracted genes
    """

    terms = []

    for gene_name in result.genes:
        # Create a term for the gene
        attributes = {"disease": result.disease}

        term = Term(kind="gene", name=gene_name, attributes=attributes)

        terms.append(term)

    return terms
