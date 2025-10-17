"""
Shared fixtures for extraction_v2 tests.

Provides common test setup and utilities for testing the DirectExtractor
and related components.
"""

import pytest
from typing import Dict, List

from pydantic_ai import models
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_graph import GraphRunContext

from src.interaction_finder.resources import ResourcePool
from src.interaction_finder.settings import IfetcherConfig
from src.interaction_finder.fetcher import PageFetcher
from src.interaction_finder.extraction_graph_v2.deps import ExtractionDeps
from src.interaction_finder.extraction_graph_v2.state import ExtractionState
from .fixtures import create_test_resource_pool, BRCA1_DOCUMENT, TEST_URLS

# Prevent accidental real LLM calls during all tests
models.ALLOW_MODEL_REQUESTS = False


@pytest.fixture(scope="session")
def prevent_model_requests():
    """Ensure no real model requests are made during testing."""
    original_setting = models.ALLOW_MODEL_REQUESTS
    models.ALLOW_MODEL_REQUESTS = False
    yield
    models.ALLOW_MODEL_REQUESTS = original_setting


@pytest.fixture
def sample_gene_document():
    """Sample document with gene and disease mentions for testing."""
    return """
    BRCA1 is a tumor suppressor gene that plays a critical role in DNA repair.
    The protein product of BRCA1 functions in homologous recombination repair.
    Mutations in BRCA1 are associated with hereditary breast and ovarian cancer.
    
    The p53 gene is another important tumor suppressor involved in cancer pathways.
    p53 mutations are found in over 50% of human cancers.
    
    Alzheimer's disease is a progressive neurological disorder affecting memory.
    The APP gene mutations are linked to early-onset Alzheimer's disease.
    """


@pytest.fixture
def sample_protein_document():
    """Sample document with protein and experimental data."""
    return """
    In agreement with previous results we detected two predominant proteins of 65 and 50 kDa,
    corresponding to lipoic acid-bound PDH-E2 and α-KGDH-E2, respectively.
    
    The biochemical assay confirms the presence of these protein complexes.
    Western blot analysis showed strong signals for both protein bands.
    
    These findings are consistent with our hypothesis about enzyme function.
    """


@pytest.fixture
def mock_test_model():
    """Create a TestModel for basic testing."""
    return TestModel()


def create_mock_function_model(responses: Dict[str, str]) -> FunctionModel:
    """Create a FunctionModel that returns predefined responses based on prompt content."""

    def model_function(messages: List[ModelMessage], info: AgentInfo) -> ModelResponse:
        # Get the last user message
        user_content = ""
        for message in reversed(messages):
            for part in message.parts:
                if hasattr(part, "content"):
                    user_content = part.content.lower()
                    break
            if user_content:
                break

        # Find matching response
        for trigger, response in responses.items():
            if trigger.lower() in user_content:
                return ModelResponse(parts=[TextPart(response)])

        # Default response
        return ModelResponse(parts=[TextPart("No matching response found")])

    return FunctionModel(model_function)


@pytest.fixture
def message_capture():
    """Fixture to capture messages sent to mock models."""
    captured_messages = []

    def create_capturing_model(base_response: str = "test response"):
        def model_function(
            messages: List[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            captured_messages.extend(messages)
            return ModelResponse(parts=[TextPart(base_response)])

        return FunctionModel(model_function)

    return captured_messages, create_capturing_model


@pytest.fixture
def extraction_config():
    """Create extraction config for testing."""
    config_data = {
        "task": {
            "kinds": {
                "gene": {
                    "kind": ["gene"],
                    "form": ["name"],
                    "example": ["BRCA1", "TP53"],
                },
                "disease": {
                    "kind": ["disease"],
                    "form": ["name"],
                    "example": ["breast cancer", "lung cancer"],
                },
            },
            "relation": "interaction",
            "context": "Test gene-disease interactions",
        },
        "agents": {"_": {"llm": "test"}},
        "cache": {"directory": "test_cache"},
    }
    return IfetcherConfig(**config_data)


@pytest.fixture
def test_deps(extraction_config):
    """Create test dependencies."""
    page_fetcher = PageFetcher(extraction_config)
    return ExtractionDeps.from_config(
        extraction_config, page_fetcher, model=TestModel()
    )


@pytest.fixture
def test_resource_pool():
    """Create resource pool with test document for extraction node testing."""
    documents = [(TEST_URLS[0], "BRCA1 Study", BRCA1_DOCUMENT)]
    return create_test_resource_pool(documents)


@pytest.fixture
def test_state(test_resource_pool):
    """Create test extraction state."""
    return ExtractionState(resource_pool=test_resource_pool)


@pytest.fixture
def test_context(test_state, test_deps):
    """Create test graph run context."""
    return GraphRunContext(state=test_state, deps=test_deps)
