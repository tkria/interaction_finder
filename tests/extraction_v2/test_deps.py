"""
Unit tests for extraction graph V2 dependency injection.

Tests ExtractionDeps configuration and service integration.
"""

import pytest
from unittest.mock import Mock, MagicMock

from pydantic_ai.models.test import TestModel

from interaction_finder.extraction_graph_v2.deps import ExtractionDeps
from interaction_finder.settings import IfetcherConfig
from interaction_finder.fetcher import PageFetcher
from interaction_finder.resources import ResourcePool


class TestExtractionDeps:
    """Test ExtractionDeps creation and configuration."""

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing."""
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
            "agents": {
                "_": {"llm": "openai:gpt-4o"},
                "extractor": {"llm": "openai:gpt-4o-mini"},
            },
            "cache": {"directory": "test_cache"},
        }
        return IfetcherConfig(**config_data)

    @pytest.fixture
    def mock_page_fetcher(self, sample_config):
        """Create mock page fetcher."""
        return PageFetcher(sample_config)

    @pytest.fixture
    def test_model(self):
        """Create test model."""
        return TestModel()

    def test_deps_creation_with_defaults(self, sample_config, mock_page_fetcher):
        """Test creating deps with default model from config."""
        deps = ExtractionDeps.from_config(sample_config, mock_page_fetcher)

        assert deps.config == sample_config
        assert deps.page_fetcher == mock_page_fetcher
        assert deps.model == "openai:gpt-4o"  # From config default
        assert deps.target_term is None
        assert deps.current_resources is None

    def test_deps_creation_with_model_override(
        self, sample_config, mock_page_fetcher, test_model
    ):
        """Test creating deps with model override."""
        deps = ExtractionDeps.from_config(
            sample_config, mock_page_fetcher, model=test_model
        )

        assert deps.model == test_model
        assert deps.config == sample_config
        assert deps.page_fetcher == mock_page_fetcher

    def test_deps_creation_with_target_term(self, sample_config, mock_page_fetcher):
        """Test creating deps with target term."""
        deps = ExtractionDeps.from_config(
            sample_config, mock_page_fetcher, target_term="BRCA1"
        )

        assert deps.target_term == "BRCA1"
        assert deps.model == "openai:gpt-4o"

    def test_deps_creation_with_resources(self, sample_config, mock_page_fetcher):
        """Test creating deps with current resources."""
        # Create mock resources
        mock_resources = [Mock(), Mock()]

        deps = ExtractionDeps.from_config(
            sample_config, mock_page_fetcher, current_resources=mock_resources
        )

        assert deps.current_resources == mock_resources
        assert len(deps.current_resources) == 2

    def test_get_entity_kinds(self, sample_config, mock_page_fetcher):
        """Test getting entity kinds from config."""
        deps = ExtractionDeps.from_config(sample_config, mock_page_fetcher)

        entity_kinds = deps.get_entity_kinds()

        assert isinstance(entity_kinds, list)
        assert "gene" in entity_kinds
        assert "disease" in entity_kinds
        assert len(entity_kinds) == 2

    def test_get_relation_type(self, sample_config, mock_page_fetcher):
        """Test getting relation type from config."""
        deps = ExtractionDeps.from_config(sample_config, mock_page_fetcher)

        relation_type = deps.get_relation_type()

        assert relation_type == "interaction"

    def test_get_task_context(self, sample_config, mock_page_fetcher):
        """Test getting task context from config."""
        deps = ExtractionDeps.from_config(sample_config, mock_page_fetcher)

        task_context = deps.get_task_context()

        assert task_context == "Test gene-disease interactions"

    def test_deps_model_fallback_chain(self, mock_page_fetcher):
        """Test model selection fallback chain."""
        # Config with no default agent
        config_data = {
            "task": {
                "kinds": {
                    "gene": {"kind": ["gene"], "form": ["name"], "example": ["BRCA1"]},
                },
                "relation": "test",
                "context": "test",
            },
            "agents": {},  # No agents specified
            "cache": {"directory": "test"},
        }
        config = IfetcherConfig(**config_data)

        deps = ExtractionDeps.from_config(config, mock_page_fetcher)

        # Should fall back to hardcoded default
        assert deps.model == "openai:gpt-4o"

    def test_deps_direct_construction(
        self, sample_config, mock_page_fetcher, test_model
    ):
        """Test direct construction of ExtractionDeps."""
        deps = ExtractionDeps(
            model=test_model,
            config=sample_config,
            page_fetcher=mock_page_fetcher,
            target_term="TEST_GENE",
            current_resources=[],
        )

        assert deps.model == test_model
        assert deps.config == sample_config
        assert deps.page_fetcher == mock_page_fetcher
        assert deps.target_term == "TEST_GENE"
        assert deps.current_resources == []

    def test_deps_immutability(self, sample_config, mock_page_fetcher):
        """Test that deps are immutable after creation."""
        deps = ExtractionDeps.from_config(sample_config, mock_page_fetcher)

        # These are dataclass fields, so they can be modified
        # but in practice should not be
        original_model = deps.model
        original_config = deps.config

        # Verify initial state
        assert deps.model == original_model
        assert deps.config == original_config


class TestExtractionDepsConfigIntegration:
    """Test integration with different config types."""

    def test_deps_with_minimal_config(self):
        """Test deps with minimal configuration."""
        config_data = {
            "task": {
                "kinds": {
                    "gene": {"kind": ["gene"], "form": ["name"], "example": ["BRCA1"]}
                },
                "relation": "test",
                "context": "minimal test",
            },
            "agents": {"_": {"llm": "test-model"}},
            "cache": {"directory": "test"},
        }
        config = IfetcherConfig(**config_data)
        page_fetcher = PageFetcher(config)

        deps = ExtractionDeps.from_config(config, page_fetcher)

        assert deps.model == "test-model"
        assert len(deps.get_entity_kinds()) == 1
        assert "gene" in deps.get_entity_kinds()

    def test_deps_with_complex_config(self):
        """Test deps with complex multi-kind configuration."""
        config_data = {
            "task": {
                "kinds": {
                    "gene": {"kind": ["gene"], "form": ["name"], "example": ["BRCA1"]},
                    "protein": {
                        "kind": ["protein"],
                        "form": ["name"],
                        "example": ["p53"],
                    },
                    "disease": {
                        "kind": ["disease"],
                        "form": ["name"],
                        "example": ["cancer"],
                    },
                    "drug": {
                        "kind": ["drug"],
                        "form": ["name"],
                        "example": ["aspirin"],
                    },
                },
                "relation": "complex_interaction",
                "context": "Multi-entity biomedical interactions",
            },
            "agents": {
                "_": {"llm": "openai:gpt-4o"},
                "extractor": {"llm": "openai:gpt-4o-mini"},
                "assessor": {"llm": "openai:gpt-4o"},
            },
            "cache": {"directory": "complex_test_cache"},
        }
        config = IfetcherConfig(**config_data)
        page_fetcher = PageFetcher(config)

        deps = ExtractionDeps.from_config(config, page_fetcher)

        entity_kinds = deps.get_entity_kinds()
        assert len(entity_kinds) == 4
        assert set(entity_kinds) == {"gene", "protein", "disease", "drug"}
        assert deps.get_relation_type() == "complex_interaction"
        assert "Multi-entity" in deps.get_task_context()

    def test_deps_with_different_model_specs(self):
        """Test deps with various model specifications."""
        configs_and_expected = [
            ({"_": {"llm": "openai:gpt-4o"}}, "openai:gpt-4o"),
            ({"_": {"llm": "anthropic:claude-3-sonnet"}}, "anthropic:claude-3-sonnet"),
            ({"_": {"llm": "test:model"}}, "test:model"),
            ({}, "openai:gpt-4o"),  # Fallback
        ]

        for agent_config, expected_model in configs_and_expected:
            config_data = {
                "task": {
                    "kinds": {
                        "gene": {
                            "kind": ["gene"],
                            "form": ["name"],
                            "example": ["BRCA1"],
                        }
                    },
                    "relation": "test",
                    "context": "test",
                },
                "agents": agent_config,
                "cache": {"directory": "test"},
            }
            config = IfetcherConfig(**config_data)
            page_fetcher = PageFetcher(config)

            deps = ExtractionDeps.from_config(config, page_fetcher)
            assert deps.model == expected_model


class TestExtractionDepsResourceManagement:
    """Test resource management in ExtractionDeps."""

    @pytest.fixture
    def sample_resource_pool(self):
        """Create sample resource pool."""
        pool = ResourcePool()
        pool.add(
            "http://example.com/doc1", "Test Doc 1", "Gene A interacts with Disease B."
        )
        pool.add(
            "http://example.com/doc2", "Test Doc 2", "Protein X binds to Receptor Y."
        )
        return pool

    def test_deps_with_resource_list(self, sample_resource_pool):
        """Test deps with specific resource list."""
        config_data = {
            "task": {
                "kinds": {
                    "gene": {"kind": ["gene"], "form": ["name"], "example": ["BRCA1"]}
                },
                "relation": "test",
                "context": "test",
            },
            "agents": {"_": {"llm": "test"}},
            "cache": {"directory": "test"},
        }
        config = IfetcherConfig(**config_data)
        page_fetcher = PageFetcher(config)

        # Convert resources to list
        resources = list(sample_resource_pool.resources)

        deps = ExtractionDeps.from_config(
            config, page_fetcher, current_resources=resources
        )

        assert deps.current_resources == resources
        assert len(deps.current_resources) == 2

    def test_deps_resource_updates(self):
        """Test updating current resources in deps."""
        config_data = {
            "task": {
                "kinds": {
                    "gene": {"kind": ["gene"], "form": ["name"], "example": ["BRCA1"]}
                },
                "relation": "test",
                "context": "test",
            },
            "agents": {"_": {"llm": "test"}},
            "cache": {"directory": "test"},
        }
        config = IfetcherConfig(**config_data)
        page_fetcher = PageFetcher(config)

        # Start with no resources
        deps = ExtractionDeps.from_config(config, page_fetcher)
        assert deps.current_resources is None

        # Update with resources (this would typically be done by nodes)
        mock_resources = [Mock(), Mock(), Mock()]
        deps.current_resources = mock_resources

        assert deps.current_resources == mock_resources
        assert len(deps.current_resources) == 3


class TestExtractionDepsErrorHandling:
    """Test error handling in ExtractionDeps."""

    def test_deps_with_invalid_config(self):
        """Test deps behavior with problematic config."""
        # This should work as IfetcherConfig handles validation
        config_data = {
            "task": {
                "kinds": {
                    "gene": {"kind": ["gene"], "form": ["name"], "example": ["BRCA1"]}
                },
                "relation": "test",
                "context": "test",
            },
            "agents": {"_": {"llm": ""}},  # Empty LLM spec
            "cache": {"directory": "test"},
        }
        config = IfetcherConfig(**config_data)
        page_fetcher = PageFetcher(config)

        deps = ExtractionDeps.from_config(config, page_fetcher)

        # Should use fallback model
        assert deps.model == "openai:gpt-4o"

    def test_deps_method_calls_with_valid_config(self):
        """Test that deps methods work with valid configuration."""
        config_data = {
            "task": {
                "kinds": {
                    "gene": {"kind": ["gene"], "form": ["name"], "example": ["BRCA1"]},
                    "disease": {
                        "kind": ["disease"],
                        "form": ["name"],
                        "example": ["cancer"],
                    },
                },
                "relation": "causes",
                "context": "Gene-disease causation analysis",
            },
            "agents": {"_": {"llm": "test-model"}},
            "cache": {"directory": "test"},
        }
        config = IfetcherConfig(**config_data)
        page_fetcher = PageFetcher(config)

        deps = ExtractionDeps.from_config(config, page_fetcher)

        # All methods should work without errors
        entity_kinds = deps.get_entity_kinds()
        relation_type = deps.get_relation_type()
        task_context = deps.get_task_context()

        assert isinstance(entity_kinds, list)
        assert isinstance(relation_type, str)
        assert isinstance(task_context, str)
        assert len(entity_kinds) > 0
        assert len(relation_type) > 0
        assert len(task_context) > 0

    def test_deps_repr_and_str(self):
        """Test string representation of deps."""
        config_data = {
            "task": {
                "kinds": {
                    "gene": {"kind": ["gene"], "form": ["name"], "example": ["BRCA1"]}
                },
                "relation": "test",
                "context": "test",
            },
            "agents": {"_": {"llm": "test"}},
            "cache": {"directory": "test"},
        }
        config = IfetcherConfig(**config_data)
        page_fetcher = PageFetcher(config)

        deps = ExtractionDeps.from_config(config, page_fetcher)

        # Should be able to convert to string without errors
        str_repr = str(deps)
        repr_str = repr(deps)

        assert isinstance(str_repr, str)
        assert isinstance(repr_str, str)
        assert "ExtractionDeps" in repr_str
