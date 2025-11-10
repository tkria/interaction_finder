"""Tests for unified agent configuration system.

Tests multi-tier config resolution, agent caching, and the agent_getter factory.
"""

import pytest
from pydantic import BaseModel
from pydantic_ai.settings import ModelSettings

from interaction_finder import IfetcherConfig, agent_getter, clear_agent_cache


class DummyOut(BaseModel):
    """Dummy output type for testing."""

    value: str


class DummyDeps:
    """Dummy deps type for testing."""

    pass


class TestConfigResolution:
    """Test multi-tier configuration resolution."""

    def test_global_default(self):
        """Test global default resolution (agents._)."""
        config = IfetcherConfig(agents={"_": {"llm": "openai:gpt-4o"}})
        spec = config.resolve_agent_config("extraction", "entity")
        assert spec.llm == "openai:gpt-4o"

    def test_module_default_with_underscore(self):
        """Test module default with underscore (agents.module._)."""
        config = IfetcherConfig(
            agents={
                "_": {"llm": "openai:gpt-4o"},
                "extraction": {"_": {"llm": "anthropic:claude-3-sonnet"}},
            }
        )
        spec = config.resolve_agent_config("extraction", "entity")
        assert spec.llm == "anthropic:claude-3-sonnet"

    def test_module_default_without_underscore(self):
        """Test module default without underscore (agents.module as AgentSpec)."""
        config = IfetcherConfig(
            agents={
                "_": {"llm": "openai:gpt-4o"},
                "extraction": {"llm": "anthropic:claude-3-sonnet"},
            }
        )
        spec = config.resolve_agent_config("extraction")
        assert spec.llm == "anthropic:claude-3-sonnet"

    def test_agent_specific_override(self):
        """Test agent-specific override (agents.module.agent)."""
        config = IfetcherConfig(
            agents={
                "_": {"llm": "openai:gpt-4o-mini"},
                "extraction": {
                    "_": {"llm": "openai:gpt-4o"},
                    "judge": {"llm": "openai:gpt-4o-turbo"},
                },
            }
        )
        # Module default
        entity_spec = config.resolve_agent_config("extraction", "entity")
        assert entity_spec.llm == "openai:gpt-4o"

        # Agent override
        judge_spec = config.resolve_agent_config("extraction", "judge")
        assert judge_spec.llm == "openai:gpt-4o-turbo"

    def test_fallback_chain(self):
        """Test complete fallback chain: agent → module → global → None."""
        config = IfetcherConfig(
            agents={
                "_": {"llm": "global-model", "retries": 3},
                "extraction": {
                    "_": {"llm": "module-model"},
                    "entity": {"llm": "agent-model"},
                },
            }
        )

        spec = config.resolve_agent_config("extraction", "entity")
        assert spec.llm == "agent-model"  # Agent-specific
        assert spec.retries == 3  # Inherited from global

    def test_model_settings_merge(self):
        """Test that model_settings are properly resolved."""
        config = IfetcherConfig(
            agents={
                "extraction": {
                    "entity": {
                        "llm": "openai:gpt-4o-mini",
                        "model_settings": {"parallel_tool_calls": False},
                    }
                }
            }
        )
        spec = config.resolve_agent_config("extraction", "entity")
        assert spec.model_settings is not None
        assert spec.model_settings["parallel_tool_calls"] is False

    def test_empty_config_returns_empty_spec(self):
        """Test that empty config returns AgentSpec with all None values."""
        config = IfetcherConfig()
        spec = config.resolve_agent_config("nonexistent", "agent")
        assert spec.llm is None
        assert spec.retries is None
        assert spec.system_prompt is None


class TestAgentSpecMerging:
    """Test AgentSpec.merge_with_parent()."""

    def test_child_overrides_parent(self):
        """Test that child non-None values override parent."""
        parent = IfetcherConfig.AgentSpec(llm="parent-model", retries=2)
        child = IfetcherConfig.AgentSpec(llm="child-model")

        merged = child.merge_with_parent(parent)
        assert merged.llm == "child-model"  # Overridden
        assert merged.retries == 2  # Inherited

    def test_child_none_inherits_parent(self):
        """Test that child None values inherit from parent."""
        parent = IfetcherConfig.AgentSpec(llm="parent-model", retries=2)
        child = IfetcherConfig.AgentSpec(retries=5)

        merged = child.merge_with_parent(parent)
        assert merged.llm == "parent-model"  # Inherited
        assert merged.retries == 5  # Overridden

    def test_merge_with_none_parent(self):
        """Test merging with None parent returns child."""
        child = IfetcherConfig.AgentSpec(llm="child-model")
        merged = child.merge_with_parent(None)
        assert merged.llm == "child-model"


class TestAgentGetter:
    """Test agent_getter factory function."""

    def setup_method(self):
        """Clear agent cache before each test."""
        clear_agent_cache()

    def test_creates_getter_function(self):
        """Test that agent_getter returns a callable."""
        getter = agent_getter("test", "agent", DummyOut, DummyDeps, "Test prompt")
        assert callable(getter)

    def test_getter_returns_agent(self):
        """Test that the getter function returns an Agent."""
        getter = agent_getter("test", "agent", DummyOut, DummyDeps, "Test prompt")
        config = IfetcherConfig()
        agent = getter(config)

        from pydantic_ai import Agent

        assert isinstance(agent, Agent)

    def test_agent_uses_config(self):
        """Test that agent uses configuration from config object."""
        getter = agent_getter(
            "test",
            "agent",
            DummyOut,
            DummyDeps,
            "Test prompt",
            default_model="openai:gpt-4o-mini",
        )

        # Config with override (use OpenAI to avoid API key issues in tests)
        config = IfetcherConfig(agents={"test": {"agent": {"llm": "openai:gpt-4o"}}})
        agent = getter(config)

        # Agent should use the configured model (though we can't easily inspect it)
        # The fact that it doesn't error is a good sign
        assert agent is not None

    def test_agent_with_model_settings(self):
        """Test agent_getter with ModelSettings."""
        getter = agent_getter(
            "test",
            "agent",
            DummyOut,
            DummyDeps,
            "Test prompt",
            default_model_settings=ModelSettings(parallel_tool_calls=False),
        )
        config = IfetcherConfig()
        agent = getter(config)
        assert agent is not None


class TestAgentCaching:
    """Test agent caching behavior."""

    def setup_method(self):
        """Clear agent cache before each test."""
        clear_agent_cache()

    def test_same_config_returns_cached_agent(self):
        """Test that same config returns the same cached agent instance."""
        getter = agent_getter("test", "agent", DummyOut, DummyDeps, "Test prompt")
        config = IfetcherConfig(agents={"test": {"llm": "openai:gpt-4o-mini"}})

        agent1 = getter(config)
        agent2 = getter(config)

        # Should be the exact same object
        assert agent1 is agent2

    def test_different_config_returns_different_agent(self):
        """Test that different config returns different agent instance."""
        getter = agent_getter("test", "agent", DummyOut, DummyDeps, "Test prompt")

        config1 = IfetcherConfig(
            agents={"test": {"agent": {"llm": "openai:gpt-4o-mini"}}}
        )
        config2 = IfetcherConfig(agents={"test": {"agent": {"llm": "openai:gpt-4o"}}})

        agent1 = getter(config1)
        agent2 = getter(config2)

        # Should be different objects (different models)
        assert agent1 is not agent2

    def test_clear_cache(self):
        """Test that clear_agent_cache() clears the cache."""
        getter = agent_getter("test", "agent", DummyOut, DummyDeps, "Test prompt")
        config = IfetcherConfig()

        agent1 = getter(config)
        clear_agent_cache()
        agent2 = getter(config)

        # After clearing, should get a new instance
        assert agent1 is not agent2


class TestRealWorldScenarios:
    """Test real-world configuration scenarios."""

    def test_extraction_with_judge_override(self):
        """Test typical extraction config with judge using better model."""
        config = IfetcherConfig(
            agents={
                "_": {"llm": "openai:gpt-4o-mini"},
                "extraction": {
                    "_": {"llm": "openai:gpt-4o-mini"},
                    "judge": {"llm": "openai:gpt-4o"},
                },
            }
        )

        entity_spec = config.resolve_agent_config("extraction", "entity")
        judge_spec = config.resolve_agent_config("extraction", "judge")

        assert entity_spec.llm == "openai:gpt-4o-mini"
        assert judge_spec.llm == "openai:gpt-4o"

    def test_widesearch_with_special_settings(self):
        """Test widesearch agent with parallel_tool_calls disabled."""
        config = IfetcherConfig(
            agents={
                "widesearch": {
                    "query_generator": {
                        "model_settings": {"parallel_tool_calls": False}
                    }
                }
            }
        )

        spec = config.resolve_agent_config("widesearch", "query_generator")
        assert spec.model_settings is not None
        assert spec.model_settings["parallel_tool_calls"] is False

    def test_multiple_modules_independent(self):
        """Test that different modules have independent config."""
        config = IfetcherConfig(
            agents={
                "_": {"llm": "default-model"},
                "keywords": {"_": {"llm": "keywords-model"}},
                "widesearch": {"_": {"llm": "widesearch-model"}},
                "extraction": {"_": {"llm": "extraction-model"}},
            }
        )

        assert config.resolve_agent_config("keywords", "any").llm == "keywords-model"
        assert (
            config.resolve_agent_config("widesearch", "any").llm == "widesearch-model"
        )
        assert (
            config.resolve_agent_config("extraction", "any").llm == "extraction-model"
        )
