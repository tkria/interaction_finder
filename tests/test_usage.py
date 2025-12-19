"""Tests for usage tracking module."""

import json
from unittest.mock import Mock

import pytest
from pydantic_ai.usage import RunUsage

from interaction_finder.usage import (
    AgentUsage,
    PipelineUsage,
    StageUsage,
    _get_model_id,
    record_usage,
)


def mock_agent(model: str) -> Mock:
    """Create a mock agent with given model string."""
    agent = Mock()
    agent.model = model
    return agent


class TestAgentUsage:
    """Tests for AgentUsage model."""

    def test_creation_with_model_and_usage(self):
        """AgentUsage stores model and usage."""
        usage = RunUsage(requests=1, input_tokens=100, output_tokens=50)
        agent_usage = AgentUsage(model="openai:gpt-4o", usage=usage)
        assert agent_usage.model == "openai:gpt-4o"
        assert agent_usage.usage.requests == 1
        assert agent_usage.usage.input_tokens == 100
        assert agent_usage.usage.output_tokens == 50

    def test_default_usage_is_empty(self):
        """AgentUsage defaults to empty RunUsage."""
        agent_usage = AgentUsage(model="openai:gpt-4o")
        assert agent_usage.usage.requests == 0
        assert agent_usage.usage.input_tokens == 0
        assert agent_usage.usage.output_tokens == 0

    def test_usage_accumulation_with_plus_equals(self):
        """Usage can be accumulated with +=."""
        agent_usage = AgentUsage(
            model="openai:gpt-4o",
            usage=RunUsage(requests=1, input_tokens=100, output_tokens=50),
        )
        agent_usage.usage += RunUsage(requests=1, input_tokens=200, output_tokens=100)
        assert agent_usage.usage.requests == 2
        assert agent_usage.usage.input_tokens == 300
        assert agent_usage.usage.output_tokens == 150


class TestRecordUsage:
    """Tests for record_usage function."""

    def test_creates_new_agent_usage(self):
        """record_usage creates new AgentUsage for unknown agent."""
        stage: StageUsage = {}
        agent = mock_agent("openai:gpt-4o")
        usage = RunUsage(requests=1, input_tokens=100, output_tokens=50)
        record_usage(stage, "entity", agent, usage)
        assert "entity" in stage
        assert stage["entity"].model == "openai:gpt-4o"
        assert stage["entity"].usage.requests == 1
        assert stage["entity"].usage.input_tokens == 100

    def test_accumulates_existing_agent_usage(self):
        """record_usage accumulates usage for existing agent."""
        stage: StageUsage = {}
        agent = mock_agent("openai:gpt-4o")
        usage1 = RunUsage(requests=1, input_tokens=100, output_tokens=50)
        usage2 = RunUsage(requests=1, input_tokens=200, output_tokens=100)
        record_usage(stage, "entity", agent, usage1)
        record_usage(stage, "entity", agent, usage2)
        assert stage["entity"].usage.requests == 2
        assert stage["entity"].usage.input_tokens == 300
        assert stage["entity"].usage.output_tokens == 150

    def test_multiple_agents_tracked_separately(self):
        """Different agents are tracked separately."""
        stage: StageUsage = {}
        agent = mock_agent("openai:gpt-4o")
        record_usage(stage, "entity", agent, RunUsage(requests=1, input_tokens=100))
        record_usage(stage, "pair_judge", agent, RunUsage(requests=2, input_tokens=500))
        assert stage["entity"].usage.requests == 1
        assert stage["pair_judge"].usage.requests == 2

    def test_accepts_result_object(self):
        """record_usage extracts usage from AgentRunResult."""
        stage: StageUsage = {}
        agent = mock_agent("openai:gpt-4o")
        # Mock a result object with usage() method
        mock_result = Mock()
        mock_result.usage.return_value = RunUsage(
            requests=3, input_tokens=500, output_tokens=100
        )
        record_usage(stage, "entity", agent, mock_result)
        assert stage["entity"].usage.requests == 3
        assert stage["entity"].usage.input_tokens == 500
        assert stage["entity"].usage.output_tokens == 100


class TestPipelineUsage:
    """Tests for PipelineUsage model."""

    def test_empty_pipeline_usage(self):
        """Empty PipelineUsage has zero totals."""
        usage = PipelineUsage()
        assert usage.total_tokens == 0
        assert usage.total_by_model() == {}

    def test_total_tokens_aggregation(self):
        """total_tokens sums across all stages and agents."""
        usage = PipelineUsage()
        record_usage(
            usage.keywords,
            "query_expander",
            mock_agent("openai:gpt-4o-mini"),
            RunUsage(input_tokens=100, output_tokens=50),
        )
        record_usage(
            usage.extraction,
            "entity",
            mock_agent("openai:gpt-4o"),
            RunUsage(input_tokens=200, output_tokens=100),
        )
        assert usage.total_tokens == 450  # 100+50 + 200+100

    def test_total_by_model_aggregation(self):
        """total_by_model aggregates tokens by model across stages."""
        usage = PipelineUsage()
        # Same model in different stages
        record_usage(
            usage.keywords,
            "query_expander",
            mock_agent("openai:gpt-4o-mini"),
            RunUsage(input_tokens=100, output_tokens=50),
        )
        record_usage(
            usage.search,
            "result_selector",
            mock_agent("openai:gpt-4o-mini"),
            RunUsage(input_tokens=200, output_tokens=100),
        )
        # Different model
        record_usage(
            usage.extraction,
            "pair_judge",
            mock_agent("openai:gpt-4o"),
            RunUsage(input_tokens=500, output_tokens=200),
        )
        totals = usage.total_by_model()
        assert totals["openai:gpt-4o-mini"] == 450  # 150 + 300
        assert totals["openai:gpt-4o"] == 700

    def test_json_serialization_roundtrip(self):
        """PipelineUsage can be serialized to JSON and back."""
        usage = PipelineUsage()
        record_usage(
            usage.keywords,
            "query_expander",
            mock_agent("openai:gpt-4o-mini"),
            RunUsage(requests=3, input_tokens=1000, output_tokens=500),
        )
        record_usage(
            usage.extraction,
            "entity",
            mock_agent("openai:gpt-4o"),
            RunUsage(requests=10, input_tokens=5000, output_tokens=1000),
        )
        # Serialize to JSON
        json_str = usage.model_dump_json()
        # Deserialize back
        restored = PipelineUsage.model_validate_json(json_str)
        # Verify equality
        assert restored.keywords["query_expander"].model == "openai:gpt-4o-mini"
        assert restored.keywords["query_expander"].usage.requests == 3
        assert restored.keywords["query_expander"].usage.input_tokens == 1000
        assert restored.extraction["entity"].model == "openai:gpt-4o"
        assert restored.extraction["entity"].usage.requests == 10
        assert restored.total_tokens == usage.total_tokens

    def test_json_structure(self):
        """Verify JSON structure matches expected format."""
        usage = PipelineUsage()
        record_usage(
            usage.extraction,
            "pair_judge",
            mock_agent("openai:gpt-4o"),
            RunUsage(
                requests=30,
                input_tokens=65393,
                output_tokens=953,
            ),
        )
        data = json.loads(usage.model_dump_json())
        assert "extraction" in data
        assert "pair_judge" in data["extraction"]
        pair_judge = data["extraction"]["pair_judge"]
        assert pair_judge["model"] == "openai:gpt-4o"
        assert pair_judge["usage"]["requests"] == 30
        assert pair_judge["usage"]["input_tokens"] == 65393
        assert pair_judge["usage"]["output_tokens"] == 953


class TestGetModelId:
    """Tests for _get_model_id function."""

    def test_get_model_id_requires_agent_with_model(self):
        """_get_model_id extracts provider:model_name from agent."""
        # Create a mock agent-like object with model attribute
        mock_model = Mock()
        mock_model.system = "openai"
        mock_model.model_name = "gpt-4o"
        mock_agent = Mock()
        mock_agent.model = mock_model
        assert _get_model_id(mock_agent) == "openai:gpt-4o"

    def test_get_model_id_anthropic(self):
        """_get_model_id works for anthropic models."""
        mock_model = Mock()
        mock_model.system = "anthropic"
        mock_model.model_name = "claude-sonnet-4-5"
        mock_agent = Mock()
        mock_agent.model = mock_model
        assert _get_model_id(mock_agent) == "anthropic:claude-sonnet-4-5"

    def test_get_model_id_string_model(self):
        """_get_model_id returns string models as-is."""
        mock_agent = Mock()
        mock_agent.model = "openai:gpt-4o"
        assert _get_model_id(mock_agent) == "openai:gpt-4o"

    def test_get_model_id_none_model(self):
        """_get_model_id returns 'unknown' for None model."""
        mock_agent = Mock()
        mock_agent.model = None
        assert _get_model_id(mock_agent) == "unknown"


class TestAgentUsageCopyDeep:
    """Tests for AgentUsage.copy_deep method."""

    def test_copy_deep_creates_independent_instance(self):
        """copy_deep creates a fully independent copy."""
        original = AgentUsage(
            model="openai:gpt-4o",
            usage=RunUsage(
                requests=5,
                input_tokens=1000,
                output_tokens=500,
                details={"cached_tokens": 200},
            ),
        )
        copy = original.copy_deep()
        # Modify original
        original.usage += RunUsage(requests=1, input_tokens=100)
        original.usage.details["new_key"] = 999
        # Copy should be unaffected
        assert copy.usage.requests == 5
        assert copy.usage.input_tokens == 1000
        assert copy.usage.details == {"cached_tokens": 200}
        assert "new_key" not in copy.usage.details

    def test_copy_deep_preserves_all_fields(self):
        """copy_deep preserves all RunUsage fields."""
        original = AgentUsage(
            model="anthropic:claude-sonnet",
            usage=RunUsage(
                requests=10,
                tool_calls=3,
                input_tokens=5000,
                output_tokens=1000,
                cache_write_tokens=100,
                cache_read_tokens=500,
                input_audio_tokens=50,
                cache_audio_read_tokens=25,
                output_audio_tokens=75,
                details={"reasoning_tokens": 200},
            ),
        )
        copy = original.copy_deep()
        assert copy.model == "anthropic:claude-sonnet"
        assert copy.usage.requests == 10
        assert copy.usage.tool_calls == 3
        assert copy.usage.input_tokens == 5000
        assert copy.usage.output_tokens == 1000
        assert copy.usage.cache_write_tokens == 100
        assert copy.usage.cache_read_tokens == 500
        assert copy.usage.input_audio_tokens == 50
        assert copy.usage.cache_audio_read_tokens == 25
        assert copy.usage.output_audio_tokens == 75
        assert copy.usage.details == {"reasoning_tokens": 200}


class TestStageUsageDeepCopy:
    """Tests for deep copying stage usage via dict comprehension."""

    def test_deep_copy_creates_independent_copies(self):
        """Dict comprehension with copy_deep creates independent copies."""
        original: StageUsage = {
            "entity": AgentUsage(
                model="openai:gpt-4o",
                usage=RunUsage(requests=5, input_tokens=1000),
            ),
            "pair_judge": AgentUsage(
                model="openai:gpt-4o-mini",
                usage=RunUsage(requests=10, input_tokens=2000),
            ),
        }
        # This is the pattern used in run.py files
        copy = {k: v.copy_deep() for k, v in original.items()}
        # Modify original
        original["entity"].usage += RunUsage(requests=1, input_tokens=100)
        # Copy should be unaffected
        assert copy["entity"].usage.requests == 5
        assert copy["entity"].usage.input_tokens == 1000
        # Different dict instances
        assert copy is not original
        assert copy["entity"] is not original["entity"]


class TestRecordUsageModelMismatch:
    """Tests for record_usage handling of model mismatches."""

    def test_same_agent_different_model_tracked_separately(self):
        """Same agent with different models tracked with model suffix."""
        stage: StageUsage = {}
        record_usage(
            stage,
            "entity",
            mock_agent("openai:gpt-4o"),
            RunUsage(requests=1, input_tokens=100),
        )
        record_usage(
            stage,
            "entity",
            mock_agent("openai:gpt-4o-mini"),
            RunUsage(requests=2, input_tokens=200),
        )
        # First model uses base key
        assert "entity" in stage
        assert stage["entity"].model == "openai:gpt-4o"
        assert stage["entity"].usage.requests == 1
        # Second model uses suffixed key
        assert "entity:openai:gpt-4o-mini" in stage
        assert stage["entity:openai:gpt-4o-mini"].model == "openai:gpt-4o-mini"
        assert stage["entity:openai:gpt-4o-mini"].usage.requests == 2

    def test_model_mismatch_accumulates_on_suffixed_key(self):
        """Multiple calls with mismatched model accumulate correctly."""
        stage: StageUsage = {}
        agent_4o = mock_agent("openai:gpt-4o")
        agent_mini = mock_agent("openai:gpt-4o-mini")
        # First call with model A
        record_usage(stage, "entity", agent_4o, RunUsage(requests=1, input_tokens=100))
        # Two calls with model B
        record_usage(
            stage, "entity", agent_mini, RunUsage(requests=1, input_tokens=200)
        )
        record_usage(
            stage, "entity", agent_mini, RunUsage(requests=1, input_tokens=300)
        )
        # Model A: 1 request, 100 tokens
        assert stage["entity"].usage.requests == 1
        assert stage["entity"].usage.input_tokens == 100
        # Model B: 2 requests, 500 tokens
        assert stage["entity:openai:gpt-4o-mini"].usage.requests == 2
        assert stage["entity:openai:gpt-4o-mini"].usage.input_tokens == 500
