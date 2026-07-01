"""Tests for session-only provider credentials (module + routes).

monkeypatch.setenv/delenv keep these from touching the real environment.
"""

from __future__ import annotations

import os

import pytest
from starlette.testclient import TestClient

from interaction_finder.settings import IfetcherConfig
from interaction_finder.web import api_keys
from interaction_finder.web.app import create_app


@pytest.fixture(autouse=True)
def clear_provider_env(monkeypatch):
    """Start each test with every provider env var unset."""
    for p in api_keys.PROVIDERS:
        monkeypatch.delenv(p.env, raising=False)


def _config(overrides: dict) -> IfetcherConfig:
    base = IfetcherConfig().model_dump()
    return IfetcherConfig.model_validate(
        IfetcherConfig.apply_overrides(base, overrides)
    )


def _shown_envs(status: dict) -> set[str]:
    return {row["env"] for row in status["shown"]}


def test_ncbi_always_shown_others_hidden_when_empty():
    status = api_keys.key_status(_config({}))
    assert "NCBI_API_KEY" in _shown_envs(status)
    # The code default is openai:gpt-4o-mini, so OpenAI is implicated even empty.
    assert "OPENAI_API_KEY" in _shown_envs(status)
    # An untouched provider is offered in the picker, not shown.
    available = {row["env"] for row in status["available"]}
    assert "GROQ_API_KEY" in available


def test_set_env_var_promotes_provider_to_shown(monkeypatch):
    assert "MISTRAL_API_KEY" not in _shown_envs(api_keys.key_status(_config({})))
    monkeypatch.setenv("MISTRAL_API_KEY", "m-key")
    status = api_keys.key_status(_config({}))
    assert "MISTRAL_API_KEY" in _shown_envs(status)
    row = next(r for r in status["shown"] if r["env"] == "MISTRAL_API_KEY")
    assert row["set"] is True


def test_configured_model_implicates_its_provider():
    status = api_keys.key_status(
        _config({"agents.extraction.pair_judge.llm": "groq:llama-3.1-70b"})
    )
    assert "GROQ_API_KEY" in _shown_envs(status)


def test_implicated_envs_collects_all_levels():
    config = _config(
        {
            "agents._.llm": "anthropic:claude-3-5-sonnet",
            "agents.extraction.pair_judge.llm": "groq:llama-3.1-70b",
        }
    )
    envs = api_keys.implicated_envs(config)
    assert {"ANTHROPIC_API_KEY", "GROQ_API_KEY"} <= envs


def test_global_override_replaces_the_code_default():
    # Overriding agents._.llm means nothing falls back to openai:gpt-4o-mini,
    # so OpenAI must not be implicated.
    config = _config({"agents._.llm": "groq:llama-3.1-70b"})
    envs = api_keys.implicated_envs(config)
    assert "GROQ_API_KEY" in envs
    assert "OPENAI_API_KEY" not in envs
    assert "OPENAI_API_KEY" not in _shown_envs(api_keys.key_status(config))


def test_code_default_implied_when_only_specific_agents_override():
    # A per-agent override leaves other agents on the code default, so OpenAI
    # (the default's provider) stays implicated.
    config = _config({"agents.extraction.pair_judge.llm": "groq:llama-3.1-70b"})
    assert {"GROQ_API_KEY", "OPENAI_API_KEY"} <= api_keys.implicated_envs(config)


def test_unknown_prefix_is_not_implicated():
    # bedrock overrides the global default and maps to no single-key row, so
    # nothing is implicated from it.
    config = _config({"agents._.llm": "bedrock:anthropic.claude-haiku-4-5"})
    assert "OPENAI_API_KEY" not in api_keys.implicated_envs(config)
    assert not any(
        row["prefix"] == "bedrock" for row in api_keys.key_status(config)["shown"]
    )


def test_secret_row_never_carries_value(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    status = api_keys.key_status(_config({}))
    row = next(r for r in status["shown"] if r["env"] == "OPENAI_API_KEY")
    assert "value" not in row
    assert "sk-secret" not in repr(status)


def test_non_secret_row_echoes_its_value(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    status = api_keys.key_status(_config({}))
    row = next(r for r in status["shown"] if r["env"] == "OPENAI_BASE_URL")
    assert row["secret"] is False
    assert row["value"] == "http://localhost:11434/v1"


def test_set_keys_sets_and_ignores_blank_secret_and_unknown():
    api_keys.set_keys(
        {"OPENAI_API_KEY": "sk-1", "ANTHROPIC_API_KEY": "  ", "BOGUS": "x"}
    )
    assert os.environ["OPENAI_API_KEY"] == "sk-1"
    assert "ANTHROPIC_API_KEY" not in os.environ
    assert "BOGUS" not in os.environ


def test_blank_secret_keeps_existing_but_blank_plain_clears(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "keep-me")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    api_keys.set_keys({"OPENAI_API_KEY": "", "OPENAI_BASE_URL": ""})
    assert os.environ["OPENAI_API_KEY"] == "keep-me"  # secret: untouched
    assert "OPENAI_BASE_URL" not in os.environ  # plain: cleared


def test_keys_routes_round_trip():
    client = TestClient(create_app())
    before = client.post("/keys/status", json={"overrides": {}}).json()
    assert "GEMINI_API_KEY" not in {r["env"] for r in before["shown"]}
    resp = client.post(
        "/keys", json={"keys": {"GEMINI_API_KEY": "g-123"}, "overrides": {}}
    )
    after = {r["env"]: r["set"] for r in resp.json()["shown"]}
    assert after["GEMINI_API_KEY"] is True
    assert "g-123" not in resp.text  # never echoed


def test_status_route_surfaces_configured_provider():
    client = TestClient(create_app())
    status = client.post(
        "/keys/status",
        json={"overrides": {"agents.search.reflector.llm": "mistral:mistral-large"}},
    ).json()
    assert "MISTRAL_API_KEY" in {r["env"] for r in status["shown"]}
