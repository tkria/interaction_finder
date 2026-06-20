"""Tests for session-only provider API keys (module + routes).

monkeypatch.setenv/delenv keep these from touching the real environment.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from interaction_finder.web import api_keys
from interaction_finder.web.app import create_app


@pytest.fixture(autouse=True)
def clear_provider_env(monkeypatch):
    """Start each test with every provider env var unset."""
    for p in api_keys.PROVIDERS:
        monkeypatch.delenv(p.env, raising=False)


def test_status_reports_unset_then_set(monkeypatch):
    status = {s["env"]: s["set"] for s in api_keys.key_status()}
    assert status["OPENAI_API_KEY"] is False
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    status = {s["env"]: s["set"] for s in api_keys.key_status()}
    assert status["OPENAI_API_KEY"] is True


def test_status_never_includes_the_value(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    blob = repr(api_keys.key_status())
    assert "sk-secret" not in blob


def test_set_keys_sets_env_and_ignores_blank_and_unknown():
    done = api_keys.set_keys(
        {"OPENAI_API_KEY": "sk-1", "ANTHROPIC_API_KEY": "  ", "BOGUS": "x"}
    )
    assert done == ["OPENAI_API_KEY"]
    import os

    assert os.environ["OPENAI_API_KEY"] == "sk-1"
    assert "ANTHROPIC_API_KEY" not in os.environ
    assert "BOGUS" not in os.environ


def test_blank_value_does_not_clear_existing(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "keep-me")
    api_keys.set_keys({"OPENAI_API_KEY": ""})
    import os

    assert os.environ["OPENAI_API_KEY"] == "keep-me"


def test_keys_routes_round_trip():
    client = TestClient(create_app())
    status = {p["env"]: p["set"] for p in client.get("/keys").json()["providers"]}
    assert status["GEMINI_API_KEY"] is False
    resp = client.post("/keys", json={"keys": {"GEMINI_API_KEY": "g-123"}})
    after = {p["env"]: p["set"] for p in resp.json()["providers"]}
    assert after["GEMINI_API_KEY"] is True
    # The response never carries the value back.
    assert "g-123" not in resp.text
