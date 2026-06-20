"""Tests for the config-editor HTTP surface (index injection + save route).

The autouse isolate_platform_dirs fixture keeps the saved default config in a
temp dir, so these never touch the real ~/.config.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from interaction_finder.web.app import create_app
from interaction_finder.web.config_store import load_default_config


@pytest.fixture
def client():
    return TestClient(create_app())


def test_index_injects_the_generated_form(client):
    body = client.get("/").text
    assert "{{ config_form }}" not in body  # token substituted
    assert 'name="stage.search.max_rounds"' in body
    assert 'id="config-dialog"' in body


def test_index_form_reflects_saved_default(client):
    # Save a default, then a fresh index render should pre-fill it.
    client.post("/config/default", json={"overrides": {"stage.search.max_rounds": 11}})
    body = create_app() and TestClient(create_app()).get("/").text
    import re

    m = re.search(r'name="stage\.search\.max_rounds"[^>]*', body)
    assert m and 'value="11"' in m.group(0)


def test_save_default_persists_only_changed_fields(client):
    resp = client.post(
        "/config/default", json={"overrides": {"stage.search.max_rounds": 7}}
    )
    assert resp.status_code == 200 and resp.json() == {"ok": True}
    saved = load_default_config()
    assert saved["stage"]["search"]["max_rounds"] == 7


def test_save_default_rejects_out_of_bounds(client):
    resp = client.post(
        "/config/default", json={"overrides": {"stage.search.max_rounds": 999}}
    )
    assert resp.status_code == 400
    assert "Invalid configuration" in resp.json()["error"]
    assert load_default_config() is None  # nothing persisted on rejection


def test_save_empty_overrides_is_ok(client):
    resp = client.post("/config/default", json={"overrides": {}})
    assert resp.status_code == 200
