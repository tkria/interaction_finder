"""Session-only provider API keys for the web UI.

The pipeline's LLM agents (pydantic-ai) and the OpenAI/PubMed search backends
read their credentials from environment variables. The UI lets a user supply
those keys when they are not already in the environment; we set them on
``os.environ`` in the running server process, so every run this session picks
them up. Nothing is written to disk -- the keys vanish on restart.

Public API: ``PROVIDERS``, ``key_status()``, ``set_keys(values)``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


# Order is the display order in the modal.
@dataclass(frozen=True)
class Provider:
    """One settable credential: its env var and a human label."""

    env: str  # environment variable name
    label: str  # shown in the UI


PROVIDERS: tuple[Provider, ...] = (
    Provider("OPENAI_API_KEY", "OpenAI"),
    Provider("ANTHROPIC_API_KEY", "Anthropic"),
    Provider("GEMINI_API_KEY", "Gemini"),
    Provider("NCBI_API_KEY", "NCBI / PubMed"),
)

_BY_ENV = {p.env: p for p in PROVIDERS}


def key_status() -> list[dict]:
    """Report each provider's env var and whether it is currently set.

    Never returns the secret value -- only ``{env, label, set}`` per provider,
    so the UI can show set/unset without echoing a key back to the browser.
    """
    return [
        {"env": p.env, "label": p.label, "set": bool(os.environ.get(p.env))}
        for p in PROVIDERS
    ]


def set_keys(values: dict[str, str]) -> list[str]:
    """Set the given provider keys on the process environment for this session.

    Parameters:
        values: Map of env-var name -> key value. Only recognised provider env
            vars are honoured; blank/whitespace values are ignored (a field
            left empty never clears an existing key).

    Returns the list of env-var names that were set, for confirmation.
    """
    set_envs = []
    for env, value in values.items():
        if env not in _BY_ENV:
            continue
        value = (value or "").strip()
        if not value:
            continue
        os.environ[env] = value
        set_envs.append(env)
    return set_envs
