"""Session-only provider credentials for the web UI.

The pipeline's LLM agents (pydantic-ai) and the OpenAI/PubMed search backends
read their credentials from environment variables. The UI lets a user supply
those values when they are not already in the environment; we set them on
``os.environ`` in the running server process, so every run this session picks
them up. Nothing is written to disk -- values vanish on restart.

The modal shows a *dynamic* set of rows rather than a fixed list: every provider
whose env var is currently set, every provider implicated by the models the
current config configures, NCBI always, plus any the user adds from the picker.

Providers are either *secret* (API keys, reported only as set/unset, never
echoed) or *plain* (a value safe to show and edit, e.g. a custom endpoint URL).

Public API: ``PROVIDERS``, ``key_status(config)``, ``set_keys(values)``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from interaction_finder.settings import IfetcherConfig


# Order is the display order in the modal.
@dataclass(frozen=True)
class Provider:
    """One settable credential.

    Fields:
        env: environment variable name.
        label: shown in the UI.
        secret: True for API keys (value never echoed; blank on save leaves an
            existing value untouched). False for plain values like an endpoint
            URL (value shown and editable; blank on save clears it).
        prefix: the ``provider:`` prefix of a model string this credential
            authenticates, used to decide which providers a config implicates.
            None for credentials not tied to a model provider (NCBI, endpoint).
    """

    env: str
    label: str
    secret: bool
    prefix: str | None


# Every single-key provider pydantic-ai 1.0 can authenticate from one env var,
# plus NCBI and the OpenAI-compatible endpoint override. Multi-value auth
# (Bedrock AWS chain, Azure endpoint+key+version) is intentionally absent: it
# cannot be expressed as one field. The prefix is the model-string provider tag.
PROVIDERS: tuple[Provider, ...] = (
    Provider("OPENAI_API_KEY", "OpenAI", True, "openai"),
    Provider("ANTHROPIC_API_KEY", "Anthropic", True, "anthropic"),
    Provider("GEMINI_API_KEY", "Gemini", True, "google-gla"),
    Provider("GOOGLE_API_KEY", "Google (Vertex/GenAI)", True, "google"),
    Provider("GROQ_API_KEY", "Groq", True, "groq"),
    Provider("MISTRAL_API_KEY", "Mistral", True, "mistral"),
    Provider("CO_API_KEY", "Cohere", True, "cohere"),
    Provider("DEEPSEEK_API_KEY", "DeepSeek", True, "deepseek"),
    Provider("GROK_API_KEY", "Grok (xAI)", True, "grok"),
    Provider("OPENROUTER_API_KEY", "OpenRouter", True, "openrouter"),
    Provider("TOGETHER_API_KEY", "Together", True, "together"),
    Provider("FIREWORKS_API_KEY", "Fireworks", True, "fireworks"),
    Provider("CEREBRAS_API_KEY", "Cerebras", True, "cerebras"),
    Provider("MOONSHOTAI_API_KEY", "Moonshot", True, "moonshotai"),
    Provider("HF_TOKEN", "Hugging Face", True, "huggingface"),
    Provider("NCBI_API_KEY", "NCBI / PubMed", True, None),
    Provider("OPENAI_BASE_URL", "Custom OpenAI endpoint", False, None),
)

_BY_ENV = {p.env: p for p in PROVIDERS}
_BY_PREFIX = {p.prefix: p for p in PROVIDERS if p.prefix}
# Rows that appear even when unset and unimplicated.
_ALWAYS_SHOWN = frozenset({"NCBI_API_KEY"})


def key_status(config: IfetcherConfig | None = None) -> dict:
    """Report which credential rows the modal shows and which it offers to add.

    A provider is *shown* when its env var is set, when a model the config
    configures implicates it, or when it is always shown (NCBI). Everything else
    the registry knows is *available* to add from the picker.

    Parameters:
        config: the effective config to resolve implicated providers from; when
            None, only set + always-shown providers are shown.

    Returns:
        ``{"shown": [row, ...], "available": [{env, label}, ...]}`` where each
        row is ``{env, label, secret, prefix, set}`` and additionally carries
        ``value`` for non-secret providers (secrets are never echoed).
    """
    implicated = implicated_envs(config) if config is not None else set()
    shown_envs = {
        p.env
        for p in PROVIDERS
        if os.environ.get(p.env) or p.env in implicated or p.env in _ALWAYS_SHOWN
    }
    shown = [_row(p) for p in PROVIDERS if p.env in shown_envs]
    available = [
        {"env": p.env, "label": p.label} for p in PROVIDERS if p.env not in shown_envs
    ]
    return {"shown": shown, "available": available}


def set_keys(values: dict[str, str]) -> None:
    """Apply provider values to the process environment for this session.

    Parameters:
        values: map of env-var name -> value. Only recognised providers are
            honoured. For *secret* providers a blank value is ignored (a field
            left empty never clears an existing key, since a hidden secret
            cannot be re-typed). For *plain* providers a blank value clears the
            env var (the visible field can legitimately be emptied).
    """
    for env, raw in values.items():
        provider = _BY_ENV.get(env)
        if provider is None:
            continue
        value = (raw or "").strip()
        if value:
            os.environ[env] = value
        elif not provider.secret:
            os.environ.pop(env, None)


def implicated_envs(config: IfetcherConfig) -> set[str]:
    """Return env vars for the providers the config's configured models need.

    Collects every ``llm`` string appearing anywhere under ``config.agents``
    plus the effective global default -- the code default only when no
    ``agents._.llm`` overrides it -- then maps each ``provider:`` prefix to its
    registry env var. Unknown prefixes (e.g. a ``bedrock:`` model with no
    single-key row) are skipped.

    Walking the config's own agent entries -- rather than a hardcoded agent list
    -- keeps this correct as agents are added: a non-default provider can only
    appear via an explicit ``agents.*.llm`` value.
    """
    global_default = config.resolve_agent_config("").llm or _CODE_DEFAULT_MODEL
    models = {global_default}
    models.update(_agent_llm_strings(config.agents))
    envs = set()
    for model in models:
        provider = _BY_PREFIX.get(model.split(":", 1)[0])
        if provider:
            envs.add(provider.env)
    return envs


# The default_model baked into agent_getter(); agents without an override use it.
_CODE_DEFAULT_MODEL = "openai:gpt-4o-mini"


def _row(provider: Provider) -> dict:
    """Build one status row; include the value only for non-secret providers."""
    current = os.environ.get(provider.env)
    row = {
        "env": provider.env,
        "label": provider.label,
        "secret": provider.secret,
        "prefix": provider.prefix,
        "set": bool(current),
    }
    if not provider.secret:
        row["value"] = current or ""
    return row


def _agent_llm_strings(
    agents: dict | IfetcherConfig.AgentSpec,
) -> set[str]:
    """Collect every non-None ``llm`` from the nested agents config structure.

    ``agents`` is a single AgentSpec, or a dict whose values are AgentSpecs
    (module defaults) or dicts of AgentSpecs (per-agent overrides).
    """
    if isinstance(agents, IfetcherConfig.AgentSpec):
        return {agents.llm} if agents.llm else set()
    models = set()
    for value in agents.values():
        if isinstance(value, IfetcherConfig.AgentSpec):
            if value.llm:
                models.add(value.llm)
        elif isinstance(value, dict):
            models.update(spec.llm for spec in value.values() if spec.llm)
    return models
