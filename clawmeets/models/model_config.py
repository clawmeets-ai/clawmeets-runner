# SPDX-License-Identifier: MIT
"""
clawmeets/models/model_config.py

Named model configs — the data behind the "model selector" revamp.

An agent's card carries a list of named model configs plus a designated
default:

    card["model_configs"]              = [ {name, provider, model, base_url,
                                            source, created_at}, ... ]
    card["default_model_config_name"]  = "<name>" | None

Every function in this module takes the *raw card dict*, mutates its
``model_configs`` / ``default_model_config_name`` in place, and (for the
mutating ops) returns the affected entry / card. Persistence is the caller's
job — routes call ``agent.update_card(...)`` under the agent write lock.

The module is pure (no clawmeets imports, only stdlib), so both the server
routes and the runner invoke path can use it. ``VALID_CONFIG_PROVIDERS`` is the
single source of truth for the provider enum; ``cli_runner`` re-exports it as
``_VALID_LLM_PROVIDERS`` so the CLI and the config API validate against the same
list.
"""
from __future__ import annotations

import re
import shutil
from datetime import UTC, datetime
from typing import Optional

# Single source of truth for provider values. Bare names shell the Code CLI
# binary; the ``-api`` suffix selects the in-process BYO-key provider. Kept
# here (Layer 1, pure) and re-exported by cli_runner as ``_VALID_LLM_PROVIDERS``.
VALID_CONFIG_PROVIDERS: tuple[str, ...] = (
    "claude", "openai", "gemini", "opencode",
    "claude-api", "openai-api", "gemini-api", "openrouter-api",
    # In-process, BYO-key, native OpenRouter tool-loop (no Pydantic-AI). Coexists
    # with openrouter-api as a directly A/B-comparable alternative.
    "openrouter-native",
)

# Config name: 1–64 chars, alphanumerics + space / underscore / hyphen.
NAME_RE = re.compile(r"^[A-Za-z0-9 _-]{1,64}$")

# Registration probe (spec #4): the 3 required providers → their Code CLI
# binary. NOTE: provider ``openai`` shells the ``codex`` binary (the factory
# maps openai → CodexCLI), so we probe for ``codex``. Priority order (first
# installed becomes the initial default): claude > openai > gemini.
_PROBE_BINARIES: tuple[tuple[str, str], ...] = (
    ("claude", "claude"),
    ("openai", "codex"),
    ("gemini", "gemini"),
)


class DuplicateConfigError(ValueError):
    """Raised when a create/rename would collide with an existing name."""


class ConfigNotFoundError(ValueError):
    """Raised when a named config does not exist on the card."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _name_eq(a: object, b: object) -> bool:
    """Case-insensitive name comparison (names are user-facing labels)."""
    return (
        isinstance(a, str)
        and isinstance(b, str)
        and a.strip().lower() == b.strip().lower()
    )


def _get_list(card: dict) -> list[dict]:
    """Return the card's model_configs as a list (never mutates the card)."""
    raw = card.get("model_configs")
    return list(raw) if isinstance(raw, list) else []


def _validate_name(name: object) -> None:
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ValueError(
            f"invalid config name {name!r}: must be 1–64 chars "
            r"matching ^[A-Za-z0-9 _-]+$"
        )


def _validate_provider(provider: object) -> None:
    if provider not in VALID_CONFIG_PROVIDERS:
        raise ValueError(
            f"invalid provider {provider!r}: expected one of "
            f"{VALID_CONFIG_PROVIDERS}"
        )


def _to_wire(entry: dict) -> dict:
    """Normalize a stored entry to the canonical wire shape (all 6 keys)."""
    return {
        "name": entry.get("name"),
        "provider": entry.get("provider"),
        "model": entry.get("model") or None,
        "base_url": entry.get("base_url") or None,
        "source": entry.get("source") or "manual",
        "created_at": entry.get("created_at") or _now_iso(),
    }


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def list_configs(card: dict) -> tuple[list[dict], Optional[str]]:
    """Return ``(configs, default_name)`` in wire shape.

    ``default_name`` is coerced to None if it points at a config that no longer
    exists (defensive — should not happen once mutations keep it consistent).
    """
    configs = [_to_wire(c) for c in _get_list(card)]
    default = card.get("default_model_config_name")
    if default is not None and not any(_name_eq(c["name"], default) for c in configs):
        default = None
    return configs, default


def resolve(card: dict, name: Optional[str]) -> Optional[dict]:
    """Resolve an override to a config dict.

    Returns the config matching ``name`` (case-insensitive); if ``name`` is
    falsy or unknown, falls back to the default config; if there is no default,
    returns None. Never raises — a stale selector value degrades to the default
    rather than breaking a turn.
    """
    configs = _get_list(card)
    if name:
        match = next((c for c in configs if _name_eq(c.get("name"), name)), None)
        if match is not None:
            return _to_wire(match)
    default = card.get("default_model_config_name")
    if default:
        match = next((c for c in configs if _name_eq(c.get("name"), default)), None)
        if match is not None:
            return _to_wire(match)
    return None


# ---------------------------------------------------------------------------
# Mutations (mutate card in place)
# ---------------------------------------------------------------------------

def add_config(card: dict, cfg: dict, *, source: str = "manual") -> dict:
    """Append a new config. Raises DuplicateConfigError on a name collision.

    ``source`` / ``created_at`` are stamped by the server (read-only on the
    wire). If this is the FIRST config, it also becomes the default.
    """
    name = cfg.get("name")
    provider = cfg.get("provider")
    _validate_name(name)
    _validate_provider(provider)

    configs = _get_list(card)
    if any(_name_eq(c.get("name"), name) for c in configs):
        raise DuplicateConfigError(f"a config named {name!r} already exists")

    entry = {
        "name": name,
        "provider": provider,
        "model": cfg.get("model") or None,
        "base_url": cfg.get("base_url") or None,
        "source": source,
        "created_at": cfg.get("created_at") or _now_iso(),
    }
    configs.append(entry)
    card["model_configs"] = configs
    if not card.get("default_model_config_name"):
        card["default_model_config_name"] = name
    return _to_wire(entry)


def update_config(card: dict, name: str, patch: dict) -> dict:
    """Apply a partial patch to a config by name. Renames are allowed.

    ``patch`` carries only the fields the client provided (``model``/``base_url``
    may be explicitly null to clear). ``source`` / ``created_at`` are read-only
    and never patched. Raises ConfigNotFoundError if ``name`` is absent, or
    DuplicateConfigError if a rename collides with another config.
    """
    configs = _get_list(card)
    idx = next(
        (i for i, c in enumerate(configs) if _name_eq(c.get("name"), name)),
        None,
    )
    if idx is None:
        raise ConfigNotFoundError(f"no config named {name!r}")

    current = dict(configs[idx])
    old_name = current["name"]

    if "name" in patch:
        _validate_name(patch["name"])
        if not _name_eq(patch["name"], old_name) and any(
            _name_eq(c.get("name"), patch["name"])
            for j, c in enumerate(configs)
            if j != idx
        ):
            raise DuplicateConfigError(
                f"a config named {patch['name']!r} already exists"
            )
        current["name"] = patch["name"]
    if "provider" in patch:
        _validate_provider(patch["provider"])
        current["provider"] = patch["provider"]
    if "model" in patch:
        current["model"] = patch["model"] or None
    if "base_url" in patch:
        current["base_url"] = patch["base_url"] or None

    configs[idx] = current
    card["model_configs"] = configs

    # Follow the rename through in the default pointer.
    if "name" in patch and _name_eq(card.get("default_model_config_name"), old_name):
        card["default_model_config_name"] = current["name"]
    return _to_wire(current)


def remove_config(card: dict, name: str) -> None:
    """Delete a config by name. Delete is ALWAYS allowed (never refuses).

    Raises ConfigNotFoundError only when ``name`` is absent. Delete rules:
      - deleting the default → auto-promote the FIRST remaining config;
      - deleting the last config → ``default_model_config_name = None``.
    """
    configs = _get_list(card)
    idx = next(
        (i for i, c in enumerate(configs) if _name_eq(c.get("name"), name)),
        None,
    )
    if idx is None:
        raise ConfigNotFoundError(f"no config named {name!r}")

    removed = configs.pop(idx)
    card["model_configs"] = configs
    if _name_eq(card.get("default_model_config_name"), removed.get("name")):
        card["default_model_config_name"] = configs[0]["name"] if configs else None


def set_default(card: dict, name: str) -> None:
    """Point the default at an existing config. Raises ConfigNotFoundError."""
    configs = _get_list(card)
    match = next((c for c in configs if _name_eq(c.get("name"), name)), None)
    if match is None:
        raise ConfigNotFoundError(f"no config named {name!r}")
    card["default_model_config_name"] = match["name"]


# ---------------------------------------------------------------------------
# Runner-path mirror (spec #7)
# ---------------------------------------------------------------------------

def effective_local_settings(card: dict) -> dict:
    """Derive the singular ``llm_*`` fields the runner reads from the default.

    Returns a patch to splat over ``card["local_settings"]`` so the existing
    startup + AGENT_SETTINGS_CHANGE paths keep resolving the current default's
    provider/model/base_url. Always returns all three keys (None when there is
    no default) so switching away from a base_url config clears the stale value.
    """
    cfg = resolve(card, None)
    if cfg is None:
        return {"llm_provider": None, "llm_model": None, "llm_base_url": None}
    return {
        "llm_provider": cfg.get("provider"),
        "llm_model": cfg.get("model"),
        "llm_base_url": cfg.get("base_url"),
    }


def apply_mirror(card: dict) -> dict:
    """Splat ``effective_local_settings`` into ``card["local_settings"]``.

    Convenience used by the mutating routes so every config change re-mirrors
    the default into the runner-facing ``llm_*`` fields under the write lock.
    """
    ls = dict(card.get("local_settings") or {})
    ls.update(effective_local_settings(card))
    card["local_settings"] = ls
    return ls


# ---------------------------------------------------------------------------
# Registration probe (spec #4)
# ---------------------------------------------------------------------------

def probe_available_providers() -> list[dict]:
    """Return an auto ModelConfig dict for each Code CLI installed on this box.

    Detection = ``shutil.which(binary) is not None`` (the same signal each
    provider's ``verify_cli`` uses). Cheap (no subprocess). Each available
    provider yields ``{name, provider, model: None, base_url: None}`` in
    priority order claude > openai > gemini.
    """
    out: list[dict] = []
    for provider, binary in _PROBE_BINARIES:
        if shutil.which(binary):
            out.append(
                {"name": provider, "provider": provider, "model": None, "base_url": None}
            )
    return out


def reconcile_default_with_local_settings(card: dict) -> dict:
    """Keep the named-config default consistent with ``local_settings.llm_*``.

    When the legacy settings path sets ``local_settings.llm_provider`` (e.g.
    ``--llm-provider`` at register, or the old provider dropdown), ensure a
    matching named config exists and is the default, then re-mirror. Does NOT
    probe (so it never resurrects a deleted auto config). No-op when
    ``llm_provider`` is unset/invalid.
    """
    ls = card.get("local_settings") or {}
    provider = (ls.get("llm_provider") or "").strip().lower()
    if provider not in VALID_CONFIG_PROVIDERS:
        return card
    model = ls.get("llm_model") or None
    base_url = ls.get("llm_base_url") or None

    configs = _get_list(card)
    match = next(
        (
            c for c in configs
            if c.get("provider") == provider and (c.get("model") or None) == model
        ),
        None,
    ) or next((c for c in configs if c.get("provider") == provider), None)

    if match is None:
        name = _unique_name(card, provider)
        add_config(
            card,
            {"name": name, "provider": provider, "model": model, "base_url": base_url},
            source="auto",
        )
        card["default_model_config_name"] = name
    else:
        card["default_model_config_name"] = match["name"]
    apply_mirror(card)
    return card


def seed_probed_configs(card: dict) -> dict:
    """Seed auto configs at registration (idempotent, non-fatal).

    Adds an auto config for each installed CLI (never clobbering a manual
    config of the same name), then reconciles the default against any explicit
    ``local_settings.llm_provider`` and re-mirrors. Safe to re-run: existing
    names are skipped and an existing default is preserved.
    """
    for seed in probe_available_providers():
        try:
            add_config(card, seed, source="auto")
        except (DuplicateConfigError, ValueError):
            continue
    reconcile_default_with_local_settings(card)
    apply_mirror(card)
    return card


def _unique_name(card: dict, base: str) -> str:
    """Return ``base``, or ``base-2``/``base-3``… if the name is taken."""
    configs = _get_list(card)
    taken = {(c.get("name") or "").strip().lower() for c in configs}
    if base.lower() not in taken:
        return base
    i = 2
    while f"{base}-{i}".lower() in taken:
        i += 1
    return f"{base}-{i}"
