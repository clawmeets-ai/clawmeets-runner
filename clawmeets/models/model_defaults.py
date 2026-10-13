# SPDX-License-Identifier: MIT
"""
clawmeets/models/model_defaults.py

A user's DEFAULT model config — what every new agent starts with.

The record is deliberately CARD-SHAPED::

    {
        "model_configs": [ {name, provider, model, base_url, source, created_at}, ... ],
        "default_model_config_name": "<name>" | None,
        "local_settings": {"smart_max_attempts": int, "smart_cooldown_seconds": int},
        "updated_at": "<iso>",
    }

so every operation in :mod:`clawmeets.models.model_config` (add, update,
remove, set_default, reorder, set_routing_settings, list_configs) works on it
unchanged. There is one definition of a valid config; this module only adds
the v1 rule on top: **no API keys at the user level** (provider, model and
base URL only — a keyed provider gets its key on the agent).

Agents get a COPY at registration (:func:`seed_card_from_default`); nothing
follows the default live. "Apply to all my agents" (:func:`apply_default_to_card`)
re-copies it onto an existing card on request.

Persisted one file per user at ``{server_dir}/model-defaults/<user_id>.json``.
"""
from __future__ import annotations

import copy
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

from clawmeets.models import model_config as mc
from clawmeets.utils.file_io import FileUtil

# A record's on-disk directory, relative to the server data dir.
MODEL_DEFAULTS_DIR = "model-defaults"

# Agents in this team exist to be one specific model; "apply to all" skips them
# unless the caller names them. Mirrors ``provider_agents.PROVIDER_TEAM``.
DEFAULT_EXCLUDED_TEAMS: tuple[str, ...] = ("Model agents",)

# The name the installer and the web UI give the router config.
SMART_CONFIG_NAME = "Smart routing"

_USER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_ROUTING_KEYS = ("smart_max_attempts", "smart_cooldown_seconds")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def empty_record() -> dict:
    return {"model_configs": [], "default_model_config_name": None, "local_settings": {}}


def check_config(cfg: dict) -> None:
    """Raise ValueError if one config breaks the v1 rule: no ``api_key``, and
    no keyed provider (it could not run without one)."""
    if cfg.get("api_key"):
        raise ValueError("the default model config cannot store an API key")
    if cfg.get("provider") in mc.KEYED_PROVIDERS:
        raise ValueError(
            f"provider '{cfg.get('provider')}' needs an API key, which the "
            "default model config cannot store; add it on the agent instead"
        )


def validate_record(record: dict) -> dict:
    """Raise ValueError if any config in ``record`` breaks the v1 rule.
    Returns ``record`` so callers can chain."""
    for cfg in mc._get_list(record):
        check_config(cfg)
    return record


def build_record(
    configs: list[dict],
    default: Optional[str] = None,
    *,
    max_attempts: Optional[int] = None,
    cooldown_seconds: Optional[int] = None,
) -> dict:
    """A validated record from wire-shape configs (the PUT body / installer).

    Goes through ``model_config.add_config`` so names, providers and
    duplicates are checked by the same code agents use."""
    record = empty_record()
    for cfg in configs:
        check_config(cfg)
        mc.add_config(record, cfg, source="default")
    if default:
        mc.set_default(record, default)
    mc.set_routing_settings(
        record, max_attempts=max_attempts, cooldown_seconds=cooldown_seconds,
    )
    return validate_record(record)


def is_configured(record: Optional[dict]) -> bool:
    return bool(record and mc._get_list(record))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _path(base_dir: Path, user_id: str) -> Path:
    if not _USER_ID_RE.match(user_id or ""):
        raise ValueError(f"invalid user id {user_id!r}")
    return Path(base_dir) / MODEL_DEFAULTS_DIR / f"{user_id}.json"


def load(base_dir: Path, user_id: str) -> Optional[dict]:
    """The user's record, or None when they never set one."""
    data = FileUtil.read(_path(base_dir, user_id), "json")
    return data if isinstance(data, dict) else None


def save(base_dir: Path, user_id: str, record: dict) -> dict:
    validate_record(record)
    record["updated_at"] = datetime.now(UTC).isoformat()
    FileUtil.write(_path(base_dir, user_id), record, "json")
    return record


def delete(base_dir: Path, user_id: str) -> bool:
    path = _path(base_dir, user_id)
    if not path.exists():
        return False
    FileUtil.delete(path)
    return True


# ---------------------------------------------------------------------------
# Copying onto agents
# ---------------------------------------------------------------------------

def _copied_configs(record: dict) -> list[dict]:
    out = []
    for cfg in mc._get_list(record):
        entry = copy.deepcopy(cfg)
        entry.pop("api_key", None)
        entry["source"] = "default"
        out.append(entry)
    return out


def _copy_routing(card: dict, record: dict) -> None:
    src = record.get("local_settings") or {}
    ls = dict(card.get("local_settings") or {})
    for key in _ROUTING_KEYS:
        if src.get(key) is not None:
            ls[key] = src[key]
        else:
            ls.pop(key, None)
    card["local_settings"] = ls


def seed_card_from_default(card: dict, record: dict) -> dict:
    """Copy the record onto a FRESH card (registration only): its configs in
    order, its default and its routing knobs, then re-mirror the runner-facing
    ``local_settings.llm_*`` / ``smart_routes``."""
    card["model_configs"] = _copied_configs(record)
    card["default_model_config_name"] = record.get("default_model_config_name")
    _copy_routing(card, record)
    mc.apply_mirror(card)
    return card


def _is_keyed(cfg: dict) -> bool:
    return bool(cfg.get("api_key")) or cfg.get("provider") in mc.KEYED_PROVIDERS


def _signature(card: dict) -> tuple:
    ls = card.get("local_settings") or {}
    return (
        tuple(
            (c.get("name"), c.get("provider"), c.get("model") or None,
             c.get("base_url") or None, bool(c.get("api_key")))
            for c in mc._get_list(card)
        ),
        card.get("default_model_config_name"),
        tuple(ls.get(k) for k in _ROUTING_KEYS),
    )


def apply_default_to_card(card: dict, record: dict) -> dict:
    """Re-copy the record onto an existing card ("apply to all my agents").

    The card's keyless configs are replaced by the record's. Its KEYED
    configs (a pasted API key) are kept, placed after the record's, so a key
    the user pasted is never deleted — it becomes the last smart-routing
    fallback. A kept config whose name collides with one of the record's is
    renamed (``name-2``). Mutates ``card``; returns
    ``{"changed": bool, "removed": [...], "kept": [...]}`` so the caller can
    say exactly what moved.
    """
    before = _signature(card)
    old = mc._get_list(card)
    kept = [dict(c) for c in old if _is_keyed(c)]
    removed = [c.get("name") for c in old if not _is_keyed(c)]

    card["model_configs"] = _copied_configs(record)
    for cfg in kept:
        cfg["name"] = mc._unique_name(card, cfg.get("name") or cfg.get("provider"))
        card["model_configs"].append(cfg)
    card["default_model_config_name"] = record.get("default_model_config_name")
    _copy_routing(card, record)
    mc.apply_mirror(card)

    new_names = {(c.get("name") or "").lower() for c in mc._get_list(card)}
    return {
        "changed": _signature(card) != before,
        "removed": [n for n in removed if (n or "").lower() not in new_names],
        "kept": [c["name"] for c in kept],
    }


# ---------------------------------------------------------------------------
# Installer
# ---------------------------------------------------------------------------

def build_installer_default(providers: list[str], *, smart: bool) -> dict:
    """The record the installer writes: one config per provider, named after
    it, in the given (priority) order. With ``smart`` and two or more
    providers, a router config is added in front and made the default; with
    one provider, that provider is the default."""
    configs: list[dict] = []
    if smart and len(providers) >= 2:
        configs.append({"name": SMART_CONFIG_NAME, "provider": "smart"})
    configs.extend({"name": p, "provider": p} for p in providers)
    default = configs[0]["name"] if configs else None
    return build_record(configs, default)
