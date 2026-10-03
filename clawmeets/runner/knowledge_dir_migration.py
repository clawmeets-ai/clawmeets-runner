# SPDX-License-Identifier: MIT
"""
clawmeets/runner/knowledge_dir_migration.py — one-time move of a legacy
knowledge folder into the agent's own home.

A relative ``local_settings.knowledge_dir`` used to resolve against the
owner's config folder (``~/.clawmeets/config/<username>/``), the base the
retired ``clawmeets init`` wizard wrote ``CLAUDE.md`` profiles into. It now
resolves against AGENT_DIR (see ``FileUtil.resolve_local_dir``), so a card
still holding e.g. ``./career_coach`` would silently point at an empty
folder. ``migrate_legacy_knowledge_dir`` runs once at runner start, before
the value is resolved, and keeps the agent's files reachable:

- Old folder exists, ``AGENT_DIR/knowledge`` does not, and no sibling agent
  of the same owner names the same relative path → the folder is MOVED to
  ``AGENT_DIR/knowledge`` and the card is rewritten to ``./knowledge``.
- Old folder exists but is shared with a sibling, or ``AGENT_DIR/knowledge``
  is already taken → nothing moves; the card is pinned to the old folder's
  absolute path so neither agent loses files.
- Otherwise (absolute / ``~`` value, or no old folder) → no-op.

The caller persists the returned value to the server so an
AGENT_SETTINGS_CHANGE echo doesn't restore the stale relative path.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

KNOWLEDGE_SUBDIR = "knowledge"
DEFAULT_KNOWLEDGE_DIR = f"./{KNOWLEDGE_SUBDIR}"


def _is_relative(raw: str) -> bool:
    return bool(raw) and not raw.startswith(("~", "/"))


def _sibling_shares(agent_dir: Path, owner_username: str, legacy_base: Path, old: Path) -> bool:
    """True when another of the owner's agents names the same legacy folder,
    either still by its relative path or already pinned to its absolute one
    (a sibling that migrated first) — moving it would strand that sibling."""
    prefix = f"{owner_username}-"
    target = old.resolve()
    for entry in agent_dir.parent.iterdir():
        if entry == agent_dir or not entry.is_dir() or not entry.name.startswith(prefix):
            continue
        try:
            card = json.loads((entry / "card.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        other = (card.get("local_settings") or {}).get("knowledge_dir") or ""
        if not other:
            continue
        other_path = legacy_base / other if _is_relative(other) else Path(other).expanduser()
        if other_path.resolve() == target:
            return True
    return False


def _prune_empty_parents(path: Path, stop: Path) -> None:
    """Remove now-empty folders from ``path`` up to (not including) ``stop``."""
    for parent in path.parents:
        if parent == stop or stop not in parent.parents:
            return
        try:
            parent.rmdir()
        except OSError:
            return


def migrate_legacy_knowledge_dir(
    agent_dir: Path,
    owner_username: str,
    legacy_base: Path,
) -> Optional[str]:
    """Move or pin a legacy knowledge folder; return the new card value.

    Returns None when the card was left alone. On a change, card.json's
    ``local_settings.knowledge_dir`` has already been rewritten.
    """
    card_path = agent_dir / "card.json"
    card = json.loads(card_path.read_text())
    local_settings = card.get("local_settings") or {}
    raw = local_settings.get("knowledge_dir") or ""
    if not _is_relative(raw):
        return None
    old = legacy_base / raw
    if not old.is_dir():
        return None

    new = agent_dir / KNOWLEDGE_SUBDIR
    if new.exists() or _sibling_shares(agent_dir, owner_username, legacy_base, old):
        value = str(old.resolve())
        logger.warning(
            f"knowledge_dir {raw!r}: legacy folder {old} is shared or "
            f"{new} already exists; pinning card to {value}"
        )
    else:
        shutil.move(str(old), str(new))
        _prune_empty_parents(old, legacy_base)
        value = DEFAULT_KNOWLEDGE_DIR
        logger.info(f"knowledge_dir {raw!r}: moved {old} -> {new}")

    local_settings["knowledge_dir"] = value
    card["local_settings"] = local_settings
    card_path.write_text(json.dumps(card, indent=2, default=str))
    return value
