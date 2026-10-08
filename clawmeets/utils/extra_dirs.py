# SPDX-License-Identifier: MIT
"""
clawmeets/utils/extra_dirs.py
The agent's extra directories — existing folders on the agent's computer
(``local_settings.extra_dirs``) that the agent works in alongside its
per-project sandbox, typically a tool checkout such as OpenMontage.

The sandbox stays the cwd; each extra directory is wired into every
provider the same way, so none of it depends on one CLI's auto-loading:

- **Access**: appended to the invocation's ``additional_dirs`` (Claude/Codex
  ``--add-dir``, Gemini ``--include-directories``, agy ``--add-dir``, the
  in-process providers' read roots; opencode reads absolute paths freely).
- **Skills**: the folder's own ``.agents/skills`` and ``.claude/skills`` join
  ``skill_source_dirs`` as the LOWEST layer, so they land in every provider's
  materialized skill tree and never shadow a clawmeets skill of the same name.
- **Instructions**: the prompt names the folder's instruction file
  (``AGENTS.md`` / ``CLAUDE.md`` / ``GEMINI.md``) and tells the agent to read
  it. Claude also auto-loads ``CLAUDE.md`` from added dirs; the prompt line
  is what carries it to the other providers.

Paths resolve exactly like ``knowledge_dir`` (``FileUtil.resolve_local_dir``).
A folder that does not exist is dropped with a warning — never created, since
it names something the user owns.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .file_io import FileUtil

logger = logging.getLogger(__name__)

EXTRA_DIRS_KEY = "extra_dirs"

# Checked in order; the first one present is the folder's instruction file.
# AGENTS.md first: it is the cross-tool convention, and CLAUDE.md is usually
# a Claude-specific copy or pointer to the same content.
INSTRUCTION_FILENAMES = ("AGENTS.md", "CLAUDE.md", "GEMINI.md")

_SKILL_SUBDIRS = (Path(".agents") / "skills", Path(".claude") / "skills")


def entries(local_settings: dict) -> list[str]:
    """``local_settings.extra_dirs`` → the path strings as written. Accepts a
    list (the canonical form) or one string with one path per line."""
    value = local_settings.get(EXTRA_DIRS_KEY)
    if isinstance(value, str):
        items = value.splitlines()
    elif isinstance(value, (list, tuple)):
        items = [v for v in value if isinstance(v, str)]
    else:
        return []
    return [s.strip() for s in items if s.strip()]


def resolve_extra_dirs(agent_dir: Path, local_settings: dict) -> list[Path]:
    """Settings → existing absolute folders, deduped, in the order given."""
    dirs: list[Path] = []
    for raw in entries(local_settings):
        path = FileUtil.resolve_local_dir(raw, agent_dir)
        if path is None:
            continue
        path = path.absolute()
        if not path.is_dir():
            logger.warning(f"Extra directory {raw!r} ({path}) is not a folder; skipping it")
            continue
        if path not in dirs:
            dirs.append(path)
    return dirs


def skill_roots(extra_dirs: list[Path]) -> list[Path]:
    """The extra directories' own skill folders that exist."""
    return [d / sub for d in extra_dirs for sub in _SKILL_SUBDIRS if (d / sub).is_dir()]


def instruction_file(extra_dir: Path) -> Optional[Path]:
    """The folder's instruction file, or None when it has none."""
    for name in INSTRUCTION_FILENAMES:
        if (extra_dir / name).is_file():
            return extra_dir / name
    return None
