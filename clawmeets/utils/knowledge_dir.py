# SPDX-License-Identifier: MIT
"""
clawmeets/utils/knowledge_dir.py

Resolve a `knowledge_dir` string from card.json local_settings into an
absolute `Path`.

At init time, `_resolve_knowledge_dir` in cli_init.py interprets relative
paths against `~/.clawmeets/config/<username>/` (the per-user output dir
where CLAUDE.md is written). The runner must use the same base so that the
agent at runtime points at the same folder the init flow prepared, rather
than a path relative to whatever shell the user happened to run
`clawmeets start` from.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def resolve_local_knowledge_dir(raw: str, user_config_dir: Optional[Path]) -> Optional[Path]:
    """Resolve a knowledge_dir string from card.json local_settings.

    - Absolute paths (`/foo/bar`) and `~`-prefixed paths are honored verbatim
      (the latter is expanded against the user's home).
    - Relative paths (`./foo`, `foo`, `../foo`) are joined to
      `user_config_dir` — the same base cli_init.py used when it wrote
      CLAUDE.md at init time. When `user_config_dir` is None, relative paths
      fall through to `Path(raw)` (legacy behavior, relative to CWD).
    - Empty strings return None.
    """
    if not raw:
        return None
    if raw.startswith("~"):
        return Path(raw).expanduser()
    if raw.startswith("/"):
        return Path(raw)
    if user_config_dir is None:
        return Path(raw)
    return user_config_dir / raw
