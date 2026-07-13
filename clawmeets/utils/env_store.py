# SPDX-License-Identifier: MIT
"""
clawmeets/utils/env_store.py — runner-local, file-based env-var store.

A per-agent store of environment variables that the runner injects into every
LLM/skill subprocess it spawns. Purpose: hold runner-specific secrets/config
(e.g. a credential a particular skill needs) on *this machine only* — never
synced to the server, never broadcast to chat, never committed to git.

Storage: a plaintext JSON object at ``{agent_dir}/env.json`` (mode 0600),
sibling of ``credential.json`` — the same class of runner-only, un-synced file.
The store is read *live* at spawn time (``LLMProvider._build_env``) so
``clawmeets env set`` takes effect on the next turn with no runner restart.

Precedence (lowest → highest), assembled in ``_build_env``:
    os.environ  <  agent store  <  CLAWMEETS_* identity (agent_env)
The reserved ``CLAWMEETS_`` prefix is rejected on write and dropped on read so
the store can never shadow / spoof agent identity vars.
"""
from __future__ import annotations

import os
import re
import stat
from pathlib import Path

from .file_io import FileUtil

# POSIX-ish env-var name: leading letter/underscore, then alnum/underscore.
KEY_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")

# Keys under this prefix carry agent identity/token and are runner-owned; the
# store must never write or surface them.
RESERVED_PREFIX = "CLAWMEETS_"

_STORE_FILENAME = "env.json"


def store_path(agent_dir: Path) -> Path:
    """Path to the store file for an agent (``{agent_dir}/env.json``)."""
    return Path(agent_dir) / _STORE_FILENAME


def validate_key(key: str) -> None:
    """Raise ``ValueError`` if ``key`` is not a legal, non-reserved env name."""
    if not KEY_RE.match(key):
        raise ValueError(
            f"invalid env var name {key!r}: must match {KEY_RE.pattern} "
            "(upper-case letters, digits, underscore; no leading digit)"
        )
    if key.startswith(RESERVED_PREFIX):
        raise ValueError(
            f"refusing to set {key!r}: the {RESERVED_PREFIX!r} prefix is "
            "reserved for agent identity and cannot be overridden by the store"
        )


def read_raw(agent_dir: Path) -> dict[str, str]:
    """Read the store verbatim. Missing or corrupt file → ``{}`` (never raises).

    Non-string / non-dict payloads are coerced away defensively so a hand-edited
    file can't crash a subprocess spawn.
    """
    data = FileUtil.read(store_path(agent_dir), "json")
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def load(agent_dir: Path) -> dict[str, str]:
    """Effective env overlay for injection at subprocess spawn.

    Drops any stray ``CLAWMEETS_*`` keys defensively (a hand-edited file could
    contain them even though ``set_var`` rejects them). Cheap — one small JSON
    read; called once per spawn so ``clawmeets env set`` lands live.
    """
    return {
        k: v
        for k, v in read_raw(agent_dir).items()
        if KEY_RE.match(k) and not k.startswith(RESERVED_PREFIX)
    }


def set_var(agent_dir: Path, key: str, value: str) -> None:
    """Validate ``key``, set/overwrite it in the store, and persist atomically.

    Writes the whole store with an atomic temp-file rename, then chmods the file
    to 0600 (owner read/write only) so secrets are not world-readable.
    """
    validate_key(key)
    data = read_raw(agent_dir)
    data[key] = value
    _write(agent_dir, data)


def unset_var(agent_dir: Path, key: str) -> bool:
    """Remove ``key`` from the store. Return ``True`` if it existed.

    The store file is unlinked when it becomes empty so ``env list`` on a
    cleared agent shows nothing rather than an empty ``{}`` artifact.
    """
    data = read_raw(agent_dir)
    if key not in data:
        return False
    del data[key]
    if data:
        _write(agent_dir, data)
    else:
        FileUtil.delete(store_path(agent_dir))
    return True


def _write(agent_dir: Path, data: dict[str, str]) -> None:
    """Atomically persist the store dict and lock perms to 0600."""
    path = store_path(agent_dir)
    FileUtil.write(path, data, "json", atomic=True)
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)  # 0600
