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

# A generous ceiling for one value set from the web UI (a PEM key fits), small
# enough that a pasted file cannot ride a websocket frame into someone's env.
VALUE_MAX_BYTES = 8192

_STORE_FILENAME = "env.json"

# Sidecar listing the keys an owner opted into scoped (non-owner) turns with
# ``clawmeets env set KEY VALUE --scoped-turns``. Kept apart from env.json so the
# store stays a flat ``{KEY: value}`` object every existing reader understands.
# Absent / unreadable => no key reaches a scoped turn (fail closed).
_SCOPED_FILENAME = "env.scoped-turns.json"


def store_path(agent_dir: Path) -> Path:
    """Path to the store file for an agent (``{agent_dir}/env.json``)."""
    return Path(agent_dir) / _STORE_FILENAME


def scoped_store_path(agent_dir: Path) -> Path:
    """Path to the scoped-turns key list (``{agent_dir}/env.scoped-turns.json``)."""
    return Path(agent_dir) / _SCOPED_FILENAME


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


def key_names(agent_dir: Path) -> list[str]:
    """Sorted key names of the effective store. Never the values."""
    return sorted(load(agent_dir))


def set_var(agent_dir: Path, key: str, value: str) -> None:
    """Validate ``key``, set/overwrite it in the store, and persist atomically.

    Writes the whole store with an atomic temp-file rename, then chmods the file
    to 0600 (owner read/write only) so secrets are not world-readable.
    """
    validate_key(key)
    data = read_raw(agent_dir)
    data[key] = value
    _write(agent_dir, data)


def scoped_keys(agent_dir: Path) -> list[str]:
    """Sorted names of the stored keys marked for scoped turns.

    Only names that are still in the effective store count, so unsetting a key
    can never leave a stale opt-in behind.
    """
    data = FileUtil.read(scoped_store_path(agent_dir), "json")
    if not isinstance(data, list):
        return []
    present = load(agent_dir)
    return sorted({str(k) for k in data if str(k) in present})


def load_scoped(agent_dir: Path) -> dict[str, str]:
    """The env overlay for a scoped turn: only keys marked ``--scoped-turns``."""
    store = load(agent_dir)
    return {k: store[k] for k in scoped_keys(agent_dir) if k in store}


def set_scoped(agent_dir: Path, key: str, scoped: bool) -> None:
    """Mark (or unmark) ``key`` as visible to scoped turns."""
    path = scoped_store_path(agent_dir)
    data = FileUtil.read(path, "json")
    keys = {str(k) for k in data} if isinstance(data, list) else set()
    if scoped:
        validate_key(key)
        keys.add(key)
    else:
        keys.discard(key)
    if keys:
        FileUtil.write(path, sorted(keys), "json", atomic=True)
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    else:
        FileUtil.delete(path)


def unset_var(agent_dir: Path, key: str) -> bool:
    """Remove ``key`` from the store. Return ``True`` if it existed.

    The store file is unlinked when it becomes empty so ``env list`` on a
    cleared agent shows nothing rather than an empty ``{}`` artifact.
    """
    set_scoped(agent_dir, key, False)
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
