# SPDX-License-Identifier: MIT
"""
clawmeets/models/share_token.py

Share token management for read-only project sharing via URL links.
Tokens are stored in {data_dir}/share_tokens as JSON.
"""
from __future__ import annotations

import asyncio
import secrets
import threading
from datetime import UTC, datetime
from pathlib import Path

from clawmeets.utils.file_io import FileUtil

_lock = asyncio.Lock()

SHARE_TOKENS_FILE = "share_tokens"


def _tokens_path(data_dir: Path) -> Path:
    return Path(data_dir) / SHARE_TOKENS_FILE


def _load(data_dir: Path) -> dict:
    path = _tokens_path(data_dir)
    data = FileUtil.read(path, "json")
    if data is None:
        return {"tokens": {}}
    return data


def _save(data: dict, data_dir: Path) -> None:
    FileUtil.write(_tokens_path(data_dir), data, "json")


async def generate_share_token(project_id: str, created_by: str, data_dir: Path) -> str:
    """Generate a share token for a project.

    Returns the newly generated token string.
    """
    async with _lock:
        data = _load(data_dir)
        existing = set(data["tokens"].keys())

        token = secrets.token_hex(8).upper()
        while token in existing:
            token = secrets.token_hex(8).upper()

        data["tokens"][token] = {
            "project_id": project_id,
            "created_by": created_by,
            "created_at": datetime.now(UTC).isoformat(),
            "viewers": [],
        }
        _save(data, data_dir)
        return token


def get_token_info(token: str, data_dir: Path) -> dict | None:
    """Look up a share token. Returns token entry dict or None."""
    if not token or not token.strip():
        return None
    data = _load(data_dir)
    return data["tokens"].get(token.strip().upper())


async def add_viewer(token: str, user_id: str, data_dir: Path) -> str | None:
    """Add a user as viewer for the token's project. Idempotent.

    Returns the project_id if successful, None if token is invalid.
    """
    if not token or not token.strip():
        return None

    async with _lock:
        data = _load(data_dir)
        entry = data["tokens"].get(token.strip().upper())
        if entry is None:
            return None
        if user_id not in entry["viewers"]:
            entry["viewers"].append(user_id)
            _save(data, data_dir)
        return entry["project_id"]


def is_viewer(project_id: str, user_id: str, data_dir: Path) -> bool:
    """Check if a user is a viewer of a project (across all tokens)."""
    data = _load(data_dir)
    for entry in data["tokens"].values():
        if entry["project_id"] == project_id and user_id in entry["viewers"]:
            return True
    return False


def get_viewers_for_project(project_id: str, data_dir: Path) -> list[str]:
    """Get all viewer user IDs for a project (deduplicated)."""
    data = _load(data_dir)
    viewers: set[str] = set()
    for entry in data["tokens"].values():
        if entry["project_id"] == project_id:
            viewers.update(entry["viewers"])
    return list(viewers)


# Memoized inverse index for `get_projects_for_viewer`, keyed on the token
# file's (absolute path, mtime_ns, size). The path is part of the key so two
# data dirs can never alias each other on a coincidental (mtime, size) match,
# and any write through `_save` moves the key, so the cache self-invalidates.
_viewer_index_key: tuple[str, int, int] | None = None
_viewer_index: dict[str, frozenset[str]] = {}
_viewer_index_lock = threading.Lock()


def get_projects_for_viewer(user_id: str, data_dir: Path) -> frozenset[str]:
    """Project IDs ``user_id`` can read as a share-token viewer.

    The inverse of :func:`get_viewers_for_project`, and memoized: this is on the
    chat-search query path, where it is the ONLY filesystem touch left (see
    ``server/chat_search_index``'s module docstring on the ACL mirror). Steady
    state is one ``stat()`` and zero parses.
    """
    global _viewer_index_key, _viewer_index

    path = _tokens_path(data_dir)
    try:
        st = path.stat()
    except OSError:
        return frozenset()
    key = (str(path), st.st_mtime_ns, st.st_size)
    with _viewer_index_lock:
        if _viewer_index_key != key:
            by_user: dict[str, set[str]] = {}
            for entry in (_load(data_dir).get("tokens") or {}).values():
                project_id = entry.get("project_id")
                if not project_id:
                    continue
                for viewer_id in entry.get("viewers") or []:
                    by_user.setdefault(viewer_id, set()).add(project_id)
            _viewer_index_key = key
            _viewer_index = {k: frozenset(v) for k, v in by_user.items()}
        return _viewer_index.get(user_id, frozenset())


async def remove_viewer(project_id: str, user_id: str, data_dir: Path) -> bool:
    """Remove a user from all viewer lists for a project.

    Returns True if the user was found and removed.
    """
    async with _lock:
        data = _load(data_dir)
        removed = False
        for entry in data["tokens"].values():
            if entry["project_id"] == project_id and user_id in entry["viewers"]:
                entry["viewers"].remove(user_id)
                removed = True
        if removed:
            _save(data, data_dir)
        return removed


async def revoke_all_for_project(project_id: str, data_dir: Path) -> None:
    """Remove all share tokens for a project (cleanup on project deletion)."""
    async with _lock:
        data = _load(data_dir)
        tokens_to_remove = [
            token for token, entry in data["tokens"].items()
            if entry["project_id"] == project_id
        ]
        for token in tokens_to_remove:
            del data["tokens"][token]
        if tokens_to_remove:
            _save(data, data_dir)
