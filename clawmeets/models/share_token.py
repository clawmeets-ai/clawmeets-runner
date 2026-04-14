# SPDX-License-Identifier: BUSL-1.1
"""
clawmeets/models/share_token.py

Share token management for read-only project sharing via URL links.
Tokens are stored in {data_dir}/share_tokens as JSON.
"""
from __future__ import annotations

import asyncio
import secrets
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
