# SPDX-License-Identifier: MIT
"""
clawmeets/models/invitation_code.py

Invitation code management for gated self-registration.
Codes are stored in {data_dir}/invitation_codes as JSON.
"""
from __future__ import annotations

import asyncio
import secrets
from datetime import UTC, datetime
from pathlib import Path

from clawmeets.utils.file_io import FileUtil

_lock = asyncio.Lock()

INVITATION_CODES_FILE = "invitation_codes"


def _codes_path(data_dir: Path) -> Path:
    return Path(data_dir) / INVITATION_CODES_FILE


def _load(data_dir: Path) -> dict:
    path = _codes_path(data_dir)
    data = FileUtil.read(path, "json")
    if data is None:
        return {"codes": []}
    return data


def _save(data: dict, data_dir: Path) -> None:
    FileUtil.write(_codes_path(data_dir), data, "json")


def generate_codes(n: int, data_dir: Path, *, allowed_usage_count: int = 1) -> list[str]:
    """Generate N invitation codes and append them to the codes file.

    Returns the list of newly generated codes.
    """
    data = _load(data_dir)
    existing = {entry["code"] for entry in data["codes"]}

    new_codes: list[str] = []
    now = datetime.now(UTC).isoformat()
    while len(new_codes) < n:
        code = secrets.token_hex(4).upper()
        if code not in existing:
            existing.add(code)
            new_codes.append(code)
            data["codes"].append({
                "code": code,
                "created_at": now,
                "used_by": [],
                "allowed_usage_count": allowed_usage_count,
            })

    _save(data, data_dir)
    return new_codes


def _normalize_used_by(entry: dict) -> list[str]:
    """Normalize legacy used_by formats to a list."""
    used_by = entry.get("used_by")
    if used_by is None:
        return []
    if isinstance(used_by, str):
        return [used_by]
    return used_by


async def classify(code: str | None, data_dir: Path) -> str | None:
    """Classify an invite code WITHOUT consuming it (advisory pre-check).

    Returns the specific stable error CODE
    (``INVITE_MALFORMED`` / ``INVITE_NOT_FOUND`` / ``INVITE_ALREADY_REDEEMED`` /
    ``INVITE_EXPIRED`` / ``INVITE_REVOKED``) if the code is not redeemable, or
    ``None`` if it is currently redeemable. NEVER mutates the codes file, so it
    is safe to call from the advisory ``/auth/oauth/invite/validate`` precheck
    and again just before redemption.

    Honors the additive optional ``expires_at`` / ``revoked_at`` fields (absent
    on legacy codes => never expires / not revoked). Multi-use semantics are
    preserved: a code is "already redeemed" only once ``len(used_by)`` reaches
    ``allowed_usage_count``.
    """
    if not code or not code.strip():
        return "INVITE_MALFORMED"
    c = code.strip().upper()

    entry = None
    for candidate in _load(data_dir)["codes"]:
        if candidate.get("code") == c:
            entry = candidate
            break
    if entry is None:
        return "INVITE_NOT_FOUND"

    if entry.get("revoked_at"):
        return "INVITE_REVOKED"

    exp = entry.get("expires_at")
    if exp:
        try:
            expires = datetime.fromisoformat(exp)
        except ValueError:
            expires = None
        if expires is not None:
            now = datetime.now(UTC)
            if expires.tzinfo is None:
                now = now.replace(tzinfo=None)
            if now > expires:
                return "INVITE_EXPIRED"

    used_by = _normalize_used_by(entry)
    allowed = entry.get("allowed_usage_count", 1)
    if len(used_by) >= allowed:
        return "INVITE_ALREADY_REDEEMED"

    return None


async def validate_and_consume(code: str, data_dir: Path, username: str) -> bool:
    """Check if code is valid and has remaining uses, then mark it as consumed.

    Returns True if the code was valid and consumed, False otherwise.
    """
    if not code or not code.strip():
        return False

    async with _lock:
        data = _load(data_dir)
        for entry in data["codes"]:
            if entry["code"] == code.strip().upper():
                used_by = _normalize_used_by(entry)
                allowed = entry.get("allowed_usage_count", 1)
                if len(used_by) < allowed:
                    used_by.append(username)
                    entry["used_by"] = used_by
                    _save(data, data_dir)
                    return True
        return False
