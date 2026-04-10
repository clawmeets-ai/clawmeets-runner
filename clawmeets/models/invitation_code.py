# SPDX-License-Identifier: BUSL-1.1
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


def generate_codes(n: int, data_dir: Path) -> list[str]:
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
                "used_by": None,
            })

    _save(data, data_dir)
    return new_codes


async def validate_and_consume(code: str, data_dir: Path, username: str) -> bool:
    """Check if code is valid and unused, then mark it as consumed.

    Returns True if the code was valid and consumed, False otherwise.
    """
    if not code or not code.strip():
        return False

    async with _lock:
        data = _load(data_dir)
        for entry in data["codes"]:
            if entry["code"] == code.strip().upper() and entry["used_by"] is None:
                entry["used_by"] = username
                _save(data, data_dir)
                return True
        return False
