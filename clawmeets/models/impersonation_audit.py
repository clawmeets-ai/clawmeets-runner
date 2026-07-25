# SPDX-License-Identifier: MIT
"""
clawmeets/models/impersonation_audit.py

Append-only audit log for admin impersonation (Auth model 1A). Exactly one
row is written per *served* impersonated request: an admin, holding their OWN
JWT and adding ``X-Impersonate-User: <target uuid>``, acted as another user.

Storage — the server's file-backed idiom (no SQL / no Alembic; mirrors
``models/desk_todo.py``)::

    {data_dir}/impersonation-audit/audit.json   # ordered list[ImpersonationAudit], oldest first

The whole log is a single JSON array so it stays human-greppable and globally
listable; writes are serialized under a module lock (append-one-row semantics).
``data_dir`` is the server's ``model_ctx.participants_dir`` (same base the
passwd store and desk-todo plates live under).
"""
from __future__ import annotations

import asyncio
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from clawmeets.utils.file_io import FileUtil

_lock = asyncio.Lock()

AUDIT_DIR = "impersonation-audit"
AUDIT_FILE = "audit.json"


class ImpersonationAudit(BaseModel):
    """One audited impersonated request (admin acted AS user at a point in time)."""

    id: str
    admin_id: str            # the real admin — the bearer-token owner
    admin_username: str = ""
    acted_as_user_id: str    # the impersonated target user
    acted_as_username: str = ""
    method: str              # HTTP verb of the served request
    path: str                # request path of the served request
    created_at: str          # ISO-8601 UTC


def gen_id() -> str:
    return "imp-" + secrets.token_hex(6)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _path(data_dir: Path) -> Path:
    return Path(data_dir) / AUDIT_DIR / AUDIT_FILE


def _load(data_dir: Path) -> list[ImpersonationAudit]:
    raw = FileUtil.read(_path(data_dir), "json")
    if not isinstance(raw, list):
        return []
    out: list[ImpersonationAudit] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        try:
            out.append(ImpersonationAudit.model_validate(row))
        except Exception:
            continue
    return out


def _save(data_dir: Path, rows: list[ImpersonationAudit]) -> None:
    FileUtil.write(_path(data_dir), [r.model_dump() for r in rows], "json")


async def record_impersonation(
    data_dir: Path,
    *,
    admin_id: str,
    acted_as_user_id: str,
    method: str,
    path: str,
    admin_username: str = "",
    acted_as_username: str = "",
) -> ImpersonationAudit:
    """Append exactly one audit row for a served impersonated request."""
    async with _lock:
        rows = _load(data_dir)
        row = ImpersonationAudit(
            id=gen_id(),
            admin_id=admin_id,
            admin_username=admin_username,
            acted_as_user_id=acted_as_user_id,
            acted_as_username=acted_as_username,
            method=method,
            path=path,
            created_at=_now(),
        )
        rows.append(row)
        _save(data_dir, rows)
        return row


def list_impersonations(
    data_dir: Path,
    *,
    admin_id: Optional[str] = None,
    acted_as_user_id: Optional[str] = None,
) -> list[ImpersonationAudit]:
    """Return the audit log (oldest first), optionally filtered by actor/target."""
    rows = _load(data_dir)
    if admin_id is not None:
        rows = [r for r in rows if r.admin_id == admin_id]
    if acted_as_user_id is not None:
        rows = [r for r in rows if r.acted_as_user_id == acted_as_user_id]
    return rows
