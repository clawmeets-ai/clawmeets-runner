# SPDX-License-Identifier: MIT
"""
clawmeets/models/desk_todo.py

Desk to-do store — the My Desk right-rail "plate": tasks that still sit
with the manager. Two origins land here:

  * ``self``  — the user captured a quick snippet in the rail.
  * ``agent`` — an agent team member pushed it (off a briefing / a decision
    it surfaced) WITH context, a suggested recipient, a ready-to-refine
    prompt, "what's been done", and "available facts".

Clicking a to-do opens the Task take-over (a guided dispatch surface); the
plate is an *ordered* list the manager drags to reorder, so — unlike the
one-file-per-artifact brief-tab registry — every user's plate is a single
ordered JSON document. Reorder is then a plain array rewrite under one
lock, and agent-publish is a prepend.

Storage::

    {data_dir}/desk-todos/
      <owner_user_id>.json     # ordered list[DeskTodo], newest capture first

Mutations are broadcast to the owner via ``DESK_TODO_SYNC`` (see
``server/routes/desk_todos.py``) so the desk refetches ``GET /me/desk/todos``.
"""
from __future__ import annotations

import asyncio
import secrets
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from clawmeets.utils.file_io import FileUtil

_lock = asyncio.Lock()

TODOS_DIR = "desk-todos"

# Distinguishes "key absent from a PATCH body" (leave the stored value
# untouched) from an explicit JSON ``null`` (clear the field). Used only by
# ``patch_todo`` for the three-way ``draft_recipient_*`` merge; the route
# layer detects key presence off the raw request body (see
# ``server/routes/desk_todos.py``) since a non-serializable sentinel can't be
# a FastAPI ``Body`` default.
_UNSET: object = object()


class DeskTodoSource(BaseModel):
    """The external nudge a self-captured task came from (Slack/Email/…)."""

    label: str
    icon: str


class DeskTodoFileRef(BaseModel):
    """An agent-suggested reference file — name/sub only, no bytes. Rendered
    as an informational chip in the take-over composer; on dispatch its name
    is appended to the message so the agent knows what to consult."""

    name: str
    sub: str = ""
    icon: str = "report"


class DeskTodoFact(BaseModel):
    """One key/value in the take-over's "Available & relevant" list."""

    k: str
    v: str


class DeskTodoLink(BaseModel):
    """A source/briefing the take-over can open."""

    label: str
    icon: str


class DeskTodo(BaseModel):
    """A single item on the manager's plate."""

    id: str
    owner_user_id: str
    text: str
    origin: str = "self"  # "self" | "agent"
    status: str = "open"  # "open" | "done"
    created_at: str
    updated_at: str

    # self-capture
    source: DeskTodoSource | None = None
    due: str | None = None

    # agent-published extras
    by_agent_id: str | None = None
    by_agent_name: str | None = None
    suggest_agent_id: str | None = None
    suggest_agent_name: str | None = None
    draft_prompt: str | None = None
    # Recipient the manager picked in the take-over composer, stored alongside
    # the draft. Both id and name are persisted so the plate pill still labels
    # correctly after a rename/removal; the frontend re-resolves against the
    # live roster and falls back if the agent is gone. Null until set.
    draft_recipient_id: str | None = None
    draft_recipient_name: str | None = None
    context: str | None = None
    files: list[DeskTodoFileRef] = Field(default_factory=list)
    done_steps: list[str] = Field(default_factory=list)
    available: list[DeskTodoFact] = Field(default_factory=list)
    linked: DeskTodoLink | None = None

    # set when the manager saves a draft in the take-over
    drafted: bool = False


def gen_id() -> str:
    return "t-" + secrets.token_hex(6)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _path(data_dir: Path, owner_user_id: str) -> Path:
    return Path(data_dir) / TODOS_DIR / f"{owner_user_id}.json"


def _load(data_dir: Path, owner_user_id: str) -> list[DeskTodo]:
    raw = FileUtil.read(_path(data_dir, owner_user_id), "json")
    if not isinstance(raw, list):
        return []
    out: list[DeskTodo] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        try:
            out.append(DeskTodo.model_validate(row))
        except Exception:
            continue
    return out


def _save(data_dir: Path, owner_user_id: str, todos: list[DeskTodo]) -> None:
    FileUtil.write(
        _path(data_dir, owner_user_id),
        [t.model_dump() for t in todos],
        "json",
    )


def list_todos(data_dir: Path, owner_user_id: str) -> list[DeskTodo]:
    """Return the owner's plate in stored (manager-controlled) order."""
    return _load(data_dir, owner_user_id)


def get_todo(data_dir: Path, owner_user_id: str, todo_id: str) -> DeskTodo | None:
    for t in _load(data_dir, owner_user_id):
        if t.id == todo_id:
            return t
    return None


async def add_todo(
    data_dir: Path,
    owner_user_id: str,
    text: str,
    *,
    source: DeskTodoSource | None = None,
    due: str | None = None,
) -> DeskTodo:
    """Capture a self-origin task; prepended to the plate (newest first)."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Task text cannot be empty")
    async with _lock:
        todos = _load(data_dir, owner_user_id)
        now = _now()
        todo = DeskTodo(
            id=gen_id(),
            owner_user_id=owner_user_id,
            text=text,
            origin="self",
            status="open",
            created_at=now,
            updated_at=now,
            source=source,
            due=due,
        )
        todos.insert(0, todo)
        _save(data_dir, owner_user_id, todos)
        return todo


async def publish_agent_todo(
    data_dir: Path,
    owner_user_id: str,
    *,
    by_agent_id: str,
    by_agent_name: str,
    text: str,
    due: str | None = None,
    suggest_agent_id: str | None = None,
    suggest_agent_name: str | None = None,
    draft_prompt: str | None = None,
    context: str | None = None,
    files: list[DeskTodoFileRef] | None = None,
    done_steps: list[str] | None = None,
    available: list[DeskTodoFact] | None = None,
    linked: DeskTodoLink | None = None,
) -> DeskTodo:
    """Push an agent-origin task onto the owner's plate (prepended)."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Task text cannot be empty")
    async with _lock:
        todos = _load(data_dir, owner_user_id)
        now = _now()
        todo = DeskTodo(
            id=gen_id(),
            owner_user_id=owner_user_id,
            text=text,
            origin="agent",
            status="open",
            created_at=now,
            updated_at=now,
            due=due,
            by_agent_id=by_agent_id,
            by_agent_name=by_agent_name,
            suggest_agent_id=suggest_agent_id,
            suggest_agent_name=suggest_agent_name,
            draft_prompt=draft_prompt,
            context=context,
            files=files or [],
            done_steps=done_steps or [],
            available=available or [],
            linked=linked,
        )
        todos.insert(0, todo)
        _save(data_dir, owner_user_id, todos)
        return todo


async def patch_todo(
    data_dir: Path,
    owner_user_id: str,
    todo_id: str,
    *,
    status: str | None = None,
    text: str | None = None,
    due: str | None = None,
    draft_prompt: str | None = None,
    draft_recipient_id: str | None = _UNSET,  # _UNSET → leave untouched
    draft_recipient_name: str | None = _UNSET,  # None → clear, str → set
    drafted: bool | None = None,
) -> DeskTodo | None:
    """Patch a task in place. Returns the updated task, or None if missing.

    ``draft_recipient_id`` / ``draft_recipient_name`` are three-way: ``_UNSET``
    (the default, when the caller omits them) leaves the stored value alone,
    an explicit ``None`` clears it, and a string overwrites it. The older
    scalar fields collapse absent+None into "untouched" (a ``None`` never
    clears them)."""
    async with _lock:
        todos = _load(data_dir, owner_user_id)
        target: DeskTodo | None = None
        for t in todos:
            if t.id == todo_id:
                target = t
                break
        if target is None:
            return None
        if status is not None:
            if status not in ("open", "done"):
                raise ValueError("status must be 'open' or 'done'")
            target.status = status
        if text is not None:
            text = text.strip()
            if text:
                target.text = text
        if due is not None:
            target.due = due or None
        if draft_prompt is not None:
            target.draft_prompt = draft_prompt
        if draft_recipient_id is not _UNSET:
            target.draft_recipient_id = draft_recipient_id
        if draft_recipient_name is not _UNSET:
            target.draft_recipient_name = draft_recipient_name
        if drafted is not None:
            target.drafted = drafted
        target.updated_at = _now()
        _save(data_dir, owner_user_id, todos)
        return target


async def reorder_todos(
    data_dir: Path, owner_user_id: str, ordered_ids: list[str]
) -> list[DeskTodo]:
    """Reorder the plate to match ``ordered_ids``. Ids not present are ignored;
    any todos omitted from ``ordered_ids`` are appended in their prior order so
    a partial list (e.g. only the open items) never drops the rest."""
    async with _lock:
        todos = _load(data_dir, owner_user_id)
        by_id = {t.id: t for t in todos}
        seen: set[str] = set()
        ordered: list[DeskTodo] = []
        for tid in ordered_ids:
            t = by_id.get(tid)
            if t is not None and tid not in seen:
                ordered.append(t)
                seen.add(tid)
        for t in todos:
            if t.id not in seen:
                ordered.append(t)
        _save(data_dir, owner_user_id, ordered)
        return ordered


async def delete_todo(data_dir: Path, owner_user_id: str, todo_id: str) -> bool:
    """Remove a task. Returns True if it existed."""
    async with _lock:
        todos = _load(data_dir, owner_user_id)
        kept = [t for t in todos if t.id != todo_id]
        if len(kept) == len(todos):
            return False
        _save(data_dir, owner_user_id, kept)
        return True
