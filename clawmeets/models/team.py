# SPDX-License-Identifier: BUSL-1.1
"""
clawmeets/models/team.py

Owner-scoped team metadata (sample requests). Membership itself is derived
from each agent's `user_teams: list[str]` field on its card — this module
only persists the *extra* metadata a team carries (sample requests), keyed
by `(owner_user_id, team_name)`.

Storage: one JSON file per user at `{data_dir}/teams/{user_id}.json` in
the shape:

    {
      "teams": {
        "<team_name>": {
          "sample_requests": [
            {"title": "...", "request": "...", "coordinator_hint": "..."}
          ],
          "created_at": "ISO8601"
        }
      }
    }

A Team record can exist with zero members (created before any agent is
invited), and members can exist without a record (legacy label-only teams
keep working — they just have an empty sample_requests list).
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from clawmeets.utils.file_io import FileUtil

_lock = asyncio.Lock()

TEAMS_DIR = "teams"


class SampleRequest(BaseModel):
    """A reusable project request attached to a team.

    Users click "Create Project" on one of these to jump into the
    CreateProjectPage with `name`, `request`, and `agent_teams` pre-filled.
    """

    title: str
    request: str
    coordinator_hint: Optional[str] = None


class Team(BaseModel):
    """Owner-scoped team metadata.

    `name` matches the string stored in `agent.user_teams` — it is how
    membership links up. No uniqueness check across users; two different
    users can both have a team called "career" without collision.
    """

    name: str
    sample_requests: list[SampleRequest] = Field(default_factory=list)
    created_at: str


def _user_teams_path(data_dir: Path, user_id: str) -> Path:
    return Path(data_dir) / TEAMS_DIR / f"{user_id}.json"


def _load(data_dir: Path, user_id: str) -> dict:
    path = _user_teams_path(data_dir, user_id)
    data = FileUtil.read(path, "json")
    if data is None:
        return {"teams": {}}
    if "teams" not in data:
        data["teams"] = {}
    return data


def _save(data: dict, data_dir: Path, user_id: str) -> None:
    FileUtil.write(_user_teams_path(data_dir, user_id), data, "json")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _team_from_entry(name: str, entry: dict) -> Team:
    return Team(
        name=name,
        sample_requests=[SampleRequest(**s) for s in entry.get("sample_requests", [])],
        created_at=entry.get("created_at") or _now(),
    )


def list_teams(data_dir: Path, user_id: str) -> list[Team]:
    """Return every persisted team for the given user, sorted by name."""
    data = _load(data_dir, user_id)
    return [_team_from_entry(name, entry) for name, entry in sorted(data["teams"].items())]


def get_team(data_dir: Path, user_id: str, name: str) -> Team | None:
    data = _load(data_dir, user_id)
    entry = data["teams"].get(name)
    if entry is None:
        return None
    return _team_from_entry(name, entry)


async def create_team(
    data_dir: Path,
    user_id: str,
    name: str,
    sample_requests: list[SampleRequest] | None = None,
) -> Team:
    """Upsert a team. If the team already exists, its existing sample_requests
    are preserved and new ones (matched by title) are merged in — this keeps
    re-running `init --from-url` idempotent. Returns the resulting Team.
    """
    async with _lock:
        data = _load(data_dir, user_id)
        entry = data["teams"].get(name)
        if entry is None:
            entry = {"sample_requests": [], "created_at": _now()}
            data["teams"][name] = entry

        if sample_requests:
            existing_titles = {s.get("title") for s in entry["sample_requests"]}
            for req in sample_requests:
                if req.title in existing_titles:
                    continue
                entry["sample_requests"].append(req.model_dump())
                existing_titles.add(req.title)

        _save(data, data_dir, user_id)
        return _team_from_entry(name, entry)


async def delete_team(data_dir: Path, user_id: str, name: str) -> bool:
    """Delete team metadata. Returns True if the team existed and was removed.

    Does NOT touch agent.user_teams — membership cleanup is the caller's job
    (the HTTP route handles it when `?remove_from_agents=true`).
    """
    async with _lock:
        data = _load(data_dir, user_id)
        if name not in data["teams"]:
            return False
        del data["teams"][name]
        _save(data, data_dir, user_id)
        return True


async def add_sample_request(
    data_dir: Path,
    user_id: str,
    name: str,
    sample: SampleRequest,
) -> Team:
    """Append a sample request to a team (creating the team if needed).

    Deduped by title: if a request with the same title already exists, it is
    replaced with the new content rather than appended, so the list doesn't
    grow on re-import.
    """
    async with _lock:
        data = _load(data_dir, user_id)
        entry = data["teams"].get(name)
        if entry is None:
            entry = {"sample_requests": [], "created_at": _now()}
            data["teams"][name] = entry

        replaced = False
        for i, existing in enumerate(entry["sample_requests"]):
            if existing.get("title") == sample.title:
                entry["sample_requests"][i] = sample.model_dump()
                replaced = True
                break
        if not replaced:
            entry["sample_requests"].append(sample.model_dump())

        _save(data, data_dir, user_id)
        return _team_from_entry(name, entry)


async def remove_sample_request(
    data_dir: Path,
    user_id: str,
    name: str,
    index: int,
) -> bool:
    """Remove the sample request at `index`. Returns True on success."""
    async with _lock:
        data = _load(data_dir, user_id)
        entry = data["teams"].get(name)
        if entry is None:
            return False
        if index < 0 or index >= len(entry["sample_requests"]):
            return False
        entry["sample_requests"].pop(index)
        _save(data, data_dir, user_id)
        return True
