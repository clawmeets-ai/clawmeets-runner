# SPDX-License-Identifier: MIT
"""
clawmeets/utils/agent_filter.py

Unified ``(teams, names)`` allowlist for agent invitation.

One predicate, two surfaces:
- Project candidate pool: ``project.agent_teams`` + ``project.agent_names``
- Front Desk allowlist: ``agent.front_desk_invitable_teams`` + ``agent.front_desk_invitable_agents``

Both surfaces share the same matching rule (id OR display-name OR short-name OR
team intersection), the same hard enforcement at chatroom-create, and the same
live runtime injection into the coordinator prompt.

``viewer_owner_id`` lets the caller request short-name matching scoped to a
specific user's namespace — agents owned by that user get their owner-prefix
stripped for comparison so the user can type ``dave`` instead of
``alice-dave``. For cross-owner agents, only id and full-name match (typing
``dave`` would otherwise be ambiguous when multiple users own a ``-dave``).
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from clawmeets.models.agent import Agent


def matches_invitable(
    invitee: "Agent",
    allowed_teams: set[str],
    allowed_names: set[str],
    viewer_owner_id: Optional[str] = None,
) -> bool:
    """Return True iff ``invitee`` passes the ``(teams, names)`` allowlist.

    Empty teams + empty names = no rule expressed by *this* helper; the caller
    decides what to do (typically: skip enforcement entirely).
    """
    if invitee.id in allowed_names:
        return True
    if invitee.name in allowed_names:
        return True
    if (
        viewer_owner_id
        and invitee.registered_by == viewer_owner_id
        and "-" in invitee.name
    ):
        short = invitee.name.split("-", 1)[1]
        if short in allowed_names:
            return True
    if allowed_teams and (allowed_teams & set(invitee.user_teams)):
        return True
    return False


def resolve_invitable_agents(
    candidates: Iterable["Agent"],
    allowed_teams: list[str],
    allowed_names: list[str],
    viewer_owner_id: Optional[str] = None,
) -> list["Agent"]:
    """Return the subset of ``candidates`` that passes the allowlist.

    Empty filters = pass-through (returns the full list). Caller decides
    whether "no filter" means "everyone" (project pool) or "no one"
    (Front Desk allowlist).
    """
    teams_set, names_set = set(allowed_teams), set(allowed_names)
    if not teams_set and not names_set:
        return list(candidates)
    return [
        a
        for a in candidates
        if matches_invitable(a, teams_set, names_set, viewer_owner_id)
    ]
