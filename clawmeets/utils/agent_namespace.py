# SPDX-License-Identifier: MIT
"""
clawmeets/utils/agent_namespace.py

Namespace resolution for agent names.

Storage is always globally unique ``{owner_username}-{suffix}`` — that's the
filesystem identity and what the registry guarantees. This module derives the
*short name* that an owner (or any namespace-aware reader) should see, and
resolves ``@short-name`` mentions by falling back to the project owner's
namespace when the short name isn't a registered full name.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from clawmeets.models.agent import Agent
    from clawmeets.models.context import ModelContext
    from clawmeets.models.project import Project


_MENTION_RE = re.compile(r'(?:^|(?<=[^a-zA-Z0-9_]))@([a-zA-Z][a-zA-Z0-9_-]*)')


def parse_mentions(content: Optional[str]) -> list[str]:
    """Extract @agent-name mentions from message content.

    Returns names without the ``@`` prefix, deduplicated, preserving order of
    first occurrence. Matches ``@`` only at word boundaries (so ``user@x.com``
    is rejected, ``**@agent**`` is accepted).
    """
    if not content:
        return []
    seen: set[str] = set()
    result: list[str] = []
    for m in _MENTION_RE.findall(content):
        if m not in seen:
            seen.add(m)
            result.append(m)
    return result


def short_name(full_name: str, owner_username: Optional[str]) -> str:
    """Strip ``{owner_username}-`` from ``full_name`` if present.

    Returns the full name unchanged when ``owner_username`` is None/empty
    or the prefix does not match.
    """
    if not owner_username:
        return full_name
    prefix = f"{owner_username}-"
    if full_name.startswith(prefix):
        return full_name[len(prefix):]
    return full_name


def resolve_mention(
    name: str,
    project: "Project",
    ctx: "ModelContext",
) -> Optional["Agent"]:
    """Resolve an ``@mention`` name to an Agent in the project's namespace.

    Exact match on the full registry name wins (so ``@alice-researcher`` and
    public agent names keep working). If no exact match, try the project
    owner's namespace by prefixing ``{owner_username}-`` and looking again.
    """
    from clawmeets.models.agent import Agent
    from clawmeets.models.user import User

    agent = Agent.get_by_name(name, ctx)
    if agent is not None:
        return agent

    if not project.created_by:
        return None
    owner = User.get(project.created_by, ctx)
    if owner is None:
        return None
    return Agent.get_by_name(f"{owner.username}-{name}", ctx)
