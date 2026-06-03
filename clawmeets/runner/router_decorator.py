# SPDX-License-Identifier: MIT
"""
clawmeets/runner/router_decorator.py

Cross-agent delegation LLM router (Primitive 3).

Runs on the requester agent's runner, right after the LLM produces an
``ActionBlock`` and before the local ``ActionBlockExecutor`` posts it. The
router inspects each ``reply`` action: if it contains an ``@mention`` of an
agent NOT owned by the requester's owner, the router:

1. ``POST /me/front-desk/{foreign_agent_full_name}/ensure`` (agent-bearer)
   to get-or-create a Front Desk project on the foreign agent's runner.
2. ``POST /tunnels`` to bind ``(current_local_project, current_local_room)``
   to ``(fd_project, "user-communication")``. Idempotent: skips if a matching
   binding already exists.

The original reply still posts locally as-is; the server-side
``TunnelSubscriber`` (``sync/tunnel.py``) then mirrors that MESSAGE entry
through the new binding to the FD project's ``user-communication``, where
the foreign coordinator reacts to it. Replies from the foreign coordinator
mirror back through the same binding.

The router does **not** modify the ``ActionBlock`` — its only side effect is
the FD-ensure + tunnel-bind HTTP calls. If any HTTP step fails, the failure
is logged and the original reply still posts locally (best-effort routing).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..api.actions import ActionBlock, ReplyAction
from ..utils.agent_namespace import parse_mentions

if TYPE_CHECKING:
    from ..models.context import ModelContext

logger = logging.getLogger(__name__)


async def route_to_foreign_agents(
    action_block: ActionBlock,
    model_ctx: "ModelContext",
    project_id: str,
    requester_agent_id: str,
    requester_agent_owner_id: str,
) -> None:
    """Detect foreign @mentions in reply actions and route them via FD + tunnel.

    Args:
        action_block: LLM output to inspect; not modified.
        model_ctx: Local ModelContext (for HTTP client + project lookup).
        project_id: The local project the LLM was invoked from. Becomes the
            ``local_project_id`` of any new tunnel binding.
        requester_agent_id: The requester agent's id (logging context only;
            actual auth flows via ``X-Agent-ID`` already set on every request).
        requester_agent_owner_id: ``registered_by`` of the requester agent.
            An ``@mention`` is "foreign" iff the resolved agent's
            ``registered_by`` differs from this.
    """
    client = model_ctx.client
    if client is None:
        return

    from ..models.project import Project

    project = Project.get(project_id, model_ctx)
    if project is None:
        return
    # Routing only makes sense on the requester side. Inside an FD project we
    # are already the responder — the LLM there does its work normally.
    if project.surface == "front_desk":
        return

    from ..utils.agent_namespace import resolve_mention

    # Collect (chatroom, foreign_agent_full_name) pairs across all reply actions.
    routed: set[tuple[str, str]] = set()
    for action in action_block.typed_actions():
        if not isinstance(action, ReplyAction):
            continue
        for mention in parse_mentions(action.content):
            agent = resolve_mention(mention, project, model_ctx)
            if agent is None:
                continue
            if not agent.registered_by:
                # Defensive: an agent with no owner has nothing to route to.
                continue
            if agent.registered_by == requester_agent_owner_id:
                continue
            key = (action.room, agent.name)
            if key in routed:
                continue
            routed.add(key)
            try:
                await _route_one(
                    client,
                    local_project_id=project_id,
                    local_room=action.room,
                    foreign_agent_full_name=agent.name,
                )
            except Exception:
                logger.exception(
                    "router_decorator: failed to route @%s from %s/%s "
                    "(requester=%s)",
                    agent.name,
                    project_id,
                    action.room,
                    requester_agent_id,
                )


async def _route_one(
    client,
    *,
    local_project_id: str,
    local_room: str,
    foreign_agent_full_name: str,
) -> None:
    # 1. Ensure FD project on the foreign agent's runner (idempotent).
    fd_project = await client.ensure_front_desk(foreign_agent_full_name)
    fd_project_id = fd_project["id"]

    # 2. Skip if this exact binding already exists.
    bindings = await client.list_tunnels()
    for b in bindings:
        if (
            b.get("local_project_id") == local_project_id
            and b.get("local_room") == local_room
            and b.get("fd_project_id") == fd_project_id
        ):
            return

    # 3. Create the binding.
    await client.create_tunnel(local_project_id, local_room, fd_project_id)
    logger.info(
        "router_decorator: bound %s/%s ↔ FD %s (foreign=%s)",
        local_project_id,
        local_room,
        fd_project_id,
        foreign_agent_full_name,
    )
