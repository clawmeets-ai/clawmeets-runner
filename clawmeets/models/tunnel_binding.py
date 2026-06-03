# SPDX-License-Identifier: BUSL-1.1
"""
clawmeets/models/tunnel_binding.py

Cross-project room-pair binding for the LLM-routed tunnel between a requester
agent's project room and the responder's Front Desk project's
``user-communication`` room.

A binding is server-side state. It is consulted by ``TunnelSubscriber``
(``sync/tunnel.py``) on every changelog append to decide whether the entry
needs to be mirrored to the other side of the tunnel.

Storage layout::

    {model_ctx.base_dir}/tunnels/{binding_id}.json     # one JSON per binding

Bindings are looked up by scanning the directory — N is expected to be small
(one per active cross-agent delegation). Atomic writes via ``FileUtil``.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, model_validator

from ..utils.file_io import FileUtil

if TYPE_CHECKING:
    from .context import ModelContext


class TunnelBinding(BaseModel):
    """A room-pair binding for cross-project message mirroring.

    Two requester shapes are supported (mirrors ``Project.requester_kind``):

    - ``"agent"``: created by Primitive 3's LLM router decorator when a
      requester agent's reply contains an ``@foreign-agent`` mention.
      ``requester_id`` holds the requester agent's id.
    - ``"user"``: created server-side (Primitive 4) when a human user types
      ``@foreign-agent`` or invites one into a chatroom. ``requester_id``
      holds the requesting user's id.

    Removed when the FD project completes (``PROJECT_COMPLETED``) or when a
    caller explicitly tears down the binding.
    """
    model_config = {"frozen": True}

    binding_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    local_project_id: str          # requester-side project
    local_room: str                # chatroom name on the requester side
    fd_project_id: str             # responder-side Front Desk project
    fd_project_name: str           # cached for path resolution without lookup
    foreign_agent_id: str          # responder agent (= fd_project.coordinator_id)
    foreign_agent_name: str        # cached for prompt/display
    requester_id: str              # user_id OR agent_id — see requester_kind
    requester_kind: Literal["user", "agent"] = "agent"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="before")
    @classmethod
    def _backfill_legacy_requester_fields(cls, data):
        """Back-fill ``requester_id`` from the old ``requester_agent_id`` field.

        Legacy on-disk records written before P4 used ``requester_agent_id``;
        in-place data carries no ``requester_kind``. Map them to the new shape
        without an explicit migration.
        """
        if isinstance(data, dict):
            if "requester_id" not in data and "requester_agent_id" in data:
                data = dict(data)
                data["requester_id"] = data.pop("requester_agent_id")
                data.setdefault("requester_kind", "agent")
        return data


def _tunnels_dir(ctx: "ModelContext") -> Path:
    return ctx.base_dir / "tunnels"


def list_bindings(ctx: "ModelContext") -> list[TunnelBinding]:
    """List every binding on disk. Sorted by ``binding_id`` for determinism."""
    d = _tunnels_dir(ctx)
    if not d.exists():
        return []
    result: list[TunnelBinding] = []
    for entry in sorted(d.iterdir()):
        if not entry.is_file() or entry.suffix != ".json":
            continue
        data = FileUtil.read(entry, "json")
        if data:
            result.append(TunnelBinding.model_validate(data))
    return result


def get_binding(binding_id: str, ctx: "ModelContext") -> TunnelBinding | None:
    path = _tunnels_dir(ctx) / f"{binding_id}.json"
    data = FileUtil.read(path, "json")
    if not data:
        return None
    return TunnelBinding.model_validate(data)


def save_binding(binding: TunnelBinding, ctx: "ModelContext") -> None:
    path = _tunnels_dir(ctx) / f"{binding.binding_id}.json"
    FileUtil.write(path, binding.model_dump(mode="json"), "json", atomic=True)


def delete_binding(binding_id: str, ctx: "ModelContext") -> bool:
    """Remove a binding. Returns True iff a record was deleted."""
    path = _tunnels_dir(ctx) / f"{binding_id}.json"
    if not path.exists():
        return False
    FileUtil.delete(path)
    return True


def find_bindings_for_project(
    project_id: str,
    ctx: "ModelContext",
) -> list[TunnelBinding]:
    """Return every binding where ``project_id`` sits on either end of the tunnel."""
    return [
        b for b in list_bindings(ctx)
        if b.local_project_id == project_id or b.fd_project_id == project_id
    ]
