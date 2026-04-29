"""
clawmeets/sync/reflection_completion.py
Detect reflection / lint trigger replies and update Agent timestamps.

The reflection scheduler posts trigger messages tagged with HTML-comment
markers (one for reflect, one for lint). When an agent replies to a trigger
(its reply's source_version points at the trigger), this subscriber loads
the source message from the chatroom's CHATS.ndjson, branches on which
marker is present, and updates the corresponding cursor on the agent card:

- ``REFLECT_TRIGGER_MARKER`` → ``Agent.update_last_reflected_at()``
- ``LINT_TRIGGER_MARKER``    → ``Agent.update_last_linted_at()``

Markers are duplicated here (rather than imported from
``clawmeets.server.reflection_scheduler``) to keep this Layer 0 / Layer 1
module free of any server import.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from clawmeets.models.chat_message import ChatMessage
from clawmeets.sync.changelog import ChangelogEntryType, MessagePayload
from clawmeets.sync.subscriber import ChangelogSubscriber

if TYPE_CHECKING:
    from clawmeets.models.context import ModelContext
    from clawmeets.sync.changelog import ChangelogEntry

logger = logging.getLogger("clawmeets.sync.reflection_completion")

# Must match clawmeets.server.reflection_scheduler.{REFLECT,LINT}_TRIGGER_MARKER.
REFLECT_TRIGGER_MARKER = "<!-- clawmeets:reflect-trigger -->"
LINT_TRIGGER_MARKER = "<!-- clawmeets:lint-trigger -->"


class ReflectionCompletionSubscriber(ChangelogSubscriber):
    """Update Agent.last_reflected_at / last_linted_at on memory-loop replies.

    Cheap path: only inspects MESSAGE entries that
      (a) live in a DM room (name starts with `dm-`),
      (b) carry a `source_version` (i.e., are a reply), and
      (c) are not ack messages.
    For those, the source message is loaded from the chatroom's CHATS.ndjson
    and matched against the trigger markers. If a marker matches, the sending
    agent's card.json is updated on the corresponding cursor.
    """

    def __init__(self, model_ctx: "ModelContext") -> None:
        self._model_ctx = model_ctx

    async def on_entry(
        self,
        entry: "ChangelogEntry",
        project_id: str,
        project_name: str,
    ) -> None:
        if entry.entry_type != ChangelogEntryType.MESSAGE:
            return
        payload = entry.payload
        if not isinstance(payload, MessagePayload):
            return
        if payload.is_ack:
            return
        if entry.source_version is None:
            return
        if not payload.chatroom_name.startswith("dm-"):
            return
        if not payload.from_participant_id:
            return

        # Lazy imports to keep Layer 0 cleanliness.
        from clawmeets.models.chatroom import Chatroom
        from clawmeets.models.agent import Agent

        chatroom = Chatroom.get(project_id, payload.chatroom_name, self._model_ctx)
        if chatroom is None:
            return

        source_msg = _find_message_by_version(chatroom, entry.source_version)
        if source_msg is None:
            return

        # Branch on which marker is in the source. Reflect and lint are
        # mutually exclusive — the scheduler emits one or the other.
        if REFLECT_TRIGGER_MARKER in source_msg.content:
            mode = "reflect"
        elif LINT_TRIGGER_MARKER in source_msg.content:
            mode = "lint"
        else:
            return

        agent = Agent.get(payload.from_participant_id, self._model_ctx)
        if agent is None:
            logger.warning(
                f"{mode.capitalize()} reply from unknown "
                f"agent_id={payload.from_participant_id[:8]}; skipping cursor update"
            )
            return

        if mode == "reflect":
            agent.update_last_reflected_at(payload.ts)
            logger.info(
                f"Updated last_reflected_at for agent={agent.name!r} "
                f"to {payload.ts.isoformat()}"
            )
        else:  # mode == "lint"
            agent.update_last_linted_at(payload.ts)
            logger.info(
                f"Updated last_linted_at for agent={agent.name!r} "
                f"to {payload.ts.isoformat()}"
            )


def _find_message_by_version(chatroom, version: int) -> ChatMessage | None:
    """Locate a ChatMessage in CHATS.ndjson by changelog version. None if not found."""
    for log_entry in chatroom.get_log_entries():
        if isinstance(log_entry, ChatMessage) and log_entry.version == version:
            return log_entry
    return None
