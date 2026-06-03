# SPDX-License-Identifier: BUSL-1.1
"""
clawmeets/sync/tunnel.py

Cross-project message mirror for the LLM-routed tunnel between a requester
agent's project room and the responder's Front Desk project ``user-communication``.

Lives **server-side only** — registered as a ``ChangelogSubscriber`` on every
project's runloop. On each appended MESSAGE entry it consults the binding store
(:mod:`clawmeets.models.tunnel_binding`) and, if the project+room sit on
either end of a binding, appends a mirrored copy to the other end via the
same ``ChangelogRunloopManager`` — along with the side-effects that the
``send_message_as_user`` / ``post_message`` route handlers would have done
for a directly-posted message:

- `PROJECT_REACTIVATED` (when target is `user-communication` of a non-ACTIVE project).
- `CHANGELOG_UPDATE` broadcast via `WSHub` (so the target's participants' runners get a WS push).
- `WorkTracker` bookkeeping: `create_pending_work` on forward direction; `record_response` + `BATCH_COMPLETE` emission on reverse direction.

Loop-guard: every mirrored MESSAGE entry carries `mirrored_from` on the
`ChangelogEntry`; the subscriber returns immediately when that field is set
on an incoming entry. The synthetic ``PROJECT_REACTIVATED`` and
``BATCH_COMPLETE`` entries the mirror itself emits do NOT carry
``mirrored_from`` (they're target-local control entries, not mirrors), and
the entry-type filter on `on_entry` skips them anyway.

Scope (MVP): mirror MESSAGE entries only. FILE_CREATED / FILE_UPDATED /
PROJECT_COMPLETED through the tunnel are TODOs.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from clawmeets.api.control import (
    ChangelogUpdatePayload,
    ControlEnvelope,
    ControlMessageType,
)

from .changelog import (
    BatchCompletePayload,
    ChangelogEntry,
    ChangelogEntryType,
    MessagePayload,
    MirroredFromRef,
    ProjectReactivatedPayload,
    ProjectStatus,
)
from .subscriber import ChangelogSubscriber

if TYPE_CHECKING:
    from clawmeets.models.context import ModelContext
    from clawmeets.models.work_tracker import WorkTracker
    from clawmeets.server.ws_hub import WSHub
    from clawmeets.sync.runloop_manager import ChangelogRunloopManager

logger = logging.getLogger("clawmeets.tunnel")


class TunnelSubscriber(ChangelogSubscriber):
    """Mirrors qualifying entries between bound rooms in two projects.

    Holds back-references to the server's ``runloop_manager``, ``ws_hub`` and
    ``work_tracker``. All three are late-bound after construction (the
    runloop manager and this subscriber have a chicken-and-egg setup; we
    keep the same pattern for the other two — see ``server/app.py``).
    """

    def __init__(self, model_ctx: "ModelContext") -> None:
        self._model_ctx = model_ctx
        self._runloop_manager: "ChangelogRunloopManager | None" = None
        self._ws_hub: "WSHub | None" = None
        self._work_tracker: "WorkTracker | None" = None
        self._batch_timeout: int = 1800

    def bind_runloop_manager(self, manager: "ChangelogRunloopManager") -> None:
        """Late-bind the runloop manager (constructed after this subscriber)."""
        self._runloop_manager = manager

    def bind_ws_hub(self, ws_hub: "WSHub") -> None:
        """Late-bind the WS hub (used for CHANGELOG_UPDATE broadcasts on mirror)."""
        self._ws_hub = ws_hub

    def bind_work_tracker(
        self, work_tracker: "WorkTracker", batch_timeout: int
    ) -> None:
        """Late-bind the work tracker (used for PendingWork bookkeeping on mirror)."""
        self._work_tracker = work_tracker
        self._batch_timeout = batch_timeout

    async def on_entry(
        self,
        entry: ChangelogEntry,
        project_id: str,
        project_name: str,
    ) -> None:
        # Loop-guard.
        if entry.mirrored_from is not None:
            return
        # MVP: only mirror MESSAGE entries.
        if entry.entry_type != ChangelogEntryType.MESSAGE:
            return
        if self._runloop_manager is None:
            logger.warning("TunnelSubscriber.on_entry called before bind_runloop_manager")
            return
        if not isinstance(entry.payload, MessagePayload):
            return

        # Cheap escape: import here so importing this module from tests/early
        # boot doesn't trigger the full model graph load.
        from clawmeets.models.tunnel_binding import find_bindings_for_project

        chatroom_name = entry.payload.chatroom_name
        for binding in find_bindings_for_project(project_id, self._model_ctx):
            await self._maybe_mirror(entry, project_id, chatroom_name, binding)

    async def _maybe_mirror(
        self,
        entry: ChangelogEntry,
        project_id: str,
        chatroom_name: str,
        binding,  # TunnelBinding (avoid runtime import in signature)
    ) -> None:
        from clawmeets.models.project import Project

        # Forward: requester local room → responder FD user-communication.
        if (
            project_id == binding.local_project_id
            and chatroom_name == binding.local_room
        ):
            fd_proj = Project.get(binding.fd_project_id, self._model_ctx)
            await self._mirror(
                target_project_id=binding.fd_project_id,
                target_project_name=fd_proj.name,
                target_chatroom="user-communication",
                src_project_id=project_id,
                src_payload=entry.payload,
                src_version=entry.version,
                direction="forward",
            )
            return

        # Reverse: responder FD user-communication → requester local room.
        if (
            project_id == binding.fd_project_id
            and chatroom_name == "user-communication"
        ):
            local_proj = Project.get(binding.local_project_id, self._model_ctx)
            await self._mirror(
                target_project_id=binding.local_project_id,
                target_project_name=local_proj.name,
                target_chatroom=binding.local_room,
                src_project_id=project_id,
                src_payload=entry.payload,
                src_version=entry.version,
                direction="reverse",
            )

    async def _mirror(
        self,
        target_project_id: str,
        target_project_name: str,
        target_chatroom: str,
        src_project_id: str,
        src_payload: MessagePayload,
        src_version: int,
        direction: str,  # "forward" | "reverse"
    ) -> None:
        assert self._runloop_manager is not None
        from clawmeets.models.project import Project

        # Build the mirrored payload with the target's chatroom_name. Preserve
        # sender attribution so the chat log on the target side reads as
        # "from the foreign participant"; PARTICIPANTS.ndjson will not list
        # them (ghost participant) — a follow-up will introduce PARTICIPANT_ADDED
        # mirroring so the local prompt rendering sees a clean roster.
        mirrored_payload = MessagePayload(
            chatroom_name=target_chatroom,
            id=src_payload.id,
            ts=src_payload.ts,
            from_participant_id=src_payload.from_participant_id,
            from_participant_name=src_payload.from_participant_name,
            content=src_payload.content,
            # `expects_response_from` on the source side is local to that
            # project's participants. Mirror it through on the forward path
            # (so the responder coordinator wakes up); leave it empty on the
            # reverse path (the local project's worker dispatch is governed
            # by the original @mention, not the mirrored reply).
            expects_response_from=(
                list(src_payload.expects_response_from)
                if direction == "forward"
                else []
            ),
            is_ack=src_payload.is_ack,
        )

        runloop = await self._runloop_manager.get_or_create(
            target_project_id, target_project_name
        )

        # Auto-reactivate: if the target project is non-ACTIVE and we're
        # delivering into user-communication, reactivate before the MESSAGE
        # so subscribers (and the UI) see status=ACTIVE first. Mirrors
        # `send_message_as_user`'s rule (`messages.py:972`).
        target_proj = Project.get(target_project_id, self._model_ctx)
        if (
            target_proj is not None
            and target_chatroom == "user-communication"
            and target_proj.status != ProjectStatus.ACTIVE
        ):
            await runloop.append(
                ChangelogEntryType.PROJECT_REACTIVATED,
                ProjectReactivatedPayload(),
            )

        entry = await runloop.append(
            ChangelogEntryType.MESSAGE,
            mirrored_payload,
            mirrored_from=MirroredFromRef(
                project_id=src_project_id,
                version=src_version,
            ),
        )
        logger.info(
            "tunnel: mirrored v%d %s/%s → %s/%s (%s)",
            src_version,
            src_project_id,
            src_payload.chatroom_name,
            target_project_id,
            target_chatroom,
            direction,
        )

        # Broadcast CHANGELOG_UPDATE so target-side runners get a WS push for
        # the mirrored entry (otherwise we rely on whatever polling cadence
        # the runner uses, which can throttle on completed projects).
        if self._ws_hub is not None and target_proj is not None:
            envelope = ControlEnvelope(
                type=ControlMessageType.CHANGELOG_UPDATE,
                payload=ChangelogUpdatePayload(
                    project_id=target_project_id,
                    project_name=target_project_name,
                    new_version=entry.version,
                    coordinator_id=target_proj.coordinator_id,
                ),
            )
            try:
                await self._ws_hub.broadcast_to_project(envelope, target_proj)
            except Exception:
                logger.exception("tunnel: WS broadcast failed for v%d", entry.version)

        # WorkTracker bookkeeping — skip acks (mirrors `messages.py:274` guard).
        if self._work_tracker is None or mirrored_payload.is_ack:
            return

        if direction == "forward":
            # Forward mirror = a delegation message arriving on the FD side.
            # Open a PendingWork batch on the target chatroom so the typing
            # indicator + batch-timeout machinery covers the foreign agent's
            # in-flight work.
            if mirrored_payload.expects_response_from:
                try:
                    await self._work_tracker.create_pending_work(
                        message_id=mirrored_payload.id,
                        message_version=entry.version,
                        project_id=target_project_id,
                        project_name=target_project_name,
                        chatroom_name=target_chatroom,
                        coordinator_id=mirrored_payload.from_participant_id,
                        expected_participants=list(
                            mirrored_payload.expects_response_from
                        ),
                        timeout_seconds=self._batch_timeout,
                    )
                except ValueError:
                    logger.warning(
                        "tunnel: PendingWork already exists for %s/%s",
                        target_project_id,
                        target_chatroom,
                    )
            return

        # direction == "reverse": foreign agent's reply mirroring back into
        # the requester's local room. Record the response against any open
        # PendingWork there and emit BATCH_COMPLETE if the batch closes.
        work = await self._work_tracker.record_response(
            target_project_id, target_chatroom, mirrored_payload.from_participant_id
        )
        if work is None or not work.is_complete:
            return
        await self._work_tracker.clear_pending_work(
            target_project_id, target_chatroom
        )
        batch_entry = await runloop.append(
            ChangelogEntryType.BATCH_COMPLETE,
            BatchCompletePayload(
                chatroom_name=target_chatroom,
                message_id=work.message_id,
                coordinator_id=work.coordinator_id,
                responded_participants=list(work.responded_participants),
            ),
            source_version=work.message_version,
        )
        if self._ws_hub is not None and target_proj is not None:
            envelope = ControlEnvelope(
                type=ControlMessageType.CHANGELOG_UPDATE,
                payload=ChangelogUpdatePayload(
                    project_id=target_project_id,
                    project_name=target_project_name,
                    new_version=batch_entry.version,
                    coordinator_id=target_proj.coordinator_id,
                ),
            )
            try:
                await self._ws_hub.broadcast_to_project(envelope, target_proj)
            except Exception:
                logger.exception(
                    "tunnel: WS broadcast failed for BATCH_COMPLETE v%d",
                    batch_entry.version,
                )
