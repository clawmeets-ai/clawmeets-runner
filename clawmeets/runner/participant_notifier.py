# SPDX-License-Identifier: MIT
"""
clawmeets/runner/participant_notifier.py
Participant notifier - fires callbacks after files are ready.

This subscriber runs LAST (priority 200), after ModelContext.
When callbacks fire, files are guaranteed to be ready on disk.

Calls Participant methods directly with IDs and raw data.
Participants emit actions directly via their injected ActionEmitter.
"""
from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING

from ..sync.changelog import (
    BatchCompletePayload,
    BatchTimeoutPayload,
    ChangelogEntry,
    ChangelogEntryType,
    MessagePayload,
    FilePayload,
    ProjectPlanStatePayload,
    RoomCreatedPayload,
)
from ..models.chat_message import ChatMessage
from ..models.chatroom import Chatroom
from ..sync.subscriber import ChangelogSubscriber

if TYPE_CHECKING:
    from ..models.participant import Participant

logger = logging.getLogger(__name__)


class ParticipantNotifier(ChangelogSubscriber):
    """
    Notifies participants of events after filesystem is ready.

    This subscriber:
    - Runs last (priority 200)
    - Files are already written (ModelContext done)
    - Calls Participant methods directly with IDs and raw data

    Participants handle action emission themselves via their injected
    ActionEmitter - this notifier just fires the callbacks.
    """

    def __init__(
        self,
        participant: "Participant",
    ) -> None:
        """
        Initialize the participant notifier.

        Args:
            participant: The participant to notify
        """
        self._participant = participant
        # Per-project batch bookkeeping for the plan-gate resume, both cleared
        # by `on_sync_complete`. `_plan_state_version` is the version of the
        # last PROJECT_PLAN_STATE seen in this batch; `_woken_version` is the
        # version of the last entry in this batch that started a turn for this
        # participant (an addressed message, a first user request, or one of the
        # two batch outcomes). Comparing the two — rather than merely asking
        # whether both happened — is what makes the guard correct in both
        # directions; see `on_sync_complete`.
        self._plan_state_version: dict[str, int] = {}
        self._woken_version: dict[str, int] = {}

    async def on_entry(
        self,
        entry: ChangelogEntry,
        project_id: str,
        project_name: str,
    ) -> None:
        """Notify participant of a single changelog entry."""
        match entry.entry_type:
            case ChangelogEntryType.ROOM_CREATED:
                await self._notify_room_created(entry, project_id)

            case ChangelogEntryType.MESSAGE:
                await self._notify_message(entry, project_id)

            case ChangelogEntryType.FILE_CREATED:
                await self._notify_file_created(entry, project_id)

            case ChangelogEntryType.FILE_UPDATED:
                await self._notify_file_updated(entry, project_id)

            case ChangelogEntryType.PROJECT_COMPLETED:
                await self._notify_project_completed(entry, project_id)

            case ChangelogEntryType.BATCH_COMPLETE:
                await self._notify_batch_complete(entry, project_id)

            case ChangelogEntryType.BATCH_TIMEOUT:
                await self._notify_batch_timeout(entry, project_id)

            case ChangelogEntryType.PROJECT_PLAN_STATE:
                await self._notify_plan_state(entry, project_id)

    async def _notify_room_created(
        self,
        entry: ChangelogEntry,
        project_id: str,
    ) -> None:
        """Notify of chatroom creation and invitation."""
        payload = self._extract_payload(entry, RoomCreatedPayload)

        # Notify of chatroom creation (for caching chatroom name)
        await self._participant.on_chatroom_created(
            project_id=project_id,
            chatroom_name=payload.chatroom_name,
            participants=payload.participants,
        )

    async def _notify_message(
        self,
        entry: ChangelogEntry,
        project_id: str,
    ) -> None:
        """Notify of message received.

        For the project coordinator receiving messages in user-communication room,
        checks if this is the first message and triggers on_first_user_request()
        instead of on_message(). Coordinator is determined by project.coordinator_id.
        """
        payload = self._extract_payload(entry, MessagePayload)

        # Skip messages from self to avoid infinite loops
        if payload.from_participant_id == self._participant.id:
            return

        # P4: skip foreign memberships. If our own PARTICIPANTS row in this
        # chatroom is `external=true`, the TunnelSubscriber is mirroring this
        # message to our FD project — we process it there, not here.
        if self._is_external_in(project_id, payload.chatroom_name):
            return

        # Check for first user message in user-communication (coordinator only)
        # Lazy check: ModelContext (priority 0) writes message before this runs (priority 200)
        # So count == 1 means this is the first message
        project = self._participant.get_project(project_id)

        # In DM-shaped projects (owned DM OR Front Desk host end) the coordinator
        # IS the partner agent, so it should only reply when actually addressed
        # (server populates expects_response_from via the user-comm
        # auto-coordinator fallback, or the tunnel mirror forwards it on the FD
        # host end).
        if project.is_dm_shaped and self._participant.is_coordinator_for(project):
            if self._participant.id not in payload.expects_response_from:
                return

        is_first_user_message = (
            self._participant.is_coordinator_for(project) and
            payload.chatroom_name == "user-communication" and
            project.get_chatroom(payload.chatroom_name).count_messages() == 1 and
            not project.is_dm_shaped  # Skip for DM AND Front Desk ends
        )

        if is_first_user_message:
            self._woken_version[project_id] = entry.version
            context_files = project.get_context_files()

            logger.info(
                f"First user request for coordinator {self._participant.name} "
                f"in project {project_id[:8]}, context files: {context_files}"
            )

            message = ChatMessage.from_message_payload(payload)
            await self._participant.on_first_user_request(
                project_id=project_id,
                chatroom_name=payload.chatroom_name,
                message=message,
                context_files=context_files,
                trigger_version=entry.version,
            )
        else:
            addressed = self._participant.id in payload.expects_response_from
            if addressed:
                # Recorded BEFORE the await, not after: `on_message` runs the
                # turn inline and can take minutes, and the record has to be
                # visible to `on_sync_complete` even if that turn raises.
                self._woken_version[project_id] = entry.version
            message = ChatMessage.from_message_payload(payload)
            await self._participant.on_message(
                project_id=project_id,
                chatroom_name=payload.chatroom_name,
                message=message,
                addressed_to_me=addressed,
                trigger_version=entry.version,
            )

    async def _notify_file_created(
        self,
        entry: ChangelogEntry,
        project_id: str,
    ) -> None:
        """Notify of file creation."""
        payload = self._extract_payload(entry, FilePayload)

        if self._is_external_in(project_id, payload.chatroom_name):
            return

        await self._participant.on_file_created(
            project_id=project_id,
            chatroom_name=payload.chatroom_name,
            filename=payload.filename,
            content=base64.b64decode(payload.content_b64),
            trigger_version=entry.version,
        )

    async def _notify_file_updated(
        self,
        entry: ChangelogEntry,
        project_id: str,
    ) -> None:
        """Notify of file update."""
        payload = self._extract_payload(entry, FilePayload)

        if self._is_external_in(project_id, payload.chatroom_name):
            return

        await self._participant.on_file_updated(
            project_id=project_id,
            chatroom_name=payload.chatroom_name,
            filename=payload.filename,
            content=base64.b64decode(payload.content_b64),
            trigger_version=entry.version,
        )

    def _is_external_in(self, project_id: str, chatroom_name: str) -> bool:
        """True iff this participant's PARTICIPANTS row in this chatroom carries ``external=true``.

        Foreign agents invited into another user's project's chatrooms get this
        flag; their notifier skips processing locally because the FD tunnel is
        delivering the same message to their FD project where they handle it.
        Returns False for missing chatrooms / legacy rows (safe default).
        """
        try:
            chatroom = Chatroom.get(project_id, chatroom_name, self._participant._model_ctx)
        except (ValueError, AttributeError):
            return False
        if chatroom is None:
            return False
        return chatroom.is_external_member(self._participant.id)

    async def _notify_project_completed(
        self,
        entry: ChangelogEntry,
        project_id: str,
    ) -> None:
        """Notify of project completion."""
        await self._participant.on_project_completed(
            project_id=project_id,
        )

    async def _notify_batch_complete(
        self,
        entry: ChangelogEntry,
        project_id: str,
    ) -> None:
        """Notify coordinator of batch completion from changelog."""
        payload = self._extract_payload(entry, BatchCompletePayload)

        # Only notify the coordinator who initiated this batch
        if payload.coordinator_id != self._participant.id:
            return

        self._woken_version[project_id] = entry.version
        await self._participant.on_batch_complete(
            project_id=project_id,
            chatroom_name=payload.chatroom_name,
            message_id=payload.message_id,
            responded_participants=payload.responded_participants,
            trigger_version=entry.version,
        )

    async def _notify_batch_timeout(
        self,
        entry: ChangelogEntry,
        project_id: str,
    ) -> None:
        """Notify coordinator of batch timeout from changelog."""
        payload = self._extract_payload(entry, BatchTimeoutPayload)

        # Only notify the coordinator who initiated this batch
        if payload.coordinator_id != self._participant.id:
            return

        self._woken_version[project_id] = entry.version
        await self._participant.on_batch_timeout(
            project_id=project_id,
            chatroom_name=payload.chatroom_name,
            message_id=payload.message_id,
            responded_participants=payload.responded_participants,
            timed_out_participants=payload.timed_out_participants,
            trigger_version=entry.version,
        )

    async def _notify_plan_state(
        self,
        entry: ChangelogEntry,
        project_id: str,
    ) -> None:
        """Forward ``PROJECT_PLAN_STATE`` to the participant.

        This subscriber is a router: it names the EVENT and lets the
        participant decide what the event means, exactly as
        :meth:`_notify_batch_complete` does. Whether this particular entry
        RELEASED the plan gate — and so whether a stalled coordinator should
        resume — needs the gate predicates and a memo of the previous value,
        both of which belong with the participant that owns a ``ModelContext``.

        Until now there was no arm at all: ``ModelContext`` wrote the fields
        into ``meta.json`` and the project sat still until some unrelated event
        woke it, which is how a plan whose last blocking note the user had just
        resolved could stop moving for good.

        **Records; does not dispatch.** The decision is deferred to
        :meth:`on_sync_complete` because it depends on what ELSE the batch
        delivers, and this entry — deliberately given the lower version so the
        turn it wakes reads the post-resolution count — is seen before that is
        known. Dispatching here fires a second, redundant coordinator turn on
        every route that publishes the projection and posts a message in one
        ``append_batch``.
        """
        self._extract_payload(entry, ProjectPlanStatePayload)
        self._plan_state_version[project_id] = entry.version

    async def on_state_loaded(
        self,
        project_id: str,
        project_name: str,
    ) -> None:
        """Let the participant record the plan gate as this process resumes it.

        The one read of project state guaranteed to predate every entry this
        runloop will apply, which is exactly what a *"did this entry open the
        gate"* comparison needs and what ``on_entry`` structurally cannot give:
        ModelContext (priority 0) has already applied the entry by the time this
        subscriber sees it.
        """
        await self._participant.on_plan_gate_prime(project_id)

    async def on_sync_complete(
        self,
        project_id: str,
        project_name: str,
    ) -> None:
        """Dispatch the plan-gate resume, unless a later entry already woke us.

        **The comparison is on versions, not on presence, and both directions
        matter.**

        *Suppress when the waking entry came after.* ``submit_review`` puts the
        projection in the prelude of the same ``append_batch`` as the messages —
        precisely so the turn those messages start reads the post-resolution
        note count. That turn is the intended one. Dispatching a resume as well
        spends a second LLM invocation on the same instruction and gives the
        milestone workroom two chances to be opened.

        *Do NOT suppress when the message came before.* A user message at a
        lower version than the projection started its turn against the
        PRE-resolution count — it is the turn the gate NO_OPs, and it is the
        stall this resume exists to rescue. Guarding on mere co-membership in a
        batch would swallow exactly that case.

        Both halves are popped whether or not either is used. A batch that
        raises mid-way never reaches this method at all, and what carries over
        to the next one is then the right answer anyway: versions are monotonic,
        so a comparison across two batches decides the same way it would have
        within one.
        """
        plan_version = self._plan_state_version.pop(project_id, None)
        woken_version = self._woken_version.pop(project_id, None)
        if plan_version is None:
            return
        if woken_version is not None and woken_version > plan_version:
            return
        await self._participant.on_plan_state_change(
            project_id=project_id,
            trigger_version=plan_version,
        )

    def _extract_payload(self, entry: ChangelogEntry, payload_type: type):
        """Extract and validate payload from entry."""
        if isinstance(entry.payload, payload_type):
            return entry.payload
        if isinstance(entry.payload, dict):
            return payload_type(**entry.payload)
        raise ValueError(f"Unexpected payload type: {payload_type}, got: {entry}")
