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
    RoomCreatedPayload,
)
from ..sync.subscriber import ChangelogSubscriber
from ..models.chat_message import ChatMessage

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

        # Check for first user message in user-communication (coordinator only)
        # Lazy check: ModelContext (priority 0) writes message before this runs (priority 200)
        # So count == 1 means this is the first message
        project = self._participant.get_project(project_id)

        # Skip assistant callbacks for DM projects - they're handled differently
        if project.is_dm_project and self._participant.is_coordinator_for(project):
            return

        is_first_user_message = (
            self._participant.is_coordinator_for(project) and
            payload.chatroom_name == "user-communication" and
            project.get_chatroom(payload.chatroom_name).count_messages() == 1 and
            not project.is_dm_project  # Skip for DM projects
        )

        if is_first_user_message:
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
            )
        else:
            addressed = self._participant.id in payload.expects_response_from
            message = ChatMessage.from_message_payload(payload)
            await self._participant.on_message(
                project_id=project_id,
                chatroom_name=payload.chatroom_name,
                message=message,
                addressed_to_me=addressed,
            )

    async def _notify_file_created(
        self,
        entry: ChangelogEntry,
        project_id: str,
    ) -> None:
        """Notify of file creation."""
        payload = self._extract_payload(entry, FilePayload)

        await self._participant.on_file_created(
            project_id=project_id,
            chatroom_name=payload.chatroom_name,
            filename=payload.filename,
            content=base64.b64decode(payload.content_b64),
        )

    async def _notify_file_updated(
        self,
        entry: ChangelogEntry,
        project_id: str,
    ) -> None:
        """Notify of file update."""
        payload = self._extract_payload(entry, FilePayload)

        await self._participant.on_file_updated(
            project_id=project_id,
            chatroom_name=payload.chatroom_name,
            filename=payload.filename,
            content=base64.b64decode(payload.content_b64),
        )

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

        await self._participant.on_batch_complete(
            project_id=project_id,
            chatroom_name=payload.chatroom_name,
            message_id=payload.message_id,
            responded_participants=payload.responded_participants,
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

        await self._participant.on_batch_timeout(
            project_id=project_id,
            chatroom_name=payload.chatroom_name,
            message_id=payload.message_id,
            responded_participants=payload.responded_participants,
            timed_out_participants=payload.timed_out_participants,
        )

    def _extract_payload(self, entry: ChangelogEntry, payload_type: type):
        """Extract and validate payload from entry."""
        if isinstance(entry.payload, payload_type):
            return entry.payload
        if isinstance(entry.payload, dict):
            return payload_type(**entry.payload)
        raise ValueError(f"Unexpected payload type: {payload_type}, got: {entry}")
