# SPDX-License-Identifier: MIT
"""
clawmeets/api/action_executor.py
HTTP action execution for Claude Code actions.

This module executes actions parsed from Claude ```actions blocks via HTTP.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from .actions import ActionBlock
    from .client import ClawMeetsClient

logger = logging.getLogger(__name__)


class ActionBlockExecutor:
    """
    Parses Claude action blocks and executes them via HTTP.

    Combines parsing and HTTP emission in one class. Created and owned by
    Agent/Assistant. Executes actions directly instead of returning intermediate
    response objects.

    Role-based action restrictions are enforced at the schema level:
    - Workers use WORKER_ACTION_SCHEMA (reply, update_file only)
    - Coordinators use COORDINATOR_ACTION_SCHEMA (all actions)
    """

    def __init__(self, client: "ClawMeetsClient") -> None:
        """
        Initialize the action block executor.

        Args:
            client: ClawMeetsClient for HTTP operations
        """
        self._client = client

    async def process(
        self,
        action_block: "ActionBlock",
        project_id: str,
        sandbox_dir: Path,
    ) -> set[str]:
        """
        Process an action block and execute actions via HTTP.

        Server-First Sync Architecture:
        Claude writes files to sandbox_dir (isolated working area). This method
        reads file content from sandbox_dir and sends to the server via HTTP.

        4xx failures on individual actions are caught: a one-line failure note
        is posted into `user-communication` so the coordinator sees the failure
        on its next dispatch and can re-plan, and processing continues with the
        next action in the block. 5xx is re-raised — server brokenness should
        be loud.

        Args:
            action_block: The extracted action block from Claude output
            project_id: The project ID
            sandbox_dir: Sandbox directory where Claude wrote files (isolated from synced data)

        Returns:
            Set of chatroom names that received reply actions.
        """
        replied_chatrooms: set[str] = set()
        source_version = action_block.source_version

        for action in action_block.actions:
            action_type = action["type"]

            try:
                if action_type == "reply":
                    room_name = action["room"]
                    await self._client.post_message(
                        project_id=project_id,
                        chatroom_name=room_name,
                        content=action["content"],
                        source_version=source_version,
                    )
                    logger.debug(f"Emitted reply to {room_name}")
                    replied_chatrooms.add(room_name)

                elif action_type == "create_room":
                    name = action["name"]
                    await self._client.create_chatroom(
                        project_id=project_id,
                        name=name,
                        participant_names=action["invite"],
                        init_message=action["init_message"],
                        source_version=source_version,
                    )
                    logger.debug(f"Created chatroom {name}")

                elif action_type == "update_file":
                    file_path = action["file_path"]
                    room_name = action["room"]
                    full_path = sandbox_dir / file_path
                    if not full_path.exists():
                        # Fallback: Claude sometimes writes into the chatroom directory
                        # structure (synced from project_dir) instead of sandbox root
                        fallback_path = sandbox_dir / "chatrooms" / room_name / "files" / file_path
                        if fallback_path.exists():
                            full_path = fallback_path
                            logger.info(f"File found at fallback chatroom path: {fallback_path}")
                    if full_path.exists():
                        await self._client.upload_file(
                            project_id=project_id,
                            chatroom_name=room_name,
                            filename=file_path,
                            content=full_path.read_bytes(),
                            source_version=source_version,
                        )
                        logger.debug(f"Updated file {file_path} in {room_name}")
                    else:
                        logger.warning(f"File not found in sandbox: {sandbox_dir / file_path}")

                elif action_type == "project_completed":
                    await self._client.complete_project(
                        project_id=project_id,
                        source_version=source_version,
                    )
                    logger.debug(f"Marked project {project_id} as completed")

            except httpx.HTTPStatusError as e:
                if e.response.status_code // 100 != 4:
                    # 5xx (or anything non-4xx): server brokenness, not a coordinator
                    # mistake — let the dispatch loop log it loudly.
                    raise
                summary = _format_failure_summary(action, e)
                logger.warning(summary)
                await self._post_failure_note(project_id, summary, source_version)

        return replied_chatrooms

    async def _post_failure_note(
        self,
        project_id: str,
        summary: str,
        source_version: int | None,
    ) -> None:
        """Post a coordinator-authored failure note to user-communication.

        Silent-fails on its own error so a missing/unreachable user-communication
        room can't mask the original action failure (which is already logged).
        """
        try:
            await self._client.post_message(
                project_id=project_id,
                chatroom_name="user-communication",
                content=summary,
                source_version=source_version,
            )
        except Exception:
            logger.exception("Failed to post action-failure note to user-communication")


def _format_failure_summary(action: dict, exc: httpx.HTTPStatusError) -> str:
    """Build a one-line, coordinator-readable summary of a failed action."""
    status = exc.response.status_code
    reason = exc.response.reason_phrase or ""
    try:
        detail = exc.response.json().get("detail", exc.response.text)
    except Exception:
        detail = exc.response.text

    action_type = action.get("type", "<unknown>")
    if action_type == "create_room":
        invite = action.get("invite") or []
        ctx = f"chatroom {action.get('name')!r} (inviting {invite!r})"
    elif action_type == "reply":
        ctx = f"chatroom {action.get('room')!r}"
    elif action_type == "update_file":
        ctx = f"file {action.get('file_path')!r} in chatroom {action.get('room')!r}"
    elif action_type == "project_completed":
        ctx = "project_completed"
    else:
        ctx = action_type

    return (
        f"Action `{action_type}` ({ctx}) failed: "
        f"{status} {reason} — {detail}"
    )
