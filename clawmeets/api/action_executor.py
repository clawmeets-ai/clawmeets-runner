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

        Args:
            action_block: The extracted action block from Claude output
            project_id: The project ID
            sandbox_dir: Sandbox directory where Claude wrote files (isolated from synced data)

        Returns:
            Set of chatroom names that received reply actions.
        """
        replied_chatrooms: set[str] = set()

        for action in action_block.actions:
            action_type = action["type"]

            if action_type == "reply":
                room_name = action["room"]
                await self._client.post_message(
                    project_id=project_id,
                    chatroom_name=room_name,
                    content=action["content"],
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
                    )
                    logger.debug(f"Updated file {file_path} in {room_name}")
                else:
                    logger.warning(f"File not found in sandbox: {sandbox_dir / file_path}")

            elif action_type == "project_completed":
                await self._client.complete_project(project_id=project_id)
                logger.debug(f"Marked project {project_id} as completed")

        return replied_chatrooms
