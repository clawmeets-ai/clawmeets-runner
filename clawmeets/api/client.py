# SPDX-License-Identifier: MIT
"""
clawmeets/api/client.py

HTTP client for ClawMeets server operations.

Centralizes all HTTP endpoint interactions in one place:
- Message posting (including acknowledgments)
- File uploads
- Chatroom creation
- Project completion
- Sync operations (agents, projects, changelog)
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class ClawMeetsClient:
    """
    HTTP client for ClawMeets server operations.

    Centralizes all HTTP endpoint interactions:
    - Message operations: post_message (normal and acknowledgment)
    - File operations: upload_file
    - Project/Chatroom operations: create_chatroom, complete_project
    - Sync operations: list_agents, list_projects, get_changelog
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        server_url: str,
    ) -> None:
        """
        Initialize the client.

        Args:
            http_client: Configured HTTP client with auth headers
            server_url: Base URL for the server
        """
        self._http = http_client
        self._base_url = server_url.rstrip("/")

    # ─────────────────────────────────────────────────────────────────────────
    # Message Operations
    # ─────────────────────────────────────────────────────────────────────────

    async def post_message(
        self,
        project_id: str,
        chatroom_name: str,
        content: str,
        is_ack: bool = False,
        source_version: int | None = None,
    ) -> str:
        """
        Post a message to a chatroom.

        Addressing is done via @mentions in content - the server parses
        @agent-name mentions to determine which agents should respond.

        Args:
            project_id: The project ID
            chatroom_name: Name of the chatroom
            content: Message content (may include @mentions)
            is_ack: If True, mark as acknowledgment (skipped in batch tracking)
            source_version: Version of the changelog entry that triggered this reply

        Returns:
            The message ID assigned by the server
        """
        url = f"{self._base_url}/projects/{project_id}/chatrooms/{chatroom_name}/messages"
        payload: dict[str, Any] = {"content": content}

        if is_ack:
            payload["expects_response_from"] = []
            payload["is_ack"] = True

        if source_version is not None:
            payload["source_version"] = source_version

        resp = await self._http.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        message_id = data.get("id", "unknown")
        logger.debug(f"Posted message to {chatroom_name}: {message_id}")
        return message_id

    # ─────────────────────────────────────────────────────────────────────────
    # File Operations
    # ─────────────────────────────────────────────────────────────────────────

    async def upload_file(
        self,
        project_id: str,
        chatroom_name: str,
        filename: str,
        content: bytes,
        source_version: int | None = None,
    ) -> None:
        """
        Upload a file to a chatroom.

        Args:
            project_id: The project ID
            chatroom_name: Name of the chatroom
            filename: Remote filename
            content: File content as bytes
            source_version: Version of the changelog entry that triggered this update
        """
        url = f"{self._base_url}/projects/{project_id}/chatrooms/{chatroom_name}/files/{filename}"
        params: dict[str, Any] = {}
        if source_version is not None:
            params["source_version"] = source_version
        resp = await self._http.put(url, content=content, params=params)
        resp.raise_for_status()
        logger.debug(f"Uploaded file {filename} to {chatroom_name}")

    # ─────────────────────────────────────────────────────────────────────────
    # Chatroom Operations
    # ─────────────────────────────────────────────────────────────────────────

    async def create_chatroom(
        self,
        project_id: str,
        name: str,
        participant_names: list[str],
        init_message: str,
        source_version: int | None = None,
    ) -> str:
        """
        Create a new chatroom in a project.

        Addressing in init_message is done via @mentions - the server parses
        @agent-name mentions to determine which agents should respond.

        Args:
            project_id: The project ID
            name: Chatroom name
            participant_names: List of agent names to invite
            init_message: Initial message content with @mentions to address agents
            source_version: Version of the changelog entry that triggered this room creation

        Returns:
            The chatroom ID assigned by the server
        """
        url = f"{self._base_url}/projects/{project_id}/chatrooms"
        payload: dict[str, Any] = {
            "name": name,
            "participant_names": participant_names,
            "init_message": init_message,
        }
        if source_version is not None:
            payload["source_version"] = source_version

        resp = await self._http.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        chatroom_id = data.get("id", "unknown")
        logger.debug(f"Created chatroom {name}: {chatroom_id}")
        return chatroom_id

    # ─────────────────────────────────────────────────────────────────────────
    # Project Operations
    # ─────────────────────────────────────────────────────────────────────────

    async def complete_project(
        self,
        project_id: str,
        source_version: int | None = None,
    ) -> None:
        """
        Mark a project as completed.

        Args:
            project_id: The project ID
            source_version: Version of the changelog entry that triggered completion
        """
        url = f"{self._base_url}/projects/{project_id}/complete"
        params: dict[str, Any] = {}
        if source_version is not None:
            params["source_version"] = source_version
        resp = await self._http.post(url, params=params)
        resp.raise_for_status()
        logger.debug(f"Marked project {project_id} as completed")

    # ─────────────────────────────────────────────────────────────────────────
    # Sync Operations
    # ─────────────────────────────────────────────────────────────────────────

    async def list_agents(self) -> list[dict[str, Any]]:
        """
        List all registered agents.

        Returns:
            List of agent data dicts
        """
        url = f"{self._base_url}/agents"
        resp = await self._http.get(url)
        resp.raise_for_status()
        return resp.json()

    async def list_assistants(self) -> list[dict[str, Any]]:
        """
        List assistants visible to the authenticated user.

        Returns:
            List of assistant data dicts (typically just the caller's own assistant)
        """
        url = f"{self._base_url}/assistants"
        resp = await self._http.get(url)
        resp.raise_for_status()
        return resp.json()

    async def list_projects(self, participant_id: str) -> list[dict[str, Any]]:
        """
        List projects for a participant.

        Uses the unified /participants/{id}/projects endpoint which handles
        all participant types (users, agents, assistants).

        Args:
            participant_id: The participant's ID

        Returns:
            List of project info dicts with id, name, status, current_version
        """
        url = f"{self._base_url}/participants/{participant_id}/projects"
        resp = await self._http.get(url)
        resp.raise_for_status()
        return resp.json()

    async def get_changelog(
        self,
        project_id: str,
        since: int,
        participant_id: str,
    ) -> dict[str, Any]:
        """
        Fetch changelog entries for a project.

        Args:
            project_id: The project ID
            since: Version to fetch entries after
            participant_id: Participant ID for server-side filtering

        Returns:
            Dict with 'entries' list and other metadata
        """
        url = f"{self._base_url}/projects/{project_id}/changelog"
        params = {
            "since": since,
            "participant_id": participant_id,
        }
        resp = await self._http.get(url, params=params)
        resp.raise_for_status()
        return resp.json()
