# SPDX-License-Identifier: MIT
"""
clawmeets/api/client.py

HTTP client for ClawMeets server operations.

Centralizes all HTTP endpoint interactions in one place:
- Message posting (including acknowledgments)
- File uploads
- Chatroom creation and mid-flight invites
- Project completion
- Sync operations (agents, projects, changelog)
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from .responses import (
    AgentResponse,
    ChangelogBatch,
    ParticipantProjectResponse,
)

logger = logging.getLogger(__name__)


class ClawMeetsClient:
    """
    HTTP client for ClawMeets server operations.

    Centralizes all HTTP endpoint interactions:
    - Message operations: post_message (normal and acknowledgment)
    - File operations: upload_file
    - Project/Chatroom operations: create_chatroom, invite_to_chatroom,
      complete_project
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
        source_version: int,
        is_ack: bool = False,
    ) -> str:
        """
        Post a message to a chatroom.

        Addressing is done via @mentions in content - the server parses
        @agent-name mentions to determine which agents should respond.

        Args:
            project_id: The project ID
            chatroom_name: Name of the chatroom
            content: Message content (may include @mentions)
            source_version: Version of the changelog entry that triggered this
                reply. Required — every agent-authored message reacts to a
                trigger and must link into the reply chain.
            is_ack: If True, mark as acknowledgment (skipped in batch tracking)

        Returns:
            The message ID assigned by the server
        """
        url = f"{self._base_url}/projects/{project_id}/chatrooms/{chatroom_name}/messages"
        payload: dict[str, Any] = {
            "content": content,
            "source_version": source_version,
            "is_ack": is_ack
        }

        resp = await self._http.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        message_id = data["id"]
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
        source_version: int,
    ) -> None:
        """
        Upload a file to a chatroom.

        Args:
            project_id: The project ID
            chatroom_name: Name of the chatroom
            filename: Remote filename
            content: File content as bytes
            source_version: Version of the changelog entry that triggered this
                update. Required — every agent-authored file event reacts to
                a trigger and must link into the reply chain.
        """
        url = f"{self._base_url}/projects/{project_id}/chatrooms/{chatroom_name}/files/{filename}"
        params: dict[str, Any] = {"source_version": source_version}
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
        source_version: int,
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
            source_version: Version of the changelog entry that triggered this
                room creation. Required — every agent-authored room creation
                reacts to a trigger and must link into the reply chain.

        Returns:
            The chatroom ID assigned by the server
        """
        url = f"{self._base_url}/projects/{project_id}/chatrooms"
        payload: dict[str, Any] = {
            "name": name,
            "participant_names": participant_names,
            "init_message": init_message,
            "source_version": source_version,
        }

        resp = await self._http.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        chatroom_id = data["id"]
        logger.debug(f"Created chatroom {name}: {chatroom_id}")
        return chatroom_id

    async def invite_to_chatroom(
        self,
        project_id: str,
        chatroom_name: str,
        participant_names: list[str],
        message: str,
        source_version: int,
    ) -> None:
        """
        Add agents to an EXISTING chatroom and address them.

        The counterpart to ``create_chatroom`` for a room already running.
        Membership is idempotent server-side, so re-inviting a current member
        just posts the message to them.

        Args:
            project_id: The project ID
            chatroom_name: Name of the existing chatroom
            participant_names: Agent names to add
            message: Message content with @mentions addressing the invitees —
                an un-@mentioned invitee is never triggered
            source_version: Version of the changelog entry that triggered this
                invite. Required — every agent-authored membership change
                reacts to a trigger and must link into the reply chain.
        """
        url = (
            f"{self._base_url}/projects/{project_id}"
            f"/chatrooms/{chatroom_name}/participants"
        )
        payload: dict[str, Any] = {
            "participant_names": participant_names,
            "message": message,
            "source_version": source_version,
        }

        resp = await self._http.post(url, json=payload)
        resp.raise_for_status()
        logger.debug(f"Invited {participant_names} to chatroom {chatroom_name}")

    # ─────────────────────────────────────────────────────────────────────────
    # Project Operations
    # ─────────────────────────────────────────────────────────────────────────

    async def get_project(self, project_id: str) -> dict[str, Any]:
        """Fetch one project as the server currently holds it.

        **The authority, for the one read that cannot use the local replica.**
        A runner normally answers project questions from its own synced
        ``meta.json``, which is the right source: it is what the turn's prompt
        was built from, so a validator reading the same bytes can never
        false-reject a reference the model was shown. The exception is the
        invitable allowlist, which a coordinator may widen *during its own
        turn* — and the ``PROJECT_ALLOWLIST_UPDATED`` entry that carries the
        widening cannot replay into the local copy until the turn releases the
        runloop lock it is holding. See
        :func:`~clawmeets.models.agent.refresh_invitable_agents`, the only
        caller.

        Returns the raw dict rather than a :class:`Project`. The route's own
        docstring says why one cannot be rebuilt from the dump — ``Project`` is
        frozen and carries computed fields that re-read ``meta.json`` through a
        context a reconstructed copy would not have — and the caller wants two
        plain list fields off it, not a model.
        """
        url = f"{self._base_url}/projects/{project_id}"
        resp = await self._http.get(url)
        resp.raise_for_status()
        return resp.json()

    async def complete_project(
        self,
        project_id: str,
        source_version: int,
    ) -> None:
        """
        Mark a project as completed.

        Args:
            project_id: The project ID
            source_version: Version of the changelog entry that triggered
                completion. Required — every agent-authored completion reacts
                to a trigger and must link into the reply chain.
        """
        url = f"{self._base_url}/projects/{project_id}/complete"
        params: dict[str, Any] = {"source_version": source_version}
        resp = await self._http.post(url, params=params)
        resp.raise_for_status()
        logger.debug(f"Marked project {project_id} as completed")

    async def set_project_display_name(
        self,
        project_id: str,
        display_name: str,
        expected_current: Optional[str] = None,
    ) -> None:
        """Rename a DM thread's ``display_name`` (the auto-title mutation).

        Called by the DM'd agent's runloop from the auto-title trigger, awaited
        within the turn (SF3). Authenticated as the agent (the http_client
        carries the agent bearer token) — the server binds authority to the
        project's coordinator, so only the thread's own agent may rename it.

        Args:
            project_id: The DM thread project ID.
            display_name: The model-generated title replacing "New chat".
            expected_current: Optional defensive sentinel; the auto path passes
                the literal ``"New chat"`` so a title that already flipped is a
                server-side no-op (best-effort, not an atomic CAS).
        """
        url = f"{self._base_url}/projects/{project_id}/display-name"
        payload: dict[str, Any] = {"display_name": display_name}
        if expected_current is not None:
            payload["expected_current"] = expected_current
        resp = await self._http.post(url, json=payload)
        resp.raise_for_status()
        logger.debug(f"Renamed project {project_id} display_name -> {display_name!r}")

    # ─────────────────────────────────────────────────────────────────────────
    # Sync Operations
    # ─────────────────────────────────────────────────────────────────────────

    async def list_agents(self) -> list[AgentResponse]:
        """
        List all registered agents.

        Returns:
            List of AgentResponse DTOs.
        """
        url = f"{self._base_url}/agents"
        resp = await self._http.get(url)
        resp.raise_for_status()
        return [AgentResponse.model_validate(item) for item in resp.json()]

    async def list_agents_if_changed(
        self, etag: Optional[str],
    ) -> tuple[Optional[list[AgentResponse]], Optional[str]]:
        """``GET /agents`` revalidated against the roster tag the caller last applied.

        Returns ``(None, etag)`` when the server answers 304 (roster unchanged),
        else ``(agents, new_etag)``. ``new_etag`` is None against a server that
        sends no ETag, which makes every call a full fetch — today's behaviour.
        The caller stores the tag only once it has applied the roster, so a
        failed apply is retried in full next time.
        """
        headers = {"If-None-Match": etag} if etag else {}
        resp = await self._http.get(f"{self._base_url}/agents", headers=headers)
        if resp.status_code == 304:
            return None, etag
        resp.raise_for_status()
        agents = [AgentResponse.model_validate(item) for item in resp.json()]
        return agents, resp.headers.get("ETag")

    async def list_projects(self, participant_id: str) -> list[ParticipantProjectResponse]:
        """
        List projects for a participant.

        Uses the unified /participants/{id}/projects endpoint which handles
        all participant types (users, agents, assistants).

        Args:
            participant_id: The participant's ID

        Returns:
            List of ParticipantProjectResponse DTOs.
        """
        url = f"{self._base_url}/participants/{participant_id}/projects"
        resp = await self._http.get(url)
        resp.raise_for_status()
        return [ParticipantProjectResponse.model_validate(item) for item in resp.json()]

    async def post_mcp_auth_init(
        self,
        agent_id: str,
        mcp_name: str,
        state: str,
        auth_url: str,
    ) -> None:
        """Register a pending MCP OAuth flow with the server.

        Tells the server which agent is awaiting which MCP's consent and
        hands over the authorization URL so the server can push a "Continue
        with Google" link to the agent's owner. The server later forwards
        the resulting code back to this runner over WebSocket.
        """
        url = f"{self._base_url}/agents/{agent_id}/mcps/{mcp_name}/auth-init"
        resp = await self._http.post(
            url, json={"state": state, "auth_url": auth_url}
        )
        resp.raise_for_status()
        logger.debug(f"Registered MCP auth-init for {mcp_name} (state={state[:8]}…)")

    async def post_skill_auth_init(
        self,
        agent_id: str,
        skill_name: str,
        state: str,
        auth_url: str,
    ) -> None:
        """Skill-rail sibling of ``post_mcp_auth_init``."""
        url = f"{self._base_url}/agents/{agent_id}/skills/{skill_name}/auth-init"
        resp = await self._http.post(
            url, json={"state": state, "auth_url": auth_url}
        )
        resp.raise_for_status()
        logger.debug(f"Registered skill auth-init for {skill_name} (state={state[:8]}…)")

    async def post_env_request_answer(
        self,
        agent_id: str,
        request_id: str,
        *,
        ok: bool,
        detail: str,
        keys: list[str],
    ) -> None:
        """Answer one ``AGENT_ENV_REQUEST`` with its outcome and the key names.

        Never carries a value: ``detail`` and ``keys`` name keys only.
        """
        url = f"{self._base_url}/agents/{agent_id}/env/requests/{request_id}"
        resp = await self._http.post(
            url, json={"ok": ok, "detail": detail, "keys": keys}
        )
        resp.raise_for_status()

    async def put_in_flight_turns(
        self,
        agent_id: str,
        turns_wire: list[dict[str, Any]],
    ) -> int:
        """Report the LLM turns still running on this runner; return how many
        the server restored.

        Same list as the WS auth frame's ``in_flight_turns``, for a RESYNC that
        arrives on a socket that never reconnected. Idempotent server-side.
        """
        url = f"{self._base_url}/agents/{agent_id}/in-flight-turns"
        resp = await self._http.put(url, json={"turns": turns_wire})
        resp.raise_for_status()
        return int(resp.json().get("restored", 0))

    async def get_changelog(
        self,
        project_id: str,
        since: int,
        participant_id: str,
    ) -> ChangelogBatch:
        """
        Fetch changelog entries for a project.

        Args:
            project_id: The project ID
            since: Version to fetch entries after
            participant_id: Participant ID for server-side filtering

        Returns:
            ChangelogBatch with version range + parsed entries.
        """
        url = f"{self._base_url}/projects/{project_id}/changelog"
        params = {
            "since": since,
            "participant_id": participant_id,
        }
        resp = await self._http.get(url, params=params)
        resp.raise_for_status()
        return ChangelogBatch.model_validate(resp.json())
