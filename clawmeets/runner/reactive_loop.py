# SPDX-License-Identifier: MIT
"""
clawmeets/runner/reactive_loop.py

Reactive control loop for changelog-based agent synchronization.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import httpx

from clawmeets.api.control import AgentSettingsChangePayload, AgentStatusChangePayload, CancelLLMPayload, ChangelogUpdatePayload, ControlMessageType, McpSyncPayload, ProjectDeletedPayload, SkillSyncPayload
from clawmeets.models.agent import Agent
from clawmeets.models.context import ModelContext
from clawmeets.sync.changelog import ChangelogEntry
from clawmeets.sync.git_sandbox import GitSandboxSubscriber
from clawmeets.sync.runloop_manager import ChangelogRunloopManager
from clawmeets.sync.subscriber import ChangelogSubscriber
from clawmeets.utils.file_io import FileUtil
from clawmeets.utils.knowledge_dir import resolve_local_knowledge_dir
from clawmeets.utils.notification_center import LLM_COMPLETE

from .invocation_registry import InvocationRegistry
from .participant_notifier import ParticipantNotifier

if TYPE_CHECKING:
    from clawmeets.api.client import ClawMeetsClient
    from clawmeets.models.participant import Participant
    from clawmeets.runner.mcp_manager import McpManager
    from clawmeets.runner.skill_manager import SkillManager

logger = logging.getLogger("clawmeets.runner")


class ReactiveControlLoop:
    """
    Control loop using per-project runloops via ChangelogRunloopManager.

    This loop:
    1. Receives WebSocket control envelopes
    2. Fetches changelog updates from server (with participant_id for filtering)
    3. Processes entries through per-project runloops:
       - ModelContext writes files (priority 0)
       - ParticipantNotifier fires callbacks (priority 200)
    4. Participant callbacks handle the actual work
    5. Responses are executed via ActionBlockExecutor (configured on participant)

    The key guarantee is that files are ready before callbacks fire.
    """

    def __init__(
        self,
        participant: "Participant",
        client: "ClawMeetsClient",
        model_ctx: ModelContext,
        extra_subscribers: list[ChangelogSubscriber],
        skill_manager: "SkillManager | None" = None,
        mcp_manager: "McpManager | None" = None,
        user_config_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize the reactive control loop.

        Args:
            participant: The agent/assistant this runner represents. The model_ctx
                         should have client configured for HTTP operations.
            client: ClawMeetsClient for HTTP operations (also available via model_ctx.client)
            model_ctx: Shared ModelContext for filesystem I/O (should have client configured)
            extra_subscribers: Additional changelog subscribers to insert between
                               ModelContext and ParticipantNotifier (pass [] if none)
            user_config_dir: Base for resolving relative knowledge_dir values from
                             local_settings (usually ~/.clawmeets/config/<username>/).
                             Hot-update applies this base on AGENT_SETTINGS_CHANGE.
        """
        self._participant = participant
        self._client = client
        self._model_ctx = model_ctx
        self._extra_subscribers = extra_subscribers
        self._skill_manager = skill_manager
        self._mcp_manager = mcp_manager
        self._user_config_dir = user_config_dir

        # Background auto-OAuth tasks for MCP installs. Tracked in a set so
        # asyncio doesn't GC them mid-run; the in-flight set keys by mcp_name
        # so a second trigger (e.g. reconnect catch-up) doesn't double-pop the
        # browser while a first flow is still running.
        self._auto_auth_tasks: set[asyncio.Task] = set()
        self._auto_auth_in_flight: set[str] = set()

        # Per-runner registry of in-flight LLM tasks, keyed by (project_id, room).
        # Surfaced through ModelContext so participants can register their own
        # cli.invoke calls without depending on the runner directly.
        self._invocation_registry = InvocationRegistry()
        self._model_ctx.set_invocation_registry(self._invocation_registry)

        # Create shared notifier (one per participant, shared across projects)
        self._notifier = ParticipantNotifier(participant=self._participant)

        # Build runloop factory that assembles per-project subscribers
        def _make_runloop(pid: str, pname: str, coordinator_id: str):
            subs: list[ChangelogSubscriber] = [
                model_ctx.changelog_subscriber(pid, pname),
            ]
            # Always register GitSandboxSubscriber — it reads git_url
            # from the PROJECT_CREATED payload and no-ops if empty.
            git_sub = GitSandboxSubscriber(
                sandbox_dir=model_ctx.sandbox_dir(pid, pname),
                coordinator_id=coordinator_id,
                participant_id=participant.id,
                project_dir=model_ctx.project_dir(pid, pname),
            )
            model_ctx.notification_center.subscribe(
                LLM_COMPLETE, git_sub._on_llm_complete,
            )
            subs.append(git_sub)
            subs.extend(extra_subscribers)
            subs.append(self._notifier)
            return model_ctx.changelog_dir(pid, pname), subs

        self._runloop_manager = ChangelogRunloopManager(runloop_factory=_make_runloop)

        # Track processing state
        self._running = False

    # ─────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the control loop."""
        self._running = True
        logger.info(f"ReactiveControlLoop started for {self._participant.name}")

    async def stop(self) -> None:
        """Stop the control loop."""
        self._running = False
        await self._runloop_manager.shutdown()
        logger.info(f"ReactiveControlLoop stopped for {self._participant.name}")

    # ─────────────────────────────────────────────────────────
    # Envelope Dispatch
    # ─────────────────────────────────────────────────────────

    async def dispatch(self, envelope) -> None:
        """Route an envelope to the appropriate handler."""
        if not self._running:
            raise RuntimeError("ReactiveControlLoop is not running. Call start() before dispatching envelopes.")

        match envelope.type:
            case ControlMessageType.CHANGELOG_UPDATE:
                # Payload is guaranteed to be ChangelogUpdatePayload by validator
                payload: ChangelogUpdatePayload = envelope.payload
                await self._sync_changelog(payload.project_id, payload.project_name, payload.new_version, payload.coordinator_id)

            case ControlMessageType.AGENT_STATUS_CHANGE:
                # Payload is guaranteed to be AgentStatusChangePayload by validator
                payload: AgentStatusChangePayload = envelope.payload
                logger.info(
                    f"Agent {payload.agent_name} ({payload.agent_id[:8]}...) is now {payload.new_status}"
                )
                # Update local card.json if we have it synced
                agent = Agent.get(payload.agent_id, self._model_ctx)
                if agent is not None:
                    agent.update_card(status=payload.new_status)
                    logger.debug(f"Updated local card for {payload.agent_name}: status={payload.new_status}")

            case ControlMessageType.PROJECT_DELETED:
                payload: ProjectDeletedPayload = envelope.payload
                logger.info(f"Project {payload.project_name} ({payload.project_id[:8]}...) was deleted")
                await self._cleanup_local_project(payload.project_id, payload.project_name)

            case ControlMessageType.SKILL_SYNC:
                payload: SkillSyncPayload = envelope.payload
                if self._skill_manager:
                    if payload.action == "install" and payload.skill_content:
                        self._skill_manager.install_skill(payload.skill_name, payload.skill_content)
                    elif payload.action == "uninstall":
                        self._skill_manager.uninstall_skill(payload.skill_name)
                else:
                    logger.warning("Received SKILL_SYNC but no SkillManager configured")

            case ControlMessageType.MCP_SYNC:
                payload: McpSyncPayload = envelope.payload
                if self._mcp_manager:
                    if payload.action == "install" and payload.manifest:
                        self._mcp_manager.install_mcp(payload.mcp_name, payload.manifest)
                        # If the manifest needs auth and there's no token yet,
                        # pop the browser now so the user doesn't have to hop
                        # to a terminal after clicking Install in the web UI.
                        if self._mcp_manager.needs_auth(payload.mcp_name):
                            self._spawn_auto_auth(payload.mcp_name, payload.manifest)
                    elif payload.action == "uninstall":
                        self._mcp_manager.uninstall_mcp(payload.mcp_name)
                else:
                    logger.warning("Received MCP_SYNC but no McpManager configured")

            case ControlMessageType.AGENT_SETTINGS_CHANGE:
                payload: AgentSettingsChangePayload = envelope.payload
                if payload.agent_id == self._participant.id:
                    self._apply_local_settings(payload.local_settings)

            case ControlMessageType.CANCEL_LLM:
                payload: CancelLLMPayload = envelope.payload
                if payload.agent_id != self._participant.id:
                    logger.warning(
                        f"CANCEL_LLM routed to wrong participant: "
                        f"target={payload.agent_id[:8]} self={self._participant.id[:8]} — ignoring"
                    )
                else:
                    cancelled = self._invocation_registry.cancel(
                        payload.project_id, payload.chatroom_name
                    )
                    if cancelled:
                        logger.info(
                            f"CANCEL_LLM: cancelled invocation in "
                            f"project={payload.project_id[:8]} room={payload.chatroom_name}"
                        )
                    else:
                        logger.info(
                            f"CANCEL_LLM: no in-flight invocation for "
                            f"project={payload.project_id[:8]} room={payload.chatroom_name}"
                        )

            case ControlMessageType.ACTIVE_WORK_CHANGE:
                # UI-only signal (typing indicator / sidebar active-work badge).
                # The server broadcasts it to all project participants; runners
                # ignore it and get their state from the changelog instead.
                pass

            case _:
                raise ValueError(f"Unknown control message type: {envelope.type}")

    # ─────────────────────────────────────────────────────────
    # MCP auto-OAuth
    # ─────────────────────────────────────────────────────────

    def auto_auth_pending_mcps(self) -> None:
        """Trigger auto-auth for every installed MCP server that still needs it.

        Called once on reconnect (after ``McpManager.sync_from_server``) so any
        install the server recorded while we were offline — or an auth flow
        that crashed mid-run last session — gets a browser popup now instead of
        waiting for a manual CLI hop.
        """
        if self._mcp_manager is None:
            return
        for mcp_name in self._mcp_manager.installed_mcps():
            if not self._mcp_manager.needs_auth(mcp_name):
                continue
            manifest = self._mcp_manager.get_manifest(mcp_name)
            if manifest is None:
                continue
            self._spawn_auto_auth(mcp_name, manifest)

    def _spawn_auto_auth(self, mcp_name: str, manifest: dict) -> None:
        """Fire a background OAuth task for one MCP server, with dedup."""
        if mcp_name in self._auto_auth_in_flight:
            # Don't double-pop the browser if both MCP_SYNC and reconnect
            # catch-up race on the same server.
            return
        self._auto_auth_in_flight.add(mcp_name)
        task = asyncio.create_task(self._auto_auth_mcp(mcp_name, manifest))
        self._auto_auth_tasks.add(task)

        def _done(t: asyncio.Task) -> None:
            self._auto_auth_tasks.discard(t)
            self._auto_auth_in_flight.discard(mcp_name)

        task.add_done_callback(_done)

    async def _auto_auth_mcp(self, mcp_name: str, manifest: dict) -> None:
        """Kick off OAuth for a freshly installed MCP server.

        Runs the blocking google-auth-oauthlib flow in a worker thread, bounded
        by a 5-minute consent timeout. Any failure is logged with a manual
        fallback command — never raises back into the reactive loop.
        """
        auth = manifest.get("auth") or {}
        method = auth.get("method")
        if method != "google_oauth_installed":
            logger.info(
                f"MCP {mcp_name}: auth method {method!r} is not auto-runnable; "
                f"run `clawmeets mcp auth {mcp_name}` manually when ready."
            )
            return
        scopes = auth.get("scopes") or []
        if not scopes:
            logger.warning(
                f"MCP {mcp_name}: manifest has no scopes; skipping auto-auth"
            )
            return
        if self._mcp_manager is None:
            return
        token_path = self._mcp_manager.token_path(mcp_name)
        logger.info(
            f"MCP {mcp_name}: starting automatic OAuth "
            f"(browser should open on this machine; token → {token_path})"
        )
        fallback = (
            f"clawmeets mcp auth {mcp_name} --agent {self._participant.name}"
        )
        try:
            from clawmeets.mcp.auth.google_oauth import run_installed_flow
            await asyncio.wait_for(
                asyncio.to_thread(run_installed_flow, scopes, token_path),
                timeout=300,
            )
            logger.info(f"MCP {mcp_name}: auto OAuth complete")
        except asyncio.TimeoutError:
            logger.warning(
                f"MCP {mcp_name}: OAuth timed out after 5 min. "
                f"Re-run `{fallback}` when ready."
            )
        except Exception as e:
            logger.warning(
                f"MCP {mcp_name}: auto OAuth failed: {e}. Fallback: `{fallback}`."
            )

    # ─────────────────────────────────────────────────────────
    # Local Settings
    # ─────────────────────────────────────────────────────────

    def _apply_local_settings(self, local_settings: dict) -> None:
        """Apply local_settings changes to runtime components.

        Updates ModelContext.knowledge_dirs and the provider's use_chrome flag
        so the next LLM invocation uses the new settings without restart.
        Also persists changes to local card.json.

        llm_provider / llm_model changes are persisted but NOT swapped live —
        the CLI subclass is bound at process start. Logs a warning that a
        restart is required for those fields.
        """
        # The runner's own card lives at participants_dir/card.json — NOT under
        # participants_dir/agents/, where PersistableParticipant.card_path looks.
        # Using update_card() here would miss the real self-card and instead
        # spawn an orphan agents/unknown-{id}/card.json that contains only
        # local_settings and crashes list_all() on the next startup. Read and
        # write the top-level self-card directly.
        self_card_path = self._model_ctx.participants_dir / "card.json"
        current_card = FileUtil.read(self_card_path, "json") or {}
        prior_settings = current_card.get("local_settings") or {}

        for field in ("llm_provider", "llm_model"):
            if field in local_settings and local_settings[field] != prior_settings.get(field):
                logger.warning(
                    f"{self._participant.name}: {field} changed to "
                    f"{local_settings[field]!r} — restart the runner for the new "
                    f"provider to take effect"
                )

        # Update knowledge_dirs — resolve the same way cli_runner does at
        # startup so a hot update of `./owner` lands on the same folder as
        # the initial load.
        knowledge_dir = local_settings.get("knowledge_dir", "")
        resolved = resolve_local_knowledge_dir(knowledge_dir, self._user_config_dir)
        new_dirs = [resolved] if resolved is not None else []
        self._model_ctx.update_knowledge_dirs(new_dirs)

        # Update use_chrome (no-op on providers that don't support browser tools)
        use_chrome = local_settings.get("use_chrome", False)
        if self._model_ctx.cli is not None:
            self._model_ctx.cli.use_chrome = use_chrome

        # Persist to the runner's own top-level card.json.
        current_card["local_settings"] = local_settings
        FileUtil.write(self_card_path, current_card, "json", atomic=True)

        logger.info(
            f"Applied local_settings for {self._participant.name}: "
            f"knowledge_dir={knowledge_dir!r}, use_chrome={use_chrome}, "
            f"llm_provider={local_settings.get('llm_provider')!r}, "
            f"llm_model={local_settings.get('llm_model')!r}"
        )

    # ─────────────────────────────────────────────────────────
    # Changelog Sync
    # ─────────────────────────────────────────────────────────

    async def _sync_changelog(
        self,
        project_id: str,
        project_name: str,
        new_version: int,
        coordinator_id: str,
    ) -> None:
        """Sync changelog from server and apply entries."""
        runloop = await self._runloop_manager.get_or_create(
            project_id, project_name, coordinator_id=coordinator_id,
        )

        # Skip if we're already up to date
        if new_version <= runloop.last_processed_version:
            return

        # Create fetch callback that queries server with participant filtering
        async def fetch_entries(last_version: int, target_version: int) -> list[ChangelogEntry]:
            data = await self._client.get_changelog(
                project_id=project_id,
                since=last_version,
                participant_id=self._participant.id,
            )
            raw_entries = data.get("entries", [])

            # Parse entries individually to avoid one bad entry breaking the batch
            parsed_entries = []
            for i, e in enumerate(raw_entries):
                entry = ChangelogEntry.model_validate(e)
                parsed_entries.append(entry)

            return parsed_entries

        # Sync using the runloop
        processed = await runloop.sync(
            new_version=new_version,
            fetch_callback=fetch_entries,
        )
        logger.debug(
            f"Synced {processed} entries for project {project_id[:8]}, "
            f"now at version {runloop.last_processed_version}"
        )

    # ─────────────────────────────────────────────────────────
    # Initial Catch-up
    # ─────────────────────────────────────────────────────────

    async def catch_up(self) -> None:
        """Catch up on missed events for projects and agents.

        Fetches worker agents and projects from server, syncing local state.
        The runloop's persisted state ensures already-synced entries are skipped.
        Also reconciles deleted projects (projects that exist locally but not on server).
        """
        # Sync this runner's own local_settings drift. AGENT_SETTINGS_CHANGE
        # only reaches connected runners, so a runner that was offline while
        # the user edited settings in the web UI would otherwise keep using
        # stale values until the next live edit. Run before other catch-ups
        # so subsequent steps see the fresh knowledge_dir / use_chrome.
        await self._sync_self_settings_from_server()

        # Sync worker agents (in case we missed AGENT_STATUS_CHANGE while disconnected)
        # Owner username lets AGENTS.md render owned agents by short name.
        # Usernames cannot contain hyphens, so the first ``-`` unambiguously
        # separates owner from suffix for agent names; a user participant has
        # no hyphen in its name at all.
        runner_name = self._participant.name
        owner_username = runner_name.split("-", 1)[0] if "-" in runner_name else runner_name
        await Agent.sync_from_server(
            ctx=self._model_ctx,
            exclude_ids=set(),
            owner_username=owner_username,
        )

    async def _sync_self_settings_from_server(self) -> None:
        """Re-fetch this runner's own card.json from the server and apply
        any `local_settings` drift via `_apply_local_settings`.

        Closes the offline-while-user-edits gap: AGENT_SETTINGS_CHANGE is a
        live broadcast — runners that were offline when the user saved a
        new `knowledge_dir` (or `use_chrome`, or `llm_provider`) miss it.
        Without this catch-up step, the next runtime call would use stale
        settings until the user happened to edit again with this runner online.
        """
        try:
            resp = await self._client._http.get(f"/agents/{self._participant.id}")
        except Exception as e:
            logger.warning(f"_sync_self_settings: failed to fetch server card ({e})")
            return
        if resp.status_code != 200:
            logger.debug(
                f"_sync_self_settings: GET /agents/{self._participant.id[:8]} "
                f"returned {resp.status_code}; skipping"
            )
            return
        server_settings = (resp.json() or {}).get("local_settings") or {}
        if not server_settings:
            return

        self_card_path = self._model_ctx.participants_dir / "card.json"
        local_card = FileUtil.read(self_card_path, "json") or {}
        local_settings = local_card.get("local_settings") or {}

        if server_settings == local_settings:
            return

        logger.info(
            f"_sync_self_settings: applying drift for {self._participant.name} — "
            f"server={server_settings} local={local_settings}"
        )
        self._apply_local_settings(server_settings)

        # Sync project changelogs
        server_projects = await self._fetch_server_projects()

        for project_id, info in server_projects.items():
            await self._sync_changelog(
                project_id,
                info["name"],
                info["current_version"],
                coordinator_id=info.get("coordinator_id"),
            )

        # Reconcile: clean up local projects that no longer exist on the server
        await self._reconcile_deleted_projects(set(server_projects.keys()))

    async def _fetch_server_projects(self) -> dict[str, dict]:
        """Fetch project info from server.

        Returns dict of {project_id: {name, status, current_version, coordinator_id}}.

        Uses the unified /participants/{id}/projects endpoint which handles
        all participant types (users, agents, assistants).
        """
        projects_data = await self._client.list_projects(self._participant.id)
        return {
            p["id"]: {
                "name": p["name"],
                "status": p.get("status", "active"),
                "current_version": p.get("current_version", 0),
                "coordinator_id": p.get("coordinator_id", ""),
            }
            for p in projects_data
        }

    # ─────────────────────────────────────────────────────────
    # Project Deletion Reconciliation
    # ─────────────────────────────────────────────────────────

    async def _reconcile_deleted_projects(self, server_project_ids: set[str]) -> None:
        """Remove local state for projects that no longer exist on the server.

        Called during catch_up() to handle projects deleted while this runner was offline.
        Scans local metadata directory for project dirs and compares against server.
        """
        metadata_dir = self._model_ctx.metadata_dir
        if not metadata_dir.exists():
            return

        for entry in metadata_dir.iterdir():
            if not entry.is_dir():
                continue
            # Directory names follow the pattern {name}-{uuid}
            # Extract project_id as the last 36 characters (UUID format)
            dir_name = entry.name
            if len(dir_name) < 37 or dir_name[-37] != "-":
                continue
            project_id = dir_name[-36:]
            project_name = dir_name[:-37]

            if project_id not in server_project_ids:
                logger.info(f"Reconciling deleted project: {project_name} ({project_id[:8]}...)")
                await self._cleanup_local_project(project_id, project_name)

    async def _cleanup_local_project(self, project_id: str, project_name: str) -> None:
        """Remove all local state for a deleted project."""
        # Remove runloop from manager
        await self._runloop_manager.remove(project_id)

        # Delete local directories
        dirs_to_delete = [
            self._model_ctx.project_dir(project_id, project_name),
            self._model_ctx.changelog_dir(project_id, project_name),
            self._model_ctx.sandbox_dir(project_id, project_name),
        ]
        for dir_path in dirs_to_delete:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                logger.info(f"Deleted local directory: {dir_path}")
