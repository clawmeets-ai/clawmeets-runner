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
from typing import TYPE_CHECKING, Callable, Optional

import httpx

from clawmeets.api.control import AgentRegistryChangePayload, AgentSettingsChangePayload, AgentStatusChangePayload, CancelLLMPayload, ChangelogUpdatePayload, ControlMessageType, KnowledgePackSyncPayload, McpAuthCodePayload, McpSyncPayload, ProjectDeletedPayload, SkillSyncPayload
from clawmeets.models.agent import Agent
from clawmeets.models.context import ModelContext
from clawmeets.sync.changelog import ChangelogEntry
from clawmeets.sync.git_sandbox import GitSandboxSubscriber
from clawmeets.sync.runloop_manager import ChangelogRunloopManager
from clawmeets.sync.subscriber import ChangelogSubscriber
from clawmeets.utils.file_io import FileUtil
from clawmeets.utils.knowledge_dir import resolve_local_knowledge_dir, resolve_local_dwh_dir
from clawmeets.utils.notification_center import LLM_COMPLETE

from .invocation_registry import InvocationRegistry
from .participant_notifier import ParticipantNotifier

if TYPE_CHECKING:
    from clawmeets.api.client import ClawMeetsClient
    from clawmeets.llm.base import LLMProvider
    from clawmeets.models.participant import Participant
    from clawmeets.runner.knowledge_pack_manager import KnowledgePackManager
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
        knowledge_pack_manager: "KnowledgePackManager | None" = None,
        user_config_dir: Optional[Path] = None,
        cli_factory: Optional[Callable[[str, Optional[str]], "LLMProvider"]] = None,
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
        self._knowledge_pack_manager = knowledge_pack_manager
        self._user_config_dir = user_config_dir
        self._cli_factory = cli_factory

        # Background auto-OAuth tasks for MCP installs. Tracked in a set so
        # asyncio doesn't GC them mid-run; the by-name dict lets uninstall
        # cancel an in-flight flow so a subsequent re-install can spawn a
        # fresh one (otherwise the dedup guard skips it forever).
        self._auto_auth_tasks: set[asyncio.Task] = set()
        self._auto_auth_in_flight: set[str] = set()
        self._auto_auth_tasks_by_name: dict[str, asyncio.Task] = {}
        # Serializes write-through to {knowledge_dir}/config.json so
        # multiple MCP slices arriving in one local_settings envelope (or
        # back-to-back envelopes) can't race the read-modify-write.
        self._mcp_config_write_lock = asyncio.Lock()
        # Same shape for skill_configs — the MCP and skill rails are
        # independent, so they get independent locks (avoids serializing
        # one behind the other unnecessarily).
        self._skill_config_write_lock = asyncio.Lock()
        # In the relay flow, an MCP_AUTH_CODE envelope arrives over WS while
        # the auto-auth task is awaiting it. The Future is keyed by the
        # one-shot state token the runner minted in build_authorization_url.
        self._mcp_auth_waiters: dict[str, asyncio.Future[str]] = {}

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
                        from clawmeets.runner.skill_manager import _decode_skill_files
                        sibling_files = _decode_skill_files(payload.skill_files or {})
                        self._skill_manager.install_skill(
                            payload.skill_name,
                            payload.skill_content,
                            files=sibling_files,
                        )
                    elif payload.action == "uninstall":
                        self._skill_manager.uninstall_skill(payload.skill_name)
                else:
                    logger.warning("Received SKILL_SYNC but no SkillManager configured")

            case ControlMessageType.KNOWLEDGE_PACK_SYNC:
                payload: KnowledgePackSyncPayload = envelope.payload
                if self._knowledge_pack_manager:
                    if payload.action == "install" and payload.pack_files is not None:
                        self._knowledge_pack_manager.install_pack(
                            slug=payload.pack_slug,
                            name=payload.pack_name or payload.pack_slug,
                            description=payload.pack_description or "",
                            files=payload.pack_files,
                        )
                    elif payload.action == "uninstall":
                        self._knowledge_pack_manager.uninstall_pack(payload.pack_slug)
                else:
                    logger.warning("Received KNOWLEDGE_PACK_SYNC but no KnowledgePackManager configured")

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
                        # Cancel any in-flight auto-auth before tearing down
                        # local state, so a subsequent re-install can start
                        # a fresh flow.
                        self._cancel_auto_auth(payload.mcp_name)
                        self._mcp_manager.uninstall_mcp(payload.mcp_name)
                else:
                    logger.warning("Received MCP_SYNC but no McpManager configured")

            case ControlMessageType.AGENT_SETTINGS_CHANGE:
                payload: AgentSettingsChangePayload = envelope.payload
                if payload.agent_id == self._participant.id:
                    if payload.local_settings is not None:
                        await self._apply_local_settings(payload.local_settings)
                    if payload.front_desk_invitable_agents is not None:
                        self._apply_front_desk_invitable_agents(
                            payload.front_desk_invitable_agents
                        )
                    if payload.front_desk_invitable_teams is not None:
                        self._apply_front_desk_invitable_teams(
                            payload.front_desk_invitable_teams
                        )

            case ControlMessageType.AGENT_REGISTRY_CHANGE:
                # A peer agent (owned by this runner's owner) was registered,
                # had a peer-visible field updated, or was deleted. Keep our
                # local peer-card cache fresh so the next coordinator turn's
                # `_resolve_invitable_agents_for_prompt` sees live state
                # without waiting for a runner restart.
                payload: AgentRegistryChangePayload = envelope.payload
                if payload.action == "delete":
                    # Surgical prune by id. `sync_from_server` is upsert-only
                    # for the live registry and won't remove the stale card.
                    try:
                        removed = Agent.prune_peer_card(
                            payload.changed_agent_id, self._model_ctx
                        )
                        logger.debug(
                            f"AGENT_REGISTRY_CHANGE (delete) "
                            f"{payload.changed_agent_name}: "
                            f"{'peer card pruned' if removed else 'no local peer card'}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"AGENT_REGISTRY_CHANGE delete prune failed: {e}",
                            exc_info=True,
                        )
                else:
                    runner_name = self._participant.name
                    owner_username = (
                        runner_name.split("-", 1)[0] if "-" in runner_name else runner_name
                    )
                    try:
                        await Agent.sync_from_server(
                            ctx=self._model_ctx,
                            exclude_ids={self._participant.id},
                            owner_username=owner_username,
                        )
                        logger.debug(
                            f"AGENT_REGISTRY_CHANGE ({payload.action}) "
                            f"{payload.changed_agent_name}: peers re-synced"
                        )
                    except Exception as e:
                        logger.warning(
                            f"AGENT_REGISTRY_CHANGE re-sync failed: {e}", exc_info=True
                        )

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

            case ControlMessageType.MCP_AUTH_CODE:
                payload: McpAuthCodePayload = envelope.payload
                if payload.agent_id != self._participant.id:
                    logger.warning(
                        f"MCP_AUTH_CODE routed to wrong participant: "
                        f"target={payload.agent_id[:8]} self={self._participant.id[:8]} — ignoring"
                    )
                else:
                    waiter = self._mcp_auth_waiters.pop(payload.state, None)
                    if waiter is None:
                        logger.warning(
                            f"MCP_AUTH_CODE: no in-flight waiter for state "
                            f"{payload.state[:8]}… (mcp={payload.mcp_name}) — ignoring"
                        )
                    elif not waiter.done():
                        waiter.set_result(payload.code)

            case ControlMessageType.MCP_AUTH_URL_FOR_USER:
                # User-targeted envelope; the server should never route this
                # to a runner. Log and ignore to avoid raising.
                logger.warning(
                    "MCP_AUTH_URL_FOR_USER unexpectedly delivered to runner; ignoring"
                )

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
        self._auto_auth_tasks_by_name[mcp_name] = task

        def _done(t: asyncio.Task) -> None:
            self._auto_auth_tasks.discard(t)
            self._auto_auth_in_flight.discard(mcp_name)
            # Drop the by-name pointer only if it still points at this task —
            # a re-install may have already replaced it.
            if self._auto_auth_tasks_by_name.get(mcp_name) is t:
                self._auto_auth_tasks_by_name.pop(mcp_name, None)

        task.add_done_callback(_done)

    def _cancel_auto_auth(self, mcp_name: str) -> None:
        """Cancel an in-flight auto-auth flow for an MCP server, if any.

        Called on MCP_SYNC uninstall so a subsequent re-install can start a
        fresh flow instead of being deduped against the previous (still
        parked-on-Future) attempt. The `_auto_auth_in_flight` guard is
        cleared synchronously here — the `_done` callback runs later when
        the cancellation lands, and waiter cleanup happens in the relay
        function's `finally` clause.
        """
        task = self._auto_auth_tasks_by_name.pop(mcp_name, None)
        self._auto_auth_in_flight.discard(mcp_name)
        if task is not None and not task.done():
            task.cancel()

    async def _auto_auth_mcp(self, mcp_name: str, manifest: dict) -> None:
        """Kick off OAuth for a freshly installed MCP server.

        Two modes, selected by ``CLAWMEETS_OAUTH_MODE`` (default ``local``):

        - ``local`` — open a browser on the runner machine and bind the OAuth
          callback to localhost. Works only when the user is on the runner.
        - ``relay`` — mint a consent URL pointing at the ClawMeets server's
          ``/oauth/mcp/callback``, push it to the agent's owner via the
          existing WebSocket, and await the resulting code over WS. Works for
          any deployment where the user reaches the runner only via the
          server.

        Either path bounds the consent wait at 10 minutes. Any failure is
        logged with a manual fallback command — never raises back into the
        reactive loop.
        """
        import os
        from clawmeets.mcp.auth.google_oauth import (
            build_authorization_url,
            exchange_code,
            run_installed_flow,
        )

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
        fallback = (
            f"clawmeets mcp auth {mcp_name} --agent {self._participant.name}"
        )
        mode = os.environ.get("CLAWMEETS_OAUTH_MODE", "relay").lower()

        if mode == "relay":
            await self._auto_auth_mcp_relay(
                mcp_name=mcp_name,
                scopes=scopes,
                token_path=token_path,
                fallback=fallback,
            )
            return

        # Default: local browser on the runner machine.
        logger.info(
            f"MCP {mcp_name}: starting automatic OAuth "
            f"(browser should open on this machine; token → {token_path})"
        )
        try:
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

    async def _auto_auth_mcp_relay(
        self,
        mcp_name: str,
        scopes: list[str],
        token_path: Path,
        fallback: str,
    ) -> None:
        """Server-relayed OAuth flow for remote runners.

        Runner mints state + auth URL locally, posts (state, auth_url) to the
        server (which pushes the link to the agent's owner via WebSocket),
        then awaits an MCP_AUTH_CODE envelope keyed by the same state token.
        On receipt, exchanges the code for tokens locally — the server never
        sees the access/refresh tokens.
        """
        import secrets
        from clawmeets.mcp.auth.google_oauth import (
            build_authorization_url,
            exchange_code,
        )

        server_url = self._client._base_url  # already rstripped of "/"
        redirect_uri = f"{server_url}/oauth/mcp/callback"
        state = secrets.token_urlsafe(32)

        try:
            auth_url, code_verifier = build_authorization_url(
                scopes=scopes,
                redirect_uri=redirect_uri,
                state=state,
            )
        except Exception as e:
            logger.warning(
                f"MCP {mcp_name}: failed to build authorization URL: {e}. "
                f"Fallback: `{fallback}`."
            )
            return

        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[str] = loop.create_future()
        self._mcp_auth_waiters[state] = waiter

        try:
            await self._client.post_mcp_auth_init(
                agent_id=self._participant.id,
                mcp_name=mcp_name,
                state=state,
                auth_url=auth_url,
            )
        except Exception as e:
            self._mcp_auth_waiters.pop(state, None)
            logger.warning(
                f"MCP {mcp_name}: failed to register auth-init with server: {e}. "
                f"Fallback: `{fallback}`."
            )
            return

        logger.info(
            f"MCP {mcp_name}: awaiting user consent via server relay "
            f"(state={state[:8]}…; token → {token_path})"
        )

        try:
            code = await asyncio.wait_for(waiter, timeout=600)
        except asyncio.TimeoutError:
            self._mcp_auth_waiters.pop(state, None)
            logger.warning(
                f"MCP {mcp_name}: relay OAuth timed out after 10 min. "
                f"Re-run `{fallback}` when ready."
            )
            return
        finally:
            # Belt-and-suspenders: if dispatch already popped, this is a no-op.
            self._mcp_auth_waiters.pop(state, None)

        try:
            await asyncio.to_thread(
                exchange_code,
                code,
                scopes,
                redirect_uri,
                token_path,
                code_verifier,
            )
            logger.info(f"MCP {mcp_name}: relay OAuth complete")
        except Exception as e:
            logger.warning(
                f"MCP {mcp_name}: code exchange failed: {e}. Fallback: `{fallback}`."
            )

    # ─────────────────────────────────────────────────────────
    # Local Settings
    # ─────────────────────────────────────────────────────────

    async def _apply_local_settings(self, local_settings: dict) -> None:
        """Apply local_settings changes to runtime components.

        Updates ModelContext.knowledge_dirs so the next LLM invocation uses the
        new settings without restart. Also persists changes to local card.json.

        Per-MCP config slices under ``local_settings.mcp_configs`` are written
        through to ``{knowledge_dir}/config.json`` (the file the in-tree MCPs
        actually read at tool-call time). The runner derives each slice's owned
        keys from the cached manifest's ``starter_config``, drops them, then
        splats the new segment — so the user can remove a key in the UI and it
        actually disappears, while hand-edited keys outside any managed slice
        survive untouched.

        llm_provider / llm_model changes are hot-swapped via the cli_factory
        passed in by the runner — the next LLM invocation uses the new CLI.
        In-flight invocations hold the prior CLI by reference and finish on
        it. If the new CLI fails to construct (binary not on PATH, unknown
        provider name), the prior CLI stays in service and the failure is
        logged.
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

        # Hot-swap the LLM CLI if provider or model changed. The factory is
        # passed in by cli_runner.py and closes over plugin_dirs / mcp_manager
        # / agent_env so we don't need to know construction details here.
        new_provider = local_settings.get("llm_provider") or "claude"
        new_model = local_settings.get("llm_model") or None
        prior_provider = prior_settings.get("llm_provider") or "claude"
        prior_model = prior_settings.get("llm_model") or None
        if self._cli_factory is not None and (
            new_provider != prior_provider or new_model != prior_model
        ):
            try:
                new_cli = self._cli_factory(new_provider, new_model)
                self._model_ctx.update_cli(new_cli)
                logger.info(
                    f"{self._participant.name}: hot-swapped LLM to "
                    f"provider={new_provider!r} model={new_model!r}"
                )
            except Exception as e:
                logger.error(
                    f"{self._participant.name}: hot-swap to "
                    f"provider={new_provider!r} model={new_model!r} failed: {e}. "
                    f"Keeping previous CLI in service.",
                    exc_info=True,
                )

        # Update knowledge_dirs — resolve the same way cli_runner does at
        # startup so a hot update of `./owner` lands on the same folder as
        # the initial load.
        knowledge_dir = local_settings.get("knowledge_dir", "")
        resolved = resolve_local_knowledge_dir(knowledge_dir, self._user_config_dir)
        new_dirs = [resolved] if resolved is not None else []
        self._model_ctx.update_knowledge_dirs(new_dirs)

        # Update dwh_dir (personal data warehouse root). Same resolution as
        # the initial load. None clears the prompt block.
        raw_dwh_dir = local_settings.get("dwh_dir", "")
        resolved_dwh = resolve_local_dwh_dir(raw_dwh_dir, self._user_config_dir) if raw_dwh_dir else None
        self._model_ctx.update_dwh_dir(resolved_dwh)

        # Per-MCP config write-through. Diff against prior to avoid rewriting
        # config.json when only knowledge_dir / llm_* changed.
        new_mcp = local_settings.get("mcp_configs") or {}
        prior_mcp = prior_settings.get("mcp_configs") or {}
        changed_mcp = {n: c for n, c in new_mcp.items() if prior_mcp.get(n) != c}
        if changed_mcp:
            await self._write_through_mcp_configs(changed_mcp)

        # Per-skill config write-through. Same shape as MCPs — one file per
        # skill at {agent_dir}/skill-hub/configs/<skill>.json.
        new_skill = local_settings.get("skill_configs") or {}
        prior_skill = prior_settings.get("skill_configs") or {}
        changed_skill = {n: c for n, c in new_skill.items() if prior_skill.get(n) != c}
        if changed_skill:
            await self._write_through_skill_configs(changed_skill)

        # Persist to the runner's own top-level card.json.
        current_card["local_settings"] = local_settings
        FileUtil.write(self_card_path, current_card, "json", atomic=True)

        logger.info(
            f"Applied local_settings for {self._participant.name}: "
            f"knowledge_dir={knowledge_dir!r}, "
            f"dwh_dir={raw_dwh_dir!r}, "
            f"llm_provider={local_settings.get('llm_provider')!r}, "
            f"llm_model={local_settings.get('llm_model')!r}"
        )

    async def _write_through_mcp_configs(self, changed: dict[str, dict]) -> None:
        """Write per-MCP config slices to ``{agent_dir}/mcp-hub/configs/{name}.json``.

        One file per MCP — no shared file, no namespace key. Each file is the
        MCP's full config dict at top level. Writes are atomic and serialized
        under a per-loop lock so multiple slices arriving in one envelope
        don't interleave their fs ops.
        """
        configs_dir = self._model_ctx.mcp_configs_dir
        configs_dir.mkdir(parents=True, exist_ok=True)

        async with self._mcp_config_write_lock:
            written: list[str] = []
            for mcp_name, segment in changed.items():
                path = configs_dir / f"{mcp_name}.json"
                if isinstance(segment, dict):
                    FileUtil.write(path, segment, "json", atomic=True)
                    written.append(mcp_name)
                else:
                    # Falsy / non-dict segment → erase the file. Mirrors the
                    # "uninstall removes the slice" semantics of the shared-
                    # file era.
                    if path.is_file():
                        try:
                            path.unlink()
                        except OSError as e:
                            logger.warning(
                                f"{self._participant.name}: failed to delete "
                                f"{path}: {e}"
                            )

        logger.info(
            f"Wrote MCP config slices for {self._participant.name}: "
            f"{sorted(written)} -> {configs_dir}"
        )

    async def _write_through_skill_configs(self, changed: dict[str, dict]) -> None:
        """Write per-skill config slices to ``{agent_dir}/skill-hub/configs/{name}.json``.

        Same shape as :meth:`_write_through_mcp_configs` — one file per
        skill, top-level dict, atomic write, serialized under a per-loop
        lock. Falsy / non-dict segments unlink the file (mirrors the
        "uninstall removes the slice" semantics).
        """
        configs_dir = self._model_ctx.skill_configs_dir
        configs_dir.mkdir(parents=True, exist_ok=True)

        async with self._skill_config_write_lock:
            written: list[str] = []
            for skill_name, segment in changed.items():
                path = configs_dir / f"{skill_name}.json"
                if isinstance(segment, dict):
                    FileUtil.write(path, segment, "json", atomic=True)
                    written.append(skill_name)
                else:
                    if path.is_file():
                        try:
                            path.unlink()
                        except OSError as e:
                            logger.warning(
                                f"{self._participant.name}: failed to delete "
                                f"{path}: {e}"
                            )

        logger.info(
            f"Wrote skill config slices for {self._participant.name}: "
            f"{sorted(written)} -> {configs_dir}"
        )

    def _apply_front_desk_invitable_agents(self, invitable: list[str]) -> None:
        """Persist a Front-Desk invitable-agents update to the synced agent card.

        The coordinator-prompt builder reads ``self.front_desk_invitable_agents``
        via ``_load_card()`` which resolves to
        ``participants_dir/agents/{name}-{id}/card.json`` — the same location
        ``Agent.sync_from_server`` writes to. Updating the runner's *top-level*
        self-card (where ``_apply_local_settings`` writes its mirror) wouldn't
        reach the property at all. So we go through ``Participant.update_card``
        which routes via ``_save_card`` → the correct synced-card path.

        Safe to call from the live AGENT_SETTINGS_CHANGE handler because
        ``Agent.sync_from_server`` already populated the ``agents/{name}-{id}``
        directory at startup (catch_up runs sync_from_server before any live
        envelope can arrive).
        """
        prior = list(self._participant.front_desk_invitable_agents)
        if prior == list(invitable):
            return
        self._participant.update_card(front_desk_invitable_agents=list(invitable))
        logger.info(
            f"Applied front_desk_invitable_agents for {self._participant.name}: "
            f"prior={prior} new={invitable}"
        )

    def _apply_front_desk_invitable_teams(self, invitable: list[str]) -> None:
        """Persist a Front-Desk invitable-teams update to the synced agent card.

        Sibling of ``_apply_front_desk_invitable_agents`` — both fields live on
        the synced-peer card and compose via OR in the allowlist. Same routing
        through ``update_card`` so subsequent prompt builds see the fresh value.
        """
        prior = list(self._participant.front_desk_invitable_teams)
        if prior == list(invitable):
            return
        self._participant.update_card(front_desk_invitable_teams=list(invitable))
        logger.info(
            f"Applied front_desk_invitable_teams for {self._participant.name}: "
            f"prior={prior} new={invitable}"
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
        # so subsequent steps see the fresh knowledge_dir.
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
        new `knowledge_dir` (or `llm_provider`) miss it. Without this catch-up
        step, the next runtime call would use stale settings until the user
        happened to edit again with this runner online.
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
        # `local_settings` lives only on the runner's TOP-LEVEL self-card
        # (`participants_dir/card.json`) — `Agent.sync_from_server` writes
        # synced-peer cards under `agents/{name}-{id}/`, which doesn't include
        # the runner's own runtime config. So we have to reconcile this field
        # explicitly here.
        # `front_desk_invitable_agents`, in contrast, lives on the synced-peer
        # card (where `_load_card()` reads from), and `Agent.sync_from_server`
        # in catch_up's next step writes the up-to-date server value there
        # automatically. No extra catch-up call needed for that field.
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
        await self._apply_local_settings(server_settings)

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
