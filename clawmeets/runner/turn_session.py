# SPDX-License-Identifier: MIT
"""
clawmeets/runner/turn_session.py

Per-turn orchestration of a scoped turn (plan 4.3, "Enforcement -> Runner").

``open_turn`` is the one place a turn becomes scoped:

    compute scope -> (full: nothing happens) -> require a working sandbox ->
    mint the turn token -> TurnHome -> egress listener (+ model-auth nonce,
    commit 2c) -> opted-in MCP render -> yield -> close nonce -> unbind ->
    revoke the grant -> delete the turn home

A scoped turn on a host without a working sandbox raises
:class:`SandboxUnavailable` BEFORE a token is minted; it never falls back to an
unsandboxed run. The yielded :class:`TurnSession` is what providers receive as
``invoke(turn=...)`` (it satisfies ``llm.base.ScopedTurn``).
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator, Callable, Optional

from ..llm.base import SandboxUnavailable
from ..utils import nofollow_fs
from .egress_proxy import EgressProxy
from .host_report import collect_host_report
from .mcp_manager import McpManager, emit_codex_toml
from .repo_push import RepoPusher
from .turn_home import TurnHome, turns_root
from .turn_sandbox import detect_sandbox, kill_macos_turn_leftovers, wrap_command
from .turn_scope import TurnScope, compute_turn_scope

if TYPE_CHECKING:
    from ..api.client import ClawMeetsClient
    from ..models.context import ModelContext
    from ..models.project import Project

logger = logging.getLogger(__name__)

# Provider-generated MCP files a full turn may have left in the sandbox cwd.
# Removed before a scoped turn so a stale link to mcp-hub/dist cannot load.
_STALE_CWD_CONFIGS = (".mcp.json", ".codex/config.toml", ".gemini/settings.json", "opencode.json")


def _real(p: Path) -> Path:
    return Path(os.path.realpath(p))


@dataclass
class TurnSession:
    """A live scoped turn: scope + turn home + sandbox, handed to the provider."""

    scope: TurnScope
    home: TurnHome
    agent_id: str
    server_url: str
    agent_dir: Path
    sandbox_kind: Optional[str] = None
    _closers: list[Callable[[], None]] = field(default_factory=list, repr=False)

    # -- llm.base.ScopedTurn -------------------------------------------------

    @property
    def mcp_dir(self) -> Path:
        return self.home.mcp_dir

    @property
    def scratch_dir(self) -> Path:
        return self.home.tmp

    @property
    def read_roots(self) -> tuple[Path, ...]:
        return self.scope.read_roots

    @property
    def write_roots(self) -> tuple[Path, ...]:
        return self.scope.write_roots

    @property
    def write_pins(self) -> tuple[tuple[int, int], ...]:
        return self.scope.write_pins

    @property
    def grant_id(self) -> Optional[str]:
        return self.scope.grant_id

    def pin_for(self, root: Path) -> Optional[tuple[int, int]]:
        return self.scope.pin_for(root)

    def additional_dirs(self, cwd: Path) -> list[Path]:
        return self.scope.additional_dirs(cwd)

    def environment(self, extra: Optional[dict[str, str]] = None) -> dict[str, str]:
        return self.home.environment(
            self.scope,
            self.home.egress_url or "",
            agent_id=self.agent_id,
            server_url=self.server_url,
            agent_dir=self.agent_dir,
            extra=extra,
        )

    def wrap(self, cmd: list[str], *, cwd: Path, extra_read: list[Path] = ()) -> list[str]:  # type: ignore[assignment]
        return wrap_command(
            cmd, self.scope, self.home,
            cwd=cwd, extra_read=extra_read,
            env_path=self.environment().get("PATH"),
            sandbox=self.sandbox_kind,
        )

    def kill_leftovers(self) -> None:
        """Kill what the turn left running outside its process group. Only
        sandbox-exec needs this: bwrap's pid namespace ends with the turn."""
        if self.sandbox_kind == "sandbox-exec":
            n = kill_macos_turn_leftovers(self.home.root, turns_root(self.agent_dir).parent)
            if n:
                logger.warning(f"turn: killed {n} process(es) the turn left running")

    def is_readable(self, path: Path) -> bool:
        """The in-process (-api) file tools' read check: same roots as the
        sandbox profile, real paths only, denies win."""
        rp = _real(path)
        if any(rp == d or d in rp.parents for d in self.scope.deny_roots):
            return False
        roots = (*self.scope.write_roots, *self.scope.read_roots)
        return any(rp == r or r in rp.parents for r in roots)

    def is_writable(self, path: Path) -> bool:
        rp = _real(path)
        if any(rp == d or d in rp.parents for d in self.scope.deny_roots):
            return False
        return any(rp == r or r in rp.parents for r in self.scope.write_roots)


@dataclass
class OwnerIsolatedTurn:
    """A requester turn the server answered ``sandbox: "owner_isolated"``
    (plan-owner-isolation 5a): the owner's own environment is the boundary,
    so the provider runs it exactly like a full turn (no OS wrap, no egress or
    model-auth proxy, no canary or provider requirement). The grant still
    rides every write the action executor makes, and the update_file read
    confinement to the sandbox still applies."""

    scope: TurnScope
    agent_id: str
    #: Tells the agent to invoke the provider unwrapped (``turn=None``).
    unwrapped = True

    @property
    def grant_id(self) -> Optional[str]:
        return self.scope.grant_id

    @property
    def write_roots(self) -> tuple[Path, ...]:
        return self.scope.write_roots

    @property
    def write_pins(self) -> tuple[tuple[int, int], ...]:
        return self.scope.write_pins


# One egress proxy per runner process; each turn gets its own listener.
_EGRESS: Optional[EgressProxy] = None


def get_egress_proxy() -> EgressProxy:
    global _EGRESS
    if _EGRESS is None:
        _EGRESS = EgressProxy()
    return _EGRESS


def _render_mcp(home: TurnHome, scope: TurnScope, agent_dir: Path) -> None:
    """Opted-in MCP servers only, into the turn's mcp dir and provider home."""
    manager = McpManager(agent_dir)
    servers = manager.render_scoped(home.mcp_dir, scope.mcp_servers)
    if not servers:
        return
    # User-level config in the throwaway home, so no project trust is needed.
    if home.provider == "codex":
        (home.home / ".codex" / "config.toml").write_text(emit_codex_toml(servers))
    elif home.provider == "gemini":
        (home.home / ".gemini").mkdir(exist_ok=True)
        (home.home / ".gemini" / "settings.json").write_text(
            (home.mcp_dir / ".mcp.json").read_text()
        )


def _clear_stale_cwd_configs(cwd: Path, pin: Optional[tuple[int, int]] = None) -> None:
    # The sandbox is turn-writable: a symlinked `.codex` is unlinked itself,
    # never followed into whatever it points at.
    for rel in _STALE_CWD_CONFIGS:
        try:
            nofollow_fs.remove(cwd, rel, pin=pin)
        except OSError:
            logger.warning(f"turn: could not remove stale {cwd / rel}", exc_info=True)


async def _push_after_turn(pusher: "RepoPusher", project: "Project", scope: TurnScope,
                           client, agent_id: str, chatroom: str, grant_id: str) -> None:
    """Push each clone's project branch; tell the room when it was refused."""
    from clawmeets.api.client import CURRENT_TURN_GRANT

    for sandbox in scope.write_roots:
        for clone in pusher.clones_in(sandbox):
            try:
                result = await asyncio.to_thread(pusher.push_project_branch, project.id, project.name, clone)
            except Exception:  # noqa: BLE001 - a failed push must not fail the turn
                logger.warning("turn: project-branch push failed", exc_info=True)
                continue
            if result.reason == "non_fast_forward" and client is not None:
                token = CURRENT_TURN_GRANT.set(grant_id)
                try:
                    await client.post_message(
                        project_id=project.id, chatroom_name=chatroom, source_version=0,
                        content=(f"The runner did not push `{result.branch}`: the remote has "
                                 "commits this clone lacks. Fetch and rebase on your next turn."),
                    )
                except Exception:  # noqa: BLE001
                    logger.warning("turn: push note failed", exc_info=True)
                finally:
                    CURRENT_TURN_GRANT.reset(token)


def _host_report(agent_dir: Path) -> Optional[dict]:
    """A fresh scan for this turn's mint (plan-owner-isolation 5a). A scan
    that fails sends none, which the server answers with the sandbox."""
    try:
        return collect_host_report(agent_dir).model_dump(mode="json")
    except Exception:  # noqa: BLE001
        logger.warning("turn: host report failed; minting without one", exc_info=True)
        return None


@asynccontextmanager
async def open_turn(
    model_ctx: "ModelContext",
    agent: object,
    project: "Project",
    chatroom: str,
    *,
    provider: Optional[str],
    agent_id: str,
    client: Optional["ClawMeetsClient"],
    server_url: str = "",
    egress: Optional[EgressProxy] = None,
    model_auth: object = None,
) -> AsyncIterator[Optional[TurnSession]]:
    """Open the turn's scope. Yields ``None`` for a full turn (no change).

    ``provider`` is the LLM provider's turn key (``None`` = the provider cannot
    run a scoped turn, which refuses it). ``model_auth`` is the model-auth proxy
    (commit 2c): ``attach_turn(home, provider)`` -> env dict, ``close_turn(nonce)``.
    """
    agent_dir = _real(model_ctx.base_dir)
    scoped_mcp = McpManager(agent_dir).scoped_turn_servers()
    scope = compute_turn_scope(agent, project, chatroom, model_ctx, scoped_mcp_servers=scoped_mcp)
    if not scope.scoped:
        yield None
        return

    if scope.kind == "requester" and client is not None:
        # Mint first: the server decides from this turn's fresh host report
        # whether the owner's environment is the boundary (owner isolation)
        # or the OS sandbox is required, which the checks below then demand.
        try:
            grant = await client.mint_turn_token(agent_id, project.id, scope.server_scope,
                                                 host=_host_report(agent_dir))
        except Exception as exc:
            if getattr(getattr(exc, "response", None), "status_code", None) == 409:
                # Neither sandboxable nor owner-isolated: the work stays held.
                raise SandboxUnavailable("this computer cannot serve this conversation safely: "
                                         "no OS sandbox and owner isolation is not in effect") from exc
            raise
        scope = scope.with_grant(grant["token"], grant["grant_id"])
        if grant.get("sandbox") == "owner_isolated":
            logger.info("turn: owner-isolated (the owner's environment is the boundary); running unwrapped")
            try:
                yield OwnerIsolatedTurn(scope=scope, agent_id=agent_id)
            finally:
                await _revoke(client, agent_id, grant["grant_id"])
            return
        try:
            _require_sandbox(provider)
        except SandboxUnavailable:
            await _revoke(client, agent_id, grant["grant_id"])
            raise
        async with _sandboxed(model_ctx, project, chatroom, scope, grant, provider=provider, agent_id=agent_id,
                              client=client, server_url=server_url, egress=egress,
                              model_auth=model_auth) as session:
            yield session
        return

    # A project turn (an old server's remote-content project) never sees the
    # bound remote (it may be the owner's own working copy): it fetches from a
    # runner-owned mirror, read only, and the runner pushes the project branch
    # afterwards (runner/repo_push.py).
    _require_sandbox(provider)
    if client is None:
        raise SandboxUnavailable("no server client to mint a turn token")
    mirror: Optional[Path] = None
    git_url = getattr(model_ctx, "git_url", None)
    pusher: Optional[RepoPusher] = None
    if scope.kind == "project" and git_url:
        pusher = RepoPusher(agent_dir, git_url,
                            base_branch=os.environ.get("CLAWMEETS_AGENT_GIT_BASE_BRANCH") or "main")
        mirror = await asyncio.to_thread(pusher.refresh_mirror, project.id, project.name)
        scope = dataclasses.replace(scope, read_roots=(*scope.read_roots, _real(mirror)))
    grant = await client.mint_turn_token(agent_id, project.id, scope.server_scope,
                                         host=_host_report(agent_dir))
    scope = scope.with_grant(grant["token"], grant["grant_id"])
    async with _sandboxed(model_ctx, project, chatroom, scope, grant, provider=provider, agent_id=agent_id,
                          client=client, server_url=server_url, egress=egress, model_auth=model_auth,
                          pusher=pusher, mirror=mirror) as session:
        yield session


def _require_sandbox(provider: Optional[str]) -> str:
    """The sandbox kind, or SandboxUnavailable: a sandboxed turn needs a
    provider that supports scoped turns and a working OS sandbox."""
    if provider is None:
        raise SandboxUnavailable("this agent's model provider cannot run an isolated turn")
    kind = detect_sandbox()
    if kind is None:
        raise SandboxUnavailable(
            "this computer has no working OS sandbox, so it cannot serve a "
            "conversation for someone other than the owner"
        )
    return kind


async def _revoke(client: "ClawMeetsClient", agent_id: str, grant_id: str) -> None:
    try:
        await client.revoke_turn_token(agent_id, grant_id)
    except Exception:  # noqa: BLE001 - the grant still expires server-side
        logger.warning("turn: grant revoke failed; it expires on its own", exc_info=True)


@asynccontextmanager
async def _sandboxed(
    model_ctx: "ModelContext",
    project: "Project",
    chatroom: str,
    scope: TurnScope,
    grant: dict,
    *,
    provider: Optional[str],
    agent_id: str,
    client: "ClawMeetsClient",
    server_url: str,
    egress: Optional[EgressProxy],
    model_auth: object,
    pusher: Optional[RepoPusher] = None,
    mirror: Optional[Path] = None,
) -> AsyncIterator[TurnSession]:
    """Run a minted turn in the OS sandbox: turn home, egress and model-auth
    proxies, MCP render; tear all of it down and revoke the grant after."""
    agent_dir = _real(model_ctx.base_dir)
    kind = _require_sandbox(provider)
    assert provider is not None
    home: Optional[TurnHome] = None
    egress = egress or get_egress_proxy()
    session: Optional[TurnSession] = None
    try:
        home = TurnHome.create(agent_dir, provider)
        egress.bind_for_turn(home)
        session = TurnSession(
            scope=scope, home=home, agent_id=agent_id, server_url=server_url,
            agent_dir=agent_dir, sandbox_kind=kind,
        )
        if model_auth is not None:
            home.model_env.update(model_auth.attach_turn(home, provider))  # type: ignore[attr-defined]
            nonce = home.model_nonce
            session._closers.append(lambda: model_auth.close_turn(nonce))  # type: ignore[attr-defined]
        if mirror is not None:
            home.model_env["CLAWMEETS_AGENT_GIT_URL"] = f"file://{_real(mirror)}"
        _render_mcp(home, scope, agent_dir)
        for root in scope.write_roots:
            _clear_stale_cwd_configs(root, scope.pin_for(root))
        yield session
    finally:
        if pusher is not None:
            await _push_after_turn(pusher, project, scope, client, agent_id, chatroom, grant["grant_id"])
        if session is not None:
            for close in session._closers:
                try:
                    close()
                except Exception:  # noqa: BLE001
                    logger.warning("turn: closer failed", exc_info=True)
        if home is not None:
            egress.unbind_turn(home)
        await _revoke(client, agent_id, grant["grant_id"])
        if home is not None:
            home.cleanup()
