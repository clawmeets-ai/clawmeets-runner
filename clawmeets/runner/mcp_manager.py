# SPDX-License-Identifier: MIT
"""
clawmeets/runner/mcp_manager.py

Manages the MCP-hub directory for an agent.

Mirrors SkillManager: caches manifests installed via the ClawMeets MCP Hub,
owns the per-server runtime state directory (including OAuth tokens), and
renders provider-specific MCP configs into a persistent `mcp-hub/dist/`
directory whenever the installed set changes. Each LLM provider then
symlinks its own format file into the per-invocation working dir — no
per-invocation file I/O happens here.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clawmeets.api.client import ClawMeetsClient

logger = logging.getLogger("clawmeets.runner")

TOKEN_PLACEHOLDER = "{{token_path}}"
AGENT_DIR_PLACEHOLDER = "{{agent_dir}}"


class McpManager:
    """
    Manages a local mcp-hub directory for an agent.

    Directory layout:
        {agent_dir}/mcp-hub/
        ├── manifests/
        │   └── chess.json
        ├── servers/
        │   └── chess/
        │       └── token.json        (if the MCP needs OAuth)
        └── dist/                     (rendered configs, one per provider)
            ├── .mcp.json                  (Claude format)
            ├── .gemini/settings.json      (Gemini project-scope format)
            └── .codex/config.toml         (Codex project-level format)

    The manifest (`launch` + `auth` spec) is cached per installed server. The
    `servers/{name}/` directory is where each server keeps its runtime state —
    OAuth tokens, caches, etc. Tokens never leave this directory.

    The `dist/` directory is rewritten by ``render_dist()`` whenever
    ``install_mcp`` / ``uninstall_mcp`` / ``sync_from_server`` mutate the
    installed set, mirroring how ``SkillManager`` keeps ``skill-hub/skills/``
    in sync with the server registry.
    """

    def __init__(self, agent_dir: Path) -> None:
        self.mcp_hub_dir = agent_dir / "mcp-hub"
        self.manifests_dir = self.mcp_hub_dir / "manifests"
        self.servers_dir = self.mcp_hub_dir / "servers"
        self.dist_dir = self.mcp_hub_dir / "dist"
        self._ensure_structure()

    def _ensure_structure(self) -> None:
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        self.servers_dir.mkdir(parents=True, exist_ok=True)
        self.dist_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Sync / install / uninstall ----------

    async def sync_from_server(self, client: "ClawMeetsClient", agent_id: str) -> None:
        """Catch-up: fetch installed MCPs from server, download missing manifests, remove extras."""
        try:
            resp = await client._http.get(f"/agents/{agent_id}/mcps")
            resp.raise_for_status()
            data = resp.json()
            server_mcps = set(data.get("installed_mcps", []))
        except Exception as e:
            logger.warning(f"Failed to fetch installed MCPs from server: {e}")
            return

        local_mcps = set(self.installed_mcps())

        for mcp_name in server_mcps - local_mcps:
            try:
                resp = await client._http.get(f"/mcps/{mcp_name}")
                resp.raise_for_status()
                mcp_data = resp.json()
                manifest = mcp_data.get("manifest")
                if manifest:
                    self.install_mcp(mcp_name, manifest)
                    logger.info(f"Synced MCP: {mcp_name}")
            except Exception as e:
                logger.warning(f"Failed to sync MCP {mcp_name}: {e}")

        for mcp_name in local_mcps - server_mcps:
            self.uninstall_mcp(mcp_name)
            logger.info(f"Removed uninstalled MCP: {mcp_name}")

        # Re-render once after the full catch-up so a sync that touched
        # nothing still produces an up-to-date dist (covers the case where
        # placeholders depend on agent_dir and the dist was wiped).
        self.render_dist()

    def install_mcp(self, mcp_name: str, manifest: dict) -> None:
        """Cache an MCP manifest locally, ensure its server state directory exists,
        and refresh the per-provider rendered configs under ``dist/``."""
        manifest_path = self.manifests_dir / f"{mcp_name}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        (self.servers_dir / mcp_name).mkdir(parents=True, exist_ok=True)
        self.render_dist()
        logger.info(f"Installed MCP: {mcp_name}")

    def uninstall_mcp(self, mcp_name: str) -> None:
        """Remove both the cached manifest and the server state directory,
        then refresh the rendered configs under ``dist/``."""
        manifest_path = self.manifests_dir / f"{mcp_name}.json"
        if manifest_path.exists():
            manifest_path.unlink()
        server_dir = self.servers_dir / mcp_name
        if server_dir.exists():
            shutil.rmtree(server_dir)
        self.render_dist()
        logger.info(f"Uninstalled MCP: {mcp_name}")

    # ---------- Query ----------

    def installed_mcps(self) -> list[str]:
        """Return installed MCP server names (sorted)."""
        if not self.manifests_dir.exists():
            return []
        return sorted(
            p.stem for p in self.manifests_dir.iterdir()
            if p.is_file() and p.suffix == ".json"
        )

    def get_manifest(self, mcp_name: str) -> dict | None:
        """Read a cached manifest."""
        manifest_path = self.manifests_dir / f"{mcp_name}.json"
        if not manifest_path.exists():
            return None
        return json.loads(manifest_path.read_text())

    def token_path(self, mcp_name: str) -> Path:
        """Conventional token file path for an MCP server."""
        manifest = self.get_manifest(mcp_name) or {}
        token_file = (manifest.get("auth") or {}).get("token_file", "token.json")
        return self.servers_dir / mcp_name / token_file

    def has_token(self, mcp_name: str) -> bool:
        """Whether an OAuth token file exists for the MCP server."""
        return self.token_path(mcp_name).exists()

    def needs_auth(self, mcp_name: str) -> bool:
        """Whether the MCP server needs authentication but isn't yet authenticated."""
        manifest = self.get_manifest(mcp_name)
        if not manifest:
            return False
        auth = manifest.get("auth") or {}
        if not auth.get("method"):
            return False
        return not self.has_token(mcp_name)

    # ---------- Provider-config rendering ----------

    def _resolved_servers(self) -> dict[str, dict]:
        """Build the canonical ``{name: {command, args, env}}`` map after
        placeholder substitution, with the same skip rules every provider
        agrees on: skip when authentication is required but no token exists,
        skip when the manifest carries no launch command."""
        servers: dict[str, dict] = {}
        for mcp_name in self.installed_mcps():
            manifest = self.get_manifest(mcp_name)
            if not manifest:
                continue
            if self.needs_auth(mcp_name):
                logger.warning(
                    f"Skipping MCP {mcp_name} in rendered configs — token missing. "
                    f"Run: clawmeets mcp auth {mcp_name}"
                )
                continue
            launch = manifest.get("launch") or {}
            command = launch.get("command")
            if not command:
                logger.warning(f"MCP {mcp_name} has no launch.command; skipping")
                continue

            token_path = str(self.token_path(mcp_name))
            agent_dir = str(self.mcp_hub_dir.parent)

            def _render(v):
                if not isinstance(v, str):
                    return v
                return (
                    v.replace(TOKEN_PLACEHOLDER, token_path)
                     .replace(AGENT_DIR_PLACEHOLDER, agent_dir)
                )

            env = {k: _render(v) for k, v in (launch.get("env") or {}).items()}
            args = [_render(a) for a in (launch.get("args") or [])]
            servers[mcp_name] = {
                "command": command,
                "args": args,
                "env": env,
            }
        return servers

    def render_dist(self) -> None:
        """Re-render the three provider-format configs into ``dist/``.

        Called by ``install_mcp`` / ``uninstall_mcp`` / ``sync_from_server``
        so the dist stays in sync with the installed manifests. Each LLM
        provider symlinks the relevant file into its per-invocation working
        dir at spawn time.

        When no MCP servers are usable, all three files are removed so a
        stale config never survives an uninstall.
        """
        self.dist_dir.mkdir(parents=True, exist_ok=True)
        gemini_dir = self.dist_dir / ".gemini"
        codex_dir = self.dist_dir / ".codex"
        gemini_dir.mkdir(parents=True, exist_ok=True)
        codex_dir.mkdir(parents=True, exist_ok=True)

        claude_target = self.dist_dir / ".mcp.json"
        gemini_target = gemini_dir / "settings.json"
        codex_target = codex_dir / "config.toml"

        servers = self._resolved_servers()
        if not servers:
            for target in (claude_target, gemini_target, codex_target):
                if target.exists():
                    target.unlink()
            return

        payload = {"mcpServers": servers}
        claude_target.write_text(json.dumps(payload, indent=2))
        gemini_target.write_text(json.dumps(payload, indent=2))
        codex_target.write_text(_emit_codex_toml(servers))


def _emit_codex_toml(servers: dict[str, dict]) -> str:
    """Emit `[mcp_servers.<name>] command/args/env` TOML blocks for codex.

    Strings are encoded via ``json.dumps`` — TOML basic-string escape rules
    are a subset of JSON-string rules, so a JSON-encoded string is always a
    valid TOML basic string. Arrays become bracketed comma-separated basic
    strings. Env is an inline table of basic-string-to-basic-string entries.
    """
    lines: list[str] = []
    for name in sorted(servers):
        spec = servers[name]
        lines.append(f"[mcp_servers.{json.dumps(name)}]")
        lines.append(f"command = {json.dumps(spec['command'])}")
        args_inner = ", ".join(json.dumps(a) for a in spec.get("args") or [])
        lines.append(f"args = [{args_inner}]")
        env = spec.get("env") or {}
        if env:
            env_inner = ", ".join(
                f"{json.dumps(k)} = {json.dumps(v)}" for k, v in env.items()
            )
            lines.append("env = { " + env_inner + " }")
        else:
            lines.append("env = {}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
