# SPDX-License-Identifier: MIT
"""
clawmeets/runner/mcp_manager.py

Manages the MCP-hub directory for an agent.

Mirrors SkillManager: caches manifests installed via the ClawMeets MCP Hub,
owns the per-server runtime state directory (including OAuth tokens), and
renders Claude Code's `.mcp.json` into a sandbox directory just before each
LLM invocation.
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


class McpManager:
    """
    Manages a local mcp-hub directory for an agent.

    Directory layout:
        {agent_dir}/mcp-hub/
        ├── manifests/
        │   ├── gmail.json
        │   └── google-calendar.json
        └── servers/
            ├── gmail/
            │   └── token.json
            └── google-calendar/
                └── token.json

    The manifest (`launch` + `auth` spec) is cached per installed server. The
    `servers/{name}/` directory is where each server keeps its runtime state —
    OAuth tokens, caches, etc. Tokens never leave this directory.
    """

    def __init__(self, agent_dir: Path) -> None:
        self.mcp_hub_dir = agent_dir / "mcp-hub"
        self.manifests_dir = self.mcp_hub_dir / "manifests"
        self.servers_dir = self.mcp_hub_dir / "servers"
        self._ensure_structure()

    def _ensure_structure(self) -> None:
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        self.servers_dir.mkdir(parents=True, exist_ok=True)

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

    def install_mcp(self, mcp_name: str, manifest: dict) -> None:
        """Cache an MCP manifest locally and ensure its server state directory exists."""
        manifest_path = self.manifests_dir / f"{mcp_name}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        (self.servers_dir / mcp_name).mkdir(parents=True, exist_ok=True)
        logger.info(f"Installed MCP: {mcp_name}")

    def uninstall_mcp(self, mcp_name: str) -> None:
        """Remove both the cached manifest and the server state directory."""
        manifest_path = self.manifests_dir / f"{mcp_name}.json"
        if manifest_path.exists():
            manifest_path.unlink()
        server_dir = self.servers_dir / mcp_name
        if server_dir.exists():
            shutil.rmtree(server_dir)
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

    # ---------- .mcp.json rendering ----------

    def render_mcp_json(self, sandbox_dir: Path) -> None:
        """Write {sandbox_dir}/.mcp.json from installed manifests.

        Claude Code reads `.mcp.json` from its working directory on startup,
        so each per-project sandbox gets its own copy. Servers that need
        authentication but don't have a token yet are skipped with a warning
        — listing them in .mcp.json would make Claude fail on launch.

        If no MCP servers are usable, no file is written (and any stale file
        is removed so Claude doesn't pick up outdated config).
        """
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        target = sandbox_dir / ".mcp.json"

        servers: dict[str, dict] = {}
        for mcp_name in self.installed_mcps():
            manifest = self.get_manifest(mcp_name)
            if not manifest:
                continue
            if self.needs_auth(mcp_name):
                logger.warning(
                    f"Skipping MCP {mcp_name} in .mcp.json — token missing. "
                    f"Run: clawmeets mcp auth {mcp_name}"
                )
                continue
            launch = manifest.get("launch") or {}
            command = launch.get("command")
            if not command:
                logger.warning(f"MCP {mcp_name} has no launch.command; skipping")
                continue

            token_path = str(self.token_path(mcp_name))
            env = {
                k: v.replace(TOKEN_PLACEHOLDER, token_path) if isinstance(v, str) else v
                for k, v in (launch.get("env") or {}).items()
            }
            args = [
                a.replace(TOKEN_PLACEHOLDER, token_path) if isinstance(a, str) else a
                for a in (launch.get("args") or [])
            ]
            servers[mcp_name] = {
                "command": command,
                "args": args,
                "env": env,
            }

        if not servers:
            if target.exists():
                target.unlink()
            return

        target.write_text(json.dumps({"mcpServers": servers}, indent=2))
