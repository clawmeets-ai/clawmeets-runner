# SPDX-License-Identifier: MIT
"""
clawmeets/runner/turn_home.py

The throwaway provider home of one scoped turn (plan 4.3, hardening item 1).

``TurnHome.create`` makes ``~/.clawmeets/turns/<agent>/<nonce>/`` with ``home/``
and ``tmp/`` (mode 0700) and seeds the provider's config dir with a minimal
settings file: no hooks, no MCP. The subprocess gets ``HOME`` / ``TMPDIR``
pointing there plus the provider's own config variable, so it never reads the
shared ``~/.claude`` / ``~/.codex`` / ``~/.gemini`` / ``~/.config/opencode`` (and
their transcripts), and nothing it writes outlives the turn: ``cleanup`` deletes
the whole tree.

``environment`` builds the turn's environment from ``{}``. It never copies the
runner's ``os.environ``; it only reads the few allowlisted variable NAMES out of
it (PATH, LANG, LC_*, TERM, TZ). Everything else is set explicitly.

On macOS the home also carries ``shims/``: stand-ins for system tools the
sandbox refuses to exec, placed first on ``PATH``. Claude Code shells
``security`` at start to read its login from the keychain; with the real tool
denied, the refused spawn crashes it, while "item not found" makes it fall back
to the placeholder key the model-auth proxy gives the turn. The sandbox profile
lets the turn exec these files and never write to them.
"""
from __future__ import annotations

import json
import os
import secrets
import shutil
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Optional

from ..utils import env_store

if TYPE_CHECKING:
    from .turn_scope import TurnScope

# Variables whose VALUE is taken from the runner's environment by name.
PASSTHROUGH_NAMES: tuple[str, ...] = ("PATH", "LANG", "TERM", "TZ")
PASSTHROUGH_PREFIXES: tuple[str, ...] = ("LC_",)

# Provider -> (config variables, path relative to home/). Each variable points
# into the throwaway home; the first one is the dir that gets seeded.
PROVIDER_CONFIG: Mapping[str, tuple[tuple[str, str], ...]] = {
    "claude": (("CLAUDE_CONFIG_DIR", ".claude"),),
    "codex": (("CODEX_HOME", ".codex"),),
    "gemini": (("GEMINI_CLI_HOME", "."), ("XDG_CONFIG_HOME", ".config")),
    "opencode": (("OPENCODE_CONFIG_DIR", ".config/opencode"), ("XDG_CONFIG_HOME", ".config")),
}

# Minimal seeded settings: no hooks, no MCP, nothing that phones home.
_SEED: Mapping[str, tuple[str, str]] = {
    "claude": (".claude/settings.json", json.dumps({"hooks": {}, "enableAllProjectMcpServers": False})),
    "codex": (".codex/config.toml", "# clawmeets scoped turn: no MCP, no hooks\n"),
    "gemini": (".gemini/settings.json", json.dumps({"mcpServers": {}})),
    "opencode": (".config/opencode/opencode.json", json.dumps({"$schema": "https://opencode.ai/config.json", "mcp": {}})),
}

_DIR_MODE = stat.S_IRWXU  # 0700

# name -> script. Each answers like the real tool does when it finds nothing.
MACOS_SHIMS: Mapping[str, str] = {
    "security": (
        "#!/bin/sh\n"
        "echo 'security: SecKeychainSearchCopyNext: The specified item could not be found in the keychain.' >&2\n"
        "exit 44\n"
    ),
}


def turns_root(agent_dir: Path) -> Path:
    """``~/.clawmeets/turns/<agent>/`` for ``~/.clawmeets/agents/<agent>/``."""
    agent_dir = Path(agent_dir)
    return agent_dir.parent.parent / "turns" / agent_dir.name


@dataclass
class TurnHome:
    """One turn's private HOME / TMPDIR / provider config dir."""

    root: Path
    provider: str
    nonce: str
    # Filled by EgressProxy.bind_for_turn: what the turn uses to reach the proxy.
    egress_url: Optional[str] = None
    egress_port: Optional[int] = None
    egress_socket: Optional[Path] = None
    # Filled by the model-auth proxy (2c): base URL / placeholder key env.
    model_env: dict[str, str] = field(default_factory=dict)
    model_nonce: Optional[str] = None

    @property
    def home(self) -> Path:
        return self.root / "home"

    @property
    def tmp(self) -> Path:
        return self.root / "tmp"

    @property
    def mcp_dir(self) -> Path:
        """Per-turn rendered MCP configs (opted-in servers only)."""
        return self.root / "mcp"

    @property
    def shim_dir(self) -> Path:
        """Stand-ins for denied system tools (macOS only; see the module docstring)."""
        return self.root / "shims"

    def shim_paths(self) -> list[Path]:
        """The shim files this home carries; empty when it has none."""
        if not self.shim_dir.is_dir():
            return []
        return sorted(p for p in self.shim_dir.iterdir() if p.name in MACOS_SHIMS)

    @property
    def config_dir(self) -> Optional[Path]:
        spec = PROVIDER_CONFIG.get(self.provider)
        return (self.home / spec[0][1]).resolve() if spec else None

    @classmethod
    def create(cls, agent_dir: Path, provider: str) -> "TurnHome":
        """Create and seed the turn home. Every dir is 0700."""
        base = turns_root(agent_dir)
        base.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
        os.chmod(base, _DIR_MODE)
        nonce = secrets.token_hex(16)
        root = base / nonce
        root.mkdir(mode=_DIR_MODE)
        os.chmod(root, _DIR_MODE)
        root = Path(os.path.realpath(root))
        th = cls(root=root, provider=provider, nonce=nonce)
        for d in (th.home, th.tmp, th.mcp_dir):
            d.mkdir(mode=_DIR_MODE)
        for _var, rel in PROVIDER_CONFIG.get(provider, ()):
            (th.home / rel).mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
        seed = _SEED.get(provider)
        if seed is not None:
            path = th.home / seed[0]
            path.parent.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
            path.write_text(seed[1], encoding="utf-8")
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        if sys.platform == "darwin":
            th.shim_dir.mkdir(mode=_DIR_MODE)
            for name, script in MACOS_SHIMS.items():
                shim = th.shim_dir / name
                shim.write_text(script, encoding="utf-8")
                os.chmod(shim, stat.S_IRUSR | stat.S_IXUSR)
        return th

    def environment(
        self,
        scope: "TurnScope",
        proxy_url: str,
        *,
        agent_id: str = "",
        server_url: str = "",
        agent_dir: Optional[Path] = None,
        extra: Optional[Mapping[str, str]] = None,
    ) -> dict[str, str]:
        """The turn's allowlisted environment, built from ``{}``.

        Order: env.json keys marked ``--scoped-turns`` first (so they can never
        override anything below), then the fixed allowlist, then ``extra`` (the
        provider's own non-secret flags) and the model-auth env.
        """
        env: dict[str, str] = {}
        if agent_dir is not None:
            scoped = env_store.load_scoped(agent_dir)
            env.update({k: v for k, v in scoped.items() if k in scope.inject_env_keys})
        for name in PASSTHROUGH_NAMES:
            if name in os.environ:
                env[name] = os.environ[name]
            else:
                env.pop(name, None)
        for name, value in os.environ.items():
            if name.startswith(PASSTHROUGH_PREFIXES):
                env[name] = value
        env.setdefault("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
        if self.shim_paths():
            env["PATH"] = f"{self.shim_dir}{os.pathsep}{env['PATH']}"
        env["HOME"] = str(self.home)
        env["TMPDIR"] = str(self.tmp)
        env["CLAWMEETS_AGENT_ID"] = agent_id
        env["CLAWMEETS_SERVER_URL"] = server_url
        env["CLAWMEETS_AGENT_TOKEN"] = scope.token or ""
        env["CLAWMEETS_TURN_SCOPE"] = scope.kind
        for name in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "ALL_PROXY", "all_proxy"):
            env[name] = proxy_url
        # Nothing bypasses the proxy; loopback is unreachable in the sandbox anyway.
        env["NO_PROXY"] = ""
        env["no_proxy"] = ""
        for var, rel in PROVIDER_CONFIG.get(self.provider, ()):
            env[var] = str((self.home / rel).resolve())
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        if extra:
            env.update(extra)
        env.update(self.model_env)
        return env

    def cleanup(self) -> None:
        """Delete the whole turn home. Safe to call twice."""
        shutil.rmtree(self.root, ignore_errors=True)
