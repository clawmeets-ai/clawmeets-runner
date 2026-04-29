# SPDX-License-Identifier: MIT
"""
clawmeets/mcp/servers/homekit_server.py

HomeKit MCP server. Exposes Apple Home automation through the macOS
`shortcuts` CLI. Two tools:
  - list_shortcuts: enumerate every Shortcut the user has built
  - run_shortcut: invoke a Shortcut by name, optionally passing text input

The Shortcuts app is the contract — the user builds scenes/actions there
(Bedtime, Away-mode, Set Bedroom Temp, etc.) and this MCP calls them.
We don't reinvent device primitives or duplicate the Shortcuts editor.

Requires macOS 12+ (Monterey), where `shortcuts` is built-in.
"""
from __future__ import annotations

import platform
import re
import shutil
import subprocess
from typing import Optional

_DISALLOWED_NAME_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_NAME_MAX_LEN = 200
_RUN_TIMEOUT_SECONDS = 60


def _check_platform() -> None:
    if platform.system() != "Darwin":
        raise RuntimeError(
            "homekit MCP requires macOS — the `shortcuts` CLI ships with "
            f"macOS 12+. Current platform: {platform.system()}."
        )
    if shutil.which("shortcuts") is None:
        raise RuntimeError(
            "`shortcuts` CLI not found on PATH. Requires macOS 12 (Monterey) "
            "or later, where the Shortcuts app and CLI ship by default."
        )


def _validate_name(name: str) -> None:
    if not name or not name.strip():
        raise ValueError("Shortcut name cannot be empty.")
    if len(name) > _NAME_MAX_LEN:
        raise ValueError(f"Shortcut name too long (max {_NAME_MAX_LEN} chars).")
    if _DISALLOWED_NAME_CHARS.search(name):
        raise ValueError("Shortcut name contains control characters.")


def main() -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(
            "The `mcp` package is required but missing — the clawmeets runner "
            "should bundle it by default. Try: pip install --upgrade clawmeets"
        ) from exc

    _check_platform()

    mcp = FastMCP("clawmeets-homekit")

    @mcp.tool()
    def list_shortcuts() -> list[str]:
        """List every Shortcut the user has built in the macOS Shortcuts app.

        Each Shortcut is a user-authored automation that may run Apple Home
        scenes, set device states, or perform other macOS actions. Call once
        per session and cache the result in your knowledge dir — the user's
        Shortcut library is stable across sessions.
        """
        result = subprocess.run(
            ["shortcuts", "list"],
            capture_output=True, text=True, timeout=_RUN_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"`shortcuts list` failed (exit {result.returncode}): "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    @mcp.tool()
    def run_shortcut(name: str, input: Optional[str] = None) -> dict:
        """Run a Shortcut by name. Optionally pass `input` as text the
        Shortcut can consume — useful for parameterized Shortcuts like
        "Set Bedroom Temp" that take a number.

        Returns {ok, exit_code, stdout, stderr}. If you're unsure of the
        exact name, call list_shortcuts() first.
        """
        _validate_name(name)
        cmd = ["shortcuts", "run", name, "--output-path", "-"]
        if input is not None:
            cmd.extend(["--input-path", "-"])
        try:
            result = subprocess.run(
                cmd,
                input=input,
                capture_output=True,
                text=True,
                timeout=_RUN_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "exit_code": None,
                "stdout": "",
                "stderr": f"Shortcut '{name}' timed out after {_RUN_TIMEOUT_SECONDS}s.",
            }
        return {
            "ok": result.returncode == 0,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    mcp.run()


if __name__ == "__main__":
    main()
