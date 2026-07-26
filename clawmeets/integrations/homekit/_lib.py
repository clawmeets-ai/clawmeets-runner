# SPDX-License-Identifier: MIT
"""
clawmeets/integrations/homekit/_lib.py

Apple HomeKit / Apple Home control through the macOS ``shortcuts`` CLI.
The user builds the Shortcuts (Bedtime, Away-mode, Set Bedroom Temp); this
module just enumerates and invokes them. macOS 12+ only.
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


def check_platform() -> None:
    """Raise if the runner can't drive macOS Shortcuts."""
    if platform.system() != "Darwin":
        raise RuntimeError(
            "homekit requires macOS — the `shortcuts` CLI ships with "
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


def list_shortcuts() -> list[str]:
    """List every Shortcut the user has built in the macOS Shortcuts app."""
    check_platform()
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


def run_shortcut(name: str, input_text: Optional[str] = None) -> dict:
    """Run a Shortcut by name, optionally passing text input on stdin.

    Returns ``{ok, exit_code, stdout, stderr}``.
    """
    check_platform()
    _validate_name(name)
    cmd = ["shortcuts", "run", name, "--output-path", "-"]
    if input_text is not None:
        cmd.extend(["--input-path", "-"])
    try:
        result = subprocess.run(
            cmd,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=_RUN_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "exit_code": None,
            "stdout": "",
            "stderr": f"Shortcut {name!r} timed out after {_RUN_TIMEOUT_SECONDS}s.",
        }
    return {
        "ok": result.returncode == 0,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
