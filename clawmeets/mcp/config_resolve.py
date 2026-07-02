# SPDX-License-Identifier: MIT
"""
clawmeets/mcp/config_resolve.py
Per-agent MCP config-file resolution.

Each configurable MCP server (gmail, gdrive, mailbox, etc.) accepts a
``config_file: str`` tool parameter that traditionally came from the
prompt's ``MCP CONFIG FILES`` block. That block was removed when the
prompt-builder dropped the per-config-file sidebar (the LLM had no
reason to be in the loop on a path it just passed back verbatim).

This module is the server-side replacement: when ``config_file`` is
empty, derive the path from the runner-injected ``CLAWMEETS_AGENT_DIR``
env var. The runner already writes per-MCP configs at
``{agent_dir}/mcp-hub/configs/<name>.json`` via
``runner/reactive_loop.py::_apply_local_settings`` on every
AGENT_SETTINGS_CHANGE, so the file is in the expected location whenever
the operator has configured the MCP. A missing file produces an empty
return — callers treat that as a clean noop.
"""
from __future__ import annotations

import os
from pathlib import Path


def resolve_mcp_config_path(server_name: str, explicit: str = "") -> str:
    """Return the config-file path for an MCP server's current invocation.

    Resolution order:
      1. ``explicit`` if non-empty — used verbatim, allowing the LLM to
         override (e.g. point at a one-off config) without coordination
         with the runner.
      2. ``$CLAWMEETS_AGENT_DIR/mcp-hub/configs/<server_name>.json`` —
         the runner-managed convention. Returned only when the file
         actually exists so callers can treat an empty return as a clean
         noop (consistent with the prior ``config_file=""`` contract).
      3. Empty string when neither resolves.

    Args:
        server_name: Stable MCP name (matches the registry entry; e.g.
            ``"gmail"``, ``"google-drive"``).
        explicit: Path the LLM passed in via the tool call's
            ``config_file`` arg, if any.
    """
    if explicit:
        return explicit
    agent_dir = os.environ.get("CLAWMEETS_AGENT_DIR")
    if not agent_dir:
        return ""
    candidate = Path(agent_dir) / "mcp-hub" / "configs" / f"{server_name}.json"
    return str(candidate) if candidate.exists() else ""
