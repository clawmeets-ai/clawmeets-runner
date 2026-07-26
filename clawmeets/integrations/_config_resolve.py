# SPDX-License-Identifier: MIT
"""
clawmeets/integrations/_config_resolve.py

Per-skill config + token-state path self-resolution.

Mirrors ``clawmeets/mcp/config_resolve.py`` for the CLI-shelled skill rail:
each ``clawmeets <thing> <subcmd>`` subcommand accepts ``--config`` /
``--token`` flags, but in the common case the LLM (or the user) hasn't
passed them. The runner already writes per-skill configs at
``$CLAWMEETS_AGENT_DIR/skill-hub/configs/<skill>.json`` via
``runner/reactive_loop.py::_write_through_skill_configs``, and the
``clawmeets <skill> auth`` command writes tokens at
``$CLAWMEETS_AGENT_DIR/skill-hub/state/<skill>/<token_file>``. These
resolvers consolidate the runner-side convention so each CLI subcommand
only needs to call them.
"""
from __future__ import annotations

import os
from pathlib import Path


def resolve_skill_config_path(skill_name: str, explicit: str = "") -> str:
    """Return the config-file path for a skill's current invocation.

    Resolution order:
      1. ``explicit`` if non-empty — used verbatim, allowing the LLM to
         override (e.g. point at a one-off config) without coordination
         with the runner.
      2. ``$CLAWMEETS_AGENT_DIR/skill-hub/configs/<skill>.json`` — the
         runner-managed convention. Returned only when the file actually
         exists so callers can treat an empty return as a clean noop.
      3. Empty string when neither resolves.
    """
    if explicit:
        return explicit
    agent_dir = os.environ.get("CLAWMEETS_AGENT_DIR")
    if not agent_dir:
        return ""
    candidate = Path(agent_dir) / "skill-hub" / "configs" / f"{skill_name}.json"
    return str(candidate) if candidate.exists() else ""


def resolve_skill_token_path(
    skill_name: str,
    token_file: str = "token.json",
    explicit: str = "",
) -> Path:
    """Return the token-state path for a skill's current invocation.

    Resolution order:
      1. ``explicit`` if non-empty — used verbatim.
      2. ``$CLAWMEETS_AGENT_DIR/skill-hub/state/<skill>/<token_file>`` —
         the convention written by ``clawmeets <skill> auth`` (mode 0600).

    Raises ``RuntimeError`` when neither resolves — token paths are not
    optional (config files can be missing for a clean noop; tokens
    cannot, the caller must know where the credentials live).
    """
    if explicit:
        return Path(explicit).expanduser()
    agent_dir = os.environ.get("CLAWMEETS_AGENT_DIR")
    if not agent_dir:
        raise RuntimeError(
            f"Cannot resolve token path for skill {skill_name!r}: "
            f"CLAWMEETS_AGENT_DIR is not set and --token was not passed."
        )
    return Path(agent_dir) / "skill-hub" / "state" / skill_name / token_file
