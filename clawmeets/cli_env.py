# SPDX-License-Identifier: MIT
"""
clawmeets/cli_env.py — runner-local env-var store management CLI.

Surfaces the per-agent env store (``clawmeets/utils/env_store.py``) as a
``clawmeets env`` command group, paired with the ``env-var`` skill. The store
holds runner-specific secrets/config on *this machine only*; any stored key
becomes an ordinary ``os.environ[...]`` lookup inside any skill subprocess the
runner spawns (injected in ``LLMProvider._build_env``), so there is no per-skill
wiring.

Agent selection: pass ``--agent <name>`` to target one of your agents, or omit
it and the command falls back to ``$CLAWMEETS_AGENT_DIR`` so a running agent can
self-manage its own store. Values are never echoed by ``set``/``import``, and
``list`` masks them unless ``--show-values`` is given.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import typer

from clawmeets.cli_runner import DEFAULT_DATA_DIR, _resolve_agent_dir
from clawmeets.utils import env_store

app = typer.Typer(
    name="env",
    help="Runner-local env-var store (local only, never synced). Paired skill: env-var.",
    no_args_is_help=True,
)

_MASK = "***"


def _echo(payload: dict) -> None:
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


def _resolve_base(agent: str, data_dir: Path) -> Path:
    """Resolve the agent-dir whose store to operate on.

    ``--agent`` wins (resolved by name/id under ``{data_dir}/agents/``); when
    omitted, fall back to ``$CLAWMEETS_AGENT_DIR`` so an agent can self-manage.
    Exits with an error if neither is available.
    """
    if agent:
        return _resolve_agent_dir(data_dir, agent)
    env_dir = os.environ.get("CLAWMEETS_AGENT_DIR")
    if env_dir:
        return Path(env_dir)
    typer.echo(
        "Error: no agent specified and $CLAWMEETS_AGENT_DIR is unset. "
        "Pass --agent <name>.",
        err=True,
    )
    raise typer.Exit(1)


@app.command("set")
def set_(
    key: str = typer.Argument(..., help="Env var name (^[A-Z_][A-Z0-9_]*$; no CLAWMEETS_ prefix)."),
    value: str = typer.Argument(..., help="Value (not echoed back)."),
    agent: str = typer.Option("", "--agent", "-a", help="Agent name/dir; default $CLAWMEETS_AGENT_DIR."),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
) -> None:
    """Set/overwrite one variable in the agent's store (value never echoed)."""
    base = _resolve_base(agent, data_dir)
    try:
        env_store.set_var(base, key, value)
    except ValueError as exc:
        _echo({"status": "error", "error": str(exc)})
        raise typer.Exit(1)
    _echo({"status": "ok", "key": key})


@app.command("get")
def get(
    key: str = typer.Argument(..., help="Env var name to read."),
    agent: str = typer.Option("", "--agent", "-a", help="Agent name/dir; default $CLAWMEETS_AGENT_DIR."),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
) -> None:
    """Print a variable's value, or a ``missing`` status if unset."""
    base = _resolve_base(agent, data_dir)
    store = env_store.read_raw(base)
    if key in store:
        _echo({"status": "ok", "key": key, "value": store[key]})
    else:
        _echo({"status": "missing", "key": key})


@app.command("list")
def list_(
    agent: str = typer.Option("", "--agent", "-a", help="Agent name/dir; default $CLAWMEETS_AGENT_DIR."),
    show_values: bool = typer.Option(False, "--show-values", help="Reveal values (default: masked)."),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
) -> None:
    """List stored keys. Values are masked unless ``--show-values``."""
    base = _resolve_base(agent, data_dir)
    store = env_store.read_raw(base)
    keys = sorted(store)
    values = {k: (store[k] if show_values else _MASK) for k in keys}
    _echo({"status": "ok", "keys": keys, "values": values})


@app.command("unset")
def unset(
    key: str = typer.Argument(..., help="Env var name to remove."),
    agent: str = typer.Option("", "--agent", "-a", help="Agent name/dir; default $CLAWMEETS_AGENT_DIR."),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
) -> None:
    """Remove a variable from the store (idempotent)."""
    base = _resolve_base(agent, data_dir)
    removed = env_store.unset_var(base, key)
    _echo({"status": "ok", "key": key, "removed": removed})


@app.command("import")
def import_(
    file: Path = typer.Argument(..., help="dotenv-style file of KEY=VALUE lines."),
    agent: str = typer.Option("", "--agent", "-a", help="Agent name/dir; default $CLAWMEETS_AGENT_DIR."),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
) -> None:
    """Bulk-load ``KEY=VALUE`` lines (dotenv style). Values never echoed.

    Skips blank lines and ``#`` comments; tolerates a leading ``export`` and
    surrounding quotes. Invalid/reserved keys are collected under ``skipped``.
    """
    base = _resolve_base(agent, data_dir)
    if not file.exists():
        _echo({"status": "error", "error": f"file not found: {file}"})
        raise typer.Exit(1)

    imported: list[str] = []
    skipped: list[dict] = []
    for raw in file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            skipped.append({"line": raw, "reason": "no '='"})
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        try:
            env_store.set_var(base, key, value)
        except ValueError as exc:
            skipped.append({"key": key, "reason": str(exc)})
            continue
        imported.append(key)
    _echo({"status": "ok", "imported": sorted(imported), "skipped": skipped})
