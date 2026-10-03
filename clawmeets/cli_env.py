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
import sys
from pathlib import Path
from typing import Optional

import typer

from clawmeets.cli_lifecycle import DEFAULT_DATA_DIR, get_current_user
from clawmeets.utils import env_store
from clawmeets.utils.agent_processes import prefixed_name

app = typer.Typer(
    name="env",
    help="Runner-local env-var store (local only, never synced). Paired skill: env-var.",
    no_args_is_help=True,
)

_MASK = "***"


def _echo(payload: dict) -> None:
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


def _fail(error: str) -> None:
    """Every error this group reports is the same JSON shape on stdout, so a
    skill parsing the output never meets a second format."""
    _echo({"status": "error", "error": error})
    raise typer.Exit(1)


def _match(agents_root: Path, name: str) -> list[Path]:
    """Directories that are exactly ``{name}-{id}``. Agent ids carry no hyphen,
    so ``backend`` does not also match ``backend-v2-{id}``."""
    prefix = f"{name}-"
    return [
        d for d in agents_root.iterdir()
        if d.is_dir() and d.name.startswith(prefix) and "-" not in d.name[len(prefix):]
    ]


def _resolve_agent(data_dir: Path, agent: str) -> Path:
    """Find an agent's directory the way ``clawmeets start --agent`` names it.

    Accepted, in order: the exact directory name (what the computer connection
    passes), the short name (``backend`` — prefixed with the signed-in user,
    as agent directories are), or the prefixed name (``alice-backend``).
    """
    agents_root = Path(data_dir).expanduser() / "agents"
    if not agents_root.is_dir():
        _fail(f"no agents directory at {agents_root}")
    exact = agents_root / agent
    if exact.is_dir():
        return exact
    user = get_current_user(Path(data_dir).expanduser())
    names = [prefixed_name(user, agent)] if user else []
    names.append(agent)
    for name in dict.fromkeys(names):
        matches = _match(agents_root, name)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            _fail(
                f"multiple agents match {agent!r}: {sorted(d.name for d in matches)}. "
                f"Pass the full directory name."
            )
    available = sorted(d.name for d in agents_root.iterdir() if d.is_dir())
    _fail(f"no agent matching {agent!r} under {agents_root}. Available: {available}")


def _resolve_base(agent: str, data_dir: Path) -> Path:
    """Resolve the agent-dir whose store to operate on.

    ``--agent`` wins; when omitted, fall back to ``$CLAWMEETS_AGENT_DIR`` so an
    agent can self-manage. Exits with a JSON error if neither is available.
    """
    if agent:
        return _resolve_agent(data_dir, agent)
    env_dir = os.environ.get("CLAWMEETS_AGENT_DIR")
    if env_dir:
        return Path(env_dir)
    _fail("no agent specified and $CLAWMEETS_AGENT_DIR is unset. Pass --agent <name>.")


_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}


def _dotenv_value(raw: str) -> str:
    """The value half of a dotenv line.

    A quoted value runs to its MATCHING closing quote (double quotes expand
    ``\\n``, ``\\t``, ``\\"`` and ``\\\\``; single quotes are literal) and anything
    after it — an inline comment — is ignored. An unquoted value ends at an
    inline `` #`` comment. Raises ``ValueError`` on an unterminated quote.
    """
    v = raw.strip()
    if v[:1] in ('"', "'"):
        quote, out, i = v[0], [], 1
        while i < len(v):
            ch = v[i]
            if ch == quote:
                return "".join(out)
            if quote == '"' and ch == "\\" and i + 1 < len(v):
                nxt = v[i + 1]
                out.append(_ESCAPES.get(nxt, "\\" + nxt))
                i += 2
                continue
            out.append(ch)
            i += 1
        raise ValueError(f"unterminated {quote} quote")
    for marker in (" #", "\t#"):
        cut = v.find(marker)
        if cut != -1:
            v = v[:cut]
    return v.strip()


@app.command("set")
def set_(
    key: str = typer.Argument(..., help="Env var name (^[A-Z_][A-Z0-9_]*$; no CLAWMEETS_ prefix)."),
    value: Optional[str] = typer.Argument(
        None,
        help="Value (not echoed back). Omit it to be prompted, so it stays out of shell history.",
    ),
    value_stdin: bool = typer.Option(
        False, "--value-stdin",
        help="Read the value from stdin (one trailing newline stripped) instead of argv.",
    ),
    agent: str = typer.Option("", "--agent", "-a", help="Agent name/dir; default $CLAWMEETS_AGENT_DIR."),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
) -> None:
    """Set/overwrite one variable in the agent's store (value never echoed).

    Three ways to pass the value: as an argument, on stdin (``--value-stdin`` —
    how the computer connection does it, so a secret never shows in ``ps``), or
    typed at a hidden prompt when neither is given.
    """
    if value is not None and value_stdin:
        _fail("pass the value as an argument or --value-stdin, not both")
    # Resolved BEFORE the prompt, so a wrong --agent fails before you type a
    # secret into it.
    base = _resolve_base(agent, data_dir)
    if value_stdin:
        value = sys.stdin.read()
        if value.endswith("\n"):
            value = value[:-1]
    elif value is None:
        value = typer.prompt(f"Value for {key}", hide_input=True, err=True)
    try:
        env_store.set_var(base, key, value)
    except ValueError as exc:
        _fail(str(exc))
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

    Skips blank lines and ``#`` comments; tolerates a leading ``export``,
    matched quotes and inline comments (see :func:`_dotenv_value`). Invalid/reserved keys are collected under ``skipped``.
    """
    base = _resolve_base(agent, data_dir)
    if not file.exists():
        _fail(f"file not found: {file}")

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
        key, _, rest = line.partition("=")
        key = key.strip()
        try:
            value = _dotenv_value(rest)
        except ValueError as exc:
            skipped.append({"key": key, "reason": str(exc)})
            continue
        try:
            env_store.set_var(base, key, value)
        except ValueError as exc:
            skipped.append({"key": key, "reason": str(exc)})
            continue
        imported.append(key)
    _echo({"status": "ok", "imported": sorted(imported), "skipped": skipped})
