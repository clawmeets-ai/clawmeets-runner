# SPDX-License-Identifier: MIT
"""
clawmeets/cli_daemon.py

``clawmeets computer …`` — this machine's connection to ClawMeets.

Every subcommand here is a **passthrough** to ``clawmeets-computer``, the entry
point of the separate ``clawmeets-daemon`` distribution. It deliberately
contains no daemon logic at all.

## Why a passthrough and not the daemon itself

The daemon's one promise is that it keeps a computer visible when the runner is
broken — including when the runner is broken *because one of its twenty-odd
dependencies is*. Shipping the daemon inside the runner wheel would make that
promise unkeepable: the same import that fails for an agent would fail for the
thing meant to report the failure. So it is its own distribution with three
dependencies, installed separately.

But it still has to be reachable as ``clawmeets computer …``, because that is
the verb the web UI prints, the system skill shells, and a user types. Two
distributions cannot both provide a ``clawmeets`` console script, so the runner
provides the verb and forwards.

## The visible verb is "computer"

Not "daemon". The product's vocabulary rule is that a user reads about *their
computer*, and a command line is the one place that rule would otherwise be
forced to break. ``clawmeets daemon …`` is registered as a hidden alias so
anything already written against the internal name keeps working; the PyPI
distribution stays ``clawmeets-daemon``, matching the name of its own public
mirror repo, for whoever is reading a package index rather than using the
product. (The runner is the other way round: mirror repo ``clawmeets-runner``,
distribution ``clawmeets``.)

## One account per invocation

Every subcommand takes ``--user``, forwarded like any other flag, and acts for
one clawmeets account: the logged-in one by default. A machine can host several
(the runner has always supported that — ``config/<username>/settings.json``,
``clawmeets start --user alice``), and each pairs separately, with its own key,
its own connection process and its own logs.

## The one subcommand that does more than forward

``install`` bootstraps the daemon package first, because at that moment it is by
definition not installed — the user has a pairing code from the web app and one
command to run. Everything else refuses with an install hint rather than
silently installing software on someone's machine mid-command.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Optional

import typer

app = typer.Typer(
    name="computer",
    help="Connect this computer to ClawMeets so you can see and control its "
         "agents from the web.",
    no_args_is_help=True,
)

DAEMON_DIST = "clawmeets-daemon"
DAEMON_BIN = "clawmeets-computer"
DAEMON_MODULE = "clawmeets_daemon.cli"

_INSTALL_HINT = (
    f"This computer's connection software is not installed yet.\n"
    f"Install it and connect in one step:\n"
    f"  clawmeets computer install --code XXXX-XXXX\n"
    f"(Get a code from the web app: Computers -> +)"
)


def _resolve() -> Optional[list[str]]:
    """How to invoke the daemon on this machine, or None if it is absent.

    Three candidates, in order of how likely each is to be the ONE that works:

    1. ``clawmeets-computer`` on PATH — a normal ``pip`` / ``uv tool`` install.
    2. ``~/.local/bin/clawmeets-computer`` — a ``uv tool`` / ``pipx`` install
       whose bin dir is not on the PATH of whatever shell this is running in.
       This is the common case, not an edge case.
    3. ``python -P -m clawmeets_daemon.cli`` — the daemon is importable in THIS
       interpreter (an editable checkout, or both distributions in one venv).
       Last because it only works when the two share an environment, which is
       exactly the coupling the separate distribution exists to avoid.

    ``-P`` on that last one is not decoration. ``python -m`` puts the CURRENT
    DIRECTORY on ``sys.path``, so a file named like a module the daemon imports
    — ``gettext.py``, ``typing.py``, ``httpx.py`` — sitting in whatever
    directory the user happened to be in shadows the real one and the command
    dies in an import traceback. The other two candidates are console scripts
    and never had this exposure; this one is the only path that does, and a
    daemon whose job is to work when other things are broken should not be
    breakable by a filename. (``-P`` is Python 3.11+, which both distributions
    already require.)
    """
    found = shutil.which(DAEMON_BIN)
    if found:
        return [found]
    for name in (DAEMON_BIN, f"{DAEMON_BIN}.exe"):
        candidate = os.path.expanduser(f"~/.local/bin/{name}")
        if os.path.exists(candidate):
            return [candidate]
    try:
        import importlib.util
        if importlib.util.find_spec(DAEMON_MODULE) is not None:
            return [sys.executable, "-P", "-m", DAEMON_MODULE]
    except (ImportError, ValueError):
        pass
    return None


def _forward(args: list[str], *, allow_missing: bool = False) -> None:
    """Run the daemon with ``args`` and exit with ITS exit code.

    Propagating the child's status matters: these commands are shelled by the
    ``control-computer`` system skill, which reads the exit code to decide
    whether to report success. Swallowing a failure would have an assistant tell
    a user their computer is connected when it is not.

    Output is not captured — it goes straight to the caller's terminal, so
    ``clawmeets computer status`` looks identical to ``clawmeets-computer
    status`` and there is no second place formatting can drift.
    """
    argv = _resolve()
    if argv is None:
        if allow_missing:
            return
        typer.echo(_INSTALL_HINT, err=True)
        raise typer.Exit(1)
    result = subprocess.run(argv + args, check=False)
    if result.returncode != 0:
        raise typer.Exit(result.returncode)


def _bootstrap() -> bool:
    """Install ``clawmeets-daemon`` if it is missing. True if it is now present.

    Tries ``uv tool install`` first, then ``pip install``. ``uv`` first because
    it puts the daemon in its OWN environment — which is the whole point: a
    broken runner venv must not be able to take the daemon down with it. The
    ``pip`` fallback lands it alongside the runner, which is worse but still
    works, and is better than refusing to install at all on a machine without
    ``uv``.
    """
    typer.echo(f"Installing {DAEMON_DIST}…")
    attempts: list[list[str]] = []
    uv = shutil.which("uv")
    if uv:
        attempts.append([uv, "tool", "install", DAEMON_DIST])
    attempts.append([sys.executable, "-m", "pip", "install", "--upgrade", DAEMON_DIST])

    for argv in attempts:
        if subprocess.run(argv, check=False).returncode == 0 and _resolve():
            return True
    return _resolve() is not None


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def install(ctx: typer.Context) -> None:
    """Connect this computer to your ClawMeets account.

    Installs the connection software if needed, then pairs this machine:

        clawmeets computer install --code XXXX-XXXX

    Flags are passed through untouched (``--code``, ``--server``, ``--no-start``),
    so this command never has to be kept in step with the daemon's own options —
    a new flag there works here the day it ships.
    """
    if _resolve() is None and not _bootstrap():
        typer.echo(
            f"Could not install {DAEMON_DIST} automatically. Install it by hand "
            f"and try again:\n  uv tool install {DAEMON_DIST}\n"
            f"  # or: pip install {DAEMON_DIST}",
            err=True,
        )
        raise typer.Exit(1)
    _forward(["install", *ctx.args])


def _passthrough(name: str, help_text: str) -> None:
    """Register one forwarding subcommand.

    Built in a loop rather than written out six times because there is genuinely
    nothing per-command here: the daemon owns every option, and a hand-written
    signature would be a second copy of its flags to keep in step.
    """

    @app.command(
        name,
        help=help_text,
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    )
    def _cmd(ctx: typer.Context, _name: str = name) -> None:
        _forward([_name, *ctx.args])


_passthrough("start", "Start this computer's connection in the background.")
_passthrough(
    "stop",
    "Stop this computer's connection. Your agents keep running; the computer's "
    "key stays valid (to revoke it, disconnect the computer in the web app).",
)
_passthrough("status", "Is this computer connected, and what is running on it?")
_passthrough("logs", "Show what this computer's connection has been doing.")
_passthrough("update", "Update this computer's connection software.")
# Has its own subcommands (enable / disable / status). They need no
# registration here: `allow_extra_args` forwards them untouched, which is the
# same reason a new flag on any command above works the day it ships.
_passthrough(
    "autostart",
    "Start this computer's connection automatically when you log in "
    "(enable / disable / status).",
)
