# SPDX-License-Identifier: BUSL-1.1
"""
clawmeets/cli_doctor.py — ``clawmeets doctor``.

The terminal surface of :mod:`clawmeets.doctor`. All the knowing lives there;
this module only decides what a person should read.

Two shapes, one source. The default is for a human standing in a shell: every
check on one line, and for each failure the exact command to paste, indented
underneath it. ``--json`` is the same report for the install script and any
other program, so there is no second opinion about what "working" means.

The exit code is the contract callers actually branch on: ``0`` when nothing
blocking failed, ``1`` otherwise. Non-blocking findings (a second model CLI that
is installed but signed out) print and do not change it — an install that
genuinely worked must not end in a red exit because the user also happens to
have a stale ``gemini`` on PATH.
"""
from __future__ import annotations

import typer

from clawmeets import doctor

# Marks, not words. A checklist is read by scanning the left edge, and
# "OK"/"FAIL" columns make the reader parse text to find the one line that
# matters. ASCII rather than emoji so it survives a dumb terminal and a CI log.
_PASS = "[ok]"
_FAIL = "[--]"
_WARN = "[..]"


def doctor_command(
    as_json: bool = typer.Option(
        False, "--json", help="Machine-readable report (used by the installer)."
    ),
    server: str = typer.Option(
        "", "--server", "-s", help="Server to test reachability against."
    ),
    skip_server: bool = typer.Option(
        False, "--skip-server", help="Don't test reachability (offline check)."
    ),
    user: str = typer.Option(
        "", "--user", "-u",
        help="Account to check (defaults to this computer's default account).",
    ),
    model_clis: bool = typer.Option(
        False, "--model-clis",
        help="Only the model-CLI findings, as JSON. Used by this computer's "
             "connection to report them to the server.",
    ),
) -> None:
    """Check your setup and print the exact command to fix anything broken.

    Covers the ClawMeets install, your sign-in on this computer, your assistant,
    whether a model CLI is installed AND signed in, whether this computer is
    linked and connected, whether it comes back after a restart, and whether the
    server is reachable.

    Every failure prints a command you can paste. Fix them top-down: the checks
    run in the order you hit them, so the first failure is the earliest thing
    that actually went wrong.
    """
    # A narrower question than the rest of this command, and the reason it is a
    # flag here rather than its own command: the ANSWER must come from the same
    # table doctor prints, or the web checklist and the terminal would disagree
    # about whether a model CLI is usable. Exits 0 regardless — the caller is
    # reporting an observation, not judging a setup.
    if model_clis:
        import json

        typer.echo(json.dumps(doctor.model_cli_wire()))
        raise typer.Exit(0)

    report = doctor.run_checks(
        server_url=server, include_server=not skip_server, username=user,
    )

    if as_json:
        typer.echo(report.to_json())
        raise typer.Exit(0 if report.ok else 1)

    typer.echo("")
    for check in report.checks:
        if check.ok:
            mark = _PASS
        else:
            mark = _FAIL if check.blocking else _WARN
        typer.echo(f"  {mark} {check.title}")
        if check.detail:
            typer.echo(f"       {check.detail}")
        # The fix is the point of the whole command, so it gets its own line and
        # a verb. A bare command with no "run this" reads like more diagnosis.
        if not check.ok and check.fix:
            typer.echo(f"       -> run: {check.fix}")
        if not check.ok and check.fix_note:
            typer.echo(f"       -> {check.fix_note}")

    typer.echo("")
    failures = [c for c in report.failed() if c.blocking]
    if not failures:
        warnings = [c for c in report.failed() if not c.blocking]
        typer.echo("  Everything needed is working.")
        if warnings:
            typer.echo(
                f"  ({len(warnings)} optional thing{'s' if len(warnings) > 1 else ''} "
                "marked [..] above — fix only if you meant to use it.)"
            )
        typer.echo("")
        raise typer.Exit(0)

    # Name the count and then the FIRST one specifically. "3 things need
    # fixing" leaves the reader to choose where to start; the checks are already
    # in dependency order, so we can just tell them.
    word = "thing" if len(failures) == 1 else "things"
    typer.echo(f"  {len(failures)} {word} need{'s' if len(failures) == 1 else ''} fixing.")
    first = failures[0]
    if first.fix:
        typer.echo(f"  Start here: {first.fix}")
    elif first.fix_note:
        typer.echo(f"  Start here: {first.fix_note}")
    typer.echo("")
    raise typer.Exit(1)
