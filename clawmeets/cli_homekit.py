# SPDX-License-Identifier: MIT
"""
clawmeets/cli_homekit.py

``clawmeets homekit <subcmd>`` — Apple HomeKit / Shortcuts CLI.

Subcommands:
  list-shortcuts   Enumerate every Shortcut the user has built.
  run-shortcut     Run a Shortcut by name (optionally pipe stdin as input).
"""
from __future__ import annotations

import json
import sys
from typing import Optional

import typer

from clawmeets.integrations.homekit import _lib

app = typer.Typer(
    name="homekit",
    help="Apple HomeKit via macOS `shortcuts` CLI. Paired skill: homekit. macOS 12+ only.",
    no_args_is_help=True,
)


def _emit_json(payload) -> None:
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


@app.command("list-shortcuts")
def list_shortcuts_cmd() -> None:
    """List every Shortcut the user has built."""
    try:
        _emit_json(_lib.list_shortcuts())
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command("run-shortcut")
def run_shortcut_cmd(
    name: str = typer.Argument(..., help="Shortcut name (case-sensitive)."),
    input_text: Optional[str] = typer.Option(
        None, "--input",
        help="Text input to pipe into the Shortcut. Pass '-' to read stdin.",
    ),
) -> None:
    """Run a Shortcut by name."""
    if input_text == "-":
        input_text = sys.stdin.read()
    try:
        _emit_json(_lib.run_shortcut(name, input_text=input_text))
    except (RuntimeError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
