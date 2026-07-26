# SPDX-License-Identifier: MIT
"""
clawmeets/cli_gdrive_write.py — Google Sheets read+write CLI.

Subcommands: read-rows, append-rows, update-cell, auth.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer

from clawmeets.integrations._config_resolve import resolve_skill_token_path
from clawmeets.integrations.gdrive_write import _lib

app = typer.Typer(
    name="gdrive-write",
    help="Google Sheets read+write (per-target). Paired skill: google-drive-write.",
    no_args_is_help=True,
)


def _svc(token: str):
    return _lib.build_service(resolve_skill_token_path("google-drive-write", explicit=token))


def _emit_json(payload) -> None:
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


@app.command("read-rows")
def read_rows_cmd(
    target: str = typer.Option(..., "--target", help="Logical target name from sheet_targets."),
    config: str = typer.Option("", "--config"),
    token: str = typer.Option("", "--token"),
) -> None:
    """Read all rows from the named target."""
    try:
        _emit_json(_lib.read_sheet_rows(_svc(token), config, target))
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command("append-rows")
def append_rows_cmd(
    target: str = typer.Option(..., "--target"),
    rows: str = typer.Option(
        "", "--rows",
        help="JSON list of row dicts. Pass '-' to read from stdin.",
    ),
    rows_file: Optional[Path] = typer.Option(
        None, "--rows-file",
        help="Path to a JSON file containing the rows list.",
    ),
    config: str = typer.Option("", "--config"),
    token: str = typer.Option("", "--token"),
) -> None:
    """Append rows to the named target."""
    if rows_file is not None:
        rows_json = rows_file.read_text()
    elif rows == "-":
        rows_json = sys.stdin.read()
    else:
        rows_json = rows
    try:
        rows_list = json.loads(rows_json)
    except json.JSONDecodeError as exc:
        typer.echo(f"Error: --rows is not valid JSON: {exc}", err=True)
        raise typer.Exit(2) from exc
    try:
        _emit_json(_lib.append_sheet_rows(_svc(token), config, target, rows_list))
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command("update-cell")
def update_cell_cmd(
    target: str = typer.Option(..., "--target"),
    row_id: str = typer.Option(..., "--row-id"),
    column: str = typer.Option(..., "--column"),
    value: str = typer.Option(..., "--value"),
    config: str = typer.Option("", "--config"),
    token: str = typer.Option("", "--token"),
) -> None:
    """Update one cell by id-column lookup."""
    try:
        _emit_json(_lib.update_sheet_cell(
            _svc(token), config, target,
            row_id=row_id, column=column, value=value,
        ))
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command()
def auth(
    credentials: Optional[Path] = typer.Option(None, "--credentials"),
    token: str = typer.Option("", "--token"),
) -> None:
    """Run Google OAuth for the google-drive-write skill (sheets write scope)."""
    from clawmeets.integrations.auth.google_oauth import (
        GoogleOAuthError, run_installed_flow,
    )
    token_path = resolve_skill_token_path("google-drive-write", explicit=token)
    try:
        run_installed_flow(_lib.SCOPES, token_path, credentials)
    except GoogleOAuthError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Authenticated. Token at {token_path} (0600).")


@app.command()
def revoke(
    token: str = typer.Option("", "--token", help="Override token path."),
) -> None:
    """Revoke the google-drive-write skill's Google grant and delete its token.

    Headless counterpart to the web Disconnect: POSTs the token to Google's
    revoke endpoint, then removes token.json. Idempotent.
    """
    from clawmeets.integrations.auth.google_oauth import revoke_token

    token_path = resolve_skill_token_path("google-drive-write", explicit=token)
    if revoke_token(token_path):
        typer.echo(f"Disconnected. Revoked Google grant and removed {token_path}.")
    else:
        typer.echo(f"Nothing to disconnect — no token at {token_path}.")
