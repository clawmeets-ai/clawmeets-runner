# SPDX-License-Identifier: MIT
"""
clawmeets/cli_gdrive.py — Google Drive (read-only) CLI.

Subcommands: search, get, sync, auth.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from clawmeets.integrations._config_resolve import resolve_skill_token_path
from clawmeets.integrations.gdrive import _lib

app = typer.Typer(
    name="gdrive",
    help="Google Drive read-only (search / read / sync). Paired skill: google-drive.",
    no_args_is_help=True,
)


def _drive_svc(token: str):
    return _lib.build_service(resolve_skill_token_path("google-drive", explicit=token))


def _sheets_factory(token: str):
    def _factory():
        return _lib.build_sheets_service(
            resolve_skill_token_path("google-drive", explicit=token)
        )
    return _factory


def _emit_json(payload) -> None:
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


@app.command()
def search(
    query: str = typer.Argument(..., help="Drive query syntax."),
    max_results: int = typer.Option(25, "--max"),
    token: str = typer.Option("", "--token"),
) -> None:
    """Search Drive files."""
    _emit_json(_lib.search_files(_drive_svc(token), query, max_results=max_results))


@app.command()
def get(
    file_id: str = typer.Argument(...),
    token: str = typer.Option("", "--token"),
) -> None:
    """Fetch the text body of a single file (Google-native types export as text/TSV)."""
    _emit_json(_lib.get_file_content(_drive_svc(token), file_id))


@app.command()
def sync(
    dwh: str = typer.Option(..., "--dwh"),
    config: str = typer.Option("", "--config"),
    max_runtime: int = typer.Option(1500, "--max-runtime"),
    token: str = typer.Option("", "--token"),
) -> None:
    """Sync Drive files into the warehouse per --config."""
    _emit_json(_lib.sync_to_warehouse(
        drive_svc=_drive_svc(token),
        sheets_svc_factory=_sheets_factory(token),
        dwh_dir=dwh,
        config_file=config,
        max_runtime_seconds=max_runtime,
    ))


@app.command()
def auth(
    credentials: Optional[Path] = typer.Option(None, "--credentials"),
    token: str = typer.Option("", "--token"),
) -> None:
    """Run Google OAuth for the google-drive skill (drive.readonly scope)."""
    from clawmeets.integrations.auth.google_oauth import (
        GoogleOAuthError, run_installed_flow,
    )
    token_path = resolve_skill_token_path("google-drive", explicit=token)
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
    """Revoke the google-drive skill's Google grant and delete its token.

    Headless counterpart to the web Disconnect: POSTs the token to Google's
    revoke endpoint, then removes token.json. Idempotent.
    """
    from clawmeets.integrations.auth.google_oauth import revoke_token

    token_path = resolve_skill_token_path("google-drive", explicit=token)
    if revoke_token(token_path):
        typer.echo(f"Disconnected. Revoked Google grant and removed {token_path}.")
    else:
        typer.echo(f"Nothing to disconnect — no token at {token_path}.")
