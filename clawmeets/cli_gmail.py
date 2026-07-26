# SPDX-License-Identifier: MIT
"""
clawmeets/cli_gmail.py

``clawmeets gmail <subcmd>`` — Typer surface for the gmail skill.

Subcommands:
  search       Search by query.
  get          Fetch a message (full | metadata).
  labels       List labels.
  attachment   Fetch one attachment (optionally to disk).
  send         Send a plaintext email.
  sync         Run sync_to_warehouse per --config.
  auth         Run Google OAuth (local installed-app flow).

Every subcommand resolves ``--config`` and ``--token`` from
``$CLAWMEETS_AGENT_DIR/skill-hub/{configs,state}/gmail/`` via
``clawmeets.integrations._config_resolve`` when not passed explicitly,
so the LLM-side ``Bash: clawmeets gmail ...`` invocations stay terse.
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Optional

import typer

from clawmeets.integrations._config_resolve import resolve_skill_token_path
from clawmeets.integrations.gmail import _lib

app = typer.Typer(
    name="gmail",
    help="Gmail (search / get / labels / attachment / send / sync). Paired skill: gmail.",
    no_args_is_help=True,
)


def _svc(token: str):
    token_path = resolve_skill_token_path("gmail", explicit=token)
    return _lib.build_service(token_path)


def _emit_json(payload) -> None:
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


@app.command()
def search(
    query: str = typer.Argument(..., help="Gmail search-syntax query."),
    max_results: int = typer.Option(20, "--max", help="Max results."),
    token: str = typer.Option("", "--token", help="Override token path."),
) -> None:
    """Search Gmail messages by query."""
    _emit_json(_lib.search_messages(_svc(token), query, max_results=max_results))


@app.command()
def get(
    message_id: str = typer.Argument(...),
    fmt: str = typer.Option("full", "--format", help="full | metadata."),
    token: str = typer.Option("", "--token"),
) -> None:
    """Fetch a single Gmail message."""
    _emit_json(_lib.get_message(_svc(token), message_id, format=fmt))


@app.command("labels")
def labels_cmd(
    token: str = typer.Option("", "--token"),
) -> None:
    """List all Gmail labels."""
    _emit_json(_lib.list_labels(_svc(token)))


@app.command()
def attachment(
    message_id: str = typer.Argument(...),
    attachment_id: str = typer.Argument(...),
    out: Optional[Path] = typer.Option(
        None, "--out", help="Write attachment bytes to path; metadata to stdout.",
    ),
    token: str = typer.Option("", "--token"),
) -> None:
    """Fetch one attachment from a message."""
    a = _lib.get_attachment(_svc(token), message_id, attachment_id)
    if out is not None:
        out.write_bytes(base64.b64decode(a["data_b64"]))
        meta = {k: v for k, v in a.items() if k != "data_b64"}
        meta["path"] = str(out)
        _emit_json(meta)
    else:
        _emit_json(a)


@app.command()
def send(
    to: str = typer.Option(..., "--to"),
    subject: str = typer.Option(..., "--subject"),
    body: str = typer.Option(
        "", "--body",
        help="Body text. Pass '-' to read from stdin.",
    ),
    cc: Optional[str] = typer.Option(None, "--cc"),
    bcc: Optional[str] = typer.Option(None, "--bcc"),
    token: str = typer.Option("", "--token"),
) -> None:
    """Send a plaintext email."""
    if body == "-":
        body = sys.stdin.read()
    _emit_json(_lib.send_message(_svc(token), to, subject, body, cc=cc, bcc=bcc))


@app.command()
def sync(
    dwh: str = typer.Option(..., "--dwh", help="Data warehouse root."),
    config: str = typer.Option("", "--config"),
    max_runtime: int = typer.Option(1500, "--max-runtime"),
    token: str = typer.Option("", "--token"),
) -> None:
    """Sync gmail labels into the warehouse (named-slice, watermarked)."""
    _emit_json(_lib.sync_to_warehouse(
        _svc(token), dwh,
        config_file=config, max_runtime_seconds=max_runtime,
    ))


@app.command()
def auth(
    credentials: Optional[Path] = typer.Option(
        None, "--credentials",
        help="Path to installed-app OAuth client_secrets.json.",
    ),
    token: str = typer.Option(
        "", "--token",
        help="Override token destination path.",
    ),
) -> None:
    """Run Google OAuth for the gmail skill.

    Opens the default browser to Google's consent screen and writes the
    refresh token to ``$CLAWMEETS_AGENT_DIR/skill-hub/state/gmail/token.json``
    (or ``--token``). Web-UI relay mode is driven by the runner via the
    SKILL_AUTH envelope — invoke this command for the headless / local case.
    """
    from clawmeets.integrations.auth.google_oauth import (
        GoogleOAuthError,
        run_installed_flow,
    )

    token_path = resolve_skill_token_path("gmail", explicit=token)
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
    """Revoke the gmail skill's Google grant and delete its local token.

    Headless counterpart to the web Disconnect: POSTs the token to Google's
    revoke endpoint, then removes token.json. Idempotent — a missing or
    already-invalid token is reported as already-disconnected.
    """
    from clawmeets.integrations.auth.google_oauth import revoke_token

    token_path = resolve_skill_token_path("gmail", explicit=token)
    if revoke_token(token_path):
        typer.echo(f"Disconnected. Revoked Google grant and removed {token_path}.")
    else:
        typer.echo(f"Nothing to disconnect — no token at {token_path}.")
