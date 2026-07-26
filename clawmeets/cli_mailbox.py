# SPDX-License-Identifier: MIT
"""
clawmeets/cli_mailbox.py — Generic IMAP+SMTP mailbox CLI.

Subcommands: list-folders, search, get, attachment, send, sync.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer

from clawmeets.integrations.mailbox import _lib

app = typer.Typer(
    name="mailbox",
    help="Generic IMAP+SMTP mailbox (Gmail/iCloud/Fastmail/etc.). Paired skill: mailbox.",
    no_args_is_help=True,
)


def _emit_json(payload) -> None:
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


@app.command("list-folders")
def list_folders_cmd(config: str = typer.Option("", "--config")) -> None:
    """List IMAP folders / mailboxes."""
    try:
        _emit_json(_lib.list_folders(config))
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command()
def search(
    query: str = typer.Argument(..., help="Mini-DSL: from:/to:/subject:/since:/before:/unseen/..."),
    folder: str = typer.Option("INBOX", "--folder"),
    max_results: int = typer.Option(50, "--max"),
    config: str = typer.Option("", "--config"),
) -> None:
    """Search a folder."""
    try:
        _emit_json(_lib.search_messages(config, query, folder=folder, max_results=max_results))
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command()
def get(
    uid: str = typer.Argument(...),
    folder: str = typer.Option("INBOX", "--folder"),
    config: str = typer.Option("", "--config"),
) -> None:
    """Fetch one message envelope by UID."""
    try:
        _emit_json(_lib.get_message(config, uid, folder=folder))
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command()
def attachment(
    uid: str = typer.Argument(...),
    part_id: str = typer.Argument(...),
    folder: str = typer.Option("INBOX", "--folder"),
    out: Optional[Path] = typer.Option(None, "--out"),
    config: str = typer.Option("", "--config"),
) -> None:
    """Fetch one attachment by UID + part_id."""
    try:
        att = _lib.get_attachment(config, uid, part_id, folder=folder)
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    if out is not None:
        import base64
        out.write_bytes(base64.b64decode(att["data_b64"]))
        meta = {k: v for k, v in att.items() if k != "data_b64"}
        meta["path"] = str(out)
        _emit_json(meta)
    else:
        _emit_json(att)


@app.command()
def send(
    to: str = typer.Option(..., "--to"),
    subject: str = typer.Option(..., "--subject"),
    body: str = typer.Option("", "--body", help="Body text. '-' reads stdin."),
    cc: Optional[str] = typer.Option(None, "--cc"),
    bcc: Optional[str] = typer.Option(None, "--bcc"),
    reply_to: Optional[str] = typer.Option(None, "--reply-to"),
    html: Optional[str] = typer.Option(None, "--html"),
    config: str = typer.Option("", "--config"),
) -> None:
    """Send a plaintext (optionally HTML-alternative) email via SMTP."""
    if body == "-":
        body = sys.stdin.read()
    try:
        _emit_json(_lib.send_message(
            config, to, subject, body,
            cc=cc, bcc=bcc, reply_to=reply_to, html=html,
        ))
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command()
def sync(
    dwh: str = typer.Option(..., "--dwh"),
    config: str = typer.Option("", "--config"),
    max_runtime: int = typer.Option(1500, "--max-runtime"),
) -> None:
    """Sync IMAP folders into the warehouse per --config."""
    _emit_json(_lib.sync_to_warehouse(
        dwh, config_file=config, max_runtime_seconds=max_runtime,
    ))
