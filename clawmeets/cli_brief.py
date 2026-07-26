# SPDX-License-Identifier: MIT
"""
clawmeets/cli_brief.py

``clawmeets brief <subcmd>`` — agent-facing CLI for the ``brief`` skill.

Paired with ``skills/brief/SKILL.md``: any agent asked to publish a
brief tab writes ``data.json`` + ``render.js`` in its sandbox cwd, then
shells:

    clawmeets brief upsert-tab <slug> --title "<title>" \\
        --data data.json --render-code render.js

The server stores the bundle under ``{data_dir}/brief-tabs/<user_id>/
<slug>.json`` keyed by the publishing agent's owner, and pushes a
``BRIEF_TAB_SYNC`` envelope to that owner's browser so My Desk refetches.

Auth resolved from env (the standard agent-runtime injection — same
pattern as ``clawmeets project create`` from a personal skill):

  - ``CLAWMEETS_SERVER_URL`` — server base URL
  - ``CLAWMEETS_AGENT_ID``   — UUID of the calling agent
  - ``CLAWMEETS_AGENT_TOKEN`` — agent bearer token

Subcommands:
  upsert-tab   Create or replace a tab (idempotent; safe to re-run).
  list-tabs    Show every tab the current user owns.
  delete-tab   Remove a tab the calling agent owns.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import typer

app = typer.Typer(
    name="brief",
    help="Publish briefing tabs to My Desk. Paired skill: brief.",
    no_args_is_help=True,
)


def _env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        typer.echo(
            f"Error: ${name} is not set. The brief CLI runs inside an agent "
            f"runtime that injects CLAWMEETS_SERVER_URL/AGENT_ID/AGENT_TOKEN.",
            err=True,
        )
        raise typer.Exit(1)
    return val


def _client() -> tuple[httpx.Client, dict[str, str]]:
    server = _env("CLAWMEETS_SERVER_URL").rstrip("/")
    headers = {
        "Authorization": f"Bearer {_env('CLAWMEETS_AGENT_TOKEN')}",
        "X-Agent-ID": _env("CLAWMEETS_AGENT_ID"),
    }
    return httpx.Client(base_url=server, timeout=30), headers


def _read_data(path: Path) -> dict | list:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        typer.echo(f"Error reading {path}: {e}", err=True)
        raise typer.Exit(1) from e
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        typer.echo(f"Error: {path} is not valid JSON: {e}", err=True)
        raise typer.Exit(1) from e


def _read_render_code(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as e:
        typer.echo(f"Error reading {path}: {e}", err=True)
        raise typer.Exit(1) from e


def _ok(resp: httpx.Response) -> dict | list:
    if resp.status_code >= 400:
        typer.echo(f"Error {resp.status_code}: {resp.text}", err=True)
        raise typer.Exit(1)
    if not resp.content:
        return {}
    return resp.json()


@app.command("upsert-tab")
def upsert_tab(
    slug: str = typer.Argument(..., help="Tab slug (a–z, 0–9, _, -; ≤ 80 chars)."),
    title: str = typer.Option(
        "", "--title",
        help="Tab label. Defaults to slug.",
    ),
    data: Path = typer.Option(
        ..., "--data",
        exists=True, file_okay=True, dir_okay=False, readable=True,
        help="Path to data.json (any JSON your render code understands; ≤ 64 KB).",
    ),
    render_code: Path = typer.Option(
        ..., "--render-code",
        exists=True, file_okay=True, dir_okay=False, readable=True,
        help="Path to render.js — BODY of function(mount, data, lib); ≤ 64 KB.",
    ),
) -> None:
    """Upsert a brief tab. Re-running with the same slug overwrites it."""
    body = {
        "title": title,
        "data": _read_data(data),
        "render_code_js": _read_render_code(render_code),
    }
    client, headers = _client()
    with client:
        resp = client.put(f"/me/brief/tabs/{slug}", json=body, headers=headers)
    out = _ok(resp)
    typer.echo(json.dumps(out, indent=2, ensure_ascii=False))


@app.command("list-tabs")
def list_tabs() -> None:
    """List every brief tab the calling agent's owner has."""
    client, headers = _client()
    with client:
        resp = client.get("/me/brief/tabs", headers=headers)
    typer.echo(json.dumps(_ok(resp), indent=2, ensure_ascii=False))


@app.command("delete-tab")
def delete_tab(
    slug: str = typer.Argument(..., help="Slug of the tab to delete."),
) -> None:
    """Delete a brief tab. Only the publishing agent (or owner) may
    delete; foreign agents get 403."""
    client, headers = _client()
    with client:
        resp = client.delete(f"/me/brief/tabs/{slug}", headers=headers)
    typer.echo(json.dumps(_ok(resp), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
