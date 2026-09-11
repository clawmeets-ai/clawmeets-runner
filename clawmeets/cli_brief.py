# SPDX-License-Identifier: MIT
"""
clawmeets/cli_brief.py

``clawmeets brief <subcmd>`` — agent-facing CLI for the ``brief`` skill.

Paired with ``skills/brief/SKILL.md``: any agent asked to publish a
brief tab writes ONE complete HTML document in its sandbox cwd, then
shells:

    clawmeets brief upsert-tab <slug> --title "<title>" \\
        --html briefing.html

The server stores the document byte-verbatim under
``{data_dir}/brief-tabs/<user_id>/<slug>.html``, its metadata alongside
in ``<slug>.json``, both keyed by the publishing agent's owner, and
pushes a ``BRIEF_TAB_SYNC`` cursor to that owner's browser so My Desk
refetches.

Auth resolved from env (the standard agent-runtime injection — same
pattern as ``clawmeets project create`` from a personal skill):

  - ``CLAWMEETS_SERVER_URL`` — server base URL
  - ``CLAWMEETS_AGENT_ID``   — UUID of the calling agent
  - ``CLAWMEETS_AGENT_TOKEN`` — agent bearer token

Subcommands:
  upsert-tab   Create or replace a tab (idempotent; safe to re-run).
  list-tabs    Show metadata for every tab the current user owns.
  delete-tab   Remove a tab the calling agent owns.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import typer

from clawmeets.models.brief_tab import MAX_BRIEF_HTML_BYTES

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


def _read_html(path: Path) -> str:
    """Read the briefing document.

    ``read_bytes().decode("utf-8")``, never ``read_text``:
    ``Path.read_text`` opens in universal-newline mode and rewrites
    ``\r\n`` and lone ``\r`` to ``\n``. A document the agent wrote with
    CRLF must arrive at the server as CRLF or the round trip is not
    byte-verbatim.
    """
    try:
        raw = path.read_bytes()
    except OSError as e:
        typer.echo(f"Error reading {path}: {e}", err=True)
        raise typer.Exit(1) from e
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as e:
        typer.echo(f"Error: {path} is not valid UTF-8: {e}", err=True)
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
    html: Path = typer.Option(
        ..., "--html",
        exists=True, file_okay=True, dir_okay=False, readable=True,
        help=(
            "Path to the complete HTML document "
            f"(≤ {MAX_BRIEF_HTML_BYTES} bytes / 256 KB)."
        ),
    ),
) -> None:
    """Upsert a brief tab. Re-running with the same slug overwrites it.

    Checks the cap locally, before the upload, with the same constant and
    the same two numbers the server's 413 would name. A 256 KB round trip
    to be told a number we already knew is a slow way to learn it — and
    an agent that gets the error where it wrote the file can fix the
    file, which is what "inside the limits on the first attempt"
    actually needs.
    """
    document = _read_html(html)
    size = len(document.encode("utf-8"))
    if size > MAX_BRIEF_HTML_BYTES:
        typer.echo(
            f"Error: {html} exceeds {MAX_BRIEF_HTML_BYTES} bytes (256 KB) — "
            f"got {size} bytes. Trim the document and re-run; nothing was "
            f"uploaded.",
            err=True,
        )
        raise typer.Exit(1)

    body = {"title": title, "html": document}
    client, headers = _client()
    with client:
        resp = client.put(f"/me/brief/tabs/{slug}", json=body, headers=headers)
    out = _ok(resp)

    # **A 200 IS NOT PROOF THE DOCUMENT WAS STORED.** The route returns the
    # STORED record, so the response itself says whether the server kept what
    # we sent: a server that understands this shape echoes `html`.
    #
    # The incident this exists for: a server older than this client had a PUT
    # whose body params were `title` / `data` / `render_code_js`. FastAPI
    # ignored the `html` key it did not know, defaulted the two it did, wrote
    # the record, and answered 200. The CLI printed that as success, the
    # publishing agent reported the briefing refreshed, and the owner's
    # briefing had in fact been replaced by an empty one. Nothing anywhere
    # raised a hand.
    #
    # A RESPONSE-SHAPE CHECK, NOT A VERSION NEGOTIATION. There is no version
    # handshake to hang this on and adding one to catch a skew would be a
    # protocol for a single `if`; the echo is already in hand. It only ever
    # fires on a server that cannot store what it just accepted.
    if isinstance(out, dict) and "html" not in out:
        typer.echo(
            f"Error: the server accepted the upload but stored no document — "
            f"it returned {sorted(out)} with no 'html'. That is almost "
            f"certainly a server older than this client, whose upsert ignores "
            f"the field it cannot store. Tab {slug!r} on that server may now "
            f"be EMPTY; update the server and re-run this command to restore "
            f"it.",
            err=True,
        )
        raise typer.Exit(1)

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
