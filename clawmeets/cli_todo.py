# SPDX-License-Identifier: MIT
"""
clawmeets/cli_todo.py

``clawmeets todo <subcmd>`` — agent-facing CLI for the ``desk-todo`` skill.

Paired with ``skills/desk-todo/SKILL.md``: when an agent has surfaced
something that needs the *user's own hand* (an approval, a decision, a
sign-off), it packages the task — a suggested recipient, a ready-to-refine
prompt, the context that seeds a sharper request, what it already did, and
the facts it gathered — and shells:

    clawmeets todo publish --text "Approve the Provi restock PO ($6.8k)" \\
        --suggest api_sync --draft-prompt "Review Provi PO #4471 …" \\
        --context-file ctx.md --fact "PO total::$6,821.40 · net-30" \\
        --done "Reconciled every line item against the last 3 orders" \\
        --file "PO-4471-provi.pdf::purchase order · 2pp"

The server pushes the task onto the owner's My Desk To-do rail (keyed by
the publishing agent's owner) and broadcasts ``DESK_TODO_SYNC`` so the
desk refetches live.

Auth resolved from env (standard agent-runtime injection — same pattern as
``clawmeets brief``):

  - ``CLAWMEETS_SERVER_URL`` — server base URL
  - ``CLAWMEETS_AGENT_ID``   — UUID of the calling agent
  - ``CLAWMEETS_AGENT_TOKEN`` — agent bearer token

Subcommands:
  publish   Push a to-do onto the owner's plate.
  list      Show every to-do the owner currently has.
  delete    Remove a to-do by id.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import typer

app = typer.Typer(
    name="todo",
    help="Publish to-dos to the owner's My Desk plate. Paired skill: desk-todo.",
    no_args_is_help=True,
)


def _env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        typer.echo(
            f"Error: ${name} is not set. The todo CLI runs inside an agent "
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


def _ok(resp: httpx.Response) -> dict | list:
    if resp.status_code >= 400:
        typer.echo(f"Error {resp.status_code}: {resp.text}", err=True)
        raise typer.Exit(1)
    if not resp.content:
        return {}
    return resp.json()


def _split2(raw: str, sep: str = "::") -> tuple[str, str]:
    """Split a ``"a::b"`` option value into ``(a, b)``; b defaults to ''."""
    if sep in raw:
        a, b = raw.split(sep, 1)
        return a.strip(), b.strip()
    return raw.strip(), ""


@app.command("publish")
def publish(
    text: str = typer.Option(..., "--text", help="The task title as it reads on the plate."),
    due: str = typer.Option("", "--due", help='Optional due hint, e.g. "Today" or "Fri".'),
    suggest: str = typer.Option(
        "", "--suggest",
        help="Suggested recipient agent (short or full name) — pre-selected in the take-over.",
    ),
    draft_prompt: str = typer.Option(
        "", "--draft-prompt",
        help="A ready-to-refine request that seeds the take-over composer.",
    ),
    context_file: Path = typer.Option(
        None, "--context-file",
        exists=True, file_okay=True, dir_okay=False, readable=True,
        help="Path to a small .md/.txt whose text becomes the attachable context chip (≤ 16 KB).",
    ),
    file: list[str] = typer.Option(
        None, "--file",
        help='Suggested reference file as "name::sub" (repeatable). Informational chip only.',
    ),
    done: list[str] = typer.Option(
        None, "--done",
        help='A step you already completed (repeatable) — shown under "What\'s been done".',
    ),
    fact: list[str] = typer.Option(
        None, "--fact",
        help='A key fact as "label::value" (repeatable) — shown under "Available & relevant".',
    ),
    linked: str = typer.Option(
        "", "--linked",
        help='A source to open as "label::icon" (e.g. "Finance briefing::chart").',
    ),
) -> None:
    """Publish a to-do onto the owner's My Desk plate."""
    body: dict = {"text": text}
    if due:
        body["due"] = due
    if suggest:
        body["suggest"] = suggest
    if draft_prompt:
        body["draft_prompt"] = draft_prompt
    if context_file:
        body["context"] = context_file.read_text(encoding="utf-8")
    if file:
        body["files"] = [
            {"name": n, "sub": s, "icon": "report"}
            for n, s in (_split2(f) for f in file)
            if n
        ]
    if done:
        body["done_steps"] = [d.strip() for d in done if d.strip()]
    if fact:
        body["available"] = [
            {"k": k, "v": v} for k, v in (_split2(f) for f in fact) if k
        ]
    if linked:
        label, icon = _split2(linked)
        if label:
            body["linked"] = {"label": label, "icon": icon or "chart"}

    client, headers = _client()
    with client:
        resp = client.put("/me/desk/todos", json=body, headers=headers)
    typer.echo(json.dumps(_ok(resp), indent=2, ensure_ascii=False))


@app.command("list")
def list_todos() -> None:
    """List every to-do the calling agent's owner currently has."""
    client, headers = _client()
    with client:
        resp = client.get("/me/desk/todos", headers=headers)
    typer.echo(json.dumps(_ok(resp), indent=2, ensure_ascii=False))


@app.command("delete")
def delete(
    todo_id: str = typer.Argument(..., help="Id of the to-do to delete."),
) -> None:
    """Delete a to-do by id."""
    client, headers = _client()
    with client:
        resp = client.delete(f"/me/desk/todos/{todo_id}", headers=headers)
    typer.echo(json.dumps(_ok(resp), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
