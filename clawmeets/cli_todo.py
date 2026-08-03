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
  update    Edit a to-do's text / due / draft prompt.
  done      Mark a to-do done.
  reopen    Move a done to-do back to open.
  delete    Remove a to-do by id.
  trigger   Fire a to-do's saved draft at its designated recipient.

``update`` / ``done`` / ``reopen`` and ``trigger`` are the **assistant's** verbs:
they go through credentials that only the owner's ``{username}-assistant``
resolves through, so any other agent gets a 401. ``publish`` / ``list`` /
``delete`` stay open to every agent as before (``delete`` still only retracts
what the caller published — unless the caller is the assistant).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import httpx
import typer

from clawmeets.cli_runner import resolve_dm_recipient, send_dm_as_owner

app = typer.Typer(
    name="todo",
    help="Publish, manage and fire to-dos on the owner's My Desk plate. Paired skill: desk-todo.",
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


def _owner_token() -> str:
    """The bearer to use for calls that need the OWNER's authority.

    Both legs of a trigger (the desk read and the DM dispatch) use one token.
    For the assistant the agent bearer and the assistant bearer are the same
    secret, so ``$CLAWMEETS_ASSISTANT_TOKEN`` or, absent it,
    ``$CLAWMEETS_AGENT_TOKEN`` authenticates both. This is why the dispatch side
    needs no extra check: any OTHER agent's token fails to resolve in
    ``resolve_user_from_credential`` and gets a 401 from the DM routes.
    """
    tok = os.environ.get("CLAWMEETS_ASSISTANT_TOKEN", "").strip()
    return tok or _env("CLAWMEETS_AGENT_TOKEN")


def _patch(todo_id: str, body: dict) -> dict | list:
    """PATCH one to-do and print the row. Shared by update / done / reopen."""
    client, headers = _client()
    with client:
        resp = client.patch(
            f"/me/desk/todos/{todo_id}", json=body, headers=headers
        )
    return _ok(resp)


def _find(client: httpx.Client, headers: dict[str, str], todo_id: str) -> dict:
    """The one to-do with this id, or exit 1.

    There is no ``GET /me/desk/todos/{id}`` — the plate is one document per
    owner, so the list IS the read path."""
    todos = _ok(client.get("/me/desk/todos", headers=headers))
    if isinstance(todos, list):
        for t in todos:
            if isinstance(t, dict) and t.get("id") == todo_id:
                return t
    typer.echo(f"Error: to-do {todo_id!r} is not on the plate.", err=True)
    raise typer.Exit(1)


def _noop(reason: str, detail: str) -> None:
    """Report "nothing to fire" and exit 0.

    The user asked to fire the item *if it was ready*; "it wasn't ready" is an
    answer, not a failure. ``reason`` is machine-readable so the skill can say
    why in plain words instead of guessing."""
    typer.echo(json.dumps({"sent": False, "reason": reason, "detail": detail}, indent=2))
    raise typer.Exit(0)


def _slug(s: str, n: int = 4) -> str:
    """Name the context attachment after the linked artifact.

    Ported from ``utils/todoDraft.ts::slug`` so a triggered to-do's ``.md``
    lands with the same filename the desk take-over would have given it."""
    cleaned = re.sub(r"['\".,()$—–:]", "", s.lower())
    parts = [p for p in cleaned.split() if p][:n]
    joined = re.sub(r"[^a-z0-9-]", "", "-".join(parts))
    return joined or "context"


def _dispatch_payload(item: dict) -> tuple[str, list[tuple[str, str]]]:
    """The message + attachments a triggered to-do sends.

    Byte-for-byte what the desk take-over sends (``todoDraft.ts``): the draft
    prompt, then a ``Referenced:`` line naming the agent-suggested file chips
    (they carry no bytes on either path, so naming is all either can do), with
    the ``context`` blob attached as ``{linked-label-slug}.md``.
    """
    content = (item.get("draft_prompt") or "").strip()
    refs = [
        f.get("name", "")
        for f in (item.get("files") or [])
        if isinstance(f, dict) and f.get("name")
    ]
    if refs:
        content = f"{content}\n\nReferenced: {', '.join(refs)}"

    files: list[tuple[str, str]] = []
    context = item.get("context")
    if context:
        linked = item.get("linked")
        label = linked.get("label") if isinstance(linked, dict) else None
        files.append(((_slug(label) if label else "context") + ".md", context))
    return content, files


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


@app.command("update")
def update(
    todo_id: str = typer.Argument(..., help="Id of the to-do to edit."),
    text: str = typer.Option("", "--text", help="New task title as it reads on the plate."),
    due: str = typer.Option("", "--due", help='New due hint, e.g. "Today" or "Fri".'),
    draft_prompt: str = typer.Option(
        "", "--draft-prompt", help="Replace the saved draft prompt."
    ),
) -> None:
    """Edit a to-do's text, due hint, or draft prompt.

    Only the flags you actually pass are sent, so an omitted flag never clears a
    stored field — an empty string means "untouched", never "clear". Requires the
    owner's assistant credential.
    """
    body: dict = {}
    if text:
        body["text"] = text
    if due:
        body["due"] = due
    if draft_prompt:
        body["draft_prompt"] = draft_prompt
    if not body:
        typer.echo(
            "Error: nothing to update — pass at least one of --text / --due / "
            "--draft-prompt.",
            err=True,
        )
        raise typer.Exit(1)
    typer.echo(json.dumps(_patch(todo_id, body), indent=2, ensure_ascii=False))


@app.command("done")
def done(
    todo_id: str = typer.Argument(..., help="Id of the to-do to strike off."),
) -> None:
    """Mark a to-do done — it moves to the plate's Completed drawer."""
    typer.echo(json.dumps(_patch(todo_id, {"status": "done"}), indent=2, ensure_ascii=False))


@app.command("reopen")
def reopen(
    todo_id: str = typer.Argument(..., help="Id of the to-do to put back."),
) -> None:
    """Move a done to-do back to open.

    A separate verb rather than ``done --undo`` so the skill's example lines read
    as instructions.
    """
    typer.echo(json.dumps(_patch(todo_id, {"status": "open"}), indent=2, ensure_ascii=False))


@app.command("delete")
def delete(
    todo_id: str = typer.Argument(..., help="Id of the to-do to delete."),
) -> None:
    """Delete a to-do by id.

    An ordinary agent may only retract a to-do it published itself; the owner's
    assistant may remove anything on its owner's plate.
    """
    client, headers = _client()
    with client:
        resp = client.delete(f"/me/desk/todos/{todo_id}", headers=headers)
    typer.echo(json.dumps(_ok(resp), indent=2, ensure_ascii=False))


_CONSUME_CHOICES = ("done", "delete", "keep")


@app.command("trigger")
def trigger(
    todo_id: str = typer.Argument(..., help="Id of the to-do to fire."),
    to: str = typer.Option(
        "", "--to", help="Override the recipient (short or full agent name)."
    ),
    consume: str = typer.Option(
        "done", "--consume",
        help="done | delete | keep — what to do with the item after a successful send.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print recipient + exact message; send nothing."
    ),
) -> None:
    """Fire a to-do's saved draft at its designated recipient.

    Mirrors the plate's one-click send with ONE deliberate difference: where the
    desk falls back to the assistant, then to any owned agent, this refuses.
    A voice-triggered send that silently redirects an addressed draft is worse
    than one that reports "nobody is on this" — so the recipient must be
    designated on the item (``draft_recipient_*``, then ``suggest_agent_*``) or
    supplied with ``--to``.

    Prints one JSON object either way, and exits 0 on a no-op::

        {"sent": true,  "to": "chengtao-api-sync", "project_id": "…", "consumed": "done"}
        {"sent": false, "reason": "no_draft_prompt" | "no_recipient"
                                  | "recipient_gone" | "already_done", "detail": "…"}

    ``--consume done`` (the default) marks the item done where the desk deletes
    it, so a to-do fired by voice leaves a trace in the Completed drawer and
    ``todo reopen`` can undo it. ``--consume delete`` is exact desk parity. A
    consume failure *after* a successful send is reported, never retried — the
    message is already out.
    """
    if consume not in _CONSUME_CHOICES:
        typer.echo(
            f"Error: --consume must be one of {', '.join(_CONSUME_CHOICES)}.", err=True
        )
        raise typer.Exit(1)

    client, headers = _client()
    with client:
        item = _find(client, headers, todo_id)

        if item.get("status") == "done":
            _noop("already_done", "That one is already struck off the plate.")
        content, files = _dispatch_payload(item)
        if not content:
            _noop(
                "no_draft_prompt",
                "No draft has been written for this to-do yet, so there is "
                "nothing to send.",
            )

        ref = (
            to.strip()
            or (item.get("draft_recipient_name") or "").strip()
            or (item.get("suggest_agent_name") or "").strip()
        )
        if not ref:
            _noop(
                "no_recipient",
                "Nobody is designated on this to-do. Pass --to <agent> to say "
                "who should get it.",
            )

        recipient = resolve_dm_recipient(
            client, _owner_token(), ref, agent_id=headers["X-Agent-ID"]
        )
        if recipient is None:
            _noop(
                "recipient_gone",
                f"{ref!r} does not match exactly one agent on the roster.",
            )

        if dry_run:
            typer.echo(json.dumps({
                "sent": False,
                "reason": "dry_run",
                "to": recipient,
                "content": content,
                "attachments": [name for name, _ in files],
                "consume": consume,
            }, indent=2, ensure_ascii=False))
            return

        project = send_dm_as_owner(
            client,
            _owner_token(),
            recipient,
            content,
            files=files or None,
            new_thread=True,
            # A retried trigger lands in the thread the first attempt made
            # instead of minting a second one.
            thread_key=f"todo:{todo_id}",
        )
        if project is None:
            _noop(
                "recipient_gone",
                f"Could not open a DM thread with {recipient!r}.",
            )

        out: dict = {
            "sent": True,
            "to": recipient,
            "project_id": project.get("id"),
            "consumed": consume,
        }
        if consume == "done":
            resp = client.patch(
                f"/me/desk/todos/{todo_id}", json={"status": "done"}, headers=headers
            )
        elif consume == "delete":
            resp = client.delete(f"/me/desk/todos/{todo_id}", headers=headers)
        else:
            resp = None
        if resp is not None and resp.status_code >= 400:
            # The message is already out — say the item is still on the plate
            # rather than retrying and risking a second send.
            out["consumed"] = "failed"
            out["consume_error"] = f"{resp.status_code}: {resp.text}"
        typer.echo(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
