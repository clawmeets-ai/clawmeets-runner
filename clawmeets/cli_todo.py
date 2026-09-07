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
  list      Show every to-do the owner currently has (optionally filtered by label).
  update    Edit a to-do's text / due / draft prompt / labels.
  labels    Curate the owner's label vocabulary (nested sub-app).
  done      Mark a to-do done.
  reopen    Move a done to-do back to open.
  delete    Remove a to-do by id.
  trigger   Fire a to-do's saved draft at its designated recipient.

All of ``labels`` except ``labels list`` is likewise the assistant's.

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
from clawmeets.models.desk_todo import LabelError, normalize_label

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


# The label error format: ``{"detail": "labels.<code>: <human sentence>"}``.
_LABEL_CODE = re.compile(r"^labels\.[a-z0-9_]+: ")


def _detail(resp: httpx.Response) -> str:
    """The human sentence out of an error response.

    Parses ``{"detail": "..."}``, then applies the contract's parse rule ONCE
    for the whole CLI: a ``detail`` starting with ``labels.<code>: `` has the
    prefix stripped and the remainder printed. A ``detail`` that does not match
    is printed unchanged — and that fallback is load-bearing rather than
    defensive, because every OTHER error in this codebase is uncoded and a CLI
    that assumed a code would mangle the first 401.

    Falls back to the raw body when it is not JSON.

    Note the blast radius, which is deliberate: this sits in ``_ok``, which
    every subcommand shares, so error output changes for EVERY todo verb — from
    ``Error 404: {"detail":"To-do 't-x' not found"}`` to ``Error 404: To-do
    't-x' not found``. That is an improvement, and it is why the assistant never
    reads "labels.cap_exceeded:" aloud.
    """
    try:
        body = resp.json()
    except Exception:
        return resp.text
    detail = body.get("detail") if isinstance(body, dict) else None
    if not isinstance(detail, str):
        return resp.text
    match = _LABEL_CODE.match(detail)
    return detail[match.end():] if match else detail


def _ok(resp: httpx.Response) -> dict | list:
    if resp.status_code >= 400:
        typer.echo(f"Error {resp.status_code}: {_detail(resp)}", err=True)
        raise typer.Exit(1)
    if not resp.content:
        return {}
    return resp.json()


def _norm(raw: str) -> str:
    """One label through the SERVER's normalizer, so ``--label @office``,
    ``--label Office`` and ``--label office`` all match the stored slug.

    Imported rather than re-implemented: a second grammar in the CLI is a second
    thing to get wrong, and the failure mode is silent (a filter that matches
    nothing) rather than loud."""
    try:
        return normalize_label(raw)
    except LabelError as e:
        typer.echo(f"Error: {e.sentence}", err=True)
        raise typer.Exit(1)


def _label_rows(client: httpx.Client, headers: dict[str, str]) -> list[dict] | None:
    """The owner's registry, or None if it could not be read.

    None is a real answer, not an error: the plate is readable with the registry
    gone, so a filter degrades rather than failing. The reason goes to STDERR so
    stdout stays parseable JSON."""
    try:
        resp = client.get("/me/desk/labels", headers=headers)
    except Exception as e:
        typer.echo(f"Note: could not read the label list ({e}).", err=True)
        return None
    if resp.status_code >= 400:
        typer.echo(
            f"Note: could not read the label list ({resp.status_code}); "
            f"filtering across every label together and omitting label detail.",
            err=True,
        )
        return None
    body = resp.json()
    rows = body.get("labels") if isinstance(body, dict) else body
    return rows if isinstance(rows, list) else []


def _matches(item: dict, wanted: list[str], kinds: dict[str, str], match_all: bool) -> bool:
    """OR within a kind, AND across kinds — the rail's default.

    ``--label office --label home --label next`` reads "next actions I could do
    at the office or at home". A slug with no registry row counts as a CONTEXT,
    which is the benign absence: filtering must decide something, so it takes
    the harmless default.

    ``--match-all`` collapses to strict AND over every named label.

    Filtering is client-side over the full GET. No query params are added to the
    server, ever — a client that has only seen a filtered subset cannot compute
    a correct global order to send to ``/reorder``.
    """
    carried = set(item.get("labels") or [])
    if match_all:
        return all(slug in carried for slug in wanted)
    groups: dict[str, list[str]] = {}
    for slug in wanted:
        groups.setdefault(kinds.get(slug, "context"), []).append(slug)
    return all(
        any(slug in carried for slug in group) for group in groups.values()
    )


def _label_detail(slugs: list[str], rows: list[dict]) -> list[dict]:
    """Join a to-do's slugs against the registry ALREADY loaded for ``_matches``.

    One dict per slug, same length and same order as ``labels``, so the two
    arrays index together and neither is a re-ordering of the other::

        {"slug": "wait-for", "name": "Wait For", "kind": "state", "registered": true}
        {"slug": "offce",    "name": "offce",    "kind": null,    "registered": false}

    An unregistered slug renders as its own name with a NULL kind, and the kind
    is never guessed. Note the deliberate divergence from ``_matches``, which
    counts an unregistered slug as a context: FILTERING must decide something,
    so it takes the harmless default; RENDERING must not, because a guessed kind
    is read out loud as a fact about the owner's own vocabulary.
    ``registered: false`` is what lets the assistant say "that one isn't in your
    list" — the CLI's half of the nag group the browser draws dashed.

    ``labels`` itself is NEVER rewritten. It stays exactly the slugs the server
    sent, so anything already reading it keeps working and there is one source
    of truth for what is stored.
    """
    by_slug = {r.get("slug"): r for r in rows if isinstance(r, dict)}
    out = []
    for slug in slugs:
        row = by_slug.get(slug)
        if row is None:
            out.append({"slug": slug, "name": slug, "kind": None, "registered": False})
        else:
            out.append({
                "slug": slug,
                "name": row.get("name") or slug,
                "kind": row.get("kind") or "context",
                "registered": True,
            })
    return out


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
    label: list[str] = typer.Option(
        None, "--label",
        help="A GTD context or state to file this under (repeatable). "
             "Write `office` or `@office` — both land as the same label.",
    ),
) -> None:
    """Publish a to-do onto the owner's My Desk plate.

    ``--label`` uses the owner's OWN vocabulary — run ``clawmeets todo labels
    list`` first and reuse what is there. A label you invent still shows on the
    item but stays unregistered and is treated as a context whatever you meant
    by it; publishing never creates a state and never adds to the owner's list.
    """
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
        link_label, icon = _split2(linked)
        if link_label:
            body["linked"] = {"label": link_label, "icon": icon or "chart"}
    if label:
        body["labels"] = [_norm(l) for l in label if l.strip()]

    client, headers = _client()
    with client:
        resp = client.put("/me/desk/todos", json=body, headers=headers)
    typer.echo(json.dumps(_ok(resp), indent=2, ensure_ascii=False))


@app.command("list")
def list_todos(
    label: list[str] = typer.Option(
        None, "--label",
        help="Only to-dos carrying this label (repeatable). Several contexts "
             "are OR'd together; a state narrows them.",
    ),
    match_all: bool = typer.Option(
        False, "--match-all",
        help="Require EVERY named label instead of the default OR-within-a-kind.",
    ),
    any_: bool = typer.Option(False, "--any", hidden=True),
) -> None:
    """List every to-do the calling agent's owner currently has.

    Unfiltered, this is a verbatim passthrough of the server's plate — including
    each item's ``labels`` (bare slugs, stored order). It costs nothing and works
    with the label registry gone.

    Filtered, it also fetches the registry, because the default filter cannot be
    computed without knowing each requested label's kind:

        --label office --label home --label next
            -> (office OR home) AND next     "what's next, at the office or at home"
        --label office --label home --match-all
            -> office AND home

    On the filtered path each row gains a sibling ``labels_detail`` array — same
    length and same order as ``labels`` — carrying each slug's display name,
    kind, and whether it is registered at all. That is what lets the assistant
    say "two of these are also waiting on someone" instead of reading raw slugs
    aloud. ``labels`` itself is never rewritten.

    If the registry cannot be read, ``labels_detail`` is OMITTED rather than
    emitted half-joined or with invented kinds, the filter degrades to a flat OR
    across every named label, and the reason goes to stderr so stdout stays
    parseable JSON.

    """
    # `--any` is a hidden no-op alias for the default, kept so anything that
    # already says it keeps working. It is NOT an inversion of `--match-all`:
    # under an OR default, a flag named `--any` that turned ON strict AND reads
    # backwards and the voice path gets it wrong every time. It stays out of
    # `--help` and out of the SKILL.md, which is why this note is a comment and
    # not part of the docstring typer renders.
    client, headers = _client()
    with client:
        todos = _ok(client.get("/me/desk/todos", headers=headers))
        if not label:
            typer.echo(json.dumps(todos, indent=2, ensure_ascii=False))
            return

        wanted = [_norm(l) for l in label if l.strip()]
        rows = _label_rows(client, headers)
        kinds = (
            {r.get("slug"): (r.get("kind") or "context") for r in rows}
            if rows is not None else {}
        )
        items = [
            t for t in todos
            if isinstance(t, dict) and _matches(t, wanted, kinds, match_all)
        ]
        if rows is not None:
            items = [
                {**t, "labels_detail": _label_detail(t.get("labels") or [], rows)}
                for t in items
            ]
    typer.echo(json.dumps(items, indent=2, ensure_ascii=False))


@app.command("update")
def update(
    todo_id: str = typer.Argument(..., help="Id of the to-do to edit."),
    text: str = typer.Option("", "--text", help="New task title as it reads on the plate."),
    due: str = typer.Option("", "--due", help='New due hint, e.g. "Today" or "Fri".'),
    draft_prompt: str = typer.Option(
        "", "--draft-prompt", help="Replace the saved draft prompt."
    ),
    label: list[str] = typer.Option(
        None, "--label",
        help="REPLACE the whole label set with these (repeatable).",
    ),
    add_label: list[str] = typer.Option(
        None, "--add-label", help="Add one label, leaving the rest (repeatable).",
    ),
    remove_label: list[str] = typer.Option(
        None, "--remove-label", help="Remove one label (repeatable).",
    ),
    clear_labels: bool = typer.Option(
        False, "--clear-labels", help="Remove every label from this to-do.",
    ),
) -> None:
    """Edit a to-do's text, due hint, draft prompt, or labels.

    Only the flags you actually pass are sent, so an omitted flag never clears a
    stored field — an empty string means "untouched", never "clear". Requires the
    owner's assistant credential.

    PREFER ``--add-label`` / ``--remove-label`` over ``--label``: they cannot
    clobber a label the owner just set in the browser, and a retried command
    cannot double-apply. Both are idempotent — adding a label the item already
    carries and removing one it does not are successes, not errors.
    """
    body: dict = {}
    if text:
        body["text"] = text
    if due:
        body["due"] = due
    if draft_prompt:
        body["draft_prompt"] = draft_prompt

    # The wire defines a precedence for replace-then-add-then-remove so a
    # confused client still gets a defined result. The CLI refuses the
    # combination instead, because at a human boundary "replace these AND also
    # add that" has no reading anybody would bet on. Different jobs, both right.
    if label and (add_label or remove_label or clear_labels):
        typer.echo(
            "Error: --label replaces the whole set; use --add-label / "
            "--remove-label to adjust it.",
            err=True,
        )
        raise typer.Exit(1)
    if clear_labels and (add_label or remove_label):
        typer.echo(
            "Error: --clear-labels removes every label; it cannot be combined "
            "with --add-label / --remove-label.",
            err=True,
        )
        raise typer.Exit(1)

    if clear_labels:
        body["labels"] = []
    elif label:
        body["labels"] = [_norm(l) for l in label if l.strip()]
    if add_label:
        body["add_labels"] = [_norm(l) for l in add_label if l.strip()]
    if remove_label:
        body["remove_labels"] = [_norm(l) for l in remove_label if l.strip()]

    if not body:
        typer.echo(
            "Error: nothing to update — pass at least one of --text / --due / "
            "--draft-prompt / --label / --add-label / --remove-label / "
            "--clear-labels.",
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


# ===========================================================================
# `clawmeets todo labels <verb>` — curating the owner's vocabulary
#
# Nested under the existing `todo` app rather than a new top-level command, so
# the desk-todo skill stays the single home for the plate. A label is only ever
# a property of a to-do; there is no owner question `labels list` answers on its
# own, so splitting it off would create a skill whose every useful invocation
# immediately needs the other skill's verbs.
#
# There is NO `labels set-kind` verb and there should not be one: a label's kind
# is immutable after creation, and the repair is delete-then-add, which is
# honest about detaching the slug from every item on the way through.
# ===========================================================================

labels_app = typer.Typer(
    name="labels",
    help="List and curate the owner's label vocabulary (contexts and states).",
    no_args_is_help=True,
)
app.add_typer(labels_app)


def _registry_rows() -> list[dict]:
    """The owner's registry rows. Exits 1 if they cannot be read — unlike the
    filtered ``list`` path, a ``labels`` verb is ABOUT the registry, so there is
    nothing to degrade to."""
    client, headers = _client()
    with client:
        body = _ok(client.get("/me/desk/labels", headers=headers))
    rows = body.get("labels") if isinstance(body, dict) else body
    return rows if isinstance(rows, list) else []


def _resolve_kind(raw: str) -> str:
    """The forgiving --kind parser.

    Voice will never say the word "kind", so the assistant transcribing "make
    that a state" must not have to guess a canonical spelling. Case-insensitive,
    after stripping a leading sigil and surrounding whitespace::

        state | states | ! | !state             -> "state"
        context | contexts | @ | @context       -> "context"
        any registered label's slug OR display  -> that label's kind
          name (next, !next, "Wait For")           (LOOKUP, not string match)
        omitted                                 -> "context"
        anything else                           -> exit 1, never a silent default

    Resolution by lookup is what "make it a state like Next" means, and it costs
    nothing — the caller already holds the registry.

    This is a CLI-side parse ONLY. The wire still carries the strict enum and
    still answers ``labels.invalid_kind``, because a forgiving boundary over a
    strict wire is the correct split; the inverse is how a third kind value gets
    stored by accident.
    """
    raw = (raw or "").strip()
    if not raw:
        return "context"
    if raw == "!":
        return "state"
    if raw == "@":
        return "context"
    token = raw.lstrip("@!").strip().lower()
    if token in ("state", "states"):
        return "state"
    if token in ("context", "contexts"):
        return "context"
    # Only a LOOKUP needs the registry, so the round trip is paid only by the
    # callers that actually say "the same kind as Next".
    for row in _registry_rows():
        if not isinstance(row, dict):
            continue
        if token in (
            str(row.get("slug", "")).lower(),
            str(row.get("name", "")).strip().lower(),
        ):
            return row.get("kind") or "context"
    typer.echo(
        f"Error: --kind must be 'context' or 'state' (or the name of a label "
        f"you already have); {raw!r} matched neither.",
        err=True,
    )
    raise typer.Exit(1)


@labels_app.command("list")
def labels_list() -> None:
    """Show the owner's label vocabulary, in the order it is grouped in.

    Readable by every agent — run it before you `publish --label` so you reuse
    the owner's own vocabulary instead of inventing a near-duplicate. Each row
    carries its `kind`, which is the only source of truth for whether a label is
    a context or a state; do not infer that from how the slug is spelled.
    """
    client, headers = _client()
    with client:
        resp = client.get("/me/desk/labels", headers=headers)
    typer.echo(json.dumps(_ok(resp), indent=2, ensure_ascii=False))


@labels_app.command("add")
def labels_add(
    name: str = typer.Argument(..., help='The label, e.g. "@errands" or "Waiting on bank".'),
    kind: str = typer.Option(
        "", "--kind",
        help='"context" (default) or "state". Also accepts a sigil (@ / !) or '
             "the name of a label you already have, meaning \"the same kind as "
             'that one\".',
    ),
    color: str = typer.Option(
        "", "--color",
        help="A palette token: blue green indigo teal amber pink slate plum. "
             "Omit it and the server picks the least-used one.",
    ),
) -> None:
    """Create a label. The `@` / `!` sigil is stripped — it is how the rail
    renders a kind, never part of the name."""
    body: dict = {"name": name, "kind": _resolve_kind(kind)}
    if color:
        body["color"] = color
    client, headers = _client()
    with client:
        resp = client.post("/me/desk/labels", json=body, headers=headers)
    typer.echo(json.dumps(_ok(resp), indent=2, ensure_ascii=False))


@labels_app.command("rename")
def labels_rename(
    slug: str = typer.Argument(..., help="The label to rename."),
    name: str = typer.Option(..., "--name", help="The new display name."),
) -> None:
    """Change a label's DISPLAY NAME. No to-do is touched and the slug does not
    move, so this is safe and instant.

    There is deliberately no way to change a label's kind here. A context that
    three tasks carry cannot be promoted to a state without retroactively
    breaking at-most-one-state on all three, silently, in a write the owner
    reads as a rename. The repair is `delete` then `add` — say first how many
    items that will detach it from, AND that you cannot put the new label back
    on them automatically.
    """
    client, headers = _client()
    with client:
        resp = client.patch(
            f"/me/desk/labels/{_norm(slug)}", json={"name": name}, headers=headers
        )
    typer.echo(json.dumps(_ok(resp), indent=2, ensure_ascii=False))


@labels_app.command("recolor")
def labels_recolor(
    slug: str = typer.Argument(..., help="The label to recolour."),
    color: str = typer.Option(
        ..., "--color",
        help="blue | green | indigo | teal | amber | pink | slate | plum.",
    ),
) -> None:
    """Change a label's colour. No to-do is touched."""
    client, headers = _client()
    with client:
        resp = client.patch(
            f"/me/desk/labels/{_norm(slug)}", json={"color": color}, headers=headers
        )
    typer.echo(json.dumps(_ok(resp), indent=2, ensure_ascii=False))


@labels_app.command("reorder")
def labels_reorder(
    slugs: list[str] = typer.Argument(..., help="Labels in the order you want them."),
) -> None:
    """Set the order labels are grouped and listed in.

    Position IS the order. Labels you leave out keep their relative order after
    the ones you named, and a slug the owner does not have is ignored rather
    than refused — so a partial list (say, just the contexts) never drops the
    rest.
    """
    client, headers = _client()
    with client:
        resp = client.post(
            "/me/desk/labels/reorder",
            json={"ordered_slugs": [_norm(s) for s in slugs]},
            headers=headers,
        )
    typer.echo(json.dumps(_ok(resp), indent=2, ensure_ascii=False))


@labels_app.command("merge")
def labels_merge(
    slug: str = typer.Argument(..., help="The label to fold away (usually a typo)."),
    into: str = typer.Option(..., "--into", help="The label it should become."),
) -> None:
    """Fold one label into another, rewriting every to-do that carries it.

    The typo repair: `merge offce --into office`. Both labels must be the SAME
    KIND — a context into a context, a state into a state. Anything else comes
    back `kind_mismatch`, and that is not a bug to work around: merging across
    kinds is the same kind-change that `rename` refuses, asking to be done the
    long way.

    Prints `moved`, the number of to-dos rewritten.
    """
    client, headers = _client()
    with client:
        resp = client.post(
            f"/me/desk/labels/{_norm(slug)}/merge",
            json={"into": _norm(into)},
            headers=headers,
        )
    typer.echo(json.dumps(_ok(resp), indent=2, ensure_ascii=False))


@labels_app.command("delete")
def labels_delete(
    slug: str = typer.Argument(..., help="The label to remove."),
) -> None:
    """Remove a label from the owner's list AND from every to-do carrying it.

    This NEVER deletes a to-do. An item whose only label was this one just
    becomes unlabelled. The response carries `detached_from` — say the number
    out loud: "Dropped @errands — it came off 6 items, all still on your plate."

    States delete exactly like contexts, and deleting every state is allowed.
    An owner running contexts-only is using the product correctly; it is not a
    mistake to correct or re-seed.
    """
    client, headers = _client()
    with client:
        resp = client.delete(f"/me/desk/labels/{_norm(slug)}", headers=headers)
    typer.echo(json.dumps(_ok(resp), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
