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
  publish     Push a to-do onto the owner's plate.
  list        Show every to-do the owner currently has (filter by label, state, archived).
  update      Edit a to-do's text / due / draft prompt / labels.
  labels      Curate the owner's label vocabulary (nested sub-app).
  archive     File a to-do away into the Archived drawer.
  unarchive   Put an archived to-do back on the open plate.
  associate   Link a project or DM thread to a to-do.
  dissociate  Unlink one.
  projects    List a to-do's linked projects and threads.
  delete      Remove a to-do by id.
  trigger     Fire a to-do's saved draft at its designated recipient.

**Two axes, and they never move each other.** A to-do's TICKET STATE — New /
Working / Completed — is DERIVED from the projects it spawned and is read-only
here: there is no verb that sets it, and there must not be one. ``archive`` is
the owner's own disposal and says nothing about whether the work finished. An
archived to-do whose project is still running still reads Working, and that is
correct rather than a bug to reconcile.

``done`` / ``reopen`` are the old names for ``archive`` / ``unarchive`` and
survive ONE release as hidden aliases; ``--consume done`` likewise.

All of ``labels`` except ``labels list`` is likewise the assistant's.

``update`` / ``archive`` / ``unarchive`` / ``associate`` / ``dissociate`` /
``projects`` and ``trigger`` are the **assistant's** verbs: they go through
credentials that only the owner's ``{username}-assistant`` resolves through, so
any other agent gets a 401. ``publish`` / ``list`` / ``delete`` stay open to
every agent as before (``delete`` still only retracts what the caller published
— unless the caller is the assistant).
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
from clawmeets.models.desk_todo_link import TICKET_STATES

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
    gone. Only ``labels_detail`` is lost — the FILTER is unaffected, because it
    is a flat OR over the named slugs and asks the registry nothing. The reason
    goes to STDERR so stdout stays parseable JSON."""
    try:
        resp = client.get("/me/desk/labels", headers=headers)
    except Exception as e:
        typer.echo(f"Note: could not read the label list ({e}).", err=True)
        return None
    if resp.status_code >= 400:
        typer.echo(
            f"Note: could not read the label list ({resp.status_code}); "
            f"filtering as usual and omitting label detail.",
            err=True,
        )
        return None
    body = resp.json()
    rows = body.get("labels") if isinstance(body, dict) else body
    return rows if isinstance(rows, list) else []


def _matches(item: dict, wanted: list[str], match_all: bool) -> bool:
    """A FLAT OR over every named label — the rail's default.

    ``--label office --label home`` reads "at the office or at home".
    ``--match-all`` collapses it to strict AND over every named label.

    THIS USED TO GROUP BY KIND: "OR within a kind, AND across kinds", so that
    ``--label office --label home --label next`` read "next actions I could do
    at the office or at home" — the state narrowed the contexts. With the state
    kind retired every label is a context, the grouping has exactly one group
    for every possible input, and ``all(any(...))`` over one group IS a flat OR.
    So the ``kinds`` parameter is gone rather than left in place computing a
    partition that can no longer partition: a dict whose value is the same for
    every key reads like a live dimension and invites the next reader to add a
    third kind to it. The narrowing that state used to do is ``--state``, which
    is a separate predicate AND'd against this one.

    That is a real change in meaning for one input and it is worth naming: an
    owner with a legacy ``next`` row who says ``--label office --label next``
    now gets "carries office OR next" where they used to get "carries next AND
    (office)". They want ``--state`` for the second reading.

    Filtering is client-side over the full GET. No query params are added to the
    server, ever — a client that has only seen a filtered subset cannot compute
    a correct global order to send to ``/reorder``.
    """
    carried = set(item.get("labels") or [])
    if match_all:
        return all(slug in carried for slug in wanted)
    return any(slug in carried for slug in wanted)


def _label_detail(slugs: list[str], rows: list[dict]) -> list[dict]:
    """Join a to-do's slugs against the registry ALREADY loaded for ``_matches``.

    One dict per slug, same length and same order as ``labels``, so the two
    arrays index together and neither is a re-ordering of the other::

        {"slug": "wait-for", "name": "Wait For", "kind": "context", "registered": true}
        {"slug": "offce",    "name": "offce",    "kind": null,     "registered": false}

    An unregistered slug renders as its own name with a NULL kind, and the kind
    is never guessed — a guessed kind is read out loud as a fact about the
    owner's own vocabulary.

    ``kind`` is ``"context"`` for every registered row now: the state kind is
    retired and the server projects a stored one onto ``context`` as it reads.
    The key stays rather than being dropped, because callers parse this array
    and a disappearing key is a break where a constant value is not. It no
    longer distinguishes anything, so nothing should branch on it; ``state`` on
    the to-do is the axis that carries meaning.
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
        help="A GTD context to file this under (repeatable). "
             "Write `office` or `@office` — both land as the same label.",
    ),
) -> None:
    """Publish a to-do onto the owner's My Desk plate.

    ``--label`` uses the owner's OWN vocabulary — run ``clawmeets todo labels
    list`` first and reuse what is there. A label you invent still shows on the
    item but stays unregistered and never joins the owner's list. Every label is
    a context; a label cannot say the work is running, and nothing you pass here
    moves the item's ``state`` — that is derived from the projects linked to it.
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


def _norm_states(raw: list[str] | None) -> list[str]:
    """Validate the ``--state`` values, or exit 1 naming what is allowed.

    A typo here must be LOUD. The alternative — silently matching nothing — is
    indistinguishable from "you have no Working to-dos", which is an answer the
    assistant would read out as a fact about the owner's plate.
    """
    wanted = [v.strip().lower() for v in (raw or []) if v and v.strip()]
    bad = [v for v in wanted if v not in TICKET_STATES]
    if bad:
        typer.echo(
            f"Error: --state must be one of {', '.join(TICKET_STATES)} "
            f"(got {', '.join(repr(b) for b in bad)}).",
            err=True,
        )
        raise typer.Exit(1)
    # Deduped, so `--state new --state new` is not a different predicate.
    return list(dict.fromkeys(wanted))


@app.command("list")
def list_todos(
    label: list[str] = typer.Option(
        None, "--label",
        help="Only to-dos carrying this label (repeatable). Several labels are "
             "OR'd together.",
    ),
    match_all: bool = typer.Option(
        False, "--match-all",
        help="Require EVERY named label instead of the default OR. "
             "Labels only — it does not apply to --state.",
    ),
    state: list[str] = typer.Option(
        None, "--state",
        help="Only to-dos in this ticket state: new | working | completed "
             "(repeatable, OR'd together).",
    ),
    archived: bool = typer.Option(
        None, "--archived/--no-archived",
        help="Only archived / only unarchived. Default: both.",
    ),
    any_: bool = typer.Option(False, "--any", hidden=True),
) -> None:
    """List every to-do the calling agent's owner currently has.

    Unfiltered, this is a verbatim passthrough of the server's plate — including
    each item's ``labels`` (bare slugs, stored order), its derived ``state``, its
    ``archived`` flag and its ``association_count``. It costs nothing and works
    with the label registry gone.

    THREE INDEPENDENT PREDICATES, AND'd together::

        --state new --state working          (new OR working)
        --label office --label home          (office OR home — see below)
        --archived / --no-archived           tri-state; default is EVERYTHING

        --state working --label office       working AND at the office

    ``--state`` is the derived ticket state and the plate's own narrowing axis:
    the label vocabulary is contexts only, so "what am I actually working on"
    is a ``--state`` question and never a ``--label`` one.

    ``--match-all`` is LABEL-ONLY and deliberately does not extend to
    ``--state``: a to-do has exactly one state, so requiring two at once is
    unsatisfiable and a flag that always returns nothing is worse than no flag.

    ``--archived`` defaults to NEITHER — that is, everything, which is what this
    command has always returned and what lets the assistant answer "did I
    already file that one away?" without a second call.

    FILTERING IS CLIENT-SIDE, over the full fetch, for every one of the three.
    No query parameter is added to the server, ever: a client that has only seen
    a filtered subset cannot compute a correct global order to send to
    ``/reorder``.

    Only ``--label`` fetches the label registry, and now only to BUILD
    ``labels_detail`` — the label predicate itself is a flat OR over the named
    slugs and needs nothing from the registry. A ``--state``-only or
    ``--archived``-only call therefore still makes exactly ONE request, and
    still carries no ``labels_detail``: that array cannot be built without the
    registry, so it stays tied to ``--label`` rather than being emitted
    half-joined.

    On the ``--label`` path each row gains a sibling ``labels_detail`` array —
    same length and same order as ``labels`` — carrying each slug's display
    name, kind, and whether it is registered at all. ``labels`` itself is never
    rewritten. If the registry cannot be read, ``labels_detail`` is OMITTED and
    the reason goes to stderr so stdout stays parseable JSON; the FILTER is
    unaffected, because it no longer depends on the registry to decide anything.

    There is deliberately no ``--group-by``. This command emits a JSON ARRAY;
    a grouping flag would have to turn it into an object and break every parser
    reading it. Grouping is the browser's.
    """
    # `--any` is a hidden no-op alias for the default, kept so anything that
    # already says it keeps working. It is NOT an inversion of `--match-all`:
    # under an OR default, a flag named `--any` that turned ON strict AND reads
    # backwards and the voice path gets it wrong every time. It stays out of
    # `--help` and out of the SKILL.md, which is why this note is a comment and
    # not part of the docstring typer renders.
    wanted_states = _norm_states(state)
    client, headers = _client()
    with client:
        todos = _ok(client.get("/me/desk/todos", headers=headers))
        if not (label or wanted_states or archived is not None):
            # Verbatim passthrough — not re-serialized from a filtered copy, so
            # a row shape this CLI has never heard of still survives the trip.
            typer.echo(json.dumps(todos, indent=2, ensure_ascii=False))
            return

        items = [t for t in todos if isinstance(t, dict)]
        if archived is not None:
            items = [t for t in items if bool(t.get("archived")) is archived]
        if wanted_states:
            # A row from a server too old to derive `state` carries None, which
            # matches no requested state — it is filtered out rather than
            # guessed at, for the same reason `_label_detail` never invents a
            # kind.
            items = [t for t in items if t.get("state") in wanted_states]

        if label:
            wanted = [_norm(l) for l in label if l.strip()]
            # The predicate no longer needs the registry — it is a flat OR over
            # the named slugs. The fetch stays because ``labels_detail`` needs
            # it, and it stays INSIDE this branch for the same reason it always
            # did: a --state-only or --archived-only call must cost one request.
            items = [t for t in items if _matches(t, wanted, match_all)]
            rows = _label_rows(client, headers)
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


# The deprecation window for the pre-association verb names. ONE release: the
# skill file is re-read every turn, but an assistant that has fired `todo done` a
# hundred times will try it again out of habit, and the failure mode without an
# alias is `No such command` on the owner's plate — a refusal the owner reads as
# "the assistant is broken", not as "that verb was renamed". Hidden, so they
# stay out of `--help` and out of the SKILL.md and nothing new learns them.
# All three retire together, with the request-side `status` shim on PATCH.
_DEPRECATED = (
    "Note: `clawmeets todo {old}` is deprecated and will be removed in the "
    "next release — use `clawmeets todo {new}`."
)


@app.command("archive")
def archive(
    todo_id: str = typer.Argument(..., help="Id of the to-do to file away."),
) -> None:
    """File a to-do away — it moves to the plate's Archived drawer.

    THIS IS A DISPOSAL, NOT A VERDICT ON THE WORK. Archiving says the owner is
    done looking at the row; it says nothing about whether the project behind it
    finished, and it never moves the derived ticket state. A to-do whose project
    is still running reads Working after this and that is correct — the two axes
    are independent in both directions.
    """
    typer.echo(json.dumps(_patch(todo_id, {"archived": True}), indent=2, ensure_ascii=False))


@app.command("unarchive")
def unarchive(
    todo_id: str = typer.Argument(..., help="Id of the to-do to put back."),
) -> None:
    """Put an archived to-do back on the open plate.

    A separate verb rather than ``archive --undo`` so the skill's example lines
    read as instructions.
    """
    typer.echo(json.dumps(_patch(todo_id, {"archived": False}), indent=2, ensure_ascii=False))


@app.command("done", hidden=True)
def done_alias(
    todo_id: str = typer.Argument(..., help="Deprecated — use `todo archive`."),
) -> None:
    """Deprecated alias for ``todo archive``."""
    typer.echo(_DEPRECATED.format(old="done", new="archive"), err=True)
    archive(todo_id)


@app.command("reopen", hidden=True)
def reopen_alias(
    todo_id: str = typer.Argument(..., help="Deprecated — use `todo unarchive`."),
) -> None:
    """Deprecated alias for ``todo unarchive``."""
    typer.echo(_DEPRECATED.format(old="reopen", new="unarchive"), err=True)
    unarchive(todo_id)


@app.command("associate")
def associate(
    todo_id: str = typer.Argument(..., help="Id of the to-do."),
    project_id: str = typer.Argument(
        ..., help="Id of the project or DM thread to link to it."
    ),
) -> None:
    """Link a project or DM thread to a to-do — TO-DO FIRST, then the project.

    The argument order is fixed and the composer's failure sentence names this
    exact string, so do not swap them: a to-do id where a project id belongs
    fails with the project refusal, which reads as though the to-do did not
    exist.

    A DM thread and a regular project share ONE id space here and neither is
    special-cased. That matters because the desk's **command** button spawns a
    DM thread, which is the most common way a linked project comes into being.

    Linking is what makes a to-do read Working, and an associated project
    completing is what makes it read Completed. Nothing here touches
    ``archived`` — the owner's disposal is theirs alone.

    Idempotent: linking a project the to-do already carries is a success, not an
    error, so a retried command cannot double-apply.

    Two refusals, both plain sentences to read out as written:

    * the id matches nothing you can see — check it, or it was deleted;
    * it belongs to someone else — you can only link what you created.
    """
    client, headers = _client()
    with client:
        resp = client.post(
            f"/me/desk/todos/{todo_id}/projects",
            json={"project_id": project_id},
            headers=headers,
        )
    typer.echo(json.dumps(_ok(resp), indent=2, ensure_ascii=False))


@app.command("dissociate")
def dissociate(
    todo_id: str = typer.Argument(..., help="Id of the to-do."),
    project_id: str = typer.Argument(..., help="Id of the project to unlink."),
) -> None:
    """Unlink a project or DM thread from a to-do.

    This is also the only way to clear a DANGLING link — one whose project has
    been deleted — so it deliberately does not check that the id still resolves.
    A to-do showing a link count it cannot explain is cleared with this.

    Idempotent in the other direction: removing a link the to-do does not carry
    is a success.
    """
    client, headers = _client()
    with client:
        resp = client.delete(
            f"/me/desk/todos/{todo_id}/projects/{project_id}", headers=headers
        )
    typer.echo(json.dumps(_ok(resp), indent=2, ensure_ascii=False))


@app.command("projects")
def projects(
    todo_id: str = typer.Argument(..., help="Id of the to-do."),
) -> None:
    """List the projects and DM threads linked to a to-do — what makes it read
    New, Working or Completed, and why.

    Every row carries ``id``, ``resolved`` and ``contributes``. A row that does
    not resolve is LISTED rather than hidden, carrying only those three: a link
    the owner cannot see is one they cannot remove either, and ``dissociate`` is
    the way out. ``contributes: false`` on a resolved row means a FAILED
    project, which stops counting towards the state exactly as a deleted one
    does — that is what keeps a to-do whose only project failed from reading
    Working forever.
    """
    client, headers = _client()
    with client:
        resp = client.get(f"/me/desk/todos/{todo_id}/projects", headers=headers)
    typer.echo(json.dumps(_ok(resp), indent=2, ensure_ascii=False))


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


_CONSUME_CHOICES = ("archive", "delete", "keep")

# `done` survives ONE release as a hidden synonym for `archive`, on exactly the
# reasoning the `todo done` / `todo reopen` verb aliases get: an assistant that
# has fired `--consume done` a hundred times will fire it again, and the failure
# mode without it is a hard refusal AFTER the message has already been sent —
# the one moment in this command where a refusal costs the owner something.
# Retires on the same schedule as those aliases.
_CONSUME_ALIASES = {"done": "archive"}


@app.command("trigger")
def trigger(
    todo_id: str = typer.Argument(..., help="Id of the to-do to fire."),
    to: str = typer.Option(
        "", "--to", help="Override the recipient (short or full agent name)."
    ),
    consume: str = typer.Option(
        "archive", "--consume",
        help="archive | delete | keep — what to do with the item after a "
             "successful send.",
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

        {"sent": true,  "to": "chengtao-api-sync", "project_id": "…", "consumed": "archive"}
        {"sent": false, "reason": "no_draft_prompt" | "no_recipient"
                                  | "recipient_gone" | "already_archived", "detail": "…"}

    ``--consume archive`` (the default) files the item away where the desk
    deletes it, so a to-do fired by voice leaves a trace in the Archived drawer
    and ``todo unarchive`` can undo it. ``--consume delete`` is exact desk
    parity. A consume failure *after* a successful send is reported, never
    retried — the message is already out.

    Note what firing this does NOT do: it does not associate the thread it opens
    with the to-do, so the item's ticket state does not move to Working. A DM
    thread never completes — no user path, no browser control, and an agent in
    the owner's DM runs on an action set that cannot emit ``project_completed``
    — so a to-do linked only that way would read Working for as long as it
    existed. Archiving IS the disposal for a triggered item, which is why it is
    the default.
    """
    if consume in _CONSUME_ALIASES:
        typer.echo(
            f"Note: `--consume {consume}` is deprecated and will be removed in "
            f"the next release — use `--consume {_CONSUME_ALIASES[consume]}`.",
            err=True,
        )
        consume = _CONSUME_ALIASES[consume]
    if consume not in _CONSUME_CHOICES:
        typer.echo(
            f"Error: --consume must be one of {', '.join(_CONSUME_CHOICES)}.", err=True
        )
        raise typer.Exit(1)

    client, headers = _client()
    with client:
        item = _find(client, headers, todo_id)

        if item.get("archived"):
            _noop("already_archived", "That one is already filed away.")
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
        if consume == "archive":
            resp = client.patch(
                f"/me/desk/todos/{todo_id}", json={"archived": True}, headers=headers
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
# honest about detaching the slug from every item on the way through. With the
# state kind retired there is also nothing left to set it TO — but the rule
# outlives the vocabulary, so the absence stays deliberate rather than
# accidental.
# ===========================================================================

labels_app = typer.Typer(
    name="labels",
    help="List and curate the owner's label vocabulary of contexts.",
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
    """The forgiving --kind parser. Every path that succeeds returns
    ``"context"`` — the only kind there is.

    Voice will never say the word "kind", so the assistant transcribing "make
    that a context" must not have to guess a canonical spelling.
    Case-insensitive, after stripping a leading sigil and surrounding
    whitespace::

        context | contexts | @ | @context       -> "context"
        any registered label's slug OR display  -> that label's kind, which the
          name (office, @office, "the office")     server reports as "context"
        omitted                                 -> "context"
        state | states | ! | !state             -> exit 1, NAMING the retirement
        anything else                           -> exit 1, never a silent default

    THE RETIRED TOKENS REFUSE HERE RATHER THAN ON THE WIRE, and that is the one
    place this parser is deliberately un-forgiving. Mapping ``state`` quietly to
    ``context`` would let an assistant that asked for a state believe it got
    one, which is the same lie ``labels.invalid_kind`` exists to refuse — and
    forwarding it to be refused by the server costs a round trip to say
    something the CLI already knows. The sentence names ``--state`` because the
    caller wants lifecycle and lifecycle still exists; it just is not a label.

    This is a CLI-side parse ONLY. The wire still carries the strict enum and
    still answers ``labels.invalid_kind``, because a forgiving boundary over a
    strict wire is the correct split; the inverse is how a third kind value gets
    stored by accident.
    """
    raw = (raw or "").strip()
    if not raw:
        return "context"
    if raw == "@":
        return "context"
    token = raw.lstrip("@!").strip().lower()
    if raw == "!" or token in ("state", "states"):
        typer.echo(
            "Error: the 'state' kind is retired — every label is a context "
            "now. Whether an item's work is running is derived on its `state` "
            "(new / working / completed) from the projects linked to it; filter "
            "on it with `clawmeets todo list --state working`. Add this as an "
            "ordinary label and it will still group the plate.",
            err=True,
        )
        raise typer.Exit(1)
    if token in ("context", "contexts"):
        return "context"
    # Only a LOOKUP needs the registry, so the round trip is paid only by the
    # callers that actually say "the same kind as @office".
    for row in _registry_rows():
        if not isinstance(row, dict):
            continue
        if token in (
            str(row.get("slug", "")).lower(),
            str(row.get("name", "")).strip().lower(),
        ):
            # The server already projects a retired state onto "context", so
            # this is "context" for every row a current server can send. It is
            # re-projected here anyway rather than forwarded: an older server in
            # front of a newer CLI would otherwise hand back "state" and we
            # would put a value on the wire that this CLI just refused to accept
            # from its own caller.
            stored = row.get("kind") or "context"
            return "context" if stored == "state" else stored
    typer.echo(
        f"Error: --kind must be 'context' (or the name of a label you already "
        f"have); {raw!r} matched neither.",
        err=True,
    )
    raise typer.Exit(1)


@labels_app.command("list")
def labels_list() -> None:
    """Show the owner's label vocabulary, in the order it is grouped in.

    Readable by every agent — run it before you `publish --label` so you reuse
    the owner's own vocabulary instead of inventing a near-duplicate. Every row
    comes back `kind: "context"`; the state kind is retired, so the field no
    longer distinguishes anything and nothing should branch on it. Whether an
    item's work is running is its `state`, not one of these.
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
        help='"context" — the only kind, and the default, so you can omit it. '
             "Also accepts @ or the name of a label you already have. The "
             '"state" kind is retired; an item\'s state is derived, not labelled.',
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

    There is deliberately no way to change a label's kind here, and with the
    state kind retired there is no other kind to change one to. The repair for a
    slug that is genuinely wrong is still `delete` then `add` — say first how
    many items that will detach it from, AND that you cannot put the new label
    back on them automatically.
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
    KIND, which for anything created now means both are contexts and the check
    never fires. It can still fire on a registry written before the state kind
    was retired: such a row reads as a context but is STORED as a state, and
    merging it into a real context comes back `kind_mismatch`. That is the
    server refusing to delete a row of the owner's across a distinction it can
    no longer show them — not a bug to work around. Leave it, or `delete` it
    deliberately if the owner asks.

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

    Every label deletes the same way and deleting all of them is allowed.
    An owner running with an empty vocabulary is using the product correctly;
    it is not a mistake to correct or re-seed.
    """
    client, headers = _client()
    with client:
        resp = client.delete(f"/me/desk/labels/{_norm(slug)}", headers=headers)
    typer.echo(json.dumps(_ok(resp), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
