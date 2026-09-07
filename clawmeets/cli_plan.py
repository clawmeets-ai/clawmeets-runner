# SPDX-License-Identifier: MIT
"""
clawmeets/cli_plan.py

``clawmeets plan <subcmd>`` — the terminal surface over the collaborative
project plan (``shared-context/PLAN.md`` and its sidecar).

**This module is transport.** Every write goes through ``PUT/POST …/plan…``,
which goes through ``apply_edits``; every text operation is a call into
:mod:`clawmeets.models.plan_markdown`. There is no parsing, no slicing and no
heading logic here, for the same reason there is none in the routes: a second
implementation of "where does section ``m2`` start" is how a terminal and a
browser come to disagree about what a plan says.

Modelled on ``cli_sop.py`` / ``cli_client.py``: auth from the injected agent
runtime with ``--token`` / ``--server`` escape hatches, human text by default
and ``--json`` on request, exit ``1`` with stderr on error.

**Every command works on every regular project, including pre-feature ones.**
There is no adoption step — the sidecar is created lazily by the first write.
A project with no ``shared-context/PLAN.md`` (a DM, the Front Desk) is ``404``.

Subcommands (§5)
----------------
  create      Seed PLAN.md from --body-file or the one template.
  show        The document, a section, a note, the index, the versions.
  update      Replace/append/retitle/delete one section. Keeper or owner.
  note        A comment, with or without a proposal. Everyone's channel.
  resolve     apply / reject / answered / dismiss. --apply on the go-note
              is how a plan is accepted, and --apply on any later note is
              how the owner re-signs what it changed.
  list-notes  Filter on every documented axis; --thread reads a thread.
  conflicts   The keeper's refused writes, with the command that closes each.
  consult     Seat the specialists in `shared-context` so they can be reached.

Flags this CLI deliberately does NOT have
-----------------------------------------
``--patch-file`` / any diff input (AC-4.4 — no surface anywhere accepts a
diff), ``--base`` (the CLI captures the base itself and there is no whole-body
form), ``--plan-mode``, ``--spec`` / ``--tracker`` / ``--scope`` (there is no
scope classification to filter on, §2.1).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import httpx
import typer

from clawmeets.cli_runner import DEFAULT_SERVER, _http, _resolve_project_ref
from clawmeets.models.plan_markdown import (
    append_to_section,
    parse_sections,
    quote_names_criterion,
    retitle_section,
    section_extent,
)
from clawmeets.models.project_plan import (
    PlanHistoryEntry,
    PlanInputError,
    PlanNote,
    PlanReviewRound,
    check_note_text,
    pending_notes,
    render_batch_message,
    review_room_for,
)

app = typer.Typer(
    name="plan",
    help="The collaborative project plan. Paired skill: plan.",
    no_args_is_help=True,
)

#: Every command takes the project first, as a name, a display name or an id.
PROJECT_ARG = typer.Argument(..., help="Project name, display name, or id.")


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------


def _fail(message: str) -> "typer.Exit":
    typer.echo(f"Error: {message}", err=True)
    return typer.Exit(1)


def _headers(token: Optional[str]) -> dict[str, str]:
    """``--token`` overrides; otherwise ``_http``'s default headers already
    carry the per-process agent identity the runner injects."""
    return {"Authorization": f"Bearer {token}"} if token else {}


def _detail(resp: httpx.Response) -> Any:
    """The server's own message, never a status code on its own.

    FastAPI wraps a raised ``HTTPException`` in ``{"detail": …}``; the plan
    routes put a **structured** body there for ``409`` (the stale sections and
    the conflict-note ids) and a plain string for everything else. Both come
    back untouched so the caller can print what the server actually said.
    """
    try:
        payload = resp.json()
    except ValueError:
        return resp.text
    return payload.get("detail", payload) if isinstance(payload, dict) else payload


def _ok(resp: httpx.Response) -> Any:
    if resp.status_code >= 400:
        detail = _detail(resp)
        if isinstance(detail, dict):
            detail = detail.get("error") or json.dumps(detail)
        raise _fail(f"{resp.status_code}: {detail}")
    if not resp.content:
        return {}
    return resp.json()


def _echo_json(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _pid(client: httpx.Client, token: Optional[str], ref: str) -> str:
    return _resolve_project_ref(client, token, ref)


def _url(pid: str, suffix: str = "") -> str:
    return f"/projects/{pid}/plan{suffix}"


def _index(client: httpx.Client, headers: dict[str, str], pid: str, **params) -> dict:
    """``GET /plan`` — the body, the derived index, the notes, and (for the
    owner) the tray. One read backs every command here."""
    return _ok(client.get(_url(pid), headers=headers, params=params or None))


def _read(text: Optional[str], path: Optional[Path], *, what: str) -> Optional[str]:
    """Resolve a body from an inline flag or a file, never both.

    Checked before any HTTP call so a mistake costs nothing.
    """
    if text is not None and path is not None:
        raise _fail(f"pass either the inline {what} or its file, not both.")
    if path is not None:
        if not path.exists():
            raise _fail(f"{path} does not exist.")
        return path.read_text(encoding="utf-8")
    return text


def _section_text(body: str, slug: str) -> Optional[str]:
    """The section's current text — the ``base`` every write is checked against.

    Delegated to phase 1 rather than sliced here. **This is the whole of "there
    is no ``--base`` flag"**: the base is the section text the CLI just read, so
    asking a caller to supply one would be asking them to retype what the read
    already returned.
    """
    extent = section_extent(body, slug)
    return None if extent is None else body[extent[0]:extent[1]]


def _resolve_ac(index: dict, ac: str, project: str) -> tuple[str, str]:
    """Resolve ``--ac AC-2.3`` to the ``(section, quote)`` pair a note carries.

    Read off ``index["criteria"]`` — the derived index the server already
    returned — so there is no parsing here, for the same reason ``_section_text``
    delegates its slicing: a second answer to *"where is AC-2.3"* is how a
    terminal and a browser come to disagree about what a plan says.

    An unknown id is refused and the refusal names where the ids are, mirroring
    the ``--reply-to`` refusal. An id written twice **warns and takes the
    first** rather than refusing — the same "say it, do not refuse it" stance
    ``_show_sections`` takes on a duplicate heading, and first in document order
    is the definition.
    """
    hits = [c for c in index.get("criteria", []) if c["id"].upper() == ac.upper()]
    if not hits:
        raise _fail(
            f"{ac} is not on this plan. "
            f"`clawmeets plan show {project} --sections` prints the criteria."
        )
    if len(hits) > 1:
        typer.echo(
            f"⚠ {ac} appears {len(hits)} times; anchoring to the first, "
            f"in `{hits[0]['section'] or '(preamble)'}`.",
            err=True,
        )
    return hits[0]["section"], hits[0]["quote"]


def _proposal_of(client, headers: dict[str, str], pid: str, note_id: str) -> str:
    """The note's proposed text. Best-effort: a refusal that prints one text is
    still better than one that prints none."""
    try:
        index = _index(client, headers, pid, note=note_id)
    except typer.Exit:
        return ""
    for note in index.get("notes", []):
        if note["id"] == note_id:
            return note.get("proposal") or ""
    return ""


def _header_line(index: dict) -> str:
    """§5.2's header: the revision, and acceptance state when there is any."""
    parts = [f"rev {index.get('revision', 1)} · {str(index.get('sha', ''))[:12]}"]
    # Printed in full: it is a value to be COPIED — into a review batch, into a
    # bug report — and a display truncation would be a version nobody can
    # reconstruct from the surface that showed it to them.
    if index.get("spec_digest"):
        parts.append(f"spec {index['spec_digest']}")
    if index.get("accepted"):
        rev = index.get("accepted_revision")
        parts.append(
            f"Accepted: revision {rev} — current revision {index.get('revision')}"
        )
        if index.get("changed_since_acceptance"):
            parts.append("⚠ the plan changed since acceptance")
    keeper = index.get("keeper")
    if keeper:
        parts.append(f"kept by @{keeper}")
    return "   ".join(parts)


def _print_stale(detail: dict) -> None:
    """A refusal, printed as **information rather than a failure**.

    §5.3: there is no retry loop and no ``--force`` on ``update``. The ``409``
    already carries your text beside what the section says now, so the CLI never
    needs a second request to tell you what happened — and the conflict note it
    filed means the refusal is not a dead end (§4.3).
    """
    for row in detail.get("stale", []):
        typer.echo(f"\n  section `{row['section']}` — what it says now:", err=True)
        for line in (row.get("current") or "").splitlines() or ["(deleted)"]:
            typer.echo(f"    {line}", err=True)
    for nid in detail.get("note_ids", []):
        typer.echo(f"\n  Your text is note {nid} for the user.", err=True)


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


@app.command("create", help="Create PLAN.md from --body-file or the one template. 409 if it exists.")
def create(
    project: str = PROJECT_ARG,
    body_file: Optional[Path] = typer.Option(
        None, "--body-file", help="Seed Markdown. Omitted, the one template is used."
    ),
    title: str = typer.Option("", "--title", help="Document title; defaults to the display name."),
    template: str = typer.Option(
        "lean", "--template", help="The only template. Present so a caller need not know that."
    ),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    json_out: bool = typer.Option(False, "--json"),
):
    """Create PLAN.md. ``409`` if it already exists.

    Exists for the manual case and for a deleted PLAN.md; **its ordinary caller
    is ``POST /projects``**, which passes the coordinator's v1 through as
    ``plan_body``. There is no state in which a plan is unwritten.
    """
    if template != "lean":
        raise _fail(f"unknown template {template!r} — `lean` is the only one (§5.1).")
    body = _read(None, body_file, what="body")
    headers = _headers(token)
    with _http(server) as client:
        pid = _pid(client, token, project)
        payload: dict[str, Any] = {"title": title}
        if body is not None:
            payload["body"] = body
        result = _ok(client.post(_url(pid), json=payload, headers=headers))
    if json_out:
        _echo_json(result)
        return
    typer.echo(f"Created PLAN.md — revision {result.get('revision', 1)}")


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


def _show_sections(index: dict) -> None:
    """The diagnostic that makes derived structure inspectable (§5.2).

    The ``warning`` column is **derived, not a wire member**: two sections with
    equal ``heading`` and slugs ``x``/``x-2`` *are* the duplicate-heading
    warning, and §6.B's ``PlanIndex`` is a closed enumeration precisely so a
    fact the response already carries does not get a second name.
    """
    sections = index.get("sections", [])
    counts: dict[str, int] = {}
    for sec in sections:
        counts[sec["heading"]] = counts.get(sec["heading"], 0) + 1
    typer.echo(_header_line(index))
    typer.echo(f"{'id':<22} {'heading':<34} {'depth':<6} {'boxes':<7} warning")
    for sec in sections:
        boxes = sec.get("boxes") or [0, 0]
        box = f"{boxes[0]}/{boxes[1]}" if boxes[1] else "—"
        heading = sec["heading"] or "(preamble)"
        warn = "duplicate heading" if counts.get(sec["heading"], 0) > 1 else ""
        clipped = heading if len(heading) <= 33 else heading[:32] + "…"
        typer.echo(f"{sec['id']:<22} {clipped:<34} {sec['depth']:<6} {box:<7} {warn}")


def _show_criteria(index: dict) -> None:
    """The criteria index, printed under the sections table — the ids
    ``plan note --ac`` takes.

    Nothing at all when the plan has no criteria: a plan that does not use the
    convention should not grow a heading for an empty table. ``duplicate id`` is
    derived here exactly as ``duplicate heading`` is above, from two rows
    sharing a value, rather than being a wire member §6.B does not have.
    """
    criteria = index.get("criteria", [])
    if not criteria:
        return
    counts: dict[str, int] = {}
    for row in criteria:
        counts[row["id"]] = counts.get(row["id"], 0) + 1
    typer.echo("")
    typer.echo(f"{'criterion':<12} {'section':<22} {'done':<6} {'text':<44} warning")
    for row in criteria:
        done = "—" if row.get("checked") is None else ("yes" if row["checked"] else "no")
        text = row.get("quote") or row.get("text", "").strip()
        clipped = text if len(text) <= 43 else text[:42] + "…"
        warn = "duplicate id" if counts[row["id"]] > 1 else ""
        typer.echo(
            f"{row['id']:<12} {(row['section'] or '(preamble)'):<22} "
            f"{done:<6} {clipped:<44} {warn}"
        )


def _show_note(index: dict, note_id: str, diff: bool) -> None:
    notes = index.get("notes", [])
    note = next((n for n in notes if n["id"] == note_id), None)
    if note is None:
        raise _fail(f"note {note_id!r} is not on this plan.")
    typer.echo(f"{note['id']}  [{note['status']}/{note['kind']}]  "
               f"by {note['by'] or '—'} to {note['to'] or '(nobody)'}  "
               f"section `{note['section'] or '—'}`  {note.get('at', '')[:10]}")
    if note.get("comment"):
        typer.echo(f"\n{note['comment']}")
    if note.get("quote"):
        typer.echo(f"\nQuoted excerpt:\n> " + "\n> ".join(note["quote"].splitlines()))
    if note.get("section_changed"):
        typer.echo("\n⚠ this section changed since the note was written")
    if note.get("section_missing"):
        typer.echo("\n⚠ this section no longer exists (dangling)")
    if diff:
        if not note.get("display_diff"):
            typer.echo("\n(no proposal on this note — nothing to diff)")
        else:
            typer.echo("\n" + note["display_diff"])
    elif note.get("proposal"):
        typer.echo(f"\nProposed text:\n{note['proposal']}")
    for reply in note.get("replies", []):
        typer.echo(f"  reply: {reply}")


def _show_versions(rows: list[dict]) -> None:
    typer.echo(f"{'rev':<5} {'sha':<14} {'by':<26} {'when':<12} proposed by / note")
    for row in rows:
        who = ", ".join(row.get("proposed_by") or []) or "—"
        note = f" · {row['note']}" if row.get("note") else ""
        typer.echo(
            f"{row['revision']:<5} {row['sha'][:12]:<14} {row['by'] or '—':<26} "
            f"{row['at'][:10]:<12} {who}{note}"
        )


@app.command("show", help="Read the plan: the file, a section, a note, the index, the versions.")
def show(
    project: str = PROJECT_ARG,
    section: str = typer.Option("", "--section", help="One section, by slug."),
    note: str = typer.Option("", "--note", help="One note, by id."),
    diff: bool = typer.Option(False, "--diff", help="With --note: render its proposal as a diff."),
    clean: bool = typer.Option(
        False, "--clean",
        help="Strip EVERY HTML comment — the 'print the plan' mode. A render, never a write.",
    ),
    sections: bool = typer.Option(False, "--sections", help="The derived index."),
    revision: bool = typer.Option(False, "--revision", help="The revision label and acceptance state."),
    versions: bool = typer.Option(False, "--versions", help="The revision history, newest first."),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    json_out: bool = typer.Option(False, "--json"),
):
    """Read the plan. No flags prints the file verbatim.

    **``--clean`` has no exception for provenance, deliberately.** It strips
    ``<!-- src: … -->`` along with everything else, so an agent reading the
    contract sees the fact without the citation. That is a consequence of
    provenance being an inert comment — a carve-out is what would make an inert
    comment stop being inert. It is a **render**: the stored bytes are untouched,
    exactly as ``spec_digest``'s normalization is a digest input and never a
    write.
    """
    headers = _headers(token)
    with _http(server) as client:
        pid = _pid(client, token, project)
        if versions:
            rows = _ok(client.get(_url(pid, "/versions"), headers=headers))
            if json_out:
                _echo_json(rows)
            else:
                _show_versions(rows)
            return
        params: dict[str, Any] = {}
        if section:
            params["section"] = section
        if note:
            params["note"] = note
        if clean:
            params["clean"] = True
        if sections:
            params["sections"] = True
        index = _index(client, headers, pid, **params)

    if json_out:
        _echo_json(index)
        return
    if sections:
        _show_sections(index)
        _show_criteria(index)
    elif note:
        _show_note(index, note, diff)
    elif revision:
        typer.echo(_header_line(index))
    else:
        typer.echo(index.get("body", ""))


# ---------------------------------------------------------------------------
# update — keeper or owner, one section, no retry
# ---------------------------------------------------------------------------


@app.command("update", help="Write one section. Owner or keeper only; no whole-body form, no retry.")
def update(
    project: str = PROJECT_ARG,
    section: str = typer.Option(..., "--section", help="The section slug. Required."),
    body_file: Optional[Path] = typer.Option(None, "--body-file", help="The section's new text."),
    text: Optional[str] = typer.Option(None, "--text", help="The section's new text, inline."),
    append: bool = typer.Option(False, "--append", help="Append to the section instead of replacing it."),
    title: str = typer.Option("", "--title", help="Rewrite the heading text. Re-slugs the section."),
    delete: bool = typer.Option(False, "--delete", help="Remove the section."),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    json_out: bool = typer.Option(False, "--json"),
):
    """Write one section. **Owner or keeper only.**

    ``--section`` is required and there is no whole-body form: the section is the
    unit of change, located by heading, so a write lands correctly at any
    revision (§2.3).

    **No ``--base`` and no retry.** The base is the section text this command
    just read. On ``409`` it prints what the section says now beside what you
    wrote and stops — a refusal is information, and the conflict note it filed
    is how it gets resolved (§4.3).
    """
    supplied = _read(text, body_file, what="text")
    if delete and (supplied is not None or append or title):
        raise _fail("--delete takes no text, no --append and no --title.")
    if not delete and supplied is None and not title:
        raise _fail("one of --body-file / --text / --title / --delete is required.")
    if append and supplied is None:
        raise _fail("--append needs --body-file or --text.")

    headers = _headers(token)
    with _http(server) as client:
        pid = _pid(client, token, project)
        index = _index(client, headers, pid)
        body = index.get("body", "")
        base = _section_text(body, section)
        create = False
        if base is None and not delete:
            # A create (§2.3) — a new section at the end. **It says so now.**
            # The empty base used to carry this on its own, and that was the
            # collision: a section that had been DELETED also has no current
            # text, so a proposal filed against one read as a create and was
            # appended beside the copy it meant to replace. `create` is the
            # fact; the empty base is only what a create happens to compare
            # against. This is the one command that genuinely means it.
            base, create = "", True
        elif base is None:
            raise _fail(f"section {section!r} does not exist; nothing to delete.")

        new_text = _new_section_text(
            body, section, supplied, append=append, title=title, delete=delete
        )
        # ``create`` rides the wire ONLY when it is true. It is an opt-in marker
        # on the one command that means it, not a fourth member every edit has
        # to carry: the replace payload stays the three the funnel has always
        # taken, and a client that never creates never has to know the flag
        # exists.
        edit: dict[str, Any] = {"section": section, "text": new_text, "base": base}
        if create:
            edit["create"] = True
        resp = client.put(_url(pid), json={"edits": [edit]}, headers=headers)
        if resp.status_code == 403:
            # **Print the server's own refusal; do not compose one here.**
            #
            # This arm used to print a fixed line sending the caller to
            # `plan note … --to <keeper> --edit-file`. M2 made that exact call
            # refuse that exact caller, so the client was overwriting the
            # server's corrected remedy with a stale one — and it was the only
            # surface that printed either, so the server's text was never seen.
            #
            # M3 makes it worse than stale: a **coordinator** whose spec edit is
            # refused after acceptance gets a 403 too, and would have been told
            # it is a worker and pointed at a command it may not run. There is
            # one rule and the server owns it; a second copy here is a second
            # answer waiting to disagree.
            raise _fail(str(_detail(resp)))
        if resp.status_code == 409:
            detail = _detail(resp)
            typer.echo(
                f"Someone changed `{section}` while you were writing it.", err=True
            )
            if isinstance(detail, dict):
                _print_stale(detail)
            typer.echo(f"  clawmeets plan show {project} --section {section}", err=True)
            raise typer.Exit(1)
        result = _ok(resp)

    if json_out:
        _echo_json(result)
        return
    if result.get("noop"):
        typer.echo("No change — the section already says that.")
    else:
        typer.echo(
            f"Wrote {', '.join(result.get('sections') or [section])} — "
            f"revision {result.get('revision')}"
        )


def _new_section_text(
    body: str,
    section: str,
    supplied: Optional[str],
    *,
    append: bool,
    title: str,
    delete: bool,
) -> str:
    """The section's replacement text, built **entirely by phase 1**.

    ``--append`` and ``--title`` are conveniences over ``append_to_section`` and
    ``retitle_section``; neither is re-implemented here, because "where does the
    heading line end" is grammar and this module has none.
    """
    if delete:
        return ""                      # SectionEdit.text == "" deletes (§2.3).
    if append and supplied is not None:
        spliced = append_to_section(body, section, supplied)
        extent = section_extent(spliced, section)
        new_text = spliced[extent[0]:extent[1]] if extent else supplied
    elif supplied is not None:
        new_text = supplied
    else:
        new_text = _section_text(body, section) or ""
    if title:
        parsed = parse_sections(new_text)
        if not parsed or not parsed[0].heading:
            raise _fail(f"section {section!r} has no heading line to retitle.")
        new_text = retitle_section(new_text, parsed[0].id, title)
    return new_text


# ---------------------------------------------------------------------------
# note — everyone's channel, and the only one for a worker
# ---------------------------------------------------------------------------


@app.command(
    "note",
    help="File a note: a comment, with or without a proposal. The user and the coordinator only.",
)
def note(
    project: str = PROJECT_ARG,
    section: str = typer.Option("", "--section", help="Section slug (advisory; required with --edit-file)."),
    to: list[str] = typer.Option(
        None, "--to",
        help="Agent name or `user`. Repeatable → N SIBLING notes. Omitted = recorded, never sent.",
    ),
    comment: Optional[str] = typer.Option(None, "--comment", "-m", help="What you want to say."),
    comment_file: Optional[Path] = typer.Option(None, "--comment-file"),
    edit_file: Optional[Path] = typer.Option(
        None, "--edit-file",
        help="The proposal: the section's REPLACEMENT TEXT. Requires --section. Never a diff.",
    ),
    quote: str = typer.Option("", "--quote", help="The excerpt this is about."),
    ac: str = typer.Option(
        "", "--ac",
        help="Anchor to an acceptance criterion, e.g. AC-2.3. Fills in --section and --quote.",
    ),
    reply_to: str = typer.Option("", "--reply-to", help="Parent note id — makes this a thread reply."),
    as_user: bool = typer.Option(
        False,
        "--as-user",
        help="Act as the project owner. The owner's ASSISTANT only, in the "
             "owner's own DM, on their explicit request.",
    ),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    json_out: bool = typer.Option(False, "--json"),
):
    """File a note. **Never refused for staleness** — a note cannot clobber
    anything, so a view several revisions behind is still a real objection.

    ``--to`` repeated creates **N sibling notes** with independent ids and
    statuses: one note with one status cannot represent *"two agreed, one
    objected."*

    ``--edit-file`` is the proposal and it is the section's replacement text.
    **There is no ``--patch-file`` and no way to submit a diff** (AC-4.4); the
    CLI captures ``base_section`` for you from the section as it reads now.

    **There is no ``--deviation`` flag, and there is nothing to replace it
    with.** A deviation is a coordinator note to the user filed after the user
    accepted the plan — the server derives it from ``at``, ``to`` and ``by`` on
    every read. So after acceptance this command already files one and the flag
    would have been redundant; before acceptance nothing has been agreed and the
    flag would have been a claim the plan cannot support. ``--section`` still
    carries the locator: ``section`` + ``base_section`` name the clause the work
    departed from. This makes a deviation **recordable and visible; nothing here
    makes one detected.**

    ``--ac AC-2.3`` is the finest address a note can carry: it resolves to the
    criterion's section **and** its text, so the note renders against that one
    line instead of against the whole milestone. The id is not a stored field —
    it rides inside the quote, so this is an ordinary quoted note and a
    renumbered criterion re-anchors by text rather than dangling on a dead id.
    """
    body_comment = _read(comment, comment_file, what="comment")
    proposal = _read(None, edit_file, what="proposal")
    # REFUSED ALONGSIDE, never merged. `--ac` IS a (section, quote) pair, so
    # letting an explicit `--section` win would file a note whose quote is not
    # in the section it names — precisely the loose note `--ac` exists to
    # prevent, arrived at by a route that looks like it worked. The refusal
    # names the flag to DROP, not the one to keep: the caller typed both and
    # only they know which they meant.
    if ac and (section or quote):
        clash = " and ".join(
            f"--{f}" for f, on in (("section", section), ("quote", quote)) if on
        )
        raise _fail(f"--ac already resolves to a section and a quote; drop {clash}.")
    # `--reply-to` and `--ac` are the two exemptions, and neither is a
    # loosening: the section is still REQUIRED, it is just resolved below —
    # off the parent, or off the criterion — instead of typed again. Without
    # the first, answering a question with the change it asks for (the one
    # shape that renders a diff the user can Accept) cannot be written as the
    # documented one-liner, and an agent that omits `--section` gets a refusal
    # for a field it has no reason to know. The second is the same bargain: a
    # criterion id already names the section it lives in.
    if proposal is not None and not section and not reply_to and not ac:
        raise _fail("--edit-file is a replacement for a section; pass --section too.")
    if not body_comment and proposal is None:
        raise _fail("a note needs one of --comment / --comment-file / --edit-file.")
    # The body is assembled **before** the first request, so the guard below can
    # walk the thing that actually goes on the wire rather than a hand-copied
    # list of some of its fields. ``base_section`` is the one member the server
    # supplies; it is filled in below, after the index read, and is text the
    # server already decoded from UTF-8 so it can never be the offender.
    payload: dict = {
        "section": section,
        "to": list(to or []),
        "comment": body_comment or "",
        "proposal": proposal or "",
        "base_section": "",
        "quote": quote,
        "reply_to": reply_to,
    }
    # The server owns this rule and refuses it too (``_validate_notes_locked``
    # per field, ``_save`` for the sidecar as a whole), but the server never
    # sees this request: ``httpx`` encodes the JSON body with
    # ``ensure_ascii=False``, so a lone surrogate raises ``UnicodeEncodeError``
    # **inside the client** and the terminal gets a bare traceback with no
    # message and no exit-code contract. A CLI can mint one where a browser
    # cannot — POSIX decodes ``argv`` with ``surrogateescape``, so any byte
    # sequence that is not valid UTF-8 arrives as lone surrogates.
    #
    # **Every string in the payload, not three of them.** ``--section`` and
    # ``--reply-to`` come from ``argv`` exactly as ``--quote`` does and reached
    # the same bare traceback; so does each ``--to``. Looping over the payload
    # means a flag added later is covered the day it is added, instead of the
    # day someone remembers to extend a list.
    try:
        # `--ac` never reaches the wire — it resolves into `section` and
        # `quote` below — but it does reach a refusal message, and echoing a
        # lone surrogate raises the same bare traceback the payload loop exists
        # to prevent. Checked here rather than added to the payload, because it
        # is an address the CLI consumes and not a field the note carries.
        check_note_text("ac", ac)
        for field, value in payload.items():
            for text in value if isinstance(value, list) else [value]:
                if isinstance(text, str):
                    check_note_text(field, text)
    except PlanInputError as exc:
        raise _fail(str(exc)) from None

    headers = _headers(token)
    with _http(server, as_user=as_user) as client:
        pid = _pid(client, token, project)
        # ONE read backs both of the paths below, and `--ac --edit-file` needs
        # both halves of it: the criteria to resolve the anchor, and the body to
        # capture the base for the section it resolved to.
        index = _index(client, headers, pid) if (ac or proposal is not None) else {}
        if ac:
            section, quote = _resolve_ac(index, ac, project)
            payload["section"] = section
            payload["quote"] = quote
        if proposal is not None:
            if not section:
                # The parent's section, off the index this call already fetches
                # — no second request. The server derives the same value for a
                # comment-only reply; a PROPOSAL cannot wait for it, because
                # `base_section` is captured HERE and is keyed on the section.
                parent = next(
                    (n for n in index.get("notes", []) if n.get("id") == reply_to), None
                )
                if parent is None:
                    raise _fail(
                        f"--reply-to {reply_to} is not a note on this plan. "
                        f"`clawmeets plan list-notes {project}` prints the ids."
                    )
                section = parent.get("section") or ""
                if not section:
                    raise _fail(
                        f"note {reply_to} is on the document as a whole, so there "
                        f"is no section for --edit-file to replace; pass --section."
                    )
                payload["section"] = section
            payload["base_section"] = _section_text(index.get("body", ""), section) or ""
        result = _ok(client.post(_url(pid, "/notes"), json=payload, headers=headers))

    if json_out:
        _echo_json(result)
        return
    if as_user:
        # Named out loud, for the same reason `resolve` names it: the only guard
        # on this flag is a prompt, so the transcript has to carry the fact.
        typer.echo("Acting as the project owner, not as this agent.")
    for nid in result.get("note_ids", []):
        typer.echo(nid)


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------


@app.command("resolve", help="Close a note out: --apply / --reject / --answered / --dismiss.")
def resolve(
    project: str = PROJECT_ARG,
    note_id: str = typer.Argument(..., help="The note id, e.g. n-a91f."),
    apply: bool = typer.Option(False, "--apply", help="Take the proposal at the current revision."),
    force: bool = typer.Option(False, "--force", help="With --apply, when the section has changed."),
    reject: bool = typer.Option(False, "--reject", help="Decline it; the proposal is kept."),
    answered: bool = typer.Option(False, "--answered"),
    dismiss: bool = typer.Option(False, "--dismiss"),
    reason: str = typer.Option("", "--reason", help="Required with --reject."),
    as_user: bool = typer.Option(
        False,
        "--as-user",
        help="Act as the project owner. The owner's ASSISTANT only, in the "
             "owner's own DM, on their explicit request.",
    ),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    json_out: bool = typer.Option(False, "--json"),
):
    """Close a note out. Exactly one action flag.

    ``--apply`` is the CLI equivalent of an ``accept`` row: one ``SectionEdit``,
    the note's proposal with the section's **current** text as ``base``, through
    ``apply_edits``. It sets ``applied``, collapses the proposal and keeps the
    comment and ``base_section`` so the thread still reads.

    **``--force`` is required when the section changed since the proposal was
    written, and printing both texts before doing it is the point of the flag.**
    In the tab the same rule is the *"this section changed"* banner and a second
    click. This is §2.6's **cross-batch** half and it is untouched by the tray's
    two-rows-one-section collapse, which is a different mechanism entirely.

    **``--apply``/``--reject`` are the OWNER's alone** (M2 AC-2.1) — they are
    decisions about the document, not reports; the addressee may ``--answered``
    / ``--dismiss`` its own note either way.

    **``--as-user``** is the owner's assistant acting for the owner, in the
    owner's own DM, on their explicit request. It carries more weight than it
    used to: with ``plan approve`` gone, ``--apply`` on the **go-note** is how a
    plan is accepted, so this flag is the whole of the CLI-only user's route to
    approving their own project. ``plan list-notes <project> --to user`` prints
    the id.
    """
    chosen = [n for n, on in
              (("apply", apply), ("reject", reject), ("answered", answered), ("dismiss", dismiss))
              if on]
    if len(chosen) != 1:
        raise _fail("pass exactly one of --apply / --reject / --answered / --dismiss.")
    action = chosen[0]
    if action == "reject" and not reason:
        raise _fail("--reject needs --reason.")
    if force and action != "apply":
        raise _fail("--force only applies to --apply.")

    headers = _headers(token)
    with _http(server, as_user=as_user) as client:
        pid = _pid(client, token, project)
        resp = client.post(
            _url(pid, f"/notes/{note_id}/resolve"),
            json={"action": action, "reason": reason, "force": force},
            headers=headers,
        )
        if resp.status_code == 409 and action == "apply" and not force:
            detail = _detail(resp)
            typer.echo(
                f"`{note_id}` was written against an older version of its section.",
                err=True,
            )
            if isinstance(detail, dict) and detail.get("stale"):
                row = detail["stale"][0]
                typer.echo("\n  What it says now:", err=True)
                for line in (row.get("current") or "").splitlines():
                    typer.echo(f"    {line}", err=True)
                # The proposal is NOT on the refusal. `StaleSection.text` is
                # deliberately not a wire member (§6) — it exists so
                # `file_conflict_note` can carry the attempted text — so the
                # second half of "print both texts" comes from the note itself.
                # One extra read, on the error path only.
                proposed = _proposal_of(client, headers, pid, note_id)
                if proposed:
                    typer.echo("\n  What the proposal would put there:", err=True)
                    for line in proposed.splitlines():
                        typer.echo(f"    {line}", err=True)
            typer.echo(
                f"\n  Apply it anyway: clawmeets plan resolve {project} {note_id} "
                f"--apply --force",
                err=True,
            )
            raise typer.Exit(1)
        result = _ok(resp)

    if json_out:
        _echo_json(result)
        return
    if as_user:
        typer.echo("Acting as the project owner, not as this agent.")
    moved = ", ".join(result.get("sections") or [])
    typer.echo(
        f"{note_id} → {action}" + (f" · wrote {moved} · revision {result.get('revision')}"
                                   if moved else "")
    )


# ---------------------------------------------------------------------------
# list-notes / conflicts
# ---------------------------------------------------------------------------


def _note_row(note: dict) -> str:
    marks = "".join(
        m for m, on in (
            ("!", note.get("conflict")),
            ("~", note.get("section_changed")),
            ("?", note.get("section_missing")),
            (">", note.get("sent")),
        ) if on
    )
    return (
        f"{note['id']:<10} {note['status']:<10} {note['kind']:<10} "
        f"{(note['section'] or '—'):<20} {(note['by'] or '—'):<24} "
        f"→ {(note['to'] or '(nobody)'):<24} {marks:<4} "
        f"{(note.get('comment') or '').splitlines()[0][:48] if note.get('comment') else ''}"
    )


@app.command("list-notes", help="List notes, filtered on any documented axis. --thread reads a thread.")
def list_notes(
    project: str = PROJECT_ARG,
    status: str = typer.Option("", "--status"),
    section: str = typer.Option("", "--section"),
    to: str = typer.Option("", "--to"),
    by: str = typer.Option("", "--by"),
    round_id: str = typer.Option("", "--round"),
    kind: str = typer.Option("", "--kind", help="note|proposal|reply|deviation (all DERIVED)"),
    ac: str = typer.Option("", "--ac", help="Notes filed against one criterion, e.g. AC-2.3."),
    dangling: bool = typer.Option(False, "--dangling", help="Its section no longer exists."),
    stale: bool = typer.Option(False, "--stale", help="Its section changed since it was written."),
    conflicted: bool = typer.Option(False, "--conflicted", help="A refused write left it behind."),
    thread: str = typer.Option("", "--thread", help="A parent and every reply, in time order."),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    json_out: bool = typer.Option(False, "--json"),
):
    """Read-only, always current, and filterable on every documented axis.

    ``--thread`` prints a parent and every reply in time order, because **an
    agent cannot resolve a note whose id it has no way to learn.**
    """
    headers = _headers(token)
    with _http(server) as client:
        pid = _pid(client, token, project)
        notes = _index(client, headers, pid).get("notes", [])

    if thread:
        rows = _thread(notes, thread)
    else:
        rows = [n for n in notes if _matches(
            n, status=status, section=section, to=to, by=by, round_id=round_id,
            kind=kind, ac=ac, dangling=dangling, stale=stale, conflicted=conflicted,
        )]

    if json_out:
        _echo_json(rows)
        return
    if not rows:
        typer.echo("(no notes match)")
        return
    for row in rows:
        typer.echo(_note_row(row))


def _matches(note: dict, **f) -> bool:
    if f["status"] and note["status"] != f["status"]:
        return False
    if f["section"] and note["section"] != f["section"]:
        return False
    if f["to"] and note["to"] != f["to"]:
        return False
    if f["by"] and note["by"] != f["by"]:
        return False
    if f["round_id"] and note.get("round") != f["round_id"]:
        return False
    if f["kind"] and note["kind"] != f["kind"]:
        return False
    # Matched against the QUOTE, because that is where the criterion id lives.
    # `--ac` files an ordinary quoted note and stores no id of its own (there is
    # no `PlanNote.ac`), which is what keeps every other predicate here working
    # untouched — and what makes a note anchored to a renumbered criterion
    # re-anchor by text instead of dangling on a dead label.
    #
    # Delegated rather than written as `ac in quote`: that substring test is
    # true of `AC-2.1` inside `AC-2.10`, so it would quietly fold ten other
    # criteria into one filter. Where a criterion label ends is grammar, and it
    # is answered in `plan_markdown` for the same reason nothing here parses a
    # heading.
    if f["ac"] and not quote_names_criterion(note.get("quote") or "", f["ac"]):
        return False
    if f["dangling"] and not note.get("section_missing"):
        return False
    if f["stale"] and not note.get("section_changed"):
        return False
    if f["conflicted"] and not note.get("conflict"):
        return False
    return True


def _thread(notes: list[dict], parent_id: str) -> list[dict]:
    """Parent first, then every reply in time order.

    Replies are matched on ``reply_to`` and sorted on ``at`` — the same ordering
    §5.7's review message and the tab use, so a thread reads the same in all
    three places (AC-5.9).
    """
    parent = next((n for n in notes if n["id"] == parent_id), None)
    if parent is None:
        raise _fail(f"note {parent_id!r} is not on this plan.")
    replies = sorted(
        (n for n in notes if n.get("reply_to") == parent_id), key=lambda n: n.get("at", "")
    )
    return [parent, *replies]


@app.command("conflicts", help="Blocked writes, beside what each section says now, with the fix.")
def conflicts(
    project: str = PROJECT_ARG,
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    json_out: bool = typer.Option(False, "--json"),
):
    """Every blocked write, beside what its section says now, with the exact
    command that closes it.

    A refused write is never a dead end (§4.3): it left a ``conflict=true`` note
    carrying what the writer wanted beside what it read. This command is the
    other half of that promise — **a list with no command to act on is a list
    nobody clears.**
    """
    headers = _headers(token)
    with _http(server) as client:
        pid = _pid(client, token, project)
        index = _index(client, headers, pid)
    body = index.get("body", "")
    rows = [n for n in index.get("notes", [])
            if n.get("conflict") and n["status"] == "open"]

    if json_out:
        _echo_json(rows)
        return
    if not rows:
        typer.echo("No blocked writes.")
        return
    for note in rows:
        current = _section_text(body, note["section"])
        typer.echo(f"\n{note['id']}  section `{note['section']}`  by {note['by']}")
        typer.echo(f"  {note.get('comment', '')}")
        typer.echo("  What it says now:")
        for line in (current or "(the section no longer exists)").splitlines():
            typer.echo(f"    {line}")
        if current is None:
            typer.echo(f"  Close it:  clawmeets plan resolve {project} {note['id']} --dismiss")
        else:
            typer.echo(
                f"  Take it:   clawmeets plan resolve {project} {note['id']} --apply --force\n"
                f"  Drop it:   clawmeets plan resolve {project} {note['id']} --dismiss"
            )


# ---------------------------------------------------------------------------
# consult — the coordinator's door to the specialists in `shared-context`
# ---------------------------------------------------------------------------


@app.command(
    "consult",
    help="Sync the specialist roster into `shared-context` so you can reach them there.",
)
def consult(
    project: str = PROJECT_ARG,
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    json_out: bool = typer.Option(False, "--json"),
):
    """Seat the specialist roster in ``shared-context``, and print who is in it.

    **A specialist is reached in `shared-context`, never by a plan note.**
    ``shared-context`` is the project's one consultation room and it is there
    from the moment the project is created — but seeded with the coordinator
    alone. During ``spec-ing`` a coordinator may not invite anyone into it: the
    pre-approval gate refuses every ``create_room`` action, and the auto-add
    that fills the room lives inside that same refused handler. So before the
    user's first review round the coordinator had a room with nobody in it.
    This command fills it.

    It does not post anything. Once the roster is in, you consult in the
    ordinary way, with **one** ``reply`` action into the room that
    ``@``-mentions every agent you want an answer from: one message is one
    batch, and one batch is one wake-up when they have all answered. A second
    addressed message into the same room collides on the batch key and is
    dropped with a warning.

    Idempotent **per participant**: run it as often as you like. An unchanged
    roster appends nothing; an agent registered since the last run is seated by
    the next one.
    """
    headers = _headers(token)
    with _http(server) as client:
        pid = _pid(client, token, project)
        out = _ok(client.post(_url(pid, "/review-room"), headers=headers))

    if json_out:
        _echo_json(out)
        return
    room = out["room"]
    members = out.get("participants", [])
    # `created` is the server's "the changelog moved" flag — a room creation on
    # a legacy call, a membership append on this one. Reported as what it now
    # means; the key keeps its name so an older runner wheel still reads it.
    typer.echo(f"`{room}` {'roster updated' if out.get('created') else 'roster already current'}")
    typer.echo(f"  in the room: {', '.join(members) if members else '(nobody)'}")
    typer.echo(
        f"\n  Consult with ONE reply into `{room}`, @-mentioning every agent you\n"
        f"  need an answer from. One message is one batch; a second addressed\n"
        f"  message into the same room is swallowed."
    )


# ---------------------------------------------------------------------------
# Phase 5 — the review batch: the room, the template, the ledger
# ---------------------------------------------------------------------------


def _as_notes(rows: list[dict]) -> list[PlanNote]:
    """Wire rows back into ``PlanNote``.

    ``PlanNoteView`` is ``PlanNote`` plus five derived members; dropping the
    extras is what lets the CLI hand the **same** objects to the **same**
    renderer the server uses, so ``--dry-run`` and the real send cannot diverge.
    """
    fields = set(PlanNote.model_fields)
    return [PlanNote(**{k: v for k, v in row.items() if k in fields}) for row in rows]


def _as_rounds(rows: list[dict]) -> list[PlanReviewRound]:
    return [PlanReviewRound(**row) for row in rows]


def _batch_set(
    notes: list[PlanNote],
    rounds: list[PlanReviewRound],
    *,
    from_notes: tuple[str, ...],
    only: tuple[str, ...],
    sections: tuple[str, ...],
) -> list[PlanNote]:
    """Which notes this batch carries (§5.7).

    ``--from-notes`` is the terminal equivalent of a tray and is taken exactly
    as given — including a note already out, which the **server** refuses with
    the round it went in. Otherwise the default set is
    :func:`pending_notes`, narrowed by ``--only`` / ``--section``. The
    filtering is here because the flags are here; the *definition* of "not
    already out" is the model's, called rather than restated.
    """
    by_id = {n.id: n for n in notes}
    if from_notes:
        missing = [nid for nid in from_notes if nid not in by_id]
        if missing:
            raise _fail(f"unknown note id(s): {', '.join(missing)}")
        chosen = [by_id[nid] for nid in from_notes]
    else:
        chosen = pending_notes(notes, rounds)
    if only:
        chosen = [n for n in chosen if n.to in only]
    if sections:
        chosen = [n for n in chosen if n.section in sections]
    return chosen


def _group(notes: list[PlanNote], keeper_name: str) -> dict[str, list[PlanNote]]:
    """One message per addressee — a note sent by id goes to whoever it is
    **addressed to**, which is what makes a ``to=user`` note reach the user."""
    out: dict[str, list[PlanNote]] = {}
    for note in notes:
        out.setdefault(note.to or keeper_name, []).append(note)
    return out


@app.command("review", help="Send a review batch. Coordinator or owner only.")
def review(
    project: str = PROJECT_ARG,
    room: str = typer.Option("", "--room", help="Pin the room; skips resolution."),
    from_notes: list[str] = typer.Option(
        None, "--from-notes", help="Send exactly these note ids. Repeatable."
    ),
    only: list[str] = typer.Option(None, "--only", help="Only notes addressed to this agent."),
    section: list[str] = typer.Option(None, "--section", help="Only notes on this section."),
    resend: bool = typer.Option(False, "--resend", help="Send again even if the digest has not moved."),
    batch_comment: str = typer.Option("", "--batch-comment", help="A line above the notes."),
    max_notes: int = typer.Option(10, "--max-notes", help="Abort BEFORE sending or creating."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Render the messages and write nothing."),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    json_out: bool = typer.Option(False, "--json"),
):
    """Send the batch. **Coordinator or owner only.**

    Room resolution, and each branch is a decision (§5.7):

    * every addressee ∈ {the coordinator, ``user``} → ``user-communication``,
      and **no room is created, ever**.
    * anyone else → ``shared-context``, the project's one consultation room,
      reused by every later batch.

    The common case creates nothing: you are reviewing a document the
    coordinator keeps, so **your review is the reply**. When ``shared-context``
    needs its roster, that roster is appended **through the runloop,
    in-process** — the fix for **B9**, which is why an owner holding a user JWT
    can send a batch that needs one.

    **Batches may overlap.** The send ledger is per *note*, not per round, so a
    second batch carries only what the first did not, in the same room, while
    the first keeps running. Editing a note's text moves its digest and re-sends
    **that note only**. The one refusal is naming a note already out.

    ``--max-notes`` aborts **before** sending or creating anything — a partial
    send with a room already created is the state that cannot be undone.
    """
    headers = _headers(token)
    with _http(server) as client:
        pid = _pid(client, token, project)
        index = _index(client, headers, pid)
        rounds = _as_rounds(_ok(client.get(_url(pid, "/rounds"), headers=headers)))
        notes = _as_notes(index.get("notes", []))
        keeper_name = index.get("keeper", "")

        chosen = _batch_set(
            notes, rounds,
            from_notes=tuple(from_notes or []),
            only=tuple(only or []),
            sections=tuple(section or []),
        )
        if not chosen:
            typer.echo("Nothing to send — no open note is waiting on an addressee.")
            return
        if max_notes and len(chosen) > max_notes:
            raise _fail(
                f"{len(chosen)} notes exceeds --max-notes {max_notes}; nothing was "
                f"sent and no room was created. Narrow it with --only/--section, "
                f"or raise --max-notes."
            )

        grouped = _group(chosen, keeper_name)
        resolved_room = room or review_room_for(grouped, keeper_name)

        if dry_run:
            # The header names the project the way the server will name it —
            # `display_name` first — so the preview is the message, not an
            # approximation of it.
            record = _ok(client.get(f"/projects/{pid}", headers=headers))
            _dry_run(
                index, notes, grouped, resolved_room, batch_comment,
                round_no=len(rounds) + 1,
                ref=record.get("name") or project,
                title=record.get("display_name") or record.get("name") or project,
            )
            return

        resp = client.post(
            _url(pid, "/review"),
            json={
                "note_ids": [n.id for n in chosen],
                "room": resolved_room,
                "resend": resend,
                "batch_comment": batch_comment,
            },
            headers=headers,
        )
        if resp.status_code == 403:
            # A 403 that does not say what to do instead is how an agent retries
            # forever — the same rule `plan update` follows. The coordinator fans
            # out work; a worker's channel is the note it already wrote.
            typer.echo(
                f"Error: workers do not send review batches; the coordinator or "
                f"the owner does.\n"
                f"  Your notes are already filed and will ride the next batch:\n"
                f"    clawmeets plan list-notes {project} --by <you>",
                err=True,
            )
            raise typer.Exit(1)
        result = _ok(resp)

    if json_out:
        _echo_json(result)
        return
    for name in result.get("unresolved", []):
        typer.echo(
            f"Warning: @{name} resolves to nobody in `{result.get('room')}` — "
            f"skipped; every other note still sent.",
            err=True,
        )
    sent = result.get("sends", {})
    typer.echo(
        f"{result['round_id']} → {result.get('room') or '(no room needed)'}"
        + (" (room opened / roster seated)" if result.get("room_created") else "")
    )
    for addressee in result.get("addressees", []):
        ids = [nid for nid, s in sent.items() if s.get("to") == addressee]
        typer.echo(f"  @{addressee}: {len(ids)} note(s) — {', '.join(sorted(ids))}")
    if result.get("revision"):
        typer.echo(f"  revision {result['revision']}")


def _dry_run(
    index: dict,
    notes: list[PlanNote],
    grouped: dict[str, list[PlanNote]],
    room: str,
    batch_comment: str,
    *,
    round_no: int,
    ref: str,
    title: str,
) -> None:
    """Render the template complete and **write nothing** — no bytes, no room,
    no send-ledger row.

    Rendered by the **same function the server calls**, with the same data. A
    dry run that used a second renderer would be a preview of a message nobody
    ever receives.
    """
    history = [PlanHistoryEntry(**row) for row in index.get("history", [])]
    typer.echo(f"# would post into `{room or '(no room needed)'}`\n")
    for i, (addressee, batch) in enumerate(sorted(grouped.items())):
        if i:
            typer.echo("\n" + "=" * 72 + "\n")
        typer.echo(render_batch_message(
            addressee, batch, batch_comment,
            body=index.get("body", ""),
            project_ref=ref,
            project_title=title,
            keeper_name=index.get("keeper", ""),
            round_no=round_no,
            revision=index.get("revision", 1),
            all_notes=notes,
            history=history,
        ))


@app.command("review-status", help="Per addressee: sent / replied / resolved, and the round's state.")
def review_status(
    project: str = PROJECT_ARG,
    round_id: str = typer.Option("", "--round", help="One round, e.g. r-9f2a."),
    close: bool = typer.Option(False, "--close", help="Force --round closed."),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    json_out: bool = typer.Option(False, "--json"),
):
    """Per addressee: ``sent`` / ``replied`` / ``resolved``, with note ids.

    Feedback is defined **mechanically**, which is what makes the loop legible
    without anyone reporting status: **replied** once a note exists with
    ``reply_to=<id>`` created after the round opened; **resolved** once the
    status leaves ``open``; a round **auto-closes** when every note in it is
    non-open. So: notes accumulate → you triage and submit one review → agents
    reply and propose → ``review-status`` shows the round dry → the next review
    goes out.
    """
    if close and not round_id:
        raise _fail("--close needs --round; closing every open round at once is not a thing.")

    headers = _headers(token)
    with _http(server) as client:
        pid = _pid(client, token, project)
        if close:
            _ok(client.post(_url(pid, f"/rounds/{round_id}/close"), headers=headers))
        rounds = _as_rounds(_ok(client.get(_url(pid, "/rounds"), headers=headers)))
        notes = _as_notes(_index(client, headers, pid).get("notes", []))

    if round_id:
        rounds = [r for r in rounds if r.round_id == round_id]
        if not rounds:
            raise _fail(f"round {round_id!r} is not on this plan.")

    report = [
        {
            "round_id": r.round_id,
            "room": r.room,
            "opened_by": r.opened_by,
            "opened_at": r.opened_at,
            "closed_at": r.closed_at,
            "unresolved": r.unresolved,
            "notes": [
                {"id": nid, "to": send.to, "state": _feedback(notes, nid, r.opened_at)}
                for nid, send in sorted(r.sends.items())
            ],
        }
        for r in rounds
    ]
    if json_out:
        _echo_json(report)
        return
    if not report:
        typer.echo("No review rounds yet.")
        return
    for row in report:
        state = "closed" if row["closed_at"] else "open"
        typer.echo(f"\n{row['round_id']}  {state}  room `{row['room'] or '—'}`  "
                   f"by {row['opened_by']}  {row['opened_at'][:16]}")
        for name in row["unresolved"]:
            typer.echo(f"  ! @{name} resolved to nobody and was skipped")
        for note in row["notes"]:
            typer.echo(f"  {note['id']:<10} → @{note['to']:<24} {note['state']}")


def _feedback(notes: list[PlanNote], note_id: str, opened_at: str) -> str:
    """§5.8's three states, in precedence order.

    ``resolved`` outranks ``replied``: a note that was answered and then closed
    is closed, and reporting it as *replied* would leave it looking like it is
    still someone's turn.
    """
    note = next((n for n in notes if n.id == note_id), None)
    if note is None:
        return "sent"
    if note.status != "open":
        return "resolved"
    if any(n.reply_to == note_id and n.at > opened_at for n in notes):
        return "replied"
    return "sent"
