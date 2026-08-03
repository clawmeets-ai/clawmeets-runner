# SPDX-License-Identifier: MIT
"""
clawmeets/cli_sop.py

``clawmeets sop <subcmd>`` — CLI for the owner's My Desk SOP library, the
stored, reusable prompts they hand an agent over and over.

Paired with ``skills/desk-sop/SKILL.md``. Unlike ``clawmeets todo``, this group
is **assistant-only** by design: the SOP library is a curated personal surface,
so ``/me/desk/sops`` accepts the owner's browser JWT or the owner's own
``{username}-assistant`` bearer and nothing else. Every verb here 401s for any
other agent — including ``list``. That is the correct failure, and ``_ok``
surfaces the server's message verbatim rather than masking it.

Auth resolved from env (standard agent-runtime injection, same as
``clawmeets todo`` / ``clawmeets brief``):

  - ``CLAWMEETS_SERVER_URL``  — server base URL
  - ``CLAWMEETS_AGENT_ID``    — UUID of the calling agent
  - ``CLAWMEETS_AGENT_TOKEN`` — agent bearer token

Subcommands:
  list      Show the whole library.
  show      One SOP, plus its parsed blanks and resolved recipient.
  create    Store a new SOP.
  update    Partially edit one.
  delete    Remove one.
  trigger   Fill the blanks from --set pairs and DM the result to its agent.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import httpx
import typer

from clawmeets.cli_runner import resolve_dm_recipient, send_dm_as_owner
from clawmeets.models.sop_template import fill, parse_blanks, unknown_labels

app = typer.Typer(
    name="sop",
    help="Manage and fire the owner's My Desk SOP library. Paired skill: desk-sop.",
    no_args_is_help=True,
)


def _env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        typer.echo(
            f"Error: ${name} is not set. The sop CLI runs inside an agent "
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


def _owner_token() -> str:
    """The bearer for calls needing the OWNER's authority — see
    ``cli_todo._owner_token``. For the assistant the agent bearer and the
    assistant bearer are the same secret."""
    tok = os.environ.get("CLAWMEETS_ASSISTANT_TOKEN", "").strip()
    return tok or _env("CLAWMEETS_AGENT_TOKEN")


def _echo(payload) -> None:
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


def _find(client: httpx.Client, headers: dict[str, str], sop_id: str) -> dict:
    """The one SOP with this id, or exit 1.

    There is no ``GET /me/desk/sops/{id}`` — the library is one document per
    owner, so the list IS the read path."""
    sops = _ok(client.get("/me/desk/sops", headers=headers))
    if isinstance(sops, list):
        for s in sops:
            if isinstance(s, dict) and s.get("id") == sop_id:
                return s
    typer.echo(f"Error: SOP {sop_id!r} is not in the library.", err=True)
    raise typer.Exit(1)


def _read_body(body: str, body_file: Path | None, *, required: bool) -> str | None:
    """Resolve the SOP body from ``--body`` or ``--body-file``.

    Exactly one, checked before any HTTP call so a mistake costs nothing.
    Returns None when neither was given and none is required (an ``update``
    that leaves the body alone).
    """
    if body and body_file:
        typer.echo("Error: pass either --body or --body-file, not both.", err=True)
        raise typer.Exit(1)
    if body_file:
        return body_file.read_text(encoding="utf-8")
    if body:
        return body
    if required:
        typer.echo("Error: one of --body / --body-file is required.", err=True)
        raise typer.Exit(1)
    return None


def _parse_set(pairs: list[str] | None) -> dict[str, str]:
    """Parse repeated ``--set "Label=value"`` into a dict.

    Splits at the FIRST ``=`` only, so a value may contain more of them (a URL,
    a query). An empty label is an error rather than a silently dropped value.
    Duplicate labels fold last-wins.
    """
    out: dict[str, str] = {}
    for raw in pairs or []:
        if "=" not in raw:
            typer.echo(
                f"Error: --set {raw!r} is not in \"Label=value\" form.", err=True
            )
            raise typer.Exit(1)
        label, value = raw.split("=", 1)
        if not label.strip():
            typer.echo(f"Error: --set {raw!r} has an empty label.", err=True)
            raise typer.Exit(1)
        out[label.strip()] = value
    return out


def _recipient_view(
    client: httpx.Client, headers: dict[str, str], sop: dict
) -> dict | None:
    """Who this SOP dispatches to, and whether they still resolve.

    Answers "who will this go to?" before anything is sent, so the assistant can
    say so in the same breath as asking for the blanks."""
    name = (sop.get("agent_name") or "").strip()
    if not name:
        return None
    resolved = resolve_dm_recipient(
        client, _owner_token(), name, agent_id=headers["X-Agent-ID"]
    )
    return {
        "id": sop.get("agent_id"),
        "name": name,
        "resolved_name": resolved,
        "live": resolved is not None,
    }


@app.command("list")
def list_sops() -> None:
    """List the owner's whole SOP library, newest first."""
    client, headers = _client()
    with client:
        _echo(_ok(client.get("/me/desk/sops", headers=headers)))


@app.command("show")
def show(
    sop_id: str = typer.Argument(..., help="Id of the SOP to inspect."),
) -> None:
    """Print one SOP with everything needed to conduct the interview in one call.

    On top of the stored fields: ``blanks`` from the shared template grammar
    (label, kind, options, default) so the assistant asks precise questions and
    can offer the ``select`` values, and ``recipient`` — the SOP's agent plus
    whether it still resolves in the owner's roster.

    This is step 2 of the two-turn protocol: read the blanks, ask the user, END
    THE TURN. See ``skills/desk-sop/SKILL.md`` §4.
    """
    client, headers = _client()
    with client:
        sop = _find(client, headers, sop_id)
        out = dict(sop)
        out["blanks"] = [b.model_dump() for b in parse_blanks(sop.get("body") or "")]
        out["recipient"] = _recipient_view(client, headers, sop)
        _echo(out)


@app.command("create")
def create(
    title: str = typer.Option(..., "--title", help="Card title as it reads in the rail."),
    body: str = typer.Option("", "--body", help="The prompt template, blanks included."),
    body_file: Path = typer.Option(
        None, "--body-file",
        exists=True, file_okay=True, dir_okay=False, readable=True,
        help="Read the template from a file instead (easier for multi-line bodies).",
    ),
    desc: str = typer.Option("", "--desc", help="Optional one-line description."),
    agent: str = typer.Option(
        "", "--agent",
        help="Default recipient (short or full agent name); resolved server-side.",
    ),
) -> None:
    """Store a new SOP in the owner's library.

    Write blanks as ``{{Label|kind:config}}`` — ``text`` / ``number`` /
    ``select`` / ``date`` / ``agent``, e.g.
    ``{{Inventory|select:Chelsea,Warehouse}}``. See ``clawmeets sop show`` for
    how they come back out.
    """
    resolved_body = _read_body(body, body_file, required=True)
    payload: dict = {"title": title, "body": resolved_body}
    if desc:
        payload["desc"] = desc
    if agent:
        payload["agent_name"] = agent
    client, headers = _client()
    with client:
        _echo(_ok(client.post("/me/desk/sops", json=payload, headers=headers)))


@app.command("update")
def update(
    sop_id: str = typer.Argument(..., help="Id of the SOP to edit."),
    title: str = typer.Option("", "--title", help="New card title."),
    desc: str = typer.Option("", "--desc", help="New description."),
    body: str = typer.Option("", "--body", help="New template body."),
    body_file: Path = typer.Option(
        None, "--body-file",
        exists=True, file_okay=True, dir_okay=False, readable=True,
        help="Read the new template body from a file.",
    ),
    agent: str = typer.Option("", "--agent", help="New default recipient."),
    clear_agent: bool = typer.Option(
        False, "--clear-agent", help="Unassign the recipient entirely."
    ),
) -> None:
    """Partially update one SOP.

    Only the keys you pass are sent, so an omitted flag never clears a stored
    field. ``--clear-agent`` sends an explicit JSON null for the recipient pair
    (the route's three-way key-presence protocol); passing it together with
    ``--agent`` is contradictory and exits 1.
    """
    if agent and clear_agent:
        typer.echo(
            "Error: pass either --agent or --clear-agent, not both.", err=True
        )
        raise typer.Exit(1)
    resolved_body = _read_body(body, body_file, required=False)
    payload: dict = {}
    if title:
        payload["title"] = title
    if desc:
        payload["desc"] = desc
    if resolved_body is not None:
        payload["body"] = resolved_body
    if agent:
        payload["agent_name"] = agent
    if clear_agent:
        payload["agent_id"] = None
        payload["agent_name"] = None
    if not payload:
        typer.echo(
            "Error: nothing to update — pass at least one of --title / --desc / "
            "--body / --body-file / --agent / --clear-agent.",
            err=True,
        )
        raise typer.Exit(1)
    client, headers = _client()
    with client:
        _echo(_ok(client.patch(f"/me/desk/sops/{sop_id}", json=payload, headers=headers)))


@app.command("delete")
def delete(
    sop_id: str = typer.Argument(..., help="Id of the SOP to remove."),
) -> None:
    """Remove an SOP from the library."""
    client, headers = _client()
    with client:
        _echo(_ok(client.delete(f"/me/desk/sops/{sop_id}", headers=headers)))


@app.command("trigger")
def trigger(
    sop_id: str = typer.Argument(..., help="Id of the SOP to fire."),
    set_: list[str] = typer.Option(
        None, "--set", help='One blank as "Label=value" (repeatable).'
    ),
    to: str = typer.Option(
        "", "--to",
        help="Recipient override, or supply one when the SOP has none.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the filled message + recipient; send nothing."
    ),
    allow_unfilled: bool = typer.Option(
        False, "--allow-unfilled",
        help="Send with unanswered blanks left as literal {{…}} text.",
    ),
) -> None:
    """Fill the template from ``--set`` pairs and DM the result to the SOP's agent.

    This verb does NOT ask for values — it cannot. It is one process invocation,
    and the calling agent's turn ends the moment it asks the user something. The
    interview belongs to the skill: ``sop show`` hands it the questions, it asks
    and stops, and the user's reply brings it back to run this. See
    ``skills/desk-sop/SKILL.md`` §4.

    Exits 1 — not a no-op — on an unknown ``--set`` label (typo protection: a
    silently dropped value would send a half-filled template), on a missing
    value (unless ``--allow-unfilled``), and on no resolvable recipient with no
    ``--to``. An SOP is addressed by construction, so an unaddressable one is a
    library problem to report, not a send to guess at.
    """
    values = _parse_set(set_)

    client, headers = _client()
    with client:
        sop = _find(client, headers, sop_id)
        body = sop.get("body") or ""

        stray = unknown_labels(body, values)
        if stray:
            typer.echo(
                f"Error: --set names blanks this SOP does not have: "
                f"{', '.join(sorted(stray))}. Its blanks are: "
                f"{', '.join(b.label for b in parse_blanks(body)) or '(none)'}.",
                err=True,
            )
            raise typer.Exit(1)

        text, missing = fill(body, values)
        if missing and not allow_unfilled:
            typer.echo(
                f"Error: no value given for: {', '.join(missing)}. Ask the user "
                f"for these and re-run with --set, or pass --allow-unfilled to "
                f"send the template text as-is.",
                err=True,
            )
            raise typer.Exit(1)

        # A select value outside its option list is accepted — a set is a
        # shortcut, never a cage (the frontend's rule) — but say so, because it
        # is also what a typo looks like. Labels match the way `fill` matches
        # them, or a lowercased --set would silently skip the warning.
        lowered = {k.strip().lower(): v for k, v in values.items()}
        for blank in parse_blanks(body):
            supplied = lowered.get(blank.label.strip().lower())
            if blank.kind == "select" and blank.options and supplied and supplied not in blank.options:
                typer.echo(
                    f"Note: {blank.label!r} = {supplied!r} is not one of "
                    f"{', '.join(blank.options)} — sending it anyway.",
                    err=True,
                )

        ref = to.strip() or (sop.get("agent_name") or "").strip()
        if not ref:
            typer.echo(
                "Error: this SOP has no recipient. Pass --to <agent> to say who "
                "should get it, or set one with `clawmeets sop update "
                f"{sop_id} --agent <agent>`.",
                err=True,
            )
            raise typer.Exit(1)
        recipient = resolve_dm_recipient(
            client, _owner_token(), ref, agent_id=headers["X-Agent-ID"]
        )
        if recipient is None:
            typer.echo(
                f"Error: {ref!r} does not match exactly one agent on the roster.",
                err=True,
            )
            raise typer.Exit(1)

        if dry_run:
            _echo({
                "sent": False,
                "reason": "dry_run",
                "to": recipient,
                "content": text,
                "unfilled": missing,
            })
            return

        # Keyed on the filled text, so a retried identical send reuses its
        # thread while running the same SOP with different values opens a new
        # one — those are different jobs.
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
        project = send_dm_as_owner(
            client,
            _owner_token(),
            recipient,
            text,
            new_thread=True,
            thread_key=f"sop:{sop_id}:{digest}",
        )
        if project is None:
            typer.echo(
                f"Error: could not open a DM thread with {recipient!r}.", err=True
            )
            raise typer.Exit(1)
        _echo({"sent": True, "to": recipient, "project_id": project.get("id")})


if __name__ == "__main__":
    app()
