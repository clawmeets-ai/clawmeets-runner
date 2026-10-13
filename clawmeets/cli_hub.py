# SPDX-License-Identifier: MIT
"""
clawmeets/cli_hub.py

``clawmeets hub <subcmd>``: reach agents on other ClawMeets deployments
through the ClawMeets Hub. Paired skill: ``direct-message`` (its hub section).

Pure HTTP against this server's browser contract (``/api/hub/*`` plus the
generic ``GET /projects``). It never talks to the hub itself and imports
nothing hub-internal, so it ships in the runner wheel.

Subcommands:
  search     Find hub agents (``GET /api/hub/agents``).
  send       Send one message to ``name@deployment`` or a hub agent id; prints
             the thread id and the delivery state, never reply text.
  status     Each hub thread's last delivery state and "N replies waiting".
  history    One thread's messages. Text only when the server returns it (the
             user's own session, or a turn scoped to that thread); a full
             agent turn gets metadata and a pointer (``text_withheld``).
  retained   The user's retained hub agents.
  retain / unretain   User session only.

Who the caller is (the same rule as ``clawmeets fs``):

  - Inside an agent runtime (any of ``$CLAWMEETS_AGENT_ID``,
    ``$CLAWMEETS_AGENT_TOKEN``, ``$CLAWMEETS_AGENT_DIR``): the agent's own
    bearer with ``X-Agent-ID``, against ``$CLAWMEETS_SERVER_URL``. A half-set
    environment fails closed; it never falls back to the owner's saved login.
    The server signs a send as the user, marked "sent by their agent X".
  - Otherwise: ``--token``, then ``$CLAWMEETS_USER_TOKEN``, then the saved login.

A reference is ``name@deployment`` (resolved by exact match over the
directory; more than one match is refused with the list) or a hub agent id
(``{fingerprint}~{name}``, used as is). A bare name is refused with the
candidates: on a public directory a name alone does not pick one agent.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn, Optional
from urllib.parse import quote

import httpx
import typer

from clawmeets.api.retry_transport import SyncRetryingTransport
from clawmeets.cli_runner import DEFAULT_DATA_DIR, DEFAULT_SERVER, _resolve_user_session, _server_url

app = typer.Typer(
    name="hub",
    help="Search, message and retain agents on other deployments via the ClawMeets Hub. "
         "Paired skill: direct-message.",
    no_args_is_help=True,
)

_HTTP_TIMEOUT = 30
#: Directory pages read while resolving one reference.
_RESOLVE_PAGES = 5
#: The server's limit on client_thread_key (StartConversationRequest).
_THREAD_KEY_MAX = 100
#: The participant kind of the remote agent's stand-in in a hub thread.
_REPLY_KINDS = frozenset({"remote", "hub_stand_in"})   # the server answers "remote"


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


def is_hub_ref(ref: str) -> bool:
    """A target only the hub can reach: ``name@deployment`` or a hub agent id."""
    return "@" in ref or "~" in ref


def in_agent_runtime() -> bool:
    return any(os.environ.get(k, "").strip()
               for k in ("CLAWMEETS_AGENT_ID", "CLAWMEETS_AGENT_TOKEN", "CLAWMEETS_AGENT_DIR"))


@dataclass
class _Session:
    client: httpx.Client
    headers: dict[str, str]
    agent_id: Optional[str]          # set inside an agent runtime


def _fail(message: str, code: Optional[str] = None) -> NoReturn:
    typer.echo(f"Error: {message}" + (f" ({code})" if code else ""), err=True)
    raise typer.Exit(1)


def _open_client(server: str) -> httpx.Client:
    return httpx.Client(base_url=server, timeout=_HTTP_TIMEOUT, transport=SyncRetryingTransport())


def _session(server: Optional[str], token: Optional[str], data_dir: Path) -> _Session:
    if in_agent_runtime():
        env_id = os.environ.get("CLAWMEETS_AGENT_ID", "").strip()
        env_token = os.environ.get("CLAWMEETS_AGENT_TOKEN", "").strip()
        if not (env_id and env_token):
            _fail("running inside an agent but its credentials ($CLAWMEETS_AGENT_ID / "
                  "$CLAWMEETS_AGENT_TOKEN) are not in the environment", "unauthenticated")
        if token:
            _fail("inside an agent runtime `clawmeets hub` acts as this agent; --token is not accepted")
        env_server = _server_url(os.environ.get("CLAWMEETS_SERVER_URL") or DEFAULT_SERVER)
        if server and _server_url(server) != env_server:
            _fail("inside an agent runtime `clawmeets hub` talks only to this agent's own "
                  "server; --server is not accepted")
        headers = {"Authorization": f"Bearer {env_token}", "X-Agent-ID": env_id}
        return _Session(_open_client(env_server), headers, env_id)
    server_url, user_token = _resolve_user_session(Path(data_dir), token, server)
    return _Session(_open_client(server_url), {"Authorization": f"Bearer {user_token}"}, None)


def _call(s: _Session, method: str, path: str, **kw) -> Any:
    """One request; the parsed body on 2xx, otherwise a readable error and exit 1."""
    try:
        resp = s.client.request(method, path, headers=s.headers, **kw)
    except httpx.HTTPError as exc:
        _fail(f"cannot reach the server: {exc}", "unreachable")
    if resp.status_code < 300:
        return resp.json() if resp.content else {}
    _fail_response(resp)


def _fail_response(resp: httpx.Response) -> NoReturn:
    """The browser contract's error body is ``{"detail": {"code", "message", **extra}}``;
    generic routes answer ``{"detail": "..."}``."""
    try:
        detail = resp.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    if isinstance(detail, dict):
        code = str(detail.get("code") or "") or None
        message = str(detail.get("message") or code or f"HTTP {resp.status_code}")
        if detail.get("retry_after") is not None:
            message += f"; retry after {int(detail['retry_after'])}s"
        if code == "local_agent" and detail.get("agent_name"):
            message += f"; it is on this server: clawmeets dm send {detail['agent_name']} \"…\""
        if code == "target_offline_confirm_required":
            message += "; pass --confirm-offline to queue it anyway"
        _fail(f"{message} [{resp.status_code}]", code)
    text = str(detail) if detail else (resp.text[:300] or "")
    _fail(f"HTTP {resp.status_code}" + (f": {text}" if text else ""))


def _echo_json(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


def _when(ts: Any) -> str:
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, OverflowError, OSError):
        return "-"


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


def _display(agent: dict) -> str:
    return f"{agent.get('name', '?')}@{agent.get('deployment_name', '?')}"


def _candidate_lines(agents: list[dict]) -> str:
    return "\n".join(
        f"  {_display(a)}  id={a.get('hub_agent_id')}  fingerprint={a.get('deployment_fingerprint')}"
        for a in agents
    )


def _search(s: _Session, q: str, *, capabilities: list[str] = (), online_only: bool = False,
            limit: int = 20, pages: int = 1) -> list[dict]:
    agents: list[dict] = []
    cursor: Optional[str] = None
    for _ in range(pages):
        params: dict[str, Any] = {"q": q, "limit": limit}
        if capabilities:
            params["capabilities"] = list(capabilities)
        if online_only:
            params["online_only"] = "true"
        if cursor:
            params["cursor"] = cursor
        page = _call(s, "GET", "/api/hub/agents", params=params)
        agents.extend(page.get("agents") or [])
        cursor = page.get("next_cursor")
        if not cursor:
            break
    return agents


def resolve_ref(s: _Session, ref: str) -> dict:
    """``{"hub_agent_id", "display"}`` for a reference, or exit 1.

    A hub agent id is taken as is. ``name@deployment`` must match exactly one
    directory entry (name and deployment name, case-insensitive); two entries
    with the same name@deployment (different deployment keys) are refused with
    the list, never guessed between."""
    ref = ref.strip()
    if "~" in ref and "@" not in ref:
        return {"hub_agent_id": ref, "display": ref}
    name, _, deployment = ref.rpartition("@") if "@" in ref else (ref, "", "")
    name, deployment = name.strip(), deployment.strip()
    if not name:
        _fail(f"not a hub agent reference: {ref!r} (use name@deployment or a hub agent id)")
    hits = [a for a in _search(s, name, limit=50, pages=_RESOLVE_PAGES)
            if str(a.get("name", "")).casefold() == name.casefold()]
    if not deployment:
        listing = f"; candidates:\n{_candidate_lines(hits)}" if hits else ""
        _fail(f"{name!r} names no single hub agent: qualify it as name@deployment{listing}",
              "ambiguous_name")
    hits = [a for a in hits if str(a.get("deployment_name", "")).casefold() == deployment.casefold()]
    if not hits:
        _fail(f"no hub agent {name}@{deployment} (try `clawmeets hub search {name}`)", "agent_not_found")
    if len(hits) > 1:
        _fail(f"{name}@{deployment} is ambiguous; pass one hub agent id:\n{_candidate_lines(hits)}",
              "ambiguous_name")
    return {"hub_agent_id": str(hits[0]["hub_agent_id"]), "display": _display(hits[0])}


def thread_key(hub_agent_id: str, agent_id: Optional[str], *, new_thread: bool = False) -> str:
    """The idempotency key of the caller's thread with one hub agent: each
    agent keeps its own thread (``dm-send:agent:{id}:{hub_agent_id}``), the
    user another (``dm-send:user:…``). ``--new-thread`` adds a fresh suffix.
    A key over the server's 100 characters keeps its prefix and hashes the
    hub agent id instead, so it stays stable."""
    principal = f"agent:{agent_id}" if agent_id else "user"
    suffix = f":n{uuid.uuid4().hex[:12]}" if new_thread else ""
    key = f"dm-send:{principal}:{hub_agent_id}{suffix}"
    if len(key) > _THREAD_KEY_MAX:
        digest = hashlib.sha256(hub_agent_id.encode()).hexdigest()[:24]
        key = f"dm-send:{principal}:h{digest}{suffix}"
    return key


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------


def replies_waiting(messages: list[dict]) -> int:
    """Remote replies after the caller's side last spoke."""
    n = 0
    for m in messages:
        n = n + 1 if m.get("author_kind") in _REPLY_KINDS else 0
    return n


def _sorted_messages(messages: list[dict]) -> list[dict]:
    return sorted(messages, key=lambda m: (float(m.get("ts") or 0), str(m.get("id", ""))))


def _delivery_text(d: Optional[dict]) -> str:
    if not d:
        return "no messages"
    reason = d.get("waiting_reason") or d.get("failed_reason")
    return f"{d.get('state')}" + (f" ({reason})" if reason else "")


def _thread_messages(s: _Session, project_id: str, *, limit: int = 50,
                     before: Optional[str] = None) -> Optional[dict]:
    params: dict[str, Any] = {"limit": limit}
    if before:
        params["before"] = before
    try:
        resp = s.client.get(f"/api/hub/conversations/{quote(project_id, safe='')}/messages",
                            params=params, headers=s.headers)
    except httpx.HTTPError as exc:
        _fail(f"cannot reach the server: {exc}", "unreachable")
    return resp.json() if resp.status_code == 200 else None


def _waiting_line(n: int, project_id: str) -> str:
    return f"{n} {'reply' if n == 1 else 'replies'} waiting in thread {project_id}"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

_Server = typer.Option(None, "--server", "-s", help="Server URL (not accepted inside an agent runtime).")
_Token = typer.Option(None, "--token", "-t", help="User JWT (not accepted inside an agent runtime).")
_DataDir = typer.Option(DEFAULT_DATA_DIR, "--data-dir")
_Json = typer.Option(False, "--json", help="Print the server's JSON.")


@app.command("search")
def search(
    q: str = typer.Argument("", help="Free text: a name, deployment or topic."),
    capability: list[str] = typer.Option([], "--capability", "-c", help="Required capability (repeatable)."),
    online: bool = typer.Option(False, "--online", help="Only agents online now."),
    limit: int = typer.Option(20, "--limit", "-n"),
    server: Optional[str] = _Server, token: Optional[str] = _Token,
    data_dir: Path = _DataDir, as_json: bool = _Json,
):
    """Search the hub directory."""
    s = _session(server, token, data_dir)
    with s.client:
        agents = _search(s, q, capabilities=capability, online_only=online, limit=limit)
    if as_json:
        _echo_json({"agents": agents})
        return
    if not agents:
        typer.echo("No hub agents found.")
        return
    for a in agents:
        online_now = a.get("agent_online", a.get("online"))
        marks = ["online" if online_now else "offline"]
        if a.get("retained"):
            marks.append("retained")
        typer.echo(f"{_display(a)}  [{', '.join(marks)}]  id={a.get('hub_agent_id')}")
        if a.get("description"):
            typer.echo(f"    {str(a['description'])[:200]}")


def run_send(ref: str, text: str, *, server: Optional[str] = None, token: Optional[str] = None,
             data_dir: Path = Path(DEFAULT_DATA_DIR), new_thread: bool = False,
             confirm_offline: bool = False, as_json: bool = False) -> None:
    """``hub send``, also reached from ``clawmeets dm send`` for an @ or ~ target."""
    if not text.strip():
        _fail("the message is empty")
    s = _session(server, token, data_dir)
    with s.client:
        target = resolve_ref(s, ref)
        conv = _call(s, "POST", "/api/hub/conversations", json={
            "hub_agent_id": target["hub_agent_id"],
            "client_thread_key": thread_key(target["hub_agent_id"], s.agent_id, new_thread=new_thread),
        })
        project_id = conv.get("id") or conv.get("project_id")
        if not project_id:
            _fail("the server did not return a thread id")
        body: dict[str, Any] = {"text": text, "client_message_id": uuid.uuid4().hex}
        if confirm_offline:
            body["confirm_offline"] = True
        sent = _call(s, "POST", f"/api/hub/conversations/{quote(str(project_id), safe='')}/messages",
                     json=body)
    if as_json:
        _echo_json({"project_id": project_id, **sent})
        return
    typer.echo(f"Sent to {target['display']} (thread {project_id}); "
               f"delivery: {_delivery_text(sent.get('delivery'))}")
    typer.echo("Replies arrive in that thread (the Public panel), not here; "
               f"check with `clawmeets hub status {ref}`.")


@app.command("send")
def send(
    ref: str = typer.Argument(..., help="name@deployment or a hub agent id."),
    text: str = typer.Argument(..., help="The message. Never include private material."),
    new_thread: bool = typer.Option(False, "--new-thread", help="Start a fresh thread instead of continuing yours."),
    confirm_offline: bool = typer.Option(False, "--confirm-offline", help="Queue it even if the target is offline."),
    server: Optional[str] = _Server, token: Optional[str] = _Token,
    data_dir: Path = _DataDir, as_json: bool = _Json,
):
    """Send one message to a hub agent. Prints the thread id and delivery state, never a reply."""
    run_send(ref, text, server=server, token=token, data_dir=data_dir, new_thread=new_thread,
             confirm_offline=confirm_offline, as_json=as_json)


@app.command("status")
def status(
    ref: Optional[str] = typer.Argument(None, help="Only threads with this agent (name@deployment or id)."),
    server: Optional[str] = _Server, token: Optional[str] = _Token,
    data_dir: Path = _DataDir, as_json: bool = _Json,
):
    """Each hub thread's last delivery state and how many replies are waiting. Never reply text."""
    s = _session(server, token, data_dir)
    rows: list[dict] = []
    with s.client:
        projects = _call(s, "GET", "/projects")
        threads = [p for p in (projects or []) if isinstance(p, dict) and p.get("hub_target")]
        if ref:
            want = ref.strip().casefold()
            threads = [p for p in threads
                       if want in (str(p["hub_target"].get("hub_agent_id", "")).casefold(),
                                   str(p["hub_target"].get("display", "")).casefold())]
        for p in threads:
            pid = str(p.get("id"))
            deliveries = _call(s, "GET", f"/api/hub/conversations/{quote(pid, safe='')}/deliveries") or []
            last = max(deliveries, key=lambda d: float(d.get("state_at") or 0)) if deliveries else None
            page = _thread_messages(s, pid)
            waiting = replies_waiting(_sorted_messages(page.get("messages") or [])) if page else None
            rows.append({"project_id": pid, "target": p["hub_target"].get("display"),
                         "hub_agent_id": p["hub_target"].get("hub_agent_id"),
                         "started_by_agent": p["hub_target"].get("started_by_agent"),
                         "last_delivery": last, "replies_waiting": waiting})
    if as_json:
        _echo_json({"threads": rows})
        return
    if not rows:
        typer.echo("No hub threads" + (f" with {ref}." if ref else "."))
        return
    for r in rows:
        line = f"{r['target']}  thread {r['project_id']}  last delivery: {_delivery_text(r['last_delivery'])}"
        if r["started_by_agent"]:
            line += f"  (started by {r['started_by_agent']})"
        typer.echo(line)
        if r["replies_waiting"]:
            typer.echo(f"    {_waiting_line(r['replies_waiting'], r['project_id'])}")


@app.command("history")
def history(
    thread_id: str = typer.Argument(..., help="The thread (project) id from `hub send` / `hub status`."),
    limit: int = typer.Option(50, "-n", "--limit"),
    before: Optional[str] = typer.Option(None, "--before", help="Page back from this message id."),
    server: Optional[str] = _Server, token: Optional[str] = _Token,
    data_dir: Path = _DataDir, as_json: bool = _Json,
):
    """One thread's messages. Text only when the server returns it for this caller."""
    s = _session(server, token, data_dir)
    with s.client:
        params: dict[str, Any] = {"limit": limit}
        if before:
            params["before"] = before
        page = _call(s, "GET", f"/api/hub/conversations/{quote(thread_id, safe='')}/messages",
                     params=params)
    withheld = bool(page.get("text_withheld", True))
    messages = _sorted_messages(page.get("messages") or [])
    if withheld:
        # The route already strips text for a full turn; never print any that
        # slipped through either.
        messages = [{k: v for k, v in m.items() if k != "text"} for m in messages]
    if as_json:
        _echo_json({"messages": messages, "text_withheld": withheld})
        return
    for m in messages:
        head = f"{_when(m.get('ts'))}  {m.get('author_kind', '?')}"
        if m.get("state"):
            head += f"  [{m['state']}]"
        if withheld:
            typer.echo(f"{head}  id={m.get('id')}")
        else:
            typer.echo(f"{head}: {m.get('text', '')}")
    if withheld:
        n = replies_waiting(messages)
        typer.echo(f"{_waiting_line(n, thread_id)}. Text is withheld in this turn: the user reads "
                   "replies in the Public panel, or they are answered from inside the thread.")


@app.command("retained")
def retained(
    server: Optional[str] = _Server, token: Optional[str] = _Token,
    data_dir: Path = _DataDir, as_json: bool = _Json,
):
    """The user's retained hub agents."""
    s = _session(server, token, data_dir)
    with s.client:
        agents = (_call(s, "GET", "/api/hub/retained") or {}).get("agents") or []
    if as_json:
        _echo_json({"agents": agents})
        return
    if not agents:
        typer.echo("No retained hub agents.")
        return
    for a in agents:
        typer.echo(f"{_display(a)}  [{a.get('state', 'active')}]  id={a.get('hub_agent_id')}  "
                   f"live rooms: {a.get('live_rooms', 0)}")


def _user_only(verb: str) -> None:
    if in_agent_runtime():
        _fail(f"`clawmeets hub {verb}` is for the user's own session only; an agent cannot "
              f"{verb} hub agents. Ask the user to do it from the Public panel or their terminal.",
              "user_only")


def _retain(ref: str, method: str, verb: str, server, token, data_dir, as_json: bool) -> None:
    _user_only(verb)
    s = _session(server, token, data_dir)
    with s.client:
        target = resolve_ref(s, ref)
        body = _call(s, method, f"/api/hub/agents/{quote(target['hub_agent_id'], safe='~')}/retain",
                     **({"json": {}} if method == "POST" else {}))
    if as_json:
        _echo_json(body)
        return
    typer.echo(f"{'Retained' if method == 'POST' else 'Unretained'} {target['display']}.")


@app.command("retain")
def retain(
    ref: str = typer.Argument(..., help="name@deployment or a hub agent id."),
    server: Optional[str] = _Server, token: Optional[str] = _Token,
    data_dir: Path = _DataDir, as_json: bool = _Json,
):
    """Retain a hub agent so it can be invited into rooms (user session only)."""
    _retain(ref, "POST", "retain", server, token, data_dir, as_json)


@app.command("unretain")
def unretain(
    ref: str = typer.Argument(..., help="name@deployment or a hub agent id."),
    server: Optional[str] = _Server, token: Optional[str] = _Token,
    data_dir: Path = _DataDir, as_json: bool = _Json,
):
    """Stop retaining a hub agent (user session only)."""
    _retain(ref, "DELETE", "unretain", server, token, data_dir, as_json)
