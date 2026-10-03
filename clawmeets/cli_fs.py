# SPDX-License-Identifier: BUSL-1.1
"""
clawmeets/cli_fs.py

``clawmeets fs --agent <name> <op> <path>`` — an agent's home folder (its
``AGENT_DIR``) from the command line. Paired skill: ``fs``.

Every op goes through the server's ``/agents/{id}/fs/*`` endpoints
(``server/routes/agent_fs.py``), which relay it to the agent's own runner, so
the access rules are the server's and the runner's, never this module's: the
same answer the UI gets, from whichever machine the command runs on.

Ops: ``ls`` (list a folder), ``cat`` (print a text file), ``write`` (create or
replace one file), ``mkdir``, ``rm`` (file, link, or folder with its contents),
``download`` (exact bytes to a local file). Paths are home-relative; a leading
``/`` is the home root, so ``clawmeets fs --agent XYZ ls /`` lists XYZ's home.

Who the caller is:

  - Inside an agent runtime (any of ``$CLAWMEETS_AGENT_ID``,
    ``$CLAWMEETS_AGENT_TOKEN``, ``$CLAWMEETS_AGENT_DIR`` injected by the
    runner): the agent's own bearer with ``X-Agent-ID``. If the id or token is
    missing (e.g. a sandbox stripped ``*TOKEN*`` variables) the command fails
    rather than fall back to anything else. On its
    own folder that is ``self``; the owner's assistant reaching another of the
    owner's agents is ``assistant``; any other agent is refused by the server.
    ``--agent`` defaults to the agent itself.
    Inside a runtime the command never falls back to a user login, and
    ``--token`` and a ``--server`` other than ``$CLAWMEETS_SERVER_URL`` are
    refused: a worker running on its owner's machine must not
    pick up the owner's saved session and with it every peer's folder.
  - Otherwise (a human at a terminal): ``--token``, then
    ``$CLAWMEETS_USER_TOKEN``, then the saved login — the owner.

``--agent`` is resolved among the caller's OWNER's own agents only (id, full
name, or the short name after ``{username}-``), never among other accounts'
public agents, before the server authorizes the op.

Exit codes: 0 ok; 1 refused or failed; 3 result unknown (a write, mkdir or rm
got no answer — timeout, dropped connection, or a 5xx without the server's error body — but may
have happened; a fresh listing of the parent follows);
4 the agent is offline or its runner is too old.
"""
from __future__ import annotations

import json
import os
import posixpath
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NoReturn, Optional

import httpx
import typer

from clawmeets.api.fs_protocol import FS_MAX_CONTENT_BYTES, FS_TRANSFER_TIMEOUT_SECONDS
from clawmeets.cli_runner import DEFAULT_DATA_DIR, DEFAULT_SERVER, _resolve_user_session, _server_url

app = typer.Typer(
    name="fs",
    help="Browse and manage an agent's home folder (relayed to its runner). Paired skill: fs.",
    no_args_is_help=True,
)

EXIT_RESULT_UNKNOWN = 3
EXIT_OFFLINE = 4
_OFFLINE_CODES = frozenset({"agent_offline", "runner_too_old"})
_MUTATING_METHODS = frozenset({"PUT", "POST", "DELETE"})


def _unsafe_char(c: str) -> bool:
    return unicodedata.category(c) in ("Cc", "Cf", "Zl", "Zp")


# Above the server's longest relay wait, so the server answers a slow transfer
# (504 result_unknown / timeout) before this client gives up on it.
_HTTP_TIMEOUT = httpx.Timeout(FS_TRANSFER_TIMEOUT_SECONDS + 30, connect=15)
# The listing shown after a result-unknown answer: a small op, so it must not
# add another full transfer wait on top of the one that just ran out.
_REFRESH_TIMEOUT = httpx.Timeout(45, connect=15)


@dataclass
class _Options:
    agent: Optional[str]
    token: Optional[str]
    server: Optional[str]
    data_dir: Path
    json: bool


@dataclass
class _Target:
    client: httpx.Client
    headers: dict[str, str]
    agent_id: str
    agent_name: str
    json: bool
    method: str = "GET"

    def call(self, method: str, op_path: str, path: str, **kw) -> httpx.Response:
        url = f"/agents/{self.agent_id}/fs/{op_path}"
        self.method = method
        try:
            return self.client.request(method, url, params={"path": path}, headers=self.headers, **kw)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            # The request never left this machine, so nothing happened.
            _fail("unreachable", f"cannot reach the server: {exc}")
        except httpx.HTTPError as exc:
            # Sent, then the answer was lost (timeout, reset, disconnect): a
            # change may have landed.
            if method in _MUTATING_METHODS:
                why = ("the server did not answer in time" if isinstance(exc, httpx.TimeoutException)
                       else f"the connection was lost before an answer: {exc}")
                _result_unknown(self, path, why)
            if isinstance(exc, httpx.TimeoutException):
                _fail("timeout", "the server did not answer in time")
            _fail("unreachable", f"cannot reach the server: {exc}")


def _fail(code: str, message: str, exit_code: int = 1) -> NoReturn:
    typer.echo(f"Error: {message} ({code})", err=True)
    raise typer.Exit(exit_code)


def _error_of(resp: httpx.Response) -> tuple[Optional[str], str, bool]:
    """``(code, message, result_unknown)`` from the server's error envelope;
    ``code`` is None when the body is not that envelope (e.g. a proxy's page)."""
    try:
        err = resp.json().get("error") or {}
    except (ValueError, AttributeError):
        err = {}
    if not isinstance(err, dict):
        err = {}
    code = str(err["code"]) if err.get("code") else None
    message = str(err.get("message") or resp.text[:500] or f"HTTP {resp.status_code}")
    return code, message, bool(err.get("result_unknown"))


def _parent(path: str) -> str:
    norm = path.replace("\\", "/").rstrip("/")
    return posixpath.dirname(norm) or "/"


def _result_unknown(t: _Target, path: str, why: str) -> NoReturn:
    """A change that may or may not have happened: say so, then show what is
    there now rather than guessing."""
    typer.echo(
        f"Result unknown: {why}. The change to {path!r} may or may not have happened; "
        "do not retry blindly. Current contents of the parent folder:",
        err=True,
    )
    parent = _parent(path)
    try:
        resp = t.client.get(f"/agents/{t.agent_id}/fs/list", params={"path": parent},
                            headers=t.headers, timeout=_REFRESH_TIMEOUT)
    except httpx.HTTPError as exc:
        typer.echo(f"(could not refresh {parent!r}: {exc})", err=True)
        raise typer.Exit(EXIT_RESULT_UNKNOWN)
    if resp.status_code == 200:
        _print_listing(resp.json(), as_json=t.json)
    else:
        code, message, _ = _error_of(resp)
        typer.echo(f"(could not refresh {parent!r}: {message} ({code or f'http_{resp.status_code}'}))", err=True)
    raise typer.Exit(EXIT_RESULT_UNKNOWN)


def _check(t: _Target, resp: httpx.Response, path: str) -> httpx.Response:
    if resp.status_code < 400:
        return resp
    code, message, unknown = _error_of(resp)
    if unknown:
        _result_unknown(t, path, message)
    if code is None and t.method in _MUTATING_METHODS and resp.status_code >= 500:
        _result_unknown(t, path, f"HTTP {resp.status_code} without the server's answer")
    code = code or f"http_{resp.status_code}"
    if code == "agent_offline":
        _fail(code, f"agent {t.agent_name} is offline (its runner is not connected): {message}", EXIT_OFFLINE)
    if code == "runner_too_old":
        _fail(code, f"agent {t.agent_name}'s runner is too old for home-folder access; "
                    f"upgrade and restart it: {message}", EXIT_OFFLINE)
    _fail(code, message)


# ---------------------------------------------------------------------------
# Who is asking, and which agent
# ---------------------------------------------------------------------------


def _owned_agents(client: httpx.Client, headers: dict[str, str], self_id: Optional[str]) -> list[dict]:
    """The caller's owner's own agents (``GET /agents`` also returns other
    accounts' public agents; those are dropped)."""
    try:
        resp = client.get("/agents", headers=headers)
        if self_id:
            owner = None
            agents = resp.json() if resp.status_code == 200 else []
            for a in agents:
                if isinstance(a, dict) and a.get("id") == self_id:
                    owner = a.get("registered_by")
        else:
            me = client.get("/auth/user/me", headers=headers)
            owner = me.json().get("id") if me.status_code == 200 else None
            agents = resp.json() if resp.status_code == 200 else []
    except (httpx.HTTPError, ValueError) as exc:
        _fail("unreachable", f"cannot list your agents: {exc}")
    if not owner:
        _fail("unauthenticated", "could not resolve who you are; log in or check the token")
    return [a for a in agents if isinstance(a, dict) and a.get("registered_by") == owner]


def resolve_agent(agents: list[dict], ref: str) -> dict:
    """``ref`` (id, full name, or short name) among ``agents``, or exit 1.
    Exact matches win over case-insensitive ones; more than one match at the
    same tier is an error rather than a guess."""
    tiers = (
        lambda a: a.get("id") == ref,
        lambda a: a.get("name") == ref,
        lambda a: str(a.get("name", "")).endswith(f"-{ref}"),
        lambda a: str(a.get("name", "")).lower() == ref.lower(),
        lambda a: str(a.get("name", "")).lower().endswith(f"-{ref.lower()}"),
    )
    for match in tiers:
        found = [a for a in agents if match(a)]
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            names = ", ".join(sorted(str(a.get("name")) for a in found))
            _fail("ambiguous_agent", f"{ref!r} matches several of your agents: {names}")
    _fail("agent_not_found", f"none of your agents is named {ref!r}")


def _open_client(server: str) -> httpx.Client:
    return httpx.Client(base_url=server, timeout=_HTTP_TIMEOUT)


def _target(ctx: typer.Context) -> _Target:
    o: _Options = ctx.obj
    env_id = os.environ.get("CLAWMEETS_AGENT_ID", "").strip()
    env_token = os.environ.get("CLAWMEETS_AGENT_TOKEN", "").strip()
    env_dir = os.environ.get("CLAWMEETS_AGENT_DIR", "").strip()
    if env_id or env_token or env_dir:
        # Inside an agent runtime: this agent or nothing. A half-set env must
        # fail closed, never fall through to the owner's saved login below.
        if not (env_id and env_token):
            _fail("unauthenticated", "running inside an agent but its credentials "
                                     "($CLAWMEETS_AGENT_ID / $CLAWMEETS_AGENT_TOKEN) are not in the environment")
        if o.token:
            _fail("invalid_request", "inside an agent runtime `clawmeets fs` acts as this agent; "
                                     "--token is not accepted")
        server = _server_url(os.environ.get("CLAWMEETS_SERVER_URL") or DEFAULT_SERVER)
        if o.server and _server_url(o.server) != server:
            _fail("invalid_request", "inside an agent runtime `clawmeets fs` talks only to "
                                     "this agent's own server; --server is not accepted")
        headers = {"Authorization": f"Bearer {env_token}", "X-Agent-ID": env_id}
        self_id: Optional[str] = env_id
    else:
        server, token = _resolve_user_session(o.data_dir, o.token, o.server)
        headers = {"Authorization": f"Bearer {token}"}
        self_id = None
    client = _open_client(server)
    ctx.call_on_close(client.close)

    ref = o.agent
    if not ref or ref == "self":
        if self_id is None:
            _fail("invalid_request", "pass --agent <name> (only an agent can mean itself)")
        ref = self_id
    agent = resolve_agent(_owned_agents(client, headers, self_id), ref)
    return _Target(client, headers, str(agent["id"]), str(agent.get("name") or agent["id"]), o.json)


@app.callback()
def main(
    ctx: typer.Context,
    agent: Optional[str] = typer.Option(
        None, "--agent", "-a",
        help="Whose home folder: id, full or short name, among your owner's agents "
             "(default: this agent, inside an agent runtime).",
    ),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="User JWT (acts as the owner). Not accepted inside an agent runtime."),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
    as_json: bool = typer.Option(False, "--json", help="Print the server's JSON instead of text."),
):
    """Browse and manage an agent's home folder. Paths are home-relative; `/` is the home root."""
    ctx.obj = _Options(agent, token, server, Path(data_dir), as_json)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _echo_json(payload) -> None:
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


def _human_size(n: int) -> str:
    for unit in ("B", "K", "M", "G"):
        if n < 1024 or unit == "G":
            return f"{n}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return str(n)  # pragma: no cover


def _marks(e: dict) -> str:
    marks = []
    if e.get("type") == "symlink":
        marks.append("link: listed only, delete-only")
    elif not e.get("readable", True):
        marks.append("no access")
    if not e.get("writable", True) and e.get("type") != "symlink":
        marks.append("read-only")
    if e.get("tier") == "secret":
        marks.append("secret")
    return f"  [{', '.join(marks)}]" if marks else ""


def _print_listing(body: dict, *, as_json: bool) -> None:
    if as_json:
        _echo_json(body)
        return
    head = f"{body.get('path') or '/'}"
    if not body.get("writable", True):
        head += "  [read-only]"
    typer.echo(head)
    kinds = {"dir": "d", "file": "-", "symlink": "l", "other": "?"}
    for e in body.get("entries", []):
        when = datetime.fromtimestamp(e.get("mtime") or 0).strftime("%Y-%m-%d %H:%M")
        size = "" if e.get("type") == "dir" else _human_size(int(e.get("size") or 0))
        # Names are agent-chosen: escape control, format (bidi) and line/paragraph
        # separators so one cannot forge rows, a terminal escape or a reordered name.
        raw = str(e.get("name", ""))
        name = raw.encode("unicode_escape").decode("ascii") if any(_unsafe_char(c) for c in raw) else raw
        name += "/" if e.get("type") == "dir" else ""
        typer.echo(f"{kinds.get(e.get('type'), '?')} {size:>7}  {when}  {name}{_marks(e)}")
    if body.get("truncated"):
        typer.echo(f"(truncated: showing the first {len(body.get('entries', []))} entries)")


# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------


@app.command("ls")
def ls(ctx: typer.Context, path: str = typer.Argument("/", help="Folder, home-relative.")):
    """List one folder. Entries you cannot change are marked \\[read-only]."""
    t = _target(ctx)
    _print_listing(_check(t, t.call("GET", "list", path), path).json(), as_json=t.json)


@app.command("cat")
def cat(ctx: typer.Context, path: str = typer.Argument(..., help="File, home-relative.")):
    """Print a text file. A binary file is refused here; use `download`."""
    t = _target(ctx)
    body = _check(t, t.call("GET", "file", path), path).json()
    if t.json:
        _echo_json(body)
        return
    if not body.get("is_text"):
        _fail("not_text", f"{path!r} is not UTF-8 text ({body.get('size')} bytes); "
                          f"use `clawmeets fs download {path}`")
    typer.echo(body.get("text") or "", nl=False)


@app.command("write")
def write(
    ctx: typer.Context,
    path: str = typer.Argument(..., help="File to create or replace, home-relative."),
    from_file: Optional[Path] = typer.Option(None, "--from", "-f", help="Upload this local file's bytes."),
    text: Optional[str] = typer.Option(None, "--text", help="Write this text (UTF-8)."),
    stdin: bool = typer.Option(False, "--stdin", help="Read the content from stdin (same as --from -)."),
):
    """Create or replace one file from --from, --text, or --stdin."""
    if from_file is not None and str(from_file) == "-":
        from_file, stdin = None, True
    if sum((from_file is not None, text is not None, stdin)) > 1:
        _fail("invalid_request", "pass only one of --from, --text, --stdin")
    if from_file is not None:
        if not from_file.is_file():
            _fail("invalid_request", f"{str(from_file)!r} is not a local file")
        if from_file.stat().st_size > FS_MAX_CONTENT_BYTES:
            _fail("too_large", f"files larger than {FS_MAX_CONTENT_BYTES} bytes cannot be uploaded")
        data = from_file.read_bytes()
    elif text is not None:
        data = text.encode("utf-8")
    elif stdin:
        data = sys.stdin.buffer.read(FS_MAX_CONTENT_BYTES + 1)
    else:
        # Never read stdin implicitly: in a tool shell it is already at EOF, so
        # a forgotten flag would silently empty the target file.
        _fail("invalid_request", "give the content with --from FILE, --text TEXT, or --stdin")
    if len(data) > FS_MAX_CONTENT_BYTES:
        _fail("too_large", f"files larger than {FS_MAX_CONTENT_BYTES} bytes cannot be uploaded")
    t = _target(ctx)
    body = _check(t, t.call("PUT", "file", path, content=data), path).json()
    if t.json:
        _echo_json(body)
    else:
        typer.echo(f"wrote {len(data)} bytes to {path}")


@app.command("mkdir")
def mkdir(ctx: typer.Context, path: str = typer.Argument(..., help="Folder to make; its parent must exist.")):
    """Make one folder."""
    t = _target(ctx)
    body = _check(t, t.call("POST", "folder", path), path).json()
    if t.json:
        _echo_json(body)
    else:
        typer.echo(f"made {path}")


@app.command("rm")
def rm(ctx: typer.Context, path: str = typer.Argument(..., help="File, link, or folder (with its contents).")):
    """Delete a file, a link, or a folder and everything in it. No prompt."""
    t = _target(ctx)
    body = _check(t, t.call("DELETE", "entry", path), path).json()
    if t.json:
        _echo_json(body)
    else:
        typer.echo(f"deleted {path}")


@app.command("download")
def download(
    ctx: typer.Context,
    path: str = typer.Argument(..., help="File, home-relative."),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Local file to write (`-` for stdout). Default: the file's name in the current folder."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing local file."),
):
    """Save one file's exact bytes locally (works for binary files)."""
    dest: Optional[Path] = None
    if str(out) != "-":
        dest = out or Path(path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or "download")
        if dest.is_dir():
            _fail("invalid_request", f"{str(dest)!r} is a folder; pass a file path with --out")
        if dest.exists() and not force:
            _fail("exists", f"{str(dest)!r} already exists; pass --force to overwrite")
    t = _target(ctx)
    data = _check(t, t.call("GET", "download", path), path).content
    if dest is None:
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
        return
    # A fresh temp file (never a pre-existing path that could be a link),
    # removed if the final rename does not happen.
    fd, tmp = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".part", dir=dest.parent or ".")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    if t.json:
        _echo_json({"path": path, "saved_to": str(dest), "bytes": len(data)})
    else:
        typer.echo(f"saved {len(data)} bytes to {dest}")
