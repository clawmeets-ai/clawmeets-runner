# SPDX-License-Identifier: MIT
"""
clawmeets/cli_http_api.py — Generic HTTP-API sync CLI.
"""
from __future__ import annotations

import json

import typer

from clawmeets.integrations.http_api import _client, _lib

app = typer.Typer(
    name="http-api",
    help="Generic HTTP-API sync (REST + JSON/CSV/TSV). Paired skill: http-api.",
    no_args_is_help=True,
)


@app.command()
def sync(
    dwh: str = typer.Option(..., "--dwh"),
    config: str = typer.Option("", "--config"),
    max_runtime: int = typer.Option(1500, "--max-runtime"),
) -> None:
    """Run configured HTTP endpoints into the warehouse per --config."""
    typer.echo(json.dumps(_lib.sync_to_warehouse(
        dwh, config_file=config, max_runtime_seconds=max_runtime,
    ), indent=2, ensure_ascii=False))


# --- Ad-hoc HTTP client (get / post / put / patch / delete) ------------------
# Independent of `sync`. A session jar (--save-session / --session) makes login
# just another request that writes cookies. Redirects are OFF by default so a
# login's 302 auth signal is not masked as a final 200. See skills/http-api.
#
# get() takes no body; post/put/patch/delete share the body-carrying option set
# below. All delegate to _client.run(), whose int return becomes the exit code.


@app.command()
def get(
    url: str = typer.Argument(..., help="Request URL."),
    header: list[str] = typer.Option(
        [], "-H", "--header",
        help="Repeatable 'Name: value'. $VAR / ${VAR} expanded from env.",
    ),
    query: list[str] = typer.Option([], "--query", help="Repeatable 'key=value'."),
    session: str = typer.Option("", "--session", help="Load cookies from a jar (curl -b)."),
    save_session: str = typer.Option(
        "", "--save-session", help="Write captured cookies to a 0600 jar (curl -c).",
    ),
    follow: bool = typer.Option(
        False, "--follow/--no-follow", help="Follow 3xx redirects (default: off).",
    ),
    output: str = typer.Option("body", "--output", help="body | json | full."),
    timeout: float = typer.Option(30.0, "--timeout", help="Per-request timeout (s)."),
    fail: bool = typer.Option(False, "--fail", help="Exit 22 on HTTP status >= 400."),
) -> None:
    """GET a URL, optionally replaying a session jar."""
    raise typer.Exit(_client.run(
        method="GET", url=url, header=header, query=query, data=[], json_body=[],
        session=session, save_session=save_session, follow=follow,
        output=output, timeout=timeout, fail=fail,
    ))


def _run_body_command(
    method: str,
    url: str,
    header: list[str],
    query: list[str],
    data: list[str],
    json_body: list[str],
    session: str,
    save_session: str,
    follow: bool,
    output: str,
    timeout: float,
    fail: bool,
) -> None:
    """Shared body-carrying request path for post/put/patch/delete."""
    raise typer.Exit(_client.run(
        method=method, url=url, header=header, query=query, data=data,
        json_body=json_body, session=session, save_session=save_session,
        follow=follow, output=output, timeout=timeout, fail=fail,
    ))


@app.command()
def post(
    url: str = typer.Argument(..., help="Request URL."),
    header: list[str] = typer.Option([], "-H", "--header"),
    query: list[str] = typer.Option([], "--query"),
    data: list[str] = typer.Option([], "--data"),
    json_body: list[str] = typer.Option([], "--json"),
    session: str = typer.Option("", "--session"),
    save_session: str = typer.Option("", "--save-session"),
    follow: bool = typer.Option(False, "--follow/--no-follow"),
    output: str = typer.Option("body", "--output"),
    timeout: float = typer.Option(30.0, "--timeout"),
    fail: bool = typer.Option(False, "--fail"),
) -> None:
    """POST (form via --data or JSON via --json). Login = POST + --save-session."""
    _run_body_command("POST", url, header, query, data, json_body,
                      session, save_session, follow, output, timeout, fail)


@app.command()
def put(
    url: str = typer.Argument(..., help="Request URL."),
    header: list[str] = typer.Option([], "-H", "--header"),
    query: list[str] = typer.Option([], "--query"),
    data: list[str] = typer.Option([], "--data"),
    json_body: list[str] = typer.Option([], "--json"),
    session: str = typer.Option("", "--session"),
    save_session: str = typer.Option("", "--save-session"),
    follow: bool = typer.Option(False, "--follow/--no-follow"),
    output: str = typer.Option("body", "--output"),
    timeout: float = typer.Option(30.0, "--timeout"),
    fail: bool = typer.Option(False, "--fail"),
) -> None:
    """PUT (form via --data or JSON via --json)."""
    _run_body_command("PUT", url, header, query, data, json_body,
                      session, save_session, follow, output, timeout, fail)


@app.command()
def patch(
    url: str = typer.Argument(..., help="Request URL."),
    header: list[str] = typer.Option([], "-H", "--header"),
    query: list[str] = typer.Option([], "--query"),
    data: list[str] = typer.Option([], "--data"),
    json_body: list[str] = typer.Option([], "--json"),
    session: str = typer.Option("", "--session"),
    save_session: str = typer.Option("", "--save-session"),
    follow: bool = typer.Option(False, "--follow/--no-follow"),
    output: str = typer.Option("body", "--output"),
    timeout: float = typer.Option(30.0, "--timeout"),
    fail: bool = typer.Option(False, "--fail"),
) -> None:
    """PATCH (form via --data or JSON via --json)."""
    _run_body_command("PATCH", url, header, query, data, json_body,
                      session, save_session, follow, output, timeout, fail)


@app.command()
def delete(
    url: str = typer.Argument(..., help="Request URL."),
    header: list[str] = typer.Option([], "-H", "--header"),
    query: list[str] = typer.Option([], "--query"),
    data: list[str] = typer.Option([], "--data"),
    json_body: list[str] = typer.Option([], "--json"),
    session: str = typer.Option("", "--session"),
    save_session: str = typer.Option("", "--save-session"),
    follow: bool = typer.Option(False, "--follow/--no-follow"),
    output: str = typer.Option("body", "--output"),
    timeout: float = typer.Option(30.0, "--timeout"),
    fail: bool = typer.Option(False, "--fail"),
) -> None:
    """DELETE (optionally with a --data / --json body)."""
    _run_body_command("DELETE", url, header, query, data, json_body,
                      session, save_session, follow, output, timeout, fail)
