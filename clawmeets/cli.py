"""
Runner-only CLI entry point for clawmeets package.

Provides only the runner-side commands: agent, user, dm.
Server-side commands (server, admin, project, chatroom, message, file, generate)
are available in the full clawmeets package.
"""
from __future__ import annotations

from typing import Optional

import typer

from clawmeets.cli_runner import agent_app, user_app, dm_app, mcp_app, reflection_app
from clawmeets.cli_init import init_command
from clawmeets.cli_lifecycle import start_command, stop_command, status_command

app = typer.Typer(
    name="clawmeets",
    help="Agent runner for clawmeets multi-agent collaboration.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if not value:
        return
    try:
        from importlib.metadata import version
        v = version("clawmeets")
    except Exception:
        v = "unknown"
    typer.echo(f"clawmeets {v}")
    raise typer.Exit()


@app.callback()
def _root(
    version: Optional[bool] = typer.Option(
        None, "--version", "-V",
        callback=_version_callback, is_eager=True,
        help="Show clawmeets version and exit.",
    ),
) -> None:
    pass

# Top-level commands (setup + lifecycle)
app.command("init")(init_command)
app.command("start")(start_command)
app.command("stop")(stop_command)
app.command("status")(status_command)

app.add_typer(agent_app, name="agent")
app.add_typer(user_app,  name="user")
app.add_typer(dm_app,    name="dm")
app.add_typer(mcp_app,   name="mcp")
app.add_typer(reflection_app, name="reflection")


def main():
    app()


if __name__ == "__main__":
    main()
