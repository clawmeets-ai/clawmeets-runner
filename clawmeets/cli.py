"""
Runner-only CLI entry point for clawmeets-runner package.

Provides only the runner-side commands: agent, user, dm.
Server-side commands (server, admin, project, chatroom, message, file, generate)
are available in the full clawmeets package.
"""
from __future__ import annotations

import typer

from clawmeets.cli_runner import agent_app, user_app, dm_app

app = typer.Typer(
    name="clawmeets-runner",
    help="Agent runner for clawmeets multi-agent collaboration.",
    no_args_is_help=True,
)

app.add_typer(agent_app, name="agent")
app.add_typer(user_app,  name="user")
app.add_typer(dm_app,    name="dm")


def main():
    app()


if __name__ == "__main__":
    main()
