# SPDX-License-Identifier: MIT
"""
Runner-only CLI entry point for clawmeets package.

Provides every command group a skill can shell: the runner-side groups (agent,
user, dm, mcp, skill, …), the client-side resource groups (project, chatroom,
message, file — pure HTTP, from cli_client), and the integration groups behind
the hub skills (gmail, gcal, gdrive, browser, …).

Only the genuinely server-side groups are absent — `server` and `admin` need
uvicorn + clawmeets.server.app, which the wheel does not carry.

Group names here must match clawmeets/cli.py exactly; a skill shelling
`clawmeets <group>` has no idea which package it landed in.
"""
from __future__ import annotations

from typing import Optional

import typer

from clawmeets.cli_runner import (
    agent_app,
    user_app,
    dm_app,
    mcp_app,
    reflection_app,
    assistant_app,
    agent_team_app,
    team_app,
    knowledge_pack_app,
    bootstrap_app,
)
from clawmeets.cli_lifecycle import start_command, stop_command, status_command
from clawmeets.cli_skill import skill_app, schedule_app
from clawmeets.cli_env import app as env_app
from clawmeets.cli_consult import consult_command
from clawmeets.cli_client import proj_app, room_app, msg_app, file_app

# Integration groups — each is the CLI surface of one hub skill.
from clawmeets.cli_gmail import app as gmail_app
from clawmeets.cli_gcal import app as gcal_app
from clawmeets.cli_gdrive import app as gdrive_app
from clawmeets.cli_gdrive_write import app as gdrive_write_app
from clawmeets.cli_browser import app as browser_app
from clawmeets.cli_caldav import app as caldav_app
from clawmeets.cli_mailbox import app as mailbox_app
from clawmeets.cli_media import app as media_app
from clawmeets.cli_homekit import app as homekit_app
from clawmeets.cli_osxphotos import app as osxphotos_app
from clawmeets.cli_database import app as database_app
from clawmeets.cli_http_api import app as http_api_app
from clawmeets.cli_brief import app as brief_app
from clawmeets.cli_todo import app as todo_app
from clawmeets.cli_dwh import app as dwh_app
from clawmeets.cli_knowledge_dir import app as knowledge_dir_app
from clawmeets.cli_etl import app as etl_app
from clawmeets.cli_website_monitor import app as website_monitor_app
from clawmeets.cli_om import app as om_app
from clawmeets.cli_ib import app as ib_app

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
app.command("start")(start_command)
app.command("stop")(stop_command)
app.command("status")(status_command)
app.command("consult")(consult_command)

app.add_typer(assistant_app, name="assistant")
app.add_typer(agent_team_app, name="agent-team")
app.add_typer(agent_app, name="agent")
app.add_typer(user_app,  name="user")
app.add_typer(dm_app,    name="dm")
app.add_typer(mcp_app,   name="mcp")
app.add_typer(reflection_app, name="reflection")
app.add_typer(team_app,           name="team")
app.add_typer(knowledge_pack_app, name="knowledge-pack")
app.add_typer(bootstrap_app,      name="bootstrap")
app.add_typer(schedule_app,       name="schedule")
app.add_typer(skill_app,          name="skill")
app.add_typer(env_app,            name="env")

# Client-side resource groups (pure HTTP; shelled by the bundled system skills
# propose-project / manage-project-roster / post-chat-message / *-completion-report).
app.add_typer(proj_app, name="project")
app.add_typer(room_app, name="chatroom")
app.add_typer(msg_app,  name="message")
app.add_typer(file_app, name="file")

# Integration groups (one per hub skill).
app.add_typer(gmail_app, name="gmail")
app.add_typer(gcal_app, name="gcal")
app.add_typer(gdrive_app, name="gdrive")
app.add_typer(gdrive_write_app, name="gdrive-write")
app.add_typer(browser_app, name="browser")
app.add_typer(caldav_app, name="caldav")
app.add_typer(mailbox_app, name="mailbox")
app.add_typer(media_app, name="media")
app.add_typer(homekit_app, name="homekit")
app.add_typer(osxphotos_app, name="osxphotos")
app.add_typer(database_app, name="database")
app.add_typer(http_api_app, name="http-api")
app.add_typer(brief_app, name="brief")
app.add_typer(todo_app, name="todo")
app.add_typer(dwh_app, name="dwh")
app.add_typer(knowledge_dir_app, name="knowledge-dir")
app.add_typer(etl_app, name="etl")
app.add_typer(website_monitor_app, name="website-monitor")
app.add_typer(om_app, name="om")
app.add_typer(ib_app, name="ib")


def main():
    app()


if __name__ == "__main__":
    main()
