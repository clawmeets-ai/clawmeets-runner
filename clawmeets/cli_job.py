# SPDX-License-Identifier: MIT
"""Runner-safe CLI for the managed background-job lifecycle."""
from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import typer

from clawmeets.api.managed_job_client import JobApiError, ManagedJobClient

app = typer.Typer(help="Register and update durable background jobs.", no_args_is_help=True)

_DURATION = re.compile(r"^\s*(\d+)\s*([smhd]?)\s*$")
_UNIT_SECONDS = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_duration(text: str) -> int:
    """'90m', '6h', '3600s', '2d' or bare seconds -> seconds."""
    match = _DURATION.match(text.lower())
    if not match:
        raise typer.BadParameter(f"{text!r} is not a duration like 90m, 6h or 3600s")
    return int(match.group(1)) * _UNIT_SECONDS[match.group(2)]


def _fail(message: str) -> None:
    typer.echo(message if message.startswith("Error") else f"Error: {message}", err=True)
    raise typer.Exit(1)


def _client() -> ManagedJobClient:
    try:
        return ManagedJobClient.from_env()
    except JobApiError as exc:
        _fail(exc.detail)


def _emit(row: dict) -> None:
    typer.echo(json.dumps(row, indent=2, sort_keys=True))


@app.command("start")
def start(
    project: str = typer.Option(..., "--project"),
    room: str = typer.Option(..., "--room"),
    pending_message: str | None = typer.Option(None, "--pending-message"),
    deadline: str | None = typer.Option(
        None, "--deadline",
        help="Absolute cap, e.g. 90m, 6h, 3600s. Omitted = no absolute deadline "
             "(the heartbeat lease still applies).",
    ),
    deadline_seconds: int | None = typer.Option(None, "--deadline-seconds", min=10, hidden=True),
    heartbeat_timeout: int = typer.Option(300, "--heartbeat-timeout", min=10),
    step: str | None = typer.Option(None, "--step"),
) -> None:
    """Register long-running work before the current invocation exits."""
    seconds = parse_duration(deadline) if deadline else deadline_seconds
    if seconds is not None and seconds < 10:
        raise typer.BadParameter("the deadline must be at least 10 seconds", param_hint="--deadline")
    with _client() as client:
        try:
            job = client.start(
                project_id=project,
                chatroom_name=room,
                pending_message_id=pending_message,
                deadline_at=(
                    (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat()
                    if seconds is not None else None
                ),
                heartbeat_timeout_seconds=heartbeat_timeout,
                step=step,
            )
        except (JobApiError, httpx.HTTPError) as exc:
            _fail(str(exc))
    _emit(job)
    if seconds is None:
        typer.echo(
            f"Note: no absolute deadline; the job stalls after {heartbeat_timeout}s "
            f"without a heartbeat.",
            err=True,
        )


@app.command("inspect")
def inspect(job_id: str) -> None:
    with _client() as client:
        try:
            _emit(client.inspect(job_id))
        except (JobApiError, httpx.HTTPError) as exc:
            _fail(str(exc))


def _event_command(operation: str):
    def command(
        job_id: str,
        sequence: int = typer.Option(..., "--sequence", min=0),
        summary: str | None = typer.Option(None, "--summary"),
        step: str | None = typer.Option(None, "--step"),
        checkpoint: str | None = typer.Option(None, "--checkpoint"),
        reason: str | None = typer.Option(None, "--reason"),
    ) -> None:
        _event(operation, job_id, sequence, summary=summary, step=step,
               checkpoint=checkpoint, reason=reason)
    command.__name__ = operation
    return command


def _closing_command(operation: str):
    def command(
        job_id: str,
        sequence: int = typer.Option(..., "--sequence", min=0),
        summary: str | None = typer.Option(None, "--summary"),
        step: str | None = typer.Option(None, "--step"),
        checkpoint: str | None = typer.Option(None, "--checkpoint"),
        reason: str | None = typer.Option(None, "--reason"),
        no_wake: bool = typer.Option(
            False, "--no-wake",
            help="Do not wake this agent with the closing line (use when closing "
                 "the job yourself, inside your own turn).",
        ),
    ) -> None:
        _event(operation, job_id, sequence, summary=summary, step=step,
               checkpoint=checkpoint, reason=reason, wake_owner=not no_wake)
    command.__name__ = operation
    return command


def _event(operation: str, job_id: str, sequence: int, **fields) -> None:
    with _client() as client:
        try:
            # A stale sequence is not retried here: an agent typing a command
            # should learn its number was wrong, not have it silently fixed.
            _emit(client.event(job_id, operation, expected_sequence=sequence,
                               retry_stale=False, **fields))
        except (JobApiError, httpx.HTTPError) as exc:
            _fail(str(exc))


for _operation in ("heartbeat", "progress", "cancel"):
    app.command(_operation)(_event_command(_operation))
for _operation in ("complete", "fail"):
    app.command(_operation)(_closing_command(_operation))


@app.command("run")
def run(
    job_id: str = typer.Argument(..., help="Id from `clawmeets job start`."),
    command: list[str] = typer.Argument(..., help="The command, after --."),
    heartbeat_every: float | None = typer.Option(
        None, "--heartbeat-every", help="Seconds between heartbeats (default: heartbeat timeout / 3)."),
    on_close: str = typer.Option(
        "stop", "--on-close",
        help="When the server closes the job (stall, deadline, cancel): 'stop' the command or 'keep' it running."),
    grace: float = typer.Option(10, "--grace", help="Seconds between SIGTERM and SIGKILL."),
    detach: bool = typer.Option(False, "--detach", help="Run in the background and return at once."),
    log: Path | None = typer.Option(None, "--log", help="Append the command's output here."),
) -> None:
    """Run COMMAND under a job: heartbeats, progress lines, and complete/fail on exit.

    Lines the command prints starting with `CLAWMEETS_JOB_PROGRESS ` are posted
    as progress. Exit 0 completes the job; any other exit fails it.
    """
    from clawmeets.runner.job_supervisor import spawn_detached, supervise

    if on_close not in ("stop", "keep"):
        raise typer.BadParameter("must be 'stop' or 'keep'", param_hint="--on-close")
    if detach:
        log_path = log or Path.cwd() / f".clawmeets-job-{job_id[:8]}.log"
        _emit(spawn_detached(job_id, command, log_path=log_path,
                             heartbeat_every=heartbeat_every, on_close=on_close, grace=grace))
        return
    try:
        code = supervise(job_id, command, heartbeat_every=heartbeat_every,
                         on_close=on_close, grace=grace, log_path=log)
    except (JobApiError, httpx.HTTPError) as exc:
        _fail(str(exc))
    raise typer.Exit(code)
