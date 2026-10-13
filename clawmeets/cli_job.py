# SPDX-License-Identifier: MIT
"""Runner-safe CLI for the managed background-job lifecycle."""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta

import httpx
import typer

from clawmeets.api.retry_transport import SyncRetryingTransport

app = typer.Typer(help="Register and update durable background jobs.", no_args_is_help=True)


def _env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        typer.echo(f"Error: ${name} is not set", err=True)
        raise typer.Exit(1)
    return value


def _client() -> tuple[httpx.Client, dict[str, str]]:
    return (
        httpx.Client(
            base_url=_env("CLAWMEETS_SERVER_URL").rstrip("/"),
            timeout=30,
            transport=SyncRetryingTransport(),
        ),
        {
            "Authorization": f"Bearer {_env('CLAWMEETS_AGENT_TOKEN')}",
            "X-Agent-ID": _env("CLAWMEETS_AGENT_ID"),
        },
    )


def _ok(response: httpx.Response) -> dict:
    if response.is_success:
        return response.json()
    try:
        detail = response.json().get("detail", response.text)
    except Exception:
        detail = response.text
    typer.echo(f"Error ({response.status_code}): {detail}", err=True)
    raise typer.Exit(1)


def _emit(row: dict) -> None:
    typer.echo(json.dumps(row, indent=2, sort_keys=True))


@app.command("start")
def start(
    project: str = typer.Option(..., "--project"),
    room: str = typer.Option(..., "--room"),
    pending_message: str | None = typer.Option(None, "--pending-message"),
    deadline_seconds: int = typer.Option(..., "--deadline-seconds", min=10),
    heartbeat_timeout: int = typer.Option(300, "--heartbeat-timeout", min=10),
    step: str | None = typer.Option(None, "--step"),
) -> None:
    """Register long-running work before the current invocation exits."""
    deadline = datetime.now(UTC) + timedelta(seconds=deadline_seconds)
    payload = {
        "project_id": project,
        "chatroom_name": room,
        "deadline_at": deadline.isoformat(),
        "heartbeat_timeout_seconds": heartbeat_timeout,
        "step": step,
    }
    if pending_message:
        payload["pending_message_id"] = pending_message
    client, headers = _client()
    with client:
        _emit(_ok(client.post("/jobs", json=payload, headers=headers)))


@app.command("inspect")
def inspect(job_id: str) -> None:
    client, headers = _client()
    with client:
        _emit(_ok(client.get(f"/jobs/{job_id}", headers=headers)))


def _event(
    operation: str,
    job_id: str,
    sequence: int,
    summary: str | None,
    step: str | None,
    checkpoint: str | None,
    reason: str | None,
) -> None:
    client, headers = _client()
    payload = {
        "expected_sequence": sequence,
        "summary": summary,
        "step": step,
        "checkpoint": checkpoint,
        "reason": reason,
    }
    with client:
        _emit(_ok(client.post(f"/jobs/{job_id}/{operation}", json=payload, headers=headers)))


def _event_command(operation: str):
    def command(
        job_id: str,
        sequence: int = typer.Option(..., "--sequence", min=0),
        summary: str | None = typer.Option(None, "--summary"),
        step: str | None = typer.Option(None, "--step"),
        checkpoint: str | None = typer.Option(None, "--checkpoint"),
        reason: str | None = typer.Option(None, "--reason"),
    ) -> None:
        _event(operation, job_id, sequence, summary, step, checkpoint, reason)
    command.__name__ = operation
    return command


for _operation in ("heartbeat", "progress", "complete", "fail", "cancel"):
    app.command(_operation)(_event_command(_operation))
