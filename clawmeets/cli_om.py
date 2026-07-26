# SPDX-License-Identifier: MIT
"""
clawmeets/cli_om.py

``clawmeets om <subcmd>`` — OpenMontage stage-handoff harness.
Paired skills: om-stage (worker) / om-produce (coordinator).

Subcommands:
  stage-begin    Sync the state fork, acquire the single-writer lease, and
                 print the stage briefing (director skill, prior checkpoints,
                 checkpoint path binding) as JSON.
  stage-commit   Validate the checkpoint, make the data-only commit, push,
                 release the lease, and print the handoff receipt as JSON.
  stage-abort    Release the lease and reset the clone (coordinator-directed
                 recovery).
"""
from __future__ import annotations

import json

import typer

from clawmeets.integrations.openmontage import _lib

app = typer.Typer(
    name="om",
    help="OpenMontage pipeline stage handoffs. Paired skills: om-stage / om-produce.",
    no_args_is_help=True,
)


def _emit_json(payload) -> None:
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


def _run(fn, *args, **kwargs) -> None:
    try:
        _emit_json(fn(*args, **kwargs))
    except (RuntimeError, ValueError) as exc:
        _emit_json({"status": "error", "error": str(exc)})
        raise typer.Exit(1) from exc


@app.command("stage-begin")
def stage_begin_cmd(
    project: str = typer.Argument(..., help="Project slug (one video production)."),
    stage: str = typer.Argument(..., help="Pipeline stage name (e.g. script)."),
    pipeline: str = typer.Option(
        ..., "--pipeline", help="Pipeline manifest name (e.g. animated-explainer)."
    ),
    workdir: str = typer.Option(
        _lib.DEFAULT_WORKDIR, "--workdir",
        help="State-fork clone location, relative to the sandbox cwd.",
    ),
    config: str = typer.Option(
        "", "--config",
        help="Config file (default: $CLAWMEETS_AGENT_DIR/skill-hub/configs/om-stage.json).",
    ),
) -> None:
    """Prepare a stage and print the briefing JSON."""
    _run(
        _lib.stage_begin, project, stage, pipeline,
        workdir=workdir, explicit_config=config,
    )


@app.command("stage-commit")
def stage_commit_cmd(
    project: str = typer.Argument(...),
    stage: str = typer.Argument(...),
    workdir: str = typer.Option(_lib.DEFAULT_WORKDIR, "--workdir"),
    config: str = typer.Option("", "--config"),
) -> None:
    """Commit + push the stage's data and print the handoff receipt JSON."""
    _run(
        _lib.stage_commit, project, stage,
        workdir=workdir, explicit_config=config,
    )


@app.command("stage-abort")
def stage_abort_cmd(
    project: str = typer.Argument(...),
    workdir: str = typer.Option(_lib.DEFAULT_WORKDIR, "--workdir"),
    config: str = typer.Option("", "--config"),
    force: bool = typer.Option(
        False, "--force",
        help="Break another agent's lease (coordinator's instruction only).",
    ),
) -> None:
    """Release the lease and reset the clone to the pushed state."""
    _run(
        _lib.stage_abort, project,
        workdir=workdir, explicit_config=config, force=force,
    )
