# SPDX-License-Identifier: MIT
"""
clawmeets/cli_database.py — Generic SQL-database sync CLI.
"""
from __future__ import annotations

import json

import typer

from clawmeets.integrations.database import _lib

app = typer.Typer(
    name="database",
    help="Generic SQL database sync (sqlalchemy-based). Paired skill: database.",
    no_args_is_help=True,
)


@app.command()
def sync(
    dwh: str = typer.Option(..., "--dwh"),
    config: str = typer.Option("", "--config"),
    max_runtime: int = typer.Option(1500, "--max-runtime"),
) -> None:
    """Run configured SQL queries into the warehouse per --config."""
    typer.echo(json.dumps(_lib.sync_to_warehouse(
        dwh, config_file=config, max_runtime_seconds=max_runtime,
    ), indent=2, ensure_ascii=False))
