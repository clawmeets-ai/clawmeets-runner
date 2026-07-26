# SPDX-License-Identifier: MIT
"""
clawmeets/cli_etl.py — Derived-table ETL CLI (deterministic harness).
"""
from __future__ import annotations

import json

import typer

from clawmeets.integrations.etl import _lib

app = typer.Typer(
    name="etl",
    help="Derived-table ETL over the data warehouse. Paired skill: etl.",
    no_args_is_help=True,
)


@app.command("load-candidates")
def load_candidates(
    rule: str = typer.Argument(..., help="Rule name from etl.json"),
    dwh: str = typer.Option(..., "--dwh"),
    config: str = typer.Option("", "--config"),
) -> None:
    """Pull candidates from each source's delta log into derived/.<rule>.candidates.json."""
    typer.echo(json.dumps(
        _lib.load_candidates(dwh, rule, config_file=config),
        indent=2, ensure_ascii=False,
    ))


@app.command("merge")
def merge(
    rule: str = typer.Argument(..., help="Rule name from etl.json"),
    dwh: str = typer.Option(..., "--dwh"),
    config: str = typer.Option("", "--config"),
) -> None:
    """Validate the batch ndjson, merge into the derived TSV, advance state."""
    typer.echo(json.dumps(
        _lib.merge(dwh, rule, config_file=config),
        indent=2, ensure_ascii=False,
    ))
