# SPDX-License-Identifier: MIT
"""
clawmeets/cli_website_monitor.py — Website-monitor crawler bookkeeping CLI.
"""
from __future__ import annotations

import json

import typer

from clawmeets.integrations.website_monitor import _lib

app = typer.Typer(
    name="website-monitor",
    help="Website-monitor crawl bookkeeping. Paired skill: website-monitor.",
    no_args_is_help=True,
)


@app.command("begin")
def begin(
    rule: str = typer.Argument(..., help="Rule name from website-monitor.json"),
    dwh: str = typer.Option(..., "--dwh"),
    config: str = typer.Option("", "--config"),
) -> None:
    """Validate the rule and open a fresh batch file for the crawl to append to."""
    typer.echo(json.dumps(
        _lib.begin(dwh, rule, config_file=config), indent=2, ensure_ascii=False))


@app.command("merge")
def merge(
    rule: str = typer.Argument(..., help="Rule name from website-monitor.json"),
    dwh: str = typer.Option(..., "--dwh"),
    config: str = typer.Option("", "--config"),
) -> None:
    """Fold the crawl batch into raw/website/<rule>/{snapshot.tsv, deltas/}."""
    typer.echo(json.dumps(
        _lib.merge(dwh, rule, config_file=config), indent=2, ensure_ascii=False))
