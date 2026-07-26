# SPDX-License-Identifier: MIT
"""
clawmeets/cli_dwh.py — Data-warehouse maintenance CLI.

`clawmeets dwh catalog` (re)builds ``{dwh}/CATALOG.md`` — the aggregate
discovery index over raw snapshots + derived tables. Syncs and ETL merges
refresh it automatically; this command is the manual / scheduled entry point
(e.g. shelled from the recurring `today` project) and the way to seed a
catalog for an externally-populated warehouse.
"""
from __future__ import annotations

import typer

from clawmeets.integrations._sync_warehouse import refresh_catalog

app = typer.Typer(
    name="dwh",
    help="Data-warehouse maintenance (catalog index).",
    no_args_is_help=True,
)


@app.command("catalog")
def catalog(
    dwh: str = typer.Option(
        ..., "--dwh", envvar="CLAWMEETS_DWH_DIR",
        help="Data-warehouse root. Read it from the `Data warehouse :` line of "
             "your prompt's runtime-context block.",
    ),
) -> None:
    """Rebuild ``{dwh}/CATALOG.md`` from the current raw + derived tables."""
    err = refresh_catalog(dwh)
    if err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Wrote {dwh.rstrip('/')}/CATALOG.md")
