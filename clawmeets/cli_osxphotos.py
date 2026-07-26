# SPDX-License-Identifier: MIT
"""
clawmeets/cli_osxphotos.py — macOS Photos library CLI.

Subcommands: list-albums, list-photos, export, export-jpeg, sync.
"""
from __future__ import annotations

import json
from typing import Optional

import typer

from clawmeets.integrations.osxphotos import _lib

app = typer.Typer(
    name="osxphotos",
    help="macOS Photos library (read-only metadata + paths + sync). Paired skill: osxphotos. macOS only.",
    no_args_is_help=True,
)


def _emit_json(payload) -> None:
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


@app.command("list-albums")
def list_albums_cmd() -> None:
    """List every album with photo counts + date range."""
    try:
        _emit_json(_lib.list_albums())
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command("list-photos")
def list_photos_cmd(
    album: Optional[str] = typer.Option(None, "--album"),
    year: Optional[int] = typer.Option(None, "--year"),
    limit: Optional[int] = typer.Option(None, "--limit"),
) -> None:
    """List photos (metadata + paths, no bytes)."""
    try:
        _emit_json(_lib.list_photos(album=album, year=year, limit=limit))
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command()
def export(
    uuid: str = typer.Argument(...),
    dest_dir: str = typer.Option(..., "--dest"),
) -> None:
    """Force-download an iCloud-optimized photo."""
    try:
        _emit_json(_lib.export_photo(uuid, dest_dir))
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command("export-jpeg")
def export_jpeg_cmd(
    uuid: str = typer.Argument(...),
    dest_dir: str = typer.Option(..., "--dest"),
    max_dim: int = typer.Option(1200, "--max-dim"),
    quality: int = typer.Option(65, "--quality"),
) -> None:
    """Transcode to a JPEG sized to fit Claude Code's 256 KB Read cap."""
    try:
        _emit_json(_lib.export_photo_as_jpeg(
            uuid, dest_dir, max_dim=max_dim, quality=quality,
        ))
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command()
def sync(
    dwh: str = typer.Option(..., "--dwh"),
    config: str = typer.Option("", "--config"),
    max_runtime: int = typer.Option(1500, "--max-runtime"),
) -> None:
    """Sync newly-added photos into the warehouse."""
    try:
        _emit_json(_lib.sync_to_warehouse(
            dwh, config_file=config, max_runtime_seconds=max_runtime,
        ))
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
