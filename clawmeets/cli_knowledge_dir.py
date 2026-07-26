# SPDX-License-Identifier: MIT
"""
clawmeets/cli_knowledge_dir.py — proprietary-knowledge index maintenance CLI.

`clawmeets knowledge-dir reindex` rebuilds ``{agent_dir}/memory/REFERENCES.md``
— the deterministic map of the user's knowledge_dir files (filename + size +
content preview). The runner rebuilds it automatically at startup and on a
knowledge_dir settings change; this command is the on-demand path for when the
user edits the knowledge folder out-of-band ("refresh your knowledge index").

The paired `consult-proprietary-knowledge` skill shells this with the resolved
knowledge_dir path(s) it reads from its prompt's `User-curated reference
material` line, so no relative-path resolution is needed here.
"""
from __future__ import annotations

from pathlib import Path

import typer

from clawmeets.runner.references_index import build_references_index

app = typer.Typer(
    name="knowledge-dir",
    help="Proprietary-knowledge index maintenance (REFERENCES.md).",
    no_args_is_help=True,
)


@app.command("reindex")
def reindex(
    knowledge_dir: list[str] = typer.Option(
        ..., "--knowledge-dir", "-k",
        help="Absolute knowledge_dir path to index. Repeat for multiple. Read "
             "these from the `User-curated reference material` line of your prompt.",
    ),
    agent_dir: str = typer.Option(
        ..., "--agent-dir", envvar="CLAWMEETS_AGENT_DIR",
        help="Agent home dir (REFERENCES.md is written under its memory/).",
    ),
) -> None:
    """Rebuild ``{agent_dir}/memory/REFERENCES.md`` from the knowledge_dir(s)."""
    memory_dir = Path(agent_dir).expanduser() / "memory"
    err = build_references_index(memory_dir, [Path(d) for d in knowledge_dir])
    if err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Reindexed {memory_dir / 'REFERENCES.md'}")
