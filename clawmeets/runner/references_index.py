# SPDX-License-Identifier: MIT
"""
clawmeets/runner/references_index.py

Deterministic builder for ``{agent_dir}/memory/REFERENCES.md`` — the index of
the user's proprietary reference files (the configured ``knowledge_dir``).

Previously REFERENCES.md was LLM-authored lazily by the
``consult-proprietary-knowledge`` skill, which left it stale/missing whenever
the agent read it directly (the prompt points at it). This builder makes it
runner-owned and always-fresh, exactly like the knowledge-pack index
(``KNOWLEDGE_PACKS.md``) and the dwh catalog (``CATALOG.md``): one bullet per
file in the shared knowledge-index format (``utils.knowledge_index``), with a
deterministic first-words **content preview** as the per-file "consult when".

The preview is a *map* — it tells the agent which files exist and roughly what
each opens with. Targeted lookups ("which file mentions X") are handled live by
the grep/find ``consult-proprietary-knowledge`` skill, which also covers any
staleness from nested-file edits between rebuilds.

Rebuilt at runner startup, on a knowledge_dir ``AGENT_SETTINGS_CHANGE``, and on
demand via ``clawmeets knowledge-dir reindex``.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

from clawmeets.utils.knowledge_index import (
    INDEX_PREAMBLE,
    file_preview,
    render_freshness_header,
    render_index_entry,
)

logger = logging.getLogger("clawmeets.runner")

INDEX_FILENAME = "REFERENCES.md"

# Names/dirs to skip while walking a knowledge_dir. Mirrors the exclusions the
# old consult skill used so the index never lists the agent's own memory files
# or scaffolding that happens to live alongside reference material.
_SKIP_DIR_NAMES = frozenset({"learnings", "skills", "config"})
_SKIP_FILE_NAMES = frozenset({
    "USER.md", "REFERENCES.md", "KNOWLEDGE_PACKS.md", "CLAUDE.md", "README.md",
})

MAX_FILES = 300
"""Cap on total files listed across all knowledge_dirs so the index stays
bounded. The grep skill covers anything beyond the cap."""


def _human_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def _iter_files(root: Path):
    """Yield reference files under ``root``, depth-first sorted, skipping
    dotfiles/dotdirs and the excluded dir/file names."""
    for entry in sorted(root.rglob("*")):
        if not entry.is_file():
            continue
        rel_parts = entry.relative_to(root).parts
        if any(p.startswith(".") for p in rel_parts):
            continue
        if any(p in _SKIP_DIR_NAMES for p in rel_parts[:-1]):
            continue
        if entry.name in _SKIP_FILE_NAMES:
            continue
        yield entry


def build_references_index(
    memory_dir: Path, knowledge_dirs: list[Path] | None
) -> Optional[str]:
    """(Re)build ``{memory_dir}/REFERENCES.md`` from the knowledge_dir(s).

    Deterministic + idempotent. Returns None on success / no-op, or a one-line
    error string. Never raises. With no knowledge_dirs the index is removed so
    the prompt doesn't point at a stale file.
    """
    index_path = Path(memory_dir) / INDEX_FILENAME
    dirs = [Path(d) for d in (knowledge_dirs or []) if Path(d).is_dir()]

    if not dirs:
        try:
            index_path.unlink(missing_ok=True)
        except OSError as exc:
            return f"could not remove stale REFERENCES.md: {exc}"
        return None

    mtimes: dict[str, int] = {}
    sections: list[str] = []
    total = 0
    truncated = 0
    for d in dirs:
        try:
            mtimes[d.as_posix()] = int(d.stat().st_mtime)
        except OSError:
            mtimes[d.as_posix()] = 0
        entries: list[str] = []
        for f in _iter_files(d):
            if total >= MAX_FILES:
                truncated += 1
                continue
            total += 1
            try:
                size = _human_size(f.stat().st_size)
            except OSError:
                size = ""
            entries.append(render_index_entry(
                f.relative_to(d).as_posix(), f.as_posix(),
                meta=size, when=file_preview(f),
            ))
        sections.append(f"## {d.as_posix()}")
        sections.append("")
        sections.extend(entries or ["_(no reference files)_"])
        sections.append("")

    parts = [
        render_freshness_header({
            "last_built": datetime.now(UTC).isoformat(),
            **{f"mtime:{k}": v for k, v in mtimes.items()},
        }),
        "",
        "# Proprietary knowledge index",
        "",
        INDEX_PREAMBLE,
        "",
        *sections,
    ]
    if truncated:
        parts.append(
            f"_(… {truncated} more file(s) over the {MAX_FILES}-file cap — "
            "use the proprietary-knowledge search to grep the rest.)_"
        )

    try:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text("\n".join(parts).rstrip("\n") + "\n")
    except OSError as exc:
        return f"REFERENCES.md write failed: {exc}"
    logger.info("Built REFERENCES.md: %d file(s) across %d dir(s)", total, len(dirs))
    return None
