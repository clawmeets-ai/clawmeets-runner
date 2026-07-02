# SPDX-License-Identifier: MIT
"""
clawmeets/utils/knowledge_index.py

The shared *knowledge-index contract* — one markdown shape reused by every
place the agent discovers domain knowledge: installed knowledge packs
(``memory/KNOWLEDGE_PACKS.md``), the data-warehouse catalog
(``{dwh}/CATALOG.md``), and the proprietary-reference index
(``memory/REFERENCES.md``).

Storage stays decoupled — three separate index files, no cross-pointers — but
they all render entries the same way so the agent's consumption is identical
regardless of source: read the index, match the one-line *consult when*, then
open only the file it points at. The always-on prompt enumerates whichever
indexes exist (see ``llm/prompt_builder._build_knowledge_precedence``).

All three indexes are built deterministically in Python and import the helpers
here: ``runner/knowledge_pack_manager`` (packs),
``runner/references_index.build_references_index`` (knowledge_dir →
``REFERENCES.md``, filename + content preview), and
``integrations/_sync_warehouse.refresh_catalog`` (dwh). The
``consult-proprietary-knowledge`` skill is now a *reader/searcher* of
``REFERENCES.md`` (grep/find over the live files), not its author.

Canonical entry::

    - [label](/abs/path) (meta) — consult when <concrete signal>

``meta`` (row counts, mtime, size) is optional context shown in parentheses;
``when`` is the discriminator the agent matches against the task at hand.
Paths are always absolute so the Read tool resolves them regardless of the
agent's working directory.
"""
from __future__ import annotations

from pathlib import Path

# Mirror of the frontend's TEXT_EXTENSIONS set
# (web/frontend/src/components/settings/KnowledgePacksSection.tsx) so the
# runner classifies pack/knowledge_dir files the same way the upload UI does.
TEXT_EXTENSIONS = frozenset({
    "md", "markdown", "txt", "json", "jsonl", "ndjson", "yaml", "yml",
    "toml", "ini", "csv", "tsv", "html", "htm", "xml", "svg",
    "py", "js", "jsx", "ts", "tsx", "css", "scss", "sh", "bash",
    "rb", "go", "rs", "java", "c", "cc", "cpp", "h", "hpp",
    "log", "env", "gitignore", "editorconfig",
})

PREVIEW_WORDS = 30
"""Default number of leading words `file_preview` returns — enough to act as a
'consult when' discriminator without bloating the index (which the agent Reads
on demand, not every turn)."""

_PREVIEW_READ_BYTES = 4096
"""Cap on bytes read for a preview so large files don't get loaded whole."""

INDEX_PREAMBLE = (
    "Read a file when its *consult when* line below matches the task at hand. "
    "Paths are absolute so they resolve regardless of your current working dir."
)


def render_index_entry(
    label: str,
    abs_path: str,
    *,
    when: str = "",
    meta: str = "",
    indent: int = 0,
) -> str:
    """Render one canonical index bullet.

    ``label``     human-readable name (filename, pack name, table path).
    ``abs_path``  absolute path the agent's Read tool opens.
    ``when``      the one-line "consult when <signal>" discriminator (no trailing
                  period needed). Concrete signals beat generic phrasings.
    ``meta``      optional parenthetical context (e.g. "1.2 KB", "812 rows,
                  synced 2026-06-20"). Rendered as ``(meta)`` before the dash.
    ``indent``    nesting depth; each level adds two leading spaces (used for
                  per-file bullets under a pack heading).
    """
    pad = "  " * max(indent, 0)
    line = f"{pad}- [{label}]({abs_path})"
    if meta:
        line += f" ({meta})"
    when = (when or "").strip().replace("\n", " ")
    if when:
        line += f" — {when}"
    return line


def render_freshness_header(mapping: dict[str, object]) -> str:
    """Render a minimal YAML front-matter block for an index file.

    ``mapping`` is rendered as flat ``key: value`` lines between ``---``
    fences. Used by deterministic builders to stamp a last-built marker (and,
    for REFERENCES.md, per-dir mtimes) so a consumer can cheaply tell whether
    the index is current. Values are emitted verbatim — pre-format dicts into
    your own indented block before calling if you need nesting.
    """
    lines = ["---"]
    for key, value in mapping.items():
        lines.append(f"{key}: {value}")
    lines.append("---")
    return "\n".join(lines)


def is_text_file(path) -> bool:
    """Whether a path looks like a text file, by extension (mirrors the upload
    UI). No extension ⇒ treated as text (READMEs, dotfiles like `.env`)."""
    name = Path(path).name
    dot = name.rfind(".")
    if dot < 0:
        return True
    return name[dot + 1:].lower() in TEXT_EXTENSIONS


def _strip_front_matter(text: str) -> str:
    """Drop a leading YAML front-matter block (``---\\n…\\n---``) so previews
    show real content, not metadata. No-op if absent."""
    if not text.startswith("---"):
        return text
    end = text.find("\n---", 3)
    if end == -1:
        return text
    rest = text[end + 4:]
    return rest.lstrip("\n") if rest.lstrip("\n") else text


def file_preview(path, *, max_words: int = PREVIEW_WORDS) -> str:
    """First ``max_words`` words of a text file as a one-line preview — the
    deterministic 'consult when' signal for the index. Reads at most
    ``_PREVIEW_READ_BYTES`` so large files don't load whole; strips a leading
    YAML front-matter block; collapses all whitespace; appends ``…`` when
    truncated. Returns ``""`` for binary / empty / unreadable files (the caller
    falls back to a size-only entry). Never raises."""
    if not is_text_file(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="strict") as f:
            head = f.read(_PREVIEW_READ_BYTES)
    except (OSError, UnicodeDecodeError):
        return ""
    words = _strip_front_matter(head).split()
    if not words:
        return ""
    preview = " ".join(words[:max_words])
    if len(words) > max_words:
        preview += " …"
    return preview
