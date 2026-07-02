# SPDX-License-Identifier: MIT
"""
clawmeets/runner/knowledge_pack_manager.py

Manages user-installed knowledge packs for an agent.

A knowledge pack is a user-curated, named *bundle of files* (markdown
typically). Installs arrive over the WebSocket as ``KNOWLEDGE_PACK_SYNC``
envelopes carrying the pack's full file set; the manager rewrites
``{agent_dir}/knowledge_packs/<slug>/`` from scratch on every install event
and rebuilds the AUTHORITATIVE-layer index at
``{agent_dir}/memory/KNOWLEDGE_PACKS.md`` so the prompt builder picks it up
on the next LLM invocation.

Layout::

    {agent_dir}/
    ├── knowledge_packs/
    │   ├── _meta.json                 { "<slug>": {name, description, files} }
    │   ├── wine-recommendations/
    │   │   ├── fine-dining-tactics.md
    │   │   ├── by-the-glass-strategy.md
    │   │   └── notes/                 (nested directories preserved from upload)
    │   │       └── cellar-curation.md
    │   └── design-refs/
    │       ├── moodboard.png          (binary files supported)
    │       └── voice-guide.md
    └── memory/
        └── KNOWLEDGE_PACKS.md         index file rebuilt on every install/uninstall

The index file is a convenience for the LLM (rendered into the prompt's
AUTHORITATIVE block). ``_meta.json`` is the manager's source of truth and
lets us rebuild the index after a single-pack install without re-fetching
every installed pack from the server. The index references pack files by
**absolute path** so the agent's Read tool resolves regardless of working
directory — the content lives in ``knowledge_packs/`` (sibling of memory/)
to keep memory/ as small text-only indexes.

Wire format: install events carry ``files`` as
``{relative_path: {"content_b64": <base64>}}``. The manager writes raw bytes
to disk; consumers (Read tool, etc.) handle them just like any other file.
"""
from __future__ import annotations

import base64
import json
import logging
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from clawmeets.utils.knowledge_index import file_preview, render_index_entry

if TYPE_CHECKING:
    from clawmeets.api.client import ClawMeetsClient
    from clawmeets.models.context import ModelContext

logger = logging.getLogger("clawmeets.runner")

INDEX_FILENAME = "KNOWLEDGE_PACKS.md"
META_FILENAME = "_meta.json"

# Per-segment filename rule, mirrored from clawmeets/models/knowledge_pack.py
# so the runner can defensively re-validate paths from the wire without
# depending on the server-side model package.
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _safe_path_parts(path: str) -> list[str] | None:
    """Return the path segments for a server-supplied pack file path, or None
    if it fails any safety check (absolute, backslash, traversal, empty
    segment, malformed segment).
    """
    if not path or "\\" in path or path.startswith("/"):
        return None
    parts = path.split("/")
    for seg in parts:
        if not seg or seg == ".." or not _SEGMENT_RE.match(seg):
            return None
    return parts


class KnowledgePackManager:
    """Per-agent installer for user-curated knowledge packs (multi-file)."""

    def __init__(self, model_ctx: "ModelContext") -> None:
        self._model_ctx = model_ctx

    # ─────────────────────────────────────────────────────────
    # Path helpers — derived from agent_dir (model_ctx.base_dir on the
    # runner). Pack content lives under agent_dir/knowledge_packs/; the
    # index lives under agent_dir/memory/.
    # ─────────────────────────────────────────────────────────

    def _packs_dir(self) -> Path:
        return self._model_ctx.packs_dir

    def _index_path(self) -> Path:
        return self._model_ctx.memory_dir / INDEX_FILENAME

    def _meta_path(self) -> Path:
        return self._packs_dir() / META_FILENAME

    def _pack_dir(self, slug: str) -> Path:
        return self._packs_dir() / slug

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    def install_pack(
        self,
        slug: str,
        name: str,
        description: str,
        files: dict[str, dict],
    ) -> None:
        """Rewrite a pack's directory + meta + index from the given files dict.

        ``files`` is ``{relative_path: {"content_b64": <base64>}}``. Paths may
        contain ``/`` for nested files.

        Idempotent — calling repeatedly with the same files leaves the disk
        layout identical. Called both on initial install and on every
        re-install (server fans out a fresh KNOWLEDGE_PACK_SYNC after every
        edit), so wiping + rewriting is the simplest safe behavior.
        """
        pack_dir = self._pack_dir(slug)

        # Wipe any prior contents so removed files don't linger.
        if pack_dir.exists():
            shutil.rmtree(pack_dir)
        pack_dir.mkdir(parents=True, exist_ok=True)

        written: list[str] = []
        for path, entry in (files or {}).items():
            parts = _safe_path_parts(path)
            if parts is None:
                logger.warning(
                    "Skipping suspicious filepath in pack %r: %r", slug, path,
                )
                continue
            content_b64 = (entry or {}).get("content_b64") or ""
            try:
                body = base64.b64decode(content_b64, validate=True)
            except (ValueError, TypeError) as e:
                logger.warning(
                    "Skipping malformed base64 for %s in pack %r: %s",
                    path, slug, e,
                )
                continue
            target = pack_dir.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
            written.append(path)

        meta = self._read_meta()
        meta[slug] = {
            "name": name,
            "description": description,
            "files": sorted(written),
            "file_descriptions": {
                path: str((files.get(path) or {}).get("description") or "")
                for path in written
                if str((files.get(path) or {}).get("description") or "").strip()
            },
        }
        self._write_meta(meta)
        self._rebuild_index(meta)
        logger.info(
            "Installed knowledge pack: %s (%d file%s)",
            slug, len(written), "" if len(written) == 1 else "s",
        )

    def uninstall_pack(self, slug: str) -> None:
        """Delete the pack directory + meta entry + rebuild the index."""
        pack_dir = self._pack_dir(slug)
        if pack_dir.exists():
            shutil.rmtree(pack_dir)

        meta = self._read_meta()
        meta.pop(slug, None)
        self._write_meta(meta)
        self._rebuild_index(meta)
        logger.info("Uninstalled knowledge pack: %s", slug)

    def installed_packs(self) -> list[str]:
        """List installed pack slugs (alphabetical)."""
        return sorted(self._read_meta().keys())

    async def sync_from_server(self, client: "ClawMeetsClient", agent_id: str) -> None:
        """Reconcile local installed packs with the server's authoritative list.

        Fetches the full installed-packs payload (slug + name + description +
        files dict + updated_at) and rewrites the local knowledge-packs
        directory and index from scratch. Adds new packs, refreshes existing
        ones, and removes packs the server no longer reports.
        """
        try:
            resp = await client._http.get(f"/agents/{agent_id}/knowledge-packs")
            resp.raise_for_status()
            data = resp.json()
            server_packs = data.get("installed_packs") or []
        except Exception as e:
            logger.warning("Failed to fetch installed knowledge packs from server: %s", e)
            return

        pdir = self._packs_dir()
        pdir.mkdir(parents=True, exist_ok=True)

        server_slugs: set[str] = set()
        meta: dict[str, dict] = {}
        for pack in server_packs:
            slug = pack.get("slug")
            if not slug:
                continue
            server_slugs.add(slug)
            files = pack.get("files") or {}
            if not isinstance(files, dict):
                files = {}
            written = self._write_pack_dir(slug, files)
            meta[slug] = {
                "name": pack.get("name") or slug,
                "description": pack.get("description") or "",
                "files": sorted(written),
                "file_descriptions": {
                    path: str((files.get(path) or {}).get("description") or "")
                    for path in written
                    if str((files.get(path) or {}).get("description") or "").strip()
                },
            }

        # Drop local pack dirs (and the legacy <slug>.md flat files) the
        # server no longer reports.
        for entry in pdir.iterdir():
            if entry.name == META_FILENAME:
                continue
            slug = entry.stem if entry.is_file() else entry.name
            if slug in server_slugs:
                continue
            if entry.is_dir():
                shutil.rmtree(entry)
            elif entry.is_file():
                entry.unlink()

        self._write_meta(meta)
        self._rebuild_index(meta)
        logger.info("Synced %d knowledge pack(s) from server", len(server_slugs))

    # ─────────────────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────────────────

    def _write_pack_dir(self, slug: str, files: dict[str, dict]) -> list[str]:
        """Materialize a pack's files on disk. ``files`` is
        ``{relative_path: {"content_b64": ...}}``. Returns the list of paths
        actually written (rejecting any that fail the defensive validator)."""
        pack_dir = self._pack_dir(slug)
        if pack_dir.exists():
            shutil.rmtree(pack_dir)
        pack_dir.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        for path, entry in files.items():
            parts = _safe_path_parts(path)
            if parts is None:
                logger.warning(
                    "Skipping suspicious filepath in pack %r: %r", slug, path,
                )
                continue
            content_b64 = (entry or {}).get("content_b64") or ""
            try:
                body = base64.b64decode(content_b64, validate=True)
            except (ValueError, TypeError) as e:
                logger.warning(
                    "Skipping malformed base64 for %s in pack %r: %s",
                    path, slug, e,
                )
                continue
            target = pack_dir.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
            written.append(path)
        return written

    def _read_meta(self) -> dict[str, dict]:
        path = self._meta_path()
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text()) or {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_meta(self, meta: dict[str, dict]) -> None:
        path = self._meta_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, indent=2, sort_keys=True))

    def _rebuild_index(self, meta: dict[str, dict]) -> None:
        """Rebuild ``memory/KNOWLEDGE_PACKS.md`` from ``_meta.json``.

        Each pack contributes an entry that lists the pack name, optional
        description, and one indented bullet per file pointing at the file's
        **absolute path** under ``knowledge_packs/``. The index lives in
        ``memory/`` (next to USER.md / REFERENCES.md / learnings/INDEX.md)
        but content lives in the sibling ``knowledge_packs/`` dir, so the
        agent's Read tool can resolve the link regardless of working dir.
        """
        index_path = self._index_path()
        if not meta:
            if index_path.exists():
                index_path.unlink()
            return

        packs_dir = self._packs_dir()
        index_path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Installed knowledge packs", ""]
        for slug in sorted(meta.keys()):
            entry = meta[slug]
            description = (entry.get("description") or "").strip().replace("\n", " ")
            name = (entry.get("name") or slug).strip()
            tail = f" — {description}" if description else ""
            lines.append(f"- **{name}** (`{slug}`){tail}")
            fdesc = entry.get("file_descriptions") or {}
            for filename in entry.get("files") or []:
                # Absolute path so the agent's Read tool resolves regardless of
                # working dir. 'consult when' = the user's curated hint if set,
                # else a deterministic first-words preview of the file content
                # (fills bulk uploads that carry no per-file description).
                file_path = packs_dir / slug / filename
                when = str(fdesc.get(filename) or "").strip()
                if not when:
                    when = file_preview(file_path)
                lines.append(render_index_entry(
                    filename, file_path.as_posix(), when=when, indent=1,
                ))
        index_path.write_text("\n".join(lines) + "\n")
