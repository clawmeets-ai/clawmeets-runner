# SPDX-License-Identifier: BUSL-1.1
"""
clawmeets/models/knowledge_pack.py

Owner-scoped knowledge packs — user-curated, named bundles of files (text or
binary) that any of the user's agents can install.

A knowledge pack lives in two places:
- Server registry: this module — a directory tree at
  ``{data_dir}/knowledge-packs/<username>/<slug>/`` holding one ``_meta.json``
  sidecar plus the pack's files as real siblings on disk. Subdirectories are
  preserved (uploaded folder structure).
- Per-agent install: when the user installs a pack onto an agent, the runner
  writes one file per pack-file under
  ``{knowledge_dir}/knowledge-packs/<slug>/`` (with the same nested layout)
  and indexes the pack in ``{knowledge_dir}/KNOWLEDGE_PACKS.md`` (handled by
  ``runner/knowledge_pack_manager.py``).

The pack is the user's explicit promotion of agent-produced content into the
AUTHORITATIVE memory layer. The user is the editor — same authority pattern
as ``USER.md`` and ``REFERENCES.md``.

Storage layout::

    {data_dir}/knowledge-packs/
      <username>/                # username; validated by validate_name
        <slug>/                  # slug; validated by validate_slug
          _meta.json             # {"name", "description", "created_at", "updated_at"}
          <path/to/file>         # one real file per pack-file (validate_filepath)
          ...
        <other-slug>/
          _meta.json
          ...

On the wire, file content is always base64-encoded raw bytes — the server
treats every file as opaque bytes, and the frontend decides per-file
(by extension) whether to UTF-8 decode for display.

Why filesystem instead of one JSON-blob: human-greppable on disk; admin can
``ls`` / ``cat`` per-pack content; per-user filesystem ACLs work; no giant
blob to read on every list call.
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from clawmeets.utils.file_io import FileUtil
from clawmeets.utils.validation import validate_name

_lock = asyncio.Lock()

PACKS_DIR = "knowledge-packs"
META_FILENAME = "_meta.json"

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

MAX_FILEPATH_LEN = 200


class PackFile(BaseModel):
    """A single file inside a knowledge pack. Content is raw bytes,
    base64-encoded for transport. The server treats every pack file as opaque
    bytes; the frontend decides per-file (by extension) whether to UTF-8
    decode for display."""

    content_b64: str


class KnowledgePack(BaseModel):
    """A user-curated, named bundle of files."""

    slug: str
    name: str
    description: str = ""
    files: dict[str, PackFile] = Field(default_factory=dict)
    created_at: str
    updated_at: str = Field(default="")


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def validate_slug(slug: str) -> str:
    """Normalize and validate a pack slug. Raises ValueError if invalid."""
    cleaned = (slug or "").strip().lower()
    if not cleaned:
        raise ValueError("Pack slug cannot be empty")
    if len(cleaned) > 80:
        raise ValueError("Pack slug must be 80 characters or fewer")
    if not _SLUG_RE.match(cleaned):
        raise ValueError(
            "Pack slug must start with [a-z0-9] and contain only lowercase "
            "letters, digits, hyphens, and underscores"
        )
    return cleaned


def validate_filename(filename: str) -> str:
    """Normalize and validate a single pack filename segment.

    Disallows path separators, hidden files (leading dot), and anything that
    would let a caller escape the pack directory at install time. Trims
    whitespace. Used for individual path segments inside ``validate_filepath``
    and as the historical single-segment validator.
    """
    cleaned = (filename or "").strip()
    if not cleaned:
        raise ValueError("Filename cannot be empty")
    if len(cleaned) > 100:
        raise ValueError("Filename must be 100 characters or fewer")
    if "/" in cleaned or "\\" in cleaned or ".." in cleaned:
        raise ValueError("Filename cannot contain path separators or '..'")
    if not _FILENAME_RE.match(cleaned):
        raise ValueError(
            "Filename must start with an alphanumeric character and contain "
            "only letters, digits, dots, hyphens, and underscores"
        )
    return cleaned


def validate_filepath(filepath: str) -> str:
    """Normalize and validate a relative pack file path. Accepts forward-slash
    separated segments where each segment satisfies ``validate_filename``.

    Rejects absolute paths, backslashes, empty segments, and any total length
    over ``MAX_FILEPATH_LEN``. Returns the normalized forward-slash path.
    """
    cleaned = (filepath or "").strip()
    if not cleaned:
        raise ValueError("File path cannot be empty")
    if len(cleaned) > MAX_FILEPATH_LEN:
        raise ValueError(f"File path must be {MAX_FILEPATH_LEN} characters or fewer")
    if "\\" in cleaned:
        raise ValueError("File path cannot contain backslashes")
    if cleaned.startswith("/"):
        raise ValueError("File path must be relative")
    segments = cleaned.split("/")
    validated: list[str] = []
    for seg in segments:
        validated.append(validate_filename(seg))
    return "/".join(validated)


def _validate_username(username: str) -> str:
    """Defensive re-check at the model boundary. Usernames are already
    validated at registration via ``validate_name``; this guards against
    a caller passing something else (an id, a path, an empty string).
    """
    cleaned = (username or "").strip()
    if not cleaned:
        raise ValueError("Username cannot be empty")
    return validate_name(cleaned)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _packs_root(data_dir: Path) -> Path:
    return Path(data_dir) / PACKS_DIR


def _user_dir(data_dir: Path, username: str) -> Path:
    return _packs_root(data_dir) / _validate_username(username)


def _pack_dir(data_dir: Path, username: str, slug: str) -> Path:
    return _user_dir(data_dir, username) / validate_slug(slug)


def _meta_path(pack_dir: Path) -> Path:
    return pack_dir / META_FILENAME


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Internal load / save
# ---------------------------------------------------------------------------


def _load_pack(pack_dir: Path, slug: str) -> KnowledgePack | None:
    """Build a KnowledgePack from a pack directory on disk. Returns None if
    the directory doesn't exist or has no ``_meta.json`` (treated as
    not-a-pack — the dir might be partial / mid-write or stale).

    Walks the pack directory recursively so nested files surface with their
    full relative path (forward-slash on every platform).
    """
    if not pack_dir.is_dir():
        return None
    meta = FileUtil.read(_meta_path(pack_dir), "json")
    if not isinstance(meta, dict):
        return None

    files: dict[str, PackFile] = {}
    for entry in pack_dir.rglob("*"):
        if not entry.is_file():
            continue
        if entry.name == META_FILENAME and entry.parent == pack_dir:
            continue
        rel = entry.relative_to(pack_dir).as_posix()
        try:
            validate_filepath(rel)
        except ValueError:
            continue
        try:
            files[rel] = PackFile(
                content_b64=base64.b64encode(entry.read_bytes()).decode("ascii"),
            )
        except OSError:
            continue

    return KnowledgePack(
        slug=slug,
        name=meta.get("name") or slug,
        description=meta.get("description") or "",
        files=files,
        created_at=meta.get("created_at") or _now(),
        updated_at=meta.get("updated_at") or meta.get("created_at") or _now(),
    )


def _write_meta(
    pack_dir: Path,
    *,
    name: str,
    description: str,
    created_at: str,
    updated_at: str,
) -> None:
    pack_dir.mkdir(parents=True, exist_ok=True)
    FileUtil.write(
        _meta_path(pack_dir),
        {
            "name": name,
            "description": description,
            "created_at": created_at,
            "updated_at": updated_at,
        },
        "json",
    )


def _read_meta(pack_dir: Path) -> dict:
    meta = FileUtil.read(_meta_path(pack_dir), "json")
    if not isinstance(meta, dict):
        return {}
    return meta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_packs(data_dir: Path, username: str) -> list[KnowledgePack]:
    """Return every persisted pack for the given user, sorted by slug."""
    user_dir = _user_dir(data_dir, username)
    if not user_dir.is_dir():
        return []

    packs: list[KnowledgePack] = []
    for entry in sorted(user_dir.iterdir()):
        if not entry.is_dir():
            continue
        try:
            slug = validate_slug(entry.name)
        except ValueError:
            continue
        pack = _load_pack(entry, slug)
        if pack is not None:
            packs.append(pack)
    return packs


def get_pack(data_dir: Path, username: str, slug: str) -> KnowledgePack | None:
    pack_dir = _pack_dir(data_dir, username, slug)
    return _load_pack(pack_dir, validate_slug(slug))


async def create_pack(
    data_dir: Path,
    username: str,
    slug: str,
    name: str,
    description: str = "",
    files: dict[str, str] | None = None,
) -> KnowledgePack:
    """Create a new pack. ``files`` is ``{relative_path: base64_content}``.

    Raises ValueError if the slug already exists or any path is invalid.
    """
    slug = validate_slug(slug)
    validated_files: dict[str, bytes] = {}
    for raw_path, body in (files or {}).items():
        path = validate_filepath(raw_path)
        try:
            validated_files[path] = base64.b64decode(body or "", validate=True)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid base64 content for {raw_path!r}: {e}")

    async with _lock:
        pack_dir = _pack_dir(data_dir, username, slug)
        if pack_dir.exists():
            raise ValueError(f"Knowledge pack {slug!r} already exists")

        now = _now()
        _write_meta(
            pack_dir,
            name=name,
            description=description,
            created_at=now,
            updated_at=now,
        )
        for rel_path, body_bytes in validated_files.items():
            target = pack_dir.joinpath(*rel_path.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body_bytes)

        pack = _load_pack(pack_dir, slug)
        assert pack is not None  # just wrote it
        return pack


async def update_pack(
    data_dir: Path,
    username: str,
    slug: str,
    name: str | None = None,
    description: str | None = None,
) -> KnowledgePack | None:
    """Patch a pack's metadata (name / description). Returns None if the
    pack does not exist. File-level edits go through
    ``upsert_file`` / ``delete_file``.
    """
    slug = validate_slug(slug)
    async with _lock:
        pack_dir = _pack_dir(data_dir, username, slug)
        if not pack_dir.is_dir():
            return None

        meta = _read_meta(pack_dir)
        if name is not None:
            meta["name"] = name
        if description is not None:
            meta["description"] = description
        meta.setdefault("created_at", _now())
        meta["updated_at"] = _now()
        _write_meta(
            pack_dir,
            name=meta.get("name") or slug,
            description=meta.get("description") or "",
            created_at=meta["created_at"],
            updated_at=meta["updated_at"],
        )
        return _load_pack(pack_dir, slug)


async def upsert_file(
    data_dir: Path,
    username: str,
    slug: str,
    filepath: str,
    content_b64: str,
) -> KnowledgePack | None:
    """Add or replace a single file inside a pack. ``content_b64`` is the
    base64-encoded raw bytes. Returns the updated pack, or None if the pack
    does not exist. Path validation enforced.
    """
    slug = validate_slug(slug)
    clean = validate_filepath(filepath)
    try:
        body_bytes = base64.b64decode(content_b64 or "", validate=True)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid base64 content: {e}")
    async with _lock:
        pack_dir = _pack_dir(data_dir, username, slug)
        if not pack_dir.is_dir():
            return None

        target = pack_dir.joinpath(*clean.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body_bytes)

        meta = _read_meta(pack_dir)
        meta.setdefault("created_at", _now())
        meta["updated_at"] = _now()
        _write_meta(
            pack_dir,
            name=meta.get("name") or slug,
            description=meta.get("description") or "",
            created_at=meta["created_at"],
            updated_at=meta["updated_at"],
        )
        return _load_pack(pack_dir, slug)


async def delete_file(
    data_dir: Path,
    username: str,
    slug: str,
    filepath: str,
) -> KnowledgePack | None:
    """Remove a file from a pack. Sweeps emptied parent directories up to the
    pack root. Returns the updated pack, or None if the pack or path does not
    exist.
    """
    slug = validate_slug(slug)
    clean = validate_filepath(filepath)
    async with _lock:
        pack_dir = _pack_dir(data_dir, username, slug)
        if not pack_dir.is_dir():
            return None
        target = pack_dir.joinpath(*clean.split("/"))
        if not target.is_file():
            return None
        target.unlink()

        parent = target.parent
        while parent != pack_dir and parent.is_dir():
            try:
                next(parent.iterdir())
                break
            except StopIteration:
                parent.rmdir()
                parent = parent.parent

        meta = _read_meta(pack_dir)
        meta.setdefault("created_at", _now())
        meta["updated_at"] = _now()
        _write_meta(
            pack_dir,
            name=meta.get("name") or slug,
            description=meta.get("description") or "",
            created_at=meta["created_at"],
            updated_at=meta["updated_at"],
        )
        return _load_pack(pack_dir, slug)


async def delete_pack(data_dir: Path, username: str, slug: str) -> bool:
    """Delete a pack directory entirely. Returns True if it existed."""
    slug = validate_slug(slug)
    async with _lock:
        pack_dir = _pack_dir(data_dir, username, slug)
        if not pack_dir.is_dir():
            return False
        shutil.rmtree(pack_dir)
        return True
