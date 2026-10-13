# SPDX-License-Identifier: MIT
"""
clawmeets/models/agent_bundle.py

Install a specialist pool agent's files into its local agent home.

A pool entry in a setup.json carries a ``bundle``: the agent's folder
(``dir``, relative to the setup.json) and every file to install with its
sha256. ``clawmeets agent-team register`` fetches the bundle after writing the
agent's card and places each file where the runner already reads it:

    role.md            -> knowledge/CLAUDE.md       (the role)
    learnings/...      -> memory/learnings/...      (field knowledge)
    knowledge/...      -> knowledge/...             (reference material)
    skills/...         -> personal-skill-hub/skills/...  (private skills)

Nothing else is installable: evals/ and SOURCES.md stay in the pool.

Re-installing never deletes. A file whose current content differs from what
the last install wrote (recorded in ``pool-bundle.json``) was changed locally,
by the user or by the agent itself, and is kept and reported unless
``overwrite_local``. ``learnings/INDEX.md`` is always merged instead of
replaced, because reflect and personalize keep adding topics to it.

Imports stay stdlib + httpx + ``clawmeets.utils.file_io`` so the module is safe
to ship in the runner wheel (``models/`` is copied wholesale).
"""
from __future__ import annotations

import json
import re
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

from ..utils.file_io import FileUtil

# Bundle path (exact file or "<dir>/" prefix) -> place in the agent home.
BUNDLE_ROUTES: dict[str, str] = {
    "role.md": "knowledge/CLAUDE.md",
    "learnings/": "memory/learnings/",
    "knowledge/": "knowledge/",
    "skills/": "personal-skill-hub/skills/",
}

# Written last into the agent home: bundle path -> sha256 of what was placed.
INSTALL_RECORD = "pool-bundle.json"

# Merged line by line rather than replaced.
LEARNINGS_INDEX = "learnings/INDEX.md"

# The role owns the knowledge dir's instruction-file slot: no other bundle
# path may land there (an AGENTS.md beside it would also shadow it for tools
# that prefer AGENTS.md). Compared case-insensitively.
RESERVED_DESTINATIONS = frozenset({"knowledge/agents.md", "knowledge/claude.md", "knowledge/gemini.md"})

_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


class BundleError(Exception):
    """A bundle that cannot be installed as listed; nothing was written."""


@dataclass
class InstallReport:
    written: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    overwritten_local_edits: list[str] = field(default_factory=list)
    kept_local_edits: list[str] = field(default_factory=list)
    merged: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [
            f"{len(self.written)} written",
            f"{len(self.unchanged)} unchanged",
        ]
        if self.merged:
            parts.append(f"{len(self.merged)} merged")
        if self.overwritten_local_edits:
            parts.append(f"{len(self.overwritten_local_edits)} local edits overwritten")
        if self.kept_local_edits:
            parts.append(
                f"{len(self.kept_local_edits)} local edits kept "
                f"({', '.join(self.kept_local_edits)}; --overwrite-local replaces them)"
            )
        return ", ".join(parts)


def check_relpath(path: str) -> str:
    """Return ``path`` if it is a plain relative POSIX path, else raise.
    Refuses empty segments, any segment starting with "." (so "..", ".git"),
    a leading "/", a backslash, a drive letter and control characters."""
    if not isinstance(path, str) or not path:
        raise BundleError(f"bundle path {path!r} is empty")
    if _CONTROL_RE.search(path):
        raise BundleError(f"bundle path {path!r} contains a control character")
    if "\\" in path or path.startswith("/") or _DRIVE_RE.match(path):
        raise BundleError(f"bundle path {path!r} is not a relative POSIX path")
    if any(not seg or seg.startswith(".") for seg in path.split("/")):
        raise BundleError(f"bundle path {path!r} has an empty or dot segment")
    return path


def destination(path: str) -> str:
    """Where bundle path ``path`` lands, relative to the agent home."""
    check_relpath(path)
    for source, dest in BUNDLE_ROUTES.items():
        if source.endswith("/"):
            if path.startswith(source) and len(path) > len(source):
                return dest + path[len(source):]
        elif path == source:
            return dest
    raise BundleError(f"bundle path {path!r} is not installable (allowed: {', '.join(BUNDLE_ROUTES)})")


def template_base_of(url_or_path: str) -> str:
    """The location a setup.json's relative ``bundle.dir`` resolves against:
    the parent URL of an http(s) URL, else the parent directory."""
    if url_or_path.startswith(("http://", "https://")):
        return url_or_path.rsplit("/", 1)[0]
    if url_or_path.startswith("file://"):
        url_or_path = url_or_path[len("file://"):]
    return str(Path(url_or_path).expanduser().resolve().parent)


def check_bundle(bundle: dict) -> tuple[str, list[tuple[str, str]]]:
    """Validate a setup.json ``bundle`` without fetching anything and return
    ``(dir, [(path, sha256)])``. Raises BundleError when any path is unsafe or
    not installable, has no valid sha, or would land on the same file as
    another path or on the role's slot (destinations compared
    case-insensitively, since macOS and Windows file systems are)."""
    if not isinstance(bundle, dict):
        raise BundleError("bundle must be an object")
    bundle_dir = check_relpath(bundle.get("dir") or "")
    files = bundle.get("files")
    if not isinstance(files, list) or not files:
        raise BundleError("bundle.files must be a non-empty list")
    entries: list[tuple[str, str]] = []
    landed: dict[str, str] = {}
    for item in files:
        path = (item or {}).get("path")
        sha = (item or {}).get("sha256")
        dest = destination(path).casefold()
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise BundleError(f"bundle file {path!r} has no valid sha256")
        if dest in RESERVED_DESTINATIONS and path != "role.md":
            raise BundleError(f"bundle file {path!r} would replace the role; put the role in role.md")
        if dest in landed:
            raise BundleError(f"bundle files {landed[dest]!r} and {path!r} install to the same file")
        landed[dest] = path
        entries.append((path, sha))
    return bundle_dir, entries


def bundle_digest(files: list[dict]) -> str:
    """One sha256 over a bundle's ``[{path, sha256}]`` list, independent of
    order. The eval report records it, so a results file can be tied to the
    exact files it evaluated."""
    lines = sorted(f"{f['path']}\0{f['sha256']}" for f in files)
    return FileUtil.sha256("\n".join(lines).encode("utf-8"))


def fetch_bundle(
    bundle: dict, template_base: str, *, http: Optional[httpx.Client] = None,
) -> dict[str, bytes]:
    """Fetch every listed file and verify its sha256. Fetches all files before
    returning, so a single bad file means nothing gets installed. Never reads
    anything the bundle does not list."""
    bundle_dir, entries = check_bundle(bundle)
    remote = template_base.startswith(("http://", "https://"))
    if not remote:
        root = (Path(template_base) / bundle_dir).resolve()
    out: dict[str, bytes] = {}
    for path, sha in entries:
        if remote:
            url = "/".join([
                template_base.rstrip("/"),
                urllib.parse.quote(bundle_dir),
                urllib.parse.quote(path),
            ])
            try:
                if http is None:
                    resp = httpx.get(url, timeout=30, follow_redirects=True)
                else:
                    resp = http.get(url, timeout=30, follow_redirects=True)
                resp.raise_for_status()
            except httpx.HTTPError as e:
                raise BundleError(f"could not fetch {path!r}: {e}") from e
            data = resp.content
        else:
            source = (root / path).resolve()
            if not source.is_relative_to(root):
                raise BundleError(f"bundle file {path!r} leads outside {bundle_dir}/")
            try:
                data = source.read_bytes()
            except OSError as e:
                raise BundleError(f"could not read {path!r}: {e}") from e
        if FileUtil.sha256(data) != sha:
            raise BundleError(f"bundle file {path!r} does not match its sha256")
        out[path] = data
    return out


def _merge_index(local: str, bundle: str) -> str:
    """``local`` with every non-blank bundle line it lacks appended, in order."""
    have = {line.strip() for line in local.splitlines() if line.strip()}
    missing = [line for line in bundle.splitlines() if line.strip() and line.strip() not in have]
    if not missing:
        return local
    sep = "" if not local or local.endswith("\n") else "\n"
    return local + sep + "\n".join(missing) + "\n"


def _resolved_destinations(files: dict[str, bytes], agent_dir: Path) -> dict[str, Path]:
    home = agent_dir.resolve()
    dests: dict[str, Path] = {}
    for path in sorted(files):
        dest = (agent_dir / destination(path)).resolve()
        if not dest.is_relative_to(home):
            raise BundleError(f"bundle file {path!r} would land outside the agent home")
        dests[path] = dest
    return dests


def installed_pool(agent_dir: Path) -> Optional[str]:
    """The pool named in ``agent_dir``'s install record, or None when there is
    no record or it names no pool. The takeover guard trusts only this."""
    record = FileUtil.read(agent_dir / INSTALL_RECORD, format="json", default={}) or {}
    pool = record.get("pool") if isinstance(record, dict) else None
    return pool if isinstance(pool, str) and pool else None


def install_bundle(
    files: dict[str, bytes], agent_dir: Path, *, pool: str, overwrite_local: bool = False,
) -> InstallReport:
    """Place fetched bundle files into ``agent_dir`` (see module docstring).
    Every destination is checked before anything is written; the install
    record, which names ``pool``, is written last, so an interrupted install
    is safe to re-run."""
    if not pool:
        raise BundleError("an install needs the pool it comes from")
    dests = _resolved_destinations(files, agent_dir)
    record_path = agent_dir / INSTALL_RECORD
    previous: Optional[dict[str, str]] = None
    if record_path.exists():
        previous = dict((FileUtil.read(record_path, format="json", default={}) or {}).get("files") or {})
    record = dict(previous or {})
    report = InstallReport()

    for path, dest in dests.items():
        data = files[path]
        new_sha = FileUtil.sha256(data)
        if not dest.exists():
            FileUtil.write(dest, data, format="bytes")
            report.written.append(path)
            record[path] = new_sha
            continue
        current = dest.read_bytes()
        if path == LEARNINGS_INDEX:
            local_text = current.decode("utf-8", errors="replace")
            merged = _merge_index(local_text, data.decode("utf-8", errors="replace"))
            if merged == local_text:
                report.unchanged.append(path)
            else:
                FileUtil.write(dest, merged, format="text")
                report.merged.append(path)
            record[path] = new_sha
            continue
        current_sha = FileUtil.sha256(current)
        if current_sha == new_sha:
            report.unchanged.append(path)
            record[path] = new_sha
            continue
        # First install: the bundle owns every path it lists.
        changed_locally = previous is not None and previous.get(path) != current_sha
        if changed_locally and not overwrite_local:
            report.kept_local_edits.append(path)
            continue
        FileUtil.write(dest, data, format="bytes")
        (report.overwritten_local_edits if changed_locally else report.written).append(path)
        record[path] = new_sha

    FileUtil.write(record_path, {"pool": pool, "files": dict(sorted(record.items()))}, format="json")
    return report
