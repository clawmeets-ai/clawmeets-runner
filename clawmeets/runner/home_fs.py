# SPDX-License-Identifier: MIT
"""
clawmeets/runner/home_fs.py

Path guard + file executor for the agent home-folder browser.

Every browser / ``clawmeets fs`` operation on an agent's home folder ends
up here, on the runner that hosts the agent. The server authenticates and
authorizes the caller and tells us WHO it is (``CallerKind``); this module
decides WHAT that caller may do to WHICH path, and does it without ever
leaving the home folder.

Safety model (the same on POSIX and Windows; the per-platform primitives
are ``_PosixOps`` here and ``home_fs_win.WindowsOps``):

* Paths are home-relative strings. ``/`` and ``\\`` are both separators, a
  leading separator means the home root. Empty, ``.``, ``..``, NUL /
  control characters, and percent- or Unicode-compatibility-encoded forms
  of any of those are refused before touching the disk. Nothing is ever
  URL-decoded here; the decoded form is only used to *refuse*. On Windows
  the Win32 escape forms are refused too: ``:`` anywhere (drive letters,
  drive-relative paths, alternate data streams), reserved device names
  with or without an extension, names ending in a dot or space, and the
  wildcard / quoting characters. UNC paths and the ``?`` / ``.`` device
  prefixes start with a doubled separator, so they split into an empty
  component and are refused like any other.
* The walk opens the home folder, then every component one at a time with
  ``O_NOFOLLOW`` relative to its parent's descriptor, and runs the final
  operation relative to the last descriptor (``dir_fd=``). A component
  swapped for a symlink mid-operation fails the open instead of being
  followed. Symlinks are listed (never their target) and can only be
  deleted. On Windows every open is ``NtCreateFile`` relative to the
  parent's handle with ``FILE_OPEN_REPARSE_POINT``; junctions and every
  other reparse point are treated exactly like symlinks, and an open whose
  real name differs from the requested one (an 8.3 short name) is refused.
* Regular files with more than one hardlink cannot be read or written
  (``fstat`` on the opened descriptor); deleting the link is allowed.
* Access tiers (``Tier``) are decided on the casefolded, NFKC-normalized
  name AND on identity: every directory opened during the walk, and the
  final file, is compared by ``(st_dev, st_ino)`` against the protected
  entries, so a case/Unicode variant that the filesystem folds onto a
  protected entry is treated as that entry.
* Writes go to a temp file in the same directory and are ``os.replace``d
  over the target, which replaces a symlink rather than following it.
* The home folder's own ``(st_dev, st_ino)`` is pinned when ``HomeFs`` is
  built; an op whose root descriptor no longer matches (home renamed and a
  symlink left in its place) is refused. Nothing on another device is
  entered, read or deleted, so a bind mount or volume inside the home
  folder cannot be used to reach outside it.
* Folder deletes use our own descriptor-based recursive delete. A check
  pass over the whole tree runs first, so a symlinked subfolder, a
  different device, a protected identity anywhere beneath, or a tree deeper
  than ``MAX_DELETE_DEPTH`` normally refuses before anything is removed.
  The delete pass re-applies every check, so a concurrent change between
  the passes can leave a partial delete but can never escape the home
  folder or remove a protected file.

All public methods on ``HomeFs`` are blocking; ``HomeFs.run`` executes one
in a worker thread so the runner's event loop (heartbeats, changelog
pushes) never stalls on a large read or recursive delete.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import secrets
import stat
import sys
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from urllib.parse import unquote


class CallerKind(str, Enum):
    """Who the server says is calling, relative to the target agent."""

    OWNER = "owner"          # the human owner (UI session or their own login)
    SELF = "self"            # the agent operating on its own home folder
    ASSISTANT = "assistant"  # the owner's assistant operating on a peer


class Tier(str, Enum):
    """Access tier of a home-relative path. See ``classify``."""

    SECRET = "secret"            # listed + locked; read by owner/self only
    SYSTEM = "system"            # runner/server managed; read-only for all
    PERSONAL_SKILLS = "personal_skills"  # read by all; written by owner only
    OPEN = "open"                # fully manageable


class FsOp(str, Enum):
    LIST = "list"
    READ = "read"
    WRITE = "write"
    MKDIR = "mkdir"
    DELETE = "delete"


class HomeFsError(Exception):
    """A refused or failed operation. ``code`` is stable and sent on the wire."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# Tier table. Keys are casefolded NFKC names, home-relative.
# ---------------------------------------------------------------------------

# Top-level files holding tokens / API keys. ``card.json`` carries the BYO
# ``llm_api_key``, raw ``model_configs[].api_key`` and the skill/MCP configs
# the locked config files are built from, so it is a secret too.
_SECRET_ROOT_FILES = frozenset({"credential.json", "env.json", "card.json"})
# Subtrees holding tokens / API keys. ``mcp-hub/servers`` holds the per-MCP
# OAuth ``token.json``, ``skill-hub/state`` the per-skill OAuth state and
# browser cookies, and ``agents`` the synced peer cards (with their
# ``skill_configs``) — all secrets in substance even though the plan only
# names ``configs``.
_SECRET_SUBTREES = (
    ("agents",),
    ("mcp-hub", "configs"),
    ("mcp-hub", "servers"),
    ("skill-hub", "configs"),
    ("skill-hub", "state"),
)
_SYSTEM_ROOT_FILES = frozenset({"agent.pid", "agents.md"})
_SYSTEM_ROOT_DIRS = frozenset({
    "metadata", "projects", "system-skill-hub",
    "mcp-hub", "skill-hub", "knowledge_packs",
})
_PERSONAL_SKILLS_DIR = "personal-skill-hub"

# Path / list / size caps. The server applies its own (smaller) wire caps;
# these are the runner's last line of defence.
MAX_PATH_CHARS = 4096
MAX_COMPONENT_BYTES = 255
DEFAULT_MAX_ENTRIES = 2000
DEFAULT_MAX_READ_BYTES = 25 * 1024 * 1024
DEFAULT_MAX_WRITE_BYTES = 25 * 1024 * 1024
# Folder levels a recursive delete descends; each level holds one open fd,
# so this also keeps a delete inside the default per-process fd limit.
MAX_DELETE_DEPTH = 128

_DIR_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
_FILE_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)


@dataclass(frozen=True)
class _Stat:
    """What the guard needs to know about one entry, on either platform."""

    kind: str       # "file" | "dir" | "symlink" (incl. Windows junctions) | "other"
    nlink: int
    dev: int        # st_dev / volume serial number
    ino: int        # st_ino / 128-bit file id
    size: int
    mtime: float
    mode: int = 0o644  # permission bits; only meaningful on POSIX
    reparse: bool = False  # Windows non-link reparse point (cloud placeholder, dedup, compressed)

    @property
    def id(self) -> tuple[int, int]:
        return (self.dev, self.ino)


def _unsupported_entry() -> HomeFsError:
    return HomeFsError("unsupported_entry",
                       "cloud placeholders, deduplicated and compressed reparse files cannot be opened or deleted here")


def _posix_stat(st: os.stat_result) -> _Stat:
    if stat.S_ISLNK(st.st_mode):
        kind = "symlink"
    elif stat.S_ISDIR(st.st_mode):
        kind = "dir"
    elif stat.S_ISREG(st.st_mode):
        kind = "file"
    else:
        kind = "other"
    return _Stat(kind=kind, nlink=st.st_nlink, dev=st.st_dev, ino=st.st_ino,
                 size=st.st_size, mtime=st.st_mtime, mode=stat.S_IMODE(st.st_mode))


class _PosixOps:
    """Descriptor-relative primitives the guard is built from. Directory
    handles are directory fds; file handles are fds (``os.read`` /
    ``os.write`` / ``os.close`` work on both platforms' file handles).
    Nothing here follows a link in the final component."""

    def open_root(self, path: Path) -> int:
        return os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0))

    def open_dir(self, parent: int, name: str) -> int:
        return os.open(name, _DIR_FLAGS, dir_fd=parent)

    def close_dir(self, h: int) -> None:
        os.close(h)

    def stat_dir(self, h: int) -> _Stat:
        return _posix_stat(os.fstat(h))

    def stat_file(self, fd: int) -> _Stat:
        return _posix_stat(os.fstat(fd))

    def lstat(self, parent: int, name: str) -> _Stat:
        return _posix_stat(os.stat(name, dir_fd=parent, follow_symlinks=False))

    def listdir(self, h: int) -> list[str]:
        return os.listdir(h)

    def open_read(self, parent: int, name: str) -> int:
        return os.open(name, _FILE_FLAGS, dir_fd=parent)

    def create_temp(self, parent: int, name: str, mode: int) -> int:
        return os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            mode, dir_fd=parent,
        )

    def replace(self, parent: int, src: str, dst: str) -> None:
        os.replace(src, dst, src_dir_fd=parent, dst_dir_fd=parent)

    def mkdir(self, parent: int, name: str) -> None:
        os.mkdir(name, 0o755, dir_fd=parent)

    def unlink(self, parent: int, name: str) -> None:
        os.unlink(name, dir_fd=parent)

    def rmdir(self, parent: int, name: str) -> None:
        os.rmdir(name, dir_fd=parent)


def _platform_ops():
    if sys.platform == "win32":
        from clawmeets.runner.home_fs_win import WindowsOps
        return WindowsOps()
    return _PosixOps()


def _fold(name: str) -> str:
    return unicodedata.normalize("NFKC", name).casefold()


def _is_log(key: tuple[str, ...]) -> bool:
    """Owner-only logs: runner stdout/stderr at the root, and the LLM CLI
    logs under ``metadata/`` — both can echo tokens and environment."""
    if not key or not key[-1].endswith(".log"):
        return False
    return len(key) == 1 or key[0] == "metadata"


def classify(key: tuple[str, ...]) -> Tier:
    """Tier of a home-relative path given as folded components.

    ``()`` is the home root itself, which is OPEN (its children decide).
    """
    if not key:
        return Tier.OPEN
    head = key[0]
    if len(key) == 1 and head in _SECRET_ROOT_FILES:
        return Tier.SECRET
    for subtree in _SECRET_SUBTREES:
        if key[: len(subtree)] == subtree:
            return Tier.SECRET
    if len(key) == 1 and (head in _SYSTEM_ROOT_FILES or _is_log(key)):
        return Tier.SYSTEM
    if head in _SYSTEM_ROOT_DIRS:
        return Tier.SYSTEM
    if head == _PERSONAL_SKILLS_DIR:
        return Tier.PERSONAL_SKILLS
    return Tier.OPEN


def can(op: FsOp, tier: Tier, caller: CallerKind, *, is_log: bool = False) -> bool:
    """The access matrix (plan: four tiers x three caller kinds)."""
    if op is FsOp.LIST:
        return True
    if op is FsOp.READ:
        if tier is Tier.SECRET:
            return caller in (CallerKind.OWNER, CallerKind.SELF)
        if is_log:
            return caller is CallerKind.OWNER
        return True
    # WRITE / MKDIR / DELETE
    if tier in (Tier.SECRET, Tier.SYSTEM):
        return False
    if tier is Tier.PERSONAL_SKILLS:
        return caller is CallerKind.OWNER
    return True


def _refusal(op: FsOp, tier: Tier, is_log: bool) -> HomeFsError:
    if op is FsOp.READ and tier is Tier.SECRET:
        return HomeFsError("forbidden_secret", "secret files of another agent cannot be read")
    if op is FsOp.READ and is_log:
        return HomeFsError("forbidden_owner_only", "runner logs are readable by the owner only")
    return HomeFsError("read_only", f"{tier.value} paths are read-only for this caller")


# ---------------------------------------------------------------------------
# Path parsing
# ---------------------------------------------------------------------------

def _bad_char(c: str) -> bool:
    # C0/C1 controls, plus format and line/paragraph separators (U+202E
    # RLO, zero-width characters, U+2028) that spoof names in the UI tree.
    return unicodedata.category(c) in ("Cc", "Cf", "Zl", "Zp")


def _bad_component(part: str) -> bool:
    if part in ("", ".", ".."):
        return True
    if any(_bad_char(c) for c in part):
        return True
    if len(part.encode("utf-8", "surrogatepass")) > MAX_COMPONENT_BYTES:
        return True
    # Refuse anything that *decodes* to a dangerous form — we never decode
    # it ourselves, but an upstream layer might have meant to.
    for variant in (unquote(part), unicodedata.normalize("NFKC", part),
                    unicodedata.normalize("NFKC", unquote(part))):
        if variant in (".", "..") or "/" in variant or "\\" in variant:
            return True
        if any(_bad_char(c) for c in variant):
            return True
    return False


# Win32 device names, reserved in every folder with or without an extension
# ("nul.txt" is the NUL device too). The superscript digits are reserved as
# well; NFKC folds them onto the plain ones, but list them for the raw form.
_WIN_RESERVED = frozenset({
    "con", "prn", "aux", "nul", "conin$", "conout$",
    *(f"{d}{n}" for d in ("com", "lpt") for n in "0123456789¹²³"),
})
# ':' covers drive letters, drive-relative paths and alternate data streams.
_WIN_BAD_CHARS = frozenset('<>:"|?*')


def _bad_windows_component(part: str) -> bool:
    for variant in {part, unquote(part), unicodedata.normalize("NFKC", part),
                    unicodedata.normalize("NFKC", unquote(part))}:
        if any(c in _WIN_BAD_CHARS for c in variant):
            return True
        if variant.endswith((".", " ")):
            return True
        if variant.split(".", 1)[0].rstrip(" ").casefold() in _WIN_RESERVED:
            return True
    return False


def parse_home_path(raw: str, *, windows: bool | None = None) -> tuple[str, ...]:
    """Split a wire path into validated components. ``""`` / ``"/"`` = root.

    ``windows`` adds the Win32 refusals; it defaults to the running platform.
    """
    if windows is None:
        windows = sys.platform == "win32"
    if not isinstance(raw, str):
        raise HomeFsError("invalid_path", "path must be a string")
    if len(raw) > MAX_PATH_CHARS:
        raise HomeFsError("invalid_path", "path is too long")
    if "\x00" in raw:
        raise HomeFsError("invalid_path", "path contains NUL")
    norm = raw.replace("\\", "/")
    if norm.startswith("/"):
        norm = norm[1:]
    if not norm:
        return ()
    if norm.endswith("/"):
        norm = norm[:-1]
    parts = tuple(norm.split("/"))
    for part in parts:
        if _bad_component(part) or (windows and _bad_windows_component(part)):
            raise HomeFsError("invalid_path", f"path component {part!r} is not allowed")
    return parts


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FsEntry:
    name: str
    type: str          # "file" | "dir" | "symlink" | "other"
    size: int
    mtime: float
    tier: Tier
    writable: bool     # may THIS caller write/delete it
    readable: bool     # may THIS caller read it (files) / list it (dirs)


@dataclass(frozen=True)
class FsListing:
    path: str
    tier: Tier
    writable: bool
    entries: list[FsEntry] = field(default_factory=list)
    truncated: bool = False


@dataclass(frozen=True)
class FsFile:
    path: str
    tier: Tier
    writable: bool
    size: int
    mtime: float
    content: bytes


@dataclass
class _Walk:
    """Open handles from the home root down to a folder, plus its key and
    the protected-identity map built once for this op from the root handle."""

    ops: object
    key: tuple[str, ...]      # canonical folded key of the deepest folder
    fds: list[int]            # fds[0] is the home root, fds[-1] the deepest folder
    dev: int = 0              # device / volume of the home folder
    identities: dict[tuple[int, int], tuple[str, ...]] = field(default_factory=dict)

    def identity_key(self, st: _Stat, key: tuple[str, ...]) -> tuple[str, ...]:
        canonical = self.identities.get(st.id)
        return canonical if canonical is not None else key

    def close(self) -> None:
        for fd in reversed(self.fds):
            try:
                self.ops.close_dir(fd)
            except OSError:
                pass
        self.fds.clear()


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class HomeFs:
    """Executes guarded file operations inside one agent's home folder.

    ``home`` is the runner's own AGENT_DIR — never a value taken from a
    request.
    """

    def __init__(
        self,
        home: Path,
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_read_bytes: int = DEFAULT_MAX_READ_BYTES,
        max_write_bytes: int = DEFAULT_MAX_WRITE_BYTES,
    ) -> None:
        self.home = Path(home)
        self.max_entries = max_entries
        self.max_read_bytes = max_read_bytes
        self.max_write_bytes = max_write_bytes
        self._ops = _platform_ops()
        self._windows = sys.platform == "win32"
        # Pin the home folder's identity now; every op checks its root
        # handle against it. Intermediate links (macOS /var -> /private/var)
        # are resolved here exactly as the walk's open resolves them.
        self._home_id: tuple[int, int] | None = None
        self._pin_home()

    def _pin_home(self) -> None:
        try:
            h = self._ops.open_root(self.home)
        except OSError:
            return
        try:
            self._home_id = self._ops.stat_dir(h).id
        finally:
            self._ops.close_dir(h)

    def _parse(self, path: str) -> tuple[str, ...]:
        return parse_home_path(path, windows=self._windows)

    # -- public ops -----------------------------------------------------

    async def run(self, op: FsOp, path: str, caller: CallerKind, data: bytes | None = None):
        """Run one op off the event loop."""
        return await asyncio.to_thread(self.execute, op, path, caller, data)

    def execute(self, op: FsOp, path: str, caller: CallerKind, data: bytes | None = None):
        if op is FsOp.LIST:
            return self.list(path, caller)
        if op is FsOp.READ:
            return self.read(path, caller)
        if op is FsOp.WRITE:
            if data is None:
                raise HomeFsError("invalid_request", "write needs content")
            return self.write(path, data, caller)
        if op is FsOp.MKDIR:
            return self.mkdir(path, caller)
        if op is FsOp.DELETE:
            return self.delete(path, caller)
        raise HomeFsError("invalid_request", f"unknown op {op!r}")

    def list(self, path: str, caller: CallerKind) -> FsListing:
        parts = self._parse(path)
        walk = self._walk(parts)
        try:
            fd = walk.fds[-1]
            tier = classify(walk.key)
            # Cut by sorted name so a truncated listing is the same subset
            # on every call.
            names = sorted(self._ops.listdir(fd), key=lambda n: (n.casefold(), n))
            truncated = len(names) > self.max_entries
            entries: list[FsEntry] = []
            for name in names[: self.max_entries]:
                try:
                    entries.append(self._entry(walk, fd, name, walk.key, caller))
                except FileNotFoundError:
                    continue  # deleted between listdir and stat
            entries.sort(key=lambda e: (e.type != "dir", e.name.casefold()))
            return FsListing(
                path="/".join(parts), tier=tier,
                writable=can(FsOp.WRITE, tier, caller),
                entries=entries, truncated=truncated,
            )
        except OSError as exc:
            raise self._os_error(exc) from exc
        finally:
            walk.close()

    def read(self, path: str, caller: CallerKind) -> FsFile:
        parts = self._parse(path)
        if not parts:
            raise HomeFsError("not_a_file", "the home folder is not a file")
        walk = self._walk(parts[:-1])
        fd = -1
        try:
            name = parts[-1]
            pfd = walk.fds[-1]
            st = self._lstat(pfd, name)
            if st.kind == "symlink":
                raise HomeFsError("symlink", "symlinks are never followed")
            if st.kind == "dir":
                raise HomeFsError("not_a_file", "path is a folder")
            fd = self._ops.open_read(pfd, name)
            fst = self._ops.stat_file(fd)
            if fst.kind == "symlink":
                raise HomeFsError("symlink", "symlinks are never followed")
            if fst.kind != "file":
                raise HomeFsError("not_a_file", "only regular files can be read")
            if fst.nlink > 1:
                raise HomeFsError("hardlink", "files with more than one hardlink cannot be read")
            self._same_device(walk, fst)
            key = walk.identity_key(fst, walk.key + (_fold(name),))
            tier = classify(key)
            if not can(FsOp.READ, tier, caller, is_log=_is_log(key)):
                raise _refusal(FsOp.READ, tier, _is_log(key))
            if fst.size > self.max_read_bytes:
                raise HomeFsError("too_large", f"file is larger than {self.max_read_bytes} bytes")
            chunks = []
            remaining = self.max_read_bytes + 1
            while remaining > 0:
                chunk = os.read(fd, min(remaining, 1 << 20))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            content = b"".join(chunks)
            if len(content) > self.max_read_bytes:
                raise HomeFsError("too_large", f"file is larger than {self.max_read_bytes} bytes")
            return FsFile(
                path="/".join(parts), tier=tier, writable=can(FsOp.WRITE, tier, caller),
                size=len(content), mtime=fst.mtime, content=content,
            )
        except OSError as exc:
            raise self._os_error(exc) from exc
        finally:
            if fd >= 0:
                os.close(fd)
            walk.close()

    def write(self, path: str, data: bytes, caller: CallerKind) -> FsEntry:
        parts = self._parse(path)
        if not parts:
            raise HomeFsError("not_a_file", "the home folder is not a file")
        if len(data) > self.max_write_bytes:
            raise HomeFsError("too_large", f"content is larger than {self.max_write_bytes} bytes")
        walk = self._walk(parts[:-1])
        try:
            name = parts[-1]
            pfd = walk.fds[-1]
            key = walk.key + (_fold(name),)
            existing = self._lstat(pfd, name, missing_ok=True)
            mode = 0o644
            if existing is not None:
                if existing.kind == "symlink":
                    raise HomeFsError("symlink", "symlinks can only be deleted")
                if existing.kind != "file":
                    raise HomeFsError("not_a_file", "path exists and is not a regular file")
                if existing.nlink > 1:
                    raise HomeFsError("hardlink", "files with more than one hardlink cannot be written")
                self._same_device(walk, existing)
                key = walk.identity_key(existing, key)
                mode = existing.mode
            tier = classify(key)
            if not can(FsOp.WRITE, tier, caller):
                raise _refusal(FsOp.WRITE, tier, False)
            tmp = f".{name[:200]}.clawmeets-tmp-{secrets.token_hex(6)}"
            tfd = self._ops.create_temp(pfd, tmp, mode)
            try:
                try:
                    view = memoryview(data)
                    while view:
                        n = os.write(tfd, view)
                        view = view[n:]
                    os.fsync(tfd)
                finally:
                    os.close(tfd)
                self._ops.replace(pfd, tmp, name)
            except BaseException:
                try:
                    self._ops.unlink(pfd, tmp)
                except OSError:
                    pass
                raise
            return self._entry(walk, pfd, name, walk.key, caller)
        except OSError as exc:
            raise self._os_error(exc) from exc
        finally:
            walk.close()

    def mkdir(self, path: str, caller: CallerKind) -> FsEntry:
        parts = self._parse(path)
        if not parts:
            raise HomeFsError("exists", "the home folder already exists")
        walk = self._walk(parts[:-1])
        try:
            name = parts[-1]
            pfd = walk.fds[-1]
            tier = classify(walk.key + (_fold(name),))
            if not can(FsOp.MKDIR, tier, caller):
                raise _refusal(FsOp.MKDIR, tier, False)
            try:
                self._ops.mkdir(pfd, name)
            except FileExistsError as exc:
                raise HomeFsError("exists", "a file or folder with that name already exists") from exc
            return self._entry(walk, pfd, name, walk.key, caller)
        except OSError as exc:
            raise self._os_error(exc) from exc
        finally:
            walk.close()

    def delete(self, path: str, caller: CallerKind) -> None:
        parts = self._parse(path)
        if not parts:
            raise HomeFsError("root_delete", "the home folder itself cannot be deleted")
        walk = self._walk(parts[:-1])
        try:
            name = parts[-1]
            pfd = walk.fds[-1]
            st = self._lstat(pfd, name)
            if st.reparse:
                raise _unsupported_entry()
            key = walk.key + (_fold(name),)
            if st.kind != "symlink":
                self._same_device(walk, st)
                key = walk.identity_key(st, key)
            tier = classify(key)
            if not can(FsOp.DELETE, tier, caller):
                raise _refusal(FsOp.DELETE, tier, False)
            if st.kind == "dir":
                # Check the whole tree first so a refusal deletes nothing,
                # then delete with the same checks re-applied per entry.
                self._rmtree(walk, pfd, name, dry_run=True)
                self._rmtree(walk, pfd, name, dry_run=False)
            else:
                self._ops.unlink(pfd, name)
        except OSError as exc:
            raise self._os_error(exc) from exc
        finally:
            walk.close()

    # -- internals ------------------------------------------------------

    def _protected_identities(self, root: int) -> dict[tuple[int, int], tuple[str, ...]]:
        """(dev, ino) -> canonical folded key, for every protected entry
        that currently exists. Looked up from the walk's own root handle
        without following links, so the map and the walk share one home."""
        out: dict[tuple[int, int], tuple[str, ...]] = {}
        ops = self._ops

        def add(dir_fd: int, name: str, key: tuple[str, ...]) -> None:
            try:
                st = ops.lstat(dir_fd, name)
            except (OSError, HomeFsError):
                return
            if st.kind != "symlink":
                out[st.id] = key

        try:
            names = ops.listdir(root)
        except OSError:
            names = []
        roots = {*_SECRET_ROOT_FILES, *_SYSTEM_ROOT_FILES, *_SYSTEM_ROOT_DIRS, _PERSONAL_SKILLS_DIR,
                 *(s[0] for s in _SECRET_SUBTREES)}
        roots |= {_fold(n) for n in names if _fold(n).endswith(".log")}
        for key in roots:
            add(root, self._actual_name(key), (key,))
        for parent in sorted({s[0] for s in _SECRET_SUBTREES if len(s) == 2}):
            try:
                pfd = ops.open_dir(root, parent)
            except (OSError, HomeFsError):
                continue
            try:
                for sub in _SECRET_SUBTREES:
                    if len(sub) == 2 and sub[0] == parent:
                        add(pfd, sub[1], sub)
            finally:
                ops.close_dir(pfd)
        return out

    @staticmethod
    def _actual_name(key: str) -> str:
        # Canonical on-disk spelling of a protected name. The runner creates
        # these with exactly these (lowercase) names, except AGENTS.md.
        return "AGENTS.md" if key == "agents.md" else key

    def _walk(self, parts: tuple[str, ...]) -> _Walk:
        ops = self._ops
        if self._home_id is None:
            # Missing at construction: pin on first sight. A mismatch with an
            # existing pin is never re-pinned (that is home_moved).
            self._pin_home()
            if self._home_id is None:
                raise HomeFsError("not_found", "agent home folder is missing")
        try:
            root = ops.open_root(self.home)
        except OSError as exc:
            raise HomeFsError("not_found", "agent home folder is missing") from exc
        walk = _Walk(ops=ops, key=(), fds=[root])
        try:
            rst = ops.stat_dir(root)
            if rst.id != self._home_id:
                raise HomeFsError("home_moved", "the agent home folder was moved or replaced")
            walk.dev = rst.dev
            walk.identities = self._protected_identities(root)
            for part in parts:
                cur = walk.fds[-1]
                try:
                    fd = ops.open_dir(cur, part)
                except OSError as exc:
                    st = self._lstat(cur, part, missing_ok=True)
                    if st is not None and st.kind == "symlink":
                        raise HomeFsError("symlink", "symlinks are never followed") from exc
                    if st is not None and st.kind != "dir":
                        raise HomeFsError("not_a_directory", f"{part!r} is not a folder") from exc
                    raise self._os_error(exc) from exc
                walk.fds.append(fd)
                fst = ops.stat_dir(fd)
                if fst.kind == "symlink":
                    raise HomeFsError("symlink", "symlinks are never followed")
                if fst.kind != "dir":  # pragma: no cover - O_DIRECTORY guards this
                    raise HomeFsError("not_a_directory", f"{part!r} is not a folder")
                self._same_device(walk, fst)
                walk.key = walk.identity_key(fst, walk.key + (_fold(part),))
            return walk
        except BaseException:
            walk.close()
            raise

    @staticmethod
    def _same_device(walk: _Walk, st: _Stat) -> None:
        if st.dev != walk.dev:
            raise HomeFsError("other_device", "paths on another filesystem (mount, volume) are not reachable")

    def _lstat(self, dir_fd: int, name: str, *, missing_ok: bool = False) -> _Stat | None:
        try:
            return self._ops.lstat(dir_fd, name)
        except FileNotFoundError:
            if missing_ok:
                return None
            raise HomeFsError("not_found", "no such file or folder")
        except OSError as exc:
            raise self._os_error(exc) from exc

    def _entry(self, walk: _Walk, dir_fd: int, name: str, parent_key: tuple[str, ...],
               caller: CallerKind) -> FsEntry:
        st = self._ops.lstat(dir_fd, name)
        key = parent_key + (_fold(name),)
        kind = st.kind
        if kind in ("dir", "file"):
            key = walk.identity_key(st, key)
        tier = classify(key)
        if kind == "symlink":
            readable = False
        elif kind == "file":
            readable = st.nlink <= 1 and st.dev == walk.dev and can(FsOp.READ, tier, caller, is_log=_is_log(key))
        else:
            readable = kind == "dir" and st.dev == walk.dev
        writable = can(FsOp.DELETE, tier, caller) and st.dev == walk.dev and not st.reparse
        if not self._addressable(name):
            # A name the path parser would read as something else (``victim\``
            # on POSIX parses as the folder ``victim``) must not be acted on
            # through its listing — that would aim at a different entry.
            readable = writable = False
        return FsEntry(
            name=name, type=kind, size=st.size if kind == "file" else 0,
            mtime=st.mtime, tier=tier, writable=writable, readable=readable,
        )

    def _addressable(self, name: str) -> bool:
        """True when ``name`` round-trips through the path parser as itself."""
        try:
            return parse_home_path(name, windows=self._windows) == (name,)
        except HomeFsError:
            return False

    def _rmtree(self, walk: _Walk, pfd: int, name: str, *, dry_run: bool) -> None:
        """Handle-based recursive delete of ``name`` under ``pfd``.

        Never follows a link (each folder is opened without following and
        re-checked on its own handle), never leaves the home folder's
        device, and refuses on any protected identity anywhere beneath —
        independent of the name-based tier, so a protected inode moved under
        an open folder is still caught. Walks with an explicit stack (one
        open handle per level) and refuses past ``MAX_DELETE_DEPTH``. With
        ``dry_run`` it only checks; in the delete pass an entry that
        vanished is already deleted.
        """
        ops = self._ops
        # Frames: (handle, name in parent, parent handle, child names still to visit).
        stack: list[tuple[int, str, int, list[str]]] = []

        def enter(parent_fd: int, child: str) -> None:
            if len(stack) >= MAX_DELETE_DEPTH:
                raise HomeFsError("too_deep", f"folder is nested more than {MAX_DELETE_DEPTH} levels deep")
            fd = ops.open_dir(parent_fd, child)
            try:
                st = ops.stat_dir(fd)
                if st.kind != "dir":
                    raise HomeFsError("symlink", "symlinks are never followed")
                self._check_tree_entry(walk, st)
                children = ops.listdir(fd)
            except BaseException:
                ops.close_dir(fd)
                raise
            stack.append((fd, child, parent_fd, children))

        try:
            enter(pfd, name)
            while stack:
                fd, dname, parent_fd, children = stack[-1]
                if not children:
                    stack.pop()
                    ops.close_dir(fd)
                    if not dry_run:
                        with contextlib.suppress(FileNotFoundError):
                            ops.rmdir(parent_fd, dname)
                    continue
                child = children.pop()
                try:
                    st = ops.lstat(fd, child)
                except FileNotFoundError:
                    continue
                if st.reparse:
                    raise _unsupported_entry()
                if st.kind != "symlink":
                    self._check_tree_entry(walk, st)
                if st.kind == "dir":
                    try:
                        enter(fd, child)
                    except FileNotFoundError:
                        continue
                elif not dry_run:
                    with contextlib.suppress(FileNotFoundError):
                        ops.unlink(fd, child)
        finally:
            for fd, *_ in stack:
                ops.close_dir(fd)

    def _check_tree_entry(self, walk: _Walk, st: _Stat) -> None:
        self._same_device(walk, st)
        if st.id in walk.identities:
            raise HomeFsError("read_only", "folder contains protected files")

    @staticmethod
    def _os_error(exc: OSError) -> HomeFsError:
        import errno
        if exc.errno == errno.ENOENT:
            return HomeFsError("not_found", "no such file or folder")
        if exc.errno == errno.ELOOP:
            return HomeFsError("symlink", "symlinks are never followed")
        if exc.errno == errno.ENOTDIR:
            return HomeFsError("not_a_directory", "a path component is not a folder")
        if exc.errno == errno.EISDIR:
            return HomeFsError("not_a_file", "path is a folder")
        if exc.errno == errno.EEXIST:
            return HomeFsError("exists", "a file or folder with that name already exists")
        if exc.errno == errno.ENOTEMPTY:
            return HomeFsError("not_empty", "folder is not empty")
        if exc.errno in (errno.EACCES, errno.EPERM):
            return HomeFsError("os_permission", "the operating system refused access")
        return HomeFsError("io_error", f"{exc.strerror or exc}")
