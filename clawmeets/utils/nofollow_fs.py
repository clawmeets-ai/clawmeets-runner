# SPDX-License-Identifier: MIT
"""
clawmeets/utils/nofollow_fs.py

File operations under a directory that someone else can also write, which
never follow a symlink found below it.

The runner writes, links and removes files in an agent's sandbox before each
turn (prompt file, MCP config link, skill tree, stale configs). A scoped
turn's sandbox is writable by that turn and persists, so whatever it left
there (``.agent-prompt.txt -> ~/.zshrc``, ``.claude -> ~/.claude``) must not
be followed by the runner, which runs as the owner and unsandboxed.

Every function takes a trusted ``root`` (whose parent the turn cannot write)
and a ``rel`` path below it. ``root`` itself is opened with ``O_NOFOLLOW`` and,
given a ``pin`` (its ``(st_dev, st_ino)`` from :func:`pin_dir`, taken at turn
start), must still be that directory. Each component below ``root`` is opened relative
to its parent's descriptor with ``O_NOFOLLOW``, so a component swapped for a
symlink mid-operation fails the open instead of being followed. A symlinked
directory component is unlinked by a write or remove (the link itself, never
its target) and, when the operation needs the directory, replaced by a real
one; a read finds nothing there and changes nothing. Writes go to a
fresh temp file that is ``os.replace``d over the target, which replaces a
symlink rather than writing through it. Platforms without ``dir_fd`` support
(Windows) fall back to the same checks on paths.
"""
from __future__ import annotations

import os
import secrets
import shutil
import stat
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Iterator, Mapping, Optional

_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_DIR_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | _NOFOLLOW | _CLOEXEC
_USE_FD = (
    {os.open, os.stat, os.mkdir, os.unlink, os.symlink, os.rename} <= os.supports_dir_fd
    and shutil.rmtree.avoids_symlink_attacks
)


class RootMoved(OSError):
    """``root`` is a symlink, not a directory, or not the pinned directory."""


def pin_dir(root: Path) -> tuple[int, int]:
    """``(st_dev, st_ino)`` of ``root``, which must be a real directory (not a
    symlink to one). Pass it as ``pin=`` to later calls on the same root."""
    st = os.stat(root, follow_symlinks=False)
    if not stat.S_ISDIR(st.st_mode):
        raise RootMoved(f"not a real directory: {root}")
    return (st.st_dev, st.st_ino)


def _open_root(root: Path, pin: Optional[tuple[int, int]]) -> "_Dir":
    if not _USE_FD:
        if os.path.islink(root) or not os.path.isdir(root):
            raise RootMoved(f"not a real directory: {root}")
        d = _Dir(root, None)
        st = os.stat(root)
    else:
        try:
            d = _Dir(root, os.open(root, _DIR_FLAGS))
        except OSError as exc:
            raise RootMoved(f"not a real directory: {root}") from exc
        st = os.fstat(d.fd)
    if pin is not None and (st.st_dev, st.st_ino) != tuple(pin):
        d.close()
        raise RootMoved(f"not the directory pinned at turn start: {root}")
    return d


class _Dir:
    """One directory reached without following a link: by descriptor where
    the platform has ``*at()`` calls, by path otherwise."""

    def __init__(self, path: Path, fd: Optional[int]) -> None:
        self.path = path
        self.fd = fd

    def _at(self, name: str) -> tuple[str, dict]:
        if self.fd is None:
            return str(self.path / name), {}
        return name, {"dir_fd": self.fd}

    def lstat(self, name: str) -> Optional[os.stat_result]:
        p, kw = self._at(name)
        try:
            return os.stat(p, follow_symlinks=False, **kw)
        except FileNotFoundError:
            return None

    def child(self, name: str) -> "_Dir":
        p, kw = self._at(name)
        if self.fd is None:
            if os.path.islink(p) or not os.path.isdir(p):
                raise NotADirectoryError(p)
            return _Dir(Path(p), None)
        return _Dir(self.path / name, os.open(p, _DIR_FLAGS, **kw))

    def mkdir(self, name: str) -> None:
        p, kw = self._at(name)
        os.mkdir(p, 0o755, **kw)

    def remove(self, name: str) -> None:
        """Remove ``name``: a real directory recursively, anything else
        (a symlink included) by unlinking the entry itself."""
        st = self.lstat(name)
        if st is None:
            return
        p, kw = self._at(name)
        if stat.S_ISDIR(st.st_mode):
            shutil.rmtree(p, **kw)
        else:
            os.unlink(p, **kw)

    def symlink(self, target: Path, name: str) -> None:
        p, kw = self._at(name)
        os.symlink(target, p, **kw)

    def open(self, name: str, flags: int, mode: int = 0o600) -> int:
        p, kw = self._at(name)
        return os.open(p, flags | _NOFOLLOW | _CLOEXEC, mode, **kw)

    def replace(self, src: str, dst: str) -> None:
        if self.fd is None:
            os.replace(self.path / src, self.path / dst)
        else:
            os.replace(src, dst, src_dir_fd=self.fd, dst_dir_fd=self.fd)

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)


def _parts(rel: str | Path) -> list[str]:
    parts = list(PurePosixPath(str(rel).replace(os.sep, "/")).parts)
    if not parts or any(p in ("", ".", "..", "/") for p in parts):
        raise ValueError(f"not a plain relative path: {rel!r}")
    return parts


@contextmanager
def _walk(root: Path, parts: list[str], *, create: bool, prune: bool = True,
          pin: Optional[tuple[int, int]] = None) -> Iterator[Optional[_Dir]]:
    """Yield the directory ``root/parts`` reached without following a link.

    A symlink found on the way is unlinked, unless ``prune`` is off (reads,
    which change nothing and yield ``None`` there). With ``create`` a missing or
    unlinked component becomes a real directory; without it the walk yields
    ``None`` (nothing real lives under there to operate on). A ``root`` that
    is a symlink or not the ``pin``ned directory is treated the same way:
    :class:`RootMoved` with ``create``, ``None`` without.
    """
    root = Path(root)
    if create:
        root.mkdir(parents=True, exist_ok=True)
        d = _open_root(root, pin)
    else:
        try:
            d = _open_root(root, pin)
        except (RootMoved, FileNotFoundError):
            yield None
            return
    opened = [d]
    try:
        for name in parts:
            st = d.lstat(name)
            if st is not None and stat.S_ISLNK(st.st_mode):
                if not prune:
                    yield None
                    return
                d.remove(name)
                st = None
            if st is None:
                if not create:
                    yield None
                    return
                d.mkdir(name)
            elif not stat.S_ISDIR(st.st_mode):
                if not create:
                    yield None
                    return
                raise NotADirectoryError(str(d.path / name))
            d = d.child(name)
            opened.append(d)
        yield d
    finally:
        for o in opened:
            o.close()


def write_text(root: Path, rel: str | Path, text: str, *, pin: Optional[tuple[int, int]] = None) -> Path:
    """Write ``root/rel``, replacing (never writing through) a symlink there."""
    *parents, name = _parts(rel)
    with _walk(root, parents, create=True, pin=pin) as d:
        assert d is not None
        tmp = f".{name}.{secrets.token_hex(6)}.tmp"
        fd = d.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            d.replace(tmp, name)
        except BaseException:
            d.remove(tmp)
            raise
    return Path(root).joinpath(*parents, name)


def read_text(root: Path, rel: str | Path, *, pin: Optional[tuple[int, int]] = None,
              errors: str = "strict") -> Optional[str]:
    """``root/rel`` as text, or ``None`` if it is missing or is not a regular
    file reached without a symlink."""
    *parents, name = _parts(rel)
    with _walk(root, parents, create=False, prune=False, pin=pin) as d:
        if d is None:
            return None
        try:
            fd = d.open(name, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
        except OSError:
            return None
        with os.fdopen(fd, "rb") as f:
            if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                return None
            return f.read().decode("utf-8", errors)


def list_dir(root: Path, rel: str | Path = "", *,
             pin: Optional[tuple[int, int]] = None) -> Optional[list[tuple[str, bool]]]:
    """``(name, is_dir)`` for each entry of ``root/rel`` (``""`` = ``root``),
    or ``None`` if it is not a directory reached without a symlink. A symlink
    entry is listed, never followed (``is_dir`` False)."""
    parts = _parts(rel) if str(rel) not in ("", ".") else []
    with _walk(root, parts, create=False, prune=False, pin=pin) as d:
        if d is None:
            return None
        names = os.listdir(d.fd if d.fd is not None else d.path)
        out = []
        for name in names:
            st = d.lstat(name)
            out.append((name, st is not None and stat.S_ISDIR(st.st_mode)))
        return out


def remove(root: Path, rel: str | Path, *, pin: Optional[tuple[int, int]] = None) -> None:
    """Remove ``root/rel`` if present. A symlink on the way (the target or a
    parent directory) is unlinked itself; nothing outside ``root`` is touched."""
    *parents, name = _parts(rel)
    with _walk(root, parents, create=False, pin=pin) as d:
        if d is not None:
            d.remove(name)


def symlink(root: Path, rel: str | Path, target: Path) -> Path:
    """Make ``root/rel`` a symlink to ``target``, replacing whatever is there."""
    *parents, name = _parts(rel)
    with _walk(root, parents, create=True) as d:
        assert d is not None
        d.remove(name)
        d.symlink(target, name)
    return Path(root).joinpath(*parents, name)


def link_dir(root: Path, rel: str | Path, links: Mapping[str, Path]) -> Path:
    """Recreate ``root/rel`` as a fresh real directory holding one
    ``name -> target`` symlink per entry of ``links``."""
    parts = _parts(rel)
    remove(root, rel)
    with _walk(root, parts, create=True) as d:
        assert d is not None
        for name, target in links.items():
            if len(_parts(name)) != 1:
                raise ValueError(f"not a single path component: {name!r}")
            d.symlink(target, name)
    return Path(root).joinpath(*parts)
