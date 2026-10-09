# SPDX-License-Identifier: MIT
"""
clawmeets/models/file_cache.py

Stat-validated cache of parsed JSON files.

The server answers "which projects is this agent in?" and "who is on the
roster?" by scanning every project ``meta.json`` (about 1,300) and every agent
``card.json``. Parsing them all on every request is what made a reconnect storm
expensive: 228 runners catching up meant ~300,000 JSON parses. This cache keeps
each parsed file keyed by ``(mtime_ns, size, inode)`` and re-parses only when
that stat moves, so a scan where nothing changed costs one ``stat`` per file.

Why a stat check instead of invalidation hooks on every writer: project
participants and card fields are written from many places (create,
add-participant, status flips, settings, maintenance scripts, hand edits). A
hook can miss one of them; a stat cannot. Every writer in the codebase goes
through ``FileUtil.write`` (temp file + rename), so a write always changes the
inode and a stale hit is impossible even within one mtime tick.

The stat is taken BEFORE the read, so the stored key is never newer than the
stored content: a write racing the read shows up as a key mismatch on the next
call and is re-parsed then.

Callers get the SAME dict object on every hit and must treat it as read-only.
Copy it (or ``model_validate`` it, which builds fresh containers) before
mutating.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

from clawmeets.utils.file_io import FileUtil

_StatKey = tuple[int, int, int]


class ParsedFileCache:
    """Process-wide cache of parsed JSON dicts, validated by ``stat`` on read."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[_StatKey, dict]] = {}
        # Number of actual JSON parses. Tests use it to prove that a scan with
        # nothing changed parses nothing.
        self.parses = 0

    def read_json(self, path: Union[str, Path]) -> Optional[dict]:
        """Parsed contents of ``path``, or None if it is missing or not a JSON object.

        Re-parses only when the file's ``(mtime_ns, size, inode)`` moved since
        the last read. Thread-safe in the sense that matters here: concurrent
        callers can at worst parse the same file twice.

        Accepts a plain ``str`` so a scan over thousands of files can skip
        building a ``Path`` per file, which costs more than the ``stat`` itself.
        """
        path = os.fspath(path)
        try:
            st = os.stat(path)
        except OSError:
            self._entries.pop(path, None)
            return None
        key = (st.st_mtime_ns, st.st_size, st.st_ino)
        hit = self._entries.get(path)
        if hit is not None and hit[0] == key:
            return hit[1]
        data = FileUtil.read(Path(path), "json")
        self.parses += 1
        if not isinstance(data, dict):
            self._entries.pop(path, None)
            return None
        self._entries[path] = (key, data)
        return data
