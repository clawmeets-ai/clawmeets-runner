# SPDX-License-Identifier: MIT
"""
clawmeets/models/skill_version.py

The content version of a hub skill. The server advertises it in
``GET /agents/{id}/skills``, ``GET /skills/{name}`` and ``SKILL_SYNC``; the runner
records the version it installed (``skill-hub/versions.json``) and catch-up
re-downloads any skill whose server version differs, so a skill edited on the
server reaches runners that already had the old copy.

The runner records what the server said rather than re-hashing its own folder:
a skill may write files next to its SKILL.md, and re-hashing would treat that
as drift and wipe them on every catch-up.
"""
from __future__ import annotations

import hashlib


def skill_version(skill_md: str, files: dict[str, bytes]) -> str:
    """16-hex-digit hash over ``SKILL.md`` text and every sibling file.

    Sibling files are raw bytes, in sorted path order.
    """
    h = hashlib.sha256()
    h.update(b"SKILL.md\0")
    h.update(skill_md.encode("utf-8"))
    for relpath in sorted(files):
        h.update(b"\0" + relpath.encode("utf-8") + b"\0")
        h.update(files[relpath])
    return h.hexdigest()[:16]
