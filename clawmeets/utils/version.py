# SPDX-License-Identifier: MIT
"""
clawmeets/utils/version.py

Best-effort resolution of the installed ``clawmeets`` distribution version.

Used by the runner to report which clawmeets build it is executing in its
WebSocket handshake (see ``cli_runner._runner_loop``). The server stores the
reported string per live connection and relays it, unexamined, to the owning
user's browser in the ``RUNNER_VERSIONS`` push. The server does NOT match it
against anything — announcement version matching is entirely client-side
(``src/utils/versionMatch.ts``), and there is deliberately no second matcher
in Python.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("clawmeets")


def installed_clawmeets_version() -> str | None:
    """Return the installed ``clawmeets`` version, or ``None`` if undeterminable.

    NEVER raises, and never returns the string ``"unknown"``. ``None`` is the
    single, unambiguous "undeterminable" signal on the wire: the announcement
    matcher's unknown-version branch keys off a null/unparseable version, so a
    sentinel STRING like ``"unknown"`` would have to be special-cased in two
    places and would sort as a version candidate.

    Returns ``None`` when the distribution is not installed (an editable /
    source checkout raises ``PackageNotFoundError``) or metadata is unreadable
    for any other reason.

    ACCEPTED LIMITATION — monorepo editable installs report the root version.
    Both ``pyproject.toml`` files in this repo declare ``name = "clawmeets"``:
    the root one carries a placeholder version, while
    ``packaging/runner/pyproject.toml`` is the authoritative declaration of the
    *released* runner version. A runner started from a monorepo editable
    install therefore reports the root's placeholder rather than the release,
    which is old enough to match an "upgrade" announcement. Arguably correct —
    a source checkout genuinely is not a release — but it will confuse someone,
    so it is documented here rather than worked around. Do not "fix" this by
    hard-coding a release number anywhere; the numbers live in
    ``packaging/runner/pyproject.toml`` and nowhere else.

    Note the pre-existing ``_version_callback`` in ``clawmeets/cli.py`` and
    ``packaging/runner/cli.py`` DOES echo ``"unknown"`` — that is a separate,
    human-display concern and is deliberately not reused here.
    """
    try:
        from importlib.metadata import version

        return version("clawmeets")
    except Exception as e:
        logger.debug(f"Could not determine installed clawmeets version: {e}")
        return None
