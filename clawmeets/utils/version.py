# SPDX-License-Identifier: MIT
"""
clawmeets/utils/version.py

Best-effort resolution of the installed ``clawmeets`` distribution version.

Used by the runner to report which clawmeets build it is executing in its
WebSocket handshake (see ``cli_runner._runner_loop``). The server stores the
reported string per live connection and later matches it against announcement
version specs.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("clawmeets")


def installed_clawmeets_version() -> str | None:
    """Return the installed ``clawmeets`` version, or ``None`` if undeterminable.

    NEVER raises, and never returns the string ``"unknown"``. ``None`` is the
    single, unambiguous "undeterminable" signal on the wire: the announcement
    matcher's unknown-version branch keys off ``is None``, so a sentinel STRING
    like ``"unknown"`` would have to be special-cased in two places and would
    sort as a version candidate.

    Returns ``None`` when the distribution is not installed (an editable /
    source checkout raises ``PackageNotFoundError``) or metadata is unreadable
    for any other reason.

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
