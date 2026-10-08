# SPDX-License-Identifier: MIT
"""
clawmeets/models/install_token.py

The one-time token in the install command, and the progress the machine reports
back while spending it.

## What it is for

Signing up used to be followed by six commands, one of which typed a password
into a terminal. This record replaces all of it: the web app mints a token, the
user pastes one line, and the script on their machine exchanges the token for a
real session — so no password is ever typed into a shell and no code is carried
from a browser to a terminal.

It is a magic link, and it is treated like one. Holding the token is enough to
obtain a session for that account, so the window is short (one hour) and the
session exchange is SINGLE USE.

## Why the token has two parts

The wire form is ``<id>.<secret>``. The id is a public handle, is the filename,
and is what a progress report is addressed to; only the secret's SHA-256 is
stored. That split is what lets the server find the record in one file open
without the raw secret ever touching disk — the same choice
:mod:`clawmeets.models.host` makes for host tokens, for the same reason.

## Why progress outlives the claim

The session exchange is single-use, but the token keeps working for progress
reports until it expires. The steps are the whole point of the live checklist:
the web app has to be able to say "installing…", "signing in…", "your computer
is connected" while it happens, and a token that died at the claim would leave
the page blind for the rest of the install — exactly the silence this work
exists to remove.

Progress is REPORTED BY THE MACHINE and is therefore a narrative, not proof.
Nothing here decides whether the setup works; that is
:mod:`clawmeets.doctor` (locally) and the server's own observations of a live
socket (in the browser). These steps say what the script is doing, and the
checklist rows stay driven by what the server actually sees.

Storage::

    {data_dir}/install_tokens/
      <owner_user_id>/
        <token_id>.json

Owner-scoped directories so "show me this user's install progress" is one
``iterdir`` of the caller's own folder, matching ``hosts/`` and ``brief_tab``.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
import secrets
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from clawmeets.utils.file_io import FileUtil

_lock = asyncio.Lock()

INSTALL_TOKENS_DIR = "install_tokens"

# One hour. Short on purpose, and shorter than it could be: this token buys a
# session for the account, so its lifetime is the window in which a shoulder-surf
# or a shared screenshot is worth anything. The cost of being strict is low
# because the web app mints a fresh one every time the page loads — the command
# on screen is always live, and an expired one is told to refresh the page.
INSTALL_TOKEN_TTL_SECONDS = 60 * 60

# Bounds on what a machine may report, so a buggy or hostile script cannot grow
# a record without limit. Exceeding the step cap drops the oldest steps rather
# than refusing the report: the newest progress is the interesting progress.
MAX_STEPS = 40
MAX_DETAIL_CHARS = 500

_TOKEN_ID_RE = re.compile(r"^[0-9a-f]{12}$")

StepState = Literal["started", "ok", "failed", "skipped"]

# The steps the install script reports, in the order it performs them. Declared
# here rather than left to the script so the web app can render the whole list
# greyed out BEFORE the machine says anything — a checklist that grows a row at a
# time reads as a stall, not as progress.
INSTALL_STEPS: tuple[tuple[str, str], ...] = (
    ("install", "Installing ClawMeets"),
    ("link", "Signing in on this computer"),
    ("assistant", "Setting up your assistant"),
    ("providers", "Connecting your model providers"),
    ("computer", "Connecting this computer"),
    ("start", "Starting your agents"),
    ("check", "Checking everything works"),
)

INSTALL_STEP_KEYS: frozenset[str] = frozenset(key for key, _ in INSTALL_STEPS)
INSTALL_STEP_LABELS: dict[str, str] = dict(INSTALL_STEPS)


class InstallStep(BaseModel):
    """One thing the script on the machine says it did.

    ``state`` is the machine's claim. ``failed`` carries the reason in
    ``detail``, which is the only place a user learns why an install stopped
    halfway without reading a terminal they may have already closed.
    """

    key: str
    state: StepState = "started"
    detail: str = ""
    at: str = ""


class InstallToken(BaseModel):
    """One run of the install command, from minting to the last report.

    ``secret_hash`` never leaves this process — :meth:`to_wire` drops it, and
    that is the only shape any route returns. The raw secret exists once, in the
    response to the mint call.
    """

    id: str
    owner_user_id: str
    username: str
    secret_hash: str
    created_at: str
    expires_at: str
    # When the session exchange was spent. Set once; a second claim is refused.
    claimed_at: Optional[str] = None
    steps: list[InstallStep] = Field(default_factory=list)

    @property
    def is_expired(self) -> bool:
        expires = _parse_iso(self.expires_at)
        return expires is None or expires < datetime.now(UTC)

    @property
    def is_claimed(self) -> bool:
        return bool(self.claimed_at)

    def to_wire(self) -> dict:
        """The shape every route returns. No ``secret_hash``, ever."""
        data = self.model_dump(exclude={"secret_hash"})
        data["expired"] = self.is_expired
        data["claimed"] = self.is_claimed
        return data


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_iso(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def validate_token_id(token_id: str) -> str:
    """Guard the path segment before it becomes a filename.

    Ids are server-minted 12-hex strings, so anything else is a typo or a
    traversal attempt; either way it is not a token.
    """
    cleaned = (token_id or "").strip().lower()
    if not _TOKEN_ID_RE.match(cleaned):
        raise ValueError("Not a valid install token")
    return cleaned


def split_token(raw: str) -> tuple[str, str]:
    """``"a1b2c3d4e5f6.<secret>"`` -> ``(id, secret)``. Raises on anything else.

    One place parses the wire form, so the mint path and the two verifying
    routes cannot disagree about where the dot goes.
    """
    parts = (raw or "").strip().split(".", 1)
    if len(parts) != 2 or not parts[1]:
        raise ValueError("Not a valid install token")
    return validate_token_id(parts[0]), parts[1]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _root(data_dir: Path) -> Path:
    return Path(data_dir) / INSTALL_TOKENS_DIR


def _owner_dir(data_dir: Path, owner_user_id: str) -> Path:
    return _root(data_dir) / owner_user_id


def _token_path(data_dir: Path, owner_user_id: str, token_id: str) -> Path:
    return _owner_dir(data_dir, owner_user_id) / f"{validate_token_id(token_id)}.json"


def _load(path: Path) -> Optional[InstallToken]:
    data = FileUtil.read(path, "json")
    if not isinstance(data, dict):
        return None
    try:
        return InstallToken.model_validate(data)
    except Exception:
        return None


def _save(data_dir: Path, record: InstallToken) -> None:
    FileUtil.write(
        _token_path(data_dir, record.owner_user_id, record.id),
        record.model_dump(),
        "json",
    )


# ---------------------------------------------------------------------------
# Minting and lookup
# ---------------------------------------------------------------------------


async def create_install_token(
    data_dir: Path, owner_user_id: str, username: str
) -> tuple[InstallToken, str]:
    """Mint a token. Returns ``(record, raw_token)``.

    Old tokens for the same user are swept, not invalidated one by one: a user
    who reloads the welcome page ten times should not leave ten live sessions
    waiting to be claimed, and the page always shows the newest command anyway.
    Only EXPIRED and CLAIMED records are removed — an unclaimed, unexpired token
    from a second browser tab keeps working, because "I opened this on my phone
    and I'm pasting on my laptop" is a real thing people do.
    """
    async with _lock:
        _sweep(data_dir, owner_user_id)
        token_id = secrets.token_hex(6)
        secret = secrets.token_urlsafe(32)
        now = datetime.now(UTC)
        record = InstallToken(
            id=token_id,
            owner_user_id=owner_user_id,
            username=username,
            secret_hash=hashlib.sha256(secret.encode()).hexdigest(),
            created_at=now.isoformat(),
            expires_at=(
                now + timedelta(seconds=INSTALL_TOKEN_TTL_SECONDS)
            ).isoformat(),
        )
        _save(data_dir, record)
        return record, f"{token_id}.{secret}"


def _sweep(data_dir: Path, owner_user_id: str) -> None:
    """Delete this user's spent and expired tokens. Best-effort housekeeping.

    Never raises: a failure to tidy must not fail the install the user is
    starting.
    """
    owner_dir = _owner_dir(data_dir, owner_user_id)
    if not owner_dir.is_dir():
        return
    for entry in owner_dir.iterdir():
        if not entry.is_file() or entry.suffix != ".json":
            continue
        record = _load(entry)
        if record is None or record.is_expired or record.is_claimed:
            try:
                entry.unlink()
            except OSError:
                pass


def find_install_token(data_dir: Path, token_id: str) -> Optional[InstallToken]:
    """Resolve a token by id alone, scanning owner directories.

    The claim and progress routes need this: the machine presents a token and
    nothing else, so there is no owner to scope the read to yet. Nothing else
    should use it — every other reader knows the owner.
    """
    try:
        wanted = f"{validate_token_id(token_id)}.json"
    except ValueError:
        return None
    root = _root(data_dir)
    if not root.is_dir():
        return None
    for owner_dir in root.iterdir():
        if not owner_dir.is_dir():
            continue
        candidate = owner_dir / wanted
        if candidate.is_file():
            return _load(candidate)
    return None


def verify_install_token(data_dir: Path, raw: str) -> Optional[InstallToken]:
    """The record this raw token names, if the secret matches and it is live.

    Returns None for unknown, mistyped, wrong-secret and expired alike — the
    caller must not tell a stranger which it was. A CLAIMED token still verifies:
    it may no longer be exchanged for a session, but it may still report
    progress, and those are two different permissions.
    """
    try:
        token_id, secret = split_token(raw)
    except ValueError:
        return None
    record = find_install_token(data_dir, token_id)
    if record is None or record.is_expired:
        return None
    if not secrets.compare_digest(
        record.secret_hash, hashlib.sha256(secret.encode()).hexdigest()
    ):
        return None
    return record


async def claim_install_token(data_dir: Path, raw: str) -> Optional[InstallToken]:
    """Spend the session exchange. Returns the record, or None.

    Single use, and the write happens under the lock BEFORE the caller mints a
    session, so two machines racing on one token cannot both get one. An
    already-claimed token returns None — the caller reports it as spent, which is
    a different message from expired because the remedies differ (one is "you
    already ran this", the other "get a fresh command").
    """
    async with _lock:
        record = verify_install_token(data_dir, raw)
        if record is None or record.is_claimed:
            return None
        record.claimed_at = _now()
        _save(data_dir, record)
        return record


async def record_step(
    data_dir: Path,
    raw: str,
    key: str,
    state: StepState,
    detail: str = "",
) -> Optional[InstallToken]:
    """Append (or update) one reported step. Returns the record, or None.

    A repeat of the same ``key`` REPLACES the earlier entry rather than stacking:
    the script reports ``started`` and then ``ok`` for each step, and the
    checklist wants one row per step whose state moves, not two rows that
    disagree.

    An unrecognized key is dropped rather than stored. The list of steps is
    declared here (:data:`INSTALL_STEPS`) because the web app renders the whole
    thing up front; accepting an arbitrary key would let a machine add a row the
    page has no label for.
    """
    if key not in INSTALL_STEP_KEYS:
        return None
    async with _lock:
        record = verify_install_token(data_dir, raw)
        if record is None:
            return None
        step = InstallStep(
            key=key, state=state, detail=(detail or "")[:MAX_DETAIL_CHARS], at=_now()
        )
        kept = [s for s in record.steps if s.key != key]
        kept.append(step)
        record.steps = kept[-MAX_STEPS:]
        _save(data_dir, record)
        return record


def latest_for_owner(data_dir: Path, owner_user_id: str) -> Optional[InstallToken]:
    """The user's newest install run, for the live progress view.

    Newest by ``created_at`` and NOT filtered to unexpired: the page should still
    be able to show how a run that has just timed out got on, and a blank panel
    is the least informative possible answer to "what happened?".
    """
    owner_dir = _owner_dir(data_dir, owner_user_id)
    if not owner_dir.is_dir():
        return None
    records: list[InstallToken] = []
    for entry in owner_dir.iterdir():
        if not entry.is_file() or entry.suffix != ".json":
            continue
        record = _load(entry)
        if record is not None:
            records.append(record)
    if not records:
        return None
    return max(records, key=lambda r: r.created_at)
