# SPDX-License-Identifier: BUSL-1.1
"""
clawmeets/doctor.py

The ONE definition of "your setup is working", and the exact command that fixes
each way it isn't.

Three surfaces ask that question and they used to answer it differently: the
install script (did the thing I just did take?), the web app's zero state (is
this account usable yet?), and a user standing in a terminal wondering why no
agent ever replied. Three definitions meant a checklist could show green while
the runner could not invoke a model, which is the specific failure this module
exists to make impossible. Everything here is computed once, in one place, and
the other three read it — the CLI prints it (``cli_doctor``), the machine
reports it on check-in (``clawmeets_daemon.client``), and the web checklist
renders what the server was told.

## A diagnosis is not a fix

Every failing check carries :attr:`Check.fix` — a command the user can paste,
not a description of their situation. "Claude CLI not logged in" is a
diagnosis; ``claude login`` is a fix. Where no command can do it (a browser
sign-in, turning on Developer Mode) the check carries :attr:`Check.fix_note`
instead and ``fix`` stays empty, so a caller can always tell "paste this" from
"go do this" without parsing prose.

## Stdlib only, and why that is load-bearing

No pydantic, no httpx at import time, nothing from ``clawmeets.llm``. Doctor is
most valuable on a machine where the dependency stack is half-installed, and a
diagnostic that cannot import is worse than no diagnostic. Server reachability
is the one network call and it uses ``urllib``; it degrades to a failed check
rather than raising.

## Model-CLI login detection is a heuristic, and says so

There is no portable, non-interactive way to ask ``claude`` / ``codex`` /
``gemini`` / ``agy`` "are you signed in?" — the credential lives in an OS
keychain on one platform and a dotfile on another, and running the CLI for real
costs a token and a network round-trip. So :data:`MODEL_CLIS` looks for the
credential artifacts each CLI is known to write, plus the API-key environment
variables that bypass login entirely. A false "not logged in" costs the user one
redundant ``claude login``; a false "logged in" costs them a silent agent, so
the checks are deliberately biased toward reporting trouble. ``detail`` names
what was actually looked for, so a wrong answer is debuggable instead of
mysterious.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# Kept in step with clawmeets/cli_lifecycle.py rather than imported from it:
# doctor must answer on a machine where importing the CLI module fails, which is
# exactly when its answer matters most.
DEFAULT_SERVER = os.environ.get("CLAWMEETS_SERVER_URL", "https://clawmeets.ai")
DEFAULT_DATA_DIR = os.environ.get(
    "CLAWMEETS_DATA_DIR", str(Path.home() / ".clawmeets")
)

# How long any probe may take. Short on purpose: doctor runs inside the install
# script's critical path and at the top of a support conversation, and a
# diagnostic that hangs teaches people not to run it.
PROBE_TIMEOUT_SECONDS = 10

# The oldest CLI whose reports this module trusts. Bumped together with the
# bootstrap skill's floor.
MIN_CLI_VERSION = "1.2.4"


# ---------------------------------------------------------------------------
# The result shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    """One thing that is either working or fixable, never merely "wrong".

    ``id`` is the stable machine name (the web checklist keys rows off it and
    the daemon reports it); ``title`` is the sentence a user reads and obeys the
    product's vocabulary rule — it says "your computer", never "runner" or
    "daemon".

    ``ok`` is three-valued in effect: ``True`` passed, ``False`` failed, and a
    check may set ``ok=True`` with a non-empty ``fix_note`` to mean "working,
    but there is something worth doing". Nothing is reported as failed that the
    user cannot act on.
    """

    id: str
    title: str
    ok: bool
    detail: str = ""
    fix: str = ""
    fix_note: str = ""
    # False for checks that are informative but must not hold back a green
    # checklist — a second model CLI, a nice-to-have. The install script exits
    # non-zero only on a blocking failure.
    blocking: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Report:
    """Every check, plus the one-line verdict the callers actually branch on."""

    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when nothing BLOCKING failed.

        A non-blocking failure (no second model CLI) must not make the install
        script report defeat after a successful install.
        """
        return all(c.ok for c in self.checks if c.blocking)

    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    def get(self, check_id: str) -> Optional[Check]:
        return next((c for c in self.checks if c.id == check_id), None)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "checks": [c.to_dict() for c in self.checks],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ---------------------------------------------------------------------------
# Model CLIs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelCLI:
    """One code CLI the runner can drive, and how to tell it is usable.

    ``binary`` mirrors the provider class's default in ``clawmeets/llm/`` —
    ``agy`` for Antigravity, not ``antigravity``. ``tests/test_doctor.py``
    asserts the two stay in step, because a doctor that probes a binary the
    runner never shells is worse than no probe.
    """

    id: str
    label: str
    binary: str
    # Credential artifacts the CLI writes once signed in. Any one present is
    # taken as signed in. `~` is expanded at check time.
    login_paths: tuple[str, ...]
    # Environment variables that make the CLI usable WITHOUT an interactive
    # login. Any one set skips the login check entirely.
    login_env: tuple[str, ...]
    login_fix: str = ""
    login_note: str = ""
    install_fix: str = ""
    install_note: str = ""


# Ordered by how likely the target user already has one. The first PRESENT entry
# is the one the checklist reports on; the rest are informational.
MODEL_CLIS: tuple[ModelCLI, ...] = (
    ModelCLI(
        id="claude",
        label="Claude Code",
        binary="claude",
        # macOS keeps the OAuth token in the login keychain, so there is no file
        # to find there; `~/.claude.json` is still written on every platform and
        # carries the signed-in account once login completes.
        login_paths=("~/.claude/.credentials.json", "~/.claude.json"),
        login_env=("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
        login_fix="claude login",
        install_note="Install Claude Code: https://docs.anthropic.com/claude-code",
    ),
    ModelCLI(
        id="codex",
        label="Codex",
        binary="codex",
        login_paths=("~/.codex/auth.json",),
        login_env=("OPENAI_API_KEY",),
        login_fix="codex login",
        install_note="Install Codex: https://github.com/openai/codex",
    ),
    ModelCLI(
        id="antigravity",
        label="Antigravity",
        binary="agy",
        login_paths=("~/.antigravity/oauth_creds.json", "~/.config/antigravity"),
        login_env=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        # No non-interactive login verb: `agy` completes a Google sign-in in a
        # browser the first time it runs, so the fix is to run it once.
        login_note="Run `agy` once on this computer and complete the Google sign-in",
        install_note=(
            "Install the Antigravity CLI: "
            "curl -fsSL https://antigravity.google/cli/install.sh | bash"
        ),
    ),
    ModelCLI(
        id="gemini",
        label="Gemini CLI",
        binary="gemini",
        login_paths=("~/.gemini/oauth_creds.json",),
        login_env=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        login_note="Run `gemini` once on this computer and complete the sign-in",
        install_note="Install Gemini CLI: https://github.com/google-gemini/gemini-cli",
    ),
    ModelCLI(
        id="opencode",
        label="OpenCode",
        binary="opencode",
        login_paths=("~/.local/share/opencode/auth.json", "~/.config/opencode"),
        login_env=(),
        login_fix="opencode auth login",
        install_note=(
            "Install OpenCode: curl -fsSL https://opencode.ai/install | bash"
        ),
    ),
)


@dataclass(frozen=True)
class ModelCLIState:
    """What we found out about one model CLI on this machine."""

    id: str
    label: str
    binary: str
    present: bool
    logged_in: bool
    version: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _which(binary: str) -> Optional[str]:
    return shutil.which(binary)


def _probe_version(binary: str) -> str:
    """``<binary> --version``, or "" if it cannot be asked.

    Never raises. A CLI that is on PATH but refuses ``--version`` is still
    reported as present — the runner will find out the hard way and say so, and
    a doctor that calls a working CLI broken sends the user on a detour.
    """
    try:
        result = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    # Some CLIs print their version to stderr; take whichever spoke, first line
    # only (a few print a banner underneath).
    lines = (result.stdout or result.stderr or "").strip().splitlines()
    return lines[0].strip() if lines else ""


def _looks_logged_in(spec: ModelCLI) -> bool:
    """Best-effort: did this CLI leave a credential behind, or is a key set?

    See the module docstring on why this is a heuristic. Biased toward reporting
    trouble: an env var counts (it genuinely bypasses login), a file only counts
    if it exists.
    """
    if any(os.environ.get(name) for name in spec.login_env):
        return True
    return any(Path(p).expanduser().exists() for p in spec.login_paths)


def inspect_model_clis() -> list[ModelCLIState]:
    """Every model CLI in :data:`MODEL_CLIS`, present or not.

    Returns the full list rather than only what was found, so a caller can say
    "install one of these" with the real names instead of a generic complaint.
    """
    states: list[ModelCLIState] = []
    for spec in MODEL_CLIS:
        path = _which(spec.binary)
        present = path is not None
        states.append(
            ModelCLIState(
                id=spec.id,
                label=spec.label,
                binary=spec.binary,
                present=present,
                logged_in=_looks_logged_in(spec) if present else False,
                version=_probe_version(spec.binary) if present else "",
            )
        )
    return states


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _data_dir() -> Path:
    return Path(DEFAULT_DATA_DIR).expanduser()


def _read_json(path: Path) -> Optional[dict]:
    """Read a JSON object, or None. Never raises — a corrupt file is "absent"."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def check_install() -> Check:
    """Is the ``clawmeets`` CLI itself on PATH?

    Worth checking even though this code is usually running FROM that CLI: the
    install script calls doctor in a fresh shell, where the real failure is
    ``uv`` putting the binary in ``~/.local/bin`` and that directory not being on
    PATH. The fix is a PATH export, not a reinstall, and saying so saves the most
    common install-time dead end.
    """
    path = _which("clawmeets")
    if path:
        return Check(
            id="install",
            title="ClawMeets is installed",
            ok=True,
            detail=f"{path} ({_probe_version('clawmeets') or 'version unknown'})",
        )
    return Check(
        id="install",
        title="ClawMeets is installed",
        ok=False,
        detail="`clawmeets` is not on PATH",
        fix='export PATH="$HOME/.local/bin:$PATH"',
        fix_note=(
            "uv installs to ~/.local/bin. Add that line to your shell profile "
            "(~/.zshrc or ~/.bashrc) so it survives a new terminal. If it is "
            f"genuinely missing, reinstall: uv tool install --python 3.11 "
            f"'clawmeets>={MIN_CLI_VERSION}'"
        ),
    )


def check_session() -> Check:
    """Is an account signed in and remembered on this machine?

    Reads the same ``config/<user>/settings.json`` the CLI writes, directly
    rather than through ``cli_lifecycle``, so the check survives a broken
    import. A file with no ``token`` is "signed out", not "corrupt" — that is
    exactly the state ``clawmeets user logout`` leaves behind.
    """
    data_dir = _data_dir()
    current = data_dir / "config" / "current_user"
    username = ""
    try:
        username = current.read_text().strip()
    except OSError:
        pass

    if not username:
        return Check(
            id="session",
            title="You are signed in on this computer",
            ok=False,
            detail="No account is signed in",
            fix="clawmeets user login <username>",
            fix_note=(
                "Signed up in the browser? Re-run the one-line install command "
                "from the web app instead — it signs in without a password."
            ),
        )

    config = _read_json(data_dir / "config" / username / "settings.json") or {}
    if not (config.get("user") or {}).get("token"):
        return Check(
            id="session",
            title="You are signed in on this computer",
            ok=False,
            detail=f'"{username}" is set up but has no saved session',
            fix=f"clawmeets user login {username}",
        )

    return Check(
        id="session",
        title="You are signed in on this computer",
        ok=True,
        detail=f'Signed in as "{username}"',
    )


def current_username() -> str:
    """The signed-in account, or "". Shared by several checks and the CLI."""
    try:
        return (_data_dir() / "config" / "current_user").read_text().strip()
    except OSError:
        return ""


def check_assistant() -> Check:
    """Does this machine hold the user's assistant?

    Presence is a directory under ``agents/`` whose name is
    ``<user>-assistant`` — the same thing ``cli_lifecycle`` enumerates to decide
    what to start. Whether it is ONLINE is deliberately not asked here: that is
    a fact about a socket the server owns, the web checklist reads it from the
    server, and a local process check would answer a different question.
    """
    username = current_username()
    if not username:
        return Check(
            id="assistant",
            title="Your assistant is set up",
            ok=False,
            detail="No account is signed in yet",
            fix="clawmeets user login <username>",
        )
    agents = _data_dir() / "agents"
    prefix = f"{username}-assistant"
    found = [
        d.name
        for d in (agents.iterdir() if agents.is_dir() else [])
        if d.is_dir() and d.name.startswith(prefix)
    ]
    if not found:
        return Check(
            id="assistant",
            title="Your assistant is set up",
            ok=False,
            detail=f'No assistant found for "{username}"',
            fix="clawmeets assistant register",
        )
    return Check(
        id="assistant",
        title="Your assistant is set up",
        ok=True,
        detail=found[0],
    )


def check_model_cli(states: Optional[list[ModelCLIState]] = None) -> list[Check]:
    """One blocking check for "a model CLI the runner can actually use", plus
    a non-blocking note per other CLI that is installed but signed out.

    Split that way because the requirement is ONE working CLI, not all of them.
    A user with Claude Code signed in and a stale ``gemini`` on PATH is fully
    working, and failing them over the Gemini install would be the checklist
    lying in the other direction.
    """
    states = states if states is not None else inspect_model_clis()
    specs = {s.id: s for s in MODEL_CLIS}

    usable = [s for s in states if s.present and s.logged_in]
    signed_out = [s for s in states if s.present and not s.logged_in]

    if usable:
        primary = Check(
            id="model_cli",
            title="A model CLI is installed and signed in",
            ok=True,
            detail=", ".join(s.label for s in usable),
        )
        # With one CLI already working, the rest are the user's business.
        also_mention = signed_out
    elif signed_out:
        # Installed but signed out — the highest-value fix in this module, and
        # the one the old setup flow reported not at all.
        first, *rest = signed_out
        spec = specs[first.id]
        primary = Check(
            id="model_cli",
            title="A model CLI is installed and signed in",
            ok=False,
            detail=f"{first.label} is installed but not signed in",
            fix=spec.login_fix,
            fix_note=spec.login_note,
        )
        also_mention = rest
    else:
        claude = specs["claude"]
        primary = Check(
            id="model_cli",
            title="A model CLI is installed and signed in",
            ok=False,
            detail="None found. Looked for: "
            + ", ".join(s.binary for s in MODEL_CLIS),
            fix_note=(
                f"{claude.install_note}, then run `{claude.login_fix}`. "
                "Any one of Claude Code, Codex, Antigravity, Gemini CLI or "
                "OpenCode works."
            ),
        )
        also_mention = []

    # Every OTHER installed-but-signed-out CLI, reported once and non-blocking:
    # the user may have meant to use it, but one working CLI is the requirement.
    extras = [
        Check(
            id=f"model_cli.{s.id}",
            title=f"{s.label} is signed in",
            ok=False,
            detail=f"{s.label} is installed but not signed in",
            fix=specs[s.id].login_fix,
            fix_note=specs[s.id].login_note,
            blocking=False,
        )
        for s in also_mention
    ]
    return [primary, *extras]


def check_computer(username: str = "") -> Check:
    """Is this computer linked, and is its connection running?

    Reads ``~/.clawmeets/computer/<user>/{config.json,computer.pid}`` directly
    instead of importing ``clawmeets_daemon``: the interesting failure is that
    the daemon distribution is NOT installed, and an ImportError is a worse way
    to learn that than a check that says so and names the fix.

    Reports the LOCAL truth only — whether a process is alive here. Whether the
    SERVER can see it is a different fact, owned by the server and rendered by
    the web checklist, and a local check that claimed it would be guessing.
    """
    username = username or current_username()
    root = _data_dir() / "computer"
    account_dir = root / username if username else root
    config = _read_json(account_dir / "config.json")

    if not config or not config.get("host_id"):
        detail = (
            f'This computer is not linked for "{username}"'
            if username
            else "This computer is not linked"
        )
        return Check(
            id="computer",
            title="This computer is linked to your account",
            ok=False,
            detail=detail,
            fix="clawmeets computer install",
            fix_note=(
                "Needs the `clawmeets-daemon` package: "
                "uv tool install --python 3.11 clawmeets-daemon"
            ),
        )

    pid_file = account_dir / "computer.pid"
    pid = 0
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        pid = 0
    if pid and _pid_alive(pid):
        return Check(
            id="computer",
            title="This computer is linked to your account",
            ok=True,
            detail=f"Linked and connected (PID {pid})",
        )
    return Check(
        id="computer",
        title="This computer is linked to your account",
        ok=False,
        detail="Linked, but the connection is not running",
        fix="clawmeets computer start",
    )


def _pid_alive(pid: int) -> bool:
    """Does this pid name a live process? Stdlib, cross-platform, never raises."""
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
                timeout=PROBE_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return str(pid) in (out.stdout or "")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Alive, owned by someone else. Still alive.
        return True
    except OSError:
        return False
    return True


def autostart_artifact(username: str = "") -> Optional[Path]:
    """Where the login entry for ``username`` lives on this platform, or None.

    DUPLICATES the paths in ``clawmeets_daemon/autostart.py`` rather than
    importing them, for the same reason :func:`check_computer` re-reads
    ``config.json`` by hand: autostart belongs to the ``clawmeets-daemon``
    distribution, the runner wheel does not ship it, and a doctor that raised
    ImportError on a machine where that package is missing would fail exactly
    where it is most needed. ``tests/test_doctor.py`` asserts these two agree, the
    same way ``tests/test_host_protocol_parity.py`` guards the other deliberate
    duplication across this boundary.

    A PATH, not a verdict: whether the entry is also switched on is the daemon's
    own question, answered by ``clawmeets computer autostart status``.
    """
    if sys.platform == "darwin":
        label = f"ai.clawmeets.computer.{username}" if username else "ai.clawmeets.computer"
        return Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
    if sys.platform == "linux":
        unit = (
            f"clawmeets-computer-{username}.service"
            if username
            else "clawmeets-computer.service"
        )
        return Path.home() / ".config" / "systemd" / "user" / unit
    return None


def check_autostart(username: str = "") -> Check:
    """Will the connection come back by itself after a reboot?

    Non-blocking by design: everything works today without it. It is still
    checked and still reported, because almost nobody stops their agents on
    purpose — they reboot — and "my agents went offline overnight" is the single
    most common way a working setup stops being one.

    "Registered" here means the login entry we write is PRESENT. It deliberately
    does not re-derive whether the OS also has it switched on: that is the
    daemon's question, and someone who ran ``systemctl --user disable`` by hand
    made a decision this check has no business overriding.
    """
    username = username or current_username()
    path = autostart_artifact(username)
    if path is None:
        return Check(
            id="autostart",
            title="Your agents come back after a restart",
            ok=True,
            detail="Not something ClawMeets sets up on this platform",
            fix_note="Run `clawmeets computer start` after a restart.",
            blocking=False,
        )
    if path.is_file():
        return Check(
            id="autostart",
            title="Your agents come back after a restart",
            ok=True,
            detail="Starts when you log in",
            blocking=False,
        )
    return Check(
        id="autostart",
        title="Your agents come back after a restart",
        ok=False,
        detail="Not set to start when you log in",
        fix="clawmeets computer autostart enable",
        blocking=False,
    )


def check_server(server_url: str = "") -> Check:
    """Can this machine reach the server at all?

    Last in the list and first in usefulness when it fails: every check above it
    is about local state, and a machine behind a captive portal or a corporate
    proxy will pass all of them and still never see an agent reply.

    Probes ``GET /``, which is the same liveness check the deployment itself
    uses (``ops/deploy-prod``: "Health: GET / → 200 (there is no /health
    route)"). Reusing it rather than inventing ``/health`` keeps one answer to
    "is this server up?" — and means this check cannot start failing because a
    route nobody else depends on was never added.

    ``urllib``, not httpx, so it answers even when the dependency stack does not.
    """
    url = (server_url or DEFAULT_SERVER).rstrip("/")
    try:
        with urllib.request.urlopen(f"{url}/", timeout=PROBE_TIMEOUT_SECONDS) as r:
            code = r.status
    except urllib.error.HTTPError as e:
        # An HTTP error still proves we REACHED a server, which is the question
        # being asked. A deployment that serves no landing page answers 404 here
        # and is perfectly healthy, so the status is not reported to the user —
        # saying "answered 404" under a green check reads as a problem.
        return Check(
            id="server",
            title="ClawMeets is reachable",
            ok=True,
            detail=f"{url} (HTTP {e.code})",
        )
    except (urllib.error.URLError, OSError, ValueError) as e:
        return Check(
            id="server",
            title="ClawMeets is reachable",
            ok=False,
            detail=f"Could not reach {url}: {e}",
            fix="",
            fix_note=(
                "Check your network. On a VPN or corporate proxy, allow "
                f"{url}. Self-hosting? Set CLAWMEETS_SERVER_URL."
            ),
        )
    return Check(
        id="server",
        title="ClawMeets is reachable",
        ok=True,
        detail=f"{url} (HTTP {code})",
    )


# ---------------------------------------------------------------------------
# The whole report
# ---------------------------------------------------------------------------


def run_checks(
    *,
    server_url: str = "",
    include_server: bool = True,
    include_autostart: bool = True,
) -> Report:
    """Every check, in the order a user hits them.

    Order is install -> sign-in -> assistant -> model CLI -> computer ->
    autostart -> server, which is the order the install script performs them, so
    the FIRST failure in the list is also the earliest thing that went wrong.
    A reader who fixes top-down never fixes something that was only broken
    because of something above it.

    ``include_server`` / ``include_autostart`` exist for the daemon, which
    reports this on every check-in: it already knows it reached the server (it
    is talking to it over a socket) and re-probing on a timer would be a
    self-answering question on a 30-second loop.
    """
    states = inspect_model_clis()
    checks: list[Check] = [
        check_install(),
        check_session(),
        check_assistant(),
        *check_model_cli(states),
        check_computer(),
    ]
    if include_autostart:
        checks.append(check_autostart())
    if include_server:
        checks.append(check_server(server_url))
    return Report(checks=checks)


def model_cli_wire() -> list[dict]:
    """The model-CLI findings, in the shape the machine reports to the server.

    Separate from :func:`run_checks` because the daemon sends exactly this and
    nothing else about model CLIs: the server needs to answer "did we OBSERVE a
    usable model CLI on that machine?" for the web checklist, and the prose of a
    fix command is a thing the terminal shows, not a thing the server stores.
    """
    return [s.to_dict() for s in inspect_model_clis()]
