# SPDX-License-Identifier: MIT
"""The host report (plan-owner-isolation section 5a): what an unsandboxed
turn on this runner's host could read.

Every clawmeets home this OS user has is scanned: the runner's own (the
parent of its ``agents/`` folder), ``$CLAWMEETS_DATA_DIR`` and
``~/.clawmeets``. The scan lists credential files and reads only the
``agent_id`` key of each, and only whether a saved login carries a token,
a refresh token or a password. Nothing it reads leaves the host except
those ids and booleans; the server never learns a token or a key.

The runner rescans for every turn (directory listings and a few small JSON
reads), so a private agent installed later shows up at the next mint.
"""
from __future__ import annotations

import getpass
import hashlib
import json
import logging
import os
import platform
import subprocess
from pathlib import Path
from typing import Optional

from clawmeets.api.turn_protocol import HostReport

logger = logging.getLogger("clawmeets.runner.host_report")


def _machine_id() -> str:
    """IOPlatformUUID (macOS), /etc/machine-id (Linux), MachineGuid
    (Windows); "" when none can be read. Not secret."""
    system = platform.system()
    try:
        if system == "Darwin":
            out = subprocess.run(["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                                 capture_output=True, text=True, timeout=5).stdout
            for line in out.splitlines():
                if "IOPlatformUUID" in line:
                    return line.split("=", 1)[1].strip().strip('"')
        elif system == "Windows":
            import winreg  # type: ignore[import-not-found]
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography") as k:
                return str(winreg.QueryValueEx(k, "MachineGuid")[0])
        else:
            for p in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
                if os.path.exists(p):
                    return Path(p).read_text().strip()
    except Exception:  # noqa: BLE001 — an unreadable id is "", never fatal
        logger.debug("machine id unavailable", exc_info=True)
    return ""


def _os_user() -> str:
    getuid = getattr(os, "getuid", None)
    return str(getuid()) if getuid is not None else getpass.getuser()


def host_key() -> str:
    return hashlib.sha256(f"clawmeets-host-v1|{_machine_id()}|{_os_user()}".encode()).hexdigest()


def clawmeets_homes(agent_dir: Optional[Path]) -> list[Path]:
    """Every clawmeets home this OS user has, deduplicated, existing only."""
    candidates: list[Path] = []
    if agent_dir is not None:
        candidates.append(Path(agent_dir).parent.parent)
    env = os.environ.get("CLAWMEETS_DATA_DIR")
    if env:
        candidates.append(Path(env))
    candidates.append(Path.home() / ".clawmeets")
    out: list[Path] = []
    for c in candidates:
        try:
            real = c.expanduser().resolve()
        except OSError:
            continue
        if real.is_dir() and real not in out:
            out.append(real)
    return out


def _read_json(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def readable_agent_ids(homes: list[Path]) -> list[str]:
    ids: set[str] = set()
    for home in homes:
        for sub in ("agents", "assistants"):
            root = home / sub
            if not root.is_dir():
                continue
            for cred in root.glob("*/credential.json"):
                if not os.access(cred, os.R_OK):
                    continue
                data = _read_json(cred) or {}
                agent_id = data.get("agent_id")
                if isinstance(agent_id, str) and agent_id:
                    ids.add(agent_id)
    return sorted(ids)


def user_session_present(homes: list[Path]) -> bool:
    """True when any saved login could act as a ClawMeets user: a token, a
    refresh token or a legacy password (any one is enough)."""
    for home in homes:
        for settings in (home / "config").glob("*/settings.json"):
            user = (_read_json(settings) or {}).get("user")
            if isinstance(user, dict) and any(user.get(k) for k in ("token", "refresh_token", "password")):
                return True
    return False


#: The environment variable each API provider family falls back to.
_KEY_ENV = {
    "claude": ("ANTHROPIC_API_KEY",), "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"), "openrouter": ("OPENROUTER_API_KEY",),
}


def model_identity(agent_dir: Optional[Path]) -> tuple[str, Optional[str]]:
    """``(model_login, model_key_digest)`` the way a turn would resolve the
    model credential (section 5c): a key from the default model config, the
    card's ``llm_api_key``, the env store or the environment gives
    ``("api_key", HMAC-SHA256(owner id, key)[:32])``; a CLI provider with no
    key uses the OS user's CLI login (``"cli_login"``). The key never leaves
    the host; the owner-salted digest only lets the server compare it with the
    owner's other agents."""
    import hmac

    if agent_dir is None:
        return "none", None
    card = _read_json(Path(agent_dir) / "card.json") or {}
    settings = card.get("local_settings") or {}
    provider = str(settings.get("llm_provider") or "claude")
    family = provider[:-4] if provider.endswith("-api") else provider
    default = next((c for c in card.get("model_configs") or []
                    if isinstance(c, dict) and c.get("name") == card.get("default_model_config_name")), {})
    key = default.get("api_key") or settings.get("llm_api_key")
    if not key:
        store = _read_json(Path(agent_dir) / "env.json") or {}
        for name in _KEY_ENV.get(family, ()):
            key = store.get(name) or os.environ.get(name)
            if key:
                break
    if key:
        owner = str(card.get("registered_by") or "")
        digest = hmac.new(owner.encode(), str(key).encode(), hashlib.sha256).hexdigest()[:32]
        return "api_key", digest
    if not provider.endswith("-api"):
        return "cli_login", None
    return "none", None


def collect_host_report(agent_dir: Optional[Path]) -> HostReport:
    """Scan this host (sections 5a and 5c)."""
    homes = clawmeets_homes(agent_dir)
    login, digest = model_identity(agent_dir)
    return HostReport(
        host_key=host_key(),
        readable_agent_ids=readable_agent_ids(homes),
        user_session_present=user_session_present(homes),
        model_login=login,  # type: ignore[arg-type]
        model_key_digest=digest,
    )
