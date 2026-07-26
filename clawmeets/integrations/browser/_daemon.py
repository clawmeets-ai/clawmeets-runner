# SPDX-License-Identifier: MIT
"""
clawmeets/integrations/browser/_daemon.py

Long-lived per-agent browser daemon — the opt-in fast path for the
playwright-browser skill.

One daemon process owns a single patchright persistent context (the agent's
identity profile) and one page (browser tab) per project, keyed by the
project's sandbox session dir. It listens on a Unix socket and answers
newline-delimited JSON requests ``{op, key, args}`` by driving the shared
``_lib._do_*`` page bodies — the exact same logic the one-shot CLI path runs,
so behavior is identical; only the page lifetime differs.

Why a daemon (vs. cold-starting Chrome per command): it removes the ~1.5-2s
cold start AND keeps the whole session in memory — cookies (including ephemeral
``expires:-1`` session cookies that Chrome never persists to disk), localStorage,
IndexedDB, and the bot-manager sensor state — so login simply stays live with no
export/clone/seed. The browser is launched once via ``_lib._open_context`` and
NEVER reconnected over CDP, preserving patchright's Cloudflare stealth.

Run via ``clawmeets browser start`` (which spawns this detached); not meant to be
invoked by hand.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
from pathlib import Path
from typing import Optional

from clawmeets.integrations.browser import _lib


class _Daemon:
    def __init__(self, context, stop_event: asyncio.Event):
        self._context = context
        self._stop = stop_event
        # A project key owns a GROUP of tabs (a click can open new ones) with an
        # explicit active index. We never auto-switch the active tab on a popup —
        # the caller (the LLM) is told a tab opened and chooses via `switch_tab`.
        self._tabs: dict[str, list] = {}     # key -> [Page, ...]
        self._active: dict[str, int] = {}    # key -> active index into _tabs[key]
        self._locks: dict[str, asyncio.Lock] = {}
        self._create_lock = asyncio.Lock()

    def _live_tabs(self, key: str) -> list:
        """The project's non-closed tabs, pruning any that closed, with the
        active index clamped back into range."""
        tabs = [p for p in self._tabs.get(key, []) if not p.is_closed()]
        self._tabs[key] = tabs
        if tabs:
            idx = self._active.get(key, len(tabs) - 1)
            self._active[key] = min(max(idx, 0), len(tabs) - 1)
        else:
            self._active.pop(key, None)
        return tabs

    def _tabs_list(self, key: str) -> list:
        """Serializable view of the project's tabs for the caller."""
        tabs = self._live_tabs(key)
        active = self._active.get(key, 0)
        out = []
        for i, p in enumerate(tabs):
            try:
                url = p.url
            except Exception:  # noqa: BLE001 - a racing close; report best-effort
                url = None
            out.append({"index": i, "url": url, "active": i == active})
        return out

    async def _page_for(self, key: str):
        """Get-or-create the ACTIVE tab for a project key. Reuse the context's
        initial blank page for the first project, open a fresh tab afterwards."""
        async with self._create_lock:
            tabs = self._live_tabs(key)
            if tabs:
                return tabs[self._active[key]]
            if not any(self._tabs.values()) and self._context.pages:
                page = self._context.pages[0]
            else:
                page = await self._context.new_page()
            self._tabs[key] = [page]
            self._active[key] = 0
            return page

    def _adopt_new_tabs(self, key: str, before: list, result: dict) -> dict:
        """After an action, fold any newly-opened tabs into the project's group
        WITHOUT changing the active tab, and annotate the result so the caller
        can decide whether to `switch_tab` to one."""
        new = [p for p in self._context.pages
               if p not in before and not p.is_closed()]
        if not new:
            return result
        self._tabs.setdefault(key, []).extend(new)
        listing = self._tabs_list(key)
        by_page = {id(p): entry for p, entry in zip(self._live_tabs(key), listing)}
        result["opened_tabs"] = [by_page[id(p)] for p in new if id(p) in by_page]
        result["tabs"] = listing
        result["note"] = (
            "new tab(s) opened; active tab unchanged. Use `switch-tab <index>` "
            "to interact with one, or `close-tab <index>` to discard it."
        )
        return result

    def _lock_for(self, key: str) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    async def dispatch(self, req: dict) -> dict:
        op = req.get("op")
        args = req.get("args") or {}
        key = req.get("key")

        if op == "ping":
            return {"ok": True, "runtime": _lib._runtime_name()}
        if op == "shutdown":
            self._stop.set()
            return {"ok": True}
        if op == "status":
            return {
                "ok": True,
                "runtime": _lib._runtime_name(),
                "projects": sorted(self._tabs.keys()),
                "open_pages": len([p for p in self._context.pages if not p.is_closed()]),
            }

        if op == "auth_open":
            # Open (or focus) a tab and optionally load a login URL. The user
            # signs in directly in the live window; the persistent profile keeps
            # the session. No export step.
            page = await self._page_for(key or "__auth__")
            url = args.get("url")
            if url:
                await page.goto(url, wait_until="domcontentloaded")
            return {"ok": True, "url": page.url, "runtime": _lib._runtime_name()}

        if not key:
            return {"error": f"op {op!r} requires a project key"}

        # Tab-management ops act on the whole group; they must not auto-create a
        # tab, so they run before _page_for.
        if op == "tabs":
            async with self._lock_for(key):
                return {"ok": True, "tabs": self._tabs_list(key),
                        "runtime": _lib._runtime_name()}
        if op == "switch_tab":
            async with self._lock_for(key):
                tabs = self._live_tabs(key)
                idx = args.get("index")
                if not isinstance(idx, int) or idx < 0 or idx >= len(tabs):
                    return {"error": f"no tab {idx!r}; {len(tabs)} open. "
                                     "Run `tabs` to list them."}
                self._active[key] = idx
                result = await _lib._do_snapshot(tabs[idx])
                result["tabs"] = self._tabs_list(key)
                return result
        if op == "close_tab":
            async with self._lock_for(key):
                tabs = self._live_tabs(key)
                idx = args.get("index")
                if not isinstance(idx, int) or idx < 0 or idx >= len(tabs):
                    return {"error": f"no tab {idx!r}; {len(tabs)} open."}
                if len(tabs) <= 1:
                    return {"error": "cannot close the last tab of a project."}
                await tabs[idx].close()
                self._live_tabs(key)  # prune + clamp active
                return {"ok": True, "tabs": self._tabs_list(key),
                        "runtime": _lib._runtime_name()}

        page = await self._page_for(key)
        async with self._lock_for(key):
            if op == "navigate":
                return await _lib._do_navigate(page, args.get("url"), {}, args.get("settle_ms"))
            if op == "snapshot":
                return await _lib._do_snapshot(page)
            if op == "click":
                before = list(self._context.pages)
                result = await _lib._do_click(page, args["selector"], args.get("settle_ms"))
                return self._adopt_new_tabs(key, before, result)
            if op == "fill":
                return await _lib._do_fill(page, args["selector"], args["text"])
            if op == "press_key":
                before = list(self._context.pages)
                result = await _lib._do_press_key(page, args["key"], args.get("settle_ms"))
                return self._adopt_new_tabs(key, before, result)
            if op == "screenshot":
                return await _lib._do_screenshot(
                    page, args["out_path"], args.get("full_page", False)
                )
            if op == "wait_for":
                return await _lib._do_wait_for(
                    page, args["selector"], args.get("timeout_ms", 10000)
                )
        return {"error": f"unknown op: {op!r}"}

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            line = await reader.readline()
            if not line:
                return
            try:
                req = json.loads(line.decode())
            except json.JSONDecodeError as exc:
                resp = {"error": f"bad request json: {exc}"}
            else:
                try:
                    resp = await self.dispatch(req)
                except Exception as exc:  # noqa: BLE001 - report, don't crash the daemon
                    resp = {"error": f"{type(exc).__name__}: {exc}"}
            writer.write((json.dumps(resp, ensure_ascii=False) + "\n").encode())
            await writer.drain()
        finally:
            writer.close()


async def serve(*, socket_path: Path, pid_path: Path, profile_dir: Path,
                headless: Optional[bool]) -> None:
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    # Clear a stale socket from a previous (dead) daemon before binding.
    if socket_path.exists():
        socket_path.unlink()

    fp = _lib.resolve_fingerprint(_lib.load_config()[0], headless_override=headless)
    stop_event = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:  # pragma: no cover - non-unix
            pass

    async with _lib._async_playwright() as p:
        context = await _lib._open_context(p, profile_dir, fp, seed_storage_path=None)
        daemon = _Daemon(context, stop_event)
        server = await asyncio.start_unix_server(daemon.handle, path=str(socket_path))
        try:
            os.chmod(socket_path, 0o600)  # owner-only — it's a shared-temp path
        except OSError:
            pass
        pid_path.write_text(str(os.getpid()))
        print(
            f"[browser daemon] up: pid={os.getpid()} runtime={_lib._runtime_name()} "
            f"socket={socket_path}",
            file=sys.stderr, flush=True,
        )
        try:
            async with server:
                await stop_event.wait()
        finally:
            await context.close()
            for path in (socket_path, pid_path):
                try:
                    path.unlink()
                except OSError:
                    pass
            print("[browser daemon] stopped", file=sys.stderr, flush=True)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="clawmeets-browser-daemon")
    ap.add_argument("--agent-dir", default=None)
    ap.add_argument("--storage", default="personal")
    headless = ap.add_mutually_exclusive_group()
    headless.add_argument("--headless", dest="headless", action="store_true", default=None)
    headless.add_argument("--headed", dest="headless", action="store_false")
    a = ap.parse_args(argv)

    agent_dir = Path(a.agent_dir) if a.agent_dir else None
    socket_path = _lib.daemon_socket_path(agent_dir)
    pid_path = _lib.daemon_pid_path(agent_dir)
    profile_dir = _lib.daemon_profile_dir(agent_dir, storage=a.storage)

    try:
        asyncio.run(serve(
            socket_path=socket_path, pid_path=pid_path,
            profile_dir=profile_dir, headless=a.headless,
        ))
    except KeyboardInterrupt:  # pragma: no cover
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
