# SPDX-License-Identifier: MIT
"""
clawmeets/cli_model_config.py — named model configs from the terminal.

Two groups over routes that already exist on the server:

``clawmeets model-config <agent|self> …``
    An agent's named model configs (``/agents/{id}/model-configs``): list, add,
    edit, remove, default, order (= smart-routing priority), routing, smart
    on|off, and reset (re-copy the owner's default onto this one agent).
    ``self`` uses the runner-injected agent identity, so an agent changes only
    its own configs; any other name resolves through the owner / assistant
    session, which reaches every agent the owner has. Paired system skill:
    ``reconfigure-agent``.

``clawmeets model-default …``
    The owner's DEFAULT model config (``/me/model-defaults``): the copy every
    new agent starts with. show, set, order, routing, clear, apply (copy it
    onto existing agents; the "Model agents" team is skipped), and setup — the
    one-line installer's step that writes it from the model CLIs signed in on
    this computer.

Every change reaches a running agent through the server's settings-change
notice and takes effect on its NEXT turn.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import httpx
import typer

from clawmeets import doctor
from clawmeets.cli_runner import (
    DEFAULT_DATA_DIR,
    _VALID_LLM_PROVIDERS,
    _http,
    _ok,
    _resolve_session_for_config,
    _resolve_user_session,
)

model_config_app = typer.Typer(
    name="model-config",
    help="An agent's named model configs and smart routing. Paired skill: reconfigure-agent.",
    no_args_is_help=True,
)
model_default_app = typer.Typer(
    name="model-default",
    help="Your default model config: what every new agent starts with.",
    no_args_is_help=True,
)

SMART_CONFIG_NAME = "Smart routing"
MODEL_AGENTS_TEAM = "Model agents"

_AGENT_ARG = typer.Argument(..., help="Agent name or id, or 'self' (the calling agent).")


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

def _provider(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized not in _VALID_LLM_PROVIDERS:
        typer.echo(
            f"Error: provider must be one of {', '.join(_VALID_LLM_PROVIDERS)} (got {value!r}).",
            err=True,
        )
        raise typer.Exit(1)
    return normalized


def _describe(listing: dict, title: str) -> None:
    """Print a config list the way the settings page reads it."""
    configs = listing.get("configs") or []
    default = listing.get("default")
    if not configs:
        typer.echo(f"{title}: no model configs.")
        return
    smart = any(c["name"] == default and c["provider"] == "smart" for c in configs)
    typer.echo(f"{title}: " + ("smart routing" if smart else f"single model ({default})"))
    rank = 0
    for c in configs:
        if c["provider"] == "smart":
            continue
        rank += 1
        star = "*" if c["name"] == default else " "
        extra = " ".join(
            x for x in (
                c.get("model") and f"model={c['model']}",
                c.get("base_url") and f"base_url={c['base_url']}",
                c.get("api_key_set") and "key=saved",
            ) if x
        )
        prefix = f"{rank}." if smart else star
        typer.echo(f"  {prefix} {c['name']}  ({c['provider']}){'  ' + extra if extra else ''}")
    if smart:
        routing = listing.get("routing") or {}
        typer.echo(
            f"  tries up to {routing.get('max_attempts')} route(s); a failed route is "
            f"skipped for {routing.get('cooldown_seconds')}s"
        )


def _summary_line(listing: dict) -> str:
    configs = listing.get("configs") or []
    default = listing.get("default")
    routes = [c["name"] for c in configs if c["provider"] != "smart"]
    if any(c["name"] == default and c["provider"] == "smart" for c in configs):
        return "smart -> " + ", ".join(routes)
    return default or "none"


def _ordered(current: list[str], first: list[str]) -> list[str]:
    """``first`` (case-insensitive, must exist) followed by the rest of
    ``current`` in its existing order — so callers name only what moves."""
    lowered = {n.lower(): n for n in current}
    picked: list[str] = []
    for name in first:
        real = lowered.get(name.strip().lower())
        if real is None:
            typer.echo(f"Error: no model config named {name!r}.", err=True)
            raise typer.Exit(1)
        if real not in picked:
            picked.append(real)
    return picked + [n for n in current if n not in picked]


class _Target:
    """The base path of one config list — an agent's or the owner's default.
    The two route families are path-for-path the same."""

    def __init__(self, client: httpx.Client, token: str, base: str, title: str) -> None:
        self.client = client
        self.auth = {"Authorization": f"Bearer {token}"}
        self.base = base
        self.title = title

    def get(self) -> dict:
        return _ok(self.client.get(self.base, headers=self.auth))

    def call(self, method: str, suffix: str = "", body: Optional[dict] = None) -> dict:
        resp = self.client.request(method, self.base + suffix, json=body, headers=self.auth)
        if resp.status_code == 204:
            return {}
        return _ok(resp)

    def named(self, name: str) -> str:
        return "/" + quote(name, safe="")

    def set_smart(self, on: bool) -> None:
        listing = self.get()
        configs = listing.get("configs") or []
        routers = [c["name"] for c in configs if c["provider"] == "smart"]
        routes = [c["name"] for c in configs if c["provider"] != "smart"]
        if not on:
            if routes:
                self.call("PUT", "/default", {"name": routes[0]})
            return
        if not routers:
            self.call("POST", "", {"name": SMART_CONFIG_NAME, "provider": "smart"})
            routers = [SMART_CONFIG_NAME]
        self.call("PUT", "/default", {"name": routers[0]})

    def reorder(self, first: list[str]) -> dict:
        configs = self.get().get("configs") or []
        routers = [c["name"] for c in configs if c["provider"] == "smart"]
        routes = [c["name"] for c in configs if c["provider"] != "smart"]
        return self.call("PUT", "/order", {"names": routers + _ordered(routes, first)})


def _agent_target(client: httpx.Client, token: str, agent_id: str, agent_name: str) -> _Target:
    return _Target(client, token, f"/agents/{agent_id}/model-configs", agent_name)


# ---------------------------------------------------------------------------
# model-config <agent|self>
# ---------------------------------------------------------------------------

def _with_agent(agent: str, token: Optional[str], server: Optional[str], data_dir: Path):
    """(server_url, token, agent_id, agent_name); ``self`` = the calling agent."""
    return _resolve_session_for_config(data_dir, token, server, agent)


_TOKEN = typer.Option(None, "--token", "-t")
_SERVER = typer.Option(None, "--server", "-s")
_DATA_DIR = typer.Option(DEFAULT_DATA_DIR, "--data-dir")


@model_config_app.command("list")
def mc_list(agent: str = _AGENT_ARG, token: Optional[str] = _TOKEN,
            server: Optional[str] = _SERVER, data_dir: Path = _DATA_DIR):
    """Show an agent's model configs, which one answers, and the routing order."""
    server_url, tok, aid, name = _with_agent(agent, token, server, data_dir)
    with _http(server_url) as client:
        _describe(_agent_target(client, tok, aid, name).get(), name)


@model_config_app.command("add")
def mc_add(
    agent: str = _AGENT_ARG,
    name: str = typer.Argument(..., help="Config name (letters, digits, space, _ or -)."),
    provider: str = typer.Option(..., "--provider", "-p", help="Provider id, e.g. claude, openai, gemini, openrouter-api."),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    base_url: Optional[str] = typer.Option(None, "--base-url"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Required for '-api' providers. Prefer Agent Settings: a key typed into chat syncs to the server."),
    make_default: bool = typer.Option(False, "--default", help="Also make it the config that answers."),
    token: Optional[str] = _TOKEN, server: Optional[str] = _SERVER, data_dir: Path = _DATA_DIR,
):
    """Add a named model config. Under smart routing it becomes the LAST route."""
    body = {"name": name, "provider": _provider(provider), "model": model, "base_url": base_url}
    if api_key:
        body["api_key"] = api_key
    server_url, tok, aid, agent_name = _with_agent(agent, token, server, data_dir)
    with _http(server_url) as client:
        target = _agent_target(client, tok, aid, agent_name)
        target.call("POST", "", body)
        if make_default:
            target.call("PUT", "/default", {"name": name})
        _describe(target.get(), agent_name)


@model_config_app.command("edit")
def mc_edit(
    agent: str = _AGENT_ARG,
    name: str = typer.Argument(..., help="Current config name."),
    new_name: Optional[str] = typer.Option(None, "--name", help="Rename it."),
    provider: Optional[str] = typer.Option(None, "--provider", "-p"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Empty string clears."),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Empty string clears."),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Empty string clears."),
    token: Optional[str] = _TOKEN, server: Optional[str] = _SERVER, data_dir: Path = _DATA_DIR,
):
    """Change one config. Only the flags you pass are touched."""
    body: dict = {}
    for key, value in (("name", new_name), ("provider", _provider(provider)),
                       ("model", model), ("base_url", base_url), ("api_key", api_key)):
        if value is not None:
            body[key] = value
    if not body:
        typer.echo("Error: pass at least one of --name/--provider/--model/--base-url/--api-key.", err=True)
        raise typer.Exit(1)
    server_url, tok, aid, agent_name = _with_agent(agent, token, server, data_dir)
    with _http(server_url) as client:
        target = _agent_target(client, tok, aid, agent_name)
        target.call("PATCH", target.named(name), body)
        _describe(target.get(), agent_name)


@model_config_app.command("remove")
def mc_remove(agent: str = _AGENT_ARG, name: str = typer.Argument(...),
              token: Optional[str] = _TOKEN, server: Optional[str] = _SERVER,
              data_dir: Path = _DATA_DIR):
    """Delete one config. Removing the one that answers promotes the next."""
    server_url, tok, aid, agent_name = _with_agent(agent, token, server, data_dir)
    with _http(server_url) as client:
        target = _agent_target(client, tok, aid, agent_name)
        target.call("DELETE", target.named(name))
        _describe(target.get(), agent_name)


@model_config_app.command("default")
def mc_default(agent: str = _AGENT_ARG, name: str = typer.Argument(...),
               token: Optional[str] = _TOKEN, server: Optional[str] = _SERVER,
               data_dir: Path = _DATA_DIR):
    """Make one config the one that answers.

    Naming a single model ends smart routing; naming the smart-routing config
    turns it on."""
    server_url, tok, aid, agent_name = _with_agent(agent, token, server, data_dir)
    with _http(server_url) as client:
        target = _agent_target(client, tok, aid, agent_name)
        _describe(target.call("PUT", "/default", {"name": name}), agent_name)


@model_config_app.command("smart")
def mc_smart(agent: str = _AGENT_ARG,
             state: str = typer.Argument(..., help="on | off"),
             token: Optional[str] = _TOKEN, server: Optional[str] = _SERVER,
             data_dir: Path = _DATA_DIR):
    """Turn smart routing on (try every config in order) or off (route 1 answers)."""
    if state not in ("on", "off"):
        typer.echo("Error: state must be 'on' or 'off'.", err=True)
        raise typer.Exit(1)
    server_url, tok, aid, agent_name = _with_agent(agent, token, server, data_dir)
    with _http(server_url) as client:
        target = _agent_target(client, tok, aid, agent_name)
        target.set_smart(state == "on")
        _describe(target.get(), agent_name)


@model_config_app.command("order")
def mc_order(agent: str = _AGENT_ARG,
             names: list[str] = typer.Argument(..., help="Configs to try first, in order; the rest keep their order after them."),
             token: Optional[str] = _TOKEN, server: Optional[str] = _SERVER,
             data_dir: Path = _DATA_DIR):
    """Set the smart-routing priority."""
    server_url, tok, aid, agent_name = _with_agent(agent, token, server, data_dir)
    with _http(server_url) as client:
        _describe(_agent_target(client, tok, aid, agent_name).reorder(names), agent_name)


@model_config_app.command("routing")
def mc_routing(agent: str = _AGENT_ARG,
               max_attempts: Optional[int] = typer.Option(None, "--max-attempts", help="Routes one request may try (1-10)."),
               cooldown: Optional[int] = typer.Option(None, "--cooldown", help="Seconds a failed route is skipped (0-86400)."),
               token: Optional[str] = _TOKEN, server: Optional[str] = _SERVER,
               data_dir: Path = _DATA_DIR):
    """Set how many routes smart routing tries, and how long a failed one rests."""
    server_url, tok, aid, agent_name = _with_agent(agent, token, server, data_dir)
    with _http(server_url) as client:
        target = _agent_target(client, tok, aid, agent_name)
        _describe(target.call("PUT", "/routing", {
            "max_attempts": max_attempts, "cooldown_seconds": cooldown,
        }), agent_name)


@model_config_app.command("reset")
def mc_reset(agent: str = typer.Argument(..., help="Agent name or id."),
             token: Optional[str] = _TOKEN, server: Optional[str] = _SERVER,
             data_dir: Path = _DATA_DIR):
    """Re-copy your default model config onto this agent.

    Its configs with an API key are kept, after the default's. Needs your
    login or your assistant: an agent cannot reset itself to its owner's
    default."""
    server_url, tok = _resolve_user_session(data_dir, token, server)
    with _http(server_url) as client:
        result = _ok(client.post(
            "/me/model-defaults/apply",
            json={"agents": [agent], "exclude_teams": []},
            headers={"Authorization": f"Bearer {tok}"},
        ))
    _print_apply(result)


# ---------------------------------------------------------------------------
# model-default
# ---------------------------------------------------------------------------

_DEFAULT_BASE = "/me/model-defaults"


def _default_target(data_dir: Path, token: Optional[str], server: Optional[str],
                    as_user: Optional[str] = None):
    server_url, tok = _resolve_user_session(data_dir, token, server, as_user=as_user)
    client = _http(server_url)
    return client, _Target(client, tok, _DEFAULT_BASE, "Your default")


_AS_USER = typer.Option(None, "--as-user", help="Account whose saved session to use.")


@model_default_app.command("show")
def md_show(token: Optional[str] = _TOKEN, server: Optional[str] = _SERVER,
            data_dir: Path = _DATA_DIR, as_user: Optional[str] = _AS_USER):
    """Show your default model config."""
    client, target = _default_target(data_dir, token, server, as_user)
    with client:
        listing = target.get()
    if not listing.get("configured"):
        typer.echo("Your default: not set. New agents use the model tools found on the server.")
        return
    _describe(listing, "Your default")


@model_default_app.command("set")
def md_set(
    providers: list[str] = typer.Argument(..., help="Providers in priority order, e.g. claude openai gemini. No '-api' providers: the default stores no keys."),
    smart: bool = typer.Option(True, "--smart/--single", help="With two or more providers, route between them (default) or use the first only."),
    token: Optional[str] = _TOKEN, server: Optional[str] = _SERVER,
    data_dir: Path = _DATA_DIR, as_user: Optional[str] = _AS_USER,
):
    """Replace your default model config.

    Existing agents are not changed; run `clawmeets model-default apply` for
    that."""
    normalized = [_provider(p) for p in providers]
    client, target = _default_target(data_dir, token, server, as_user)
    with client:
        listing = _put_default(target, normalized, smart=smart)
    _describe(listing, "Your default")
    typer.echo("New agents start with a copy of this. Run `clawmeets model-default apply` to update existing ones.")


@model_default_app.command("order")
def md_order(names: list[str] = typer.Argument(..., help="Configs to try first, in order."),
             token: Optional[str] = _TOKEN, server: Optional[str] = _SERVER,
             data_dir: Path = _DATA_DIR, as_user: Optional[str] = _AS_USER):
    """Set the smart-routing priority of your default."""
    client, target = _default_target(data_dir, token, server, as_user)
    with client:
        _describe(target.reorder(names), "Your default")


@model_default_app.command("routing")
def md_routing(max_attempts: Optional[int] = typer.Option(None, "--max-attempts"),
               cooldown: Optional[int] = typer.Option(None, "--cooldown"),
               token: Optional[str] = _TOKEN, server: Optional[str] = _SERVER,
               data_dir: Path = _DATA_DIR, as_user: Optional[str] = _AS_USER):
    """Set the smart-routing knobs of your default."""
    client, target = _default_target(data_dir, token, server, as_user)
    with client:
        _describe(target.call("PUT", "/routing", {
            "max_attempts": max_attempts, "cooldown_seconds": cooldown,
        }), "Your default")


@model_default_app.command("clear")
def md_clear(token: Optional[str] = _TOKEN, server: Optional[str] = _SERVER,
             data_dir: Path = _DATA_DIR, as_user: Optional[str] = _AS_USER):
    """Delete your default. New agents go back to the server's model tools."""
    client, target = _default_target(data_dir, token, server, as_user)
    with client:
        target.call("DELETE")
    typer.echo("Your default model config is cleared.")


def _print_apply(result: dict) -> None:
    verb = "Would change" if result.get("dry_run") else "Changed"
    changed = result.get("changed") or []
    if not changed:
        typer.echo("No agent needed a change.")
    for a in changed:
        notes = []
        if a.get("removed"):
            notes.append("removed " + ", ".join(a["removed"]))
        if a.get("kept"):
            notes.append("kept keyed " + ", ".join(a["kept"]))
        typer.echo(f"  {verb} {a['agent_name']}" + (f" ({'; '.join(notes)})" if notes else ""))
    for s in result.get("skipped") or []:
        typer.echo(f"  Skipped {s['agent_name']}: {s['reason']}")
    if changed and not result.get("dry_run"):
        typer.echo("Each agent switches on its next turn.")


@model_default_app.command("apply")
def md_apply(
    agent: Optional[list[str]] = typer.Option(None, "--agent", "-a", help="Only these agents (repeatable). Default: all of yours."),
    include_model_agents: bool = typer.Option(False, "--include-model-agents", help=f"Also change agents in the {MODEL_AGENTS_TEAM!r} team."),
    dry_run: bool = typer.Option(False, "--dry-run", help="List what would change; change nothing."),
    token: Optional[str] = _TOKEN, server: Optional[str] = _SERVER,
    data_dir: Path = _DATA_DIR, as_user: Optional[str] = _AS_USER,
):
    """Copy your default onto your existing agents.

    Configs holding an API key are kept, after the default's providers."""
    client, target = _default_target(data_dir, token, server, as_user)
    body: dict = {"dry_run": dry_run}
    if include_model_agents:
        body["exclude_teams"] = []
    if agent:
        body["agents"] = list(agent)
    with client:
        result = target.call("POST", "/apply", body)
    _print_apply(result)


def _put_default(target: _Target, providers: list[str], *, smart: bool) -> dict:
    configs: list[dict] = []
    if smart and len(providers) >= 2:
        configs.append({"name": SMART_CONFIG_NAME, "provider": "smart"})
    configs.extend({"name": p, "provider": p} for p in providers)
    return target.call("PUT", "", {
        "configs": configs, "default": configs[0]["name"] if configs else None,
    })


def _detected_providers() -> list[tuple[str, str]]:
    """(provider, label) for each model CLI installed AND signed in here, in
    ``doctor.MODEL_CLIS`` order — the same order ``pick_provider`` uses."""
    usable = {s.id for s in doctor.inspect_model_clis() if s.present and s.logged_in}
    return [(spec.provider, spec.label) for spec in doctor.MODEL_CLIS if spec.id in usable]


def _ask_order(found: list[tuple[str, str]]) -> list[tuple[str, str]]:
    for i, (_, label) in enumerate(found, 1):
        typer.echo(f"    {i}. {label}")
    default = ",".join(str(i) for i in range(1, len(found) + 1))
    while True:
        raw = typer.prompt("  Order to try them (Enter keeps this)", default=default)
        try:
            picks = [int(x) for x in raw.replace(" ", "").split(",") if x]
        except ValueError:
            picks = []
        if picks and all(1 <= p <= len(found) for p in picks) and len(set(picks)) == len(picks):
            chosen = [found[p - 1] for p in picks]
            return chosen + [f for f in found if f not in chosen]
        typer.echo(f"  Use numbers 1-{len(found)} separated by commas, e.g. {default}.")


@model_default_app.command("setup")
def md_setup(
    ask: bool = typer.Option(True, "--ask/--no-ask", help="Ask in the terminal. Without a terminal it never asks."),
    token: Optional[str] = _TOKEN, server: Optional[str] = _SERVER,
    data_dir: Path = _DATA_DIR, as_user: Optional[str] = _AS_USER,
):
    """Set your default from the model CLIs signed in on this computer.

    Run by the one-line installer, before your assistant is created.

    One CLI: it becomes the default. Two or more: asks whether to turn on smart
    routing and in what order to try them; with no terminal, smart routing in
    the order found. An existing default is kept (asked, with a terminal)."""
    interactive = ask and sys.stdin.isatty()
    client, target = _default_target(data_dir, token, server, as_user)
    with client:
        current = target.get()
        if current.get("configured"):
            typer.echo(f"  Your default model: {_summary_line(current)}")
            if not interactive or typer.confirm("  Keep it?", default=True):
                return

        found = _detected_providers()
        if not found:
            typer.echo("  No signed-in model CLI found on this computer; "
                       "`clawmeets doctor` shows how to add one.")
            return
        smart = len(found) >= 2
        if smart and interactive:
            typer.echo("  Found: " + ", ".join(label for _, label in found))
            smart = typer.confirm(
                "  Enable smart routing (try them in order, moving on when one fails)?",
                default=True,
            )
            if smart:
                found = _ask_order(found)
            else:
                typer.echo("  Which one should answer?")
                found = _ask_order(found)[:1]
        _put_default(target, [p for p, _ in found], smart=smart)
    labels = {p: label for p, label in found}
    if smart:
        chosen = "smart routing -> " + ", ".join(labels[p] for p, _ in found)
    else:
        chosen = labels[found[0][0]]
    typer.echo(f"  Default model for your agents: {chosen}")
    typer.echo("  Change it any time: Account Settings -> Default model, or `clawmeets model-default`.")
