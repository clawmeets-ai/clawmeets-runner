# SPDX-License-Identifier: MIT
"""
clawmeets/cli_consult.py

`clawmeets consult` — invoke another LLM CLI for a focused second-opinion
answer. Used by the `/clawmeets:consult` skill.

Coordinator LLM names the provider(s) (single, or comma-list for parallel
fan-out) and a focused sub-question. consult shells out to each provider's
CLI, captures stdout, and prints back the answer — fenced per-provider in
multi-provider mode. Per-provider failures (binary missing, timeout, non-zero
exit) print as `--- <provider> (FAILED: <reason>) ---` so the coordinator
sees partial results, not a hard error.

Examples
--------
    clawmeets consult claude "what jazz clubs are in greenwich village?"
    clawmeets consult claude,gemini "best UWS dim sum brunch this saturday?"
    clawmeets consult openai "what are the best cocktail bars in LES?"
    clawmeets consult --config $CLAWMEETS_AGENT_DIR/skill-hub/configs/consult.json \\
        gemini "<focused question>"        # threads providers.gemini.model

Validation
----------
Provider must be in {"claude", "openai", "gemini"}. `composite` is rejected
to prevent nesting. `--model` is honored in single-provider mode only;
`--config` model pins apply per-provider in both single and parallel modes
(explicit `--model` wins).

Observability
-------------
One row per sub-call is appended to `{CLAWMEETS_AGENT_DIR}/metadata/
consults.ndjson` when that env var is set. The `error` field is capped at
2 KB so concurrent appends stay under PIPE_BUF for atomic POSIX O_APPEND.
Cost.json is NOT touched — raw CLI invocations don't surface uniform
token counts.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

from clawmeets.utils.file_io import FileUtil
from clawmeets.utils.jsonc import parse_jsonc

logger = logging.getLogger(__name__)

_VALID_PROVIDERS = ("claude", "openai", "gemini")
_DEFAULT_TIMEOUT_SECONDS = 600


def _provider_command(provider: str, model: Optional[str], question: str) -> list[str]:
    if provider == "claude":
        cmd = [
            "claude",
            "--print",
            "--no-session-persistence",
            "--permission-mode", "bypassPermissions",
        ]
        if model:
            cmd += ["--model", model]
        cmd.append(question)
        return cmd
    if provider == "gemini":
        cmd = ["gemini"]
        if model:
            cmd += ["-m", model]
        cmd += ["-p", question]
        return cmd
    if provider == "openai":
        cmd = ["codex", "exec", "--skip-git-repo-check", "--sandbox", "read-only"]
        if model:
            cmd += ["-m", model]
        cmd.append(question)
        return cmd
    raise ValueError(f"unknown provider {provider!r}")


async def _run_one(
    provider: str,
    model: Optional[str],
    question: str,
    timeout: int,
) -> dict:
    started = time.monotonic()
    cmd = _provider_command(provider, model, question)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return {
            "provider": provider, "model": model, "ok": False,
            "answer": None, "error": f"{cmd[0]} not on PATH",
            "duration_ms": int((time.monotonic() - started) * 1000),
            "exit_code": -1,
        }

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            pass
        return {
            "provider": provider, "model": model, "ok": False,
            "answer": None, "error": f"timeout after {timeout}s",
            "duration_ms": int((time.monotonic() - started) * 1000),
            "exit_code": -1,
        }

    answer = (stdout or b"").decode("utf-8", errors="replace").strip()
    err = (stderr or b"").decode("utf-8", errors="replace").strip()
    duration_ms = int((time.monotonic() - started) * 1000)

    if proc.returncode != 0:
        return {
            "provider": provider, "model": model, "ok": False,
            "answer": answer or None,
            "error": err or f"exit code {proc.returncode}",
            "duration_ms": duration_ms,
            "exit_code": proc.returncode,
        }
    return {
        "provider": provider, "model": model, "ok": True,
        "answer": answer,
        "error": None,
        "duration_ms": duration_ms,
        "exit_code": 0,
    }


def _log_consult(result: dict, question: str) -> None:
    agent_dir = os.environ.get("CLAWMEETS_AGENT_DIR")
    if not agent_dir:
        return
    log_path = Path(agent_dir) / "metadata" / "consults.ndjson"
    # Cap error at 2 KB so a giant stderr can't push a row past PIPE_BUF
    # (~4 KB) and break POSIX O_APPEND atomicity for concurrent writers.
    err = result.get("error")
    if err and len(err) > 2048:
        err = err[:2045] + "..."
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "provider": result["provider"],
        "model": result.get("model"),
        "question_chars": len(question),
        "answer_chars": len(result.get("answer") or ""),
        "duration_ms": result["duration_ms"],
        "exit_code": result["exit_code"],
        "ok": result["ok"],
        "error": err,
    }
    try:
        FileUtil.write(log_path, entry, "ndjson", mode="a")
    except Exception as e:
        logger.warning("failed to log consult to %s: %s", log_path, e)


def _load_config_models(config_path: Optional[str]) -> dict[str, str]:
    """Load per-provider model pins from a consult skill config file.

    Returns a {provider_name_lower: model} dict for every provider whose
    config has a non-null `model`. Tolerant: missing file / parse error /
    bad shape → returns {} and emits a stderr warning.
    """
    if not config_path:
        return {}
    try:
        text = Path(config_path).read_text(encoding="utf-8")
        cfg = parse_jsonc(text)
    except Exception as e:
        typer.echo(
            f"Warning: --config {config_path} unreadable ({e}); "
            "proceeding without model pins.",
            err=True,
        )
        return {}
    out: dict[str, str] = {}
    providers = cfg.get("providers") if isinstance(cfg, dict) else None
    if not isinstance(providers, dict):
        return out
    for name, spec in providers.items():
        if isinstance(spec, dict):
            m = spec.get("model")
            if m:
                out[name.lower()] = m
    return out


def consult_command(
    providers: str = typer.Argument(
        ...,
        help=(
            "Provider name, or comma-separated list for parallel "
            f"consultation. One of {_VALID_PROVIDERS}."
        ),
    ),
    question: str = typer.Argument(
        ...,
        help=(
            "The focused sub-question to send to each provider. Keep it "
            "short (1-3 sentences); do NOT paste full conversation context."
        ),
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help=(
            "Provider-specific model override (single-provider mode only; "
            "ignored when multiple providers are listed). Wins over any "
            "model pinned in --config."
        ),
    ),
    timeout: int = typer.Option(
        _DEFAULT_TIMEOUT_SECONDS, "--timeout",
        help="Per-provider timeout in seconds.",
    ),
    config: Optional[str] = typer.Option(
        None, "--config",
        help=(
            "Path to consult.json (typically "
            "$CLAWMEETS_AGENT_DIR/skill-hub/configs/consult.json). "
            "When set, per-provider `providers.<name>.model` pins are applied "
            "to each invocation. Explicit --model overrides the pin."
        ),
    ),
) -> None:
    """Invoke another LLM CLI for a focused second-opinion answer.

    See module docstring for usage and validation rules.
    """
    names = [p.strip().lower() for p in providers.split(",") if p.strip()]
    if not names:
        typer.echo("Error: no providers specified", err=True)
        raise typer.Exit(2)
    seen: set[str] = set()
    deduped: list[str] = []
    for n in names:
        if n == "composite":
            typer.echo(
                "Error: 'composite' is not a consult target (nesting forbidden).",
                err=True,
            )
            raise typer.Exit(2)
        if n not in _VALID_PROVIDERS:
            typer.echo(
                f"Error: unknown provider {n!r} "
                f"(expected one of {_VALID_PROVIDERS}).",
                err=True,
            )
            raise typer.Exit(2)
        if n in seen:
            continue
        seen.add(n)
        deduped.append(n)

    single = len(deduped) == 1
    if model and not single:
        typer.echo(
            "Warning: --model ignored in multi-provider mode "
            "(per-provider model pins from --config still apply).",
            err=True,
        )

    config_models = _load_config_models(config)
    effective_models: dict[str, Optional[str]] = {}
    for n in deduped:
        if single and model:
            effective_models[n] = model        # explicit --model wins
        else:
            effective_models[n] = config_models.get(n)  # may be None

    async def _run_all() -> list[dict]:
        return await asyncio.gather(
            *[_run_one(n, effective_models[n], question, timeout) for n in deduped]
        )

    results = asyncio.run(_run_all())

    for r in results:
        _log_consult(r, question)

    if single:
        r = results[0]
        if not r["ok"]:
            typer.echo(f"Error consulting {r['provider']}: {r['error']}", err=True)
            raise typer.Exit(1)
        typer.echo(r["answer"])
        return

    blocks: list[str] = []
    for r in results:
        if r["ok"]:
            blocks.append(f"--- {r['provider']} ---\n{r['answer']}")
        else:
            blocks.append(f"--- {r['provider']} (FAILED: {r['error']}) ---")
    typer.echo("\n\n".join(blocks))
