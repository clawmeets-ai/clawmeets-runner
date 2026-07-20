# SPDX-License-Identifier: MIT
"""
clawmeets/llm/_harness_tools.py
Single source of truth for the native harness's tool surface.

Two kinds of tool live here:

* **Emitters** — the "JSON via tools" device carried over from the retired
  ``qwen_harness`` experiment. One ``emit_*`` tool binds 1:1 to one action block;
  its parameters map 1:1 onto that block's fields. Structure comes from the tool
  schema, not from the model free-forming JSON. A no-arg ``finalize`` tool ends the
  turn. :class:`EmitterSpec` is the ONE registry driving both the OpenAI function
  schemas offered to the model AND the assembler that folds the calls back into
  ``ActionBlock.actions`` — the structural anti-drift device.

* **Real tools** — the file / bash / skill / web harness. These are NOT
  reimplemented: they reuse the plain-Python callables already living in
  :mod:`clawmeets.llm.api_provider` (importing them does not pull in pydantic-ai,
  whose imports there are lazy), wrapped behind explicit JSON schemas as
  :class:`HarnessTool`. One harness implementation, two providers.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ..api.actions import COORDINATOR_ACTION_SCHEMA
from ._openai_wire import ToolCall
from .api_provider import (
    _build_file_tools,
    _build_skill_tool,
    _build_web_fetch_tool,
    _build_web_search_tool,
)

logger = logging.getLogger(__name__)

FINALIZE_TOOL_NAME = "finalize"


# ---------------------------------------------------------------------------
# Emitters — JSON via tools (one emit_* tool <-> one action block)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmitterSpec:
    """Binds one ``emit_*`` tool 1:1 to one action-block type.

    ``params`` is an ordered ``(name, description)`` tuple; every param is
    required. ``array_params`` names those declared as array-of-string rather
    than string (only ``emit_create_room.invite`` today).
    """

    tool_name: str
    action_type: str
    params: tuple[tuple[str, str], ...]
    array_params: frozenset[str] = field(default_factory=frozenset)


WORKER_EMITTERS: tuple[EmitterSpec, ...] = (
    EmitterSpec(
        "emit_reply",
        "reply",
        (
            ("room", "The chatroom to post the reply to."),
            ("content", "The message text (use @mentions to address agents)."),
        ),
    ),
    EmitterSpec(
        "emit_update_file",
        "update_file",
        (
            ("room", "The chatroom the file belongs to."),
            ("file_path", "Relative path of the file you wrote in your sandbox."),
        ),
    ),
)

COORDINATOR_EMITTERS: tuple[EmitterSpec, ...] = WORKER_EMITTERS + (
    EmitterSpec(
        "emit_create_room",
        "create_room",
        (
            ("name", "Name of the new chatroom."),
            ("invite", "Array of agent names to invite."),
            ("init_message", "Opening message (use @mentions to address agents)."),
        ),
        array_params=frozenset({"invite"}),
    ),
    EmitterSpec("emit_project_completed", "project_completed", ()),
)


def emitters_for(action_schema: dict) -> tuple[EmitterSpec, ...]:
    """COORDINATOR emitters iff the caller selected the coordinator schema."""
    return (
        COORDINATOR_EMITTERS
        if action_schema == COORDINATOR_ACTION_SCHEMA
        else WORKER_EMITTERS
    )


def build_emitter_schemas(emitters: tuple[EmitterSpec, ...]) -> list[dict]:
    """One OpenAI function schema per emitter (params → required props,
    ``additionalProperties: false``), plus the no-arg ``finalize`` tool."""
    schemas: list[dict] = []
    for spec in emitters:
        props: dict[str, dict] = {}
        required: list[str] = []
        for name, desc in spec.params:
            if name in spec.array_params:
                props[name] = {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": desc,
                }
            else:
                props[name] = {"type": "string", "description": desc}
            required.append(name)
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": spec.tool_name,
                    "description": (
                        f"Emit one `{spec.action_type}` action block. "
                        "Call once per block you want to produce."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": required,
                        "additionalProperties": False,
                    },
                },
            }
        )
    schemas.append(
        {
            "type": "function",
            "function": {
                "name": FINALIZE_TOOL_NAME,
                "description": (
                    "Call once, after every emit_* call, to end your turn. Takes no "
                    "arguments."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        }
    )
    return schemas


def _normalize_array_param(value: object) -> list:
    """Coerce an array param to a list (models sometimes send a JSON string)."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
        except (json.JSONDecodeError, ValueError):
            return [s]
        return parsed if isinstance(parsed, list) else [str(parsed)]
    return [value]


def assemble(
    emit_calls: list[ToolCall], emitters: tuple[EmitterSpec, ...]
) -> list[dict]:
    """Fold ordered ``emit_*`` calls into ``ActionBlock.actions``.

    Order is preserved (it is semantically meaningful). For each call: look up its
    :class:`EmitterSpec`, then build ``{"type": action_type, **only-declared-params}``
    — extra/hallucinated keys are dropped (key-exactness). A call missing a required
    param, or naming an unknown emitter, is DROPPED with a logged warning rather than
    crashing the turn (a bad tail call must not lose the good blocks). ``finalize`` is
    not an emitter and never reaches here.
    """
    by_name = {s.tool_name: s for s in emitters}
    actions: list[dict] = []
    for call in emit_calls:
        spec = by_name.get(call.name)
        if spec is None:
            logger.warning("[native] dropping unknown emitter %r", call.name)
            continue
        args = call.arguments or {}
        block: dict = {"type": spec.action_type}
        missing = [name for name, _ in spec.params if name not in args]
        if missing:
            logger.warning(
                "[native] dropping %s: missing required param(s) %s",
                call.name,
                missing,
            )
            continue
        for name, _ in spec.params:
            value = args[name]
            if name in spec.array_params:
                value = _normalize_array_param(value)
            block[name] = value
        actions.append(block)
    return actions


# ---------------------------------------------------------------------------
# Real tools — the file/bash/skill/web harness, reused from api_provider
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HarnessTool:
    """A real, side-effecting tool: an OpenAI schema + the callable behind it."""

    name: str
    schema: dict
    fn: Callable
    is_async: bool
    arg_names: tuple[str, ...]


def _tool(
    name: str,
    fn: Callable,
    *,
    is_async: bool,
    description: str,
    properties: dict[str, dict],
    required: list[str],
) -> HarnessTool:
    return HarnessTool(
        name=name,
        schema={
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        },
        fn=fn,
        is_async=is_async,
        arg_names=tuple(properties.keys()),
    )


def build_harness_tools(
    working_dir: Path,
    env: dict[str, str],
    read_roots: list[Path],
    skill_source_dirs: list[Path],
    web_budget: int,
    web_fetch_budget: int,
    enable_web: bool,
    write_roots: Optional[list[Path]] = None,
) -> list[HarnessTool]:
    """Wrap ``api_provider``'s plain callables behind explicit JSON schemas.

    Writes stay sandbox-confined exactly as ``_build_file_tools`` enforces (the
    reused callables guard writes against ``working_dir``), plus any extra
    ``write_roots`` (today the agent ``memory_dir``, so reflect/personalize
    writebacks succeed); reads span the sandbox plus ``read_roots``. This is why
    there is ONE harness implementation.
    """
    file_fns = {
        f.__name__: f
        for f in _build_file_tools(
            working_dir, env, read_roots=read_roots, write_roots=write_roots
        )
    }
    tools: list[HarnessTool] = [
        _tool(
            "read_file",
            file_fns["read_file"],
            is_async=False,
            description="Read a UTF-8 file (sandbox or a readable project/knowledge dir).",
            properties={"path": {"type": "string", "description": "File path."}},
            required=["path"],
        ),
        _tool(
            "write_file",
            file_fns["write_file"],
            is_async=False,
            description="Write a UTF-8 file under the sandbox working dir (relative path).",
            properties={
                "path": {"type": "string", "description": "Sandbox-relative path."},
                "content": {"type": "string", "description": "Full file contents."},
            },
            required=["path", "content"],
        ),
        _tool(
            "edit_file",
            file_fns["edit_file"],
            is_async=False,
            description="Replace the first occurrence of `old` with `new` in a sandbox file.",
            properties={
                "path": {"type": "string", "description": "Sandbox-relative path."},
                "old": {"type": "string", "description": "Exact text to replace."},
                "new": {"type": "string", "description": "Replacement text."},
            },
            required=["path", "old", "new"],
        ),
        _tool(
            "list_dir",
            file_fns["list_dir"],
            is_async=False,
            description="List entries of a directory (defaults to the sandbox root).",
            properties={"path": {"type": "string", "description": "Directory path."}},
            required=[],
        ),
        _tool(
            "glob",
            file_fns["glob"],
            is_async=False,
            description="Glob for files across the sandbox and readable dirs.",
            properties={"pattern": {"type": "string", "description": "Glob pattern."}},
            required=["pattern"],
        ),
        _tool(
            "grep",
            file_fns["grep"],
            is_async=False,
            description="Regex-search files under a directory; returns file:line: text.",
            properties={
                "pattern": {"type": "string", "description": "Regular expression."},
                "path": {"type": "string", "description": "Directory to search."},
            },
            required=["pattern"],
        ),
        _tool(
            "bash",
            file_fns["bash"],
            is_async=True,
            description="Run a shell command in the sandbox cwd; returns exit code + output.",
            properties={"command": {"type": "string", "description": "Shell command."}},
            required=["command"],
        ),
        _tool(
            "skill",
            _build_skill_tool(skill_source_dirs or []),
            is_async=False,
            description="Load an installed skill's SKILL.md instructions by name.",
            properties={"name": {"type": "string", "description": "Skill name."}},
            required=["name"],
        ),
    ]
    if enable_web:
        tools.append(
            _tool(
                "web_search",
                _build_web_search_tool(web_budget),
                is_async=False,
                description="Search the web and return the top results as text.",
                properties={"query": {"type": "string", "description": "Search query."}},
                required=["query"],
            )
        )
        tools.append(
            _tool(
                "web_fetch",
                _build_web_fetch_tool(web_fetch_budget),
                is_async=False,
                description="Fetch a web page and return its main text as markdown.",
                properties={"url": {"type": "string", "description": "Page URL."}},
                required=["url"],
            )
        )
    return tools


def build_mcp_tools(mcp_config_dir: Optional[Path], env: dict[str, str]) -> list[HarnessTool]:
    """MCP tool surface — a documented M2 fast-follow.

    The plan flags MCP as the heaviest new piece and explicitly allows it to ship
    later without blocking the core reply / update_file loop (§9). Returns ``[]``
    for now; the reply/update_file/file/bash/skill/web surface is fully functional
    without it. A later revision will read ``{mcp_config_dir}/.mcp.json`` and surface
    each server's tools as async :class:`HarnessTool`s, mirroring
    ``ApiLLMProvider._build_mcp_toolsets``.
    """
    if mcp_config_dir is not None:
        logger.info("[native] MCP tools not yet wired (M2 fast-follow); skipping.")
    return []
