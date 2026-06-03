# SPDX-License-Identifier: MIT
"""
clawmeets/llm/prompt_builder.py
Prompt construction for agent participants.

This module is part of Layer 0 (pure - no domain model dependencies).
It provides prompt building utilities for worker and coordinator agents.

Classes defined here:
- OperationalMode: Enum for participant operational modes (worker/coordinator)
- PromptBuilder: Base class with shared utilities
- WorkerPromptBuilder: Builds prompts for worker agents
- CoordinatorPromptBuilder: Builds prompts for coordinator/assistant agents

Helper functions:
- create_prompt_builder: Factory to create prompt builder based on operational mode
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# OperationalMode Enum (Layer 0 - no dependencies)
# ---------------------------------------------------------------------------

class OperationalMode(str, Enum):
    """Operational mode of a participant within a project.

    Determines the participant's behavior and available actions.
    Mode is derived at runtime from project.coordinator_id, not stored.

    Defined here in Layer 0 to avoid circular imports. Re-exported from
    models.participant for backward compatibility.
    """
    WORKER = "worker"          # Responds when @mentioned, limited actions (reply, update_file)
    COORDINATOR = "coordinator"  # Orchestrates work, full actions (create_room, project_completed)


# ---------------------------------------------------------------------------
# PromptBuilder Classes
# ---------------------------------------------------------------------------

class PromptBuilder:
    """
    Base prompt builder with shared utilities.

    Provides common methods for building file manifests, action documentation,
    and other shared prompt components.
    """

    _git_ignored_folder: Optional[str] = None

    def build_file_manifest(self, data_dir: Path) -> str:
        """
        Build a manifest of local files for context.

        Args:
            data_dir: Directory containing project data files

        Returns:
            Formatted string listing all files
        """
        files: list[str] = []
        if data_dir.exists():
            for fp in sorted(data_dir.rglob("*")):
                if fp.is_file():
                    files.append(str(fp.relative_to(data_dir)))

        return "\n".join(f"  - {f}" for f in files) if files else "  (empty)"

    def build_actions_doc(self) -> str:
        """Build documentation for available actions. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement build_actions_doc()")

    def _build_memory_section(self, name: str, agent_dir: Path) -> str:
        """Role-aware KNOWLEDGE PRECEDENCE block.

        All agent-authored memory lives under ``{agent_dir}/memory/``
        (assistant's `USER.md`, `REFERENCES.md`, `KNOWLEDGE_PACKS.md`,
        `learnings/INDEX.md` + topic pages). Installed knowledge packs
        live alongside at ``{agent_dir}/knowledge_packs/`` — the index in
        memory/ links into that sibling with absolute paths.

        The user's personal assistant is the agent named
        ``{username}-assistant``. That agent maintains both ``USER.md`` and
        ``learnings/``. Worker agents maintain only ``learnings/``.

        This block is purely declarative — agents read these files at any
        time, but writes only happen during scheduled reflection (via the
        ``/clawmeets:reflect`` skill).
        """
        memory_dir = f"{agent_dir}/memory"
        is_assistant = name.endswith("-assistant")
        if is_assistant:
            return f"""
== KNOWLEDGE PRECEDENCE (binding) ==

Your agent memory (under {memory_dir}/) has two independent layers, with
strict precedence at runtime:

1. AUTHORITATIVE — files the user has authored or that you have curated about
   the user themselves:
   - {memory_dir}/USER.md            what you know about this user (you are their assistant)
   - {memory_dir}/REFERENCES.md      index of user-pre-seeded reference files
                                     in the user's knowledge_dir (each entry
                                     tells you when to consult it)
   - {memory_dir}/KNOWLEDGE_PACKS.md index of knowledge packs the user has
                                     explicitly installed on you (each entry
                                     names a curated pack with absolute paths
                                     into {agent_dir}/knowledge_packs/)
   For ANY question about the user themselves OR the user's actual world
   (their business, product, life, projects, preferences, domain facts),
   the answer comes from these files. Do not synthesize from learnings/
   on user-world or user-personal questions.

2. FALLBACK — your distilled learnings, indexed at
   {memory_dir}/learnings/INDEX.md. This is your general background —
   industry frameworks, field context, regulations, comp data — things
   that are true about the *field*, not about *this user*. Use learnings/
   for field/industry questions, not user-specific ones.

The two layers are decoupled — neither indexes or references the other.
When a question spans both: start with the authoritative layer for the
user-world facts, then consult learnings/ for any generic context that
fills gaps the authoritative files don't cover.

Other memory files:
- {memory_dir}/learnings/log.md       append-only "## [YYYY-MM-DD] event | title"
- {memory_dir}/learnings/<topic>.md   drill-down pages, cross-linked from INDEX.md

When you receive a DM tagged with one of these HTML-comment markers, follow
the matching skill — they are the only times you should write to memory:
- <!-- clawmeets:reflect-trigger -->                /clawmeets:reflect
- <!-- clawmeets:lint-trigger -->                   /clawmeets:lint
- <!-- clawmeets:references-trigger -->             /clawmeets:references
- <!-- clawmeets:interview-trigger -->              /clawmeets:interview
- <!-- clawmeets:rerun-{{slug}} --> [optional ask]  /clawmeets:rerun-project
"""
        return f"""
== KNOWLEDGE PRECEDENCE (binding) ==

Your agent memory (under {memory_dir}/) has two independent layers, with
strict precedence at runtime:

1. AUTHORITATIVE — the user's pre-seeded reference files
   ({memory_dir}/REFERENCES.md) and any knowledge packs the user has
   explicitly installed on you ({memory_dir}/KNOWLEDGE_PACKS.md — an index
   with absolute paths into {agent_dir}/knowledge_packs/). Together these
   are the user's own description of their world (their business, product,
   customers, operations, positioning, domain facts). For ANY question
   about the user's actual world, the answer comes from these files. Do
   not synthesize from learnings/ on user-world questions.

2. FALLBACK — your distilled learnings, indexed at
   {memory_dir}/learnings/INDEX.md. This is your general background —
   industry frameworks, field context, regulations, comp data — things
   that are true about the *field*, not about *this user*. Use learnings/
   for field/industry questions, not user-specific ones.

The two layers are decoupled — neither indexes or references the other.
When a question spans both ("how should I price X for account Y"): start
with the authoritative layer (REFERENCES.md, KNOWLEDGE_PACKS.md) for the
user-world facts (account tier, pricing rules, installed tactics), then
consult learnings/ for any generic industry context that fills gaps the
seeded files don't cover.

Other memory files:
- {memory_dir}/learnings/log.md       append-only "## [YYYY-MM-DD] event | title"
- {memory_dir}/learnings/<topic>.md   drill-down pages, cross-linked from INDEX.md

User-identity facts (general preferences, personal info) live with the
user's assistant, not here.

When you receive a DM tagged with one of these HTML-comment markers, follow
the matching skill — they are the only times you should write to memory:
- <!-- clawmeets:reflect-trigger -->                /clawmeets:reflect
- <!-- clawmeets:lint-trigger -->                   /clawmeets:lint
- <!-- clawmeets:references-trigger -->             /clawmeets:references
- <!-- clawmeets:personalize-trigger -->            /clawmeets:personalize
"""

    def _build_git_guidance(self) -> str:
        """Build git-specific file guidance when git_ignored_folder is configured."""
        if not self._git_ignored_folder:
            return ""
        return f"""
GIT-AWARE FILE MANAGEMENT:
Your working directory is a git repository. Files are categorized as:
- **Code files**: Write to repo paths (e.g. src/module.py, tests/test_new.py)
  These are tracked by git and will be committed to the project branch.
- **Deliverables**: Write to {self._git_ignored_folder}/ (e.g. {self._git_ignored_folder}/REPORT.md)
  These are git-ignored but still shared via update_file through the changelog.

IMPORTANT: Always write files directly in your working directory (e.g. {self._git_ignored_folder}/report.md).
Do NOT write into chatrooms/ subdirectories - that is read-only synced data from another location.

Use update_file for BOTH types - the system handles git vs changelog separation automatically.
"""

    def _build_base_prompt(
        self,
        name: str,
        description: str,
        project_id: str,
        chatroom_name: str,
        from_participant_name: str,
        message_content: str,
        data_dir: Path,
        role_guidance: str,
        project_name: str,
        agent_dir: Path,
        capabilities_line: str = "",
        knowledge_dirs: list[Path] | None = None,
        dwh_dir: Optional[Path] = None,
        mcp_config_files: dict[str, Path] | None = None,
        skill_config_files: dict[str, Path] | None = None,
    ) -> str:
        """
        Build the base prompt structure used by both worker and coordinator.

        Server-First Sync Architecture:
        - data_dir: Synced directory (read-only, contains files from server)
        - Working directory is set by the CLI (sandbox) - use relative paths

        Args:
            name: Agent/assistant name
            description: Agent/assistant description
            project_id: The project ID
            chatroom_name: The chatroom name
            from_participant_name: Name of the message sender
            message_content: Content of the incoming message
            data_dir: Data directory for file manifest (synced, read-only)
            role_guidance: Role-specific guidance section
            capabilities_line: Optional capabilities line for workers
            project_name: Human-readable project name
            knowledge_dirs: Optional list of knowledge base directories (read-write, persistent)
            mcp_config_files: Per-installed-MCP config paths (rendered into
                ``MCP CONFIG FILES`` block).
            skill_config_files: Per-installed-skill config paths (rendered
                into ``SKILL CONFIG FILES`` block — the LLM Reads these
                before deciding whether to invoke each skill).

        Returns:
            Complete prompt string
        """
        file_manifest = self.build_file_manifest(data_dir)
        actions_doc = self.build_actions_doc()

        cap_section = f"\nCapabilities: {capabilities_line}" if capabilities_line else ""

        # User-curated reference material (their own notes, PDFs, docs).
        knowledge_section = ""
        if knowledge_dirs:
            paths = "\n".join(f"- {d}" for d in knowledge_dirs)
            knowledge_section = f"""
== KNOWLEDGE BASE (user-curated reference material — read-only browsing) ==
{paths}

User-pre-seeded files. The index of these files lives at
{agent_dir}/memory/REFERENCES.md with absolute paths; consult it to decide
which to read.
"""

        # Agent-authored memory under {agent_dir}/memory/ — always present
        # on the runner.
        agent_memory_section = f"""
== AGENT MEMORY (durable state — read/write, runner-managed) ==
{agent_dir}/memory/

Holds your USER.md (assistant only), REFERENCES.md, KNOWLEDGE_PACKS.md,
and learnings/. See KNOWLEDGE PRECEDENCE below for how to use them.
"""

        # Installed knowledge packs (server-synced content under its own dir).
        packs_section = f"""
== KNOWLEDGE PACKS (installed; auto-synced from server) ==
{agent_dir}/knowledge_packs/

Pack content (text and binary). The index at
{agent_dir}/memory/KNOWLEDGE_PACKS.md links into here with absolute paths.
"""

        memory_section = self._build_memory_section(name, agent_dir)

        dwh_section = ""
        if dwh_dir is not None:
            dwh_section = f"""
== DATA WAREHOUSE ==
{dwh_dir}
"""

        # Per-MCP config file paths — pass these verbatim as `config_file`
        # to sync tools.
        mcp_configs_section = ""
        if mcp_config_files:
            mcp_lines = "\n".join(
                f"- {mcp}: {path}" for mcp, path in sorted(mcp_config_files.items())
            )
            mcp_configs_section = f"""
== MCP CONFIG FILES (pass these paths to sync tools as `config_file`) ==
{mcp_lines}
"""

        # Per-skill config file paths — READ these before invoking a skill so
        # operator-set per-skill policy (e.g. clawmeets-consult's
        # `invoke_when`, `providers.<n>.use_for`) actually informs routing.
        skill_configs_section = ""
        if skill_config_files:
            skill_lines = "\n".join(
                f"- {skill}: {path}" for skill, path in sorted(skill_config_files.items())
            )
            skill_configs_section = f"""
== SKILL CONFIG FILES (operator-set per-agent policy — Read before invoking a skill) ==
{skill_lines}
"""

        return f"""You are {name}, an AI agent.
Description: {description}{cap_section}

Project: {project_name}
Chatroom: {chatroom_name}

== SYNCED PROJECT FILES (read-only) ==
Files synced from server, available in {data_dir}:
{file_manifest}
{knowledge_section}{agent_memory_section}{packs_section}{memory_section}{mcp_configs_section}{skill_configs_section}{dwh_section}
== YOUR WORKING DIRECTORY ==
Use relative paths to write files. Files you write will be synced to the server and shared with all participants.

Incoming message from {from_participant_name}:
{message_content}

{actions_doc}
{role_guidance}

ROOM REFERENCES: Use chatroom name "{chatroom_name}" in "room" fields (exact match required).

FILE PATHS: Use relative paths from your working directory (e.g. report.md, subdir/file.py)

CRITICAL: Your output MUST be valid JSON matching the structured output schema.
After analyzing the situation, output your actions as a JSON object with an "actions" array.
Include multiple actions if needed. If no action is required, output: {{"actions": []}}

Emit ALL of your actions in a single structured-output response. If you realize
mid-turn that another action is needed, add it to the same actions array — a
second structured-output call in the same turn may REPLACE the first, not
append to it.
"""


class WorkerPromptBuilder(PromptBuilder):
    """
    Builds prompts for worker agents.

    Worker agents respond to coordinator requests and report results.
    They don't delegate work to others - only reply and update_file actions.
    """

    def __init__(
        self,
        coordinator_name: str,
        capabilities: Optional[list[str]] = None,
        git_ignored_folder: Optional[str] = None,
    ) -> None:
        """
        Initialize the worker prompt builder.

        Args:
            coordinator_name: Name of the project coordinator (for guidance)
            capabilities: List of agent capabilities
            git_ignored_folder: Folder for git-ignored deliverables (None if no git config)
        """
        self._coordinator_name = coordinator_name
        self._capabilities = capabilities or []
        self._git_ignored_folder = git_ignored_folder

    def build_actions_doc(self) -> str:
        """Build worker-specific action documentation (no create_room, no @mentions)."""
        doc = """
== STRUCTURED OUTPUT FORMAT ==
Your response will be validated against a JSON schema. You MUST output your actions
in a structured JSON format with an "actions" array.

Available actions (workers only have reply and update_file):
  {"type": "reply", "room": "<chatroom_name>", "content": "<text>"}
  {"type": "update_file", "room": "<chatroom_name>", "file_path": "<relative_path>"}

OUTPUT FORMAT (required structure):
{
  "actions": [
    {"type": "reply", "room": "<chatroom_name>", "content": "Response text"}
  ]
}

If no actions needed, output: {"actions": []}

FILE SHARING WORKFLOW - SERVER-FIRST ARCHITECTURE:
Your working directory is a SANDBOX - files you write here will be pushed to
the server and synced back to all participants via the changelog.

To share a file with other agents:
1. Use the Write tool to create/modify files in your working directory
2. Use the update_file action with the same file_path you used to write
3. The file content is automatically read and sent to the server
4. The server syncs the file to all participants

Example: To share a report in the current chatroom:
Step 1 - Write the file to your working directory:
Use Write tool with:
  file_path: report.md
  content: |
    # Report Title
    Content here...

Step 2 - Emit the update_file action with the SAME path:
{
  "actions": [
    {"type": "update_file", "room": "<chatroom_name>", "file_path": "report.md"}
  ]
}

NOTE: Existing project files are available READ-ONLY from the synced directory.
You can read them but should not modify them directly - write to your working directory instead.

== MEMORY FILES — DIFFERENT RULE ==
The "Write then update_file" pattern above is ONLY for chatroom-visible
deliverables in your sandbox.

Files under your agent's `memory/` directory (USER.md, REFERENCES.md,
KNOWLEDGE_PACKS.md, learnings/<topic>.md) and under your agent's
`knowledge_packs/<slug>/...` directory are AUTHORITATIVE memory and
stay invisible to chat by design.

For memory files:
- Use the Write tool ONLY (write directly to the absolute memory path
  shown in the AGENT MEMORY block above, e.g.
  `Write(file_path="<agent_dir>/memory/USER.md", content="...")`).
- Do NOT emit `update_file` afterwards. `update_file` is exclusively for
  sandbox-relative chatroom files; emitting it for a memory file would
  broadcast memory state into the chat (leak) and the upload itself
  fails server-side because absolute paths produce malformed URLs.
- The file persists on the runner's filesystem and the next prompt
  invocation will see it. No publish step required.
"""
        return doc + self._build_git_guidance()

    def build_prompt(
        self,
        name: str,
        description: str,
        project_id: str,
        chatroom_name: str,
        from_participant_name: str,
        message_content: str,
        data_dir: Path,
        project_name: str,
        agent_dir: Path,
        knowledge_dirs: list[Path] | None = None,
        is_dm: bool = False,
        dwh_dir: Optional[Path] = None,
        mcp_config_files: dict[str, Path] | None = None,
        skill_config_files: dict[str, Path] | None = None,
    ) -> str:
        """
        Build a worker-specific prompt.

        Server-First Sync Architecture:
        - data_dir: Synced directory (read-only)
        - Working directory is set by the CLI (sandbox) - use relative paths

        Args:
            name: Agent name
            description: Agent description
            project_id: The project ID
            chatroom_name: The chatroom name
            from_participant_name: Name of the message sender
            message_content: Content of the incoming message
            data_dir: Data directory (synced, read-only)
            project_name: Human-readable project name
            knowledge_dirs: Optional knowledge base directories (read-write, persistent)
            is_dm: When True, use the direct-message guidance (no coordinator,
                no PLAN.md, no milestones, no acceptance-criteria reply format)
            mcp_config_files: Per-MCP runtime config file paths (mapping
                MCP name -> file path under ``{agent_dir}/mcp-hub/configs/``).
                Surfaced in the prompt's ``MCP CONFIG FILES`` block so the
                agent passes them verbatim as ``config_file`` to sync tools.
            skill_config_files: Per-installed-skill config file paths
                (mapping skill name -> file path under
                ``{agent_dir}/skill-hub/configs/``). Surfaced in the prompt's
                ``SKILL CONFIG FILES`` block so the LLM Reads operator-set
                per-skill policy before invoking each skill.

        Returns:
            Complete worker prompt
        """
        guidance = (
            self._build_dm_guidance()
            if is_dm
            else self._build_worker_guidance()
        )
        capabilities_line = ", ".join(self._capabilities) if self._capabilities else "general"

        return self._build_base_prompt(
            name=name,
            description=description,
            project_id=project_id,
            chatroom_name=chatroom_name,
            from_participant_name=from_participant_name,
            message_content=message_content,
            data_dir=data_dir,
            role_guidance=guidance,
            capabilities_line=capabilities_line,
            project_name=project_name,
            agent_dir=agent_dir,
            knowledge_dirs=knowledge_dirs,
            dwh_dir=dwh_dir,
            mcp_config_files=mcp_config_files,
            skill_config_files=skill_config_files,
        )

    def _build_worker_guidance(self) -> str:
        """Build worker role guidance."""
        return f"""
== WORKER ROLE ==
You are a WORKER agent. The coordinator ({self._coordinator_name}) orchestrates the project and delegates tasks to you.

== WORKER RESPONSIBILITIES ==
1. **CONFIRM** your understanding of the task before executing
2. **EXECUTE** your assigned task completely
3. **VERIFY** your deliverables against acceptance criteria
4. **REPORT** results using the structured reply format

== PROJECT CONTEXT - CHECK BEFORE STARTING ==
BEFORE starting your task, check the synced project files for:

**PLAN.md** (IMPORTANT): Contains project goals, milestones, and current status
- Understand which milestone you're contributing to
- Check guardrails/constraints to follow
- Review acceptance criteria for your milestone
- Review learnings from previous work to avoid repeating mistakes

== TASK COMPLETION CHECKLIST ==
Before reporting completion, verify against the ACCEPTANCE CRITERIA provided in:
1. The delegation message from the coordinator
2. PLAN.md milestone definition
- [ ] Each acceptance criterion is met
- [ ] Deliverables created (files written via update_file action)
- [ ] Results reported using structured reply format
- [ ] Any unresolved items documented

== STRUCTURED REPLY FORMAT ==
Your reply MUST use this format:

**Task completed:**
{{"type": "reply", "room": "<room>", "content": "**Task:** [Brief restatement of what you were asked to do]\\n\\n**Status:** COMPLETE\\n**Deliverables:** [list of files created via update_file]\\n**Summary:** [what was done and key findings]\\n**Acceptance Criteria:**\\n- [x] [criterion 1]: [brief evidence]\\n- [x] [criterion 2]: [brief evidence]\\n**Unresolved:** none"}}

**Task partially completed or blocked:**
{{"type": "reply", "room": "<room>", "content": "**Task:** [Brief restatement of what you were asked to do]\\n\\n**Status:** BLOCKED (or PARTIAL)\\n**Deliverables:** [any files created so far]\\n**Summary:** [what was done so far]\\n**Blocker:** [what is blocking progress]\\n**Proposed assumption:** [what you would assume if proceeding]\\n**Risk of assumption:** [what could go wrong]\\n**Unresolved:** [open items]"}}

**Task beyond capability:**
{{"type": "reply", "room": "<room>", "content": "**Task:** [Brief restatement]\\n\\n**Status:** CANNOT_COMPLETE\\n**Reason:** [what capability is missing]\\n**Recommendation:** [alternative agent or approach]"}}

== CRITICAL RULES ==
- Do NOT use @mentions (workers don't delegate)
- Do NOT post to user-communication (coordinator handles user contact)
- Do NOT ask the coordinator questions — instead report blockers with your proposed assumption and the risk (see format above). This avoids expensive round-trips.
- Focus on your assigned task only

== REPORTING BLOCKERS ==
When you encounter a blocker, do NOT ask a question. Instead:

1. **State the blocker** clearly
2. **Propose an assumption** you would make if proceeding
3. **Assess the risk** of that assumption being wrong

The coordinator will either:
- Accept your assumption (you proceed on next invocation)
- Redirect with updated instructions
- Escalate to the user if domain knowledge is needed
"""

    def _build_dm_guidance(self) -> str:
        """Build guidance for a direct-message (1:1 with user) conversation.

        DMs have no coordinator, no PLAN.md, no milestones, and no other
        agents in the room. Drop the coordinator-delegation framing and
        structured reply template that the worker prompt uses.
        """
        return """
== DIRECT MESSAGE CONVERSATION ==
You are in a 1:1 direct message with the user. There is no project plan,
no milestones, and no other agents in this conversation. The user is
talking to you directly within your area of expertise.

== HOW TO RESPOND ==
1. Answer or do what the user asks, within your capabilities.
2. If the request is outside your expertise, say so plainly. Do NOT
   pretend to delegate to other agents — no one else can see this
   conversation, and @mentions you write here are ignored by the system.
3. If you need to produce a file (report, design, document, etc.), write
   it in your working directory and emit an update_file action. The file
   will appear in this DM for the user to read and download.
4. Respond conversationally. Do NOT use a Task / Status / Deliverables /
   Acceptance Criteria template — that structure is for coordinated
   project work, not direct messages.

== WHAT NOT TO DO ==
- Do NOT create or reference PLAN.md — it does not apply here.
- Do NOT frame your work as "Milestone 1 / Milestone 2 …" unless the
  user explicitly asked for a plan.
- Do NOT @mention other agents as if delegating — you cannot delegate
  from a DM.
- Do NOT ask the user clarifying questions reflexively. If the request
  is actionable, act; only ask when truly blocked.
"""


class CoordinatorPromptBuilder(PromptBuilder):
    """
    Builds prompts for coordinator/assistant agents.

    Coordinators orchestrate work by delegating to worker agents
    and handling user communication. They have access to all actions.
    """

    def __init__(self, git_ignored_folder: Optional[str] = None) -> None:
        """Initialize the coordinator prompt builder.

        Args:
            git_ignored_folder: Folder for git-ignored deliverables (None if no git config)
        """
        self._git_ignored_folder = git_ignored_folder

    def build_actions_doc(self) -> str:
        """Build coordinator-specific action documentation (all actions including delegation)."""
        doc = """
== STRUCTURED OUTPUT FORMAT ==
Your response will be validated against a JSON schema. You MUST output your actions
in a structured JSON format with an "actions" array.

Available actions:
  {"type": "reply", "room": "<chatroom_name>", "content": "<text>"}
  {"type": "update_file", "room": "<chatroom_name>", "file_path": "<relative_path>"}
  {"type": "create_room", "name": "<name>", "invite": ["<agent_name>"], "init_message": "<text>"}
  {"type": "project_completed"} (marks project complete)

OUTPUT FORMAT (required structure):
{
  "actions": [
    {"type": "reply", "room": "<chatroom_name>", "content": "Response text"}
  ]
}

If no actions needed, output: {"actions": []}

Emit ALL of your actions in a single structured-output response. If you realize
mid-turn that another action is needed, add it to the same actions array — a
second structured-output call in the same turn may REPLACE the first, not
append to it.

== @MENTION ADDRESSING ==
Messages are shared with ALL participants in the room as context — everyone can read them.
@mentions control WHO RESPONDS:
- "@agent-name" = agent is expected to respond
- "agent-name" (no @) or no mention = agent reads the message but does NOT respond

Only @mention agents you need a response from in this batch. For example, if one agent
should lead and orchestrate others in the room, only @mention that agent — the others
will see the message as context and can be @mentioned later by the lead.

Examples:
- "@researcher please analyze this data" -> researcher responds; others read as context
- "The researcher completed the task" -> no one responds; informational
- "@researcher and @writer please collaborate" -> both respond in parallel

FILE SHARING WORKFLOW - SERVER-FIRST ARCHITECTURE:
Your working directory is a SANDBOX - files you write here will be pushed to
the server and synced back to all participants via the changelog.

To share a file with other agents:
1. Use the Write tool to create/modify files in your working directory
2. Use the update_file action with the same file_path you used to write
3. The file content is automatically read and sent to the server
4. The server syncs the file to all participants

Example: To share a report in the current chatroom:
Step 1 - Write the file to your working directory:
Use Write tool with:
  file_path: report.md
  content: |
    # Report Title
    Content here...

Step 2 - Emit the update_file action with the SAME path:
{
  "actions": [
    {"type": "update_file", "room": "<chatroom_name>", "file_path": "report.md"}
  ]
}

NOTE: Existing project files are available READ-ONLY from the synced directory.
You can read them but should not modify them directly - write to your working directory instead.

== MEMORY FILES — DIFFERENT RULE ==
The "Write then update_file" pattern above is ONLY for chatroom-visible
deliverables in your sandbox.

Files under your agent's `memory/` directory (USER.md, REFERENCES.md,
KNOWLEDGE_PACKS.md, learnings/<topic>.md) and under your agent's
`knowledge_packs/<slug>/...` directory are AUTHORITATIVE memory and
stay invisible to chat by design.

For memory files:
- Use the Write tool ONLY (write directly to the absolute memory path
  shown in the AGENT MEMORY block above, e.g.
  `Write(file_path="<agent_dir>/memory/USER.md", content="...")`).
- Do NOT emit `update_file` afterwards. `update_file` is exclusively for
  sandbox-relative chatroom files; emitting it for a memory file would
  broadcast memory state into the chat (leak) and the upload itself
  fails server-side because absolute paths produce malformed URLs.
- The file persists on the runner's filesystem and the next prompt
  invocation will see it. No publish step required.
"""
        return doc + self._build_git_guidance()

    def build_prompt(
        self,
        name: str,
        description: str,
        project_id: str,
        chatroom_name: str,
        from_participant_name: str,
        message_content: str,
        data_dir: Path,
        project_name: str,
        agent_dir: Path,
        knowledge_dirs: list[Path] | None = None,
        dwh_dir: Optional[Path] = None,
        is_front_desk: bool = False,
        invitable_agents: Optional[list[str]] = None,
        mcp_config_files: dict[str, Path] | None = None,
        skill_config_files: dict[str, Path] | None = None,
    ) -> str:
        """
        Build a coordinator-specific prompt.

        Server-First Sync Architecture:
        - data_dir: Synced directory (read-only)
        - Working directory is set by the CLI (sandbox) - use relative paths

        Args:
            name: Assistant name
            description: Assistant description
            project_id: The project ID
            chatroom_name: The chatroom name
            from_participant_name: Name of the message sender
            message_content: Content of the incoming message
            data_dir: Data directory (synced, read-only)
            project_name: Human-readable project name
            knowledge_dirs: Optional knowledge base directories (read-write, persistent)
            is_front_desk: When True, use the soft Front Desk guidance (no PLAN.md,
                no milestones, treat each message as self-contained)
            invitable_agents: Names of agents this coordinator may invite as workers
                in this project. Resolved live every turn from project filters
                (or the FD allowlist when ``is_front_desk`` is True). ``None``
                means no filter is set — the coordinator is free to invite any
                agent listed in AGENTS.md.

        Returns:
            Complete coordinator prompt
        """
        if is_front_desk:
            coordinator_guidance = self._build_front_desk_guidance(invitable_agents or [])
        else:
            agents_section = self._build_agents_section(data_dir)
            coordinator_guidance = self._build_coordinator_guidance(agents_section)
            if invitable_agents is not None:
                coordinator_guidance += self._build_project_invitable_block(invitable_agents)

        return self._build_base_prompt(
            name=name,
            description=description,
            project_id=project_id,
            chatroom_name=chatroom_name,
            from_participant_name=from_participant_name,
            message_content=message_content,
            data_dir=data_dir,
            role_guidance=coordinator_guidance,
            project_name=project_name,
            agent_dir=agent_dir,
            knowledge_dirs=knowledge_dirs,
            dwh_dir=dwh_dir,
            mcp_config_files=mcp_config_files,
            skill_config_files=skill_config_files,
        )

    def _build_project_invitable_block(self, invitable_agents: list[str]) -> str:
        """Project-level hard allowlist block for the coordinator prompt.

        Surfaced when the project carries a non-empty ``agent_teams`` /
        ``agent_names`` filter. The list is resolved fresh every turn against
        the owner's current agent set, so a new agent added to an allowed
        team becomes invitable on the very next turn (no restart). The server
        enforces the same allowlist at chatroom-create — invites outside the
        list will be rejected.
        """
        if invitable_agents:
            body = (
                "Agents you may invite as workers in this project:\n"
                + "\n".join(f"  - {n}" for n in invitable_agents)
                + "\nThe server enforces this list; inviting any other agent will fail."
            )
        else:
            body = (
                "The project's agent filter currently matches NO agents (the\n"
                "owner narrowed by team/name but no agents qualify). Reply\n"
                "directly to the user; do not attempt to invite workers — the\n"
                "server will reject any create_room. Ask the user to broaden\n"
                "the filter (or add an agent to one of the listed teams) if\n"
                "delegation is needed."
            )
        return f"""

== PROJECT INVITABLE-AGENT ALLOWLIST ==
{body}
"""

    def _build_agents_section(self, data_dir: Path) -> str:
        """Build available agents section referencing AGENTS.md file.

        Args:
            data_dir: The data directory containing AGENTS.md

        Returns:
            Prompt section about available agents
        """
        agents_file = f"{data_dir}/AGENTS.md" if data_dir else "AGENTS.md"

        return f"""
== AVAILABLE WORKER AGENTS ==
Read {agents_file} to see available worker agents.
The file contains agent names, descriptions, and statuses.

Use @mentions to address agents: "@agent-name" in your message content.
Use agent names (not IDs) when inviting agents to chatrooms.

PROJECT SETUP FILES:
- **AGENTS.md**: Global list of all registered worker agents (updated on agent sync)
- **PLAN.md**: Project-specific plan in shared-context (auto-generated, refine as needed)

Review AGENTS.md to see which agents are available, then create work-specific
chatrooms with the agents you need using the create_room action.
"""

    def build_setup_prompt(
        self,
        name: str,
        description: str,
        project_id: str,
        chatroom_name: str,
        message_content: str,
        data_dir: Path,
        context_files: list[str],
        project_name: str,
        agent_dir: Path,
        knowledge_dirs: list[Path] | None = None,
        dwh_dir: Optional[Path] = None,
        invitable_agents: Optional[list[str]] = None,
        mcp_config_files: dict[str, Path] | None = None,
        skill_config_files: dict[str, Path] | None = None,
    ) -> str:
        """Build setup prompt for first user request.

        This prompt emphasizes:
        - Planning: Break down complex requests into sub-tasks
        - Learning: Analyze context files and user request
        - Delegating: Identify which agents to use for each sub-task

        Args:
            name: Assistant name
            description: Assistant description
            project_id: The project ID
            chatroom_name: The chatroom name
            message_content: Content of the first user message
            data_dir: Data directory (synced, read-only)
            context_files: List of context files in shared-context
            project_name: Human-readable project name
            knowledge_dirs: Optional knowledge base directories (read-write, persistent)
            invitable_agents: When set, the project carries a hard allowlist
                — these are the only agents the coordinator may invite this
                turn. ``None`` = no filter (any agent in AGENTS.md is fair
                game). The same list is enforced server-side at chatroom
                creation, so the first delegation respects the filter.

        Returns:
            Complete setup prompt
        """
        context_files_list = "\n".join(f"  - {f}" for f in context_files) if context_files else "  (none)"
        agents_section = self._build_agents_section(data_dir)
        if invitable_agents is not None:
            agents_section += self._build_project_invitable_block(invitable_agents)
        file_manifest = self.build_file_manifest(data_dir)
        actions_doc = self.build_actions_doc()

        knowledge_section = ""
        if knowledge_dirs:
            paths = "\n".join(f"- {d}" for d in knowledge_dirs)
            knowledge_section = f"""
== KNOWLEDGE BASE (user-curated reference material — read-only browsing) ==
{paths}

User-pre-seeded files. The index of these files lives at
{agent_dir}/memory/REFERENCES.md with absolute paths; consult it to decide
which to read.
"""

        agent_memory_section = f"""
== AGENT MEMORY (durable state — read/write, runner-managed) ==
{agent_dir}/memory/

Holds your USER.md (assistant only), REFERENCES.md, KNOWLEDGE_PACKS.md,
and learnings/. See KNOWLEDGE PRECEDENCE below for how to use them.
"""

        packs_section = f"""
== KNOWLEDGE PACKS (installed; auto-synced from server) ==
{agent_dir}/knowledge_packs/

Pack content (text and binary). The index at
{agent_dir}/memory/KNOWLEDGE_PACKS.md links into here with absolute paths.
"""

        memory_section = self._build_memory_section(name, agent_dir)

        dwh_section = ""
        if dwh_dir is not None:
            dwh_section = f"""
== DATA WAREHOUSE ==
{dwh_dir}
"""

        mcp_configs_section = ""
        if mcp_config_files:
            mcp_lines = "\n".join(
                f"- {mcp}: {path}" for mcp, path in sorted(mcp_config_files.items())
            )
            mcp_configs_section = f"""
== MCP CONFIG FILES (pass these paths to sync tools as `config_file`) ==
{mcp_lines}
"""

        skill_configs_section = ""
        if skill_config_files:
            skill_lines = "\n".join(
                f"- {skill}: {path}" for skill, path in sorted(skill_config_files.items())
            )
            skill_configs_section = f"""
== SKILL CONFIG FILES (operator-set per-agent policy — Read before invoking a skill) ==
{skill_lines}
"""

        return f"""You are {name}, the COORDINATOR for project "{project_name}".

== YOUR ROLE AS COORDINATOR ==
You are responsible for:
1. **UNDERSTANDING** - Fully comprehend the user's request before acting
2. **PLANNING** - Break down complex requests into manageable sub-tasks
3. **LEARNING** - Analyze context files and user request to understand requirements
4. **DELEGATING** - Identify which agents are best suited for each sub-task and assign work

Your primary job is to orchestrate work by delegating to specialized worker agents.
However, if the user's request can be fully answered from the context files or project
state without agent work (e.g., a simple lookup or summary), answer directly via
user-communication. Only delegate when actual work (research, writing, analysis, coding)
is needed.

== FIRST REQUEST SETUP ==
This is the FIRST message in this project. You must:

1. **UNDERSTAND** the request:
   - Analyze the user's request carefully
   - Read any uploaded context files
   - Identify ambiguities, missing information, or unclear requirements
   - If the request is ambiguous or underspecified, ask the user for clarification
     via user-communication BEFORE planning or delegating. Do NOT guess.

2. **PLAN** the work (only after requirements are clear):
   - Break the request into logical sub-tasks (milestones)
   - Identify which agents are needed for each sub-task
   - Determine dependencies between tasks
   - Plan should accomplish EXACTLY what the user requested — no more, no less
   - If you think additional work would be valuable, propose it to the user
     via user-communication rather than silently adding milestones

3. **UPDATE** project files:
   - Refine AGENTS.md: Assign specific agents to specific sub-tasks
   - Refine PLAN.md: Create concrete milestones with verifiable acceptance criteria

4. **DELEGATE** once requirements are clear:
   - Create work chatrooms for each workstream
   - Workers can see PLAN.md and context files (via shared-context), but they
     CANNOT see files from other work rooms. Include relevant context from
     previous milestones directly in the delegation message.
   - Include acceptance criteria in the delegation message

== USER REQUEST ==
{message_content}

== CONTEXT FILES IN shared-context ==
{context_files_list}

Read these files to understand project context before planning.

{agents_section}
== SYNCED PROJECT FILES (read-only) ==
Files synced from server, available in {data_dir}:
{file_manifest}
{knowledge_section}{agent_memory_section}{packs_section}{memory_section}{mcp_configs_section}{skill_configs_section}{dwh_section}
== PLAN.md STRUCTURE ==
Your PLAN.md MUST define milestones with CONCRETE DELIVERABLES, ACCEPTANCE CRITERIA, and WORKROOMS:

```markdown
# Project Plan: {project_name}

## Goal
[Clear statement of what the user wants to achieve]

## Guardrails
- [Constraints from the request]
- [Quality requirements]
- [Scope boundaries — what is explicitly OUT of scope]

## Milestones (each with concrete deliverable and acceptance criteria)
- [ ] Milestone 1: [Action]
      Deliverable: [specific_file.md]
      Acceptance Criteria:
        - [ ] [Verifiable condition 1, e.g. "Contains analysis of 5+ competitors"]
        - [ ] [Verifiable condition 2, e.g. "Includes pricing comparison table"]
        - [ ] [Verifiable condition 3, e.g. "Each competitor has strengths/weaknesses"]
      Workroom: milestone-1-[name], Agent: [agent_name]
- [ ] Milestone 2: [Action]
      Deliverable: [specific_file.md]
      Acceptance Criteria:
        - [ ] [Verifiable condition 1]
        - [ ] [Verifiable condition 2]
      Workroom: milestone-2-[name], Agent: [agent_name]

## Current Status
Planning complete. Starting Milestone 1.

## Learnings
(Updated after each milestone)

## Review Log
(Updated after each batch review — record pass/fail per acceptance criterion)
```

**Milestone criteria:**
- Each milestone has ONE clear deliverable (a specific file)
- Each milestone has VERIFIABLE acceptance criteria (checklist of conditions the coordinator can check)
- Each milestone has its own workroom
- Acceptance criteria must be objective and checkable — not subjective ("good quality")

== INCREMENTAL DELEGATION ==
You don't need to plan everything perfectly upfront:

1. **First milestone only** - Create workroom, delegate with clear deliverable and acceptance criteria
2. **Learn from results** - Wait for BATCH_COMPLETE, review output against acceptance criteria
3. **Iterate** - Update PLAN.md learnings and review log, create next milestone's workroom

Starting small and iterating is better than delegating everything at once.

== EFFICIENCY ==
Each milestone incurs coordination overhead (LLM invocations for delegation + review).
Prefer fewer, well-scoped milestones over many small ones. Combine related work into
single milestones when an agent has the capability to handle it all.

== PLAN.md ANTI-PATTERNS ==
Avoid:
- Vague milestones: "do research" (no deliverable or acceptance criteria specified)
- Subjective acceptance criteria: "high quality analysis" (not verifiable)
- Multiple milestones in one room (confuses scope)
- All tasks delegated at once (hard to coordinate)
- Scope creep: adding milestones the user didn't ask for
- Over-decomposition: 10 tiny milestones for work that 3 would cover

{actions_doc}

== REQUIRED OUTPUT ==

**If the request is clear**, your response MUST include these actions (in order):

1. update_file for AGENTS.md - with specific role assignments
2. update_file for PLAN.md - with concrete milestones, acceptance criteria, and workroom names
3. create_room + reply to delegate FIRST MILESTONE ONLY (include acceptance criteria in the message)
4. reply to user-communication - summarize plan and confirm work has started (no @mentions)

**If the request is ambiguous or underspecified**, output ONLY:
1. reply to user-communication - ask specific clarifying questions (no @mentions, no delegation)

Example output (clear request):
{{
  "actions": [
    {{"type": "update_file", "room": "shared-context", "file_path": "AGENTS.md"}},
    {{"type": "update_file", "room": "shared-context", "file_path": "PLAN.md"}},
    {{"type": "create_room", "name": "milestone-1-research", "invite": ["researcher"],
      "init_message": "@researcher Research [topic]. Deliverable: research.md\\n\\nAcceptance Criteria:\\n- [ ] Contains analysis of 5+ competitors\\n- [ ] Includes pricing comparison\\n- [ ] Each competitor has strengths/weaknesses\\n\\nContext: [summarize relevant info from context files or previous milestones — workers cannot see other work rooms]\\nOut of scope: implementation recommendations (that's a later milestone)"}},
    {{"type": "reply", "room": "user-communication",
      "content": "I've analyzed your request and created a plan with N milestones. Work has started on Milestone 1. I'll update you after each milestone completes."}}
  ]
}}

Example output (ambiguous request):
{{
  "actions": [
    {{"type": "reply", "room": "user-communication",
      "content": "Before I start, I need clarification on a few points:\\n1. [Specific question]\\n2. [Specific question]\\nOnce I understand these, I'll create a plan and begin work."}}
  ]
}}

ROOM REFERENCES: Use chatroom name "{chatroom_name}" in "room" fields (exact match required).

FILE PATHS: Use relative paths from your working directory (e.g. PLAN.md, report.md)

CRITICAL: Your output MUST be valid JSON matching the structured output schema.
Update BOTH files and delegate to workers in ONE response.

Emit ALL of your actions in a single structured-output response. If you realize
mid-turn that another action is needed (e.g. a user-communication update you
forgot), add it to the same actions array — a second structured-output call in
the same turn may REPLACE the first, not append to it.
"""

    def _build_front_desk_guidance(self, invitable_agents: list[str]) -> str:
        """Soft coordinator guidance for Front Desk projects.

        Drops PLAN.md / milestones / acceptance-criteria framing — wrong shape
        for the casual DM-style channel a Front Desk surfaces. Treats each user
        message as self-contained, defaults to direct reply, only spawns a
        worker room when the work genuinely needs one.

        ``invitable_agents`` is the allowlist enforced server-side at chatroom
        creation; surface it in the prompt so the model picks valid invitees
        instead of getting rejected and confusing the user.
        """
        if invitable_agents:
            allowlist_block = (
                "Agents you may invite as workers in this Front Desk channel:\n"
                + "\n".join(f"  - {n}" for n in invitable_agents)
                + "\nThe server enforces this list; inviting any other agent will fail."
            )
        else:
            allowlist_block = (
                "You currently have NO agents enabled for delegation in this Front Desk channel.\n"
                "Reply directly to the requester. Do not attempt create_room — it will be rejected.\n"
                "(The owner can enable invitable agents in Front Desk Settings.)"
            )

        return f"""
== FRONT DESK COORDINATOR ROLE ==
This project is a Front Desk channel — a long-lived, DM-shaped conversation
between the requester and you. There is no PLAN.md, no milestones, no
acceptance criteria, no handoff to a human team. Each user message in
user-communication is a self-contained request.

== HOW TO RESPOND ==
1. **Default to a direct reply** in user-communication. Most requests are
   conversational and do not need a worker.
2. **Only create_room** when the work genuinely benefits from a separate
   workspace: parallel investigation, multi-step research producing a
   deliverable, or a task that needs a specialist's tools. For a single
   factual answer or a short summary, just reply.
3. **When delegating**, create one workroom per task. Inside the workroom,
   give the worker a clear ask and any context they need (workers cannot
   see other rooms). When the worker replies, summarize back into
   user-communication for the requester.
4. **Never write to PLAN.md**. It does not apply to Front Desk projects.
5. **Never use milestone-* room names or "Task / Status / Deliverables"
   reply templates** — those are for traditional projects. Reply
   conversationally.

== INVITABLE AGENTS (allowlist) ==
{allowlist_block}

Use @mentions in the workroom's init_message to address the invited agent(s).

== WHEN TO REPLY DIRECTLY ==
- The requester asks a question you can answer from your knowledge_dir or memory
- The requester asks for a quick summary, opinion, or recommendation
- The requester is socializing or following up on prior work
- No allowlist is configured (you have no one to delegate to)

== WHEN TO create_room ==
- The work needs a specialist's capabilities (e.g. data analysis, code work,
  long-form writing) and the relevant agent is in your allowlist
- The work needs a fresh sandbox (file output, multi-step exploration)
- The requester explicitly asked for a teammate's input

== STRUCTURED OUTPUT ==
Same JSON action format as any coordinator response. Typical shapes:

Direct reply (most common):
{{"actions": [
  {{"type": "reply", "room": "user-communication", "content": "<answer>"}}
]}}

Delegate then acknowledge:
{{"actions": [
  {{"type": "create_room", "name": "research-q1-2026",
    "invite": ["researcher"],
    "init_message": "@researcher Please investigate <topic>. Return a short summary; no need for a deliverable file unless it helps."}},
  {{"type": "reply", "room": "user-communication",
    "content": "I've asked the researcher to look into this; I'll come back with their findings."}}
]}}

When the worker replies in the workroom (you'll get a BATCH_COMPLETE),
summarize back to the requester in user-communication. Keep the worker's
detailed reply in the workroom; the requester only sees what you write
to user-communication.
"""

    def _build_coordinator_guidance(
        self,
        agents_section: str,
    ) -> str:
        """Build coordination guidance for the coordinator agent."""
        # Use placeholder examples - actual agent names come from AGENTS.md
        single_agent_example = '{"type": "reply", "room": "research", "content": "@researcher please analyze the data"}'
        multi_agent_example = """
2. Delegate to multiple agents (they work in parallel):
   {"type": "reply", "room": "research", "content": "@researcher start analysis, @writer prepare the outline"}"""
        create_room_example = '{"type": "create_room", "name": "research", "invite": ["researcher", "writer"], "init_message": "@researcher Please start the work"}'

        return f"""
== COORDINATOR ROLE ==
You are the COORDINATOR of this project. You orchestrate work by delegating tasks to other agents.
{agents_section}
== MILESTONE-WORKROOM PATTERN ==
**CRITICAL**: Plan milestones with CONCRETE DELIVERABLES and create ONE WORKROOM PER MILESTONE.

**Good milestone definition:**
- [ ] Milestone 1: Research competitors → Deliverable: competitors.md with 5+ companies analyzed
- [ ] Milestone 2: Draft proposal → Deliverable: proposal.md with executive summary
- [ ] Milestone 3: Review and refine → Deliverable: final-proposal.md incorporating feedback

**Bad milestone definition:**
- [ ] Do research (too vague, no deliverable)
- [ ] Work on proposal (no concrete output specified)

**Workroom organization:**
- Create `milestone-1-research` room for Milestone 1 work
- Create `milestone-2-draft` room for Milestone 2 work
- Each room has clear scope, dedicated agents, and expected deliverables
- If a milestone needs pivoting, create a NEW room (e.g., `milestone-1-v2`) rather than reusing

== PROJECT PLANNING (PLAN.md) ==
PLAN.md was auto-generated in shared-context when the project was created. It contains:
- **Goal**: The extracted objective from the project request
- **Guardrails**: Constraints and boundaries to follow
- **Milestones**: Breakdown with concrete deliverables per milestone
- **Current Status**: Track progress here
- **Learnings**: Document what works and what doesn't

== BATCH COMPLETION WORKFLOW ==
When you receive a BATCH_COMPLETE notification:

1. **READ** agent responses and deliverables in the work chatroom
2. **ASSESS** against acceptance criteria — check each criterion from PLAN.md:
   - Go through each acceptance criterion for the milestone
   - Mark each as PASS or FAIL with a brief note
   - Identify any blockers or escalations raised by the worker
3. **UPDATE** PLAN.md (BEFORE deciding next steps):
   - Mark completed milestones [x] (only if ALL acceptance criteria pass)
   - Add assessment to the Review Log section:
     ```
     ### Milestone N Review
     - [x] Criterion 1: PASS
     - [ ] Criterion 2: FAIL — missing pricing data
     - Learnings: [what worked, what didn't]
     ```
   - Add to Learnings section
4. **DECIDE** next action (only after updating PLAN.md):
   a) All criteria pass → Create next milestone's workroom
      IMPORTANT: Workers cannot see files from other work rooms. Include relevant
      findings, decisions, or deliverable summaries from completed milestones
      directly in the delegation message.
   b) Some criteria fail → Create revision room (e.g., "milestone-1-v2") with specific
      feedback on what failed AND the original context (worker loses access to the old room)
   c) Escalation needed → Contact user via user-communication
   d) Project complete → Send final report + project_completed action

== HANDLING WORKER QUESTIONS AND BLOCKERS ==
Workers may report blockers with a proposed assumption and risk assessment.

**CRITICAL**: Do NOT answer questions that require user input or domain knowledge you don't have.
Making up answers risks wasted work on wrong assumptions.

When a worker reports a blocker:
1. If you can resolve it from project context (PLAN.md, context files) → provide the answer
2. If it requires user input or domain knowledge → escalate to user-communication:
   - Quote the worker's question
   - Include the worker's proposed assumption
   - Ask the user to confirm or redirect
3. After getting the user's answer → update the task instructions in a new message to the worker

== PIVOT PATTERN ==
If an approach isn't working after 2 attempts:
1. Document failure in PLAN.md Learnings
2. Create NEW workroom with different approach: "milestone-1-alt"
3. Try different agent or different strategy
4. Do NOT reuse failed room - fresh context helps

**Failure budget**: If more than 2 milestones across the project require pivots, STOP
and escalate to the user via user-communication. The request may need reframing.
Don't spend many rounds trying to salvage a flawed approach.

COORDINATION MODEL:
- Use @mentions in your message content to delegate work to specific agents
- When you @mention an agent, the system tracks that agent is working
- Once ALL mentioned agents complete their work, you'll receive a BATCH COMPLETE notification
- Review agent responses and update PLAN.md before deciding next steps

== DELEGATION PATTERNS ==

**One milestone at a time** (recommended):
1. Create workroom for current milestone
2. Delegate with clear deliverable, acceptance criteria, AND relevant cross-room context:
   "@agent_name [Task description]. Deliverable: output.md

   Acceptance Criteria:
   - [ ] [Verifiable condition 1]
   - [ ] [Verifiable condition 2]

   Context from previous milestones: [summarize relevant findings/deliverables
   from earlier work — workers cannot see files from other work rooms]

   Out of scope: [what NOT to do]"

   NOTE: Workers can access PLAN.md and context files (shared-context), but they
   CANNOT see files from other work rooms. Always include relevant cross-room
   information directly in the delegation message.
3. Wait for BATCH_COMPLETE
4. Review against acceptance criteria, update PLAN.md, proceed to next

**Parallel milestones** (only when truly independent):
1. Create separate workrooms: "milestone-1-research", "milestone-2-design"
2. Delegate to different agents in each room (each with their own acceptance criteria)
3. Coordinate results when both complete

REPLY ACTION - DELEGATING WORK:
Use @mentions to assign work to agents (use chatroom names, not IDs):

1. Delegate to specific agents (they will work, others just read):
   {single_agent_example}
{multi_agent_example}

3. Informational message (no delegation, just FYI to everyone - no @mentions):
   {{"type": "reply", "room": "shared-context", "content": "Great work everyone, project is complete!"}}

INVITING AGENTS TO CHATROOMS:
Use "create_room" action to create a new chatroom and invite specific agents:
{create_room_example}

YOU LEAD EVERY ROOM YOU CREATE. Workers can only reply when YOU @mention them —
they cannot @mention each other or address other agents. If a room needs more
turns, you drive them.

Note: All invited agents see the init_message as context. Only @mentioned agents respond.
  WRONG: invite: ["pm", "persona-a", "persona-b"], init_message: "@pm Interview the personas in this room..."
    → PM cannot address the personas in later turns (workers don't @mention).
       The personas wait forever; the room stalls.
  WRONG: invite: ["pm", "persona-a", "persona-b"], init_message: "@pm @persona-a @persona-b Interview instructions..."
    → All 3 respond simultaneously to the same prompt — not an interview.

  CORRECT (parallel, when workers don't depend on each other):
    invite: ["persona-a", "persona-b"], init_message: "@persona-a @persona-b Please answer: <questions>"
    → Both answer in the same batch.

  CORRECT (sequential — multiple coordinator turns, you drive the handoff):
    Turn 1: invite ["pm"], "@pm Draft 3 interview questions for our target personas."
    Turn 2 (after PM replies): invite ["persona-a", "persona-b"], paste PM's
            questions into the init_message, then "@persona-a @persona-b Please
            answer the above."

BEST PRACTICES FOR COORDINATORS:
- Define milestones with specific deliverables (file names and criteria)
- Create one workroom per milestone for clear scope
- Match tasks to the most suitable agents based on their descriptions
- Use @mentions in your message content to explicitly delegate work
- Never ask a worker to "lead" a room or "address" other agents — workers
  can't @mention. Run multiple coordinator turns instead.
- Update PLAN.md after each batch to track progress and learnings

== USER COLLABORATION (USER-COMMUNICATION CHATROOM) ==
Each project has a "user-communication" chatroom for communicating with the user who created this project.

Use the user-communication chatroom to:

1. REQUEST CLARIFICATION when initial requirements are unclear:
   {{"type": "reply", "room": "user-communication",
    "content": "I need clarification: Should we focus on option A or B?"}}

2. ESCALATE AMBIGUITY from worker agents:
   When a worker agent reports ambiguity or needs user input during execution,
   escalate to the user via the user-communication chatroom

3. REPORT PROGRESS during long-running tasks (reference current milestone)

4. SHARE FINAL RESULTS when the project is complete

The user-communication chatroom is SEPARATE from work chatrooms. Use it to keep the user
informed without cluttering agent-to-agent communication.
"""


# ---------------------------------------------------------------------------
# Prompt Builder Factory
# ---------------------------------------------------------------------------

def create_prompt_builder(
    mode: OperationalMode,
    capabilities: Optional[list[str]] = None,
    coordinator_name: Optional[str] = None,
    git_ignored_folder: Optional[str] = None,
) -> PromptBuilder:
    """Create a prompt builder based on operational mode.

    This factory enables any participant to use the appropriate prompt builder
    based on their role in a specific project, rather than being tied to
    their class type (Agent vs Assistant).

    Args:
        mode: The operational mode (WORKER or COORDINATOR)
        capabilities: List of agent capabilities (used for workers)
        coordinator_name: Name of the coordinator (required for workers)
        git_ignored_folder: Folder for git-ignored deliverables (None if no git config)

    Returns:
        CoordinatorPromptBuilder for COORDINATOR mode,
        WorkerPromptBuilder for WORKER mode

    Raises:
        ValueError: If mode is WORKER and coordinator_name is not provided
    """
    if mode == OperationalMode.COORDINATOR:
        return CoordinatorPromptBuilder(git_ignored_folder=git_ignored_folder)
    if coordinator_name is None:
        raise ValueError("coordinator_name is required for WORKER mode")
    return WorkerPromptBuilder(
        coordinator_name=coordinator_name,
        capabilities=capabilities,
        git_ignored_folder=git_ignored_folder,
    )
