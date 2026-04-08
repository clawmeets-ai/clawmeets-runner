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
        capabilities_line: str = "",
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

        Returns:
            Complete prompt string
        """
        file_manifest = self.build_file_manifest(data_dir)
        actions_doc = self.build_actions_doc()

        cap_section = f"\nCapabilities: {capabilities_line}" if capabilities_line else ""

        return f"""You are {name}, an AI agent.
Description: {description}{cap_section}

Project: {project_name}
Chatroom: {chatroom_name}

== SYNCED PROJECT FILES (read-only) ==
Files synced from server, available in {data_dir}:
{file_manifest}

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

        Returns:
            Complete worker prompt
        """
        worker_guidance = self._build_worker_guidance()
        capabilities_line = ", ".join(self._capabilities) if self._capabilities else "general"

        return self._build_base_prompt(
            name=name,
            description=description,
            project_id=project_id,
            chatroom_name=chatroom_name,
            from_participant_name=from_participant_name,
            message_content=message_content,
            data_dir=data_dir,
            role_guidance=worker_guidance,
            capabilities_line=capabilities_line,
            project_name=project_name,
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

        Returns:
            Complete coordinator prompt
        """
        agents_section = self._build_agents_section(data_dir)
        coordinator_guidance = self._build_coordinator_guidance(agents_section)

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
        )

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

        Returns:
            Complete setup prompt
        """
        context_files_list = "\n".join(f"  - {f}" for f in context_files) if context_files else "  (none)"
        agents_section = self._build_agents_section(data_dir)
        file_manifest = self.build_file_manifest(data_dir)
        actions_doc = self.build_actions_doc()

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

Note: All invited agents see the init_message as context. Only @mentioned agents respond.
  CORRECT: invite: ["pm", "persona-a", "persona-b"], init_message: "@pm Interview the personas in this room..."
    → PM responds; personas read the instructions as context and wait to be addressed by PM
  WRONG:   invite: ["pm", "persona-a", "persona-b"], init_message: "@pm @persona-a @persona-b Interview instructions..."
    → All 3 respond simultaneously instead of PM leading the conversation

BEST PRACTICES FOR COORDINATORS:
- Define milestones with specific deliverables (file names and criteria)
- Create one workroom per milestone for clear scope
- Match tasks to the most suitable agents based on their descriptions
- Use @mentions in your message content to explicitly delegate work
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
