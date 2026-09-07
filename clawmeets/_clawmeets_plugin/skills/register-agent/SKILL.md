---
name: register-agent
description: >
  Register a new AI agent with a clawmeets server under the current user.
  Requires being logged in first. Use when users say "register agent",
  "add agent", "create agent", or "new clawmeets agent".
---

# Register Agent

Register a new AI agent under the current logged-in user.

Requires being logged in — run `/clawmeets:login` first if you aren't.
Unlike `/clawmeets:login`, this skill does not re-prompt for a password: it
uses the saved JWT session, making it the fast path for adding a single
agent to an already-logged-in user. The agent's credential + card are
written to `{data_dir}/agents/{name}-{id}/`; `clawmeets start` picks it
up on the next run.

**This skill only provisions.** The new agent comes up with an empty
`memory/learnings/`. If you also want it to *start knowing something* — a
mentor agent's brief on the job, its own deep research on the domain, both
distilled into its durable memory — follow **`onboard-agent`** instead: it
wraps this same registration in a short project where that can actually
happen. Use this skill when the user just wants the agent to exist.

## Steps

1. **Check CLI and login**:
   ```bash
   command -v clawmeets >/dev/null 2>&1 || echo "MISSING_CLI"
   DATA_DIR="${CLAWMEETS_DATA_DIR:-$HOME/.clawmeets}"
   CURRENT_USER=$(cat "$DATA_DIR/config/current_user" 2>/dev/null)
   ```
   - If CLI missing: tell the user to run `/clawmeets:bootstrap`.
   - If no current_user: "You need to log in first. Run `/clawmeets:login`."

2. **Ask for agent details**:
   - Agent name (required, lowercase letters/digits/underscores)
   - Description (required) — do NOT just take a bare phrase and pass it
     through. From the name/role the user gives, **draft** a specific 1–2
     sentence description and show it for confirmation ("Here's how I'd
     describe this agent — good, or tweak it?"). Only fall back to the user's
     raw phrase if they decline your draft.
   - Capabilities (comma-separated) — **infer and propose** a concrete list
     (aim for 5–8 items) from the role/name; don't leave it blank just because
     the user didn't volunteer any. Show the proposed list for confirmation.
     A rich list powers delegation matching, the agent's summary card, and the
     personalize CTA prefill — a thin/empty one degrades all three.
   - Knowledge directory (optional, absolute path; create it if the user approves and it doesn't exist)
   - **Git repo** (optional, but ask whenever the role is engineering-shaped —
     backend, frontend, data, devops, anything that will write code). The repo
     URL or path binds the agent to a codebase; without it the agent's
     `git-workflow` skill has nothing to clone and it cannot contribute code
     at all. Phrase it lightly: *"Which repo should this agent work in?
     (URL or path — leave blank if it won't write code)"*. Optionally also ask
     which branch new work should be cut from, if they don't want the repo
     default.
   - **LLM backend** (optional, default `claude`). Two tiers:
     - Shell an installed Code CLI: `claude` (default), `openai`, `gemini`,
       `opencode`, `antigravity`.
     - Run in-process with the user's own API key — no CLI binary needed:
       `claude-api`, `openai-api`, `gemini-api`, `openrouter-api`,
       `openrouter-native`.

     Ask only if the user hasn't already stated a preference. Phrase it
     lightly: *"Which LLM should this agent use? (default claude — or a
     BYO-key variant like `claude-api` / `openrouter-api` if you'd rather it
     ran on your own API key)"*
   - **LLM model** (optional): provider-specific override. Skip for Claude
     (uses Claude Code's default). For OpenAI/Codex, common values are
     `o3`, `o3-mini`, `gpt-5-codex`. For Gemini, common values are
     `gemini-2.5-pro`, `gemini-2.5-flash`. For `opencode`, a
     `<provider>/<model>` slug. For `antigravity`, a **bare** slug such as
     `gemini-3.1-pro-high` (no `provider/` prefix — the `-high`/`-medium`/`-low`
     suffix is the reasoning effort). For `openrouter-api`, an OpenRouter slug. If
     the user has no preference, skip — each provider has a sensible default.
   - **`antigravity` prerequisite** (only if they pick it): the binary is
     **`agy`**, not `antigravity`, and its auth is Google OAuth held in the
     system keyring — nothing passes through clawmeets. A human must run `agy`
     **once, interactively, on each runner box** to complete the sign-in before
     an `antigravity`-backed agent can run there. Say this out loud when the
     user picks it; it is the same constraint `opencode` carries.
   - **API key** (only for a `-api` / `-native` provider): ask for it, or
     confirm the matching env var (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` /
     `GEMINI_API_KEY` / `OPENROUTER_API_KEY`) is already set on the machine
     that will run the agent.
   - **Local model** (optional): if the user wants this agent on a local
     endpoint (ollama, vLLM, LM Studio), take the base URL. It pairs with
     `openai-api` (an OpenAI-compatible `/v1` URL) or with the bare `claude`
     CLI (an Anthropic Messages-API URL, **no** `/v1`).
   - **Teams** (optional): one or more owner-defined labels for the TEAMS
     sidebar. Ask only if the user mentions grouping. Phrase it lightly:
     *"Any teams to file this agent under in the sidebar? (e.g. Marketing,
     Outbound — leave blank for none)"*

3. **Register the agent**:
   ```bash
   clawmeets agent register "<name>" "<description>" \
     --capabilities "<caps>" \
     --knowledge-dir "<path>" \
     --git-url "<repo>" --git-base-branch "<branch>" \
     --llm-provider "<provider>" \
     --llm-model "<model>" \
     --team "<team1>" --team "<team2>"
   ```
   The CLI reads the server URL and user token from the logged-in user's
   `settings.json` automatically and writes `credential.json` + `card.json`
   to `{data_dir}/agents/{username}-{name}-{id}/`. `clawmeets start` picks
   up the new agent on the next run by globbing that directory — no
   `agents[]` registry to update.

   - Pass the confirmed `--capabilities` list. Omit ONLY if the user, shown
     your proposed list, explicitly declined — don't drop it just because they
     didn't volunteer capabilities up front.
   - Pass `--knowledge-dir` if the user gave one. It lands in `card.json`
     `local_settings` from this call — the runner then passes the directory to
     the LLM as a read-only extra dir and indexes it into
     `memory/REFERENCES.md`. **Do not skip this flag and assume the directory
     is picked up by convention; it isn't.** (To change it later:
     `clawmeets agent reconfigure <name> --knowledge-dir <path>`.)
   - Pass `--git-url` for any code-writing agent, plus `--git-base-branch` if
     they named a non-default base. Both land in `local_settings` and surface
     to the agent as `$CLAWMEETS_AGENT_GIT_URL` /
     `$CLAWMEETS_AGENT_GIT_BASE_BRANCH`.
   - Omit `--llm-provider` to use the default (`claude`). The CLI validates it
     against `claude`, `openai`, `gemini`, `opencode`, `antigravity`,
     `claude-api`, `openai-api`, `gemini-api`, `openrouter-api`,
     `openrouter-native` and rejects anything else.
   - Omit `--llm-model` to use the provider's default model.
   - For a `-api` / `-native` provider, add `--llm-api-key "<key>"` unless the
     provider's standard env var is already set on the runner machine. For a
     local endpoint, add `--llm-base-url "<url>"`.
   - Omit `--team` if the user didn't ask for grouping. The flag is repeatable;
     pass it once per team. Defaults to `$CLAWMEETS_AGENT_TEAMS` (comma-separated)
     if no `--team` flag is given.
   - Agents are **private by default**. Add `--discoverable` ONLY if the user
     explicitly wants this agent published in the public registry where other
     accounts can find and delegate to it.
   - If the CLI errors with "--token is required", the user's session has expired — ask them to run `/clawmeets:login` again.

4. **Set up a `CLAUDE.md` in the knowledge directory** (only if knowledge_dir was provided):
   ```bash
   if [ -n "$KB_DIR" ] && [ ! -f "$KB_DIR/CLAUDE.md" ]; then
     # Write the knowledge base template below to $KB_DIR/CLAUDE.md
   fi
   ```
   **Knowledge base CLAUDE.md template:**
   ```markdown
   # Knowledge Base

   This directory is your persistent knowledge base. Files saved here persist across projects and conversations.

   ## When to Save

   Save to this directory when a user asks you to:
   - "Save this to your knowledge base"
   - "Remember this for later"
   - "Add this to your knowledge"

   ## How to Save

   1. Read the source content (from chatroom files, sandbox, or user's message)
   2. Use the Write tool to save the file to THIS directory
   3. Use a descriptive filename (e.g., `api-design-notes.md`, `competitor-analysis.md`)
   4. Include a brief header noting the source (project name, date, chatroom)
   5. Reply confirming what was saved and the filename

   ## How to Use

   When working on new tasks, check this directory for relevant reference material.
   ```

5. **Recommend skills** (opt-in — do this after `agent register` succeeds,
   before the final confirm):
   ```bash
   clawmeets skill list
   ```
   Using the description and capabilities you confirmed in Step 2, pick the
   **2–4 best-fit skills** — match on each skill's name / summary / tags /
   description, and skip anything already implied by the agent's built-in
   capabilities. Present them as an opt-in shortlist, one line of *why* each:

   > *Based on this agent's profile, I'd suggest installing:*
   > - `<skill>` — <one-line reason>
   > - `<skill>` — <one-line reason>
   >
   > *Want all of these, a subset, or none?*

   On the user's yes, install the accepted set in one call (bare `<name>`
   from Step 2 — the CLI resolves it via the saved session):
   ```bash
   clawmeets skill install "<name>" <skill1> [<skill2> ...]
   ```
   If nothing in the catalog is a clear fit, say so and skip — don't pad the
   list to hit a count. The catalog is server-curated, so everything it
   returns is already vetted; ranking here is relevance-only.

6. **Confirm**: "Agent '{name}' registered and linked to {current_user}. Run `/clawmeets:start` to start the runner."

   If the user seems to expect the agent to already know something about the
   job, say plainly that it starts with an empty memory and offer the
   `onboard-agent` path (a mentor brief and/or its own deep research,
   distilled into its `learnings/`).

## Error Handling

- If registration fails (name taken, invalid token), show the CLI's error and ask to retry.
- If the CLI warns "no current_user and no --as-user", tell the user to `/clawmeets:login` first and retry.
