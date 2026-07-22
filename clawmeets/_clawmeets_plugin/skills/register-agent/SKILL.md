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
   - **LLM backend** (optional, default `claude`): one of `claude`, `openai`,
     or `gemini`. Ask only if the user hasn't already stated a preference.
     Phrase it lightly: *"Which LLM should this agent use? (claude / openai /
     gemini, default claude)"*
   - **LLM model** (optional): provider-specific override. Skip for Claude
     (uses Claude Code's default). For OpenAI/Codex, common values are
     `o3`, `o3-mini`, `gpt-5-codex`. For Gemini, common values are
     `gemini-2.5-pro`, `gemini-2.5-flash`. If the user has no preference,
     skip — each provider has a sensible default.
   - **Tags** (optional): one or more owner-defined labels for the TAGS
     sidebar. Ask only if the user mentions grouping. Phrase it lightly:
     *"Any tags to file this agent under in the sidebar? (e.g. Marketing,
     Outbound — leave blank for none)"*

3. **Register the agent**:
   ```bash
   clawmeets agent register "<name>" "<description>" \
     --capabilities "<caps>" \
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
   - Omit `--llm-provider` to use the default (`claude`). The CLI validates
     the value and rejects anything outside `claude|openai|gemini`.
   - Omit `--llm-model` to use the provider's default model.
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

4.5 **Recommend skills** (opt-in — do this after `agent register` succeeds,
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

5. **Confirm**: "Agent '{name}' registered and linked to {current_user}. Run `/clawmeets:start` to start the runner."

## Error Handling

- If registration fails (name taken, invalid token), show the CLI's error and ask to retry.
- If the CLI warns "no current_user and no --as-user", tell the user to `/clawmeets:login` first and retry.
