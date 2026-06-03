---
name: register-agent
description: >
  Register a new AI agent with a clawmeets server under the current user.
  Requires being logged in first. Use when users say "register agent",
  "add agent", "create agent", or "new clawmeets agent".
---

# Register Agent

Register a new AI agent under the current logged-in user.

Requires being logged in — run `/clawmeets:init` first if you aren't.
Unlike `/clawmeets:init`, this skill does not re-prompt for a password: it
uses the saved JWT session, making it the fast path for adding a single
agent to an already-logged-in user. The agent is appended to the current
user's agent list in `settings.json`.

## Steps

1. **Check CLI and login**:
   ```bash
   command -v clawmeets >/dev/null 2>&1 || echo "MISSING_CLI"
   DATA_DIR="${CLAWMEETS_DATA_DIR:-$HOME/.clawmeets}"
   CURRENT_USER=$(cat "$DATA_DIR/config/current_user" 2>/dev/null)
   ```
   - If CLI missing: tell the user to run `/clawmeets:bootstrap`.
   - If no current_user: "You need to log in first. Run `/clawmeets:init`."

2. **Ask for agent details**:
   - Agent name (required, lowercase letters/digits/underscores)
   - Description (required)
   - Capabilities (optional, comma-separated)
   - Knowledge directory (optional, absolute path; create it if the user approves and it doesn't exist)
   - **LLM backend** (optional, default `claude`): one of `claude`, `codex`,
     or `gemini`. Ask only if the user hasn't already stated a preference.
     Phrase it lightly: *"Which LLM should this agent use? (claude / codex /
     gemini, default claude)"*
   - **LLM model** (optional): provider-specific override. Skip for Claude
     (uses Claude Code's default). For Codex, common values are `o3`,
     `o3-mini`, `gpt-5-codex`. For Gemini, common values are
     `gemini-2.5-pro`, `gemini-2.5-flash`. If the user has no preference,
     skip — each provider has a sensible default.
   - **Tags** (optional): one or more owner-defined labels for the TAGS
     sidebar. Ask only if the user mentions grouping. Phrase it lightly:
     *"Any tags to file this agent under in the sidebar? (e.g. Marketing,
     Outbound — leave blank for none)"*

3. **Register and link to settings.json in one command**:
   ```bash
   clawmeets agent register "<name>" "<description>" \
     --capabilities "<caps>" \
     --save-to-settings \
     --knowledge-dir "<kb_dir>" \
     --llm-provider "<provider>" \
     --llm-model "<model>" \
     --team "<team1>" --team "<team2>"
   ```
   The CLI reads the server URL and user token from the logged-in user's
   `settings.json` automatically, and `--save-to-settings` appends the agent
   to `agents[]`. No manual JSON parsing needed.

   - Omit `--capabilities` if the user didn't provide any.
   - Omit `--knowledge-dir` if the user didn't provide one.
   - Omit `--llm-provider` to use the default (`claude`). The CLI validates
     the value and rejects anything outside `claude|codex|gemini`.
   - Omit `--llm-model` to use the provider's default model.
   - Omit `--team` if the user didn't ask for grouping. The flag is repeatable;
     pass it once per team. Defaults to `$CLAWMEETS_AGENT_TEAMS` (comma-separated)
     if no `--team` flag is given.
   - If the CLI errors with "--token is required", the user's session has expired — ask them to run `/clawmeets:init` again.

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

5. **Confirm**: "Agent '{name}' registered and linked to {current_user}. Run `/clawmeets:start` to start the runner."

## Error Handling

- If registration fails (name taken, invalid token), show the CLI's error and ask to retry.
- If the CLI warns "no current_user and no --as-user", tell the user to `/clawmeets:init` first and retry.
