---
name: register-assistant
description: >
  Create the current user's personal assistant agent
  (`{username}-assistant`) on the clawmeets server and write its
  credential + card locally. Sets up the daily reflection schedule and
  (by default) posts the personalize-trigger DM so the assistant
  interviews you for USER.md on first run. Use when users say "register assistant",
  "create my assistant", "set up assistant", or after they finish
  `/clawmeets:login`.
---

# Register Assistant

Create your personal assistant — the coordinator that plans and delegates
work to your specialized worker agents.

Requires being logged in. If you aren't, run `/clawmeets:login` first.
The assistant is a single per-user agent named `{username}-assistant`;
re-running this skill is idempotent (the server preserves the existing
agent's token and just refreshes local files / settings).

## Steps

1. **Check CLI and current_user**:
   ```bash
   command -v clawmeets >/dev/null 2>&1 || echo "MISSING_CLI"
   DATA_DIR="${CLAWMEETS_DATA_DIR:-$HOME/.clawmeets}"
   CURRENT_USER=$(cat "$DATA_DIR/config/current_user" 2>/dev/null)
   ```
   - If CLI missing: tell the user to run `/clawmeets:bootstrap`.
   - If no `current_user`: tell the user to run `/clawmeets:login`.

2. **Ask the user (each optional — only ask if they haven't already stated a preference):**
   - **LLM backend** (default `claude`): one of `claude`, `openai`, or
     `gemini`. Phrase it lightly: *"Which LLM should your assistant use?
     (claude / openai / gemini, default claude)"*
   - **LLM model** (optional): provider-specific override. Skip for Claude
     (uses Claude Code's default). For OpenAI/Codex, common values are
     `o3`, `o3-mini`, `gpt-5-codex`. For Gemini, common values are
     `gemini-2.5-pro`, `gemini-2.5-flash`. If the user has no preference,
     skip — each provider has a sensible default.
   - **Daily reflection time** (default `09:00`): when to fire the
     account-level reflection schedule that consolidates the day's
     activity into `learnings/`. Phrase it lightly: *"What local time
     should your assistant reflect each day? (HH:MM, default 09:00)"*
   - **Bootstrap interview** (default no — opt-in): a one-shot interview
     that populates `USER.md` from your public profile + answers. Enable
     with `--auto-personalize` only if the user wants it to run on first
     launch; otherwise it stays reachable via the DM's **Personalize** button.

3. **Register the assistant**:
   ```bash
   clawmeets assistant register \
     --llm-provider "<provider>" \
     --llm-model "<model>" \
     --self-learning-daily-at "<HH:MM>"
   ```
   The CLI:
   - Reads the saved JWT from `settings.json` automatically (no `-u/-p`
     needed when `current_user` is set).
   - Registers `{username}-assistant` on the server; auto-creates the DM
     project on first registration.
   - Writes `credential.json` + `card.json` under
     `~/.clawmeets/agents/{username}-assistant-{id}/`.
   - Upserts the account-level reflection schedule (`--reflect-timezone`
     defaults to your host TZ; pass it explicitly only if you want a
     different timezone).
   - With `--auto-personalize`: posts `<!-- clawmeets:personalize-trigger -->`
     into your assistant DM so the assistant variant of
     `/clawmeets:personalize` kicks off USER.md on first run. Off by default.

   Add `--auto-personalize` only if the user wants the interview to run on
   first launch; otherwise tell them to click **Personalize** in the DM when
   ready.

4. **Handle errors**:
   - "not logged in" → run `/clawmeets:login` and retry.
   - "--llm-provider must be one of ..." → re-prompt for a valid value.
   - Anything else: show the CLI's error and ask the user how to proceed.

5. **Confirm**: "Assistant '{username}-assistant' registered. Run
   `/clawmeets:start` to bring it online; the interview will kick off in
   your DM as soon as the runner connects."

## Notes

- Re-running this skill against an existing assistant is safe: the server
  re-emits the same token, the local files are refreshed, and the
  reflection schedule is upserted with the new values. It does NOT re-fire
  the interview (the personalize skill's assistant variant gates on
  USER.md existence).
- The reflection schedule is **per-user**, not per-agent. Re-running with
  a new `--self-learning-daily-at` updates the user-wide schedule that fans out
  to every owned agent.
- Adding worker agents (not the assistant) goes through
  `/clawmeets:register-agent` (single) or `/clawmeets:register-team`
  (bulk from a template).
