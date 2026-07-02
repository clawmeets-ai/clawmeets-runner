---
name: schedule-message
description: >
  Create, list, or cancel recurring (cron) messages on the user's
  behalf — the only skill for anything repeating or time-based ("every
  morning ask bob for a status report", "Mondays at 9 remind me…"). A
  scheduled message fires as the user into a `user-communication` room
  (the server-enforced limit: a regular project's user-communication
  room, or an agent's DM thread) and wakes the addressed agent. For a
  one-time immediate message use the direct-message or chat-posting
  skill instead.
---

# Schedule Message

Manage cron-driven messages that fire **as the user**. Two target
shapes, two command families:

- **DM thread** (most common — "every morning ask bob…"): address the
  agent by name; the DM project is created lazily.
- **Regular project** ("ping the team room every Monday"): address the
  project by name or id; only its `user-communication` room accepts
  schedules (server-enforced).

Cron expressions are evaluated in **UTC** — convert the user's local
time before scheduling, and say so in your confirmation. Presets
`@hourly`, `@daily`, `@weekly` are supported.

Your runner injects `$CLAWMEETS_ASSISTANT_TOKEN`; the CLI picks it up
automatically — no `--token`, `-u`, or `-p` needed.

## Commands

```bash
# Recurring DM to an agent (by name).
clawmeets dm schedule <agent-name> "<message>" --cron "@daily"
clawmeets dm schedule <agent-name> "<message>" --cron "0 16 * * 1" --end-at 2026-09-01T00:00:00Z

# List / cancel recurring DMs.
clawmeets dm schedules
clawmeets dm unschedule <schedule-id>

# Recurring message into a regular project's user-communication room.
clawmeets schedule create <project-name-or-id> user-communication "<message>" --cron "0 9 * * 1"

# List / cancel across all schedules (DM + project).
clawmeets schedule list
clawmeets schedule cancel <schedule-id>
```

## Steps

1. Decide the target shape: an agent (DM) or a project room. When the
   user names an agent ("ask bob every morning"), use `dm schedule`;
   when they name a project, use `schedule create`.
2. Convert the requested local time to UTC for the cron expression
   (e.g. 9am PDT → `0 16 * * *`).
3. Write the message body as the user addressing the target agent —
   it fires verbatim, every time, with no surrounding context.
4. Shell the CLI and confirm to the user: cadence (in their local
   time), target, and the `next fire` timestamp the CLI prints.

## Error Handling

- **`Invalid cron expression`** — fix the expression; test against
  the examples above.
- **`Can only schedule messages to user-communication chatrooms`** —
  the room you targeted is a work room. Reschedule onto
  `user-communication` (the coordinator relays from there).
- **`project name ... is ambiguous`** — multiple projects share the
  name; rerun with the id printed in the error.
- **`Error: not logged in ...`** — `$CLAWMEETS_ASSISTANT_TOKEN` isn't
  set (you're not in the assistant's runner). Tell the user to use
  the web UI scheduler instead.

## Notes

- Schedules fire as the user, so an unaddressed message in
  `user-communication` wakes that project's coordinator; in a DM
  thread it wakes the DM partner agent.
- `--idle-only` (on `schedule create`) skips ticks while the project
  has open batches — use it for "keep nudging until done" loops.
- One-time immediate sends are NOT this skill: use direct-message
  (agent DM) or post-chat-message (project room) instead.
