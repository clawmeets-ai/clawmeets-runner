---
name: manage-project-roster
description: >
  Add or remove agents (or whole teams) on an EXISTING project's invitable
  allowlist — the set of specialists you, the coordinator, are permitted to
  pull into the project via create_room. Use when a project you coordinate
  needs an agent the owner did NOT include at creation: most often when a
  `create_room` is rejected with "not in this project's invitable allowlist"
  (the server's 403) or with "is a real agent but is NOT invitable in this
  project" (the runner's pre-flight validator, which is what you will usually
  see), or when you're planning work that clearly needs a specialist outside
  the current roster. The allowlist is a guardrail the owner set — NEVER widen it
  silently: surface the gap, propose the exact addition, and only run the CLI
  after the user approves. Assistant-only. (To set the roster AT creation
  time, that's the propose-project flow's `--agent`/`--team` flags, not this.)
---

# Manage project roster

A project's invitable allowlist (`agent_names` + `agent_teams` on its
`meta.json`) is set at creation and enforced in **two** places, which is why
the rejection you see may be worded two different ways:

- **Your own runner, before the action is ever sent** — the pre-flight action
  validator rejects the `create_room` with *"`<name>` is a real agent but is
  **NOT invitable** in this project (outside its invitable allowlist)"*. This
  fires first, so in practice it is the message you actually get.
- **The server** — `POST /chatrooms` returns **403 — "not in this project's
  invitable allowlist"** for anything that reaches it.

Same rule, same fix. This skill is the *only* way to change the allowlist
after the project exists. It shells one CLI command, which emits a
`PROJECT_ALLOWLIST_UPDATED` changelog entry that replays into the project
`meta.json` on the server and on your own runner. No restart — and you do not
have to wait for your next turn either: the validator re-reads the invitable
set (union only, never shrinking) between retry attempts, so widening the
roster and re-emitting the **same** `create_room` **within the same turn**
works.

## When this fires

- You tried to delegate to a specialist and `create_room` was rejected —
  either **"is a real agent but is NOT invitable in this project"** (your
  runner's pre-flight validator, the common case) or **403 … not in this
  project's invitable allowlist** (the server). Both mean the same thing.
- You're scoping work in a project you already coordinate and it plainly
  needs an agent the owner didn't list at creation.

If you're instead deciding whether to spin up a **new** project and who staffs
it, that's `propose-project` (its `--agent`/`--team` flags set the initial
allowlist). This skill is strictly for editing a project that already exists.

## The guardrail: ask, then add

The allowlist is a deliberate scope the **owner** chose. Do **not** widen it on
your own — and never claim you added an agent that you haven't.

1. **Surface the gap** in `user-communication`: name the specific agent (or
   team) that's missing, the project it's for, and *why* the work needs it.
2. **Propose the exact change** — e.g. "Add `<agent>` to this project's
   invitable roster so I can hand it the ETL step?"
3. **Wait for the user's approval.** Only then run the CLI.
4. **Confirm** once it lands, and continue the work (the `create_room` will
   now go through).

If the user has **already** asked for that agent on this project, step 3 is
satisfied — do not stall a turn re-asking. Run the CLI and re-emit the same
`create_room` in the same turn.

## Command

Auth is automatic in your runtime — you are the project's coordinator, so the
runner-injected `CLAWMEETS_AGENT_TOKEN` authorizes the edit. No `--token`
needed.

```bash
# Add an agent (repeatable). Accepts id, full registry name, or owner-relative
# short name — same forms propose-project uses for --agent.
clawmeets project allowlist <project_id> --agent <agent-name>

# Add a whole team (every agent carrying that user_team becomes invitable).
clawmeets project allowlist <project_id> --team <team-name>

# Remove an agent / team.
clawmeets project allowlist <project_id> --remove-agent <agent-name>
clawmeets project allowlist <project_id> --remove-team <team-name>

# Replace the entire allowlist with exactly these (instead of merging).
clawmeets project allowlist <project_id> --agent A --agent B --replace
```

Default is **MERGE**: the `--agent`/`--team` you pass are unioned onto the
current allowlist, then any `--remove-*` are dropped. `--replace` sets it to
exactly what you pass.

## Finding the project_id

- Reacting **inside** the project (either rejection above): use *this*
  project's id — it's the project you're currently coordinating.
- Otherwise list them: `clawmeets project list` (the id is the trailing UUID).

## Effect

The command prints the updated lists, e.g.:

```
Allowlist for <project-name> updated — agent_names=[...] agent_teams=[...]
```

Behind it: a `PROJECT_ALLOWLIST_UPDATED` entry replays into the project's
`meta.json` everywhere, so the new agent resolves as invitable — on your next
`create_room` attempt in this turn, and on every later turn.

**One thing not to do:** if a `create_room` is rejected for a name you know is
spelled right, do NOT retry it with spelling variants and do NOT widen the
roster again under a second spelling. Both rejections are the allowlist, not
the spelling. Widen once, re-emit the same name.

## Error handling

- **403 (Only the coordinator agent or project owner can edit the allowlist)** —
  you're not the coordinator of that project (and have no owner JWT). You can't
  edit someone else's project's roster; tell the user to do it from their side.
- **404 (Project … not found)** — wrong `project_id`. Re-resolve with
  `clawmeets project list`.

## Notes

- This only changes *who may be invited*. It does not add anyone to the project
  yet — you still `create_room` / @-mention to actually bring the agent in.
- Mirrors how `manage-team` edits an agent's `user_teams`: go through the CLI,
  never hand-edit `meta.json` — the changelog fan-out is what keeps the server
  and every runner in sync.
- Do not include any trigger marker in your reply.
