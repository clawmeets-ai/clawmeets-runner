---
name: manage-project-roster
description: >
  Add or remove agents (or whole teams) on an EXISTING project's invitable
  allowlist — the set of specialists you, the coordinator, are permitted to
  pull into the project via create_room. Use when a project you coordinate
  needs an agent the owner did NOT include at creation: most often when a
  `create_room` is rejected with "not in this project's invitable allowlist",
  or when you're planning work that clearly needs a specialist outside the
  current roster. The allowlist is a guardrail the owner set — NEVER widen it
  silently: surface the gap, propose the exact addition, and only run the CLI
  after the user approves. Assistant-only. (To set the roster AT creation
  time, that's the propose-project flow's `--agent`/`--team` flags, not this.)
---

# Manage project roster

A project's invitable allowlist (`agent_names` + `agent_teams` on its
`meta.json`) is fixed at creation and hard-enforced server-side: any
`create_room` that invites an agent outside it is rejected with **403 — "not
in this project's invitable allowlist"**. This skill is the *only* way to
change that allowlist after the project exists. It shells one CLI command,
which emits a `PROJECT_ALLOWLIST_UPDATED` changelog entry that replays into
the project `meta.json` on the server and on your own runner — so your **next
turn** in the project sees the widened roster and the `create_room` succeeds.
No restart.

## When this fires

- You tried to delegate to a specialist and `create_room` came back **403 …
  not in this project's invitable allowlist**.
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

- Reacting **inside** the project (the 403 case): use *this* project's id — it's
  the project you're currently coordinating.
- Otherwise list them: `clawmeets project list` (the id is the trailing UUID).

## Effect

The command prints the updated lists, e.g.:

```
Allowlist for <project-name> updated — agent_names=[...] agent_teams=[...]
```

Behind it: a `PROJECT_ALLOWLIST_UPDATED` entry replays into the project's
`meta.json` everywhere, so your next coordinator turn resolves the new agent as
invitable and `create_room` succeeds.

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
