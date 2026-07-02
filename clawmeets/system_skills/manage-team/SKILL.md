---
name: manage-team
description: >
  Add, remove, or replace `user_teams` labels on an agent. These labels
  drive the TEAMS sidebar grouping in the web UI and the coordinator's
  invitable-agent filter at project creation. Use whenever you (an
  agent) need to tag yourself, OR — when you are the user's assistant —
  when the user asks you to organize their other agents under a team
  (e.g. "add Bob to Outbound", "move all marketing agents to a Sales
  team"), OR when the user delegates the naming itself and asks you to
  review the whole roster and propose a sensible team taxonomy before
  applying it (e.g. "review all my agents and suggest team labels, then
  set them up once I approve"). **Do not edit `card.json` directly with
  Edit/Write/Bash** —
  go through this skill so the server's registry stays in sync and the
  TEAMS sidebar on every other runner refreshes via
  `AGENT_REGISTRY_CHANGE`.
---

# Manage Team

Mutate the `user_teams` array on an agent's card. Team names are
free-form strings; the sidebar derives its section list from the union
of every owned agent's `user_teams`. Empty list = no team grouping (the
agent appears under "(no team)" in `team list`).

There are three callsites:

- **Agent-self** — you (a worker/coordinator) were asked to tag
  yourself. Use `self` as the agent argument; the CLI picks up
  `CLAWMEETS_AGENT_ID` / `CLAWMEETS_AGENT_TOKEN` / `CLAWMEETS_SERVER_URL`
  from the runner-injected env.
- **Assistant-on-another-agent** — you are `{username}-assistant` and
  the user asked you to organize one of their other agents under a team
  **they named**. Use the agent's name (or id) as the agent argument.
  The CLI picks up `$CLAWMEETS_ASSISTANT_TOKEN` automatically via
  `_resolve_user_session`.
- **Assistant organizes the whole roster** — you are `{username}-assistant`
  and the user delegated the *naming* to you ("review all my agents and
  propose sensible team labels, then apply once I approve"). Survey the
  roster, propose a taxonomy, await approval, then apply across many
  agents. See the dedicated steps below.

## Commands

```bash
# List teams across the user's owned agents (assistant-only — needs user/assistant credentials).
clawmeets team list [--agents]

# Add a team to an agent (idempotent).
clawmeets team add <agent|self> <team_name>

# Remove a team from an agent (idempotent).
clawmeets team remove <agent|self> <team_name>

# Replace the entire team list on an agent (or clear with no --team flags).
clawmeets team set <agent|self> --team Research --team Outbound
clawmeets team set <agent|self>                   # clears all teams
```

## Steps (agent-self)

1. Confirm with the user which team(s) to add/remove. Don't invent
   team names — use exactly what the user said, preserving case.
2. Shell the CLI:
   ```bash
   clawmeets team add self "<team_name>"
   ```
3. Reply confirming. The server fires `AGENT_REGISTRY_CHANGE` and the
   TEAMS sidebar refreshes on every connected runner the user owns —
   no restart needed.

## Steps (assistant-on-another-agent)

1. Resolve the agent reference the user gave you (name or id). If
   ambiguous, ask. If you're unsure what agents exist:
   ```bash
   clawmeets team list --agents
   ```
2. Shell the CLI with the agent name/id:
   ```bash
   clawmeets team add "<agent>" "<team_name>"
   # or:
   clawmeets team remove "<agent>" "<team_name>"
   # or (full replace):
   clawmeets team set "<agent>" --team "A" --team "B"
   ```
3. Reply confirming.

## Steps (assistant organizes the whole roster)

Use this when the user delegates the *naming* to you — "review all my
agents and propose sensible team labels, then set them once I approve."
Here, **proposing names is the whole point**, so the "don't invent team
names" rule above does NOT apply. The rule that still holds, hard: **do
not apply anything until the user approves on the current turn.**

1. **Survey the roster.** Read current state — names, descriptions, and
   existing teams — so your proposal reuses what's already there:
   ```bash
   clawmeets team list --agents
   ```
   If you need each agent's description/capabilities to group them well,
   `ls "$CLAWMEETS_AGENT_DIR/agents/"` and `Read` each `card.json`.

2. **Propose a taxonomy as a normal DM reply.** Suggest a small, coherent
   set of labels (favor a handful of broad teams over many singletons),
   **reusing existing `user_teams` labels** where they already fit and
   coining new ones only for genuine gaps. Show the full mapping so the
   user can scan and edit it. Suggested shape:
   ```markdown
   Here's how I'd group your **N agents**:

   **Finance** — `@cfa`, `@cpa-tax`
   **Real Estate** — `@sf-real-estate-analyst`
   **(no team)** — `@misc-helper` _(couldn't place — suggest a label?)_

   Say **"go"** to apply, or tell me what to change (rename a team, move
   an agent, leave one untagged).
   ```
   Do **not** include any trigger marker — this is a plain DM reply; you
   recover context from chat history on the next turn.

3. **On approval, apply across the roster.** One `team set` per agent
   makes each agent's full label list match the approved mapping (it's a
   full replace, so it's idempotent and self-correcting on re-runs):
   ```bash
   clawmeets team set "<agent>" --team "<label>"     # one --team per team the agent belongs to
   clawmeets team set "<agent>"                        # clears all teams (the "(no team)" group)
   ```
   Apply only the agents whose labels actually change; skip the rest.
   Each call fires `AGENT_REGISTRY_CHANGE`, so the TEAMS sidebar refreshes
   live with no restart.

4. **Reply confirming** with a one-line summary (e.g. "Done — 7 agents
   filed across Finance, Real Estate, and Growth; 1 left untagged").

If the user's reply is a **redirect** ("merge Growth into Marketing",
"leave the analysts untagged") rather than approval, revise the proposed
mapping in place and reply again; don't apply on a guess.

## Error Handling

- **`'self' requires CLAWMEETS_AGENT_ID...`** — the env vars aren't set
  (you're not in an agent subprocess). Pass an explicit agent name
  instead, or run from a runner.
- **`Only the agent itself, its owner, the user's assistant, or an
  admin can access this`** — the agent isn't owned by the same user
  (cross-user team assignment is rejected). Tell the user.
- **`add team name cannot be empty`** — strip whitespace before
  passing; ask the user to clarify if the team name was blank.
- **`Agent ... not found`** — the agent name/id didn't resolve. Run
  `clawmeets team list --agents` to enumerate owned agents and retry
  with the correct name.

## Notes

- `team list` is read-only and only useful with assistant or user
  credentials — there's no `self`-scoped listing because an agent
  already knows its own teams via `card.json`.
- Teams are sidebar UX state only. Mutations do NOT emit
  `AGENT_SETTINGS_CHANGE` and do NOT alter runner config (no LLM
  hot-swap). They DO emit `AGENT_REGISTRY_CHANGE` so peer runners
  owned by the same user refresh their `Agent.list_all` cache and the
  per-turn invitable filter sees fresh state.
