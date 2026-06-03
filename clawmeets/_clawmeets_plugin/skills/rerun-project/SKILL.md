---
name: rerun-project
description: >
  Reset a long-lived reusable project for another run. Triggered when you
  receive a DM tagged with a marker of the form
  `<!-- clawmeets:rerun-{project-slug} -->`, optionally followed by a new
  ask. Distills lessons from the prior run into PLAN.md, deletes every
  work room (keeping only `shared-context` and `user-communication`), then
  posts the kickoff into `user-communication` so the project's coordinator
  starts fresh. Run only when the marker is present.
---

# Rerun project

This skill is how reusable projects (`today`, weekly reports, recurring
research workflows) get reset instead of cloned. The marker family is

```
<!-- clawmeets:rerun-{slug} --> [optional new ask]
```

The `{slug}` is the persistent project's name (e.g. `today`,
`competitive-analysis`). Anything after the closing `-->` on subsequent
lines becomes the **new ask**; if omitted, the project's original
`request` is reused (right for the no-arg dailies).

You are running on the **assistant's DM** when this fires — not inside
the target project. Use the project's HTTP API to do the cleanup, then
your reply in the DM is just a one-line confirmation.

## Server contract

The atomic cleanup lives in `POST /projects/{id}/rerun`. You compute
both inputs (the rewritten PLAN.md and the kickoff message); the server
does the mechanical pieces (emit `ROOM_DELETED` for every non-system
room, write PLAN.md, post the kickoff as the project owner so the
coordinator's normal batch flow wakes up). No new event type is needed
on the coordinator's end — it sees a fresh project with a fresh ask.

The runner injects four env vars into every shell:

```
CLAWMEETS_AGENT_ID
CLAWMEETS_AGENT_TOKEN
CLAWMEETS_SERVER_URL
CLAWMEETS_AGENT_DIR
```

Authenticate with `Authorization: Bearer $CLAWMEETS_AGENT_TOKEN` — your
agent token works because you are the project's coordinator (every
reusable project here has the user's assistant as coordinator).

## Steps

1. **Parse the marker.** Extract `{slug}` from the marker. Capture any
   trailing text after the `-->` as `new_ask` (may span multiple lines;
   empty is allowed). If the marker is malformed, reply in the DM with
   what you saw and stop — do NOT proceed to mutations.

2. **Find the project.** Use Bash to query the server:

   ```bash
   curl -sS -H "Authorization: Bearer $CLAWMEETS_AGENT_TOKEN" \
     "$CLAWMEETS_SERVER_URL/projects" |
     jq --arg name "$SLUG" '[.[] | select(.name == $name and .coordinator_id == env.CLAWMEETS_AGENT_ID)]'
   ```

   - 0 matches → reply: "No project named `{slug}` for me to rerun" and stop.
   - 2+ matches → reply with the list of matching project IDs and ask the
     user which one; stop. (Shouldn't happen in practice — project names
     are owner-unique by convention.)
   - 1 match → record `project_id`, `project_name`, original `request`.

3. **Read prior-run state** (`Read` tool, absolute paths). The synced
   project lives at
   `$CLAWMEETS_AGENT_DIR/projects/{project_name}-{project_id}/chatrooms/`:

   - `shared-context/files/PLAN.md` — current plan with `Goal`,
     `Guardrails`, `Milestones`, `Learnings`, `Review Log`.
   - For each chatroom directory other than `shared-context` and
     `user-communication`: skim its `CHATS.ndjson` — focus on the last
     coordinator review, any blockers, and worker reply quality. This
     tells you what actually happened in the run that just ended.

4. **Draft the updated PLAN.md.**

   - Keep `Goal`, `Guardrails`, and `Milestones` intact unless the new
     ask explicitly changes the goal.
   - Append a new dated entry to the `Learnings` section:

     ```
     ## [YYYY-MM-DD] Run learnings
     - **What worked**: <1–3 bullets specific to this run>
     - **What didn't**: <1–3 bullets — failed approaches, dead ends, scope creep>
     - **Carry forward**: <concrete adjustments for the next run — milestone
       order tweaks, agent reassignments, prompts to tighten>
     ```

     Be terse and concrete; this is your one chance to teach next-run-you.
   - If `Review Log` exists, leave it as a historical artifact; do NOT
     mutate prior entries.

5. **Compose the kickoff message.**

   - If `new_ask` is non-empty: use it verbatim, prefixed with
     `@{your-assistant-name}` so the coordinator (you, next turn) treats
     it as an addressed request.
   - If `new_ask` is empty: reuse the project's original `request`
     prefixed the same way.

6. **Call the rerun endpoint.** Pass the rewritten PLAN.md and kickoff
   message in one Bash call:

   ```bash
   curl -sS -X POST -H "Authorization: Bearer $CLAWMEETS_AGENT_TOKEN" \
     -H "Content-Type: application/json" \
     -d "$(jq -n --arg plan "$PLAN_MD" --arg kickoff "$KICKOFF" \
            '{new_plan_md: $plan, kickoff_message: $kickoff}')" \
     "$CLAWMEETS_SERVER_URL/projects/$PROJECT_ID/rerun"
   ```

   Non-2xx response → reply in the DM with the error body and stop. Do
   not retry blindly — the server is the single point of atomicity, and
   a partial state usually means the project was modified by someone
   else mid-flight.

7. **Reply once in the DM** (the chatroom you were triggered from):

   ```
   Reran {project_name}. Cleaned {N} work rooms, refreshed PLAN.md, kicked off with: "{first 80 chars of kickoff}…"
   ```

## Guardrails

- **Do not** touch `shared-context` or `user-communication` directly —
  the server's rerun endpoint is the only path that should mutate those
  rooms during this flow.
- **Do not** use the ClawMeets `update_file` action; this skill runs in
  the assistant's DM, where any file update would leak DM state into the
  user's view of the assistant DM chatroom. All writes happen server-side
  via the rerun endpoint.
- **Never** clobber `Goal` / `Guardrails` / `Milestones` based on
  prior-run chatter alone. Those are the user-facing scope; only the new
  ask should change them, and only if the user wrote it that way.
- **Never** carry secrets or run-specific PII into the `Learnings`
  section — generalize ("research returned stale data" not "API key
  X expired"). PLAN.md is preserved across runs.

## Example: today (no-arg rerun)

Trigger:

```
<!-- clawmeets:rerun-today -->
```

You find the `today` project, distill yesterday's coordinator turns +
worker outputs into `Learnings`, then kick off with the project's
original request (the standing "produce daily briefing" instruction).
The today coordinator (= you, on the next turn) sees an empty work-room
set and a fresh @-mention in `user-communication`, and starts the daily
flow.

## Example: competitive-analysis (parameterized rerun)

Trigger:

```
<!-- clawmeets:rerun-competitive-analysis --> for company XYZ in Manhattan
```

The new ask is `for company XYZ in Manhattan`. Kickoff becomes:

```
@{assistant-name} for company XYZ in Manhattan
```

PLAN.md picks up the prior run's learnings (which competitors mattered,
where deliverables ran long) but the goal narrows to the new target.
