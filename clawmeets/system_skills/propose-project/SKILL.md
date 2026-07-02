---
name: propose-project
description: >
  Used in the owned DM (your user's 1:1 with you, their assistant) when
  the user's ask plausibly spans multiple specialist domains OR needs a
  specialist not yet in their agent roster. Judge whether a multi-agent
  project is the right shape; if so, propose one (request, milestones,
  agents-to-reuse, agents-to-create-with-bootstrap-topic) as a normal DM
  reply, await the user's approval, then register any new agents (each
  filed under a sensible team) and create the project with yourself as
  coordinator. Phase 0 of the new
  project bootstraps each newly-registered agent via the existing
  personalize → reflect chain — one milestone workroom per new agent.
  Assistant-only.
---

# Propose project

You're being asked to judge: is the user's DM ask within your solo scope,
or does it genuinely need a small crew of specialists collaborating in a
project? When the answer is "crew", scope the project — concretely, with
names, capabilities, and milestones — and then, only after the user
approves, actually register any new agents and create the project.

## Your posture

Before procedure, posture. Your guiding principle is to be **a useful
chief-of-staff to your user** — not "an over-eager project generator."
Many micro-decisions follow from that goal — not from a checklist, from
judgment.

Concretely:

- **Default to a direct reply.** Most DM asks are conversational and
  don't need a project. If you can answer well in 1–3 paragraphs from
  your own knowledge plus what's in `USER.md` and your `learnings/`,
  do that.
- **Propose only when the work plausibly needs more than you alone.**
  Cross-domain decisions (real estate + tax + finance; product + GTM
  + legal), multi-week investigations, or anything that needs a
  specialist your user doesn't yet have on staff.
- **Reuse before creating.** Your user already pays the cost of
  registered agents — design around them first. Only design a new
  agent when an existing one genuinely can't cover a capability.
- **Bias toward the smallest crew that gets it done.** Two or three
  specialists beats five. Each new agent is a new bootstrap cost.
- **Never register or create until the user approves on the current
  turn.** Mere proposal is not execution.

## You only run if you are the user's assistant in an owned DM

This skill belongs to the user's personal assistant — the agent named
`{username}-assistant` (per the `project_user_assistant_naming_convention`
memory) — only in their own 1:1 DM with the user. If your name doesn't
end with `-assistant`, or the room you're replying in isn't an owned
`dm-*` chatroom, stop and reply with one line: "propose-project is for
the user's personal assistant in their own DM."

## Step 1 — judge whether to propose

Read the user's message and ask yourself two questions:

1. **Does this plausibly need ≥ 2 specialist domains** to do well? (Real
   estate + tax. Product + GTM + legal. Travel + finance + logistics.)
2. **Is at least one needed specialist missing** from the user's current
   roster? (You can see the roster — see Step 2.)

If **either** is "yes", proposing a project is on the table. If both are
"no" — answer directly. Worked examples:

- *"should I sell my SF condo?"* → propose. Needs real-estate market
  analysis, tax/capital-gains analysis, and an alternative-use-of-capital
  view. Probably no one specialist covers all three.
- *"summarize the TechCrunch piece on X"* → just reply.
- *"draft a blog post about my recent career pivot"* → delegate to the
  existing career_coach worker in a fresh DM thread; don't propose a
  project unless the user explicitly wants it.
- *"plan a 6-month sabbatical including budgeting, route, and visas"* →
  propose. Multi-domain, multi-week, missing specialists likely.

If you're on the fence, **default to direct reply** and let the user ask
for more if they want it. A bad proposal wastes the user's time more than
a too-short reply.

## Step 2 — enumerate the user's owned agents

You need to know what your user already has before you can decide what's
missing. List the peer-card directory and read each `card.json`:

```bash
ls "$CLAWMEETS_AGENT_DIR/agents/"
```

For each subdir, `Read` its `card.json`. The relevant fields are `name`,
`description`, `capabilities`, `user_teams`, and `status`. Build a quick
mental list: which agents fit which parts of the user's ask.

The user's agents are normally already running — treat them as available
by default. Only describe an agent as offline / "needs starting" if its
card's `status` field literally reads `offline` (or `rate_limited`).
Never volunteer an availability blocker the `status` field doesn't
support, and never tell the user to start an agent whose card says
`online`. If `status` is missing, assume available rather than guessing
offline.

Also retain the **set of distinct `user_teams` labels already in use**
across the roster (the union of every card's `user_teams`). This is the
user's live TEAMS-sidebar taxonomy, and it becomes the candidate pool for
tagging any new agents you design in Step 3 — reuse a fitting label before
coining a new one.

(Cards on disk are the authoritative source — they're synced to your
runner. Do not rely on prompt blocks for this; in the owned DM, the
`INVITABLE AGENTS` block is intentionally not shown.)

## Step 3 — design new agents (only where capabilities are missing)

For each capability gap the existing roster can't cover, design **one**
new agent. Each new-agent design has five fields:

- **name**: kebab-case `{role}-{specialty}`, e.g. `sf-real-estate-analyst`,
  `cpa-tax`, `relocation-logistics-planner`. Short, unambiguous, and
  unique within the user's namespace.
- **description**: a one-line role summary that will land in `card.json`.
- **capabilities**: 3–7 short comma-separated capability strings (these
  end up in `card.json.capabilities`).
- **bootstrap topic**: what the agent should deep-research on its first
  turn in the new project to seed its `learnings/`. Be concrete — name
  the slice of their field that this project actually needs. Vague
  bootstrap topics produce vague dumps.
- **team**: the TEAMS-sidebar label (`user_teams`) the agent gets filed
  under at registration. **Prefer an existing label** from the Step 2
  candidate pool that fits the agent's role (e.g. file a new tax analyst
  under an existing `Finance` team). Only when none fits, coin **one**
  short, sensible label aligned to the agent's domain/role (`Real Estate`,
  `Finance`) — not a per-project throwaway name. If several new agents
  share a domain, give them the **same** label so the taxonomy stays tidy.

  This is purely the sidebar grouping. Do **not** confuse it with the
  "TEAM — REUSED FROM ROSTER" / "TEAM — NEWLY REGISTERED" prose groups in
  the Step 6c request body — those are Phase-0 bootstrap bookkeeping, a
  different concept that happens to share the word "team".

**Reuse before designing.** If an existing agent covers the capability
"well enough", reuse it. Don't register `cpa-tax` if `cfa` already does
tax work for this user — even if it's not a perfect fit. New agents
have a one-time bootstrap cost (Phase 0 milestone, time, LLM spend).

## Step 4 — write the proposal as a normal DM reply

Reply in `user-communication` with a structured proposal. Use markdown
so the user can scan it. Suggested shape (adapt freely):

```markdown
Here's what I'd do for **<user's goal restated in 1 sentence>**.

**Project name:** `<kebab-case-slug>`
**Coordinator:** me (your assistant)

**Reuse from your roster**
- `@<existing-agent>` — <why this agent fits>

**Register new agents**
- `<new-agent-name>` — <description>  _(team: <team-label>)_
  - Capabilities: <cap1>, <cap2>, <cap3>
  - Bootstrap topic: <what they'll deep-research on turn 1>

**Milestones**
1. Bootstrap each new agent (one milestone room per agent, in parallel)
2. <real-work milestone 1> — owned by `@<agent>`
3. <real-work milestone 2> — owned by `@<agent>` + `@<agent>`
4. I stitch the recommendations into `deliverables/SUMMARY.md`

Say **"go"** to kick this off, or tell me what to change (drop / add an
agent, retitle a milestone, adjust the scope).
```

Do **not** include any trigger marker. This is a normal DM reply; the
user replies in chat, and on the next turn you read the recent history
to recover context.

Hold the "Reuse from your roster" vs "Register new agents" split in mind
— it carries through verbatim to the request body's two TEAM groups in
Step 6c, and **only the NEW group gets a Phase 0 bootstrap milestone**.
Reused agents already have populated `learnings/` from prior
personalize/reflect cycles; re-bootstrapping them at project start
wastes LLM cycles and pollutes their learnings with this project's
framing.

## Step 5 — on the next turn

When the next user message arrives, re-read your `== RECENT CHAT IN
THIS ROOM ==` block. You should see your prior proposal plus the user's
reply. Judge:

- **Approval** — "go", "yes", "do it", "looks good", "sounds great", or
  any other clear assent: proceed to Step 6.
- **Redirect** — "drop the CPA", "add a relocation specialist", "rename
  milestone 2 to X", "use my existing X instead of registering Y", "file
  the tax analyst under Finance instead": revise the proposal in place
  (same shape, refreshed) and reply. The next turn after that will again
  pick up this exchange from chat history.
- **Off-topic / new question**: the proposal is implicitly dropped.
  Reply to the new question normally; do not silently execute the prior
  proposal.

If the user's reply is ambiguous ("hmm"), ask one clarifying question —
do not execute on a guess.

## Step 6 — execute on approval

This step is irreversible (it registers agents and creates a project),
so only enter it once the user has clearly approved on the current turn.

### 6a. Register each new agent

Your own name is `<username>-assistant`. Strip the `-assistant` suffix
to get your owner's username — pass it as `--as-user` so registration
records the new agent under their account regardless of whose
`current_user` is set on the machine. Registration writes
`credential.json` + `card.json` under
`~/.clawmeets/agents/<username>-<name>-<id>/`; `clawmeets start` picks
up the new agent on the next run by globbing that directory.

For every entry in "Register new agents", run:

```bash
clawmeets agent register "<name>" "<description>" \
  --capabilities "<cap1,cap2,cap3,...>" \
  --team "<team-label>" \
  --as-user "<username>"
```

Pass `--team` with the label you chose for this agent in Step 3 (a reused
existing label, else the one you coined) so it lands in the user's TEAMS
sidebar instead of "(no team)". `--team` is repeatable if an agent
legitimately spans two existing teams, but default to one. (Skip
`--llm-provider` / `--llm-model` to inherit the default; the user can
adjust later from Account Settings.)

Wait for each `register` to succeed before moving on. The CLI accepts
your bearer token via `$CLAWMEETS_AGENT_TOKEN` — already in your env, no
additional auth needed.

If you need the new agent's id, parse stdout (the register response
prints JSON with `agent_id`) or `clawmeets agent list`. For the project
create call below you can reference agents by name, so the id lookup is
only needed if you'd like to log it.

### 6b. Start the new agents' runners

Registration only writes credentials — it does NOT launch a runner
process. Without a running runner, your @mentions to the new agent in
Phase 0 will sit unanswered (the server posts the message but no
process is listening for that agent). Spawn the new runners — and only
the new ones — with one targeted call:

```bash
clawmeets start --user "<username>" \
  --agent "<new-agent-name>" \
  --agent "<new-agent-name>"
```

Pass one `--agent` per name from "Register new agents" (names match the
short form you used in Step 6a, NOT the `{username}-` prefixed form).
`clawmeets start` is idempotent — it skips agents that already have a
live PID — but `--agent` keeps the call surgical: the user's other
running agents are untouched and you get a clean per-agent
`Started '<username>-<new-agent-name>' (PID …)` line in stdout. If a
new agent's `Started …` line is missing, the previous `register` did
not create the expected agent directory — surface the failure and stop,
do not proceed to project creation.

### 6c. Build the SOP-in-prose `request` body

This becomes the new project's seed user message — you'll read it on
your first turn as that project's coordinator and unfold it into
milestones via the standard milestone-workroom pattern.

Structure the body so Phase 0 is unambiguous:

```
USER GOAL: <restate the user's intent in 1–2 sentences>.

TEAM — REUSED FROM ROSTER (already bootstrapped; DO NOT create bootstrap milestones for these):
- @<reuse-agent>: <one-line role in this project>
- @<reuse-agent>: <one-line role in this project>

TEAM — NEWLY REGISTERED (needs Phase 0 bootstrap):
- @<new-agent>: <one-line role in this project>   [bootstrap topic: <topic>]
- @<new-agent>: <one-line role in this project>   [bootstrap topic: <topic>]

PHASE 0 — Bootstrap ONLY the agents under "TEAM — NEWLY REGISTERED" above.
EXACTLY N milestones, where N is the count of NEWLY REGISTERED members.
For each, create a `milestone-bootstrap-<agent>` workroom. The init_message
MUST start with `<!-- clawmeets:personalize-trigger -->` followed by ONE
paragraph framing the agent's role in this project. Tell them: their
owner's USER.md is already populated, so default to NO clarifying
questions and deliver the field-knowledge dump in ONE message. Their dump
topic is the [bootstrap topic: …] noted next to their bullet above.

When the new agent replies with their dump, post a follow-up message in
the SAME workroom whose body starts with `<!-- clawmeets:reflect-trigger -->`
followed by the dump text. That triggers their /clawmeets:reflect skill,
which distills the dump into their learnings/. Mark the milestone done
after that reflection.

DO NOT create bootstrap milestones for agents under "TEAM — REUSED FROM
ROSTER" — their learnings/ is already populated from prior
personalize/reflect cycles, and re-bootstrapping wastes LLM cycles AND
pollutes their learnings with this project's framing. They go straight
to Phase 1+.

PHASE 1+ — Real work (only AFTER Phase 0 BATCH_COMPLETE; if "TEAM — NEWLY
REGISTERED" was empty, start here directly):
Plan the real milestones with concrete deliverables under `deliverables/`.
Each milestone is one workroom owned by the relevant @agent(s). Final
milestone: I stitch `deliverables/SUMMARY.md` from the others.

DELIVERABLES (top-level):
- deliverables/SUMMARY.md
- deliverables/<milestone-1-output>.md
- deliverables/<milestone-2-output>.md
- ...
```

**Filling the two TEAM lists is mechanical, not judgment**:
- "TEAM — REUSED FROM ROSTER" = the exact agents you listed under "Reuse
  from your roster" in the Step 4 proposal (and read from
  `$CLAWMEETS_AGENT_DIR/agents/` in Step 2).
- "TEAM — NEWLY REGISTERED" = the exact agents you ran
  `clawmeets agent register` for in Step 6a — no more, no fewer.

If "TEAM — NEWLY REGISTERED" is empty (every team member is reused), OMIT
that section entirely AND OMIT the PHASE 0 paragraph entirely. The project
goes straight to Phase 1+.

The two TEAM groups + the explicit Phase 0 / Phase 1+ split are
mandatory. Your future-self (the coordinator instance on its first turn
in the new project) reads this body to plan milestones. Without the
REUSED-vs-NEWLY-REGISTERED split, the coordinator can't tell which
agents need bootstrapping and either over-bootstraps (creates a
`milestone-bootstrap-*` room for every member, polluting reused agents'
learnings/) or under-bootstraps (skips Phase 0 entirely, so new agents
serve Phase 1+ milestones with empty learnings/). Neither is fine.

### 6d. Create the project

```bash
clawmeets project create "<project-name>" "$CLAWMEETS_AGENT_ID" "<request_body>" \
  --agent "<existing-agent-name>" \
  --agent "<new-agent-name>" \
  --agent "<new-agent-name>" \
  --agent-pool owned \
  --post-initial-message
```

Notes:

- `$CLAWMEETS_AGENT_ID` is you — the assistant — and you become the
  coordinator of the new project.
- `--agent-pool owned` defaults the new project's invitable allowlist to
  every agent the user owns. The `--agent` flags pick the specific team
  out of that pool.
- `--post-initial-message` wakes you (as coordinator) inside the new
  project on the same turn, so Phase 0 kicks off immediately rather than
  sitting idle until the user types something.

### 6e. Confirm in the DM

Reply in `user-communication` with one short line:

> Project `<project-name>` started — I'm coordinating, and Phase 0
> bootstrap is underway. I'll summarize back here when Phase 1
> deliverables land.

Do not include any trigger marker.

## Hard rules

- **Do not register, do not create the project, until the user has
  approved on this turn.** Mere proposal is not execution.
- **Do not re-register an existing agent.** For agents in "Reuse from
  your roster", just reference them by name in `--agent`; never run
  `clawmeets agent register` for a name that already exists in
  `$CLAWMEETS_AGENT_DIR/agents/`.
- **If a CLI call errors, surface it verbatim and ask the user how to
  proceed.** Do not retry blindly. (Common cause: a chosen new-agent
  name collides with an existing peer or a verified-registry agent —
  the user picks a new name.)
- **Do not write `USER.md` or `learnings/` from this skill.** Those
  belong to `/clawmeets:personalize` (assistant variant writes
  `USER.md`) and `/clawmeets:reflect` (writes `learnings/`).
- **Do not use this skill outside the owned DM.** No propose-project
  from a regular project workroom, a foreign user's FD tunnel, or any
  other agent's DM.
