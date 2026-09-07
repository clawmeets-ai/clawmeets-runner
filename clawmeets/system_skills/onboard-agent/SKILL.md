---
name: onboard-agent
description: >
  Register a new agent AND get its memory populated, in one short project you
  coordinate. Use in your user's own DM when they ask you to add an agent and
  you want it to start with knowledge rather than a blank `learnings/`. Two
  optional halves — a mentor agent dumps what it already knows about the job,
  and the new agent deep-researches its own domain — then ONE reflect pass
  distills whichever ran into the new agent's durable memory. Both halves are
  optional; with both off this is just `register-agent`. Assistant-only.
---

# Onboard an agent

`register-agent` writes `credential.json` and `card.json`. That is all it
does — the agent comes up with an empty `memory/learnings/` and no idea what
your user cares about. This skill wraps that registration in a small project
you coordinate, so the agent's first real turn is spent *learning the job*.

Two things can fill its memory, and **each is independently optional**:

- **a mentor's knowledge dump** — what an agent already on this team knows
  about the user's domain, house conventions, and traps;
- **the new agent's own deep research** — the slice of its field a
  practitioner serving *this* user needs on day one.

Both feed a single `/clawmeets:reflect` pass at the end, which is what makes
any of it durable.

## Why a project and not a DM round-trip

You cannot do this from the DM alone. `clawmeets dm send` posts and returns —
nothing wakes you when the answer lands, so an onboarding doc DM'd to a mentor
would sit unread in a thread nobody is watching.

In a project you get the wake-up for free: when the agents you `@`-mentioned in
a room have all replied, the server fires `BATCH_COMPLETE` at you, the
coordinator, and you continue in the same room. That is the entire reason for
the project shape.

It also fixes a real bug. `clawmeets assistant register --auto-personalize` and
`clawmeets agent-team register --auto-personalize` post a bare
`<!-- clawmeets:personalize-trigger -->`; the agent replies with a 1500–3000
word field-knowledge dump; **nothing ever posts the matching reflect-trigger
back**, and the personalize skill explicitly forbids the agent from writing its
own `learnings/`. The dump dies as chat text. Here it doesn't.

## Step 1 — collect the agent's profile

Same fields as `register-agent`, and it is worth following that skill for the
prompting detail (draft the description, propose 5–8 capabilities, don't leave
either thin). You need:

name · description · capabilities · team (`--team`, the TEAMS-sidebar label) ·
knowledge dir · LLM provider/model · git repo, if the role is engineering-shaped.

## Step 2 — set the two switches

Ask the user only what you can't infer, in one line each. Defaults:

| Switch | Default | Turn it OFF when |
|---|---|---|
| **Mentor brief** | ON, with **you** as the mentor | The user says no mentor, or you genuinely hold nothing about this domain that the agent won't research better itself |
| **Deep research** | ON | The user says skip it, or an agent being *re-*onboarded already covers the ground in its `learnings/` |

**Who the mentor is.** Default is you — the assistant. You just drafted this
agent's description and capabilities and you hold `USER.md`, so you are usually
the best-informed party. Route to a **peer** agent only when the user names one
or when a peer clearly owns the domain (a senior backend agent briefing a new
junior one). The difference is mechanical, not structural: as mentor you write
the brief yourself in the room; a peer mentor is `@`-mentioned and writes it.

**If both switches are OFF**, do not create a project. Register and start the
agent from the DM (Step 3 minus the project), tell the user it's up with an
empty `learnings/`, and stop. An empty onboarding project on their desk is
worse than no project.

## Step 3 — create the project, then provision

Create the project **first**, so the coordinator turn it wakes is the one that
does the work.

```bash
clawmeets project create "onboard-<name>" "$CLAWMEETS_AGENT_ID" "$(cat /tmp/onboard-brief.md)" \
  --display-name "Onboarding: <name>" \
  --agent-pool owned \
  --agent "<planned-agent-name>" \
  --agent "<mentor-name>"          # omit when you are the mentor
```

Follow **`create-project`** for the call's mechanics (slug vs `--display-name`,
one create = one kickoff, what to do on a 4xx).

**Listing the planned name before the agent exists is legal, and is the trick
that makes this work.** `POST /projects` stores `agent_names` verbatim as
plain strings; the invitable check runs later, at *chatroom* create, and
`Project.matches_invitable` matches an agent's id, its full registry name, **or
the owner-relative short name** for agents the project owner registered. So the
name simply starts matching the moment you register it in the next step.

Two consequences worth expecting rather than debugging:

- On your **first** turn in this project the `INVITABLE AGENTS` block resolves
  against agents that actually exist, so with a solo (you-as-mentor) onboarding
  it renders **empty**. That reads like "you can't delegate." Ignore it — you
  create no room on that turn.
- If the name you actually register differs from the one you pre-listed, widen
  the list rather than re-creating the project:
  `clawmeets project allowlist <project-id> --agent <actual-name>`. The update
  replays before your next turn; no restart.

### Provision (milestone 1 — no workroom)

```bash
clawmeets agent register "<name>" "<description>" \
  --capabilities "<cap1,cap2,…>" \
  --team "<team-label>" \
  [--knowledge-dir "<path>"] \
  [--git-url "<repo>" [--git-base-branch "<branch>"]] \
  [--llm-provider "<provider>" [--llm-model "<model>"]] \
  --as-user "<owner-username>"

clawmeets start --user "<owner-username>" --agent "<name>"
```

Set everything you can **at register time**. `--knowledge-dir`, `--git-url` and
`--git-base-branch` all land in `card.json` `local_settings` from this one call;
chasing it with `clawmeets agent reconfigure` is only for what you learn later.

**Hard gate: `clawmeets start` must print `Started '<owner>-<name>' (PID …)`.**
If that line is missing, stop the project here and tell the user. Registration
only writes credentials — with no runner process, your `@`-mention in the next
milestone is delivered to nobody, and the symptom is not an error but a silent
30-minute `BATCH_TIMEOUT`.

---

## The bootstrap contract

*This section is the shared unit. This skill runs it once, for the agent being
onboarded. `propose-project` runs it once per newly-registered member during
its Phase 0. It is written to be read standalone: it takes `agent_name`,
`bootstrap_topic`, `mentor`, and the two switches from Step 2 — nothing else.*

**One room. Mentor and new agent both in it. Serial by `@`-mention.**

```
create_room  milestone-bootstrap-<agent_name>
participants: you (coordinator) + <mentor, if a peer> + <agent_name>
```

The room is the transport. Every agent's prompt carries a
`== RECENT CHAT IN THIS ROOM ==` block — the last 10 non-ack messages, **whole
bodies, no truncation, own messages included** — so the new agent reads the
mentor's dump simply by being in the room when it was posted. Room files sync
to every participant too, so anything the mentor writes to
`onboarding-<agent_name>.md` is readable by the new agent directly. Nothing has
to be copied, re-inlined, or re-posted.

Only `@`-mentioned participants are expected to reply (the server resolves
mentions to `expects_response_from`; an un-mentioned participant reads and
stays silent). That is what makes three turns in one room serial.

### Turn 1 — the mentor's knowledge dump *(skip if the switch is OFF)*

**If a peer is the mentor**, this is the room's opening message, `@`-mentioning
the mentor and nobody else. Ask for what a practitioner already on this team
knows that the new agent cannot look up: the user's domain and business, house
conventions, who owns what, the traps, the things that have already gone wrong.
Tell them to post it as their reply **and** save it to
`onboarding-<agent_name>.md` in this room.

**If you are the mentor** (the default), there is no separate turn: write the
brief yourself as the room's opening message, save the same file, and put the
turn-2 trigger in that same message.

Either way you end this turn holding the brief, and the file is the artifact
your user can read later.

**Do not create a knowledge pack.** `knowledge_packs/` is the user's curated
shelf; agents do not write to it. Offer the promotion at the end (Step 5) and
let them decide.

### Turn 2 — the agent's deep research *(skip if the switch is OFF)*

Message `@<agent_name>`, body starting with the marker on its own line:

```
<!-- clawmeets:personalize-trigger -->
```

Then, in prose: one paragraph framing its role and why this user registered it;
a pointer to the mentor's brief above in this room (name what it covers — don't
just say "see above"); the bootstrap topic, **sharpened by the brief** if there
is one; and the standing instruction that `USER.md` is already populated, so it
should skip clarifying questions and deliver the dump in ONE message.

The topic is what makes or breaks this turn. "Your field" produces a generic
primer. Name the slice this user actually needs, in the vocabulary the mentor
just used.

*(With the mentor switch OFF, this is the room's opening message and there is
no brief to point at — the topic comes from the capabilities the user asked
for.)*

### Turn 3 — reflect, once

Message `@<agent_name>`, body starting with:

```
<!-- clawmeets:reflect-trigger -->
```

Then one short paragraph: state that **no `== Recent activity ==` transcript is
attached and the room conversation in their prompt is the source for this
cycle** — the mentor's brief and their own dump — and to distill both into
`learnings/`.

**One reflect pass covers both halves**, because both are messages in this
room. Two triggers would not have helped anyway: reflect is idempotent per day,
so a second one on the same date replies "Already reflected for today" and
skips both its passes.

Two things to respect:

- **Post it as the immediate next message after the dump.** The history window
  is 10 messages; this room holds 3–5, so you have room to spare — but don't
  spend it on chatter between the dump and the trigger.
- **Skip this turn entirely only if both switches were OFF** — in which case
  you never created the room.

`last_reflected_at` is not bumped by an in-room reflect (the subscriber that
moves that cursor only watches `user-communication`). Harmless: the nightly
schedule still fires for this agent, and reflect's own one-per-day guard makes
that firing a no-op.

---

## Step 4 — mark the milestone and close

Standard coordinator bookkeeping: assess the milestone on `BATCH_COMPLETE`,
tick its checkbox in `## Milestones` (`clawmeets plan update <project> --section
milestones --body-file <f>`), post the pass/fail detail into the milestone's own
chatroom, then `project_completed`. Do not add a log section to the plan — the
room is the log. There is no Phase 1 here — the onboarding *is* the project.

An onboarding project is a regular project, so it starts in `spec-ing` and the
server will refuse a workroom until the plan is accepted. Fill the plan, point
the user at the approval note the server filed for them, then **stop** — you
cannot accept it yourself, and there is no command to try. Acceptance is the
user applying that note, which writes *"User approves the plan."* into the
plan's `## Approval` section; **that line is your go signal, and a "looks good"
in chat is not**. You do not need to hold the turn open: their acceptance wakes
you, and that is the turn in which you open the first workroom.

## Step 5 — report back in the DM

One short message to your user:

> `<name>` is registered, running, and onboarded. It now holds
> `<one line: what the mentor covered / what it researched>` in its own memory.
> The brief is saved at `onboarding-<name>.md` in the onboarding project — say
> the word and I'll turn it into a knowledge pack you can install on your other
> agents.

Offer the pack; never create one unasked.

## Hard rules

- **Never run the `knowledge-pack` CLI from this skill.** Promotion of a brief
  into a pack is user-initiated, always.
- **Never write the new agent's `learnings/` yourself.** That is
  `/clawmeets:reflect`'s job, and only the agent can write its own memory.
- **Don't run the bootstrap contract on an agent you reused.** It already has
  populated `learnings/`; re-bootstrapping burns cycles and pollutes it with
  this project's framing.
- **One create, one kickoff.** Don't follow `project create` with a second
  coordinator-directed message — you'll mint a duplicate set of workrooms.
