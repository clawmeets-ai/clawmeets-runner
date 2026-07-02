---
name: personalize
description: >
  Personalize yourself on first run by reading USER.md and delivering a
  tailored field-knowledge dump for this user. Triggered by
  `<!-- clawmeets:personalize-trigger -->` — typically posted by
  `clawmeets agent-team register`'s fan-out or the "Personalize" button
  in a DM. Worker/coordinator variant of /clawmeets:personalize (audience:
  worker + coordinator; assistant variant runs the multi-turn USER.md
  interview instead). First check your existing learnings/ — if the
  bootstrap topic is already covered from prior work, skip (or narrow to a
  delta) the dump so a re-bootstrapped / reused agent doesn't deep-research
  the same ground twice. Otherwise read what's known about the user, judge
  whether 1–3 user-specific facts are still missing that would substantially raise
  the quality of the dump (e.g. industry + geography for a career coach),
  ask only those, then reply with a ~1500–3000 word field-knowledge dump
  anchored on what's now known.
---

# Personalize (worker/coordinator variant)

You're being asked to deliver a tailored field-knowledge dump — the slice
of your domain an experienced practitioner serving *this user* would
actually need on day one. Before writing, judge whether a couple of
targeted questions about the user would substantially raise the quality
of what you produce. If yes, ask them first. If no, go straight to the
dump.

## Your posture

Before procedure, posture. Your guiding principle is to be **genuinely
useful to this specific user** — not "informative about your field in
general." Many micro-decisions follow from that goal — not from a
checklist, from judgment.

Concretely:

- **Bias toward inferring, not asking.** Most of what you need is
  usually already in `USER.md` (role, industry, geography, voice,
  recurring themes). Mine it first. Only ask about gaps that genuinely
  change the dump.
- **Ask only the important questions.** This is not a screening
  interview. Skip anything you can infer or anything that wouldn't
  materially reshape your research. One sharp question beats three
  shallow ones.
- **Use judgment, not compliance.** When the user pushes back or
  redirects, that's signal — adjust. Don't drift into a generic field
  primer just because the questions weren't answered cleanly.

The procedure below is in service of these principles. Don't follow it
past the point where you have enough.

## Step 1 — read what you already know

Read `$CLAWMEETS_AGENT_DIR/memory/USER.md` if it exists (it may not —
the user might not have personalized their assistant yet, and that's
fine). Mine it for anything relevant to your role.

Also read `$CLAWMEETS_AGENT_DIR/memory/learnings/INDEX.md` if it exists,
plus any `learnings/<topic>.md` page whose one-liner looks related to the
bootstrap topic you've been handed. This is what tells you whether you've
already deep-researched this ground in a prior project. (Empty or missing
learnings/ is the common newly-registered case — that just means you have
nothing yet and will do the full dump below.)

Re-read the trigger DM itself — the frontend inlines your role and
capabilities as a reminder. Your own `== AGENT ID ==` block in this
prompt also carries them.

## Step 1.5 — already-covered check (skip redundant research)

Before spending a full deep-research dump, decide whether you even need
one. This is the de-dup gate — its whole job is to keep a reused /
re-bootstrapped agent from researching ground it already owns.

1. Identify the **bootstrap topic** from the trigger DM's framing
   paragraph (the coordinator names it explicitly, e.g. "bootstrap topic:
   SF condo capital-gains rules").
2. Compare it against your `learnings/INDEX.md` + any matching topic pages
   you read in Step 1. Then branch:

   - **Fully covered** — your learnings already give you working depth on
     this exact slice. Reply with a single line, e.g. *"I already have
     deep coverage of `<topic>` from prior work — ready for Phase 1."*
     **Drop the marker** (this is your final reply — see "Multi-turn
     convention"). No questions, no dump. Stop here.
   - **Partially covered** — you have some of it but this project needs a
     slice you haven't researched. Default to **no** clarifying questions;
     deliver a **scoped 2–4 paragraph dump on the uncovered delta only**,
     opening with a one-line lede naming what you already had (so the
     coordinator and your future reflect pass know it's a delta, not a full
     dump). Drop the marker. Skip the rest of this skill.
   - **Not covered** (empty/irrelevant learnings/, the common
     newly-registered case) — proceed to Step 2 for the full dump.

The full-dump path (Steps 2–4) runs **only** on the "not covered" branch.

## Step 2 — gap evaluation (judgment, not checklist)

Ask yourself: **given my role and capabilities, which 1–3 user-specific
facts would substantially raise the quality of a tailored field-
knowledge dump?** Then: are any of them missing from `USER.md` and not
inferable from what you've already read?

This is pure judgment — the right questions vary wildly by role:

- A **career coach** usually needs the user's industry, geography, and
  career stage / seniority. Without those, "career advice" collapses to
  generic platitudes.
- A **nutritionist** usually needs dietary constraints, activity level,
  and health goals. Without those, recommendations are unsafe or
  pointless.
- A **financial advisor** usually needs jurisdiction, income bracket,
  and risk tolerance. Without those, suggestions risk being illegal or
  wildly off-base.
- A **software-architecture coach** usually needs the team's stack,
  team size, and the actual problem space — not just "I do
  engineering."

Do **not** treat these as a template. Reason from *your* role to *your*
gaps. If nothing material is missing — proceed straight to Step 4 on
turn 1.

## Step 3 — ask (only if needed; turn 1)

If you identified genuine gaps in Step 2:

1. Open with one short sentence acknowledging what you're about to do.
2. Ask **at most 3** targeted questions — one batched message, not
   drip-feed. Explain *briefly* why each matters to the dump (one
   half-line each).
3. **End the reply with the marker comment** so the next turn knows
   you're still mid-personalize (see "Multi-turn convention" below).

If you identified no material gaps: skip to Step 4 immediately on
turn 1 — don't ask filler questions.

## Step 4 — write the dump and close

When you have what you need (turn 1 if no questions were needed, or
turn 2+ once answers are in):

1. Reply with a **~1500–3000 word field-knowledge dump** in the DM —
   the slice of your field an experienced practitioner serving *this
   user* would actually need on day one. Anchor every section on what
   you now know about them; don't drift into a generic primer. (This
   ~1500–3000 word figure is the "not covered" default. If Step 1.5 sent
   you down the partial-coverage branch, you already delivered the shorter
   delta there and never reach this step.)
2. Useful shape (adapt freely): a one-line "framing for you" lede,
   then 4–8 sections covering the highest-leverage sub-areas of your
   field for this user, then a short "what I'd watch next" tail. Cite
   the user's own situation in the prose where it sharpens a point.
3. **Do not** include the marker in this final reply — its absence
   signals "personalize complete" to your future-self on any later DM.

## Multi-turn convention — marker echo

Marker dispatch is keyed on inbound DM markers, but the user's reply on
turn 2 won't carry the marker. To keep yourself anchored to "personalize
in progress", **end every personalize reply with the marker comment**
until you write the final dump. The marker stays as an HTML comment so
the user doesn't see it; it stays in `CHATS.ndjson`; on the next turn
the runner detects the marker on your prior reply and re-prepends it to
the inbound message you see — so you'll see the marker on the
"Incoming message from {user}" block of every personalize turn.

```
<!-- clawmeets:personalize-trigger -->
```

Drop the marker on the final field-knowledge-dump reply. From that turn
on, the runner stops re-prepending and any further DMs are normal chat
(not personalize continuations).

## Hard rules

- **Run-scoped only.** The answers the user gives in turn 2 inform
  *this* dump. **Do not write `USER.md`** — that's assistant-owned and
  is filled by the assistant variant of `/clawmeets:personalize`.
  **Do not write `learnings/`** either — durable distillation is
  `/clawmeets:reflect`'s job, on its own cadence.
- **The Step 1.5 coverage check is read-only on `learnings/`.** You read
  it to decide whether to skip; you never write it here. Distillation of
  any delta you do deliver still happens later via `/clawmeets:reflect`.
- **Do not call Gmail, Calendar, or Photos tools.** This skill is chat
  + WebFetch only (and only if a user-supplied URL is in play).
- **Do not invent user facts.** If the user didn't say it and `USER.md`
  didn't say it, leave it out of the dump's user-specific framing.
- **Do not use `update_file`.** The dump is a chat reply, not a file
  artifact.
