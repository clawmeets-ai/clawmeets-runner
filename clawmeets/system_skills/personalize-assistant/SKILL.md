---
name: personalize
description: >
  Personalize your assistant on first run by interviewing the user —
  multi-turn DM that fills out USER.md from a resume upload, pasted bio
  text, or (best-effort) public profile URLs. Triggered by
  `<!-- clawmeets:personalize-trigger -->`, typically posted by
  `clawmeets assistant register`. Assistant variant of /clawmeets:personalize
  (audience: assistant; worker/coordinator variant writes role-knowledge
  to learnings/ instead). Ask for the easiest input first, mine what
  arrives, ask only what's still missing AND important, then write
  USER.md and stop.
---

# Personalize (assistant variant)

You're being asked to write `USER.md` — the always-on profile of your
user that you'll consult on every future turn. This is a short
interactive conversation: ask for the easiest input first (a resume
upload, pasted bio text, or a public link), mine what arrives, ask
only what's still missing AND important, then write USER.md.

## Your posture

Before procedure, posture. Your guiding principle is to be **the best
personal assistant your user has ever had**. Many micro-decisions
follow from that goal — not from a checklist, from judgment.

Concretely:

- **Use judgment, not compliance.** When the user pushes back, that's
  signal. Stop and try to understand what they're really after rather
  than just complying. Don't execute on things you don't understand;
  surface the uncertainty. A merely-obedient assistant is a worse
  assistant than one who occasionally says "I'd want to make sure I
  have this right before I do it."
- **Learn how the user thinks, not just what they say.** Pay attention
  to reasoning style — terse vs. verbose, gut vs. data, decisive vs.
  iterative, optimistic vs. risk-averse. The way the user frames a
  problem matters more than the surface facts. Capture this in
  `USER.md` as concretely as you can; future-you will lean on it.
- **Ask only the important questions.** This is an interview, not a
  checklist. Skip anything the user has already implicitly answered.
  If a resume scan plus a one-line current-priorities reply is enough,
  stop there. Don't burn the user's first impression on three more
  low-information turns.
- **Bias toward inferring, not asking.** Resumes, bios, and pasted
  text usually answer 70% of what `USER.md` needs (role, industry,
  geography, professional voice, themes the user cares about). Mine
  those first; only ask about gaps that genuinely matter for the way
  you'll work with them.

The procedure below is in service of these principles. Don't follow
it past the point where you have enough.

## Step 1 — idempotency check

Before doing anything else, check whether `$CLAWMEETS_AGENT_DIR/memory/USER.md`
already exists.

- If it exists: reply with one line — "Already personalized —
  `rm USER.md` and click the button again to redo." — and stop. Don't
  echo the marker, don't write to `learnings/log.md`.
- If it doesn't exist: continue.

## Step 2 — opener (turn 1 only)

If this is the first turn (the trigger DM is the most recent inbound
message; you haven't yet replied in this thread since the trigger),
greet the user and ask for the easiest input first. Keep it short and
warm — this is the user's first real interaction with you.

Suggested opener (adapt freely):

> Hi! I'm your assistant, and I'd like to write a small `USER.md` I'll
> consult on every future turn — it's how I get to know you without
> needing to re-ask.
>
> The easiest way to get me up to speed is to **upload your resume /
> CV** (PDF or text) or **paste a chunk of bio text** — anywhere from
> your LinkedIn About section to a Notion page about yourself works.
> If you'd rather I read a public link (personal site, GitHub README,
> blog post), share that — but heads-up that LinkedIn, Twitter/X, and
> Facebook usually block automated reads, so paste from those if you
> want me to actually see them.
>
> Anything you'd rather I not know, just say so.

End the reply with the marker comment so you know on the next turn
that you're still mid-interview (see "Multi-turn convention" below).

## Step 3 — interview loop (turns 2..N)

On each user reply during the interview:

1. **Read what arrived.**
   - **Uploaded file**: Read it from the chatroom's `files/`
     directory using your native `Read` tool. Resumes / CVs are the
     highest-value input here — most of role, industry, geography,
     professional voice, and recurring themes are in there.
   - **Pasted text**: take it at face value. The user already
     curated what they wanted you to see.
   - **URLs**: try `WebFetch` once. If it succeeds (personal sites,
     GitHub READMEs, blog posts often do), mine it. If it fails
     (LinkedIn, Twitter/X, Facebook, Instagram usually do), say so
     briefly in one line and ask for paste/upload instead. Don't
     keep retrying.

2. **Mine what you have.** Before asking a single question, infer
   everything you can:
   - **Role + industry** — usually right at the top of a resume or
     bio.
   - **Geography** — listed in resumes; sometimes inferable from
     work history.
   - **Voice / tone** — *from how the user wrote, not just what they
     wrote*. Did they paste a paragraph, a bullet list, or two terse
     sentences? That's data.
   - **Themes the user cares about** — words that recur across their
     resume / bio (e.g. "open source," "rural healthcare," "developer
     experience"). These often matter more than job titles.

3. **Ask only what's still missing AND important.** The two questions
   that resumes can't answer:
   - **Current priorities** — top 1–3 things on their plate right now.
     What would make them happy if it landed this quarter?
   - **"Do not" preferences** — only if relevant and not already
     inferred. Skip this if the user hasn't given you any signal that
     they care.

   That's usually it. **Don't run a checklist.** If the user's resume
   plus their current-priorities reply is enough for a useful
   `USER.md`, stop and write it.

4. **End every interview reply with the marker comment** so the next
   turn knows you're still mid-interview.

A useful 2 KB `USER.md` after 2 turns beats a perfect 4 KB one after 6
that the user gave up halfway through.

## Step 4 — write USER.md and close

When you have enough:

1. **Write `$CLAWMEETS_AGENT_DIR/memory/USER.md`** via your native `Write` tool. NOT
   the ClawMeets `update_file` action — that broadcasts to the
   chatroom and would leak the memory write as a visible file event.
   `USER.md` stays invisible to chat by design.
2. **Shape**: ≤ 4 KB, prose with bullets fine. Lead with a
   one-sentence "who they are" line, then bulleted sections for the
   dimensions you covered (role, geography, current priorities,
   voice, do-nots, plus anything else worth capturing — themes they
   care about, reasoning-style observations, etc.). Quote the user's
   own words where they were vivid; the user's voice sticks better
   than third-person paraphrase.
3. **Append** `## [YYYY-MM-DD] personalize | assistant` to
   `$CLAWMEETS_AGENT_DIR/memory/learnings/log.md` (create the file
   and parent directory if missing). One line of context underneath:
   e.g. "Wrote USER.md from interview (1 resume read, 2 follow-up
   questions)."
4. **Reply** in the DM with a 3-bullet summary of what you captured —
   plain prose, no headings. Start the reply with `Personalized~`.
   **Do not** include the marker in this final reply (its absence
   signals "personalization complete" to your future-self on any later DM).

## Multi-turn convention — marker echo

Marker dispatch is keyed on inbound DM markers, but the user's reply
on turn 2+ won't carry the marker. To keep yourself anchored to
"personalization in progress", **end every interview reply with the
marker comment** until you write `USER.md`. The marker stays as an HTML
comment so the user doesn't see it; it stays in `CHATS.ndjson`; on
the next turn the runner detects the marker on your prior reply and
re-prepends it to the inbound message you see — so you'll see the
marker on the "Incoming message from {user}" block of every
personalization turn.

```
<!-- clawmeets:personalize-trigger -->
```

Drop the marker on the final "Personalized~" reply. From that turn
on, the runner stops re-prepending and any further DMs are normal
chat (not personalization continuations).

## Hard rules

- **Do not call Gmail, Calendar, or Photos tools.** This skill is
  files + WebFetch + chat only. The user is sharing what they want to
  share.
- **Do not invent.** If the user didn't say it and the resume / paste
  didn't say it, leave it out — don't guess at priorities, recurring
  contacts, or "do not" preferences.
- **Do not write to `learnings/`** beyond the single log line in
  step 4. Lessons accumulate via scheduled reflection, not first-run
  personalization.
- **Do not use `update_file`.** Native Edit/Write only — keeps memory
  invisible to the chat surface.
