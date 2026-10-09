# SPDX-License-Identifier: MIT
"""
clawmeets/models/thinking_partner_plan.py

The PLAN.md v1 that the "Add a thinking partner for big decisions" starter
to-do creates its project with (``clawmeets project create --plan-file``). The
assistant fills in the one ``<…>`` line under Goal (the owner's areas, from
USER.md) and passes the rest through word for word, so the project opens with
its plan already written and the owner's one click — dismissing the start
gate — starts it. Same shape as ``models/onboarding_plan.py``; see that module
for why a plan is kept apart from the to-do that carries it.

The scope is fixed here on purpose. Left to draft its own plan, the assistant
consulted thought_partner, which asked for two more research areas and put a
scope call in front of the owner before anything had started. Finding the real
question (AC-3.4) and narrowing with cheap tests (AC-3.5) are now in the
template, and the request tells the assistant not to consult anyone on it.

Two orderings in here are product decisions, pinned in
tests/test_desk_todo_seed.py:

* thought_partner deep-researches its craft BEFORE it receives the brain dump,
  the same rule as every agent onboarding registers. The assistant writes the
  brain dump in parallel, so the rule costs no wall-clock time.
* thought_partner is registered by M1, AFTER the owner dismisses the start
  gate, not by the to-do. Same as onboarding's team: the to-do itself leaves
  nothing behind, and an owner who never dismisses the gate gets no agent.
  This works because a project's allowlist is a list of names matched
  against the live roster each turn, so ``--agent thought_partner`` can be
  named at create before the agent exists.
* The gaps table is built only once both are in hand, and every brain-dump
  item gets a row, including the ones no method fits.
"""
from __future__ import annotations

# The to-do prompt wraps this in onboarding's PLAN_TEMPLATE_START/END markers:
# each prompt carries exactly one plan, so one pair of markers serves both.
THINKING_PARTNER_PLAN_TEMPLATE = """\
# Add a thinking partner for big decisions

## Goal

thought_partner is a running agent I can bring decisions and strategic
questions to, across every area of my work and life, and it starts out already
knowing how I decide rather than with a blank memory.

A thinking partner is only useful if it can question me well and widen my
options from day one. That takes practitioner-grade method (how good
consultants and coaches question people; when each ideation technique helps or
misleads) and an honest picture of my own decision history, with the places
that picture is thin called out instead of papered over.

My areas: <each business, role or part of life from my USER.md, comma-separated>

## Not Authorized

- Firing any to-do. I start each one myself.
- Registering any agent other than thought_partner (M1 registers it).
- Contacting anyone outside my own agents.
- Putting secrets (passwords, keys, account numbers) into the brain dump.

## Acceptance Criteria

### G1: thought_partner is live and equipped

- [ ] **AC-1.1** — thought_partner is in my agent list, online, with a role
      description that reflects my areas and background from USER.md.
- [ ] **AC-1.2** — thought_partner can run the grill-me and broaden-options
      skills when asked.

### G2: thought_partner knows how I decide

- [ ] **AC-2.1** — thought_partner's memory holds an account of how I decide,
      each item tagged observed (I said or did it, with a pointer to where),
      inferred, or unknown. It covers: past decisions and the reasons behind
      them; which bets held up and which didn't; the assumptions I keep
      making; the questions I tend to avoid; how I react when challenged (dig
      in, go quiet, ask for data, or move to action); decisions I reversed or
      let drift, and what finally moved me; my default speed and risk appetite
      in each of my areas; who I consult and whose view outweighs data for me;
      where what I ask for differs from what I then do; and the decisions open
      right now, with deadlines where they are already known.
- [ ] **AC-2.2** — Everywhere that account is thin, guessed, or unsupported, it
      says so explicitly rather than filling the gap.

### G3: thought_partner knows its craft to practitioner depth

- [ ] **AC-3.1** — thought_partner's memory describes at least 8 named
      questioning techniques used by skilled consultants and coaches (for
      example clean language, Socratic questioning, the GROW model, Mom
      Test-style questions about past behaviour, laddering / "5 whys", silence
      and reflective listening), each with an example question and one way it
      fails, sourced from practitioner or research work rather than listicles,
      with sources a reader can open.
- [ ] **AC-3.2** — For each ideation method (analogy, constraint flip,
      assumption reversal, pre-mortem, perspective shift, plus the three
      baseline options: do nothing, do both, one then the other) its memory
      says when the method works and when it fails or misleads, and when not
      to brainstorm at all because the real problem is framing or commitment
      rather than too few options. Every works/fails claim cites a source a
      reader can open, and each method has at least one empirical source (for
      example Klein's pre-mortem work, Mitchell, Russo & Pennington 1989 on
      prospective hindsight, Gick & Holyoak on analogical transfer).
- [ ] **AC-3.3** — The research is reconciled against my decision history in a
      gaps table: one row per blind spot or decision pattern from the brain
      dump, the method thought_partner would use for it (or "none fits"), and
      a confidence rating (supported, inferred, or guessed) traced to the
      brain-dump item behind it. A brain-dump item that maps to no method gets
      its own row.
- [ ] **AC-3.4** — thought_partner's memory explains how practitioners find
      the real decision behind a stated request (reframing, issue-tree /
      hypothesis framing, and the evidence on how often decisions solve the
      wrong problem, e.g. Nutt), with at least two concrete reframing moves it
      can use in conversation, with sources a reader can open.
- [ ] **AC-3.5** — thought_partner's memory explains how to narrow a wide set
      of options to 2-3 (screening on impact, effort and reversibility) and how
      to design the cheapest test that would change the decision (a prediction
      written in advance plus a kill criterion), including the ways cheap tests
      mislead (too small to mean anything, built only to confirm, sunk cost),
      with sources a reader can open.

### G4: I hear back

- [ ] **AC-4.1** — I receive, in my DM with my assistant, a self-contained
      summary of what thought_partner now knows about how I decide and the
      full list of gaps it flagged.

## Milestones <!-- layer: detail -->

### M1: Register and equip thought_partner  <!-- advances: AC-1.1, AC-1.2 -->

- [ ] **M1** — assistant
- Register thought_partner with your register-agent skill, unless it is
  already on my roster, and make sure it is running.
    Name:      thought_partner  (keep this exact name; its starter messages
               are tied to it)
    Industry:  decision-making and strategic thinking for my work and life,
               across my areas
    Expertise: consultant-style interviewing that finds the real question
               behind a request; structured brainstorming (analogies,
               constraint flips, assumption reversal, pre-mortems,
               perspective shifts) followed by narrowing to 2-3 options with
               cheap tests
    Mentor:    my assistant
  Build its role description from my areas and background in USER.md.
- Install grill-me and broaden-options on it with your install-skill skill
  and confirm both show on it. File it under a team with your manage-team
  skill.
- Then add it here, before any milestone gives it work:
  `clawmeets project allowlist <project> --agent thought_partner`.

### M2: Deep research  <!-- advances: AC-3.1, AC-3.2, AC-3.4, AC-3.5 -->

- [ ] **M2** — thought_partner, in parallel with M3
- Research to practitioner depth: questioning technique, finding the real
  question behind a request, ideation methods and when each fails, and
  narrowing to 2-3 options with cheap tests. Cite sources a reader can open;
  prefer practitioner and research work over listicles. This makes it an
  industry veteran before it learns about me.

### M3: Brain dump  <!-- advances: AC-2.1, AC-2.2 -->

- [ ] **M3** — assistant (mentor), in parallel with M2
- Write onboarding-thought_partner-from-assistant.md from my USER.md, your
  ledger on me and our past project history, covering every item in AC-2.1
  and tagging each observed, inferred or unknown. Never ask me for due dates;
  an unknown deadline is written down as unknown. Summaries only; leave out
  secrets.
- Hand it to thought_partner once M2 is finished, not before.

### M4: Reconcile and memorize  <!-- advances: AC-3.3 -->

- [ ] **M4** — thought_partner
- Reconcile the research against the brain dump in the gaps table, flagging
  every place my practice departs from what the research found.
- Run your reflect skill once to commit the research, the brain dump and the
  gaps table to memory, then post an inventory of what you know plus every
  gap or conflict you flagged.

### M5: Report back  <!-- advances: AC-4.1 -->

- [ ] **M5** — assistant
- In my DM with you (the conversation this project was spawned from), send a
  summary I can read with nothing else open: what thought_partner now knows
  about how I decide, and every gap it flagged.
- Wrap up with a completion report.
"""
