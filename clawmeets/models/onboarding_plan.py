# SPDX-License-Identifier: MIT
"""
clawmeets/models/onboarding_plan.py

The PLAN.md v1 that the "Get to know me & staff my priorities" starter to-do
creates its project with (``clawmeets project create --plan-file``). The
assistant fills in the three ``<…>`` roster lines under Goal and passes the
rest through word for word, so the project opens with its plan already written
and the owner's one click — dismissing the start gate — starts it.

Kept apart from ``models/desk_todo.py`` because it is a project document, not
a to-do: the plan skill's rules govern it. Goal, Not Authorized and Acceptance
Criteria are the spec layer and lock once the owner accepts; ``## Milestones``
carries ``<!-- layer: detail -->`` so the coordinator can tick and re-cut them
freely. Every criterion is claimed by a milestone's ``advances:`` marker, which
is what keeps the coverage check satisfied when the milestones move.

Two orderings in here are product decisions, pinned in
tests/test_desk_todo_seed.py:

* Next tasks are proposed with all the confirmed priorities in view at once —
  1 to 3 of them in total, not 1 to 3 per priority.
* A new agent deep-researches its field BEFORE the assistant mentors it.
  Research turns a fresh model into an industry veteran; the brain dump then
  turns that veteran into one who knows this owner's institution, and the agent
  can only recognise where the house departs from the field once it knows the
  field.
"""
from __future__ import annotations

# Marker lines around the template inside the to-do prompt, so tests (and
# anyone editing it) can address the plan without parsing prose.
PLAN_TEMPLATE_START = "--- ONBOARDING PLAN TEMPLATE ---"
PLAN_TEMPLATE_END = "--- END TEMPLATE ---"

ONBOARDING_PLAN_TEMPLATE = """\
# Get to know me & staff my priorities

## Goal

My assistant knows me well enough to write my USER.md, my top priorities are
ranked, my next tasks are on my plate, and every task is addressed to the agent
(or the assistant-coordinated team) that will do it, ready for me to fire.
Nothing is started on my behalf.

Model agents on my roster: <names, or "none">
Existing agents: <name - one-line job, one per line>
Max new agents: 3

## Not Authorized

- Firing any to-do. I start each one myself.
- Writing USER.md before I confirm it.
- Registering an agent I have not approved, or more than the max above.

## Acceptance Criteria

### G1: My assistant knows me

- [ ] **AC-1.1** — My USER.md covers my role, company, industry, how I like to
      work and my top priorities, ranked, and I confirmed it.

### G2: My next tasks are on my plate

- [ ] **AC-2.1** — Every next task I confirmed is an open to-do on my plate,
      none duplicated.

### G3: Every task has someone to do it

- [ ] **AC-3.1** — Every new agent I approved is registered, mentored by my
      assistant, and can say what it knows about its field and about me, and
      what it flagged.
- [ ] **AC-3.2** — Every task from this project has a recipient and a first
      assignment I could fire without editing. A task that needs several agents
      is addressed to my assistant and says it will coordinate them in a
      project.

## Milestones <!-- layer: detail -->

### M1: Brain dump from my model agents  <!-- advances: AC-1.1 -->

- [ ] **M1** — every model agent, in parallel, one workroom each
- Send each the same brief: "Brief my assistant on me as a mentor would, from
  what your model has seen of my work on this computer: instruction files and
  saved memories, projects and repos, conventions and stack, decisions and
  why, what failed, recurring topics. Then give your read of (a) my top
  priorities right now, ranked, with the evidence for each, and (b) the
  USER.md basics: role, company, industry, how I like to work. Mark each item
  observed or inferred. Summaries only; leave out secrets."
- With no model agents, skip this; M2 interviews me directly.

### M2: Reconcile and confirm who I am  <!-- advances: AC-1.1 -->

- [ ] **M2** — assistant
- Merge the answers into one ledger (fact, which agents said it, agreed or
  conflicting or single-source) and commit it to your memory, noting which
  model agent said what.
- In user-communication, show me the draft USER.md and my ranked priorities,
  with every conflict side by side. Ask only what is still missing and
  important, easiest input first (a resume, a bio, a link). Never ask me for
  due dates. If USER.md already exists, fill only the gaps.
- Once I confirm, write USER.md with your personalize skill.

### M3: Propose my next tasks  <!-- advances: AC-2.1 -->

- [ ] **M3** — assistant
- With all my confirmed priorities in mind, propose 1-3 next concrete tasks
  and let me correct, add or cut them.

### M4: Put the tasks on my plate  <!-- advances: AC-2.1 -->

- [ ] **M4** — assistant
- Run `clawmeets todo list --no-archived` and reuse a to-do that already
  covers a task. Publish only when none does:
  `clawmeets todo publish --text "<title>" --draft-prompt "<the task>"`.
- List the to-do ids here.

### M5: Propose, register and onboard my team  <!-- advances: AC-3.1 -->

- [ ] **M5** — assistant, then each new agent
- Propose a team table: agent, new or existing, industry domain, expertise
  ("B2B SaaS pricing analyst", not "Researcher"), to-dos it owns. Reuse my
  existing agents first; propose a new one only where nothing fits, and no
  more than the max. Every new agent's mentor is my assistant.
- Interview me on it and change the table to what I approve.
- Register each approved agent, top priority first, with your register-agent
  skill and a role description built from its domain and expertise. Make sure
  it is running, then add it here:
  `clawmeets project allowlist <project> --agent <name>`.
- Onboard each new agent in its own workroom, in parallel, in this order:
  a. Deep research. The agent researches its domain to practitioner depth:
     the state of the art, standard tools and their trade-offs, common failure
     modes, where the field is heading. This makes it an industry veteran.
  b. Brain dump. My assistant mentors it with everything proprietary or
     non-obvious it now knows that touches this agent's domain and to-dos: me
     and my priorities, our conventions, decisions already made and why, what
     failed, the people and systems involved. This makes it a veteran of how
     WE work. The agent flags every place our practice departs from what its
     research found.
  c. Memorize. The agent reflects, commits both to memory, and posts an
     inventory of what it knows plus every gap or conflict it flagged.

### M6: Address every to-do  <!-- advances: AC-3.2 -->

- [ ] **M6** — assistant
- Update each to-do from M4 in place (skip any no longer open; never publish
  a duplicate):
  - One agent can do it:
    `clawmeets todo update <id> --recipient <agent> --draft-prompt "<first assignment>"`
  - It needs several agents: `--recipient` is my assistant, and the draft
    opens "Create a project you coordinate with @a, @b ...", says what each
    agent owns, and ends with the first deliverable.
  - It stays with my assistant: recipient is my assistant.
- A first assignment states the goal, current status, the very next
  deliverable and what makes it good enough. Keep any existing draft below it.
- Hand off each to-do as soon as its agent finishes onboarding.
- Wrap up with a completion report: the team (name, job, what it knows, what
  it flagged), a USER.md summary, and my to-dos in priority order (title,
  recipient, first deliverable). Don't archive the starter to-do; it follows
  this project.
"""
