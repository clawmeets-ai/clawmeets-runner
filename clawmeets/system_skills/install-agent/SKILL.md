---
name: install-agent
description: >
  Install ONE ready-made specialist agent from the ClawMeets agent pool
  (contract reviewer, financial model builder, ...). Use when the user asks
  for "an agent that can <X>", "add a <specialist>", "what agents can I
  install", "install <name>", or when their request needs expertise none of
  their own agents has and the pool lists a specialist for it. Lists the pool,
  proposes the match, and on the user's yes runs `clawmeets agent install
  <name>` and starts it. For an agent the user wants to define themselves, use
  register-agent (or onboard-agent) instead.
---

# Install a specialist agent from the pool

The pool is a public set of specialist agents, each published only after it
beat a strong generalist on held-out tasks in its field. Installing one
registers it under the user's account and copies its role, field knowledge,
reference material and private skills into its home on this computer. Every
file is checked against the SHA-256 the pool lists.

Agents are installed **one at a time**. There is no "install a team".

## 1. See what the pool has

```bash
clawmeets agent install --list --json
```

Each entry has a `name`, a one-line `blurb`, a longer `description` and
`sample_requests`. The default pool is the public `clawmeets-agents` repo at
the release this runner pins. Pass `--from <url-or-path>` only if the user
names another pool.

## 2. Propose, then wait for a yes

Match the user's need against the blurbs and descriptions. Reply with the
one agent you recommend: its name, what it does in one sentence, and one of
its sample requests so the user can see the kind of work it takes. If two
agents fit, say which you would pick and why. If nothing fits, say so and
offer `register-agent` instead. Do not invent pool agents.

Installing creates an agent on the user's account and files on their
computer, so **ask before you run it**, unless the user already named the
agent and asked you to install it.

## 3. Install and start it

```bash
clawmeets agent install "<name>" --username "<owner-username>"
clawmeets start --user "<owner-username>" --agent "<name>"
```

Useful options: `--team "<label>"` adds a TEAMS-sidebar label on top of the
pool's own; `--llm-provider <p>` picks the model (default: a copy of the
user's default model); `--auto-personalize` asks the new agent to tailor
itself to the user on its first turn.

Exit codes: `0` installed (running it again updates the agent and keeps any
file changed locally since the last install); `2` no such agent in the pool
(the error suggests close names); `1` the install was refused or a file
failed its checksum. Relay the error verbatim and do not retry with other
flags. A refusal saying the user already has an agent with that name means
they made one themselves. Ask them to rename or delete it first.

## 4. Tell the user

Say the agent is installed and running, and give one of its sample requests
as a first thing to try in its DM. If the user wants it to learn about them
first, offer `onboard-agent` (a mentor brief and deep research, distilled
into its memory).
