---
name: control-computer
description: >
  Answer "is my computer up?", "why is nothing replying?", "what's running on
  my laptop?" and "start the budget analyst on my desktop" — the questions
  about a MACHINE rather than a process on this one. Use when the user asks
  about a computer by name, asks why agents look offline, asks to start or stop
  an agent that lives on a DIFFERENT machine, or asks what ClawMeets is
  allowed to do on their computer. Drives the five allowed actions (start,
  stop, restart, report status, update the connection software) on any
  connected computer. Never deletes anything. Assistant-only.
---

# Control Computer (the machine, not the process)

`control-agent` reaches agents on **this** machine, by process id. This skill
reaches agents on **any** of the user's connected computers, because each of
those machines keeps a connection open to ClawMeets that survives every agent on
it dying.

That last clause is the reason this skill exists. When a user says "my agents
stopped replying", there used to be no way to tell these apart:

- the computer is off,
- the computer is on and nothing is running on it,
- the computer is on, agents are running, and something else is wrong.

They all looked like silence. Someone sat in the second one for two and a half
days believing the product was down. **Naming which of the three it is, before
offering any remedy, is the job.**

## Decision flow

### 0. Is this a delete request? Then stop here
Deleting an agent is the user's own action in the browser. It is not on the
allowed list, the machine refuses it, and you must not try. Hand off to
`control-agent`, which has the full answer, and say plainly that it is theirs
to do.

### 1. Is it about a machine, or a process?
- "Is `<agent>` running?" / "start `<agent>`" with **no machine mentioned**, and
  the agent lives here → that is `control-agent`. Use it.
- Anything naming a computer, asking why a whole group is quiet, or targeting an
  agent whose directory is **not** on this machine → you are in the right place.

### 2. Look before you act
```bash
clawmeets computer status
```
That reports THIS machine only (is its connection up, and what a fresh scan of
its agents directory says). For the user's other computers, read the computers
list from the API as yourself:
```bash
curl -s -H "Authorization: Bearer $CLAWMEETS_AGENT_TOKEN" \
     -H "X-Agent-ID: $CLAWMEETS_AGENT_ID" \
     "$CLAWMEETS_SERVER_URL/me/computers"
```
Each row carries `id`, `name`, `status`, `running_count`, `agent_count`,
`last_seen_at`, and an `agents` array with one entry per agent and its `state`.

### 3. Name the state before offering a remedy
Read `status` and say which of these it is, in the user's words — never
"runner", never "daemon", never "host":

| `status` | What to say | The remedy |
|---|---|---|
| `online`, `running_count > 0` | "`<name>` is on and N of M agents are running." | none needed |
| `online`, `running_count == 0` | "`<name>` is on, but none of your agents are running — that's why they look offline." | start them (step 4) |
| `not_answering` | "`<name>` looks awake but ClawMeets has lost contact with it, so I can't reach it from here." | they run `clawmeets computer start` **on that machine** |
| `off` | "`<name>` is off, asleep, or not on the internet. Nothing has run there since `<last_seen_at>`." | open the computer; queued messages are still queued |
| `revoked` | "`<name>` is disconnected from ClawMeets, so I can't see or control it." | reconnect with a fresh code from the web app |

Two facts worth volunteering, because they are the ones that stop a user
resending work or waiting on us:

- Messages sent to an agent whose computer was down are **still queued** and
  arrive when it reconnects. They do not need to resend.
- A down computer is **their machine, not a ClawMeets outage**.

An agent whose `state` is `crashed` exited on its own since it was last
started — say "it stopped on its own", not "it's stopped". The difference is
what tells a user something went wrong rather than that they turned it off.

### 4. Act — the five allowed actions, and nothing else
```bash
curl -s -X POST \
  -H "Authorization: Bearer $CLAWMEETS_AGENT_TOKEN" \
  -H "X-Agent-ID: $CLAWMEETS_AGENT_ID" \
  -H "Content-Type: application/json" \
  -d '{"action":"start","agent":"<short-name>"}' \
  "$CLAWMEETS_SERVER_URL/me/computers/<computer-id>/commands"
```

`action` is one of exactly these:

| Action | Needs an agent | What it does |
|---|---|---|
| `start` | yes | Start one of the user's agents on that computer |
| `stop` | yes | Stop one of them |
| `restart` | yes | Stop then start, as two steps |
| `status` | no | Make the computer re-report what is running |
| `update` | no | Update that computer's own connection software |

Anything else is refused — by the server before it sends, and again by the
machine before it runs. If the user asks for something outside this list, say so
plainly and name what you *can* do; do not look for another route to it.

The command is accepted and carried out asynchronously; the machine reports the
outcome on its next check-in. So **verify, do not assume**: wait a couple of
seconds, re-read `/me/computers`, and report what the agent's `state` actually
says. A `409` means the computer is not reachable right now — do not retry in a
loop, fall back to step 5.

### 5. When you cannot reach the machine
A `not_answering` or `off` computer cannot be driven from here, and pretending
otherwise is worse than saying so. Give the exact command for the user to run
**on that machine**, whole and copyable:

```bash
clawmeets computer start          # reconnect the computer to ClawMeets
clawmeets start                   # start every agent on it
clawmeets start --agent <agent>   # start just one
clawmeets computer logs --tail 50 # why the connection is failing
```

If that computer is shared by more than one ClawMeets account, every one of
those commands takes `--user <account>`; without it they act for whichever
account is logged in on the machine. Only add it when you know the user has
more than one account there — it is noise otherwise.

## Connecting a new computer

You cannot connect one, and that is deliberate: pairing is the moment the user
grants ClawMeets permission to act on a machine, so the code is minted only for
a human in their browser. Tell them the path and what it grants:

> In ClawMeets, open **Computers** in the left rail and press **+**. You'll get
> a code that's good for 15 minutes, and one command to run on that computer.
> Connecting lets me start, stop and restart your own agents there, see which
> are running, and keep its connection software current — nothing else. You can
> disconnect it at any time from the same page, and its key stops working
> immediately.

## What ClawMeets may never do on a computer

Say these out loud if the user asks, or if they seem uneasy about the whole
idea. They are enforced in two places, not one — the server will not send an
unlisted command, and the machine will not run one:

- Run any other command
- Open, read, copy or send their files
- Install or change anything else
- Delete an agent — only they can, in the browser
- Reach any other computer or account

## Vocabulary

The user reads about **their computer**, or the machine's own name ("MacBook
Pro"). The words "runner", "daemon", "host" and "process" do not appear in
anything you write to them — not in a status line, not in an apology, not in a
command explanation. The CLI verb is `clawmeets computer …` for the same
reason.
