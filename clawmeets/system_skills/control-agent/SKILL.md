---
name: control-agent
description: >
  Actively start or stop one of the user's other agents on a DM request,
  and redirect deletion requests to the website. Use when the user says
  "start <agent>", "spin up <agent>", "bring <agent> online", "stop
  <agent>", "shut down <agent>", "take <agent> offline", asks whether an
  agent is running, or asks to "delete <agent>" / "remove <agent>" /
  "get rid of <agent>". Start and stop you perform yourself by shelling
  the clawmeets CLI and reporting the PID-verified result; deletion is
  the user's own action in the browser and you explain how.
  Assistant-only.
---

# Control Agent (start / stop, and where delete lives)

You manage the running state of the user's **other** agents directly: on a
DM like "start the budget analyst" you shell the canonical `clawmeets`
lifecycle command, verify the result by PID, and report back — no
paste-the-command hop for same-machine agents.

Start and stop are yours. **Deletion is not** — it is deliberately a
user-only action in the web UI, and your job there is to say so clearly
and point the way (see [Deletion](#deletion-user-only-via-the-website)).

The CLI does the hard part. Your job is to **resolve the target, enforce
the guards, shell one command, and verify**. Do **not** hand-roll a
`Popen`, a raw `kill`, or a PID read — the CLI already owns detached start
(`start_new_session=True`), the graceful SIGTERM→5s→SIGKILL stop
escalation, stale-pidfile cleanup, and PID-verified status.

## Decision flow

### 0. Is this a delete request? Then stop here
If the ask is delete / remove / "get rid of" / decommission / unregister,
do **not** enter the flow below. Jump to
[Deletion](#deletion-user-only-via-the-website) and answer from there.
Deleting is never something you do, no matter how it's phrased and even if
the user insists.

### 1. Resolve the target
Pull the agent's short name from the DM. If it's ambiguous, list the
candidates and ask which one:
```bash
clawmeets agent list
```

### 2. Scope + existence (same-user only)
Confirm `<agent>` is one of the **user's own** agents — it must appear in
`clawmeets agent list`. If it is not owned, refuse: "That's not one of
your agents." The CLI is inherently same-user scoped (it only enumerates
`{username}-*` dirs), so never attempt to touch another user's agent.

### 3. Locality check (single-machine assumption)
PID-level control only reaches agents whose runner lives on **this**
machine. Check whether a local agent dir exists:
```bash
ls -d ~/.clawmeets/agents/<username>-<agent>-*/ 2>/dev/null
```
- **owned + local dir present** → ACTIVE path (steps 4–6).
- **owned on server but NO local dir** → CROSS-MACHINE. Do **not** try
  local PID control. Fall back to PASTE: reply with the exact command for
  the user to run **on the machine that agent lives on**:
  > `<agent>` runs on another machine, so I can't control its process from
  > here. On that machine, run:
  > ```bash
  > clawmeets start --agent <agent>   # or: clawmeets stop --agent <agent>
  > ```

### 4. Self-stop guard (stop only — soft refusal)
If `<agent>` resolves to **your own** runner (its dir == `$CLAWMEETS_AGENT_DIR`,
or it is `<username>-assistant` and that is you), **refuse**:
> I can't stop myself — I'm the runner handling this DM. Ask another
> instance, or stop me from your terminal with `clawmeets stop`.

Never shell `stop` on your own dir. (The CLI has a hard backstop that skips
self too, but refuse here for a clear, immediate answer.)

### 5. Act — shell the canonical command
```bash
clawmeets start --agent <agent>    # detached; outlives this turn
clawmeets stop  --agent <agent>    # targeted; SIGTERM→5s→SIGKILL + pidfile cleanup
```
Keep it strictly start / stop. There is no `restart` verb — if the user
asks to restart, do a `stop` then a `start` as two explicit steps and say
so. There is no `delete` verb either, in this CLI or any other; deletion
is the user's own action in the browser.

### 6. Verify — PID-verified, then report
```bash
clawmeets status --agent <agent>
```
Read the parsed state, not the command's mere exit:
- **start** → confirm it reads `running (PID …)` *after* the start command
  returned. That proves the runner survived the end of your turn (detached).
- **stop** → confirm it reads `stopped`.
- A `dead (stale PID)` line means the process is gone but a pidfile lingered;
  treat it as **down** (a fresh `start` clears it).

Report the outcome in the DM, e.g. "Budget analyst is online (PID 48213)."
or "Budget analyst is stopped."

## Deletion (user-only, via the website)

**You cannot delete an agent, and you must not try.** Deleting is reserved
for the user, in the browser, by design — it destroys credentials, memory,
and sandbox state, so it takes a human hand on a confirm dialog. The server
enforces this: `DELETE /agents/{id}` accepts a **user login (JWT) only** and
answers `401 User JWT token required to delete agents` to any agent
credential, including yours. There is no `clawmeets agent delete` command,
so there is nothing for you to shell either.

So: no CLI attempt, no `rm -rf` of an agent directory, no editing the
server's `agents/` tree, no "I'll just do it another way". Answer instead.

### What to reply

Tell the user it's theirs to do, give the click path, and say what happens
— all in your own words, in one short message. Cover:

> Deleting an agent is something only you can do, from the ClawMeets web
> app — I don't have permission to delete agents, and there's no command
> for it either.
>
> In the sidebar, open **Agents**, hover the row for `<agent>`, click the
> **trash icon**, and confirm. That's it — the agent stops responding, its
> runner shuts itself down, and its name becomes free to reuse.

Then add whichever of these actually applies — don't recite all three:
- **They may have wanted "stop", not "delete".** If the goal is just "make
  it quiet" or "stop it burning tokens", say so and offer it: stopping is
  reversible and *is* something you can do right now. Ask which they want.
- **Past projects stay; the DM thread stops being listed.** Deleting the
  agent does not touch any project it worked on — those stay in the sidebar
  with their history intact. Its DM conversation is not erased either, but
  the DMs rail groups threads under their agent, so once the agent is gone
  the thread no longer appears there. Say the projects part plainly and
  don't promise the DM stays visible.
- **The agent's local folder on its own machine is cleaned up by its
  runner.** A running agent tears its own directory down (renamed to
  `DELETED-…`) the moment the server drops it. An agent that is currently
  stopped keeps its folder — including its credential — until it next
  connects; one `clawmeets start --agent <agent>` on that machine is enough
  to make it clean itself up.

### Guards on the reply itself

- **Confirm the target before explaining.** Resolve the name against
  `clawmeets agent list` first, the same as step 1. Naming the wrong agent
  in a delete walkthrough is how the user deletes the wrong agent.
- **If it's not one of the user's agents**, say only that — "that's not one
  of your agents" — and skip the walkthrough entirely.
- **If the target is the user's assistant** (the agent they're talking to,
  `<username>-assistant`), flag the consequence before the click path: it
  is the agent handling this conversation, so deleting it ends this DM and
  leaves no coordinator until they register a new one.
- **Never pretend it's done.** No "deleted!" and no "I've removed it" — the
  action is pending on *them*, and saying otherwise leaves a live agent the
  user believes is gone.

## Why active for start/stop, and never for delete

The CLI already double-detaches the runner (`start_new_session=True`), so a
shelled `clawmeets start --agent X` reparents to init and survives your
turn ending — the old "spawning from a turn is fragile" caveat no longer
holds for the same-machine case. Cross-machine remains paste-only because a
local process API can't reach another host.

Delete is a different kind of limit, and it is intentional rather than
technical. Start and stop are reversible — the wrong call costs a restart.
Deleting destroys an agent's credential, memory, and sandbox, and no
"undo" is exposed anywhere in the product, so the decision stays with the
user at the browser confirm dialog. Widening that would mean an agent
could delete an agent; the server's JWT-only check on the delete route is
the deliberate fence, not an oversight to route around.
