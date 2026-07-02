---
name: control-agent
description: >
  Actively start or stop one of the user's other agents on a DM request.
  Use when the user says "start <agent>", "spin up <agent>", "bring <agent>
  online", "stop <agent>", "shut down <agent>", "take <agent> offline", or
  asks whether an agent is running. You perform the action yourself by
  shelling the clawmeets CLI and report the PID-verified result.
  Assistant-only.
---

# Control Agent (active start / stop)

You manage the lifecycle of the user's **other** agents directly: on a DM
like "start the budget analyst" you shell the canonical `clawmeets`
lifecycle command, verify the result by PID, and report back — no
paste-the-command hop for same-machine agents.

The CLI does the hard part. Your job is to **resolve the target, enforce
the guards, shell one command, and verify**. Do **not** hand-roll a
`Popen`, a raw `kill`, or a PID read — the CLI already owns detached start
(`start_new_session=True`), the graceful SIGTERM→5s→SIGKILL stop
escalation, stale-pidfile cleanup, and PID-verified status.

## Decision flow

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
so.

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

## Why active, not paste

The CLI already double-detaches the runner (`start_new_session=True`), so a
shelled `clawmeets start --agent X` reparents to init and survives your
turn ending — the old "spawning from a turn is fragile" caveat no longer
holds for the same-machine case. Cross-machine remains paste-only because a
local process API can't reach another host.
