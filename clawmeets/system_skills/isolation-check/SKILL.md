---
name: isolation-check
description: >
  Re-run an agent's isolation check inside its running process, so it can take
  conversations from other people (hub, Front Desk, channels) again. Invoke
  when the owner asks ("re-run the isolation check", "why isn't my agent
  answering hub messages", "I signed Claude in again"), after a turn failed
  with a sandbox or login error, or when the owner forwards the "can't take
  conversations from other people right now" email. Workers and coordinators
  check themselves (`self`); the owner's assistant can check any of the
  owner's agents by name.
---

# The isolation check

A conversation with someone outside the owner's account runs in a sandbox, and
an agent takes such conversations only while its runner has proved the sandbox
holds. The proof is the **isolation check**: at start, and after a model
provider, model or login change, the runner runs one real sandboxed model turn
plus a set of probes. If it fails, those conversations are **held**, not lost:
the sender is told it is waiting, the owner gets an email, and the work is
delivered as soon as a check passes (or closed after 72 hours on the hub,
7 days otherwise).

Re-running the check is cheap to ask for but makes one real, paid model call,
so do not loop it.

## When to run it

- The owner asks you to.
- The owner says they signed the model CLI in again (`claude login`, …).
- A previous check failed with a **temporary** problem (rate limit, 5xx,
  timeout) a few minutes ago.
- The owner forwards the email saying conversations are waiting.

Do **not** run it from a conversation with someone outside the owner's
account — the command refuses there anyway. Act only in your owner's own DM.

## Command

```bash
clawmeets agent isolation-check self          # check yourself (worker / coordinator)
clawmeets agent isolation-check <agent>       # assistant: one of the owner's agents
clawmeets agent isolation-check <agent> --status    # last result, start nothing
clawmeets agent isolation-check <agent> --no-wait   # start it and return
```

It waits for the result (up to 15 minutes; `--timeout` changes that). The
agent keeps serving while the check runs; only a failure stops it.

| Exit | Meaning | What to tell the owner |
|---|---|---|
| 0 | Passed | It takes conversations from other people again; waiting ones are being delivered. |
| 1 | Failed | Quote the printed reason and next step (below). |
| 3 | Still running at `--timeout` | Check later with `--status`. |
| 4 | Agent not running, or its runner too old | Start it (`clawmeets start --agent <name>`) or update and restart it; a start runs the check. |

A failure is labelled with what to do next:

- **temporary** — the provider or network had a bad minute. Run it again in a
  few minutes.
- **needs_action** — something must be fixed first. A login error means the
  model CLI on the agent's computer must be signed in again by the owner;
  "no working OS sandbox" means the computer cannot sandbox (macOS
  `sandbox-exec`, Linux `bwrap`). Then run it again.
- **unsupported** — this model provider (the in-process `-api` providers,
  smart routing) cannot run isolated conversations yet. Re-running will not
  change that; say so plainly.
- **crashed** — the check itself raised; the traceback is in the agent's
  `stderr.log`.

Report the outcome in one or two sentences: passed or not, why, and the one
next step. Never paste the full failure output unless the owner asks.
