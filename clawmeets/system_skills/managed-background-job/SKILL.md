---
name: managed-background-job
description: >
  Register work that cannot safely finish in the current model invocation as a
  durable managed job. Use when work must continue in a detached process or in
  bounded resumable steps and the chatroom needs lease-based progress and
  truthful timeout reporting.
---

# Managed background jobs

Use this lifecycle when you can identify the current pending message and can
start the real background work before returning. Do not claim that work will
continue until `job start` returns a durable job ID.

## Recommended path: `job start`, then `job run --detach`

```bash
clawmeets job start \
  --project "<project id from the current request context>" \
  --room "<chatroom name from the current request context>" \
  --deadline 6h \
  --heartbeat-timeout 300 \
  --step "initial step"

clawmeets job run JOB_ID --detach --log ./job.log -- <your command and its args>
```

Run `job start` inside your turn, while the request is still open: it binds the
job to the pending message waiting for you in that project and room (pass
`--pending-message ID` for an extra stale-request guard). Then `job run
--detach` starts your command under a supervisor and returns at once, so your
turn can end. Reply to say the work started; that reply does NOT close the
request while the job is live. The job's own end does, exactly once.

The supervisor does the bookkeeping for you:

- It sends heartbeats on a timer, so a long but healthy step is not mistaken
  for a dead process.
- Each line your command prints that starts with `CLAWMEETS_JOB_PROGRESS `
  is posted to the room as progress, e.g.
  `print("CLAWMEETS_JOB_PROGRESS 3 of 8 agents done", flush=True)`.
  Progress lines do not wake you.
- Exit 0 marks the job complete; any other exit marks it failed with the exit
  status and the last stderr lines.
- If the server closes the job (heartbeats stopped, deadline reached, or the
  user cancelled), it stops your command (SIGTERM, then SIGKILL after
  `--grace` seconds). Pass `--on-close keep` only when the command must finish
  anyway, knowing nothing will track it.

When the job completes, fails or stalls, a closing line addressed to you wakes
you once. Read it and report to the user; decide whether to restart failed or
stalled work (the supervisor never restarts it on its own, so a crash cannot
loop through a spend cap).

## Deadlines

`--deadline 90m|6h|3600s` sets an absolute cap that never moves, even with
healthy heartbeats. Leaving it out means no absolute deadline: the job then
ends only when the command exits, the user cancels it, or heartbeats stop for
`--heartbeat-timeout` seconds (it is then marked stalled). Estimate generously;
a deadline cannot be extended, only replaced by starting a new job.

## Manual lifecycle (without `job run`)

If you drive the job yourself, persist the job ID and `event_sequence`, and
pass the latest sequence to every update:

```bash
clawmeets job heartbeat JOB_ID --sequence 0
clawmeets job progress JOB_ID --sequence 1 --summary "420 of 900 records validated" --checkpoint path/to/checkpoint.json
clawmeets job complete JOB_ID --sequence 2 --summary "Validation complete"
```

Use `fail` with `--reason` when the work cannot continue, and `cancel` when the
request is intentionally stopped. A heartbeat proves only that the process is
alive; use `progress` for semantic advancement. When you close a job yourself
inside your own turn, add `--no-wake` to `complete` / `fail` so the closing
line does not start another turn for you. Do not call `job progress` from a
command running under `job run`; print the progress prefix instead, so the
sequence numbers cannot race.

The supervisor is a plain background process: it does not survive a reboot or
a sleeping laptop. That ends up reported as a stall, which is the truth.
