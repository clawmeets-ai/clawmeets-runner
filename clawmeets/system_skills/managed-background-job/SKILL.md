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

```bash
clawmeets job start \
  --project "<project id from the current request context>" \
  --room "<chatroom name from the current request context>" \
  --deadline-seconds 21600 \
  --heartbeat-timeout 300 \
  --step "initial step"
```

The server binds the job to the pending message currently waiting for this
agent in that project and room. Pass `--pending-message ID` only when you want
an additional stale-request guard.

Persist the returned job ID and `event_sequence` beside the work. Every update
must use the latest sequence returned by the previous command:

```bash
clawmeets job heartbeat JOB_ID --sequence 0
clawmeets job progress JOB_ID --sequence 1 --summary "420 of 900 records validated" --checkpoint path/to/checkpoint.json
clawmeets job complete JOB_ID --sequence 2 --summary "Validation complete"
```

Use `fail` with `--reason` when the work cannot continue, and `cancel` when the
request is intentionally stopped. A heartbeat proves only that the process is
alive; use `progress` for semantic advancement. The absolute deadline never
moves, even when heartbeats are healthy.

The first release supervises lifecycle state; it does not automatically create
a detached process or schedule another model invocation. Start and supervise
the process with the execution mechanism available to the runner, then bind it
to this job immediately.
