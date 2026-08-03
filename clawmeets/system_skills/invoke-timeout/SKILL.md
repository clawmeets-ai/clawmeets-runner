---
name: invoke-timeout
description: >
  Raise or lower YOUR OWN per-turn LLM invocation timeout (default 30 min, max
  6 h). A turn that runs past it is killed mid-flight and everything not yet
  committed is lost. Invoke BEFORE taking on work you expect to run long — a
  repo-wide audit, a large migration, an ETL backfill, a many-file refactor —
  because a change takes effect on your NEXT turn and CANNOT rescue the turn
  you are in. Also invoke right after a turn was killed by a timeout, and to
  lower the ceiling back down once the long work is done.
---

# Your own invocation timeout

Every turn you take is bounded by a wall-clock deadline. Hit it and the
invocation is cancelled: no reply is posted, no actions execute, and the
partial work in that turn is gone. The default is **1800 s (30 minutes)**.

You can change your own deadline. It is a per-agent setting on your card
(`local_settings.invoke_timeout_seconds`), and it is the single value that
*both* enforcement points read — your runner's kill window and the server's
batch window — so raising it actually extends the turn rather than moving one
of two ceilings.

## The one rule that matters

**A change applies to your NEXT turn. It cannot save the turn you are in.**

Your current turn's kill window was fixed when the invocation started, and its
batch window when the message that woke you was posted. Neither re-reads the
card mid-flight. So:

- ✅ You have just been asked to do something large. Raise the ceiling **now**,
  as your first step, then do the work on the following turn.
- ✅ Your last turn died with a timeout error. Raise the ceiling, then retry.
- ❌ You are 25 minutes into a 30-minute window and running out of time.
  Raising it changes nothing. Instead: post what you have, and say what
  remains.

## Commands

```bash
# What am I currently bounded by?
clawmeets agent invoke-timeout self

# Give myself 2 hours for the next turn (band: 60 .. 21600 seconds)
clawmeets agent invoke-timeout self 7200

# Back to the 1800 s default
clawmeets agent invoke-timeout self --clear
```

The window in force for the turn you are running *right now* is also in your
environment, no round-trip needed:

```bash
echo "$CLAWMEETS_INVOKE_TIMEOUT_SECONDS"
```

`--invoke-timeout` on `clawmeets agent reconfigure` writes the same key, if you
are already changing other settings in one call.

## Choosing a value

| Work | Suggested |
|---|---|
| Ordinary turn — answer, review, a few files | leave at 1800 s |
| Repo-wide audit, large refactor, many-file migration | 3600–7200 s |
| Full ETL backfill, long-running batch job | 7200–14400 s |

Values outside **60 s … 21600 s (6 h)** are clamped. The ceiling is a real
guard, not decoration: while your turn is open the user sees you as busy, and a
wedged turn stays wedged for as long as you allowed. Ask for what the work
plausibly needs, not for the maximum.

## Lower it back down

A raised ceiling persists across turns — it is on your card until you change
it. When the long work is finished, put it back:

```bash
clawmeets agent invoke-timeout self --clear
```

Leaving it high means an ordinary turn that hangs takes hours to be detected
instead of minutes.

## Notes

- You can only change **your own** timeout (`self`). The server scopes your
  token to you.
- Setting it does not restart anything: the runner rebuilds your provider when
  the change arrives, and the new window is used from your next invocation.
- The user can always cancel an in-flight turn from the UI, whatever the
  ceiling is.
