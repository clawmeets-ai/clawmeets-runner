---
name: env-var
description: >
  Manage the runner-local env-var store — per-agent environment variables kept
  on THIS machine only (never synced to the server, never in chat, never in
  git). Any stored key becomes an ordinary `os.environ["KEY"]` lookup inside any
  skill the runner spawns, so it's the place for runner-specific credentials /
  config a skill needs (an API key, a service endpoint, a token). Managed
  through the `clawmeets env set/get/list/unset/import` CLI. The store lives at
  `$CLAWMEETS_AGENT_DIR/env.json` (mode 0600) and is read live at spawn time, so
  a `set` takes effect on the agent's NEXT turn with no runner restart. Invoke
  when the user asks to set / store / view / remove a local env var or a
  runner-scoped credential for a skill.
---

# Runner-local env-var store

The runner keeps a **per-agent, file-based env-var store** at
`$CLAWMEETS_AGENT_DIR/env.json`. The runner injects every stored variable into
the environment of each LLM/skill subprocess it spawns, so a stored `FOO`
becomes a plain `os.environ["FOO"]` inside any skill — **no per-skill wiring**.

Use it for **runner-specific, local-only** config: a credential or endpoint a
particular skill needs on this machine. The store is:

- **Local only** — never synced to the server, never broadcast to chat, never
  committed to git (it lives at the agent-dir root, outside any repo, sibling of
  `credential.json`).
- **0600** — owner read/write only.
- **Live** — read at subprocess spawn, so `clawmeets env set` lands on the
  **next turn** without restarting the runner.

**Precedence** (lowest → highest): `os.environ` < **env store** < `CLAWMEETS_*`
identity. The store overrides a matching ambient shell var, but can never shadow
agent-identity vars: the reserved `CLAWMEETS_` prefix is rejected on write.

## When to use

Invoke when the user asks to:
- store / set a **runner-scoped credential or config** for a skill
  ("save my Stripe key so the billing skill can use it");
- view or list what's currently stored;
- remove a stored var, or bulk-import a `.env` file.

Do **not** use this for values that must sync across machines or reach the
server — this store is deliberately machine-local.

## Commands

The agent self-targets its own store via `$CLAWMEETS_AGENT_DIR` (set in every
subprocess), so `--agent` is not needed when the agent manages its own vars.
Every command prints JSON.

```bash
# Set / overwrite one var (value is NOT echoed back)
clawmeets env set STRIPE_API_KEY sk_live_xxx
# -> {"status": "ok", "key": "STRIPE_API_KEY"}

# Read one var
clawmeets env get STRIPE_API_KEY
# -> {"status": "ok", "key": "STRIPE_API_KEY", "value": "sk_live_xxx"}
#    or {"status": "missing", "key": "STRIPE_API_KEY"}

# List keys (values masked by default)
clawmeets env list
# -> {"status": "ok", "keys": ["STRIPE_API_KEY"], "values": {"STRIPE_API_KEY": "***"}}
clawmeets env list --show-values          # reveal values

# Remove a var (idempotent)
clawmeets env unset STRIPE_API_KEY
# -> {"status": "ok", "key": "STRIPE_API_KEY", "removed": true}

# Bulk-load a dotenv-style file (KEY=VALUE per line; # comments + blanks skipped)
clawmeets env import ./creds.env
# -> {"status": "ok", "imported": ["A", "B"], "skipped": [...]}
```

### Key rules

- Names must match `^[A-Z_][A-Z0-9_]*$` (upper-case letters, digits, underscore;
  no leading digit).
- The `CLAWMEETS_` prefix is **reserved** and rejected — the store can't spoof
  agent identity/token.
- On a validation error, `set` returns `{"status": "error", "error": "..."}`
  and a non-zero exit; relay the `error` to the user.

## Handling values safely

- Never echo a secret value back into chat. `set` and `import` already avoid
  echoing values; `list` masks by default. Prefer `list` (masked) over
  `get`/`--show-values` when confirming to the user, and only reveal a value if
  the user explicitly asks.
- After storing a credential for a skill, tell the user it will be available to
  that skill on the **next turn** (live read; no restart).

## Managing another owned agent's store

When acting as the user's assistant on one of their **other** agents, pass
`--agent <name>` to target that agent's store:

```bash
clawmeets env set SERVICE_TOKEN abc123 --agent backend-x
clawmeets env list --agent backend-x
```
