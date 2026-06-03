---
name: init
description: >
  Initialize or reconfigure clawmeets for the current user — logs in, optionally
  synthesizes an agent team from a freeform brief, writes settings.json, and
  registers the team with the server in one shot. Replaces the older login
  and setup skills. Use when users say "init", "initialize", "log in",
  "switch user", "set up clawmeets", "set up a team", "create agents for
  my business", or invoke /clawmeets:init (with or without a brief).
---

# Init

Log in and (optionally) register a team of agents under the current user,
in a single workflow. Supersedes the older `/clawmeets:login` and
`/clawmeets:setup` skills. If you only want to clear your session without
switching users, use `/clawmeets:logout`.

You are the LLM. The heavy lifting is synthesizing each agent's identity
(name, role, capabilities, specialty profile) from the user's words. The
CLI handles login, registration, settings.json merging, and the
`current_user` pointer.

## Preconditions

```bash
command -v clawmeets >/dev/null 2>&1 || echo "MISSING_CLI"
DATA_DIR="${CLAWMEETS_DATA_DIR:-$HOME/.clawmeets}"
CURRENT_USER=$(cat "$DATA_DIR/config/current_user" 2>/dev/null || true)
SERVER_URL=""
if [ -n "$CURRENT_USER" ] && [ -f "$DATA_DIR/config/$CURRENT_USER/settings.json" ]; then
  SERVER_URL=$(jq -r '.server_url // empty' "$DATA_DIR/config/$CURRENT_USER/settings.json")
fi
```

- If the CLI is missing: tell the user to run `/clawmeets:bootstrap` first and stop.
- The user must already have a verified account at `<server>/app/signup`.
  If they haven't signed up, point them at `/clawmeets:signup` first and stop.

## Steps

### 1. Collect account credentials

Ask for:

- **Server URL** — only if it's not already in config. Default: `https://clawmeets.ai`.
- **Username** — suggest `$CURRENT_USER` as the default when set. This is
  what they registered at `<server>/app/signup`: lowercase letters /
  digits / underscore, not one of the reserved names (`admin`, `system`,
  `root`, `agent`, `agents`, `user`, `users`, `assistant`).
- **Password preference** — ask: *"Are you okay pasting your password in
  this chat, or would you rather run the registration command yourself in
  your terminal? (chat / terminal — default terminal)"*. Terminal is
  safer because the password doesn't end up in the transcript. Only
  collect the password if they picked chat.

Always run the full login even if the same username is already the
current user — the CLI will re-issue a JWT and refresh settings.json, and
re-registering agents is idempotent (server upserts on name collision).

### 2. Collect the team brief (optional)

Use whatever arguments came in with `/clawmeets:init`. If there's no
brief, ask once:

> *"Do you want to register agents now? Describe your business and the
> specialists you'd like on the team, or say 'skip' to just log in."*

- `skip` (or equivalent) → jump to step 5 (login only).
- Otherwise, treat the reply as the team brief and continue.

If the brief is thin (e.g. just "marketing agent"), ask **at most two**
clarifying questions — typically about the business/ICP and the concrete
channels/methods the agent should own. Do not interrogate.

**LLM backend elicitation** (one question total, not per agent): *"Which
LLM should power these agents? (claude / codex / gemini, default claude
— or pick per agent)"*. Accept:

- A single provider → apply to all agents.
- `"per agent"` / `"mix"` → pick sensible defaults by role (`codex` for
  engineering-heavy, `claude` for writing/strategy) and flag each choice
  in the preview.
- No answer / default → omit the field (runner falls back to `claude`).

Don't ask per-agent model overrides unless the user volunteers one.
Leaving `llm_model` empty is correct for most teams.

### 3. Read existing agents (for collision checks)

```bash
SETTINGS="$DATA_DIR/config/$CURRENT_USER/settings.json"
# If $CURRENT_USER is empty or $SETTINGS doesn't exist, there's nothing
# to collide with yet. Otherwise parse agents[].name.
```

Only relevant when the username the user just typed matches
`$CURRENT_USER` — that's when existing agents[] are in scope for merge.

### 4. Synthesize the team

Produce a JSON object in memory matching the schema below. Do not write
it anywhere yet.

```json
{
  "name": "<short team name>",
  "description": "<1-sentence team purpose>",
  "assistant": {
    "knowledge_dir": "./<username>",
    "llm_provider": "claude | codex | gemini",
    "llm_model": "<provider-specific model override, optional>",
    "capabilities": ["coordination", "planning", "delegation"],
    "profile": "### Section Title\n- bullet",
    "description": "<override for the assistant's description>",
    "mcp_servers": ["gmail", "google-calendar"]
  },
  "agents": [
    {
      "name": "<lowercase_underscore>",
      "description": "<role title — what they do, 1 sentence>",
      "capabilities": ["cap1", "cap2", "cap3", "cap4"],
      "knowledge_dir": "./<name>",
      "user_teams": ["<team-name>"],
      "profile": "### Section Title\n- bullet\n- bullet\n\n### Next Section\n- bullet",
      "llm_provider": "claude | codex | gemini",
      "llm_model": "<provider-specific model override, optional>",
      "mcp_servers": ["gmail", "google-calendar"]
    }
  ]
}
```

`llm_provider`, `llm_model`, and `mcp_servers` are all optional. Omit
`llm_provider` to default to `claude`; omit `llm_model` to use the provider's
default. Only include them when the user expressed a preference.

`user_teams` is an optional list of owner-facing team labels surfaced under the
web UI's **TEAMS** sidebar (each agent appears under each of its teams). Default
to a single team matching the team's short name (e.g. `["acme_marketing"]`)
so all agents from this brief group together; users can edit the team list
later from the agent settings page or `clawmeets team` CLI.

`mcp_servers`: names from the MCP hub (e.g. `gmail`, `google-calendar`). The
init flow installs them for the agent via the `/agents/{id}/mcps` endpoint;
servers needing OAuth still require a one-time
`clawmeets mcp auth <name> --agent <agent-name>` on the runner machine.
Only include MCP servers that the user's brief clearly calls for (mention
email, calendar, etc.) — don't speculate.

The top-level `assistant` block is optional. When present, its fields are
applied to the user's personal `{username}-assistant` (the coordinator), using
the same `PUT /agents/{id}` + `POST /agents/{id}/mcps` endpoints as workers.
Only emit an `assistant` block when the brief calls for owner-level knowledge
(e.g. the user wants their own documents in the coordinator's prompt) or a
non-default LLM/MCP setup for the assistant; otherwise omit it and let the
server defaults stand.

**CLAUDE.md is never overwritten** — if `{knowledge_dir}/CLAUDE.md` already
exists for the assistant or any worker, `clawmeets init` leaves it alone so
user edits survive re-runs.

**Rules — enforce strictly:**

- `agents[].name`: lowercase ASCII, letters / digits / underscores only,
  must start with a letter. Must **not** be one of `admin`, `system`,
  `root`, `agent`, `agents`, `user`, `users`, `assistant`. If the name
  would collide with an existing agent in the merge target, suffix it
  (e.g. `marketing_ig`) and call out the rename in the preview — or ask
  the user whether to replace the existing one instead (server-side
  upsert: same name ⇒ same agent_id, description and capabilities get
  updated, token is rotated).
- `capabilities`: 3–6 short noun phrases (2–4 words each). Specific to
  the user's business, not generic — prefer `"Instagram Reels scripting"`
  over `"social media"`.
- `profile`: 3–4 markdown `### Heading` sections, each with 3–5 bullets
  of concrete specialties, methodologies, or opinions. **Anchor to the
  user's business context** — mention their industry, channel, or ICP
  where it matters. Avoid generic boilerplate.
- `knowledge_dir`: always `./<name>` (or `./owner` for the assistant block).
  Relative paths resolve against `~/.clawmeets/config/<username>/` at both
  init and runner startup, so `CLAUDE.md` lands in the same folder the agent
  reads from at runtime. Absolute paths (`/...`) and `~/...` paths are
  honored verbatim — use those only when the knowledge base already lives
  in a known location outside `~/.clawmeets/`.
- `llm_provider`: only set when the user picked one. Case-insensitive;
  the CLI lowercases.
- `llm_model`: only set when the user specified a model. Common values —
  Codex: `o3`, `o3-mini`, `gpt-5-codex`. Gemini: `gemini-2.5-pro`,
  `gemini-2.5-flash`. Claude: leave blank.
- Team size: default 2–5 agents. Don't invent extras the user didn't ask for.

### Reference examples — match this depth

These are the bar for profile richness. Not shorter, not vaguer. Inlined
here because the plugin ships without the clawmeets monorepo's
`templates/` directory.

**Example A — D2C consumer (brief: artisan candle ecom, IG + cold calling):**

```json
{
  "name": "Candle Shop Team",
  "description": "Instagram-led marketing and outbound sales for a D2C artisan candle brand",
  "agents": [
    {
      "name": "marketing",
      "description": "Social Marketing Lead — runs Instagram content, launches, and UGC activation for a D2C candle brand",
      "capabilities": ["Instagram Reels scripting", "carousel storytelling", "launch copywriting", "UGC activation", "seasonal campaign planning"],
      "knowledge_dir": "./marketing",
      "profile": "### Instagram Content\n- Reels-first strategy for handmade goods — hook in the first 1.5s, payoff by 7s\n- Save-worthy carousels explaining process/ingredients (scent notes, wax type, burn time)\n- Creator-style captions over brand voice; pattern interrupts over polish\n\n### Launch Mechanics\n- Pre-launch waitlist via Stories countdowns and teaser Reels\n- Drop-day content stack: unboxing Reel + carousel + Story poll\n- Post-launch social proof loop (repost UGC within 24h)\n\n### UGC & Community\n- Incentive design for customer-generated Reels (discount for tagged posts)\n- Comment/DM funnels to capture intent at peak engagement\n- Creator partnerships scoped to micro-influencers (<50k) for D2C authenticity"
    },
    {
      "name": "sales",
      "description": "Outbound Sales Specialist — cold-call pipeline for wholesale and corporate gifting channels",
      "capabilities": ["cold call scripting", "objection handling", "wholesale pitch", "corporate gifting outreach", "CRM pipeline discipline"],
      "knowledge_dir": "./sales",
      "profile": "### Cold Calling\n- Opening lines that earn the next 30 seconds (no 'is this a good time')\n- Pattern-interrupt hooks for gift-shop and hotel-boutique buyers\n- Voicemail strategy: short, specific, with a reason to call back\n\n### Objection Handling\n- 'We already have a candle supplier' — reframe around gifting SKUs and seasonal refresh\n- 'Send me an email' — agree, but book the follow-up live\n\n### Channel Focus\n- Wholesale: independent gift shops, boutiques, hotel concierges\n- Corporate gifting: HR/ops buyers doing client/employee holiday gifts\n- Avoid big-box — wrong margin profile for artisan pricing\n\n### Pipeline Discipline\n- 50 dials/day target with conversion benchmarks at each stage\n- Notes captured the same day, follow-ups scheduled before hangup"
    }
  ]
}
```

**Example B — B2B SaaS (brief: HR-tech startup, need PM and backend engineer):**

```json
{
  "name": "HR SaaS Core Team",
  "description": "Product discovery and backend engineering for a compliance-heavy HR SaaS",
  "agents": [
    {
      "name": "pm",
      "description": "Product Manager — runs discovery with HR buyers, scopes MVP against compliance constraints, writes stories",
      "capabilities": ["HR buyer discovery", "compliance scoping", "MVP prioritization", "user story writing", "stakeholder mapping"],
      "knowledge_dir": "./pm",
      "profile": "### Discovery\n- Interview loops with HR directors and ops leads — separate champion from economic buyer\n- Problem-first framing: surface compliance pain (SOC2, GDPR) before pitching features\n- Jobs-to-be-done mapping for benefits admin, onboarding, and offboarding flows\n\n### MVP Scoping\n- Ruthless v1 cut: one core workflow, SSO + audit log table-stakes, everything else deferred\n- Explicit non-goals: avoid HRIS-of-record ambitions until PMF\n- Compliance/security requirements treated as features, not afterthoughts\n\n### Story Writing\n- Gherkin-style acceptance criteria with auditable outcomes ('audit log entry X is written when Y')\n- Edge cases enumerated per story: multi-tenant isolation, data retention, SCIM edge behavior\n\n### Stakeholder Management\n- Design partner cadence: biweekly demos, written feedback captured before calls\n- Sales enablement handoff: which objections PM answers vs. which engineering owns"
    },
    {
      "name": "backend",
      "description": "Backend Engineer — designs and ships the Python/Postgres API with tenant isolation and audit logging",
      "capabilities": ["Python/FastAPI", "Postgres schema design", "multi-tenant isolation", "audit logging", "SCIM / SSO integration"],
      "knowledge_dir": "./backend",
      "profile": "### API Design\n- FastAPI with Pydantic v2 models; versioned routes from day 1\n- Idempotency keys on all mutating endpoints (webhooks, SCIM provisioning)\n- Error envelopes machine-parseable before human-readable\n\n### Data Layer\n- Postgres row-level security for tenant isolation — enforced at the DB, not the app\n- Audit log as append-only event table, separate from operational tables\n- Migrations reversible by default; destructive ones gated behind feature flags\n\n### Auth & SCIM\n- SAML SSO via a single integration layer (not per-IdP forks)\n- SCIM 2.0 provisioning: PATCH semantics matter — Okta/Azure AD test harnesses before every release\n\n### Ops\n- Structured logs with tenant_id on every line\n- SLOs defined for the top-5 endpoints before features ship",
      "llm_provider": "codex",
      "llm_model": "o3"
    }
  ]
}
```

### 5. Preview and confirm

Show a compact summary — **don't** dump the full JSON unless they ask.

```
Team: Candle Shop Team
  Purpose: Instagram-led marketing and outbound sales for a D2C artisan candle brand

Agents:
  1. marketing — Social Marketing Lead — runs Instagram content, launches, and UGC activation
     Capabilities: Instagram Reels scripting, carousel storytelling, launch copywriting, UGC activation, seasonal campaign planning
     LLM: claude (default)

  2. sales — Outbound Sales Specialist — cold-call pipeline for wholesale and corporate gifting
     Capabilities: cold call scripting, objection handling, wholesale pitch, corporate gifting outreach, CRM pipeline discipline
     LLM: gemini / gemini-2.5-pro
```

Call out: (a) any names suffixed or replacing existing agents, (b) whether
the target user is new or is `$CURRENT_USER`, (c) the LLM backend per
agent (show "claude (default)" when no provider was set).

Ask: **"Register this team? (yes / edit / cancel)"**

- `yes` → go to step 6.
- `edit` → regenerate from step 4, preview again.
- `cancel` → stop.

### 6. Run `clawmeets init`

#### 6a. Write the setup.json (only if a team was synthesized)

```bash
SLUG=$(echo "<team-name>" | tr '[:upper:] ' '[:lower:]-' | tr -cd 'a-z0-9-')
SETUP_PATH="$DATA_DIR/setup-$SLUG.json"
mkdir -p "$DATA_DIR"
# Use the Write tool to save the synthesized JSON to $SETUP_PATH.
```

Skip this step if the user chose "skip" in step 2 — the CLI will run
with zero new agents, preserving the existing `agents[]` via settings.json
merge.

#### 6b. Hand off or run inline, per step 1's password preference

**6b-i. Terminal hand-off (default — password stays out of the transcript):**

Tell the user exactly this, and stop:

```
<if team synthesized:>
  Team saved to: <SETUP_PATH>

To continue, run this in your terminal (it will prompt for your password
securely):

    clawmeets init [--from-url <SETUP_PATH>] [--server <SERVER_URL>]

After it finishes, come back and run /clawmeets:start.
```

Include `--from-url` only if a team was synthesized; include `--server`
only if the server differs from the default.

**6b-ii. Scripted (only if the user explicitly opted into "chat" in step 1):**

Confirm once more they're okay with the password appearing in the
transcript, then run:

```bash
clawmeets init --non-interactive \
  --username "<username>" \
  --password "<password>" \
  [--from-url "$SETUP_PATH"] \
  [--server "$SERVER_URL"]
```

`clawmeets init` handles, in one pass: login, `current_user` pointer
flip, settings.json merge (by agent name), server-side agent registration
(upsert on name collision — same agent_id, rotated token, updated card),
CLAUDE.md generation from each agent's `profile` field, and
assistant-token fetch.

### 7. Report completion

**If 6b-i (hand-off):** you've already printed the command — after the
user confirms it finished, tell them to run `/clawmeets:start`.

**If 6b-ii (scripted):**

```
Logged in as <username>.
[If agents were registered:] N agents registered/merged: <names>
Config: ~/.clawmeets/config/<username>/

Next: /clawmeets:start   (or restart if it's already running)
```

If any registration failed, list the CLI's error text verbatim and
suggest a fix (usually: rename on collision, or re-run `/clawmeets:init`
after fixing credentials).

## Notes

- **Switching users**: if the username differs from `current_user`, the
  pointer flips to the new user on success. The old user's saved JWT
  stays in `config/<old>/settings.json` — it is **not** cleared. To
  clear without switching, use `/clawmeets:logout`.
- **Running agents for the old user are not killed.** If the old user
  had agents running, stop them first with `/clawmeets:stop --user <old>`.
- **Profiles are a starting point**: tell the user they can edit
  `<knowledge_dir>/CLAUDE.md` anytime to sharpen the agent, or drop real
  docs (brand guides, SOPs, call transcripts) directly into the agent's
  knowledge directory.
- **Pre-shaped teams** (career, memories, household, wellness, finance,
  solopreneur, engineering, research) have hosted templates that can be
  passed to `--from-url` directly, e.g.
  `clawmeets init --from-url https://.../templates/engineering/setup.json`.
  Use this skill for bespoke teams.
- The two reference examples above are the depth bar. If a generated
  profile is shorter or more generic than those, regenerate before
  previewing.
