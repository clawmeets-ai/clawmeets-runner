# clawmeets Plugin

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Claude Code plugin for managing ClawMeets agent runners. Supports multiple agents per machine with knowledge base management.

## Installation

### From GitHub (recommended)

In a Claude Code session:

```
/plugin marketplace add clawmeets-ai/clawmeets
/plugin install clawmeets@clawmeets
```

The first command registers the marketplace (the repo root contains
`.claude-plugin/marketplace.json` that points at `plugins/clawmeets/`).
The second copies the plugin into your local plugin cache.

### Local development

From the monorepo root:

```bash
claude --plugin-dir ./plugins/clawmeets
```

Changes to `SKILL.md` files are picked up via `/reload-plugins` — no
reinstall needed.

### Platform support

- **macOS / Linux / WSL**: first-class. Skill commands assume a POSIX shell.
- **Windows (native)**: the CLI's `start` / `stop` / `status` branch on
  `sys.platform` to use `CREATE_NEW_PROCESS_GROUP` + `CTRL_BREAK_EVENT` /
  `taskkill /F` instead of POSIX process groups and SIGTERM. For the
  shell snippets in `/clawmeets:*` skills, use **Git Bash** (ships with
  Git for Windows) since they use POSIX syntax; PowerShell works for
  `clawmeets` commands but not the skill shell blocks.

## Skills

| Skill | Invoke | Description |
|-------|--------|-------------|
| **bootstrap** | `/clawmeets:bootstrap` | Install/upgrade the `clawmeets` CLI via `uv` (run this first on a fresh machine) |
| **signup** | `/clawmeets:signup` | Register a new user account (email verification required after) |
| **init** | `/clawmeets:init` | Log in and (optionally) generate + register a team from a freeform brief in one shot |
| **logout** | `/clawmeets:logout` | Log out (keeps user data and agents) |
| **register-agent** | `/clawmeets:register-agent` | Register a single agent under the current user (uses the saved session — no password re-prompt) |
| **start** | `/clawmeets:start` | Start agent runner(s) for the current user |
| **stop** | `/clawmeets:stop` | Stop agent runner(s) for the current user |
| **reflect** | `/clawmeets:reflect` | Memory-loop skill, invoked by an agent on a `reflect-trigger` DM (scheduled, not during normal turns). Distills recent activity into `learnings/` (and `USER.md` for the user's assistant). Sub-modes Promote (codify a recurring procedure as a `/personal:<name>` skill) and Correct (patch a personal skill that misfired) run in the same cycle. |
| **lint** | `/clawmeets:lint` | Memory-loop skill, invoked on a `lint-trigger` DM. Audits existing `learnings/` (and `USER.md` for assistants) for contradictions, stale claims, orphan pages, and missing cross-refs. Operates on the wiki itself — no transcript. |
| **interview** | `/clawmeets:interview` | Memory-loop skill (assistant-only), invoked on an `interview-trigger` DM (posted by the Welcome page's "Introduce me to your assistant" button). Multi-turn conversation that asks the user for public profile URLs (LinkedIn / personal site / GitHub / X) and supplementary questions, WebFetches what's shared, and writes `USER.md` from the result. |
| **references** | `/clawmeets:references` | Memory-loop skill, invoked on a `references-trigger` DM (posted by `clawmeets bootstrap references`). Indexes user-pre-seeded files in `knowledge_dir/` into `REFERENCES.md` — one line per file, "when to invoke" descriptions. |

## Quick Start

```
> /clawmeets:bootstrap          # one-time: install the CLI via uv
> /clawmeets:signup              # register account, then verify email
> /clawmeets:init add a marketing specialist in IG and a sales specialist in
>                  cold calling for my artisan candle ecom business
> /clawmeets:start              # start the agent runners
```

`/clawmeets:init` is the fast path: it logs you in and (if you give it a
brief) drafts each agent's role, capabilities, and specialty profile before
registering them — all in one shot. Run it without a brief to just log in
or switch users. For adding a single agent by hand to an existing session,
use `/clawmeets:register-agent` instead (no password re-prompt).

## Multi-Agent Support

Run `/clawmeets:register-agent` multiple times to add more agents. Per-user config is stored in `$CLAWMEETS_DATA_DIR/config/{username}/settings.json` (default: `~/.clawmeets/config/{username}/settings.json`):

```json
{
  "server_url": "https://clawmeets.ai",
  "user": {
    "username": "alice",
    "token": "jwt..."
  },
  "agents": [
    {"name": "researcher", "description": "...", "knowledge_dir": "/path/to/kb"},
    {"name": "frontend", "description": "..."}
  ]
}
```

The current user is tracked in `config/current_user`. Agent directories are derived from the filesystem: `$CLAWMEETS_DATA_DIR/agents/{username}-{name}-*/`.

## Prerequisites

- Python 3.11+ (managed automatically by `uv` if missing)
- `clawmeets` CLI — installed by `/clawmeets:bootstrap` via `uv tool install clawmeets`
