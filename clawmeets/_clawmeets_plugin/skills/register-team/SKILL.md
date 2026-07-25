---
name: register-team
description: >
  Bulk-register a team of worker agents from a `setup.json` template URL
  (or a shipped template short name like `career`, `data`, `engineering`).
  Use when users say "register team", "register agents", "add a team",
  "set up specialists from template", or paste a setup.json URL.
---

# Register Team

Bulk-register a team of worker agents from a `setup.json` template.

This skill is a **thin wrapper** around `clawmeets agent-team register
<url>`. It does NOT synthesize a `setup.json` from a freeform brief —
the user must provide a URL or pick a shipped template by short name.
If brief-to-setup synthesis is ever wanted, that becomes its own skill;
this one stays focused on the deterministic CLI invocation.

Requires being logged in. If you aren't, run `/clawmeets:login` first.
The personal assistant should already exist (run
`/clawmeets:register-assistant` if not — workers coordinate through it).

## Steps

1. **Check CLI and current_user**:
   ```bash
   command -v clawmeets >/dev/null 2>&1 || echo "MISSING_CLI"
   DATA_DIR="${CLAWMEETS_DATA_DIR:-$HOME/.clawmeets}"
   CURRENT_USER=$(cat "$DATA_DIR/config/current_user" 2>/dev/null)
   ```
   - If CLI missing: tell the user to run `/clawmeets:bootstrap`.
   - If no `current_user`: tell the user to run `/clawmeets:login`.

2. **Resolve the template source**. Accept either:
   - **Absolute URL** to a `setup.json` (e.g.
     `https://clawmeets.ai/templates/career/setup.json`), or
   - **Shipped template short name** — resolve to
     `{server_url}/templates/{name}/setup.json`. Read `server_url` from
     `~/.clawmeets/config/{current_user}/settings.json`.

   Shipped template short names (current as of writing — check the
   Welcome page if uncertain): `career`, `customer_success`, `data`,
   `engineering`, `finance`, `information`, `marketing`, `memories`,
   `news`, `nyc`, `personal_data`, `restaurant`, `retail`, `sales`,
   `shopping`, `solopreneur`, `chess`.

3. **Optional: ask whether to register only a subset of the template's
   workers.** Each `setup.json` has an `agents: [...]` list with names;
   the user can pass `--agent NAME` (repeatable) to register only those.
   By default every worker in the template is registered.

4. **Register the team**:
   ```bash
   clawmeets agent-team register "<url>" \
     --agent "<name1>" --agent "<name2>"
   ```
   The CLI:
   - Reads the saved JWT from `settings.json` automatically.
   - Fetches the `setup.json` from the URL (or local path — both work).
   - Registers each worker; writes per-agent `credential.json` +
     `card.json` under `~/.clawmeets/agents/{worker-name}-{id}/`.
   - With `--auto-personalize`: posts `<!-- clawmeets:personalize-trigger -->`
     into each worker's `dm-{worker-name}` chatroom (so the worker
     self-personalizes on its own runner). Off by default — otherwise each
     worker stays reachable via the **Personalize** button in its DM.

   Optional flags:
   - `--llm-provider <p>` overrides the per-agent provider in the
     template for every worker in this run (`claude` | `openai` | `gemini`).
   - `--auto-personalize` enables the personalize-trigger DM fan-out (off by
     default).

5. **Handle errors**:
   - "not logged in" → run `/clawmeets:login` and retry.
   - "Template contains no agent definitions" → the URL is wrong, points
     at an empty file, or returns HTML. Re-verify the URL.
   - "--agent name(s) not in template" → the CLI lists what's available;
     re-prompt the user with that list.
   - HTTP 4xx fetching the URL → likely a typo in the short name, or the
     server requires auth on the templates endpoint.

6. **Recommend skills (batched).** After the workers register, run
   `clawmeets skill list` once. For each registered worker, read its
   `description` + `capabilities` from the template's `setup.json` and pick
   the 2–4 best-fit skills (same matching rule as single-agent). Present the
   picks as **one summary table** — do not prompt per agent:

   | Agent | Suggested skills | Why |
   |---|---|---|
   | research | `deep-research` | multi-source, fact-checked reports |
   | … | … | … |

   Then ask once: *"Install all / pick per agent / skip?"* On approval,
   install per worker:
   ```bash
   clawmeets skill install "<worker-name>" <skill1> [<skill2> ...]
   ```
   Skip any worker with no clear fit. If the run is fully non-interactive
   (e.g. a scripted run with no user to confirm against), skip this step.

7. **Confirm**: "Registered {N} agents: {names}. Run `clawmeets start`
   to bring them online." If the personalize-trigger DMs were posted,
   add: "Each worker will self-personalize on first connect."

## Notes

- **Re-runs are additive, not destructive.** Re-running with the same
  template re-registers the listed workers (server preserves existing
  tokens); use `--agent NAME` to register a single worker out of a
  template you previously partially-registered.
- **One worker at a time?** Use `/clawmeets:register-agent` (single
  agent, you supply name + description) — `register-team` is for
  template-driven bulk setup.
- **Where to find URLs?** The Welcome page in the web UI (`/welcome`)
  lists every shipped template with a copy-button that gives you the
  exact `clawmeets agent-team register <url>` command.
