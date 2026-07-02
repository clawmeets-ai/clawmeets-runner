---
name: update-skill-config
description: >
  Add, modify, or remove an entry in a per-agent skill config file
  (the JSON files under `{agent_dir}/skill-hub/configs/<name>.json`,
  carrying API keys, sync windows, source/rule definitions, etc.).
  Use whenever you (an agent) need to mutate your own skill config,
  OR — when you are the user's assistant — when you need to do so on
  behalf of another agent the user owns. **Do not edit those files
  directly with Edit/Write/Bash** — go through this skill so the
  server's per-agent config mirror stays in sync with the runner's
  file; a direct edit silently drifts and can be clobbered by the
  next AGENT_SETTINGS_CHANGE broadcast.
---

# Update Skill Config

Update the runtime config for a configurable skill. Common cases: API
keys, sync windows, ETL source/rule definitions.

There are two callsites for this skill:

- **Agent-self** — you (a worker/coordinator) were asked to modify
  your own `skill-hub/configs/<name>.json`. Use `self` as the agent
  argument and rely on the runner-injected env (`CLAWMEETS_AGENT_ID`
  / `CLAWMEETS_AGENT_TOKEN` / `CLAWMEETS_SERVER_URL`).
- **Assistant-on-another-agent** — you are `{username}-assistant` and
  the user asked you to configure a skill on one of their other
  agents. Use the agent's name and `$CLAWMEETS_ASSISTANT_TOKEN`.

## Steps (agent-self)

1. **Fetch the current config and starter template** so you know the
   shape:
   ```bash
   curl -s -H "Authorization: Bearer $CLAWMEETS_AGENT_TOKEN" \
        "$CLAWMEETS_SERVER_URL/agents/$CLAWMEETS_AGENT_ID/skills/<skill_name>/config" \
        | jq .
   ```
   The response carries `config` (current), `starter_config` (template
   with `${VAR}` placeholders for secrets), and `starter_config_text`
   (the JSONC source with comments).

2. **Build the full new config**. The PUT replaces the entry whole —
   carry over the existing keys and apply your delta on top. Do not
   invent values for missing secrets; ask the user.

3. **Write the config JSON** to a scratch file in cwd:
   ```bash
   tmpfile=$(mktemp).json
   cat > "$tmpfile" <<'JSON'
   { "...": "..." }
   JSON
   ```

4. **Apply via the CLI**:
   ```bash
   clawmeets skill set-config self <skill_name> "$tmpfile"
   ```
   The CLI picks up your agent token from the env, PUTs to the server,
   which persists the entry to `card.local_settings.skill_configs` and
   broadcasts `AGENT_SETTINGS_CHANGE` — the runner's
   `_apply_local_settings` writes through to
   `{agent_dir}/skill-hub/configs/<skill_name>.json` immediately.

5. **Clean up the scratch file** and confirm.

## Steps (assistant-on-another-agent)

Same flow, but resolve `<agent>` to a name/id and use
`$CLAWMEETS_ASSISTANT_TOKEN` instead of `$CLAWMEETS_AGENT_TOKEN`:

```bash
curl -s -H "Authorization: Bearer $CLAWMEETS_ASSISTANT_TOKEN" \
     "$CLAWMEETS_SERVER_URL/agents/<agent_id>/skills/<skill_name>/config" \
     | jq .
# ... build $tmpfile ...
clawmeets skill set-config <agent> <skill_name> "$tmpfile"
```

## Error handling

- "skill not installed on agent" → run `/clawmeets:install-skill` first.
- "invalid JSON" → show the user what you wrote and fix.
- "'self' requires CLAWMEETS_AGENT_ID..." → the env vars aren't set;
  you're probably not in an agent subprocess. Pass an explicit agent
  name instead.
