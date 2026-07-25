---
name: git-workflow
description: >
  Your standard workflow for making code changes to the EXISTING codebase in the
  git repo you are bound to. Invoke whenever a task asks you to write, edit, or
  fix code AND you have a bound repo (the FILES & STATE block shows a "Bound git
  repo" line and `$CLAWMEETS_AGENT_GIT_URL` is set). Clones the repo into your
  sandbox, cuts a per-request branch, and extends the code already there (never
  scaffold a new/parallel project), commits, pushes, and announces the branch so
  a human can open a PR. Do NOT use the `update_file` action for source code —
  code lives in git, not in chat-synced files.
---

# Git workflow

You are bound to one git repo. This skill is how your code changes reach it:
**clone → branch → edit → commit → push → announce**. There is no automatic
merge — you push a branch and a human (or the user) opens the PR.

Your working directory is the per-project sandbox. Everything below happens
under `./repos/` inside it, so non-code outputs (notes, deliverables) stay
outside the working tree.

## 0. Resolve your binding

The runner injects these env vars (`Write`/`Edit` do not expand env vars —
resolve them to literal strings with `Bash` first):

```bash
echo "$CLAWMEETS_AGENT_GIT_URL"          # the repo you push to (required)
echo "$CLAWMEETS_AGENT_GIT_BASE_BRANCH"  # branch to cut from (may be empty)
echo "$CLAWMEETS_AGENT_ID"               # your stable id (branch disambiguator)
```

If `$CLAWMEETS_AGENT_GIT_URL` is empty you have no bound repo — do not use this
skill; share outputs with `update_file` instead.

**Prerequisite:** the machine running you must already have push access to the
repo (an SSH key or a git credential helper). If a `git push` fails with an
auth error, stop and say so in chat — do not attempt to reconfigure credentials.

## 1. Clone (once) or fetch

Derive the repo dir name from the URL basename (strip a trailing `.git`):

```bash
REPO_DIR="repos/$(basename "$CLAWMEETS_AGENT_GIT_URL" .git)"
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" fetch origin
else
  git clone "$CLAWMEETS_AGENT_GIT_URL" "$REPO_DIR"
fi
```

Run every later `git` command with `git -C "$REPO_DIR" …` (or `cd` into it).

## 2. Pick the base branch

Use `$CLAWMEETS_AGENT_GIT_BASE_BRANCH` if set; otherwise the repo's default:

```bash
BASE="${CLAWMEETS_AGENT_GIT_BASE_BRANCH:-$(git -C "$REPO_DIR" remote show origin \
  | sed -n 's/.*HEAD branch: //p')}"
```

## 3. Branch per request

One branch per coding request (this project), namespaced by you so several
agents can share a repo without colliding:

```
<request-slug>/<your-short-name>
```

- `<request-slug>` = the **Project** name from your prompt's identity block,
  lowercased and reduced to `[a-z0-9._-]` (it is already a kebab slug).
- `<your-short-name>` = your own name (the part after your owner's username),
  same sanitization. If unsure, use a short prefix of `$CLAWMEETS_AGENT_ID`.

```bash
SLUG=$(printf '%s' "<project-name>" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9._-' '-' | sed 's/-\+/-/g;s/^-//;s/-$//')
BRANCH="$SLUG/<your-short-name>"
git -C "$REPO_DIR" checkout -B "$BRANCH" "origin/$BASE"
```

If the branch already exists on the remote (you're iterating on the same
request), check it out and `git pull --rebase origin "$BRANCH"` instead of
recutting it from base.

## 4. Read repo conventions, then make changes

**This repo already contains a working codebase — you are extending it, not
starting fresh.** Before editing, explore what is already there (`ls`, read the
README and the relevant entry points) and integrate your change into the
existing modules, structure, and conventions. Do NOT create a new standalone
project scaffold alongside the existing code.

Also read `$CLAWMEETS_AGENT_DIR/memory/REPO.md` if it exists — it holds the
architecture, conventions, and gotchas you've accumulated for this repo. Follow
them. (You don't write `REPO.md` here; your reflection cycle keeps it current.)

Make your edits inside `$REPO_DIR`.

## 5. Commit and push

```bash
git -C "$REPO_DIR" add -A
git -C "$REPO_DIR" commit -m "<clear, scoped message>"
git -C "$REPO_DIR" push -u origin "$BRANCH"
```

Write real commit messages (what changed and why), not "wip". On a push
conflict, `git -C "$REPO_DIR" pull --rebase origin "$BRANCH"`, resolve, push
again.

## 6. Announce the branch

In your `reply`, tell the team the pushed branch name (and repo) so the user can
review and open a PR — e.g. *"Pushed `feature-x/backend-x`; ready to PR into
`main`."* The branch is your deliverable hand-off, the same way `update_file`
is for non-code work.
