---
name: git-workflow
description: >
  Your standard workflow for making code changes to the EXISTING codebase in the
  git repo you are bound to. Invoke whenever a task asks you to write, edit, or
  fix code AND you have a bound repo (the FILES & STATE block shows a "Bound git
  repo" line and `$CLAWMEETS_AGENT_GIT_URL` is set). Clones the repo into your
  sandbox, joins the shared project branch (one per repo, per project), and
  extends the code already there (never scaffold a new/parallel project),
  commits, pushes, and announces the branch so a human can open a PR. Do NOT use
  the `update_file` action for source code — code lives in git, not in
  chat-synced files.
---

# Git workflow

You are bound to one git repo. This skill is how your code changes reach it:
**clone → join the project branch → read what teammates landed → edit → commit
→ push → announce**. There is no automatic merge — the project branch is the
deliverable and a human (or the user) opens one PR from it.

Your working directory is the per-project sandbox. Everything below happens
under `./repos/` inside it, so non-code outputs (notes, deliverables) stay
outside the working tree.

## 0. Resolve your binding

The runner injects these env vars (`Write`/`Edit` do not expand env vars —
resolve them to literal strings with `Bash` first):

```bash
echo "$CLAWMEETS_AGENT_GIT_URL"          # the repo you push to (required)
echo "$CLAWMEETS_AGENT_GIT_BASE_BRANCH"  # branch to cut from (may be empty)
echo "$CLAWMEETS_AGENT_ID"               # your stable id
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

`$CLAWMEETS_AGENT_GIT_BASE_BRANCH` is *per-agent* config, so two agents on one
project can disagree about it. **The base is only consulted when the project
branch does not exist yet** (§3). Once it exists on the remote, whoever created
it fixed the base — nobody re-derives it. Never rebase the project branch onto
a different base mid-project.

## 3. Join the project branch

All coding agents on a project push to **one shared branch**:

```
project/<project-slug>
```

- `<project-slug>` = the **Project** name from your prompt's identity block,
  lowercased and reduced to `[a-z0-9._-]` (it is already a kebab slug).
- Derive it yourself — do not wait for anyone to announce it. If the
  coordinator's delegation explicitly names a different branch, that override
  wins; otherwise the derived name is the source of truth.

```bash
SLUG=$(printf '%s' "<project-name>" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9._-' '-' | sed 's/-\+/-/g;s/^-//;s/-$//')
BRANCH="project/$SLUG"

if git -C "$REPO_DIR" ls-remote --exit-code origin "refs/heads/$BRANCH" >/dev/null 2>&1; then
  # ADOPT — teammates (or an earlier turn of yours) already pushed here
  git -C "$REPO_DIR" fetch origin "$BRANCH"
  git -C "$REPO_DIR" checkout -B "$BRANCH" "origin/$BRANCH"
else
  # CREATE — you're first on this repo for this project.
  # Claim the branch BEFORE editing, so a teammate starting at the same moment
  # adopts yours instead of cutting a second one.
  git -C "$REPO_DIR" checkout -B "$BRANCH" "origin/$BASE"
  if ! git -C "$REPO_DIR" push -u origin "$BRANCH"; then
    # Lost the race — someone created it between the ls-remote and the push.
    # Fall back to ADOPT; never force your version over theirs.
    git -C "$REPO_DIR" fetch origin "$BRANCH"
    git -C "$REPO_DIR" checkout -B "$BRANCH" "origin/$BRANCH"
  fi
fi
```

**Adopt-the-remote invariant — the one rule that must not be broken:** if
`origin/$BRANCH` exists, you branch from **`origin/$BRANCH`**, never from
`origin/$BASE`. Re-cutting from base and pushing discards every teammate commit
on the branch.

### 3a. Scope: one branch per (repo, project)

The branch is shared per **repo**, not globally. The `git_url` binding is
per-agent, so a project can span several repos — e.g. a frontend agent bound to
`web.git` and a backend agent bound to `api.git`.

- **Same repo** — every agent lands on the same `project/<slug>` branch, sees
  each other's commits, and the project ships as **one PR**. This is where the
  consistency win comes from.
- **Different repos** — each repo gets its own `project/<slug>` branch,
  containing only the agents bound to that repo. They never see each other's
  code and the project ships as **one PR per repo**. That is expected, not a
  failure: it degrades to the old isolated-branch behaviour, and cross-repo
  agents coordinate the way they always have — through the spec, the shared
  context, and chat. Nothing about the procedure above changes; the same commands
  produce the right result in both cases.

You do not need to know which case you're in before you start. Find out in §3b.

### 3b. Read what teammates already landed

Do this **before** you edit — the shared branch only buys consistency if you
build on what's there instead of re-inferring the design from the spec.

```bash
git -C "$REPO_DIR" log --oneline "origin/$BASE..HEAD"
git -C "$REPO_DIR" diff --stat "origin/$BASE..HEAD"
```

- **Commits from other agents** (see the `Agent:` trailer in §5) ⇒ you share the
  repo with them. Read the files they touched and follow the interfaces,
  naming, and patterns they established. If their code contradicts your reading
  of the spec, their code wins for anything already merged — say so in your
  reply rather than quietly re-doing it.
- **Only your own commits, or none** ⇒ you're alone in this repo (teammates are
  on other repos, or you're first). Work from the spec and the shared-context
  files as usual, and be explicit in your reply about the interface you're
  exposing so the agent on the other repo can match it.

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

Because the branch is shared, every commit carries an `Agent:` trailer so
`git log` still attributes each hunk to whoever wrote it:

```bash
git -C "$REPO_DIR" add -A
git -C "$REPO_DIR" commit -m "<clear, scoped message>" -m "Agent: <your-short-name>"
```

Then push, rebasing onto whatever teammates landed while you were working:

```bash
for i in 1 2 3; do
  git -C "$REPO_DIR" pull --rebase origin "$BRANCH" || break
  git -C "$REPO_DIR" push origin "$BRANCH" && break
done
```

Write real commit messages (what changed and why), not "wip".

**Two hard rules on a shared branch:**

- **Never force-push** (`push --force`, `--force-with-lease`, `push +ref`) and
  never rewrite already-pushed history (`rebase -i`, `commit --amend` on a
  pushed commit, `reset --hard` followed by a push). On a personal branch that
  is harmless; here it silently destroys a teammate's commits.
- **On a rebase conflict, stop.** Run `git -C "$REPO_DIR" rebase --abort`, leave
  the branch as it is, and report in chat which files conflicted and with whose
  commit. Do NOT resolve it yourself — you cannot see the intent behind the
  other agent's concurrent edit, and a wrong resolution corrupts their work in a
  way nobody reviews. The coordinator serializes the two of you and re-runs.

## 6. Announce the branch and repo

In your `reply`, name **both the repo and the branch** you pushed to — e.g.
*"Pushed to `project/feature-x` in `api.git`; ready to PR into `main`."* The
repo matters: when a project spans several repos there is one branch per repo
(§3a), and the coordinator's completion report needs to know which repos to
diff. Also say briefly what you added to the branch, so the next agent on it
knows what to build on without reading the whole diff.

The branch is the project's deliverable hand-off, the same way `update_file` is
for non-code work.
