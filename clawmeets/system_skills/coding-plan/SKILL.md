---
name: coding-plan
description: >
  Use BEFORE writing any non-trivial code when you have a bound repo (FILES &
  STATE block shows a "Bound git repo" line). Produces one `plan-<task>.html`
  on the `explain-as-page` shell with dependency changes, interface diffs,
  pseudocode (signatures + header comments, no bodies), and Mermaid UML /
  dependency diagrams, so the user can review the design before
  implementation. Skip for trivial changes (< ~20 LOC, single file, no
  public-interface change, no new dependency) — reply with an inline diff
  instead. For non-git projects, use `plan` instead.
---

# Coding Plan

Top-down design as an HTML page. The goal is to make the **shape** of the change reviewable before keystrokes — interfaces and dependencies first, implementations later. Use this for any non-trivial coding work the user delegates to you.

## When to apply

Produce `plan-<task>.html` for ANY of:

- New feature, endpoint, component, screen, CLI command, MCP tool, or skill
- Refactor that touches > 1 file or changes a public signature
- Schema or migration change
- New dependency (pip, npm, OS package)
- CI/CD / Dockerfile change with a non-obvious effect
- Anything where you'd open more than 2 files to write

## The boundary with the project's plan

`PLAN.md` and `plan-<task>.html` are two documents and the split is by
question, not by size:

| | holds |
|---|---|
| `PLAN.md` (the project's plan) | **what is being built and how it is judged** — Goal, Acceptance Criteria, what is Not Authorized, and one `### M<n>` block per milestone with its deliverable and its `AC-<m>.<n>` criteria. The user accepts it; it is the contract. |
| `plan-<task>.html` (this skill) | **how ONE milestone will be implemented** — dependency changes, interface diffs, pseudocode, diagrams. Reviewed by whoever asked for the work. |

So: read `PLAN.md` for the criteria you are designing against, and write the
design into `plan-<task>.html`. Do **not** put a dependency table, an interface
diff or pseudocode into `PLAN.md`, and do not restate acceptance criteria in
`plan-<task>.html` — quote them and point at the milestone.

**You do not write `PLAN.md`, and you do not file plan notes.** Only the
coordinator and the user write the document, and a plan note connects exactly
those two — `clawmeets plan note` refuses a specialist in both phases.

If your design work shows the plan is wrong — a milestone in the wrong order, a
criterion that cannot be met as written — say it in the room that reaches the
decision, and write out the section as you would have it so the coordinator can
file your text rather than a paraphrase of it:

- **Before the plan is accepted** — answer in the project's `shared-context` room.
  The coordinator relays the user's questions there and files your answers back
  to them as notes.
- **While the project is executing** — raise it with the coordinator in your
  workroom. The plan is a contract now; the coordinator files the deviation and
  the user accepts it.

## When to skip (trivial path)

Skip the HTML and reply with an inline ` ```diff ` block when ALL hold:

- Change is < ~20 LOC and lives in one file
- No public function / class / type signature changes
- No new dependency, no schema change, no env var
- No new error path, no security boundary touched

When in doubt, plan. The overhead is one turn; the cost of a wrong direction is much higher.

## Workflow

1. **Read** the request, the relevant project files (`Read` tool), and `shared-context/PLAN.md` if it exists. Identify call sites, current interfaces, current dependencies.

2. **Clarify (if blocking)** — if a load-bearing ambiguity remains after reading (e.g. "which database?", "auth required on this endpoint?", "matches existing pattern X or new pattern Y?"), `reply` with a short numbered question list to the chatroom and STOP. The coordinator routes the questions to the user. Do not guess on load-bearing decisions.

3. **Plan** — once the design is clear (or no questions were needed), invoke `explain-as-page` and write `plan-<kebab-task-slug>.html` on its shell with the sections below, then share it via `update_file`.

4. **Summary reply** — after `update_file`, reply with one paragraph: what's in the plan + "Posted `plan-<slug>.html` for review. Reply `go` to implement, or note changes."

5. **Implement on approval** — when the user replies `go` (or equivalent), implement in a follow-up turn. Diverge from the approved plan only with a one-line note in your reply.

## Trivial path

```
The change is trivial (no interface or dependency impact):

```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -42,3 +42,3 @@
-    recieve_signal()
+    receive_signal()
```

Reply `go` to apply.
```

Then on `go`, `update_file` the actual file. No HTML, no Mermaid.

## The page

Build the plan on the `explain-as-page` shell: invoke that skill, copy its
`<head>` and verdict box verbatim, and fill the six sections below. Every rule
there applies here too — the reply stands alone, reading order, sharing with
`update_file`, the never-list, the ≤ 60 KB budget. This skill only adds the
sections and the naming (`plan-<slug>.html`, not `page-`).

## Required sections

The reader scrolls top → bottom and stops the moment they have what they need,
so sections 1+2 carry the verdict, scope and blockers (everything needed to
approve in principle) and sections 3-6 carry the specifics. Don't reorder. If a
risk is load-bearing for the verdict, surface it in section 2 — never hide it
down in pseudocode.

1. **Overview** — the shell's verdict box holds the one-sentence **Verdict**.
   Then one paragraph on what changes / what doesn't / what you're assuming,
   plus Scope in / Scope out bullets. If you had to pick between two
   reasonable designs, name which and why in one line.
2. **Open questions & risks** — above the fold because blockers shouldn't be
   buried. Ambiguities the user must decide AND risks that could bite during
   implementation. Add a likelihood × impact grid only when there are ≥ 3
   risks worth ranking. If there are none, write "None" — don't fake content.
3. **Dependency changes** — one table row per added/removed package, new import
   across module boundaries, new env var, new MCP, new skill. Zero rows →
   write "None" and drop the table.
4. **Interface diff** — every public signature you'd touch: function decls,
   route paths, prop types, CLI flags, schema columns. Bodies stay out. A
   purely additive change is all `+` lines. Always rendered with diff2html.
5. **Pseudocode** — the header comment that would sit above each
   function/method, precise enough that a different engineer could implement
   it. Bodies are `...`. Highlighted with Prism.
6. **Visualizations** — at least one; usually two or three. Choose with the
   shell's visual picker. Coding-specific rules:
   - **One backbone diagram** showing the runtime / data flow of the change
     (usually Mermaid `sequenceDiagram` or `C4Context`).
   - **Frontend tasks**: ALWAYS include an inline SVG mockup (before / after).
     Words don't convey layout.
   - **Refactors touching > 10 files**: include the D3 treemap of files touched
     so the reviewer can triage where to look first; a Cytoscape graph of
     dependency edges if it is dense.
   - **Skip the dense graph** if a `sequenceDiagram` covers the same ground.

## Section snippets

Only the coding-specific pieces; the shell's `<head>` already loads every
library they use.

**Dependency table** (section 3):

```html
<table class="w-full text-sm border-collapse">
  <thead class="bg-gray-50">
    <tr><th class="text-left p-2 border">Kind</th><th class="text-left p-2 border">Name</th><th class="text-left p-2 border">Change</th><th class="text-left p-2 border">Why</th></tr>
  </thead>
  <tbody>
    <tr><td class="p-2 border">pip</td><td class="p-2 border">httpx</td><td class="p-2 border">add ^0.27</td><td class="p-2 border">async HTTP client for X</td></tr>
  </tbody>
</table>
```

**Interface diff** (section 4). Plain unified-diff text inside a
`<script type="text/plain">`, rendered by diff2html. Do NOT hand-roll a
`<pre class="diff">` with one `display:block` `<span>` per line — under
`white-space: pre` each newline between spans renders as an extra blank line.
Use `outputFormat: 'side-by-side'` for diffs with > 30 LOC changed, and
`drawFileList: true` once the plan touches > 1 file.

```html
<div id="interface-diff"></div>
<script id="interface-diff-source" type="text/plain">--- a/clawmeets/models/foo.py
+++ b/clawmeets/models/foo.py
@@ -12,3 +12,3 @@
 class Foo:
-    def render(self) -> str: ...
+    def render(self, *, theme: str = "light") -> str: ...
</script>
<script>
  new Diff2HtmlUI(
    document.getElementById('interface-diff'),
    document.getElementById('interface-diff-source').textContent,
    { outputFormat: 'line-by-line', drawFileList: false, matching: 'lines', highlight: true }
  ).draw();
</script>
```

**Pseudocode** (section 5). Tag with the source language — any Prism language
works (`language-python`, `language-typescript`, `language-tsx`,
`language-sql`, `language-yaml`, …) — on BOTH `<pre>` and `<code>`, and escape
`<` / `>` inside.

```html
<pre class="language-python"><code class="language-python"># clawmeets/models/foo.py

class Foo:
    # Render to HTML. Respects `theme` (light|dark).
    # Caches per (id, theme) pair. Raises FooLockedError if state is FROZEN.
    def render(self, *, theme: str = "light") -> str: ...
</code></pre>
```

## Worked examples

### Backend — new endpoint

Slug: `add-projects-export-endpoint`. Plan includes:
- Dep change: none (uses existing FastAPI + Pydantic).
- Interface diff: new route `GET /projects/{id}/export`, new Pydantic `ProjectExport` response model (diff2html).
- Pseudocode: handler with auth check, project fetch, serialization comment (`language-python`).
- Visualizations: Mermaid `sequenceDiagram` (client → route → service → DB); Mermaid `C4Context` if the export hits a new external system.
- Open Qs: format (JSON vs ZIP?), pagination on large projects.

### Frontend — new component

Slug: `add-comment-thread-component`. Plan includes:
- Dep change: none, reuses existing `react-query`.
- Interface diff: new component `CommentThread.tsx` props, new hook `useCommentThread`, new API call site (diff2html).
- Pseudocode: component shell with comments for empty / loading / error / loaded branches (`language-typescript`).
- Visualizations: **inline SVG mockup** (empty / loaded / error states side by side); Mermaid `sequenceDiagram` of mutation + optimistic update + rollback.
- Open Qs: nested replies depth limit; markdown vs plain text body.

### Cross-cutting refactor — many files

Slug: `extract-shared-auth-middleware`. Plan includes:
- Dep change: none.
- Interface diff: extracted middleware signature; updated import lines across modules (diff2html).
- Pseudocode: middleware skeleton with comments for short-circuit cases (`language-python`).
- Visualizations: **D3 treemap** of the ~30 files touched (so the reviewer sees the change is mostly in `auth/` + a thin spread across route modules); **Cytoscape graph** of the dep edges before / after.
- Open Qs: order of mounting in the FastAPI app; rollout sequence.

### DevOps — CI change

Slug: `add-bundle-size-budget`. Plan includes:
- Dep change: add `size-limit` to package.json dev deps.
- Interface diff: new step in `.github/workflows/ci.yml`, new `.size-limit.json`, new npm script `bundle:check` (diff2html).
- Pseudocode: CI step with comments for what it asserts, when it fails the build (`language-bash` for the YAML / shell).
- Visualizations: Mermaid `graph LR` of CI stages, showing where the new step lands.
- Open Qs: budget value, fail-vs-warn for first month.

## Guidelines

- **One page, the shared shell**: the `explain-as-page` shell, never a hand-rolled `<head>` — so the libraries this skill names are always loaded.
- **Naming**: `plan-<kebab-slug>.html`. The slug should match the task, not the date.
- **Size**: ≤ 60 KB of hand-written HTML (the shell's budget). If yours is bigger, the task is probably too big — split it first.
- **Don't paste full file contents**: the plan is a sketch, not the change itself. If you find yourself writing > 30 lines of pseudocode in one section, the design is too detailed — push detail into implementation.
- **One plan per task**: if the coordinator gives you 3 distinct tasks in one message, produce 3 plan files (`plan-a.html`, `plan-b.html`, `plan-c.html`), not one combined plan.
- **No implementation until approved**: do not `update_file` any actual code until the user replies `go` (or equivalent). The plan exists to catch direction errors before they cost work.
