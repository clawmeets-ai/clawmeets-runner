---
name: coding-plan
description: >
  Use BEFORE writing any non-trivial code in a git-tracked project (FILES &
  STATE block shows a `Git repo` line). Produces a single self-contained
  `plan-<task>.html` with dependency changes, interface diffs, pseudocode
  (signatures + header comments, no bodies), and Mermaid UML / dependency
  diagrams, so the user can review the design before implementation.
  Skip for trivial changes (< ~20 LOC, single file, no public-interface
  change, no new dependency) — reply with an inline diff instead. For
  non-git projects, use `plan` instead.
---

# Coding Plan

Top-down design as an HTML artifact. The goal is to make the **shape** of the change reviewable before keystrokes — interfaces and dependencies first, implementations later. Use this for any non-trivial coding work the user delegates to you.

## When to apply

Produce `plan-<task>.html` for ANY of:

- New feature, endpoint, component, screen, CLI command, MCP tool, or skill
- Refactor that touches > 1 file or changes a public signature
- Schema or migration change
- New dependency (pip, npm, OS package)
- CI/CD / Dockerfile change with a non-obvious effect
- Anything where you'd open more than 2 files to write

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

3. **Plan** — once the design is clear (or no questions were needed), write `plan-<kebab-task-slug>.html` via `update_file`. Use the template below. Keep the file self-contained and ≤ 60 KB.

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

## HTML template

Copy this skeleton verbatim and fill the six sections. Single file, CDN-only, no external assets.

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Plan: TASK_TITLE</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <script>mermaid.initialize({ startOnLoad: true, theme: 'neutral' });</script>
  <!-- Prism: syntax highlighting for pseudocode / sample code in ANY language.
       The autoloader fetches whichever grammar each `language-*` class needs
       (python, typescript, tsx, json, sql, yaml, go, rust, …) on demand, so you
       never pre-declare component scripts. Keep the theme stylesheet above it. -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/themes/prism.min.css">
  <script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-core.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/plugins/autoloader/prism-autoloader.min.js"></script>
  <!-- diff2html: real diff viewer for interface signatures (Section 3).
       Author a plain unified-diff string and call Diff2HtmlUI on a div —
       it handles `+`/`-`/`@@`, file headers, and (optionally) side-by-side
       layout out of the box. Highlight.js is loaded so diff hunks get
       syntax-highlighted to match the rest of the doc. -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/diff2html@3.4.51/bundles/css/diff2html.min.css">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/highlight.js@11.9.0/styles/github.min.css">
  <script src="https://cdn.jsdelivr.net/npm/highlight.js@11.9.0/lib/index.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/diff2html@3.4.51/bundles/js/diff2html-ui.min.js"></script>
  <style>
    /* Generic code block fallback when no Prism language class is set. */
    pre.code { background: #f6f8fa; padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 13px; }
  </style>
</head>
<body class="bg-white text-gray-900 max-w-4xl mx-auto px-6 py-8 space-y-10">

  <header>
    <h1 class="text-2xl font-bold">Plan: TASK_TITLE</h1>
    <p class="text-sm text-gray-500">Author: AGENT_NAME · Chatroom: ROOM</p>
  </header>

  <!-- 1. Overview -->
  <section>
    <h2 class="text-xl font-semibold mb-2">1. Overview</h2>
    <p class="text-lg font-semibold text-gray-900">Verdict: ONE-SENTENCE bottom line — the reader should be able to read just this and know whether to dive in.</p>
    <p class="mt-3">One paragraph: what problem this solves, what changes at a high level, what stays the same.</p>
    <ul class="list-disc ml-6 mt-2 text-sm">
      <li><strong>Scope in:</strong> …</li>
      <li><strong>Scope out:</strong> …</li>
      <li><strong>Assumptions:</strong> …</li>
    </ul>
  </section>

  <!-- 2. Open questions & risks (above the fold — surfaces blockers BEFORE the reader invests in pseudocode / diagrams) -->
  <section>
    <h2 class="text-xl font-semibold mb-2">2. Open questions &amp; risks</h2>
    <ul class="list-disc ml-6 text-sm mb-6">
      <li><strong>Q:</strong> Should `theme` default to user preference instead of "light"? — needs product call.</li>
      <li><strong>Risk:</strong> Cache key collides with existing key if two themes share a hash prefix.</li>
    </ul>

    <!-- Risk matrix: keep when ≥3 risks. Drop when 0-2 — the list above suffices. -->
    <h3 class="font-semibold text-sm mb-2">Risk matrix</h3>
    <div class="grid grid-cols-[120px_1fr_1fr] gap-1 text-xs max-w-2xl">
      <div></div>
      <div class="text-center font-semibold text-gray-600">Low impact</div>
      <div class="text-center font-semibold text-gray-600">High impact</div>

      <div class="text-right font-semibold pr-2 self-center">High likelihood</div>
      <div class="bg-yellow-50 border border-yellow-200 p-2 rounded">Cache key collision (theme prefix)</div>
      <div class="bg-red-50 border border-red-200 p-2 rounded">— none —</div>

      <div class="text-right font-semibold pr-2 self-center">Low likelihood</div>
      <div class="bg-green-50 border border-green-200 p-2 rounded">— none —</div>
      <div class="bg-yellow-50 border border-yellow-200 p-2 rounded">Schema migration partially applied on rollback</div>
    </div>
  </section>

  <!-- 3. Dependency changes -->
  <section>
    <h2 class="text-xl font-semibold mb-2">3. Dependency changes</h2>
    <table class="w-full text-sm border-collapse">
      <thead class="bg-gray-50">
        <tr><th class="text-left p-2 border">Kind</th><th class="text-left p-2 border">Name</th><th class="text-left p-2 border">Change</th><th class="text-left p-2 border">Why</th></tr>
      </thead>
      <tbody>
        <tr><td class="p-2 border">pip</td><td class="p-2 border">httpx</td><td class="p-2 border">add ^0.27</td><td class="p-2 border">async HTTP client for X</td></tr>
        <!-- one row per package / import / module added or removed -->
      </tbody>
    </table>
  </section>

  <!-- 4. Interface diff -->
  <section>
    <h2 class="text-xl font-semibold mb-2">4. Interface diff</h2>
    <!-- ALWAYS render interface diffs through this diff2html block (plain
         unified-diff text in the `<script type="text/plain">` source +
         `Diff2HtmlUI`). Do NOT hand-roll a `<pre class="diff">` with one
         `display:block` <span> per line: under `white-space: pre` the literal
         newline between each </span> and the next <span> renders as its own
         extra blank line, double-spacing the whole diff. diff2html handles
         layout, indentation, and `+`/`-`/`@@` correctly. -->
    <p class="text-sm text-gray-600 mb-2">Public signatures only — function/class declarations, HTTP routes, prop types, env vars. Plain unified-diff text in the `<script type="text/plain">` block below; diff2html renders it.</p>

    <div id="interface-diff"></div>

    <script id="interface-diff-source" type="text/plain">--- a/clawmeets/models/foo.py
+++ b/clawmeets/models/foo.py
@@ -12,3 +12,3 @@
 class Foo:
-    def render(self) -> str: ...
+    def render(self, *, theme: str = "light") -> str: ...
</script>
    <script>
      // toggle outputFormat to 'side-by-side' for wider screens / larger diffs
      const ifDiff = document.getElementById('interface-diff-source').textContent;
      const ifUi = new Diff2HtmlUI(
        document.getElementById('interface-diff'),
        ifDiff,
        { outputFormat: 'line-by-line', drawFileList: false, matching: 'lines', highlight: true }
      );
      ifUi.draw();
    </script>
  </section>

  <!-- 5. Pseudocode -->
  <section>
    <h2 class="text-xl font-semibold mb-2">5. Pseudocode</h2>
    <p class="text-sm text-gray-600 mb-2">Header comments + signatures only. No bodies. Tag with the source language — any Prism language works (`language-python`, `language-typescript`, `language-tsx`, `language-json`, `language-sql`, `language-yaml`, …); the autoloader fetches the grammar on demand. Put the class on both `&lt;pre&gt;` and `&lt;code&gt;` so the theme background applies.</p>
    <pre class="language-python"><code class="language-python"># clawmeets/models/foo.py

class Foo:
    # Render to HTML. Respects `theme` (light|dark).
    # Caches per (id, theme) pair. Raises FooLockedError if state is FROZEN.
    def render(self, *, theme: str = "light") -> str: ...

    # Internal: build the cache key. Pure function.
    def _cache_key(self, theme: str) -> str: ...
</code></pre>
  </section>

  <!-- 6. Visualizations -->
  <section>
    <h2 class="text-xl font-semibold mb-2">6. Visualizations</h2>
    <p class="text-sm text-gray-600 mb-2">Pick the kinds that match the change. See "Visualization picker" in the skill body for guidance on which to use when; the boilerplate below shows one of each — keep what helps, delete the rest.</p>

    <h3 class="font-semibold mt-4">Sequence (Mermaid)</h3>
    <pre class="mermaid">
sequenceDiagram
    participant U as User
    participant API as FastAPI
    participant DB as Postgres
    U->>API: POST /foo
    API->>DB: INSERT foo
    DB-->>API: id
    API-->>U: 201 {id}
    </pre>

    <h3 class="font-semibold mt-4">Module dependencies (Cytoscape — survives >20 nodes, zoom/pan)</h3>
    <div id="cy-deps" style="width:100%;height:420px;border:1px solid #e5e7eb;border-radius:6px"></div>
    <script src="https://cdn.jsdelivr.net/npm/cytoscape@3.30.1/dist/cytoscape.min.js"></script>
    <script>
      // Replace `elements` with your actual node + edge list. `cose` is a
      // good force-directed default; switch to `breadthfirst` for a tree.
      cytoscape({
        container: document.getElementById('cy-deps'),
        elements: [
          { data: { id: 'routes/foo.py' } },
          { data: { id: 'models/foo.py' } },
          { data: { id: 'db/session.py' } },
          { data: { source: 'routes/foo.py', target: 'models/foo.py' } },
          { data: { source: 'models/foo.py', target: 'db/session.py' } },
        ],
        layout: { name: 'cose', animate: false, padding: 24 },
        style: [
          { selector: 'node', style: {
              label: 'data(id)', 'background-color': '#6b2aa0', color: '#fff',
              'text-valign': 'center', 'text-halign': 'center',
              'font-size': 10, 'text-wrap': 'wrap', 'text-max-width': 90,
              shape: 'roundrectangle', width: 'label', height: 28, padding: 6,
          }},
          { selector: 'edge', style: {
              'curve-style': 'bezier', 'target-arrow-shape': 'triangle',
              'line-color': '#6b7280', 'target-arrow-color': '#6b7280', width: 1,
          }},
        ],
      });
    </script>

    <h3 class="font-semibold mt-4">Architecture (Mermaid C4 — context / container / component)</h3>
    <pre class="mermaid">
C4Context
    Person(user, "Reviewer")
    System(api, "ClawMeets API", "FastAPI + WebSocket")
    System_Ext(db, "Postgres")
    System_Ext(cache, "Redis")
    Rel(user, api, "HTTPS")
    Rel(api, db, "SQL")
    Rel(api, cache, "Sessions")
    </pre>

    <h3 class="font-semibold mt-4">UI mockup (inline SVG — frontend changes)</h3>
    <p class="text-xs text-gray-500 mb-1">Sketch before / after at low fidelity. The point is layout + new affordances, not pixel parity.</p>
    <svg viewBox="0 0 720 240" class="w-full max-w-2xl border rounded">
      <!-- Before -->
      <text x="16" y="20" font-family="ui-sans-serif" font-size="11" fill="#6b7280">BEFORE</text>
      <rect x="16" y="32" width="320" height="180" fill="#fafafa" stroke="#e5e7eb"/>
      <rect x="32" y="48" width="288" height="32" fill="#fff" stroke="#e5e7eb"/>
      <text x="44" y="68" font-family="ui-sans-serif" font-size="13" fill="#111">Project · 3 rooms</text>
      <rect x="32" y="96" width="288" height="100" fill="#fff" stroke="#e5e7eb"/>
      <text x="44" y="118" font-family="ui-sans-serif" font-size="12" fill="#6b7280">(no status pill)</text>
      <!-- After -->
      <text x="384" y="20" font-family="ui-sans-serif" font-size="11" fill="#6b2aa0">AFTER</text>
      <rect x="384" y="32" width="320" height="180" fill="#fafafa" stroke="#e5e7eb"/>
      <rect x="400" y="48" width="288" height="32" fill="#fff" stroke="#e5e7eb"/>
      <text x="412" y="68" font-family="ui-sans-serif" font-size="13" fill="#111">Project · 3 rooms</text>
      <rect x="624" y="54" width="56" height="20" fill="#ede9fe" stroke="#6b2aa0" rx="10"/>
      <text x="652" y="68" font-family="ui-sans-serif" font-size="10" fill="#6b2aa0" text-anchor="middle">ACTIVE</text>
      <rect x="400" y="96" width="288" height="100" fill="#fff" stroke="#e5e7eb"/>
    </svg>

    <h3 class="font-semibold mt-4">Scope map (D3 treemap — where the change lives)</h3>
    <div id="scope-map" style="width:100%;height:280px"></div>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script>
      // value = LOC touched per file. Group by top-level dir for the
      // tile color so the reader sees "this is mostly an auth/ change."
      const scope = { name: 'repo', children: [
        { name: 'auth/', children: [
          { name: 'login.py', value: 80 },
          { name: 'session.py', value: 30 },
        ]},
        { name: 'routes/', children: [
          { name: 'auth.py', value: 50 },
        ]},
        { name: 'tests/', children: [
          { name: 'test_login.py', value: 40 },
        ]},
      ]};
      const root = d3.hierarchy(scope).sum(d => d.value).sort((a,b) => b.value - a.value);
      const w = document.getElementById('scope-map').clientWidth, h = 280;
      d3.treemap().size([w, h]).padding(2)(root);
      const sel = d3.select('#scope-map').append('svg').attr('width', w).attr('height', h);
      const color = d3.scaleOrdinal(d3.schemeTableau10).domain(root.children.map(d => d.data.name));
      const leaf = sel.selectAll('g').data(root.leaves()).join('g')
        .attr('transform', d => `translate(${d.x0},${d.y0})`);
      leaf.append('rect')
        .attr('width', d => d.x1 - d.x0).attr('height', d => d.y1 - d.y0)
        .attr('fill', d => color(d.parent.data.name)).attr('opacity', 0.85);
      leaf.append('text').attr('x', 6).attr('y', 16)
        .attr('font-family', 'ui-sans-serif').attr('font-size', 11).attr('fill', '#111')
        .text(d => d.data.name);
    </script>
  </section>

</body>
</html>
```

## Filling the template

**Reading-order contract.** The reader scrolls top → bottom and stops the moment they have what they need. So the doc is ordered **high level first, details after**: sections 1+2 carry the verdict, scope, and blockers (everything the reviewer needs to decide whether to approve in principle); sections 3-6 carry the specifics they drill into only if they want to. Don't reorder. If a risk is load-bearing for the verdict, surface it in section 2 — never hide it down in pseudocode.

- **Section 1 — Overview**: lead with a one-sentence **Verdict** — the bottom line the reader could repeat to a colleague. Then one paragraph on what changes / what doesn't / what you're assuming, plus the Scope in/out bullets. If you had to pick between two reasonable designs, name which and why in one line.
- **Section 2 — Open questions & risks**: this is ABOVE THE FOLD because blockers shouldn't be buried. List ambiguities the user must decide AND risks that could bite during implementation. Keep the risk matrix only when there are ≥3 risks worth ranking; a short bulleted list reads better otherwise. If there are zero open questions and zero non-trivial risks, write "None" — don't fake content.
- **Section 3 — Dependency changes**: one row per added/removed package, new import across module boundaries, new env var, new MCP, new skill. If the row count is 0, write "None" and remove the table.
- **Section 4 — Interface diff**: every public signature you'd touch — function decls, route paths, prop types, CLI flags, schema columns. Bodies stay out. If the change is purely additive (new file, new function), show the new signature as all `+` lines.
- **Section 5 — Pseudocode**: include the header comment that would sit above each function/method. The comment should describe behavior precisely enough that a different engineer could implement it. Bodies are `...`.
- **Section 6 — Visualizations**: at least one; usually two or three. Use the picker table below to choose. Delete the boilerplate examples you don't keep — empty `<div id="cy-deps">` / `<svg>` shells in the final HTML are dead weight.

## Visualization picker

The boilerplate ships with one of each so you have a working starting point. Pick what serves THIS change and delete the rest. Don't render an empty mockup just because the template had one.

| Pick | When | Authoring cost |
|---|---|---|
| **Mermaid `sequenceDiagram`** | Request flow, async coordination, message-passing across components | Low — declarative |
| **Mermaid `classDiagram`** | New types + their relationships (fields, methods, inheritance) | Low |
| **Mermaid `erDiagram`** | DB schema change (tables, FKs, cardinalities) | Low |
| **Mermaid `graph TD` / `graph LR`** | Small module-dep or data-flow graph (< ~15 nodes) | Low |
| **Mermaid `C4Context` / `C4Container`** | Architectural / systems-level view across layers (user / system / external systems). Mermaid v10+ ships it. | Low–medium |
| **Cytoscape.js graph** | Dense module / call / dependency graph (> ~15 nodes). Survives the size, gives zoom + pan + click | Medium — author elements JSON |
| **Inline SVG mockup** | Frontend / UI / component change. Sketch before / after layout at low fidelity — layout + new affordances, not pixel parity | Medium — SVG by hand, but you're good at it |
| **D3 treemap (scope map)** | Cross-cutting refactor across many files. Tiles sized by LOC touched, colored by top-level dir — shows "the change is mostly in `auth/`" at a glance | Medium |

Rules of thumb:

- **One mandatory backbone diagram** that shows the runtime/data flow of the change (usually `sequenceDiagram` or `C4Context`).
- **Frontend tasks**: ALWAYS include an SVG mockup. Words don't convey layout.
- **Refactors touching > 10 files**: include the D3 treemap so the reviewer can triage where to look first.
- **Skip the dense graph** if a `sequenceDiagram` covers the same ground — don't double up.

## Diagram & viz notes

- **Mermaid**: wrap each diagram in `<pre class="mermaid">…</pre>` — auto-renders on page load. Keep nodes labeled with short identifiers; long labels go in quotes: `A["Some long label"]`. Use `C4Context` / `C4Container` for architectural views (Mermaid v10+).
- **Prism** (pseudocode / sample code): any language works — the autoloader loads the grammar on demand, so you never list component scripts. Put the `language-*` class on BOTH `<pre>` and `<code>` so the theme background applies, and always tag real code with a language class (don't fall back to bare `pre.code`, which renders unhighlighted). Escape `<` as `&lt;` and `>` as `&gt;` inside `<code>` blocks — otherwise the HTML breaks.
- **diff2html** (interface diff): put the raw unified diff inside `<script type="text/plain">` (NOT a `<pre>`) so the browser doesn't render the angle brackets or normalize whitespace. Then `new Diff2HtmlUI(target, diffText, opts).draw()`. `outputFormat: 'line-by-line'` is the right default; switch to `'side-by-side'` for diffs with > 30 LOC changed where horizontal comparison helps. Set `drawFileList: true` once the plan touches > 1 file. The library highlights `+`/`-`/`@@` for you — do NOT hand-wrap span tags.
- **Cytoscape**: load the script BEFORE your inline `<script>` that calls `cytoscape({...})`. The container `<div>` needs an explicit `height` (CSS or attribute), otherwise the graph renders at 0px.
- **Inline SVG**: use a `viewBox` so the sketch scales with the page; use `text-anchor="middle"` for centered labels; reuse Tailwind purple `#6b2aa0` for the "new affordance" highlight so the doc reads consistently with the rest of the product.
- **D3 treemap**: read `clientWidth` AFTER the container is in the DOM; the snippet in the template already handles this. Color tiles by their parent (top-level dir), not their leaf — the reviewer wants to see "this is mostly an `auth/` change" at a glance.

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

- **Single file**: everything in one `.html`, CDN-only — Tailwind, Mermaid, Prism, diff2html, highlight.js, Cytoscape, D3 all load from `cdn.jsdelivr.net` / `d3js.org`. No local assets.
- **Naming**: `plan-<kebab-slug>.html`. The slug should match the task, not the date.
- **Size**: target ≤ 80 KB (CDN scripts don't count toward this — they're URLs in your file, not bytes). If your hand-written HTML alone exceeds 80 KB, the task is probably too big — split it first.
- **Don't paste full file contents**: the plan is a sketch, not the change itself. If you find yourself writing > 30 lines of pseudocode in one section, the design is too detailed — push detail into implementation.
- **One plan per task**: if the coordinator gives you 3 distinct tasks in one message, produce 3 plan files (`plan-a.html`, `plan-b.html`, `plan-c.html`), not one combined plan.
- **No implementation until approved**: do not `update_file` any actual code until the user replies `go` (or equivalent). The plan exists to catch direction errors before they cost work.
