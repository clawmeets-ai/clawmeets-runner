---
name: explain-as-page
description: >
  Use when an answer needs a diagram, a comparison of 3+ options, or more
  than about two screens of detail. Write the decision in the chat reply as
  usual, then attach ONE `page-<slug>.html` with `update_file`. Not for short
  answers, and not for interactive tools (dashboards, forms, calculators —
  that is `web-artifacts`). Coding plans use this shell via `coding-plan`.
---

# Explain as a page

Some answers don't fit in a chat message: they need a picture, a side-by-side
comparison, or more detail than anyone will scroll through. For those, the
reply still carries the decision, and ONE HTML page carries the evidence. The
user reads the reply first and opens the page only if they want the detail.

This skill owns the page: when to make one, the HTML shell, the reading
order, the visuals, and how it is shared. Other skills (`coding-plan` today)
build on the same shell and add their own sections.

## When a page is worth it

Make a page only when at least one of these holds:

- **A diagram is needed** — a flow, an architecture, a schema, a timeline,
  a layout sketch. Words would take a paragraph to say what a picture says.
- **3 or more options are compared** across several attributes — a table
  with real columns, not a bulleted list.
- **More than about two screens of detail** — evidence the user may want
  to audit, but should not have to scroll through in chat.

Otherwise: reply only. A short answer, a status update, a yes/no, a single
recommendation with two reasons — none of these get a page. When in doubt,
don't make one; a missing page costs the user nothing, a needless one costs
them a click and a reread.

## The reply still stands alone

The page is attached to a reply, never a substitute for one. The reply must
make sense without opening the page:

- The **decision surface** stays in the Markdown reply: the verdict in one
  sentence, any decision the user must make with your recommended default,
  and the outcomes. The page holds the evidence.
- Never write "see the page" as the answer. Write the answer, then say what
  the page adds: "The page has the sequence diagram and the full option
  table."
- Email notifications, connected channels and hub conversations show the
  reply only. A reader there must still be able to decide.

## Draft in Markdown first

Before writing HTML, outline the page as Markdown in your head or your
scratch notes: the verdict, the section headings, one line per section on
what it shows. Then render that outline into the shell. This keeps the page
and the reply consistent — the reply's verdict and the page's verdict must
be the same sentence.

## The shell

Copy this verbatim and fill the body. Single file; libraries load from CDNs
in the reader's browser, so keep the `<head>` as is — a library you don't use
costs nothing.

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PAGE_TITLE</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <!-- Mermaid: flow / sequence / class / ER / C4 diagrams from text. -->
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <script>mermaid.initialize({ startOnLoad: true, theme: 'neutral' });</script>
  <!-- Prism: syntax highlighting for code in ANY language. The autoloader
       fetches whichever grammar each `language-*` class needs on demand. -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/themes/prism.min.css">
  <script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-core.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/plugins/autoloader/prism-autoloader.min.js"></script>
  <!-- diff2html (+ highlight.js): a real viewer for unified diffs. -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/diff2html@3.4.51/bundles/css/diff2html.min.css">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/highlight.js@11.9.0/styles/github.min.css">
  <script src="https://cdn.jsdelivr.net/npm/highlight.js@11.9.0/lib/index.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/diff2html@3.4.51/bundles/js/diff2html-ui.min.js"></script>
  <!-- Cytoscape: dense graphs (> ~15 nodes) with zoom + pan. -->
  <script src="https://cdn.jsdelivr.net/npm/cytoscape@3.30.1/dist/cytoscape.min.js"></script>
  <!-- D3: treemaps and custom charts. -->
  <script src="https://d3js.org/d3.v7.min.js"></script>
</head>
<body class="bg-white text-gray-900 max-w-4xl mx-auto px-6 py-8 space-y-10">

  <header>
    <h1 class="text-2xl font-bold">PAGE_TITLE</h1>
    <p class="text-sm text-gray-500">Author: AGENT_NAME · Room: ROOM</p>
  </header>

  <!-- Verdict box: the same sentence as the reply's first line. -->
  <section class="border-l-4 border-purple-700 bg-purple-50 p-4 rounded">
    <p class="text-lg font-semibold">Verdict: ONE SENTENCE.</p>
    <ul class="list-disc ml-6 mt-2 text-sm">
      <li><strong>Decision needed:</strong> … (recommended: …)</li>
      <li><strong>Outcome:</strong> what is now true …</li>
    </ul>
  </section>

  <!-- Evidence sections: one <section> per question the reader may ask. -->
  <section>
    <h2 class="text-xl font-semibold mb-2">SECTION_TITLE</h2>
    <p>…</p>
  </section>

</body>
</html>
```

## Reading order

The reader scrolls top to bottom and stops the moment they have what they
need. So: **verdict → decisions needed → outcomes → evidence.** The same
rule as a chat reply.

- The verdict box comes first and matches the reply.
- A risk or open question that could change the verdict goes right under
  the verdict box, never deep in an evidence section.
- Evidence sections are ordered by what the reader is most likely to want,
  not by the order you did the work.
- No positional references ("see section 3", "as above"). Each section is
  readable on its own.

## Picking a visual

Pick what serves THIS answer. Don't include a visual because the shell can
draw it.

| Pick | When | Authoring cost |
|---|---|---|
| **Mermaid `sequenceDiagram`** | Request flow, async coordination, message-passing across components | Low — declarative |
| **Mermaid `classDiagram`** | Types and their relationships (fields, methods, inheritance) | Low |
| **Mermaid `erDiagram`** | Database schema (tables, FKs, cardinalities) | Low |
| **Mermaid `graph TD` / `graph LR`** | Small dependency or data-flow graph (< ~15 nodes) | Low |
| **Mermaid `C4Context` / `C4Container`** | Systems-level view across layers (user / system / external systems) | Low–medium |
| **Cytoscape.js graph** | Dense graph (> ~15 nodes). Survives the size, gives zoom + pan | Medium — author elements JSON |
| **Inline SVG mockup** | Any UI or layout question. Low fidelity: layout and new affordances, not pixel parity | Medium |
| **D3 treemap** | "Where does the weight sit?" — e.g. files touched sized by LOC, colored by top-level dir | Medium |
| **HTML table** | Comparing 3+ options across attributes | Low |

Notes:

- **Mermaid**: wrap each diagram in `<pre class="mermaid">…</pre>`; it
  renders on load. Short node ids; long labels in quotes: `A["Long label"]`.
- **Prism**: put the `language-*` class on BOTH `<pre>` and `<code>`.
  Escape `<` as `&lt;` and `>` as `&gt;` inside `<code>`.
- **diff2html**: put the raw unified diff inside
  `<script type="text/plain" id="…">` (NOT a `<pre>`), then
  `new Diff2HtmlUI(target, text, { outputFormat: 'line-by-line', drawFileList: false, matching: 'lines', highlight: true }).draw()`.
  Never hand-roll a diff out of `<span>` lines.
- **Cytoscape**: the container `<div>` needs an explicit height, or the
  graph renders at 0px. `layout: { name: 'cose' }` is a good default.
- **Inline SVG**: use a `viewBox` so it scales; purple `#6b2aa0` marks what
  is new.
- **D3**: read the container's `clientWidth` after it is in the DOM.

## Sharing the page

1. Write the file into your sandbox (your working directory) with the Write
   tool. Name it `page-<kebab-slug>.html`; the slug names the question, not
   the date. Never save it anywhere else.
2. Emit `update_file` with the same relative path in the same turn as the
   reply. It shows up as a file pill under the message; clicking it opens
   the page in the web app's viewer.
3. Updating the page later: rewrite the same file and `update_file` it
   again. Don't create `page-x-v2.html`.

One page per reply. If you find yourself wanting two, the reply is
answering two questions — split them.

## Never

- **No update checks, telemetry or tracking pixels.** The page must not
  call home to anyone.
- **No network calls from the agent's computer** to build the page — no
  rendering service, no installer, no `npx`. You write text; that's all.
- **Never open a browser or a file viewer** on the computer you run on.
  The user reads the page in the web app.
- **No secrets, tokens, or private paths** in the page. It is a file in the
  room; anyone who can read the room can read it.
- **No local assets.** Images are inline SVG or `data:` URIs. CDN
  `<script>` and `<link>` tags are fine — they load in the reader's
  browser, not on your computer.

## Size

Keep the hand-written HTML ≤ 60 KB (CDN libraries are URLs, not bytes).
Bigger usually means the page answers more than one question — split it,
or cut evidence the user won't read.

## Building on the shell

Another skill that needs a page (`coding-plan`, later the completion
reports) does not copy the shell. It says "use the `explain-as-page`
shell", keeps the head and the verdict box, and defines its own evidence
sections and naming (`coding-plan` uses `plan-<slug>.html`). Everything in
this file — the reply rule, reading order, sharing, the never-list, the
size budget — still applies to those pages.
