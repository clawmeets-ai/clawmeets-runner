# SPDX-License-Identifier: MIT
"""
clawmeets/utils/email_templates.py
HTML + plain-text builders for notification emails.

Exposes three builders, one per notification type. Each returns
``(subject, plain_text_body, html_body)``. Callers pass the tuple
directly to ``send_notification_email``.
"""
from __future__ import annotations

import html as _html
import logging
import re
from typing import Optional
from urllib.parse import quote

from ..models.project import Project

logger = logging.getLogger("clawmeets.email_templates")

_BRAND = "ClawMeets"
_COLOR_HEADING = "#111827"
_COLOR_BODY = "#1f2937"
_COLOR_MUTED = "#6b7280"
# Primary CTA matches the `bg-purple-600` button used across the web app
# (New Project button, Sign in / Sign up, settings save buttons, Send).
_COLOR_ACCENT = "#9333ea"        # tailwind purple-600
_COLOR_ACCENT_HOVER = "#7e22ce"  # tailwind purple-700
# Wordmark colors match <BrandName> in the web app (Claw / Meets, two tones).
_COLOR_BRAND_CLAW = "#2d2272"
_COLOR_BRAND_MEETS = "#7c3aed"
_COLOR_BG = "#f9fafb"
_COLOR_CARD = "#ffffff"
_COLOR_BORDER = "#e5e7eb"
_FONT_STACK = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
    "'Helvetica Neue', Arial, sans-serif"
)


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _render_markdown_html(text: str) -> str:
    """Render markdown to HTML, preferring markdown-it-py when available."""
    if not text:
        return ""
    try:
        from markdown_it import MarkdownIt  # type: ignore
    except ImportError:
        return _render_markdown_fallback(text)
    md = MarkdownIt("commonmark", {"breaks": True, "linkify": True})
    return md.render(text).strip()


def _render_markdown_fallback(text: str) -> str:
    """Minimal markdown-to-HTML converter for when markdown-it-py is missing.

    Handles the subset agents actually use: paragraphs, blank-line breaks,
    `**bold**`, `*italic*`, ``code``, `[text](url)`, `#`-headings, `-`/`*`
    bullets, and `1.`-numbered lists. Unknown syntax passes through as text.
    """
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Heading
        m = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if m:
            level = len(m.group(1))
            content = _inline(m.group(2))
            out.append(f"<h{level} style=\"margin:16px 0 8px;color:{_COLOR_HEADING};\">{content}</h{level}>")
            i += 1
            continue

        # Unordered list
        if re.match(r"^[-*]\s+", stripped):
            items: list[str] = []
            while i < len(lines) and re.match(r"^[-*]\s+", lines[i].strip()):
                item = re.sub(r"^[-*]\s+", "", lines[i].strip())
                items.append(f"<li style=\"margin:4px 0;\">{_inline(item)}</li>")
                i += 1
            out.append(
                f"<ul style=\"margin:8px 0;padding-left:20px;color:{_COLOR_BODY};\">"
                + "".join(items)
                + "</ul>"
            )
            continue

        # Ordered list
        if re.match(r"^\d+\.\s+", stripped):
            items = []
            while i < len(lines) and re.match(r"^\d+\.\s+", lines[i].strip()):
                item = re.sub(r"^\d+\.\s+", "", lines[i].strip())
                items.append(f"<li style=\"margin:4px 0;\">{_inline(item)}</li>")
                i += 1
            out.append(
                f"<ol style=\"margin:8px 0;padding-left:22px;color:{_COLOR_BODY};\">"
                + "".join(items)
                + "</ol>"
            )
            continue

        # Paragraph (consume until blank line)
        buf: list[str] = [stripped]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(
            r"^(#{1,6}\s|[-*]\s|\d+\.\s)", lines[i].strip()
        ):
            buf.append(lines[i].strip())
            i += 1
        joined = " ".join(buf)
        out.append(
            f"<p style=\"margin:12px 0;color:{_COLOR_BODY};line-height:1.55;\">"
            f"{_inline(joined)}</p>"
        )

    return "\n".join(out)


def _inline(text: str) -> str:
    """Apply inline markdown: bold, italic, code, links. Escapes HTML."""
    escaped = _html.escape(text)
    # Links: [text](url) — do before emphasis so bracket chars aren't munged
    escaped = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: f"<a href=\"{m.group(2)}\" style=\"color:{_COLOR_BRAND_MEETS};text-decoration:underline;\">{m.group(1)}</a>",
        escaped,
    )
    # Inline code
    escaped = re.sub(
        r"`([^`]+)`",
        r"<code style=\"background:#f3f4f6;padding:1px 4px;border-radius:3px;font-family:'SF Mono',Menlo,monospace;font-size:0.9em;\">\1</code>",
        escaped,
    )
    # Bold
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    # Italic (single *; avoid matching inside the bold already converted)
    escaped = re.sub(r"(?<!\*)\*(?!\s)([^*]+?)(?<!\s)\*(?!\*)", r"<em>\1</em>", escaped)
    return escaped


def _strip_markdown(text: str) -> str:
    """Strip markdown syntax for plain-text email bodies."""
    if not text:
        return ""
    # Code blocks: drop fences, keep content
    text = re.sub(r"```[\w-]*\n?", "", text)
    text = re.sub(r"```", "", text)
    # Headings: drop leading #
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Bold/italic markers
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*(?!\s)([^*]+?)(?<!\s)\*(?!\*)", r"\1", text)
    # Inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Links: [text](url) -> text (url)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GENERIC_ROLES = {"assistant", "agent", "bot", "helper"}


def _humanize_agent_name(name: str, username: Optional[str] = None) -> str:
    """Turn slugs like ``chuswine-bc`` into ``BC`` or ``Chuswine BC``.

    If the slug is prefixed with the username (common when agents belong
    to a user), strip the redundant prefix — unless what's left is a
    generic role word like ``assistant``, in which case keep the prefix
    so the result stays recognizable (``Chuswine Assistant``).
    """
    if not name:
        return "your agent"
    display = name
    if username:
        pref = f"{username}-"
        if display.lower().startswith(pref.lower()) and len(display) > len(pref):
            remainder = display[len(pref):]
            if remainder.lower() not in _GENERIC_ROLES:
                display = remainder
    parts = re.split(r"[-_\s]+", display)
    titled = " ".join(_title_part(p) for p in parts if p)
    return titled or name


def _title_part(part: str) -> str:
    """Title-case a token; treat short consonant-only tokens as acronyms."""
    if len(part) <= 3 and not re.search(r"[aeiouAEIOU]", part):
        return part.upper()
    return part.capitalize()


def _first_sentence(text: str, max_len: int = 90, min_len: int = 40) -> str:
    """Return a short, sentence-ish fragment suitable for a subject preview.

    Prefers sentence boundaries that give at least ``min_len`` chars, so a
    leading ``"Got it."`` doesn't swallow the whole preview. Falls back to
    the first line, then to a clean ellipsized cut.
    """
    clean = _strip_markdown(text)
    # Take the first line/paragraph, but keep joining short leading lines
    # until we clear min_len so "Got it.\n\nHere's the plan" becomes
    # "Got it. Here's the plan" rather than just "Got it."
    first_para = re.split(r"\n\s*\n", clean, maxsplit=1)[0]
    flat = re.sub(r"\s+", " ", first_para).strip()
    if not flat:
        return ""

    # Collect sentence boundaries inside the window
    boundaries = [m.end() for m in re.finditer(r"[.!?](\s|$)", flat[: max_len + 30])]
    candidate: Optional[str] = None
    for end in boundaries:
        snippet = flat[:end].strip()
        if len(snippet) >= min_len and len(snippet) <= max_len:
            candidate = snippet
            break
        if len(snippet) > max_len:
            break
    if candidate is None and boundaries:
        # Last-ditch: accept a short sentence if the whole first line is short
        if len(flat) <= max_len:
            candidate = flat
    if candidate is None:
        if len(flat) <= max_len:
            candidate = flat
        else:
            candidate = flat[: max_len - 1].rstrip() + "\u2026"
    return candidate


def _chatroom_url(app_base_url: str, project_id: str, chatroom_name: str) -> str:
    base = app_base_url.rstrip("/")
    return f"{base}/app/projects/{project_id}/chatrooms/{quote(chatroom_name, safe='')}"


def _project_url(app_base_url: str, project_id: str) -> str:
    return f"{app_base_url.rstrip('/')}/app/projects/{project_id}"


# ---------------------------------------------------------------------------
# HTML shell
# ---------------------------------------------------------------------------

def _render_shell(
    *,
    preheader: str,
    headline: str,
    body_html: str,
    cta_url: str,
    cta_label: str,
) -> str:
    """Render the shared branded HTML shell around a body fragment."""
    safe_preheader = _html.escape(preheader or "")
    safe_headline = _html.escape(headline)
    safe_cta_label = _html.escape(cta_label)
    safe_cta_url = _html.escape(cta_url, quote=True)
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>{_html.escape(_BRAND)}</title>
  </head>
  <body style="margin:0;padding:0;background:{_COLOR_BG};font-family:{_FONT_STACK};color:{_COLOR_BODY};">
    <span style="display:none !important;visibility:hidden;opacity:0;color:transparent;height:0;width:0;overflow:hidden;">{safe_preheader}</span>
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background:{_COLOR_BG};padding:24px 0;">
      <tr>
        <td align="center">
          <table role="presentation" width="600" cellpadding="0" cellspacing="0" border="0" style="max-width:600px;width:100%;background:{_COLOR_CARD};border:1px solid {_COLOR_BORDER};border-radius:12px;overflow:hidden;">
            <tr>
              <td style="padding:20px 28px 0;">
                <div style="font-size:20px;font-weight:700;letter-spacing:-0.01em;">
                  <span style="color:{_COLOR_BRAND_CLAW};">Claw</span><span style="color:{_COLOR_BRAND_MEETS};">Meets</span>
                </div>
              </td>
            </tr>
            <tr>
              <td style="padding:16px 28px 8px;">
                <h1 style="margin:0;font-size:20px;line-height:1.35;color:{_COLOR_HEADING};font-weight:600;">{safe_headline}</h1>
              </td>
            </tr>
            <tr>
              <td style="padding:4px 28px 8px;color:{_COLOR_BODY};font-size:15px;line-height:1.55;">
                {body_html}
              </td>
            </tr>
            <tr>
              <td style="padding:16px 28px 24px;">
                <a href="{safe_cta_url}" style="display:inline-block;background:{_COLOR_ACCENT};color:#ffffff;text-decoration:none;padding:10px 18px;border-radius:8px;font-weight:600;font-size:14px;">{safe_cta_label}</a>
              </td>
            </tr>
            <tr>
              <td style="padding:16px 28px 24px;border-top:1px solid {_COLOR_BORDER};color:{_COLOR_MUTED};font-size:12px;line-height:1.5;">
                You received this because you have notifications enabled for your ClawMeets account.
                <br>
                <a href="{_html.escape(cta_url.split('/app/')[0])}/app" style="color:{_COLOR_MUTED};text-decoration:underline;">Open ClawMeets</a>
                &nbsp;&middot;&nbsp;
                <a href="{_html.escape(cta_url.split('/app/')[0])}/app/settings" style="color:{_COLOR_MUTED};text-decoration:underline;">Notification settings</a>
              </td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>"""


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_assistant_message_email(
    *,
    username: str,
    assistant_name: str,
    project: Project,
    chatroom_name: str,
    content: str,
    app_base_url: str,
) -> tuple[str, str, str]:
    """Assistant (coordinator) posted a message to the user-communication room."""
    display_name = _humanize_agent_name(assistant_name, username)
    preview = _first_sentence(content, max_len=110)
    subject = (
        f"{display_name}: {preview}"
        if preview
        else f"New message from {display_name} in \u201c{project.name}\u201d"
    )
    cta_url = _chatroom_url(app_base_url, project.id, chatroom_name)

    body_html = (
        f"<p style=\"margin:0 0 12px;color:{_COLOR_BODY};line-height:1.55;\">"
        f"Hi {_html.escape(username)} \U0001F44B &mdash; your assistant "
        f"<strong>{_html.escape(display_name)}</strong> left a note in project "
        f"<strong>{_html.escape(project.name)}</strong>."
        f"</p>"
        f"<div style=\"border-left:3px solid {_COLOR_BRAND_MEETS};background:#f3f4f6;"
        f"padding:12px 16px;border-radius:4px;margin:12px 0;\">"
        f"{_render_markdown_html(content) or '<em>(no content)</em>'}"
        f"</div>"
    )
    html = _render_shell(
        preheader=preview or f"New update in {project.name}",
        headline=f"New message from {display_name}",
        body_html=body_html,
        cta_url=cta_url,
        cta_label="Open chatroom",
    )
    plain = (
        f"Hi {username},\n\n"
        f"Your assistant \"{display_name}\" left a note in project \"{project.name}\".\n\n"
        f"{_strip_markdown(content) or '(no content)'}\n\n"
        f"Open: {cta_url}\n\n"
        f"— {_BRAND}\n"
    )
    return subject, plain, html


def build_dm_response_email(
    *,
    username: str,
    agent_name: str,
    project: Project,
    chatroom_name: str,
    content: str,
    app_base_url: str,
) -> tuple[str, str, str]:
    """DM chatroom batch completed — the agent responded to the user's DM."""
    display_name = _humanize_agent_name(agent_name, username)
    preview = _first_sentence(content, max_len=110)
    subject = (
        f"{display_name}: {preview}"
        if preview
        else f"{display_name} replied to your DM"
    )
    cta_url = _chatroom_url(app_base_url, project.id, chatroom_name)

    body_html = (
        f"<p style=\"margin:0 0 12px;color:{_COLOR_BODY};line-height:1.55;\">"
        f"Hi {_html.escape(username)} \U0001F44B &mdash; "
        f"<strong>{_html.escape(display_name)}</strong> replied to your direct message."
        f"</p>"
        f"<div style=\"border-left:3px solid {_COLOR_BRAND_MEETS};background:#f3f4f6;"
        f"padding:12px 16px;border-radius:4px;margin:12px 0;\">"
        f"{_render_markdown_html(content) or '<em>(no content)</em>'}"
        f"</div>"
    )
    html = _render_shell(
        preheader=preview or f"{display_name} replied to your DM",
        headline=f"{display_name} replied",
        body_html=body_html,
        cta_url=cta_url,
        cta_label="View conversation",
    )
    plain = (
        f"Hi {username},\n\n"
        f"{display_name} replied to your direct message.\n\n"
        f"{_strip_markdown(content) or '(no content)'}\n\n"
        f"Open: {cta_url}\n\n"
        f"— {_BRAND}\n"
    )
    return subject, plain, html


def build_project_completed_email(
    *,
    username: str,
    project: Project,
    app_base_url: str,
) -> tuple[str, str, str]:
    """Project marked as completed."""
    subject = f"\u201c{project.name}\u201d is complete"
    cta_url = _project_url(app_base_url, project.id)

    body_html = (
        f"<p style=\"margin:0 0 12px;color:{_COLOR_BODY};line-height:1.55;\">"
        f"Hi {_html.escape(username)} \U0001F44B &mdash; your project "
        f"<strong>{_html.escape(project.name)}</strong> has been marked complete."
        f"</p>"
        f"<p style=\"margin:12px 0;color:{_COLOR_MUTED};font-size:14px;line-height:1.55;\">"
        f"All deliverables are available in the chatrooms. You can review the final "
        f"state and any artifacts from the project page."
        f"</p>"
    )
    html = _render_shell(
        preheader=f"{project.name} \u2014 all deliverables are ready",
        headline=f"{project.name} is complete",
        body_html=body_html,
        cta_url=cta_url,
        cta_label="View project",
    )
    plain = (
        f"Hi {username},\n\n"
        f"Your project \"{project.name}\" has been marked complete.\n\n"
        f"All deliverables are available in the chatrooms. Open the project to review.\n\n"
        f"Open: {cta_url}\n\n"
        f"— {_BRAND}\n"
    )
    return subject, plain, html
