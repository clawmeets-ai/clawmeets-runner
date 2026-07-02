# SPDX-License-Identifier: MIT
"""
clawmeets/utils/email.py
Generic email infrastructure: value object (``EmailMessage``), pluggable
``Mailer`` transports (SendGrid + console), and reusable text/HTML
rendering primitives. ClawMeets-specific templates live in
``clawmeets/server/email.py``.
"""
from __future__ import annotations

import html as _html
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Optional

import sendgrid
from markdown_it import MarkdownIt
from pydantic import BaseModel, Field
from sendgrid.helpers.mail import Mail, Email, To, Content, Bcc

logger = logging.getLogger("clawmeets.email")

SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY")
SENDGRID_FROM_EMAIL = os.environ.get("SENDGRID_FROM_EMAIL", "info@clawmeets.ai")
SENDGRID_FROM_NAME = os.environ.get("SENDGRID_FROM_NAME", "ClawMeets AI")

# ---------------------------------------------------------------------------
# Branded HTML shell — palette / typography used by render_notification_html.
# Self-contained: nothing here references agents, projects, or chatrooms.
# ---------------------------------------------------------------------------

_BRAND = "ClawMeets"
_COLOR_HEADING = "#111827"
_COLOR_BODY = "#1f2937"
_COLOR_MUTED = "#6b7280"
_COLOR_ACCENT = "#9333ea"        # tailwind purple-600
_COLOR_ACCENT_HOVER = "#7e22ce"  # tailwind purple-700
_COLOR_BRAND_CLAW = "#2d2272"
_COLOR_BRAND_MEETS = "#7c3aed"
_COLOR_BG = "#f9fafb"
_COLOR_CARD = "#ffffff"
_COLOR_BORDER = "#e5e7eb"
_FONT_STACK = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
    "'Helvetica Neue', Arial, sans-serif"
)


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
# EmailMessage — value object + generic rendering primitives
# ---------------------------------------------------------------------------

class EmailMessage(BaseModel):
    to_email: str
    subject: str
    body: str
    html_body: Optional[str] = None
    bcc: list[str] = Field(default_factory=list)
    log_label: str = "Email"

    # ------------------------------------------------------------------
    # Text utilities
    # ------------------------------------------------------------------

    @staticmethod
    def strip_markdown(text: str) -> str:
        """Strip markdown syntax for plain-text email bodies."""
        if not text:
            return ""
        text = re.sub(r"```[\w-]*\n?", "", text)
        text = re.sub(r"```", "", text)
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"(?<!\*)\*(?!\s)([^*]+?)(?<!\s)\*(?!\*)", r"\1", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
        return text.strip()

    @staticmethod
    def first_sentence(text: str, max_len: int = 90, min_len: int = 40) -> str:
        """Return a short, sentence-ish fragment for a subject preview.

        Prefers sentence boundaries that give at least ``min_len`` chars,
        so a leading ``"Got it."`` doesn't swallow the whole preview.
        Falls back to the first line, then to a clean ellipsized cut.
        """
        clean = EmailMessage.strip_markdown(text)
        first_para = re.split(r"\n\s*\n", clean, maxsplit=1)[0]
        flat = re.sub(r"\s+", " ", first_para).strip()
        if not flat:
            return ""
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
            if len(flat) <= max_len:
                candidate = flat
        if candidate is None:
            if len(flat) <= max_len:
                candidate = flat
            else:
                candidate = flat[: max_len - 1].rstrip() + "…"
        return candidate

    @staticmethod
    def render_markdown_html(text: str) -> str:
        """Render markdown to HTML via markdown-it-py."""
        if not text:
            return ""
        md = MarkdownIt("commonmark", {"breaks": True, "linkify": True})
        return md.render(text).strip()

    @staticmethod
    def render_notification_html(
        *,
        preheader: str,
        headline: str,
        intro_html: str,
        quote_content_md: Optional[str] = None,
        secondary_text: Optional[str] = None,
        cta_url: str,
        cta_label: str,
    ) -> str:
        """Render a notification email's full HTML.

        ``intro_html`` is inserted verbatim and must already be escaped /
        contain trusted inline tags. ``quote_content_md`` is rendered
        through markdown into a brand-bordered quote block.
        ``secondary_text`` is plain text shown as a muted follow-up
        paragraph.
        """
        parts: list[str] = [
            f'<p style="margin:0 0 12px;color:{_COLOR_BODY};line-height:1.55;">'
            f'{intro_html}'
            f'</p>'
        ]
        if quote_content_md is not None:
            rendered = EmailMessage.render_markdown_html(quote_content_md) or "<em>(no content)</em>"
            parts.append(
                f'<div style="border-left:3px solid {_COLOR_BRAND_MEETS};background:#f3f4f6;'
                f'padding:12px 16px;border-radius:4px;margin:12px 0;">'
                f'{rendered}'
                f'</div>'
            )
        if secondary_text:
            parts.append(
                f'<p style="margin:12px 0;color:{_COLOR_MUTED};font-size:14px;line-height:1.55;">'
                f'{_html.escape(secondary_text)}'
                f'</p>'
            )
        return _render_shell(
            preheader=preheader,
            headline=headline,
            body_html="".join(parts),
            cta_url=cta_url,
            cta_label=cta_label,
        )


# ---------------------------------------------------------------------------
# Mailer transports
# ---------------------------------------------------------------------------

class Mailer(ABC):
    @abstractmethod
    async def send(self, message: EmailMessage) -> None: ...


class SendGridMailer(Mailer):
    def __init__(self, api_key: str, from_email: str, from_name: str) -> None:
        self._api_key = api_key
        self._from_email = from_email
        self._from_name = from_name

    async def send(self, message: EmailMessage) -> None:
        sg = sendgrid.SendGridAPIClient(api_key=self._api_key)
        mail = Mail(
            from_email=Email(self._from_email, self._from_name),
            to_emails=To(message.to_email),
            subject=message.subject,
            plain_text_content=Content("text/plain", message.body),
        )
        if message.html_body:
            mail.add_content(Content("text/html", message.html_body))
        for bcc_addr in message.bcc:
            mail.add_bcc(Bcc(bcc_addr))
        try:
            response = sg.send(mail)
            logger.info(
                f"{message.log_label} sent to {message.to_email} "
                f"(status={response.status_code})"
            )
        except Exception as e:
            logger.error(
                f"Failed to send {message.log_label.lower()} to {message.to_email}: {e}"
            )
            raise


class ConsoleMailer(Mailer):
    async def send(self, message: EmailMessage) -> None:
        logger.info(
            f"[EMAIL FALLBACK] {message.log_label} for {message.to_email}"
        )
        print(f"\n--- {message.log_label} (SendGrid not configured) ---")
        print(f"To: {message.to_email}")
        for bcc_addr in message.bcc:
            print(f"BCC: {bcc_addr}")
        print(f"Subject: {message.subject}")
        print(f"Body: {message.body}")
        if message.html_body:
            print(f"[HTML body suppressed in console — {len(message.html_body)} chars]")
        print("-" * (len(message.log_label) + 32) + "\n")


def get_mailer() -> Mailer:
    if SENDGRID_API_KEY:
        return SendGridMailer(SENDGRID_API_KEY, SENDGRID_FROM_EMAIL, SENDGRID_FROM_NAME)
    return ConsoleMailer()
