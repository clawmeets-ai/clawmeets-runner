# SPDX-License-Identifier: MIT
"""
clawmeets/mcp/servers/gmail_server.py

Gmail MCP server. Exposes search, read, and send as MCP tools, backed by
google-api-python-client. Runs as a stdio subprocess of Claude Code.

Reads the OAuth token from the path in CLAWMEETS_GMAIL_TOKEN_FILE.
"""
from __future__ import annotations

import base64
import os
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


def _token_path() -> Path:
    p = os.environ.get("CLAWMEETS_GMAIL_TOKEN_FILE")
    if not p:
        raise RuntimeError(
            "CLAWMEETS_GMAIL_TOKEN_FILE is not set. The Gmail MCP server is "
            "expected to be launched by the clawmeets runner, which sets this "
            "via the mcps/gmail/mcp.json launch spec."
        )
    return Path(p)


def _service():
    from googleapiclient.discovery import build
    from clawmeets.mcp.auth.google_oauth import load_credentials

    creds = load_credentials(_token_path(), SCOPES)
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def main() -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(
            "The `mcp` package is required but missing — the clawmeets runner "
            "should bundle it by default. Try: pip install --upgrade clawmeets"
        ) from exc

    mcp = FastMCP("clawmeets-gmail")

    @mcp.tool()
    def search_messages(query: str, max_results: int = 20) -> list[dict]:
        """Search Gmail using the standard query syntax.

        Returns a list of {id, thread_id, snippet, from, subject, date}.
        """
        svc = _service()
        resp = svc.users().messages().list(
            userId="me", q=query, maxResults=max_results
        ).execute()
        out: list[dict] = []
        for m in resp.get("messages", []):
            full = svc.users().messages().get(
                userId="me", id=m["id"], format="metadata",
                metadataHeaders=["From", "Subject", "Date"],
            ).execute()
            headers = {h["name"]: h["value"] for h in full.get("payload", {}).get("headers", [])}
            out.append({
                "id": full["id"],
                "thread_id": full.get("threadId"),
                "snippet": full.get("snippet", ""),
                "from": headers.get("From", ""),
                "subject": headers.get("Subject", ""),
                "date": headers.get("Date", ""),
            })
        return out

    @mcp.tool()
    def get_message(message_id: str, format: str = "full") -> dict:
        """Fetch a full Gmail message. `format` is 'full' (default) or 'metadata'."""
        svc = _service()
        return svc.users().messages().get(
            userId="me", id=message_id, format=format,
        ).execute()

    @mcp.tool()
    def list_labels() -> list[dict]:
        """List all Gmail labels on the account."""
        svc = _service()
        resp = svc.users().labels().list(userId="me").execute()
        return [{"id": lbl["id"], "name": lbl["name"]} for lbl in resp.get("labels", [])]

    @mcp.tool()
    def send_message(
        to: str,
        subject: str,
        body: str,
        cc: Optional[str] = None,
        bcc: Optional[str] = None,
    ) -> dict:
        """Send a plaintext email. Returns the new message's id + thread_id."""
        svc = _service()
        msg = EmailMessage()
        msg["To"] = to
        msg["Subject"] = subject
        if cc:
            msg["Cc"] = cc
        if bcc:
            msg["Bcc"] = bcc
        msg.set_content(body)
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        sent = svc.users().messages().send(
            userId="me", body={"raw": raw},
        ).execute()
        return {"id": sent.get("id"), "thread_id": sent.get("threadId")}

    mcp.run()


if __name__ == "__main__":
    main()
