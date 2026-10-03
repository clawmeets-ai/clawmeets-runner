# SPDX-License-Identifier: MIT
"""
clawmeets/integrations/gmail/_lib.py

Pure-Python Gmail integration. Drives ``clawmeets gmail <subcmd>`` via
``clawmeets/cli_gmail.py``; the paired skill ``skills/gmail/SKILL.md``
teaches the LLM when to shell which subcommand.

Carried over verbatim from the MCP-era ``clawmeets/mcp/servers/gmail_server.py``
minus the ``FastMCP`` wrapping and the ``CLAWMEETS_GMAIL_TOKEN_FILE`` env-var
indirection — every function now takes ``token_path: Path`` explicitly so the
CLI can resolve it via ``clawmeets.integrations._config_resolve``.
"""
from __future__ import annotations

import base64
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


def build_service(token_path: Path):
    """Build a Gmail API client backed by the cached token at ``token_path``."""
    from googleapiclient.discovery import build
    from clawmeets.integrations.auth.google_oauth import load_credentials

    creds = load_credentials(token_path, SCOPES)
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


# ---------------------------------------------------------------------------
# Tool bodies (search / get / labels / attachment / send / archive)
# ---------------------------------------------------------------------------

INBOX_LABEL = "INBOX"
BATCH_MODIFY_MAX_IDS = 1000  # Gmail API hard limit per batchModify call


def search_messages(svc, query: str, max_results: int = 20) -> list[dict]:
    """Search Gmail using the standard query syntax.

    Returns a list of ``{id, thread_id, snippet, from, subject, date}``.
    """
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


def get_message(svc, message_id: str, format: str = "full") -> dict:
    """Fetch a full Gmail message. ``format`` is 'full' (default) or 'metadata'."""
    return svc.users().messages().get(
        userId="me", id=message_id, format=format,
    ).execute()


def list_labels(svc) -> list[dict]:
    """List all Gmail labels on the account."""
    resp = svc.users().labels().list(userId="me").execute()
    return [{"id": lbl["id"], "name": lbl["name"]} for lbl in resp.get("labels", [])]


def get_attachment(svc, message_id: str, attachment_id: str) -> dict:
    """Fetch an attachment whose body was stubbed by ``get_message(format="full")``.

    Returns ``{filename, mime_type, size, data_b64}`` where ``data_b64`` is
    **standard** (not URL-safe) base64.
    """
    att = svc.users().messages().attachments().get(
        userId="me", messageId=message_id, id=attachment_id,
    ).execute()
    url_safe_b64 = att.get("data", "")
    size = att.get("size", 0)

    if url_safe_b64:
        raw_bytes = base64.urlsafe_b64decode(url_safe_b64 + "==")
        data_b64 = base64.b64encode(raw_bytes).decode()
    else:
        data_b64 = ""

    msg = svc.users().messages().get(
        userId="me", id=message_id, format="full",
    ).execute()

    def _find_part(parts: list[dict]) -> Optional[dict]:
        for p in parts or []:
            body = p.get("body") or {}
            if body.get("attachmentId") == attachment_id:
                return p
            child = _find_part(p.get("parts") or [])
            if child is not None:
                return child
        return None

    part = _find_part([msg.get("payload") or {}])
    if part is None:
        return {
            "filename": f"part-{attachment_id[:8]}",
            "mime_type": "application/octet-stream",
            "size": size,
            "data_b64": data_b64,
        }

    filename = part.get("filename") or f"part-{part.get('partId', 'unknown')}"
    mime_type = part.get("mimeType", "application/octet-stream")
    return {
        "filename": filename,
        "mime_type": mime_type,
        "size": size,
        "data_b64": data_b64,
    }


def send_message(
    svc,
    to: str,
    subject: str,
    body: str,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
) -> dict:
    """Send a plaintext email. Returns the new message's id + thread_id."""
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


def _modify_chunk(svc, ids: list[str], *, undo: bool = False) -> list[dict]:
    """Apply the INBOX label change to one chunk of <= ``BATCH_MODIFY_MAX_IDS``.

    ``batchModify`` is all-or-nothing per call and returns an empty body, so a
    chunk that raises is retried id-by-id via ``messages.modify`` to attribute
    the error to the offending id(s) instead of dropping the whole chunk.
    Returns the per-id failure records — empty list on full success. Never
    raises.
    """
    key = "addLabelIds" if undo else "removeLabelIds"
    try:
        svc.users().messages().batchModify(
            userId="me", body={"ids": list(ids), key: [INBOX_LABEL]},
        ).execute()
        return []
    except Exception:
        pass

    failures: list[dict] = []
    for message_id in ids:
        try:
            svc.users().messages().modify(
                userId="me", id=message_id, body={key: [INBOX_LABEL]},
            ).execute()
        except Exception as exc:
            failures.append({
                "id": message_id,
                "error": f"{type(exc).__name__}: {exc}"[:300],
            })
    return failures


def archive_messages(svc, message_ids: list[str], *, undo: bool = False) -> dict:
    """Archive messages by removing the ``INBOX`` label (Gmail has no archive
    endpoint — archiving *is* dropping that label). With ``undo=True`` the
    label is added back, moving the messages to the inbox.

    Idempotent: archiving an already-archived id is a silent no-op success, so
    the call is safe to retry. Nothing is deleted — archived mail stays in All
    Mail. Ids are de-duplicated (order preserved) and chunked at
    ``BATCH_MODIFY_MAX_IDS``; an empty list does zero API calls.

    Returns ``{"action", "requested", "archived": [id...],
    "failed": [{"id", "error"}...]}``.
    """
    seen: set[str] = set()
    ids: list[str] = []
    for message_id in message_ids or []:
        if message_id and message_id not in seen:
            seen.add(message_id)
            ids.append(message_id)

    action = "unarchive" if undo else "archive"
    failures: list[dict] = []
    for start in range(0, len(ids), BATCH_MODIFY_MAX_IDS):
        failures.extend(
            _modify_chunk(svc, ids[start:start + BATCH_MODIFY_MAX_IDS], undo=undo)
        )

    failed_ids = {f["id"] for f in failures}
    return {
        "action": action,
        "requested": len(ids),
        "archived": [i for i in ids if i not in failed_ids],
        "failed": failures,
    }
