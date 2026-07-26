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
import hashlib
import os
import re
import unicodedata
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

from clawmeets.integrations._config_resolve import resolve_skill_config_path
from clawmeets.integrations._sync_warehouse import (
    SyncBudget,
    run_slice_sync,
    run_slices,
    utcnow_iso,
    write_howto,
)
from clawmeets.utils.file_io import FileUtil
from clawmeets.utils.jsonc import parse_jsonc

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


def build_service(token_path: Path):
    """Build a Gmail API client backed by the cached token at ``token_path``."""
    from googleapiclient.discovery import build
    from clawmeets.integrations.auth.google_oauth import load_credentials

    creds = load_credentials(token_path, SCOPES)
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def load_config(config_file: str) -> tuple[Optional[dict], Optional[str]]:
    """Read the gmail-skill config from the supplied path, or self-resolve.

    Falls back to ``$CLAWMEETS_AGENT_DIR/skill-hub/configs/gmail.json``
    when the caller didn't pass a path — the runner writes per-skill
    configs there.

    Returns ``(cfg, err)``:
      - ``(dict, None)`` on a successfully-parsed dict-shaped config
      - ``(None, None)`` when the path is empty, missing, or empty file
        (caller treats as a clean noop — fresh installs don't error)
      - ``(None, "...")`` when the file is malformed JSONC or its root
        isn't a dict
    """
    config_file = resolve_skill_config_path("gmail", config_file)
    if not config_file:
        return None, None
    path = Path(config_file).expanduser()
    if not path.exists():
        return None, None
    try:
        raw = path.read_text()
    except OSError as exc:
        return None, f"could not read config file {path}: {exc}"
    if not raw.strip():
        return None, None
    try:
        cfg = parse_jsonc(raw)
    except Exception as exc:
        return None, f"config file is not valid JSON: {exc}"
    if not isinstance(cfg, dict):
        return None, "config file must contain a JSON object"
    return cfg, None


def _message_id_hash(rfc822_message_id: str, fallback_id: str) -> str:
    """Stable, filesystem-safe message identity. Prefers RFC822 Message-Id
    (sha256, truncated to 24 hex) for cross-source dedup with the IMAP
    mailbox integration; falls back to Gmail's own ``id`` when Message-Id
    is missing (legacy / programmatically-injected messages sometimes lack
    one)."""
    if rfc822_message_id:
        return hashlib.sha256(
            rfc822_message_id.encode("utf-8", errors="replace")
        ).hexdigest()[:24]
    return fallback_id


_FILENAME_BAD_CHARS = re.compile(r"[\x00-\x1f\x7f/\\:*?<>|\"]+")
_FILENAME_WS = re.compile(r"\s+")


def _safe_filename_segment(s: str, *, fallback: str = "file", max_len: int = 80) -> str:
    if not s:
        return fallback
    s = unicodedata.normalize("NFC", s)
    s = _FILENAME_BAD_CHARS.sub("-", s)
    s = _FILENAME_WS.sub(" ", s).strip()
    s = s.strip(".-_ ")
    if len(s) > max_len:
        s = s[:max_len].rstrip(".-_ ")
    return s or fallback


def _extract_attachments(payload: dict) -> list[dict]:
    """Walk a Gmail full-message payload recursively; return one dict per
    attachment part (parts that carry ``body.attachmentId`` + a ``filename``).
    Inline parts without filenames are skipped.
    """
    out: list[dict] = []

    def walk(part: dict) -> None:
        body = part.get("body") or {}
        attachment_id = body.get("attachmentId")
        filename = part.get("filename") or ""
        if attachment_id and filename:
            out.append({
                "part_id": part.get("partId") or "",
                "filename": filename,
                "content_type": part.get("mimeType") or "application/octet-stream",
                "size": int(body.get("size") or 0),
                "attachment_id": attachment_id,
                "path": None,
                "downloaded_at": None,
            })
        for sub in part.get("parts") or []:
            walk(sub)

    walk(payload or {})
    return out


def _download_envelope_attachments(
    *,
    envelope: dict,
    gmail_svc,
    slice_cfg: dict,
    dwh_root: Path,
    files_root: Path,
) -> None:
    raw_cap = slice_cfg.get("max_attachment_size_mb")
    try:
        cap = (int(raw_cap) if raw_cap is not None else 25) * 1024 * 1024
    except (TypeError, ValueError):
        cap = 25 * 1024 * 1024
    msg_id = envelope.get("id") or ""
    if not msg_id:
        return
    att_dir = files_root / msg_id
    for att in envelope.get("attachments", []):
        if att.get("size", 0) > cap:
            continue
        try:
            resp = gmail_svc.users().messages().attachments().get(
                userId="me", messageId=msg_id, id=att["attachment_id"],
            ).execute()
            url_safe_b64 = resp.get("data", "")
            if not url_safe_b64:
                att["download_error"] = "empty attachment payload"
                continue
            raw = base64.urlsafe_b64decode(url_safe_b64 + "==")
            safe = _safe_filename_segment(
                att.get("filename") or f"part-{att.get('part_id') or '0'}"
            )
            pid = (att.get("part_id") or "0").replace(".", "_")
            out_path = att_dir / f"{pid}-{safe}"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_name(out_path.name + ".tmp")
            tmp.write_bytes(raw)
            os.replace(tmp, out_path)
            att["path"] = str(out_path.relative_to(dwh_root))
            att["downloaded_at"] = utcnow_iso()
        except Exception as exc:
            att["download_error"] = f"{type(exc).__name__}: {exc}"[:200]


def _extract_body_parts(payload: dict) -> tuple[Optional[bytes], Optional[bytes]]:
    html_bytes: Optional[bytes] = None
    text_bytes: Optional[bytes] = None

    def walk(part: dict) -> None:
        nonlocal html_bytes, text_bytes
        body = part.get("body") or {}
        data = body.get("data")
        mime = part.get("mimeType") or ""
        if data and not body.get("attachmentId"):
            try:
                decoded = base64.urlsafe_b64decode(data + "==")
            except Exception:
                decoded = b""
            if mime == "text/html" and html_bytes is None:
                html_bytes = decoded
            elif mime == "text/plain" and text_bytes is None:
                text_bytes = decoded
        for sub in part.get("parts") or []:
            walk(sub)

    walk(payload or {})
    return html_bytes, text_bytes


def _download_envelope_body(
    *,
    envelope: dict,
    html_bytes: Optional[bytes],
    text_bytes: Optional[bytes],
    slice_cfg: dict,
    dwh_root: Path,
    files_root: Path,
) -> None:
    if not slice_cfg.get("download_body", True):
        return
    raw_cap = slice_cfg.get("max_body_size_mb")
    try:
        cap = (int(raw_cap) if raw_cap is not None else 5) * 1024 * 1024
    except (TypeError, ValueError):
        cap = 5 * 1024 * 1024
    if not (html_bytes or text_bytes):
        return
    msg_id = envelope.get("id") or ""
    if not msg_id:
        return
    msg_dir = files_root / msg_id
    msg_dir.mkdir(parents=True, exist_ok=True)
    for data, suffix, key in (
        (html_bytes, "html", "body_html_path"),
        (text_bytes, "txt",  "body_text_path"),
    ):
        if not data:
            continue
        if len(data) > cap:
            continue
        out_path = msg_dir / f"body.{suffix}"
        tmp = out_path.with_name(out_path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, out_path)
        envelope[key] = str(out_path.relative_to(dwh_root))


def _build_envelope(full_msg: dict, slice_name: str) -> tuple[Optional[dict], Optional[str]]:
    ts_ms = int(full_msg.get("internalDate", "0"))
    if ts_ms <= 0:
        return None, None
    ts_iso = datetime.fromtimestamp(ts_ms / 1000, timezone.utc).isoformat()
    payload = full_msg.get("payload") or {}
    headers = {h["name"]: h["value"] for h in payload.get("headers", [])}
    rfc822_id = headers.get("Message-Id") or headers.get("Message-ID") or ""
    html_bytes, text_bytes = _extract_body_parts(payload)
    envelope = {
        "ts": ts_iso,
        "id": full_msg.get("id"),
        "thread_id": full_msg.get("threadId"),
        "message_id": rfc822_id,
        "message_id_hash": _message_id_hash(rfc822_id, full_msg.get("id", "")),
        "labels": full_msg.get("labelIds", []),
        "headers": headers,
        "snippet": full_msg.get("snippet", ""),
        "body_html_path": None,
        "body_text_path": None,
        "attachments": _extract_attachments(payload),
        "raw": full_msg,
        "slice": slice_name,
        "__html_bytes": html_bytes,
        "__text_bytes": text_bytes,
    }
    return envelope, ts_iso


def _err_result(name: str, msg: str) -> dict:
    return {"name": name, "rows_written": 0, "watermarks": None,
            "has_more": False, "error": msg}


def _sync_one_slice(
    *,
    gmail_svc,
    slice_cfg: dict,
    dwh_dir: str,
    budget: SyncBudget,
) -> dict:
    raw_name = slice_cfg.get("name") if isinstance(slice_cfg, dict) else None
    if not isinstance(raw_name, str) or not raw_name.strip():
        return _err_result("<unnamed>", "slice config missing required 'name' field")
    try:
        name = FileUtil.validate_fs_name(raw_name)
    except ValueError as exc:
        return _err_result(raw_name, f"invalid slice name {raw_name!r}: {exc}")

    label = slice_cfg.get("label")
    if not isinstance(label, str) or not label.strip():
        return _err_result(
            name, "slice config missing required 'label' field (Gmail label name)")
    extra_query = slice_cfg.get("query") or ""
    if not isinstance(extra_query, str):
        extra_query = ""

    dwh_root = Path(dwh_dir).expanduser().resolve()
    source = f"gmail/{name}"
    base = dwh_root / "raw" / source
    files_root = base / "files"

    howto_err = write_howto(slice_cfg.get("howto"), snapshot_dir=base)
    if howto_err:
        return _err_result(name, howto_err)

    def fetch(window_start: str, window_end: str, bud: SyncBudget, emit) -> bool:
        start_epoch = int(datetime.fromisoformat(window_start.replace("Z", "+00:00")).timestamp())
        end_epoch = int(datetime.fromisoformat(window_end.replace("Z", "+00:00")).timestamp())
        q_parts = [f"label:{label}", f"after:{start_epoch}", f"before:{end_epoch}"]
        if extra_query.strip():
            q_parts.append(extra_query.strip())
        q = " ".join(q_parts)

        page_token: Optional[str] = None
        while True:
            if bud.should_stop():
                return True
            resp = gmail_svc.users().messages().list(
                userId="me", q=q, maxResults=500, pageToken=page_token,
            ).execute()
            for m in resp.get("messages", []):
                if bud.should_stop():
                    return True
                full = gmail_svc.users().messages().get(
                    userId="me", id=m["id"], format="full",
                ).execute()
                envelope, _ts_iso = _build_envelope(full, name)
                if envelope is None:
                    continue
                if slice_cfg.get("download_attachments") and envelope["attachments"]:
                    _download_envelope_attachments(
                        envelope=envelope, gmail_svc=gmail_svc, slice_cfg=slice_cfg,
                        dwh_root=dwh_root, files_root=files_root,
                    )
                _download_envelope_body(
                    envelope=envelope,
                    html_bytes=envelope.pop("__html_bytes", None),
                    text_bytes=envelope.pop("__text_bytes", None),
                    slice_cfg=slice_cfg, dwh_root=dwh_root, files_root=files_root,
                )
                bud.rows_written += 1
                emit(envelope)
            page_token = resp.get("nextPageToken")
            if not page_token:
                return False

    return run_slice_sync(
        source=source, dwh_dir=dwh_dir, budget=budget, fetch=fetch,
        id_field="id", ts_field="ts", start_at=slice_cfg.get("start_at"),
        snapshot_fmt="ndjson",
    )


# ---------------------------------------------------------------------------
# Interactive tool bodies (search / get / labels / attachment / send)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Sync entry point
# ---------------------------------------------------------------------------


def sync_to_warehouse(
    svc,
    dwh_dir: str,
    config_file: str = "",
    max_runtime_seconds: int = 1500,
) -> dict:
    """Sync new / updated Gmail messages into the personal data warehouse.

    Triggered by ``<!-- clawmeets:gmail-sync-trigger -->`` in a DM. Named-slice
    model: the config carries a ``labels_to_sync`` list, each entry a
    ``{name, label, query?, merge_policy?, ...}`` dict.
    """
    window_end = utcnow_iso()
    cfg, err = load_config(config_file)
    if err:
        return {
            "status": "error", "source": "gmail", "rows_written": 0,
            "window": [window_end, window_end], "watermarks": None,
            "has_more": False, "error": err, "per_slice": {},
        }
    if cfg is None:
        return {
            "status": "noop", "source": "gmail", "rows_written": 0,
            "window": [window_end, window_end], "watermarks": None,
            "has_more": False, "error": None, "per_slice": {},
        }

    slices = cfg.get("labels_to_sync")
    if not isinstance(slices, list) or len(slices) == 0:
        return {
            "status": "noop", "source": "gmail", "rows_written": 0,
            "window": [window_end, window_end], "watermarks": None,
            "has_more": False, "error": None, "per_slice": {},
        }

    budget = SyncBudget(max_runtime_seconds)
    return run_slices(
        source_family="gmail", slices=slices, budget=budget,
        dwh_dir=dwh_dir,
        run_one=lambda sc: _sync_one_slice(
            gmail_svc=svc, slice_cfg=sc, dwh_dir=dwh_dir, budget=budget),
    )
