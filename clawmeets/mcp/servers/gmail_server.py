# SPDX-License-Identifier: MIT
"""
clawmeets/mcp/servers/gmail_server.py

Gmail MCP server. Exposes search, read, send, and named-slice
``sync_to_warehouse`` as MCP tools, backed by google-api-python-client.
Runs as a stdio subprocess of Claude Code.

Reads the OAuth token from the path in CLAWMEETS_GMAIL_TOKEN_FILE.

Named-slice sync model: each entry in ``labels_to_sync`` (in the per-MCP
config at ``{agent_dir}/mcp-hub/configs/gmail.json``) binds one Gmail
label (and optional extra ``query``) to its own warehouse dataset under
``{dwh_dir}/sources/gmail/<name>/`` and ``{dwh_dir}/merged/gmail/<name>.json``.
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

from clawmeets.mcp.servers._sync_warehouse import (
    SyncBudget,
    _read_state,
    atomic_write_json,
    gc_old_timestamps,
    merge_json_envelopes,
    new_timestamp_dir,
    utcnow_iso,
    validate_merge_policy,
    write_howto,
)
from clawmeets.utils.jsonc import parse_jsonc
from clawmeets.utils.validation import validate_name

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


def _load_config(config_file: str) -> tuple[Optional[dict], Optional[str]]:
    """Read this MCP's config from the file path supplied by the agent.

    Returns ``(cfg, err)``:
      - ``(dict, None)`` on a successfully-parsed dict-shaped config
      - ``(None, None)`` when the file path is empty, missing, or empty
        (caller treats as a clean noop — fresh installs don't error)
      - ``(None, "...")`` when the file is malformed JSONC or its root
        isn't a dict
    """
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
    mailbox MCP; falls back to Gmail's own ``id`` when Message-Id is missing
    (legacy / programmatically-injected messages sometimes lack one)."""
    if rfc822_message_id:
        return hashlib.sha256(
            rfc822_message_id.encode("utf-8", errors="replace")
        ).hexdigest()[:24]
    return fallback_id


_FILENAME_BAD_CHARS = re.compile(r"[\x00-\x1f\x7f/\\:*?<>|\"]+")
_FILENAME_WS = re.compile(r"\s+")


def _safe_filename_segment(s: str, *, fallback: str = "file", max_len: int = 80) -> str:
    """Sanitize a string into a filename segment safe on any common filesystem.

    Copy of ``gdrive_server._safe_filename_segment`` — duplicated here to keep
    each MCP self-contained (single-process invariant).
    """
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

    Inline parts without filenames (tracking pixels, embedded CID images
    without a name header) are skipped — the same convention the rest of
    this MCP uses to distinguish "attachment" from "body / inline glue".
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


def _download_envelope_attachments_gmail(
    *,
    envelope: dict,
    gmail_svc,
    slice_cfg: dict,
    dwh_dir: str,
    slice_name: str,
) -> None:
    """Persist attachment bytes to disk and populate ``path`` /
    ``downloaded_at`` on each ``envelope.attachments[*]`` dict.

    Storage location on disk: ``{dwh_dir}/merged/gmail/<slice>.attachments/
    <gmail_id>/<part_id>-<safe_filename>``. The path lives outside the
    per-run timestamp folder so ``gc_old_timestamps`` retention trim
    doesn't delete attachment bytes the merged envelope still points at.

    ``attachments[i].path`` is recorded **warehouse-relative** (e.g.
    ``merged/gmail/inbox.attachments/<gmail_id>/<file>``) so the merged
    envelope is portable across hosts / sandboxes that mount ``dwh_dir``
    at different absolute paths. Consumers should join against their own
    ``dwh_dir``.

    Skips parts larger than ``slice_cfg.max_attachment_size_mb`` (default
    25 MB). Per-attachment errors are surfaced via a ``download_error``
    field and never raise.
    """
    raw_cap = slice_cfg.get("max_attachment_size_mb")
    try:
        cap = (int(raw_cap) if raw_cap is not None else 25) * 1024 * 1024
    except (TypeError, ValueError):
        cap = 25 * 1024 * 1024
    msg_id = envelope.get("id") or ""
    if not msg_id:
        return
    dwh_root = Path(dwh_dir).expanduser().resolve()
    att_dir = (
        dwh_root / "merged" / "gmail"
        / f"{slice_name}.attachments" / msg_id
    )
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
            # Replace dots in part_id (Gmail uses '0.1.0' for nested) to keep
            # the filename prefix readable.
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
    """Walk a Gmail full-message payload to return decoded
    ``(html_bytes, text_bytes)``. Either or both may be ``None`` if the
    message lacks that variant.

    Selects parts whose ``mimeType`` is ``text/html`` or ``text/plain``
    and whose ``body.data`` is present (i.e. inline bytes — NOT
    ``attachmentId``-referenced parts, which the Gmail API serves
    separately). Prefers the shallowest match per type so multipart
    /alternative trees that wrap the body in nested groups still find it.
    Decodes Gmail's URL-safe base64.
    """
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


def _download_envelope_body_gmail(
    *,
    envelope: dict,
    html_bytes: Optional[bytes],
    text_bytes: Optional[bytes],
    slice_cfg: dict,
    dwh_dir: str,
    slice_name: str,
) -> None:
    """Persist email body to disk; set ``body_{html,text}_path`` on ``envelope``.

    Storage location: ``{dwh_dir}/merged/gmail/<slice>.attachments/
    <gmail_id>/body.{html,txt}`` — same per-message dir as attachment
    bytes so consumers can re-open both with one path prefix.

    Paths recorded **warehouse-relative** (matching attachments). Gated by
    ``slice_cfg.download_body`` (default ``True``); capped by
    ``slice_cfg.max_body_size_mb`` (default 5 MB).
    """
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
    dwh_root = Path(dwh_dir).expanduser().resolve()
    msg_dir = (
        dwh_root / "merged" / "gmail"
        / f"{slice_name}.attachments" / msg_id
    )
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
    """Convert a Gmail full-format message dict into the warehouse envelope.

    Returns ``(envelope, ts_iso)`` or ``(None, None)`` if the message lacks
    ``internalDate`` (skip).
    """
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
        "body_html_path": None,    # populated by _download_envelope_body_gmail
        "body_text_path": None,    # populated by _download_envelope_body_gmail
        "attachments": _extract_attachments(payload),
        "raw": full_msg,
        "slice": slice_name,
        # Sidechannel — popped by _sync_one_slice before persisting.
        "__html_bytes": html_bytes,
        "__text_bytes": text_bytes,
    }
    return envelope, ts_iso


def _sync_one_slice(
    *,
    gmail_svc,
    slice_cfg: dict,
    dwh_dir: str,
    budget: SyncBudget,
    window_end: str,
) -> dict:
    """Sync a single named gmail slice; return its per-slice summary.

    Mirrors ``mailbox_server._sync_one_slice``: owns its own
    ``sync-state.json`` under ``{dwh_dir}/sources/gmail/<name>/`` and
    advances its watermark independently of sibling slices. The shared
    ``gmail_svc`` and ``budget`` are passed in so all slices in one call
    share one OAuth handshake and one wall-clock budget.
    """
    raw_name = slice_cfg.get("name") if isinstance(slice_cfg, dict) else None
    if not isinstance(raw_name, str) or not raw_name.strip():
        return {
            "name": "<unnamed>",
            "rows_written": 0, "watermarks": None,
            "has_more": False,
            "error": "slice config missing required 'name' field",
        }
    try:
        name = validate_name(raw_name)
    except ValueError as exc:
        return {
            "name": raw_name,
            "rows_written": 0, "watermarks": None,
            "has_more": False,
            "error": f"invalid slice name {raw_name!r}: {exc}",
        }

    label = slice_cfg.get("label")
    if not isinstance(label, str) or not label.strip():
        return {
            "name": name,
            "rows_written": 0, "watermarks": None,
            "has_more": False,
            "error": "slice config missing required 'label' field (Gmail label name)",
        }
    extra_query = slice_cfg.get("query") or ""
    if not isinstance(extra_query, str):
        extra_query = ""

    merge_policy, upsert_id_column, merge_err = validate_merge_policy(slice_cfg)
    if merge_err:
        return {
            "name": name,
            "rows_written": 0, "watermarks": None,
            "has_more": False,
            "error": merge_err,
        }

    dwh_root = Path(dwh_dir).expanduser().resolve()
    source = f"gmail/{name}"
    source_dir = dwh_root / "sources" / source
    state_path = source_dir / "sync-state.json"
    merged_path = dwh_root / "merged" / "gmail" / f"{name}.json"

    howto_err = write_howto(
        slice_cfg.get("howto"),
        source_dir=source_dir,
        merged_path=merged_path,
    )
    if howto_err:
        return {
            "name": name,
            "rows_written": 0, "watermarks": None,
            "has_more": False,
            "error": howto_err,
        }

    state = _read_state(state_path, source)

    if merge_policy != "replace" and state.get("last_sync_at") is None:
        start_at_raw = slice_cfg.get("start_at")
        if isinstance(start_at_raw, str) and start_at_raw.strip():
            start_at = start_at_raw.strip()
            try:
                datetime.fromisoformat(start_at.replace("Z", "+00:00"))
            except ValueError as exc:
                return {
                    "name": name,
                    "rows_written": 0, "watermarks": None,
                    "has_more": False,
                    "error": (
                        f"invalid start_at {start_at_raw!r} for slice {name!r}: "
                        f"{exc} (must be ISO-8601, e.g. '2024-01-01' or "
                        f"'2024-01-01T00:00:00Z')"
                    ),
                }
            state["low_watermark"] = start_at
            state["high_watermark"] = start_at

    low = state.get("low_watermark") or utcnow_iso()
    high = state.get("high_watermark") or utcnow_iso()

    if merge_policy == "replace":
        # Replace mode: pull every message matching the label + extra_query
        # each run; merge step rewrites the consolidated JSON. Use epoch as a
        # defensive lower bound for the `after:` query so the request still
        # parses.
        window_start: Optional[str] = "1970-01-01T00:00:00+00:00"
    else:
        window_start = max(low, high)
        if window_start >= window_end:
            return {
                "name": name,
                "rows_written": 0,
                "watermarks": {"low": low, "high": high},
                "has_more": False, "error": None,
            }

    timestamp_dir = new_timestamp_dir(source_dir)
    rows_written_start = budget.rows_written
    latest_seen = window_start or low
    has_more = False

    try:
        epoch = int(datetime.fromisoformat(window_start).timestamp())
        # Compose the query: label scope + watermark + optional extra filter.
        q_parts = [f"label:{label}", f"after:{epoch}"]
        if extra_query.strip():
            q_parts.append(extra_query.strip())
        q = " ".join(q_parts)

        page_token: Optional[str] = None
        while True:
            if budget.should_stop():
                has_more = True
                break
            resp = gmail_svc.users().messages().list(
                userId="me", q=q, maxResults=500, pageToken=page_token,
            ).execute()
            ids = [m["id"] for m in resp.get("messages", [])]
            for msg_id in ids:
                if budget.should_stop():
                    has_more = True
                    break
                full = gmail_svc.users().messages().get(
                    userId="me", id=msg_id, format="full",
                ).execute()
                envelope, ts_iso = _build_envelope(full, name)
                if envelope is None:
                    continue
                # Defensive: skip rows the API surfaced outside the window
                # (Gmail's `after:` is epoch-second precision).
                if merge_policy == "upsert" and ts_iso >= window_end:
                    continue
                if slice_cfg.get("download_attachments") and envelope["attachments"]:
                    _download_envelope_attachments_gmail(
                        envelope=envelope,
                        gmail_svc=gmail_svc,
                        slice_cfg=slice_cfg,
                        dwh_dir=dwh_dir,
                        slice_name=name,
                    )
                _download_envelope_body_gmail(
                    envelope=envelope,
                    html_bytes=envelope.pop("__html_bytes", None),
                    text_bytes=envelope.pop("__text_bytes", None),
                    slice_cfg=slice_cfg,
                    dwh_dir=dwh_dir,
                    slice_name=name,
                )
                atomic_write_json(timestamp_dir / f"{envelope['id']}.json", envelope)
                if ts_iso > latest_seen:
                    latest_seen = ts_iso
                budget.rows_written += 1
            if has_more:
                break
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        _rmdir_if_empty(timestamp_dir)
        state["last_sync_at"] = utcnow_iso()
        state["last_sync_count"] = budget.rows_written - rows_written_start
        state["last_error"] = err
        atomic_write_json(state_path, state)
        return {
            "name": name,
            "rows_written": budget.rows_written - rows_written_start,
            "watermarks": {"low": low, "high": high},
            "has_more": False, "error": err,
        }

    _rmdir_if_empty(timestamp_dir)

    if timestamp_dir.exists():
        merge_err_msg = merge_json_envelopes(
            timestamp_dir, merged_path,
            policy=merge_policy, id_column=upsert_id_column,
        )
        if merge_err_msg:
            state["last_sync_at"] = utcnow_iso()
            state["last_sync_count"] = budget.rows_written - rows_written_start
            state["last_error"] = merge_err_msg
            atomic_write_json(state_path, state)
            return {
                "name": name,
                "rows_written": budget.rows_written - rows_written_start,
                "watermarks": {"low": low, "high": high},
                "has_more": False, "error": merge_err_msg,
            }
        gc_old_timestamps(source_dir)

    if merge_policy == "replace":
        state["last_sync_at"] = utcnow_iso()
        state["last_sync_count"] = budget.rows_written - rows_written_start
        state["last_error"] = None
        atomic_write_json(state_path, state)
        return {
            "name": name,
            "rows_written": budget.rows_written - rows_written_start,
            "watermarks": {"low": low, "high": high},
            "has_more": has_more, "error": None,
        }

    new_high = window_end if not has_more else max(latest_seen, high)
    state["high_watermark"] = new_high
    state["last_sync_at"] = utcnow_iso()
    state["last_sync_count"] = budget.rows_written - rows_written_start
    state["last_error"] = None
    atomic_write_json(state_path, state)

    return {
        "name": name,
        "rows_written": budget.rows_written - rows_written_start,
        "watermarks": {"low": low, "high": new_high},
        "has_more": has_more, "error": None,
    }


def _rmdir_if_empty(path: Path) -> None:
    """Remove ``path`` if it's an empty directory; ignore otherwise."""
    try:
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    except OSError:
        pass


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
    def get_attachment(message_id: str, attachment_id: str) -> dict:
        """Fetch an attachment whose body was stubbed by ``get_message(format="full")``.

        Gmail's full-message fetch inlines small attachments (under ~32 KB)
        as ``payload.parts[i].body.data`` (URL-safe base64). Larger ones
        come back stubbed: ``body`` carries ``size`` + ``attachmentId`` but
        no ``data``. Use this tool to fetch the bytes for those stubs.

        Returns a dict ``{filename, mime_type, size, data_b64}`` where
        ``data_b64`` is **standard** (not URL-safe) base64 — the skill can
        decode it directly with ``base64.b64decode(data_b64)``. ``filename``
        is recovered by walking the parent message's payload parts to find
        the part whose ``body.attachmentId`` matches; for inline parts that
        have no ``filename`` header, falls back to ``"part-{partId}"``.

        Args:
            message_id: The Gmail message id (same id you passed to get_message).
            attachment_id: The ``body.attachmentId`` from a stubbed part.

        Used by ETL skills (and any other skill that needs verbatim
        attachment bytes for files larger than ~32 KB).
        """
        svc = _service()

        # 1. Fetch the bytes (URL-safe base64).
        att = svc.users().messages().attachments().get(
            userId="me", messageId=message_id, id=attachment_id,
        ).execute()
        url_safe_b64 = att.get("data", "")
        size = att.get("size", 0)

        # Re-encode as standard base64 so callers don't need to know about
        # URL-safe. Single decode + encode is cheap.
        if url_safe_b64:
            raw_bytes = base64.urlsafe_b64decode(url_safe_b64 + "==")  # forgiving padding
            data_b64 = base64.b64encode(raw_bytes).decode()
        else:
            data_b64 = ""

        # 2. Walk message parts to recover filename + mime_type.
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

    @mcp.tool()
    def sync_to_warehouse(
        dwh_dir: str,
        config_file: str = "",
        max_runtime_seconds: int = 1500,
    ) -> dict:
        """Sync new / updated Gmail messages into the personal data warehouse.

        Call this exactly once when you receive a DM whose body starts with
        ``<!-- clawmeets:gmail-sync-trigger -->``. Read ``dwh_dir`` from your
        prompt's ``== DATA WAREHOUSE ==`` block and ``config_file`` from the
        ``== MCP CONFIG FILES ==`` block (the path next to ``gmail``).

        Named-slice model: the config carries a ``labels_to_sync`` list, each
        entry a ``{name, label, query?, merge_policy?,
        merge_policy_upsert_id_column?, start_at?, download_attachments?,
        max_attachment_size_mb?, download_body?, max_body_size_mb?,
        howto?}`` dict. Each slice
        gets its own output directory and watermark at
        ``{dwh_dir}/sources/gmail/<name>/``; per-run envelopes land in
        ``<TIMESTAMP>/<gmail_id>.json`` siblings of ``sync-state.json``, and
        the consolidated dataset rebuilds at
        ``{dwh_dir}/merged/gmail/<name>.json`` (JSON array sorted by ``ts``)
        per the slice's ``merge_policy`` (default ``upsert`` keyed on Gmail
        message ``id``). Slices advance independently — a failure on one
        does not roll back another's watermark.

        Watermark semantics: in ``upsert`` mode, the per-slice filter is
        ``q="label:<label> after:<epoch>"`` against ``internalDate`` — i.e.
        messages whose receipt time falls in ``(window_start, window_end)``.
        Optional ``query`` field on the slice is AND'd onto the query
        verbatim (e.g. ``"from:billing@aws.com"`` or ``"has:attachment"``).
        Same Gmail ``id`` overwrites in the merged JSON.

        Empty/missing config or empty ``labels_to_sync`` list ⇒
        ``status: "noop"`` (no directories created, no API calls).

        Each row is the warehouse envelope built by ``_build_envelope``:
        ``{ts, id, thread_id, message_id, message_id_hash, labels, headers,
        snippet, body_html_path, body_text_path, attachments[], raw,
        slice}``. Body bytes are NOT inlined; when the slice has
        ``download_body: true`` (default) they are persisted to disk at
        ``{dwh_dir}/merged/gmail/<slice>.attachments/<gmail_id>/
        body.{html,txt}`` (capped by ``max_body_size_mb``, default 5) and
        the envelope carries warehouse-relative ``body_html_path`` +
        ``body_text_path`` (``None`` when the body is empty / oversize /
        download_body is off). ``attachments[]`` carries one dict per
        Gmail attachment part (``{part_id, filename, content_type, size,
        attachment_id, path, downloaded_at}``); ``path`` / ``downloaded_at``
        are populated only when the slice has ``download_attachments: true``,
        in which case the bytes land at ``{dwh_dir}/merged/gmail/
        <slice>.attachments/<gmail_id>/<part_id>-<safe_filename>`` (capped
        by ``max_attachment_size_mb``, default 25). ``message_id_hash`` is
        sha256(RFC822 Message-Id)[:24] — the same algorithm the IMAP
        mailbox MCP uses, so a slice configured with
        ``merge_policy_upsert_id_column: "message_id_hash"``
        cross-source-joins with mailbox slices cleanly.

        Args:
            dwh_dir: Personal data warehouse root.
            config_file: Path to this agent's per-MCP config file.
                Empty / missing file ⇒ noop.
            max_runtime_seconds: Wall-clock budget shared across all slices.
                Default 1500 (25 min) — graceful checkpoint before the
                LLM-side invoke timeout. The MCP returns ``has_more=true``
                when the budget elapses; the next trigger resumes.

        Returns the standard sync envelope plus a ``per_slice`` map:
        ``{status, source, rows_written, window, watermarks, has_more,
        error, per_slice}``.
        """
        window_end = utcnow_iso()
        cfg, err = _load_config(config_file)
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

        # One Gmail service shared across slices — same auth handshake.
        gmail_svc = _service()

        budget = SyncBudget(max_runtime_seconds)
        per_slice: dict[str, dict] = {}
        any_error = False
        any_has_more = False
        agg_low: Optional[str] = None
        agg_high: Optional[str] = None
        first_error: Optional[str] = None

        for slice_cfg in slices:
            s_name = slice_cfg.get("name") if isinstance(slice_cfg, dict) else None
            display_name = s_name if isinstance(s_name, str) and s_name else "<unnamed>"
            if budget.should_stop():
                any_has_more = True
                per_slice[display_name] = {
                    "name": display_name, "rows_written": 0,
                    "watermarks": None, "has_more": True, "error": None,
                }
                continue
            summary = _sync_one_slice(
                gmail_svc=gmail_svc,
                slice_cfg=slice_cfg if isinstance(slice_cfg, dict) else {},
                dwh_dir=dwh_dir,
                budget=budget,
                window_end=window_end,
            )
            per_slice[summary["name"]] = summary
            if summary.get("error"):
                any_error = True
                if first_error is None:
                    first_error = summary["error"]
            if summary.get("has_more"):
                any_has_more = True
            wms = summary.get("watermarks") or {}
            if wms.get("low"):
                agg_low = wms["low"] if agg_low is None else min(agg_low, wms["low"])
            if wms.get("high"):
                agg_high = wms["high"] if agg_high is None else max(agg_high, wms["high"])

        if any_error and budget.rows_written == 0:
            status = "error"
        elif any_has_more:
            status = "partial"
        elif budget.rows_written == 0:
            status = "noop"
        else:
            status = "ok"

        return {
            "status": status,
            "source": "gmail",
            "rows_written": budget.rows_written,
            "window": [agg_low or window_end, window_end],
            "watermarks": (
                {"low": agg_low, "high": agg_high}
                if (agg_low or agg_high) else None
            ),
            "has_more": any_has_more,
            "error": first_error,
            "per_slice": per_slice,
        }

    mcp.run()


if __name__ == "__main__":
    main()
