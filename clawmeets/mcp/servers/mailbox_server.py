# SPDX-License-Identifier: MIT
"""
clawmeets/mcp/servers/mailbox_server.py

IMAP + SMTP MCP server. Provider-agnostic mailbox access — works with any
mail server that speaks IMAP/SMTP (Gmail with app-password, iCloud, Fastmail,
Outlook, ProtonMail Bridge, self-hosted Dovecot/Postfix).

Reads its config from the per-MCP file at
``{agent_dir}/mcp-hub/configs/mailbox.json``, supplied to each tool call as
the ``config_file`` argument (the agent reads the path from its prompt's
``== MCP CONFIG FILES ==`` block). ``${VAR}`` placeholders inside the file
resolve from ``os.environ`` — same pattern as the http-api and database
MCPs. No OAuth, no token files; users export credentials as env vars on
the runner before ``clawmeets start``.

Config schema (the file at the path the agent passes):

    {
      "imap": {
        "host": "imap.gmail.com",
        "port": 993,
        "ssl": true,
        "username": "${MAILBOX_USERNAME}",
        "password": "${MAILBOX_PASSWORD}"
      },
      "smtp": {
        "host": "smtp.gmail.com",
        "port": 587,
        "starttls": true,
        "username": "${MAILBOX_USERNAME}",
        "password": "${MAILBOX_PASSWORD}",
        "from": "${MAILBOX_USERNAME}"
      },
      "folders_to_sync": [
        {
          "name": "inbox",
          "folder": "INBOX",
          "merge_policy": "upsert",
          "merge_policy_upsert_id_column": "message_id_hash",
          "download_attachments": false,        // opt-in; default false
          "max_attachment_size_mb": 25,          // cap when download_attachments=true; default 25
          "download_body": true,                 // write body.{html,txt} sidecars; default true
          "max_body_size_mb": 5,                 // cap on body sidecars; default 5
          "howto": "..."
        }
      ]
    }

``folders_to_sync`` is a list of named slices; each slice maps one IMAP
folder onto an independent warehouse dataset at
``{dwh_dir}/sources/mailbox/<name>/`` and
``{dwh_dir}/merged/mailbox/<name>.json``. Empty list / missing field ⇒
sync_to_warehouse is a no-op.

The ``smtp`` block is optional — without it, ``send_message`` raises a
clear error. The ``from`` address defaults to ``smtp.username`` when omitted.
"""
from __future__ import annotations

import base64
import email
import email.header
import email.message
import email.utils
import hashlib
import imaplib
import os
import re
import smtplib
import ssl
import unicodedata
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Optional

from clawmeets.mcp.servers._sync_warehouse import (
    SyncBudget,
    _read_state,
    atomic_write_json,
    expand_env,
    gc_old_timestamps,
    merge_json_envelopes,
    new_timestamp_dir,
    utcnow_iso,
    validate_merge_policy,
    write_howto,
)
from clawmeets.utils.jsonc import parse_jsonc
from clawmeets.utils.validation import validate_name


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


def _require_config(config_file: str) -> dict:
    """Interactive tools want a hard error on missing/bad config, not a noop."""
    cfg, err = _load_config(config_file)
    if cfg is not None:
        return cfg
    if err is None:
        raise RuntimeError(
            "mailbox config_file is required — pass the path from your "
            "`== MCP CONFIG FILES ==` prompt block (next to `mailbox`); "
            "save the config via the Configure modal in Agent Settings "
            "(see mcps/mailbox/README.md)"
        )
    raise RuntimeError(err)


def _resolve(cfg: dict, scope: dict[str, str] | None = None) -> tuple[dict, list[str]]:
    """Substitute ${VAR} from scope+os.environ across the entire config."""
    missing: list[str] = []
    expanded = expand_env(cfg, scope or {}, missing)
    return expanded, missing


def _imap_connect(imap_cfg: dict) -> imaplib.IMAP4:
    host = imap_cfg.get("host")
    port = int(imap_cfg.get("port") or (993 if imap_cfg.get("ssl", True) else 143))
    use_ssl = imap_cfg.get("ssl", True)
    if not host:
        raise RuntimeError("imap.host is required in config.json")
    if use_ssl:
        ctx = ssl.create_default_context()
        conn: imaplib.IMAP4 = imaplib.IMAP4_SSL(host, port, ssl_context=ctx)
    else:
        conn = imaplib.IMAP4(host, port)
    user = imap_cfg.get("username") or ""
    password = imap_cfg.get("password") or ""
    if not user or not password:
        raise RuntimeError(
            "imap.username and imap.password are required (resolve env vars "
            "before running)"
        )
    conn.login(user, password)
    return conn


def _smtp_connect(smtp_cfg: dict) -> smtplib.SMTP:
    host = smtp_cfg.get("host")
    port = int(smtp_cfg.get("port") or 587)
    use_starttls = smtp_cfg.get("starttls", True)
    if not host:
        raise RuntimeError("smtp.host is required in config.json")
    if port == 465:  # implicit TLS
        ctx = ssl.create_default_context()
        smtp: smtplib.SMTP = smtplib.SMTP_SSL(host, port, context=ctx)
    else:
        smtp = smtplib.SMTP(host, port)
        if use_starttls:
            smtp.starttls(context=ssl.create_default_context())
    user = smtp_cfg.get("username") or ""
    password = smtp_cfg.get("password") or ""
    if user and password:
        smtp.login(user, password)
    return smtp


def _decode_header(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    parts = email.header.decode_header(raw)
    out: list[str] = []
    for chunk, charset in parts:
        if isinstance(chunk, bytes):
            try:
                out.append(chunk.decode(charset or "utf-8", errors="replace"))
            except LookupError:
                out.append(chunk.decode("utf-8", errors="replace"))
        else:
            out.append(chunk)
    return "".join(out)


def _addr_list(raw: Any) -> list[str]:
    if not raw:
        return []
    return [
        email.utils.formataddr((name, addr)) if name else addr
        for name, addr in email.utils.getaddresses([_decode_header(raw)])
        if addr
    ]


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


def _attachment_bytes_by_part_id(raw_rfc822: bytes) -> dict[str, bytes]:
    """Walk an RFC822 message; return ``{part_id: raw_bytes}`` for every
    non-body part. Part numbering mirrors ``_normalize_message`` exactly so
    the dict keys line up with ``envelope.attachments[*].part_id``.
    """
    msg = email.message_from_bytes(raw_rfc822, _class=email.message.EmailMessage)
    out: dict[str, bytes] = {}
    counter = 0
    for part in msg.walk():
        if part.is_multipart():
            continue
        ctype = part.get_content_type() or "application/octet-stream"
        disposition = (part.get_content_disposition() or "").lower()
        filename = part.get_filename()
        if filename:
            filename = _decode_header(filename)
        is_body = (
            ctype in ("text/plain", "text/html")
            and disposition != "attachment"
            and not filename
        )
        if is_body:
            continue
        counter += 1
        try:
            payload_bytes = part.get_payload(decode=True) or b""
        except Exception:
            payload_bytes = b""
        out[str(counter)] = payload_bytes
    return out


def _download_envelope_attachments(
    *,
    envelope: dict,
    raw_rfc822: bytes,
    slice_cfg: dict,
    dwh_dir: str,
    slice_name: str,
) -> None:
    """Persist attachment bytes to disk and populate ``path`` /
    ``downloaded_at`` on each ``envelope.attachments[*]`` dict.

    Storage location on disk: ``{dwh_dir}/merged/mailbox/<slice>.attachments/
    <message_id_hash>/<part_id>-<safe_filename>``. The path lives outside
    the per-run timestamp folder so ``gc_old_timestamps`` retention trim
    doesn't delete attachment bytes the merged envelope is still pointing
    at. Same-message upserts overwrite the same path (idempotent — same
    ``message_id_hash`` + ``part_id`` means same logical attachment).

    ``attachments[i].path`` is recorded **warehouse-relative** (e.g.
    ``merged/mailbox/inbox.attachments/<hash>/<file>``) so the merged
    envelope is portable across hosts / sandboxes that mount ``dwh_dir``
    at different absolute paths. Consumers should join against their own
    ``dwh_dir``.

    Skips inline parts (typically tracking pixels / embedded CID images)
    and parts larger than ``slice_cfg.max_attachment_size_mb`` (default
    25 MB). Per-attachment errors are surfaced via a ``download_error``
    field and never raise.
    """
    raw_cap = slice_cfg.get("max_attachment_size_mb")
    try:
        cap = (int(raw_cap) if raw_cap is not None else 25) * 1024 * 1024
    except (TypeError, ValueError):
        cap = 25 * 1024 * 1024
    bytes_by_part = _attachment_bytes_by_part_id(raw_rfc822)
    dwh_root = Path(dwh_dir).expanduser().resolve()
    att_dir = (
        dwh_root / "merged" / "mailbox"
        / f"{slice_name}.attachments" / envelope["message_id_hash"]
    )
    for att in envelope.get("attachments", []):
        if att.get("disposition") == "inline":
            continue
        if att.get("size", 0) > cap:
            continue
        payload_bytes = bytes_by_part.get(att["part_id"])
        if payload_bytes is None:
            att["download_error"] = "part not found in rfc822"
            continue
        try:
            safe = _safe_filename_segment(
                att.get("filename") or f"part-{att['part_id']}"
            )
            out_path = att_dir / f"{att['part_id']}-{safe}"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_name(out_path.name + ".tmp")
            tmp.write_bytes(payload_bytes)
            os.replace(tmp, out_path)
            att["path"] = str(out_path.relative_to(dwh_root))
            att["downloaded_at"] = utcnow_iso()
        except Exception as exc:
            att["download_error"] = f"{type(exc).__name__}: {exc}"[:200]


def _download_envelope_body(
    *,
    envelope: dict,
    body_text: str,
    body_html: str,
    slice_cfg: dict,
    dwh_dir: str,
    slice_name: str,
) -> None:
    """Persist email body to disk; set ``body_{html,text}_path`` on ``envelope``.

    Storage location: ``{dwh_dir}/merged/mailbox/<slice>.attachments/
    <message_id_hash>/body.{html,txt}`` — same per-message dir as
    attachment bytes so consumers can re-open both with one path prefix.

    Paths recorded **warehouse-relative** (matching attachments). Gated by
    ``slice_cfg.download_body`` (default ``True``); capped by
    ``slice_cfg.max_body_size_mb`` (default 5 MB — real bodies are tiny;
    oversize usually means a malformed message and is silently skipped so
    the per-message envelope path stays ``None``).
    """
    if not slice_cfg.get("download_body", True):
        return
    raw_cap = slice_cfg.get("max_body_size_mb")
    try:
        cap = (int(raw_cap) if raw_cap is not None else 5) * 1024 * 1024
    except (TypeError, ValueError):
        cap = 5 * 1024 * 1024
    if not (body_html or body_text):
        return
    dwh_root = Path(dwh_dir).expanduser().resolve()
    msg_dir = (
        dwh_root / "merged" / "mailbox"
        / f"{slice_name}.attachments" / envelope["message_id_hash"]
    )
    msg_dir.mkdir(parents=True, exist_ok=True)
    for content, suffix, key in (
        (body_html, "html", "body_html_path"),
        (body_text, "txt",  "body_text_path"),
    ):
        if not content:
            continue
        data = content.encode("utf-8") if isinstance(content, str) else content
        if len(data) > cap:
            continue
        out_path = msg_dir / f"body.{suffix}"
        tmp = out_path.with_name(out_path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, out_path)
        envelope[key] = str(out_path.relative_to(dwh_root))


def _normalize_message(
    *,
    folder: str,
    uid: str,
    uidvalidity: str,
    flags: list[str],
    internaldate: datetime,
    raw_rfc822: bytes,
) -> tuple[dict, str, str]:
    """Parse RFC822 bytes into the provider-agnostic envelope.

    Returns ``(envelope, body_text, body_html)``. The body strings are
    sidechanneled out (not stored on the envelope) so the caller can hand
    them to ``_download_envelope_body`` for disk persistence; the merged
    JSON only carries ``body_{html,text}_path`` warehouse-relative paths
    (initialized to ``None`` here, populated by the download helper).
    """
    msg = email.message_from_bytes(raw_rfc822, _class=email.message.EmailMessage)

    body_text = ""
    body_html = ""
    attachments: list[dict] = []
    part_counter = [0]

    # Walk parts; collect text/html bodies + attachments (everything else).
    for part in msg.walk():
        if part.is_multipart():
            continue
        ctype = part.get_content_type() or "application/octet-stream"
        disposition = (part.get_content_disposition() or "").lower()  # 'inline'|'attachment'|''
        filename = part.get_filename()
        if filename:
            filename = _decode_header(filename)

        is_text_body = (
            ctype in ("text/plain", "text/html")
            and disposition != "attachment"
            and not filename
        )
        if is_text_body:
            try:
                payload = part.get_content()
            except Exception:
                payload = part.get_payload(decode=True) or b""
                if isinstance(payload, bytes):
                    payload = payload.decode("utf-8", errors="replace")
            if ctype == "text/plain" and not body_text:
                body_text = payload if isinstance(payload, str) else str(payload)
            elif ctype == "text/html" and not body_html:
                body_html = payload if isinstance(payload, str) else str(payload)
            continue

        # Treat as an attachment (inline images count too — disposition preserved).
        part_counter[0] += 1
        part_id = str(part_counter[0])
        try:
            payload_bytes = part.get_payload(decode=True) or b""
        except Exception:
            payload_bytes = b""
        attachments.append({
            "part_id": part_id,
            "filename": filename,
            "content_type": ctype,
            "content_id": part.get("Content-ID"),
            "disposition": disposition or "attachment",
            "size": len(payload_bytes),
            "path": None,                # populated by _download_envelope_attachments
            "downloaded_at": None,       # populated by _download_envelope_attachments
        })

    message_id = _decode_header(msg.get("Message-ID", ""))
    envelope = {
        "uid": uid,
        "uidvalidity": uidvalidity,
        "folder": folder,
        "message_id": message_id,
        "message_id_hash": _message_id_hash(folder, uid, uidvalidity, message_id),
        "date": _decode_header(msg.get("Date", "")),
        "from": _decode_header(msg.get("From", "")),
        "to": _addr_list(msg.get("To")),
        "cc": _addr_list(msg.get("Cc")),
        "bcc": _addr_list(msg.get("Bcc")),
        "reply_to": _addr_list(msg.get("Reply-To")),
        "subject": _decode_header(msg.get("Subject", "")),
        "flags": flags,
        "headers": {k: _decode_header(v) for k, v in msg.items()},
        "body_html_path": None,    # populated by _download_envelope_body
        "body_text_path": None,    # populated by _download_envelope_body
        "attachments": attachments,
    }
    return envelope, body_text, body_html


def _message_id_hash(folder: str, uid: str, uidvalidity: str, message_id: str) -> str:
    """Stable, filesystem-safe message identity. Prefers Message-ID for
    cross-folder dedup; falls back to per-folder UID under UIDVALIDITY when
    Message-ID is missing (some legacy senders omit it)."""
    if message_id:
        return hashlib.sha256(
            message_id.encode("utf-8", errors="replace")
        ).hexdigest()[:24]
    safe_folder = "".join(c if c.isalnum() or c in "-_" else "_" for c in folder)
    return f"{safe_folder}-{uidvalidity}-{uid}"


def _message_filename(folder: str, uid: str, uidvalidity: str, message_id: str) -> str:
    """Stable filename derived from ``_message_id_hash``. ``.json`` suffix."""
    return f"{_message_id_hash(folder, uid, uidvalidity, message_id)}.json"


def _imap_select_folder(conn: imaplib.IMAP4, folder: str) -> str:
    """Select a folder read-only; return its UIDVALIDITY as a string."""
    typ, data = conn.select(folder, readonly=True)
    if typ != "OK":
        raise RuntimeError(f"IMAP SELECT {folder!r} failed: {data}")
    typ, data = conn.status(folder, "(UIDVALIDITY)")
    if typ != "OK":
        raise RuntimeError(f"IMAP STATUS {folder!r} failed: {data}")
    raw = data[0].decode() if data and data[0] else ""
    # Format: '"INBOX" (UIDVALIDITY 1700000000)'
    uv = ""
    if "UIDVALIDITY" in raw:
        try:
            uv = raw.split("UIDVALIDITY", 1)[1].strip().strip(")").strip()
        except Exception:
            uv = ""
    return uv


def _parse_fetch_response(items: list) -> dict[str, dict]:
    """Parse imaplib's UID FETCH response. Returns {uid: {flags, internaldate, rfc822}}."""
    out: dict[str, dict] = {}
    pending: dict[str, dict] = {}
    for item in items:
        if isinstance(item, tuple) and len(item) >= 2:
            header_bytes, body_bytes = item[0], item[1]
            header = header_bytes.decode("utf-8", errors="replace") if isinstance(header_bytes, bytes) else str(header_bytes)
            uid = ""
            flags: list[str] = []
            idate: Optional[datetime] = None
            # Pull UID
            if "UID " in header:
                uid_chunk = header.split("UID ", 1)[1].split(" ", 1)[0].strip().strip(")")
                uid = uid_chunk
            # Pull FLAGS
            if "FLAGS (" in header:
                flag_chunk = header.split("FLAGS (", 1)[1].split(")", 1)[0]
                flags = [f for f in flag_chunk.split() if f]
            # Pull INTERNALDATE
            if "INTERNALDATE " in header:
                idate_chunk = header.split("INTERNALDATE ", 1)[1]
                # Strip leading quote+content+quote
                if idate_chunk.startswith('"'):
                    idate_chunk = idate_chunk[1:].split('"', 1)[0]
                else:
                    idate_chunk = idate_chunk.split(" ", 1)[0]
                try:
                    tup = imaplib.Internaldate2tuple(
                        b'INTERNALDATE "' + idate_chunk.encode() + b'"'
                    )
                    if tup is not None:
                        idate = datetime(*tup[:6], tzinfo=timezone.utc)
                except Exception:
                    idate = None
            if uid:
                pending[uid] = {
                    "flags": flags,
                    "internaldate": idate,
                    "rfc822": body_bytes if isinstance(body_bytes, bytes) else b"",
                }
        elif isinstance(item, bytes):
            # Trailing close-paren or status — flush pending.
            for uid, rec in pending.items():
                out[uid] = rec
            pending = {}
    for uid, rec in pending.items():
        out[uid] = rec
    return out


def _sync_one_slice(
    *,
    conn: imaplib.IMAP4,
    slice_cfg: dict,
    dwh_dir: str,
    budget: SyncBudget,
    window_end: str,
) -> dict:
    """Sync a single named mailbox slice; return its per-slice summary.

    Mirrors ``gdrive_server._sync_one_slice``: owns its own
    ``sync-state.json`` under ``{dwh_dir}/sources/mailbox/<name>/`` and
    advances its watermark independently of sibling slices. The shared
    IMAP ``conn`` (already authenticated) and ``budget`` are passed in so
    all slices in one call share one connection and one wall-clock budget.
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

    folder = slice_cfg.get("folder")
    if not isinstance(folder, str) or not folder.strip():
        return {
            "name": name,
            "rows_written": 0, "watermarks": None,
            "has_more": False,
            "error": "slice config missing required 'folder' field (IMAP folder name)",
        }

    merge_policy, upsert_id_column, merge_err = validate_merge_policy(slice_cfg)
    if merge_err:
        return {
            "name": name,
            "rows_written": 0, "watermarks": None,
            "has_more": False,
            "error": merge_err,
        }

    dwh_root = Path(dwh_dir).expanduser().resolve()
    source = f"mailbox/{name}"
    source_dir = dwh_root / "sources" / source
    state_path = source_dir / "sync-state.json"
    merged_path = dwh_root / "merged" / "mailbox" / f"{name}.json"

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
        # Replace mode: pull everything in the folder each cycle. Use the
        # IMAP epoch as a defensive lower bound so SINCE doesn't reject.
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
        window_start_dt = datetime.fromisoformat(window_start)
        # IMAP SINCE is day-granular — we round to the day; Python re-filters.
        since_date = window_start_dt.strftime("%d-%b-%Y")

        uidvalidity = _imap_select_folder(conn, folder)
        typ, data = conn.uid("SEARCH", None, "SINCE", since_date)
        if typ != "OK":
            raise RuntimeError(f"IMAP SEARCH failed: {data}")
        uids = data[0].decode().split() if data and data[0] else []
        # Process oldest-first so watermark advancement is monotonic.
        for uid in uids:
            if budget.should_stop():
                has_more = True
                break
            typ, items = conn.uid(
                "FETCH", uid, "(FLAGS INTERNALDATE BODY.PEEK[])",
            )
            if typ != "OK":
                continue
            parsed = _parse_fetch_response(items)
            rec = parsed.get(uid)
            if not rec or not rec.get("rfc822"):
                continue
            idate = rec["internaldate"]
            if idate is None:
                continue
            ts_iso = idate.astimezone(timezone.utc).isoformat()
            if merge_policy == "upsert":
                if ts_iso <= window_start or ts_iso >= window_end:
                    continue
            envelope, body_text, body_html = _normalize_message(
                folder=folder,
                uid=uid,
                uidvalidity=uidvalidity,
                flags=rec["flags"],
                internaldate=idate,
                raw_rfc822=rec["rfc822"],
            )
            envelope["ts"] = ts_iso
            envelope["slice"] = name
            if slice_cfg.get("download_attachments") and envelope["attachments"]:
                _download_envelope_attachments(
                    envelope=envelope,
                    raw_rfc822=rec["rfc822"],
                    slice_cfg=slice_cfg,
                    dwh_dir=dwh_dir,
                    slice_name=name,
                )
            _download_envelope_body(
                envelope=envelope,
                body_text=body_text,
                body_html=body_html,
                slice_cfg=slice_cfg,
                dwh_dir=dwh_dir,
                slice_name=name,
            )
            atomic_write_json(
                timestamp_dir / f"{envelope['message_id_hash']}.json",
                envelope,
            )
            if ts_iso > latest_seen:
                latest_seen = ts_iso
            budget.rows_written += 1
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

    mcp = FastMCP("clawmeets-mailbox")

    @mcp.tool()
    def list_folders(config_file: str) -> list[str]:
        """List all folders / mailboxes on the configured IMAP account.

        Useful for discovering folder names to use as the ``folder`` field
        on entries in ``folders_to_sync``.
        """
        cfg = _require_config(config_file)
        resolved, missing = _resolve(cfg)
        if missing:
            raise RuntimeError(f"unset env vars: {sorted(set(missing))}")
        imap_cfg = resolved.get("imap") or {}
        conn = _imap_connect(imap_cfg)
        try:
            typ, data = conn.list()
            if typ != "OK":
                raise RuntimeError(f"IMAP LIST failed: {data}")
            out: list[str] = []
            for line in data:
                s = line.decode() if isinstance(line, bytes) else str(line)
                # Format: (\HasNoChildren) "/" "INBOX"
                if '"' in s:
                    out.append(s.rsplit('"', 2)[-2])
            return out
        finally:
            try:
                conn.logout()
            except Exception:
                pass

    @mcp.tool()
    def search_messages(
        config_file: str,
        query: str,
        folder: str = "INBOX",
        max_results: int = 50,
    ) -> list[dict]:
        """Search a folder. ``query`` is a small DSL parsed into IMAP SEARCH:

          - ``from:alice@x.com``
          - ``to:bob@y.com``
          - ``subject:invoice``
          - ``since:2026-05-01`` (YYYY-MM-DD)
          - ``before:2026-05-08``
          - ``unseen`` / ``seen`` / ``flagged`` / ``answered``
          - bare words → SUBJECT or BODY substring

        Multiple terms AND together. Returns ``[{uid, from, subject, date,
        internal_date, snippet}]`` newest first.
        """
        cfg = _require_config(config_file)
        resolved, missing = _resolve(cfg)
        if missing:
            raise RuntimeError(f"unset env vars: {sorted(set(missing))}")
        criteria = _query_to_imap_criteria(query)
        conn = _imap_connect(resolved.get("imap") or {})
        try:
            _imap_select_folder(conn, folder)
            typ, data = conn.uid("SEARCH", None, *criteria)
            if typ != "OK":
                raise RuntimeError(f"IMAP SEARCH failed: {data}")
            uids = (data[0].decode().split() if data and data[0] else [])
            uids = uids[-max_results:][::-1]  # newest first
            if not uids:
                return []
            uid_set = ",".join(uids)
            typ, items = conn.uid(
                "FETCH", uid_set,
                "(FLAGS INTERNALDATE BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE MESSAGE-ID)])",
            )
            if typ != "OK":
                raise RuntimeError(f"IMAP FETCH failed: {items}")
            parsed = _parse_fetch_response(items)
            out: list[dict] = []
            for uid in uids:
                rec = parsed.get(uid)
                if not rec:
                    continue
                msg = email.message_from_bytes(rec["rfc822"])
                out.append({
                    "uid": uid,
                    "from": _decode_header(msg.get("From", "")),
                    "subject": _decode_header(msg.get("Subject", "")),
                    "date": _decode_header(msg.get("Date", "")),
                    "internal_date": (
                        rec["internaldate"].isoformat() if rec.get("internaldate") else None
                    ),
                    "message_id": _decode_header(msg.get("Message-ID", "")),
                })
            return out
        finally:
            try:
                conn.logout()
            except Exception:
                pass

    @mcp.tool()
    def get_message(config_file: str, uid: str, folder: str = "INBOX") -> dict:
        """Fetch one message by UID; return the full normalized envelope
        (without attachment bytes — use ``get_attachment`` for those)."""
        cfg = _require_config(config_file)
        resolved, missing = _resolve(cfg)
        if missing:
            raise RuntimeError(f"unset env vars: {sorted(set(missing))}")
        conn = _imap_connect(resolved.get("imap") or {})
        try:
            uidvalidity = _imap_select_folder(conn, folder)
            typ, items = conn.uid(
                "FETCH", uid, "(FLAGS INTERNALDATE BODY.PEEK[])",
            )
            if typ != "OK":
                raise RuntimeError(f"IMAP FETCH failed: {items}")
            parsed = _parse_fetch_response(items)
            rec = parsed.get(uid)
            if not rec:
                raise RuntimeError(f"UID {uid} not found in {folder}")
            return _normalize_message(
                folder=folder,
                uid=uid,
                uidvalidity=uidvalidity,
                flags=rec["flags"],
                internaldate=rec["internaldate"] or datetime.now(timezone.utc),
                raw_rfc822=rec["rfc822"],
            )
        finally:
            try:
                conn.logout()
            except Exception:
                pass

    @mcp.tool()
    def get_attachment(
        config_file: str, uid: str, part_id: str, folder: str = "INBOX",
    ) -> dict:
        """Fetch one attachment's bytes by ``part_id`` (the value in
        ``message.attachments[i].part_id``).

        Returns ``{filename, content_type, size, data_b64}``. ``data_b64`` is
        standard (not URL-safe) base64 — decode with
        ``base64.b64decode(data_b64)``.
        """
        cfg = _require_config(config_file)
        resolved, missing = _resolve(cfg)
        if missing:
            raise RuntimeError(f"unset env vars: {sorted(set(missing))}")
        conn = _imap_connect(resolved.get("imap") or {})
        try:
            _imap_select_folder(conn, folder)
            typ, items = conn.uid("FETCH", uid, "(BODY.PEEK[])")
            if typ != "OK":
                raise RuntimeError(f"IMAP FETCH failed: {items}")
            parsed = _parse_fetch_response(items)
            rec = parsed.get(uid)
            if not rec:
                raise RuntimeError(f"UID {uid} not found in {folder}")
            msg = email.message_from_bytes(rec["rfc822"])
            counter = 0
            for part in msg.walk():
                if part.is_multipart():
                    continue
                ctype = part.get_content_type() or "application/octet-stream"
                disposition = (part.get_content_disposition() or "").lower()
                filename = part.get_filename()
                if filename:
                    filename = _decode_header(filename)
                is_body = (
                    ctype in ("text/plain", "text/html")
                    and disposition != "attachment"
                    and not filename
                )
                if is_body:
                    continue
                counter += 1
                if str(counter) == str(part_id):
                    raw = part.get_payload(decode=True) or b""
                    return {
                        "filename": filename or f"part-{counter}",
                        "content_type": ctype,
                        "size": len(raw),
                        "data_b64": base64.b64encode(raw).decode(),
                    }
            raise RuntimeError(f"part_id {part_id!r} not found in UID {uid}")
        finally:
            try:
                conn.logout()
            except Exception:
                pass

    @mcp.tool()
    def send_message(
        config_file: str,
        to: str,
        subject: str,
        body: str,
        cc: Optional[str] = None,
        bcc: Optional[str] = None,
        reply_to: Optional[str] = None,
        html: Optional[str] = None,
    ) -> dict:
        """Send a plaintext (or HTML) email via SMTP.

        ``to`` / ``cc`` / ``bcc`` accept comma-separated address lists.
        Returns ``{message_id, accepted, refused}``.
        """
        cfg = _require_config(config_file)
        resolved, missing = _resolve(cfg)
        if missing:
            raise RuntimeError(f"unset env vars: {sorted(set(missing))}")
        smtp_cfg = resolved.get("smtp") or {}
        if not smtp_cfg.get("host"):
            raise RuntimeError("smtp block missing from config.json")
        from_addr = smtp_cfg.get("from") or smtp_cfg.get("username")
        if not from_addr:
            raise RuntimeError("smtp.from (or smtp.username) is required")

        msg = EmailMessage()
        msg["From"] = from_addr
        msg["To"] = to
        msg["Subject"] = subject
        if cc:
            msg["Cc"] = cc
        if bcc:
            msg["Bcc"] = bcc
        if reply_to:
            msg["Reply-To"] = reply_to
        msg["Message-ID"] = email.utils.make_msgid()
        msg["Date"] = email.utils.formatdate(localtime=False)
        msg.set_content(body)
        if html:
            msg.add_alternative(html, subtype="html")

        smtp = _smtp_connect(smtp_cfg)
        try:
            refused = smtp.send_message(msg)
        finally:
            try:
                smtp.quit()
            except Exception:
                pass
        return {
            "message_id": msg["Message-ID"],
            "accepted": [a for a in [to, cc, bcc] if a and a not in (refused or {})],
            "refused": list((refused or {}).keys()),
        }

    @mcp.tool()
    def sync_to_warehouse(
        dwh_dir: str,
        config_file: str,
        max_runtime_seconds: int = 1500,
    ) -> dict:
        """Sync new mail from the configured folders into the personal data warehouse.

        Call this exactly once when you receive a DM whose body starts with
        ``<!-- clawmeets:mailbox-sync-trigger -->``. Read ``dwh_dir`` from
        your prompt's ``== DATA WAREHOUSE ==`` block and ``config_file``
        from your ``== MCP CONFIG FILES ==`` block (the path next to
        ``mailbox``).

        Named-slice model: the config carries a ``folders_to_sync`` list,
        each entry a ``{name, folder, merge_policy?,
        merge_policy_upsert_id_column?, start_at?, howto?}`` dict. Each
        slice gets its own output directory and watermark at
        ``{dwh_dir}/sources/mailbox/<name>/``; per-run envelopes land in
        ``<TIMESTAMP>/<message_id_hash>.json`` siblings of
        ``sync-state.json``, and the consolidated dataset rebuilds at
        ``{dwh_dir}/merged/mailbox/<name>.json`` (JSON array sorted by
        ``ts``) per the slice's ``merge_policy`` (default ``upsert`` keyed
        on ``message_id_hash``). Slices advance independently — a failure
        on one does not roll back another's watermark.

        Watermark semantics: in ``upsert`` mode, the per-slice filter is
        ``INTERNALDATE`` (server-assigned arrival time) ``in
        (window_start, window_end)``. The IMAP ``SEARCH`` is day-rounded
        via ``SINCE``; Python re-filters to exact precision.

        Empty/missing config or empty ``folders_to_sync`` list ⇒
        ``status: "noop"`` (no directories created, no IMAP connection).

        Each row is the provider-agnostic envelope from ``_normalize_message``
        (uid, message_id, message_id_hash, from/to/cc/bcc, subject,
        body_{html,text}_path, attachments[]) plus ``ts`` (server-assigned
        arrival time, UTC) and ``slice`` (= the slice's slug). Body bytes
        are NOT inlined into the envelope JSON; when the slice has
        ``download_body: true`` (default) they are persisted to disk at
        ``{dwh_dir}/merged/mailbox/<slice>.attachments/<message_id_hash>/
        body.{html,txt}`` (capped by ``max_body_size_mb``, default 5) and
        the envelope carries ``body_html_path`` + ``body_text_path``
        (warehouse-relative, ``None`` when the body is empty / oversize /
        download_body is off). Attachment bytes are also not inlined, but
        when the slice has ``download_attachments: true`` they ARE
        persisted to disk at ``{dwh_dir}/merged/mailbox/<slice>.attachments/
        <message_id_hash>/<part_id>-<safe_filename>`` (capped by
        ``max_attachment_size_mb``, default 25) and each
        ``attachments[i]`` dict carries a ``path`` + ``downloaded_at``.
        For ad-hoc reads when ``download_attachments`` is off (or for
        skipped/oversize parts), use ``get_attachment(uid, part_id,
        folder)``.

        Args:
            dwh_dir: Personal data warehouse root.
            config_file: Path to this agent's per-MCP config file.
                Empty / missing file ⇒ noop.
            max_runtime_seconds: Wall-clock budget shared across all
                slices. Default 1500.

        Returns the standard sync envelope plus a ``per_slice`` map:
        ``{status, source, rows_written, window, watermarks, has_more,
        error, per_slice}``.
        """
        window_end = utcnow_iso()
        cfg, err = _load_config(config_file)
        if err:
            return {
                "status": "error", "source": "mailbox", "rows_written": 0,
                "window": [window_end, window_end], "watermarks": None,
                "has_more": False, "error": err, "per_slice": {},
            }
        if cfg is None:
            return {
                "status": "noop", "source": "mailbox", "rows_written": 0,
                "window": [window_end, window_end], "watermarks": None,
                "has_more": False, "error": None, "per_slice": {},
            }
        resolved, missing = _resolve(cfg)
        if missing:
            return {
                "status": "error", "source": "mailbox", "rows_written": 0,
                "window": [window_end, window_end], "watermarks": None,
                "has_more": False,
                "error": f"unset env vars: {sorted(set(missing))}",
                "per_slice": {},
            }

        slices = resolved.get("folders_to_sync")
        if not isinstance(slices, list) or len(slices) == 0:
            return {
                "status": "noop", "source": "mailbox", "rows_written": 0,
                "window": [window_end, window_end], "watermarks": None,
                "has_more": False, "error": None, "per_slice": {},
            }

        imap_cfg = resolved.get("imap") or {}

        # One IMAP connection shared across slices — each slice does its own
        # SELECT inside _sync_one_slice. Logout once at the end.
        conn = _imap_connect(imap_cfg)

        budget = SyncBudget(max_runtime_seconds)
        per_slice: dict[str, dict] = {}
        any_error = False
        any_has_more = False
        agg_low: Optional[str] = None
        agg_high: Optional[str] = None
        first_error: Optional[str] = None

        try:
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
                    conn=conn,
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
        finally:
            try:
                conn.logout()
            except Exception:
                pass

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
            "source": "mailbox",
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


def _query_to_imap_criteria(query: str) -> list[str]:
    """Parse the small DSL into an IMAP SEARCH criteria list."""
    if not query.strip():
        return ["ALL"]
    out: list[str] = []
    for token in query.split():
        if ":" in token:
            key, val = token.split(":", 1)
            key = key.lower()
            if key == "from":
                out += ["FROM", val]
            elif key == "to":
                out += ["TO", val]
            elif key == "subject":
                out += ["SUBJECT", val]
            elif key == "since":
                out += ["SINCE", _date_to_imap(val)]
            elif key == "before":
                out += ["BEFORE", _date_to_imap(val)]
            else:
                out += ["BODY", token]
        else:
            low = token.lower()
            if low in ("unseen", "seen", "flagged", "answered", "deleted"):
                out += [low.upper()]
            else:
                out += ["BODY", token]
    return out or ["ALL"]


def _date_to_imap(s: str) -> str:
    """YYYY-MM-DD → DD-Mon-YYYY (IMAP date format)."""
    try:
        d = datetime.fromisoformat(s)
    except Exception:
        return s
    return d.strftime("%d-%b-%Y")


if __name__ == "__main__":
    main()
