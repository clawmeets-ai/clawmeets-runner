# SPDX-License-Identifier: MIT
"""
clawmeets/integrations/mailbox/_lib.py

IMAP + SMTP integration. Provider-agnostic — works with Gmail (app password),
iCloud, Fastmail, Outlook, ProtonMail Bridge, self-hosted Dovecot/Postfix.
No OAuth; credentials come from env vars referenced via ``${VAR}`` in the
per-agent config at ``$CLAWMEETS_AGENT_DIR/skill-hub/configs/mailbox.json``.
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
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Optional

from clawmeets.integrations._config_resolve import resolve_skill_config_path
from clawmeets.integrations._sync_warehouse import (
    SyncBudget,
    expand_env,
    run_slice_sync,
    run_slices,
    utcnow_iso,
    write_howto,
)
from clawmeets.utils.file_io import FileUtil
from clawmeets.utils.jsonc import parse_jsonc


def load_config(config_file: str) -> tuple[Optional[dict], Optional[str]]:
    config_file = resolve_skill_config_path("mailbox", config_file)
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
    cfg, err = load_config(config_file)
    if cfg is not None:
        return cfg
    if err is None:
        raise RuntimeError(
            "mailbox config not found. Set up Agent Settings → Skills → "
            "mailbox → Configure first."
        )
    raise RuntimeError(err)


def _resolve(cfg: dict, scope: Optional[dict[str, str]] = None) -> tuple[dict, list[str]]:
    missing: list[str] = []
    expanded = expand_env(cfg, scope or {}, missing)
    return expanded, missing


def _imap_connect(imap_cfg: dict) -> imaplib.IMAP4:
    host = imap_cfg.get("host")
    port = int(imap_cfg.get("port") or (993 if imap_cfg.get("ssl", True) else 143))
    use_ssl = imap_cfg.get("ssl", True)
    if not host:
        raise RuntimeError("imap.host is required in mailbox config")
    if use_ssl:
        ctx = ssl.create_default_context()
        conn: imaplib.IMAP4 = imaplib.IMAP4_SSL(host, port, ssl_context=ctx)
    else:
        conn = imaplib.IMAP4(host, port)
    user = imap_cfg.get("username") or ""
    password = imap_cfg.get("password") or ""
    if not user or not password:
        raise RuntimeError(
            "imap.username and imap.password are required (resolve env vars first)"
        )
    conn.login(user, password)
    return conn


def _smtp_connect(smtp_cfg: dict) -> smtplib.SMTP:
    host = smtp_cfg.get("host")
    port = int(smtp_cfg.get("port") or 587)
    use_starttls = smtp_cfg.get("starttls", True)
    if not host:
        raise RuntimeError("smtp.host is required in mailbox config")
    if port == 465:
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


def _download_envelope_attachments(*, envelope, raw_rfc822, slice_cfg, dwh_root, files_root) -> None:
    raw_cap = slice_cfg.get("max_attachment_size_mb")
    try:
        cap = (int(raw_cap) if raw_cap is not None else 25) * 1024 * 1024
    except (TypeError, ValueError):
        cap = 25 * 1024 * 1024
    bytes_by_part = _attachment_bytes_by_part_id(raw_rfc822)
    att_dir = files_root / envelope["message_id_hash"]
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


def _download_envelope_body(*, envelope, body_text, body_html, slice_cfg, dwh_root, files_root) -> None:
    if not slice_cfg.get("download_body", True):
        return
    raw_cap = slice_cfg.get("max_body_size_mb")
    try:
        cap = (int(raw_cap) if raw_cap is not None else 5) * 1024 * 1024
    except (TypeError, ValueError):
        cap = 5 * 1024 * 1024
    if not (body_html or body_text):
        return
    msg_dir = files_root / envelope["message_id_hash"]
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


def _normalize_message(*, folder, uid, uidvalidity, flags, internaldate, raw_rfc822):
    msg = email.message_from_bytes(raw_rfc822, _class=email.message.EmailMessage)
    body_text = ""
    body_html = ""
    attachments: list[dict] = []
    part_counter = [0]

    for part in msg.walk():
        if part.is_multipart():
            continue
        ctype = part.get_content_type() or "application/octet-stream"
        disposition = (part.get_content_disposition() or "").lower()
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
        part_counter[0] += 1
        part_id = str(part_counter[0])
        try:
            payload_bytes = part.get_payload(decode=True) or b""
        except Exception:
            payload_bytes = b""
        attachments.append({
            "part_id": part_id, "filename": filename,
            "content_type": ctype, "content_id": part.get("Content-ID"),
            "disposition": disposition or "attachment",
            "size": len(payload_bytes),
            "path": None, "downloaded_at": None,
        })

    message_id = _decode_header(msg.get("Message-ID", ""))
    envelope = {
        "uid": uid, "uidvalidity": uidvalidity, "folder": folder,
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
        "body_html_path": None, "body_text_path": None,
        "attachments": attachments,
    }
    return envelope, body_text, body_html


def _message_id_hash(folder, uid, uidvalidity, message_id) -> str:
    if message_id:
        return hashlib.sha256(
            message_id.encode("utf-8", errors="replace")
        ).hexdigest()[:24]
    safe_folder = "".join(c if c.isalnum() or c in "-_" else "_" for c in folder)
    return f"{safe_folder}-{uidvalidity}-{uid}"


def _imap_select_folder(conn, folder: str) -> str:
    typ, data = conn.select(folder, readonly=True)
    if typ != "OK":
        raise RuntimeError(f"IMAP SELECT {folder!r} failed: {data}")
    typ, data = conn.status(folder, "(UIDVALIDITY)")
    if typ != "OK":
        raise RuntimeError(f"IMAP STATUS {folder!r} failed: {data}")
    raw = data[0].decode() if data and data[0] else ""
    uv = ""
    if "UIDVALIDITY" in raw:
        try:
            uv = raw.split("UIDVALIDITY", 1)[1].strip().strip(")").strip()
        except Exception:
            uv = ""
    return uv


def _parse_fetch_response(items: list) -> dict[str, dict]:
    out: dict[str, dict] = {}
    pending: dict[str, dict] = {}
    for item in items:
        if isinstance(item, tuple) and len(item) >= 2:
            header_bytes, body_bytes = item[0], item[1]
            header = header_bytes.decode("utf-8", errors="replace") if isinstance(header_bytes, bytes) else str(header_bytes)
            uid = ""
            flags: list[str] = []
            idate: Optional[datetime] = None
            if "UID " in header:
                uid_chunk = header.split("UID ", 1)[1].split(" ", 1)[0].strip().strip(")")
                uid = uid_chunk
            if "FLAGS (" in header:
                flag_chunk = header.split("FLAGS (", 1)[1].split(")", 1)[0]
                flags = [f for f in flag_chunk.split() if f]
            if "INTERNALDATE " in header:
                idate_chunk = header.split("INTERNALDATE ", 1)[1]
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
                    "flags": flags, "internaldate": idate,
                    "rfc822": body_bytes if isinstance(body_bytes, bytes) else b"",
                }
        elif isinstance(item, bytes):
            for uid, rec in pending.items():
                out[uid] = rec
            pending = {}
    for uid, rec in pending.items():
        out[uid] = rec
    return out


def _sync_one_slice(*, conn, slice_cfg, dwh_dir, budget):
    raw_name = slice_cfg.get("name") if isinstance(slice_cfg, dict) else None
    if not isinstance(raw_name, str) or not raw_name.strip():
        return {"name": "<unnamed>", "rows_written": 0, "watermarks": None,
                "has_more": False, "error": "slice config missing required 'name' field"}
    try:
        name = FileUtil.validate_fs_name(raw_name)
    except ValueError as exc:
        return {"name": raw_name, "rows_written": 0, "watermarks": None,
                "has_more": False, "error": f"invalid slice name {raw_name!r}: {exc}"}

    folder = slice_cfg.get("folder")
    if not isinstance(folder, str) or not folder.strip():
        return {"name": name, "rows_written": 0, "watermarks": None,
                "has_more": False,
                "error": "slice config missing required 'folder' field (IMAP folder name)"}

    dwh_root = Path(dwh_dir).expanduser().resolve()
    source = f"mailbox/{name}"
    base = dwh_root / "raw" / source
    files_root = base / "files"

    howto_err = write_howto(slice_cfg.get("howto"), snapshot_dir=base)
    if howto_err:
        return {"name": name, "rows_written": 0, "watermarks": None,
                "has_more": False, "error": howto_err}

    def fetch(window_start: str, window_end: str, bud, emit) -> bool:
        ws_dt = datetime.fromisoformat(window_start.replace("Z", "+00:00"))
        we_dt = datetime.fromisoformat(window_end.replace("Z", "+00:00"))
        since_date = ws_dt.strftime("%d-%b-%Y")
        # IMAP BEFORE is date-exclusive; +1 day so window_end's own day is included,
        # then the precise ts filter below trims to the instant.
        before_date = (we_dt + timedelta(days=1)).strftime("%d-%b-%Y")

        uidvalidity = _imap_select_folder(conn, folder)
        typ, data = conn.uid("SEARCH", None, "SINCE", since_date, "BEFORE", before_date)
        if typ != "OK":
            raise RuntimeError(f"IMAP SEARCH failed: {data}")
        uids = data[0].decode().split() if data and data[0] else []
        for uid in uids:
            if bud.should_stop():
                return True
            typ, items = conn.uid("FETCH", uid, "(FLAGS INTERNALDATE BODY.PEEK[])")
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
            if ts_iso <= window_start or ts_iso >= window_end:
                continue
            envelope, body_text, body_html = _normalize_message(
                folder=folder, uid=uid, uidvalidity=uidvalidity,
                flags=rec["flags"], internaldate=idate, raw_rfc822=rec["rfc822"],
            )
            envelope["ts"] = ts_iso
            envelope["slice"] = name
            if slice_cfg.get("download_attachments") and envelope["attachments"]:
                _download_envelope_attachments(
                    envelope=envelope, raw_rfc822=rec["rfc822"],
                    slice_cfg=slice_cfg, dwh_root=dwh_root, files_root=files_root,
                )
            _download_envelope_body(
                envelope=envelope, body_text=body_text, body_html=body_html,
                slice_cfg=slice_cfg, dwh_root=dwh_root, files_root=files_root,
            )
            bud.rows_written += 1
            emit(envelope)
        return False

    return run_slice_sync(
        source=source, dwh_dir=dwh_dir, budget=budget, fetch=fetch,
        id_field="message_id_hash", ts_field="ts",
        start_at=slice_cfg.get("start_at"), snapshot_fmt="ndjson",
    )


def _date_to_imap(s: str) -> str:
    try:
        d = datetime.fromisoformat(s)
    except Exception:
        return s
    return d.strftime("%d-%b-%Y")


def _query_to_imap_criteria(query: str) -> list[str]:
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


# ---------------------------------------------------------------------------
# Public tool bodies
# ---------------------------------------------------------------------------


def list_folders(config_file: str) -> list[str]:
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
            if '"' in s:
                out.append(s.rsplit('"', 2)[-2])
        return out
    finally:
        try:
            conn.logout()
        except Exception:
            pass


def search_messages(config_file, query, folder="INBOX", max_results=50) -> list[dict]:
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
        uids = uids[-max_results:][::-1]
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


def get_message(config_file, uid, folder="INBOX") -> dict:
    cfg = _require_config(config_file)
    resolved, missing = _resolve(cfg)
    if missing:
        raise RuntimeError(f"unset env vars: {sorted(set(missing))}")
    conn = _imap_connect(resolved.get("imap") or {})
    try:
        uidvalidity = _imap_select_folder(conn, folder)
        typ, items = conn.uid("FETCH", uid, "(FLAGS INTERNALDATE BODY.PEEK[])")
        if typ != "OK":
            raise RuntimeError(f"IMAP FETCH failed: {items}")
        parsed = _parse_fetch_response(items)
        rec = parsed.get(uid)
        if not rec:
            raise RuntimeError(f"UID {uid} not found in {folder}")
        envelope, _, _ = _normalize_message(
            folder=folder, uid=uid, uidvalidity=uidvalidity,
            flags=rec["flags"],
            internaldate=rec["internaldate"] or datetime.now(timezone.utc),
            raw_rfc822=rec["rfc822"],
        )
        return envelope
    finally:
        try:
            conn.logout()
        except Exception:
            pass


def get_attachment(config_file, uid, part_id, folder="INBOX") -> dict:
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


def send_message(config_file, to, subject, body, cc=None, bcc=None,
                 reply_to=None, html=None) -> dict:
    cfg = _require_config(config_file)
    resolved, missing = _resolve(cfg)
    if missing:
        raise RuntimeError(f"unset env vars: {sorted(set(missing))}")
    smtp_cfg = resolved.get("smtp") or {}
    if not smtp_cfg.get("host"):
        raise RuntimeError("smtp block missing from mailbox config")
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


def sync_to_warehouse(dwh_dir, config_file="", max_runtime_seconds=1500) -> dict:
    """Sync IMAP folders into the warehouse.

    Triggered by ``<!-- clawmeets:mailbox-sync-trigger -->``.
    """
    window_end = utcnow_iso()
    cfg, err = load_config(config_file)
    if err:
        return {"status": "error", "source": "mailbox", "rows_written": 0,
                "window": [window_end, window_end], "watermarks": None,
                "has_more": False, "error": err, "per_slice": {}}
    if cfg is None:
        return {"status": "noop", "source": "mailbox", "rows_written": 0,
                "window": [window_end, window_end], "watermarks": None,
                "has_more": False, "error": None, "per_slice": {}}
    resolved, missing = _resolve(cfg)
    if missing:
        return {"status": "error", "source": "mailbox", "rows_written": 0,
                "window": [window_end, window_end], "watermarks": None,
                "has_more": False,
                "error": f"unset env vars: {sorted(set(missing))}",
                "per_slice": {}}

    slices = resolved.get("folders_to_sync")
    if not isinstance(slices, list) or len(slices) == 0:
        return {"status": "noop", "source": "mailbox", "rows_written": 0,
                "window": [window_end, window_end], "watermarks": None,
                "has_more": False, "error": None, "per_slice": {}}

    imap_cfg = resolved.get("imap") or {}
    conn = _imap_connect(imap_cfg)

    budget = SyncBudget(max_runtime_seconds)
    try:
        return run_slices(
            source_family="mailbox", slices=slices, budget=budget,
            dwh_dir=dwh_dir,
            run_one=lambda sc: _sync_one_slice(
                conn=conn, slice_cfg=sc, dwh_dir=dwh_dir, budget=budget),
        )
    finally:
        try:
            conn.logout()
        except Exception:
            pass
