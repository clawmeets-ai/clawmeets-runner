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
import smtplib
import ssl
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Optional

from clawmeets.integrations._config_resolve import expand_env, resolve_skill_config_path
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
        "body_text": body_text, "body_html": body_html,
        "attachments": attachments,
    }
    return envelope


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
        envelope = _normalize_message(
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
