# SPDX-License-Identifier: MIT
"""
clawmeets/integrations/gdrive/_lib.py

Google Drive (read-only). Drives ``clawmeets gdrive <subcmd>``; paired skill:
``google-drive``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

GDOC_EXPORT_MIME = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/tab-separated-values",
    "application/vnd.google-apps.presentation": "text/plain",
}

TEXT_MIME_PREFIXES = ("text/",)
TEXT_MIME_EXACT = {"application/json", "application/xml", "application/x-yaml"}

MAX_INLINE_BYTES = 256 * 1024


def build_service(token_path: Path):
    from googleapiclient.discovery import build
    from clawmeets.integrations.auth.google_oauth import load_credentials

    creds = load_credentials(token_path, SCOPES)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _is_text_mime(mime: str) -> bool:
    return mime.startswith(TEXT_MIME_PREFIXES) or mime in TEXT_MIME_EXACT


def _fetch_body(svc, file_id: str, mime_type: str) -> Optional[str]:
    try:
        if mime_type in GDOC_EXPORT_MIME:
            export_mime = GDOC_EXPORT_MIME[mime_type]
            data = svc.files().export(fileId=file_id, mimeType=export_mime).execute()
        elif _is_text_mime(mime_type):
            data = svc.files().get_media(fileId=file_id).execute()
        else:
            return None
    except Exception:
        return None

    if not data:
        return ""
    if isinstance(data, bytes):
        if len(data) > MAX_INLINE_BYTES:
            return None
        try:
            return data.decode("utf-8", errors="replace")
        except Exception:
            return None
    return str(data)[:MAX_INLINE_BYTES]


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------


def search_files(svc, query: str, max_results: int = 25) -> list[dict]:
    """Search Drive with the standard Drive query syntax."""
    resp = svc.files().list(
        q=query, pageSize=max_results,
        fields=(
            "files(id, name, mimeType, modifiedTime, size, "
            "webViewLink, parents)"
        ),
    ).execute()
    out: list[dict] = []
    for f in resp.get("files", []):
        out.append({
            "id": f.get("id"),
            "name": f.get("name", ""),
            "mime_type": f.get("mimeType", ""),
            "modified_time": f.get("modifiedTime"),
            "size": int(f["size"]) if f.get("size") else None,
            "web_view_link": f.get("webViewLink"),
            "parents": f.get("parents", []),
        })
    return out


def get_file_content(svc, file_id: str) -> dict:
    """Fetch the text body of a single Drive file by id."""
    meta = svc.files().get(
        fileId=file_id,
        fields="id, name, mimeType, modifiedTime",
    ).execute()
    mime = meta.get("mimeType", "")
    return {
        "id": meta.get("id"),
        "name": meta.get("name", ""),
        "mime_type": mime,
        "modified_time": meta.get("modifiedTime"),
        "content": _fetch_body(svc, meta["id"], mime),
    }
