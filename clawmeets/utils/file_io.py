# SPDX-License-Identifier: MIT
"""
clawmeets/utils/file_io.py
Unified file I/O utilities for the clawmeets codebase.

FileUtil consolidates file I/O patterns scattered across the codebase,
providing consistent handling of JSON, NDJSON, text, and binary files
with optional atomic writes and base64 encoding.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Format(str, Enum):
    """Supported file formats for FileUtil.read() and FileUtil.write()."""

    JSON = "json"       # dict/list as JSON
    NDJSON = "ndjson"   # list of dicts, newline-delimited
    BYTES = "bytes"     # raw bytes
    TEXT = "text"       # string


class FileUtil:
    """Unified file I/O utilities.

    Provides consistent methods for reading and writing files in various
    formats with support for atomic writes, base64 encoding, and hash
    computation.

    Usage:
        # Read JSON with default fallback
        data = FileUtil.read(path, "json", default={})

        # Write JSON atomically
        FileUtil.write(path, {"key": "value"}, "json", atomic=True)

        # Append to NDJSON
        FileUtil.write(path, {"entry": 1}, "ndjson", mode="a")

        # Read bytes with base64 decoding
        content = FileUtil.read(path, "bytes", base64_decode=True)

        # Compute hash
        sha = FileUtil.sha256(content)
    """

    @staticmethod
    def read(
        path: Path,
        format: str = "json",
        default: Any = None,
        base64_decode: bool = False,
    ) -> Any:
        """Read file with format-aware parsing.

        Args:
            path: File path to read
            format: "json" | "ndjson" | "bytes" | "text"
            default: Return value if file doesn't exist or is invalid
            base64_decode: Decode content from base64 (applies to bytes/text)

        Returns:
            Parsed content or default value

        Format behavior:
            - json: Returns dict | list | None
            - ndjson: Returns list[dict] (empty list if file doesn't exist)
            - bytes: Returns bytes | None
            - text: Returns str | None
        """
        if path is None or not path.exists():
            # NDJSON returns empty list by convention
            if format == Format.NDJSON or format == "ndjson":
                return default if default is not None else []
            return default

        try:
            if format == Format.JSON or format == "json":
                return FileUtil._read_json(path, default)
            elif format == Format.NDJSON or format == "ndjson":
                return FileUtil._read_ndjson(path)
            elif format == Format.BYTES or format == "bytes":
                content = path.read_bytes()
                if base64_decode:
                    content = base64.b64decode(content)
                return content
            elif format == Format.TEXT or format == "text":
                content = path.read_text(encoding="utf-8")
                if base64_decode:
                    content = base64.b64decode(content.encode()).decode("utf-8")
                return content
            else:
                raise ValueError(f"Unknown format: {format}")
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to read {format} from {path}: {e}")
            # NDJSON returns empty list by convention
            if format == Format.NDJSON or format == "ndjson":
                return default if default is not None else []
            return default

    @staticmethod
    def _read_json(path: Path, default: Any) -> Any:
        """Read and parse a JSON file."""
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in {path}: {e}")
            return default

    @staticmethod
    def _read_ndjson(path: Path) -> list[dict]:
        """Read and parse an NDJSON file (newline-delimited JSON)."""
        result = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        result.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON line in {path}: {line[:50]}...")
        except OSError as e:
            logger.warning(f"Failed to read NDJSON from {path}: {e}")
        return result

    @staticmethod
    def write(
        path: Path,
        content: Any,
        format: str = "json",
        mode: str = "w",
        ensure_dir: bool = True,
        atomic: bool = True,
        base64_encode: bool = False,
    ) -> None:
        """Write file with format-aware serialization.

        Args:
            path: File path to write
            content: Content to write (type depends on format)
            format: "json" | "ndjson" | "bytes" | "text"
            mode: "w" (write/overwrite) | "a" (append, only for ndjson/text)
            ensure_dir: Create parent directories if needed
            atomic: Use temp file + rename for safe writes (ignored for append)
            base64_encode: Encode content to base64 (applies to bytes/text)

        Format behavior:
            - json: Expects dict | list, writes indented JSON
            - ndjson: Expects dict (single entry), writes one JSON line
            - bytes: Expects bytes, writes raw bytes
            - text: Expects str, writes text
        """
        if ensure_dir:
            path.parent.mkdir(parents=True, exist_ok=True)

        # Append mode: write directly (no atomic for append)
        if mode == "a":
            FileUtil._append(path, content, format, base64_encode)
            return

        # Write mode: optionally use atomic write
        if format == Format.JSON or format == "json":
            serialized = json.dumps(content, default=str, indent=2)
            FileUtil._write_text(path, serialized, atomic)
        elif format == Format.NDJSON or format == "ndjson":
            serialized = json.dumps(content, default=str) + "\n"
            FileUtil._write_text(path, serialized, atomic)
        elif format == Format.BYTES or format == "bytes":
            data = content
            if base64_encode:
                data = base64.b64encode(content)
            FileUtil._write_bytes(path, data, atomic)
        elif format == Format.TEXT or format == "text":
            data = content
            if base64_encode:
                data = base64.b64encode(content.encode()).decode("utf-8")
            FileUtil._write_text(path, data, atomic)
        else:
            raise ValueError(f"Unknown format: {format}")

    @staticmethod
    def _append(
        path: Path,
        content: Any,
        format: str,
        base64_encode: bool,
    ) -> None:
        """Append content to file (non-atomic)."""
        if format == Format.JSON or format == "json":
            raise ValueError("JSON format does not support append mode")
        elif format == Format.NDJSON or format == "ndjson":
            serialized = json.dumps(content, default=str) + "\n"
            with path.open("a", encoding="utf-8") as f:
                f.write(serialized)
        elif format == Format.BYTES or format == "bytes":
            data = content
            if base64_encode:
                data = base64.b64encode(content)
            with path.open("ab") as f:
                f.write(data)
        elif format == Format.TEXT or format == "text":
            data = content
            if base64_encode:
                data = base64.b64encode(content.encode()).decode("utf-8")
            with path.open("a", encoding="utf-8") as f:
                f.write(data)
        else:
            raise ValueError(f"Unknown format: {format}")

    @staticmethod
    def _write_text(path: Path, content: str, atomic: bool) -> None:
        """Write text content, optionally using atomic write."""
        if atomic:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(content, encoding="utf-8")
            tmp.replace(path)
        else:
            path.write_text(content, encoding="utf-8")

    @staticmethod
    def _write_bytes(path: Path, content: bytes, atomic: bool) -> None:
        """Write binary content, optionally using atomic write."""
        if atomic:
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(content)
            tmp.replace(path)
        else:
            path.write_bytes(content)

    @staticmethod
    def to_base64(content: bytes) -> str:
        """Encode bytes to base64 string.

        Args:
            content: Binary content to encode

        Returns:
            Base64-encoded string
        """
        return base64.b64encode(content).decode("utf-8")

    @staticmethod
    def from_base64(content_b64: str) -> bytes:
        """Decode base64 string to bytes.

        Args:
            content_b64: Base64-encoded string

        Returns:
            Decoded bytes
        """
        return base64.b64decode(content_b64)

    @staticmethod
    def sha256(content: bytes) -> str:
        """Compute SHA256 hash of content.

        Args:
            content: Binary content to hash

        Returns:
            Hex-encoded SHA256 hash
        """
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def exists(path: Path) -> bool:
        """Check if file exists.

        Args:
            path: Path to check

        Returns:
            True if path exists and is a file
        """
        return path is not None and path.exists() and path.is_file()

    @staticmethod
    def list_dir(path: Path) -> list[str]:
        """List files in directory (sorted, files only).

        Args:
            path: Directory path

        Returns:
            Sorted list of filenames (empty list if directory doesn't exist)
        """
        if path is None or not path.exists():
            return []
        try:
            return sorted(f.name for f in path.iterdir() if f.is_file())
        except OSError as e:
            logger.warning(f"Failed to list directory {path}: {e}")
            return []

    @staticmethod
    def list_dir_recursive(path: Path) -> list[str]:
        """List files recursively, returning paths relative to the given directory.

        Args:
            path: Directory path

        Returns:
            Sorted list of relative file paths (empty list if directory doesn't exist)
        """
        if path is None or not path.exists():
            return []
        try:
            return sorted(
                str(f.relative_to(path))
                for f in path.rglob("*")
                if f.is_file()
            )
        except OSError as e:
            logger.warning(f"Failed to list directory {path}: {e}")
            return []

    @staticmethod
    def delete(path: Path, missing_ok: bool = True, raise_on_error: bool = False) -> bool:
        """Delete a file.

        Args:
            path: File path to delete
            missing_ok: If True, don't raise/log if file doesn't exist
            raise_on_error: If True, raise OSError on failure; otherwise log and return False

        Returns:
            True if file was deleted, False otherwise
        """
        if path is None:
            return False
        try:
            if path.exists():
                path.unlink()
                logger.debug(f"Deleted file: {path}")
                return True
            return False
        except OSError as e:
            logger.error(f"Failed to delete file {path}: {e}")
            if raise_on_error:
                raise
            return False
