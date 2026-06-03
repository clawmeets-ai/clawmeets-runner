# SPDX-License-Identifier: MIT
"""
clawmeets/mcp/servers/image_gen_server.py

Image-generation MCP server. Wraps three backends behind one tool surface
and saves the generated PNG to disk, returning the local filesystem path
so downstream agents (or inline Pillow overlay code) can pick it up:

  - openai      → gpt-image-1 via /v1/images/generations
  - gemini      → Imagen (e.g. imagen-3.0-generate-002) via :predict
  - nano-banana → gemini-2.5-flash-image-preview via :generateContent

Backend selection lives in the per-agent config at
`{agent_dir}/mcp-hub/configs/image-gen.json` (`provider` field).
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROVIDERS = {"openai", "gemini", "nano-banana"}

DEFAULT_MODELS = {
    "openai":      "gpt-image-1",
    "gemini":      "imagen-3.0-generate-002",
    "nano-banana": "gemini-2.5-flash-image-preview",
}

DEFAULT_SIZES = {
    "openai":      "1024x1024",
    "gemini":      "1:1",
    "nano-banana": "",
}

OPENAI_SIZES     = {"1024x1024", "1024x1536", "1536x1024", "auto"}
OPENAI_QUALITIES = {"auto", "low", "medium", "high"}
GEMINI_ASPECTS   = {"1:1", "9:16", "16:9", "3:4", "4:3"}

OPENAI_IMAGES_URL = "https://api.openai.com/v1/images/generations"
GEMINI_BASE_URL   = "https://generativelanguage.googleapis.com/v1beta/models"


def _resolve_env(value: Any) -> Any:
    """Resolve `${VAR}` placeholders in a string against runner env. Non-strings
    pass through. Missing env vars are left as the literal placeholder so the
    caller can surface a clearer error than KeyError."""
    if not isinstance(value, str):
        return value
    if "${" not in value:
        return value
    out = value
    for key, val in os.environ.items():
        out = out.replace(f"${{{key}}}", val)
    return out


def _load_config(config_file: str) -> dict:
    """Read the per-agent config. Returns the resolved dict.

    Expected shape (all fields optional except api_key):
        {
          "provider": "openai",            // or "gemini" / "nano-banana"
          "api_key": "${OPENAI_API_KEY}",  // or "${GEMINI_API_KEY}" / literal
          "model": "gpt-image-1",
          "default_size": "1024x1024",
          "default_quality": "auto",
          "output_dir": "${CLAWMEETS_AGENT_DIR}/image-gen",
          "timeout_seconds": 120
        }
    """
    if not config_file:
        raise RuntimeError(
            "config_file is required. Pass the path from your prompt's "
            "`== MCP CONFIG FILES ==` block (next to `image-gen`)."
        )
    p = Path(config_file).expanduser()
    if not p.exists():
        raise RuntimeError(f"image-gen config file not found: {p}")
    raw = p.read_text()
    try:
        from clawmeets.utils.jsonc import parse_jsonc
        cfg = parse_jsonc(raw)
    except Exception:
        cfg = json.loads(raw)
    if not isinstance(cfg, dict):
        raise RuntimeError(
            f"image-gen config must be a JSON object, got {type(cfg).__name__}"
        )

    provider = (cfg.get("provider") or "openai").strip()
    if provider not in PROVIDERS:
        raise RuntimeError(
            f"image-gen config: `provider` {provider!r} is not supported; "
            f"choose one of: {sorted(PROVIDERS)}"
        )

    api_key = _resolve_env(cfg.get("api_key"))
    if not isinstance(api_key, str) or not api_key or api_key.startswith("${"):
        expected_env = "OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY"
        raise RuntimeError(
            f"image-gen config: `api_key` is missing or has an unresolved "
            f"${{VAR}} placeholder. For provider {provider!r} set {expected_env} "
            f"on the runner (export {expected_env}=... before `clawmeets start`) "
            f"or inline the key in the config."
        )

    output_dir_raw = _resolve_env(cfg.get("output_dir") or "")
    if not isinstance(output_dir_raw, str) or not output_dir_raw or "${" in output_dir_raw:
        raise RuntimeError(
            "image-gen config: `output_dir` is missing or has an unresolved "
            "${VAR} placeholder. Set CLAWMEETS_AGENT_DIR in the runner env or "
            "inline an absolute path."
        )
    output_dir = Path(output_dir_raw).expanduser()

    return {
        "provider": provider,
        "api_key": api_key,
        "model": cfg.get("model") or DEFAULT_MODELS[provider],
        "default_size": cfg.get("default_size") or DEFAULT_SIZES[provider],
        "default_quality": cfg.get("default_quality") or "auto",
        "output_dir": output_dir,
        "timeout_seconds": float(cfg.get("timeout_seconds") or 120),
    }


def _slug(text: str, max_len: int = 32) -> str:
    """Lowercase, ascii-only, hyphen-separated. Trimmed to max_len chars."""
    out = []
    last_dash = False
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
            last_dash = False
        elif not last_dash:
            out.append("-")
            last_dash = True
    s = "".join(out).strip("-")
    return (s[:max_len] or "image").rstrip("-")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{int(time.time() * 1000)}")
    tmp.write_bytes(data)
    tmp.replace(path)


def _generate_openai(cfg: dict, prompt: str, size: str, quality: str) -> bytes:
    """POST to the OpenAI Images endpoint and return decoded PNG bytes."""
    import httpx

    body = {
        "model": cfg["model"],
        "prompt": prompt,
        "size": size,
        "quality": quality,
        "n": 1,
    }
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=cfg["timeout_seconds"]) as client:
        resp = client.post(OPENAI_IMAGES_URL, json=body, headers=headers)
    if resp.status_code >= 400:
        try:
            err = resp.json().get("error", {})
            msg = err.get("message") or resp.text
        except Exception:
            msg = resp.text
        raise RuntimeError(f"OpenAI Images API returned {resp.status_code}: {msg}")
    payload = resp.json()
    data = payload.get("data") or []
    if not data or not isinstance(data, list):
        raise RuntimeError(f"OpenAI Images API returned no data: {payload!r}")
    b64 = data[0].get("b64_json")
    if not b64:
        raise RuntimeError(
            "OpenAI Images API response missing `b64_json` — gpt-image-1 "
            "always returns base64; check `model` in config."
        )
    return base64.b64decode(b64)


def _generate_gemini(cfg: dict, prompt: str, aspect: str) -> bytes:
    """POST to the Gemini Imagen `:predict` endpoint and return PNG bytes."""
    import httpx

    url = f"{GEMINI_BASE_URL}/{cfg['model']}:predict"
    body = {
        "instances": [{"prompt": prompt}],
        "parameters": {"sampleCount": 1, "aspectRatio": aspect},
    }
    headers = {"Content-Type": "application/json"}
    with httpx.Client(timeout=cfg["timeout_seconds"]) as client:
        resp = client.post(url, params={"key": cfg["api_key"]}, json=body, headers=headers)
    if resp.status_code >= 400:
        try:
            err = resp.json().get("error", {})
            msg = err.get("message") or resp.text
        except Exception:
            msg = resp.text
        raise RuntimeError(f"Gemini Imagen API returned {resp.status_code}: {msg}")
    payload = resp.json()
    preds = payload.get("predictions") or []
    if not preds or not isinstance(preds, list):
        raise RuntimeError(f"Gemini Imagen API returned no predictions: {payload!r}")
    b64 = preds[0].get("bytesBase64Encoded")
    if not b64:
        raise RuntimeError(
            f"Gemini Imagen response missing `bytesBase64Encoded`; likely a "
            f"safety filter rejection. Full prediction: {preds[0]!r}"
        )
    return base64.b64decode(b64)


def _generate_nano_banana(cfg: dict, prompt: str) -> bytes:
    """POST to gemini-2.5-flash-image-preview `:generateContent` and return
    PNG bytes from the first image part in the response."""
    import httpx

    url = f"{GEMINI_BASE_URL}/{cfg['model']}:generateContent"
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json"}
    with httpx.Client(timeout=cfg["timeout_seconds"]) as client:
        resp = client.post(url, params={"key": cfg["api_key"]}, json=body, headers=headers)
    if resp.status_code >= 400:
        try:
            err = resp.json().get("error", {})
            msg = err.get("message") or resp.text
        except Exception:
            msg = resp.text
        raise RuntimeError(f"Gemini generateContent returned {resp.status_code}: {msg}")
    payload = resp.json()
    candidates = payload.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini generateContent returned no candidates: {payload!r}")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    for part in parts:
        inline = part.get("inline_data") or part.get("inlineData") or {}
        mime = inline.get("mime_type") or inline.get("mimeType") or ""
        data_b64 = inline.get("data")
        if isinstance(mime, str) and mime.startswith("image/") and data_b64:
            return base64.b64decode(data_b64)
    text_parts = [p.get("text") for p in parts if p.get("text")]
    if text_parts:
        raise RuntimeError(
            "Gemini nano-banana returned text instead of an image (likely a "
            f"refusal or off-task answer): {' '.join(text_parts)[:300]}"
        )
    raise RuntimeError(
        f"Gemini nano-banana response had no image part. Raw candidate: "
        f"{candidates[0]!r}"
    )


def main() -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(
            "The `mcp` package is required but missing — the clawmeets runner "
            "should bundle it by default. Try: pip install --upgrade clawmeets"
        ) from exc

    mcp = FastMCP("clawmeets-image-gen")

    @mcp.tool()
    def generate_image(
        config_file: str,
        prompt: str,
        size: str = "",
        quality: str = "",
    ) -> dict:
        """Generate an image from a prompt and save it to disk.

        The backend (openai / gemini / nano-banana) is selected by the
        `provider` field in the per-agent config; pass the config path from
        your prompt's `== MCP CONFIG FILES ==` block (next to `image-gen`).

        ``size`` semantics depend on provider:
          - openai      — pixel size, one of "1024x1024", "1024x1536",
                          "1536x1024", "auto".
          - gemini      — aspect ratio, one of "1:1", "9:16", "16:9",
                          "3:4", "4:3".
          - nano-banana — ignored (the model picks its own size).

        ``quality`` is openai-only ("auto" | "low" | "medium" | "high") and
        is ignored for the Google backends.

        When omitted, both fall back to the per-agent config defaults.

        Returns ``{image_path, prompt_used, size_used, quality_used,
        generated_at, provider, model}``.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise RuntimeError("`prompt` is required and must be non-empty")

        cfg = _load_config(config_file)
        provider = cfg["provider"]
        resolved_size = size or cfg["default_size"]
        resolved_quality = quality or cfg["default_quality"]

        if provider == "openai":
            if resolved_size not in OPENAI_SIZES:
                raise RuntimeError(
                    f"size {resolved_size!r} not supported by gpt-image-1; "
                    f"choose one of: {sorted(OPENAI_SIZES)}"
                )
            if resolved_quality not in OPENAI_QUALITIES:
                raise RuntimeError(
                    f"quality {resolved_quality!r} not supported by gpt-image-1; "
                    f"choose one of: {sorted(OPENAI_QUALITIES)}"
                )
            png_bytes = _generate_openai(cfg, prompt, resolved_size, resolved_quality)
        elif provider == "gemini":
            if resolved_size not in GEMINI_ASPECTS:
                raise RuntimeError(
                    f"size {resolved_size!r} is not a valid Imagen aspect ratio; "
                    f"choose one of: {sorted(GEMINI_ASPECTS)}"
                )
            png_bytes = _generate_gemini(cfg, prompt, resolved_size)
            resolved_quality = "n/a"
        else:  # nano-banana
            png_bytes = _generate_nano_banana(cfg, prompt)
            resolved_size = resolved_size or "model-default"
            resolved_quality = "n/a"

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        filename = f"{ts}-{_slug(prompt)}-{digest}.png"
        out_path = (cfg["output_dir"] / filename).resolve()
        _atomic_write_bytes(out_path, png_bytes)

        return {
            "image_path": str(out_path),
            "prompt_used": prompt,
            "size_used": resolved_size,
            "quality_used": resolved_quality,
            "generated_at": ts,
            "provider": provider,
            "model": cfg["model"],
        }

    mcp.run()


if __name__ == "__main__":
    main()
