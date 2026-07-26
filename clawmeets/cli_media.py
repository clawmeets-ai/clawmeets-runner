# SPDX-License-Identifier: MIT
"""
clawmeets/cli_media.py

``clawmeets media <image|audio|video>`` — provider-agnostic media generation.

Each subcommand resolves settings in the order
``flag > config-file section > env var > default`` and writes the generated
file to disk, printing JSON metadata whose ``media_path`` is the absolute path
consumers pick up (e.g. hand to ``update_file`` to surface it in chat).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer

from clawmeets.integrations.media import _lib

app = typer.Typer(
    name="media",
    help="Generate image / audio / video from a prompt. Paired skill: media.",
    no_args_is_help=True,
)


def _emit_json(payload) -> None:
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


def _resolve_prompt(prompt: str, prompt_file: Optional[Path]) -> str:
    if prompt_file is not None:
        prompt = prompt_file.read_text()
    elif prompt == "-":
        prompt = sys.stdin.read()
    if not prompt.strip():
        typer.echo(
            "Error: provide --prompt, --prompt-file, or '--prompt -' with stdin.",
            err=True,
        )
        raise typer.Exit(2)
    return prompt


# Shared option definitions (Typer reads these at decoration time).
_PROVIDER = typer.Option("", "--provider", help="Backend provider (overrides config).")
_MODEL    = typer.Option("", "--model", help="Model id/slug (overrides config + default).")
_API_KEY  = typer.Option("", "--api-key", help="API key (overrides config + env fallback).")
_CONFIG   = typer.Option("", "--config", help="Path to the media config (else per-agent default).")
_OUT_DIR  = typer.Option("", "--output-dir", help="Directory to write the file into.")
_PROMPT   = typer.Option("", "--prompt", help="Generation prompt. Pass '-' to read stdin.")
_PROMPT_F = typer.Option(None, "--prompt-file", help="File containing the prompt.")


@app.command()
def image(
    prompt: str = _PROMPT,
    prompt_file: Optional[Path] = _PROMPT_F,
    provider: str = _PROVIDER,
    model: str = _MODEL,
    api_key: str = _API_KEY,
    config: str = _CONFIG,
    output_dir: str = _OUT_DIR,
    size: str = typer.Option(
        "", "--size",
        help="openai px (1024x1024 etc.) / gemini aspect (1:1) / ignored for nano-banana."),
    quality: str = typer.Option("", "--quality", help="OpenAI only: auto | low | medium | high."),
    input_image: list[str] = typer.Option(
        [], "--input-image",
        help="Path to a source image to edit/compose. Repeat for multiple "
             "(e.g. subject + background). Providers: openai | nano-banana."),
) -> None:
    """Generate one image; print metadata + the on-disk media_path (PNG).

    With one or more ``--input-image`` paths, edits/composites the input(s)
    instead of text-to-image (openai or nano-banana only).
    """
    prompt = _resolve_prompt(prompt, prompt_file)
    try:
        _emit_json(_lib.generate_image(
            prompt=prompt, config_file=config, provider=provider, model=model,
            api_key=api_key, output_dir=output_dir, size=size, quality=quality,
            input_images=list(input_image),
        ))
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command()
def audio(
    prompt: str = _PROMPT,
    prompt_file: Optional[Path] = _PROMPT_F,
    provider: str = _PROVIDER,
    model: str = _MODEL,
    api_key: str = _API_KEY,
    config: str = _CONFIG,
    output_dir: str = _OUT_DIR,
    voice: str = typer.Option(
        "", "--voice",
        help="Voice name/id (openai: alloy…; elevenlabs: voice_id; gemini: Kore…)."),
    audio_format: str = typer.Option(
        "", "--format", help="openai output format: mp3 | wav | opus | aac | flac."),
) -> None:
    """Generate one speech clip; print metadata + the on-disk media_path."""
    prompt = _resolve_prompt(prompt, prompt_file)
    try:
        _emit_json(_lib.generate_audio(
            prompt=prompt, config_file=config, provider=provider, model=model,
            api_key=api_key, output_dir=output_dir, voice=voice, audio_format=audio_format,
        ))
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command()
def video(
    prompt: str = _PROMPT,
    prompt_file: Optional[Path] = _PROMPT_F,
    provider: str = _PROVIDER,
    model: str = _MODEL,
    api_key: str = _API_KEY,
    config: str = _CONFIG,
    output_dir: str = _OUT_DIR,
    duration: str = typer.Option("", "--duration", help="Clip length in seconds (model-dependent)."),
    aspect_ratio: str = typer.Option("", "--aspect-ratio", help="e.g. 16:9 | 9:16 | 1:1 (model-dependent)."),
    input_image: str = typer.Option(
        "", "--input-image", help="Path to a still for image→video (passed as the model's image input)."),
) -> None:
    """Generate one video (async submit→poll→download); print the on-disk media_path."""
    prompt = _resolve_prompt(prompt, prompt_file)
    try:
        _emit_json(_lib.generate_video(
            prompt=prompt, config_file=config, provider=provider, model=model,
            api_key=api_key, output_dir=output_dir, duration=duration,
            aspect_ratio=aspect_ratio, input_image=input_image,
        ))
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
