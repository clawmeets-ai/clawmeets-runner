# SPDX-License-Identifier: MIT
"""Shared best-effort token→USD pricing via ``genai-prices`` (a core dep).

Used by every provider that has token counts but no provider-reported cost:
the in-process ``ApiLLMProvider`` (all tiers) and the subprocess CLIs whose
binaries don't surface a price (gemini, codex). ``claude_cli`` reports real
cost and doesn't need this. Always best-effort: returns 0.0 (never raises) when
the model/provider isn't priceable, so a missing price never breaks a turn.
"""
from __future__ import annotations

from typing import Optional

# clawmeets provider key -> genai-prices provider id (explicit for clarity).
PRICE_PROVIDER_ID = {
    "anthropic": "anthropic",
    "openai": "openai",
    "google": "google",
    "openrouter": "openrouter",
}


def price_usd(
    model: str,
    provider_id: Optional[str],
    input_tokens: int,
    output_tokens: int,
    cache_read: int = 0,
    cache_write: int = 0,
) -> float:
    """Best-effort USD cost for one invocation; 0.0 if the model isn't priced
    (e.g. an unmapped OpenRouter slug or a missing/empty model name).

    Token convention (genai-prices): ``input_tokens`` is the TOTAL input, with
    ``cache_read``/``cache_write`` as SUBSETS of it (uncached = input − cache_read).
    Pydantic-AI (every ``-api`` tier) reports input this way, so pass straight
    through. CLI providers that instead report ``input_tokens`` as FRESH/uncached
    (gemini-cli, where cache_read can exceed input) must add the cache to the input
    THEMSELVES before calling this — otherwise "uncached" goes negative and
    genai-prices raises (silently swallowed to $0.0 below)."""
    if not model:
        return 0.0
    try:
        from genai_prices import Usage, calc_price

        calc = calc_price(
            Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
            ),
            model_ref=model,
            provider_id=provider_id,
        )
        return float(calc.total_price)
    except Exception:  # noqa: BLE001 — unknown model/provider → no cost
        return 0.0


# OpenRouter slug vendor -> genai-prices provider id (the big three; an unknown
# vendor falls through to None → genai-prices infers from the model name).
_OPENROUTER_VENDOR = {"anthropic": "anthropic", "openai": "openai", "google": "google"}


def price_openrouter_usd(
    slug: str,
    input_tokens: int,
    output_tokens: int,
    cache_read: int = 0,
    cache_write: int = 0,
) -> float:
    """Best-effort cost for an OpenRouter run. genai-prices can't price a routed
    ``<vendor>/<model>`` slug under provider ``openrouter``, so we split the slug
    and price the UNDERLYING model at its list price (the same genai-prices basis
    every other tier uses). This is an estimate — it ignores OpenRouter's margin /
    discounted routing, and for a fallback chain we can't know which model actually
    served, so we price the first. A ``:free`` model is $0; an unpriceable vendor
    falls back to $0 (no worse than today)."""
    first = (slug or "").split(",")[0].strip()  # chain → representative first model
    if not first:
        return 0.0
    if first.endswith(":free"):  # OpenRouter free tier costs nothing
        return 0.0
    first = first.split(":")[0]  # drop a :nitro/:floor/etc. variant suffix
    if "/" in first:
        vendor, model = first.split("/", 1)
        provider_id = _OPENROUTER_VENDOR.get(vendor)  # None for unknown → infer
    else:
        model, provider_id = first, None
    return price_usd(model, provider_id, input_tokens, output_tokens, cache_read, cache_write)
