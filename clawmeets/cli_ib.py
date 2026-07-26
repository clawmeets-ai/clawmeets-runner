# SPDX-License-Identifier: MIT
"""
clawmeets/cli_ib.py — Interactive Brokers (read-only) CLI.

Subcommands: positions, cost-basis, pnl, price, history, news-providers, news,
news-article, doctor. Paired skill: ``ib`` (skills/ib/SKILL.md).

This surface is **strictly read-only**. There is deliberately no ``order`` /
``buy`` / ``sell`` / ``cancel`` / ``trade`` subcommand, and the underlying
``clawmeets/integrations/ib`` package references no order symbol — enforced by
``tests/test_ib_readonly_guard.py``.
"""
from __future__ import annotations

import json
from typing import List, Optional

import typer

from clawmeets.integrations.ib import _lib

app = typer.Typer(
    name="ib",
    help=(
        "Interactive Brokers — READ-ONLY fetch of positions / cost basis / PnL / "
        "prices / history / news for US equities + bonds. Paired skill: ib."
    ),
    no_args_is_help=True,
)


def _emit_json(payload) -> None:
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _run(fn, *args, **kwargs) -> None:
    """Call a lib fetch, converting typed IB errors into clean CLI exits.

    Errors are printed to stderr as structured JSON (so the shelling LLM can act
    on the error class) and exit non-zero — never a silent empty success.
    """
    try:
        _emit_json(fn(*args, **kwargs))
    except _lib.MarketDataNotSubscribedError as exc:
        typer.echo(json.dumps({"error": "not_subscribed", "detail": str(exc), "codes": exc.codes}), err=True)
        raise typer.Exit(3) from exc
    except _lib.ContractResolutionError as exc:
        typer.echo(json.dumps({"error": "contract_resolution", "detail": str(exc)}), err=True)
        raise typer.Exit(4) from exc
    except _lib.IBPacingError as exc:
        typer.echo(json.dumps({"error": "pacing", "detail": str(exc), "retry_after": exc.retry_after}), err=True)
        raise typer.Exit(5) from exc
    except _lib.IBConnectionError as exc:
        typer.echo(json.dumps({"error": "connection", "detail": str(exc)}), err=True)
        raise typer.Exit(6) from exc
    except _lib.IBError as exc:
        typer.echo(json.dumps({"error": "ib", "detail": str(exc)}), err=True)
        raise typer.Exit(2) from exc


@app.command()
def positions(
    account: str = typer.Option("", "--account"),
    config: str = typer.Option("", "--config"),
) -> None:
    """List open positions (account data; no market-data subscription)."""
    _run(_lib.get_positions, config_file=config, account=account)


@app.command("cost-basis")
def cost_basis_cmd(
    account: str = typer.Option("", "--account"),
    config: str = typer.Option("", "--config"),
) -> None:
    """Cost basis per position from avgCost (account data; equity + bond)."""
    _run(_lib.get_cost_basis, config_file=config, account=account)


@app.command()
def pnl(
    account: str = typer.Option("", "--account"),
    conid: Optional[List[int]] = typer.Option(None, "--conid", help="Limit to these conIds (repeatable)."),
    config: str = typer.Option("", "--config"),
) -> None:
    """Per-position PnL. Unrealized leg needs a live price (typed error if unsubscribed)."""
    _run(_lib.get_pnl, config_file=config, account=account, conids=list(conid) if conid else None)


@app.command()
def price(
    ident: str = typer.Argument(..., help="Symbol (equity) or CUSIP (bond)."),
    sec_type: str = typer.Option("STK", "--sec-type", help="STK or BOND."),
    allow_delayed: bool = typer.Option(False, "--allow-delayed", help="Permit delayed data (tagged data_type=delayed)."),
    config: str = typer.Option("", "--config"),
) -> None:
    """Snapshot current price. LIVE by default; delayed only under --allow-delayed."""
    _run(_lib.get_current_price, ident, sec_type=sec_type, allow_delayed=allow_delayed, config_file=config)


@app.command()
def history(
    ident: str = typer.Argument(..., help="Symbol (equity) or CUSIP (bond)."),
    sec_type: str = typer.Option("STK", "--sec-type", help="STK or BOND."),
    duration: str = typer.Option("1 M", "--duration"),
    bar_size: str = typer.Option("1 day", "--bar-size"),
    what: str = typer.Option("TRADES", "--what", help="TRADES / MIDPOINT / BID_ASK / YIELD (bonds)."),
    allow_delayed: bool = typer.Option(False, "--allow-delayed"),
    config: str = typer.Option("", "--config"),
) -> None:
    """Historic bars. Pacing enforced on disk; typed error if unsubscribed."""
    _run(
        _lib.get_historic_prices, ident, sec_type=sec_type, duration=duration,
        bar_size=bar_size, what=what, allow_delayed=allow_delayed, config_file=config,
    )


@app.command("news-providers")
def news_providers_cmd(
    config: str = typer.Option("", "--config"),
) -> None:
    """List subscribed news providers (this is itself the news subscription probe)."""
    _run(_lib.list_news_providers, config_file=config)


@app.command()
def news(
    ident: str = typer.Argument(..., help="Symbol (equity) or CUSIP (bond)."),
    sec_type: str = typer.Option("STK", "--sec-type", help="STK or BOND."),
    lookback_days: int = typer.Option(7, "--lookback-days"),
    provider: Optional[List[str]] = typer.Option(None, "--provider", help="Provider code(s), repeatable."),
    config: str = typer.Option("", "--config"),
) -> None:
    """Historical news headlines. Unsubscribed requested provider -> typed error."""
    _run(
        _lib.get_news_headlines, ident, sec_type=sec_type, lookback_days=lookback_days,
        providers=list(provider) if provider else None, config_file=config,
    )


@app.command("news-article")
def news_article_cmd(
    provider_code: str = typer.Argument(...),
    article_id: str = typer.Argument(...),
    config: str = typer.Option("", "--config"),
) -> None:
    """Fetch a full news-article body by (provider_code, article_id)."""
    _run(_lib.get_news_article, provider_code, article_id, config_file=config)


@app.command()
def doctor(
    config: str = typer.Option("", "--config"),
) -> None:
    """Connectivity + read-only + subscription self-check. Returns no market data."""
    _run(_lib.doctor, config_file=config)
