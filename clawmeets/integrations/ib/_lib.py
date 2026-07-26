# SPDX-License-Identifier: MIT
"""
clawmeets/integrations/ib/_lib.py

Pure-Python, **strictly read-only** Interactive Brokers integration. Drives
``clawmeets ib <subcmd>`` via ``clawmeets/cli_ib.py``; the paired skill
``skills/ib/SKILL.md`` teaches the LLM when to shell which subcommand.

All IB access goes through the ``ib_async`` socket API (the maintained fork of
the unmaintained ``ib_insync`` — API-compatible) against a human-operated
TWS / IB Gateway instance. This module fetches **positions, cost basis, PnL,
historic prices, current prices, and news** for US equities and bonds. It never
places, modifies, or cancels an order — there is deliberately no order symbol
anywhere in this package, enforced by ``tests/test_ib_readonly_guard.py``.

Design notes (map to the M1 plan + review findings F1–F6):
  * F1  Pacing is persisted to disk (``PacingLedger``). Every ``clawmeets ib``
        call is a fresh process, so an in-memory throttle enforces nothing; the
        IB pacing budget is only real if the request timestamps survive between
        processes. The ledger is an on-disk, cross-process (flock-guarded)
        timestamp record consulted at the start of every market-data call.
  * F2  Missing market-data / news subscriptions surface as a typed
        ``MarketDataNotSubscribedError`` — never a silent empty/nan success.
  * F3  A central ``_ErrorSink`` buckets IB error codes (benign / not-subscribed
        / pacing / fatal); pacing violations back off, fatal ones raise.
  * F4  Equity vs bond diverge in exactly two places: ``_resolve_contract``
        (bonds resolve CUSIP -> conId) and ``_bond_economics`` (%-of-par pricing,
        clean price quoted, accrued interest broken out as its own field).
  * F5  No credentials touch this repo or ``ib.json`` — only host/port/clientId/
        account. Connection coordinates are never logged; the account id is
        treated as sensitive.
  * F6  Delayed data never masquerades as live: default is LIVE (market data
        type 1); delayed (type 3) is only reachable behind an explicit
        ``allow_delayed`` flag and every such payload is tagged
        ``data_type="delayed"``. ``doctor`` reports the connected account's
        paper/live flag so a wrong-port connection is caught before use.
"""
from __future__ import annotations

import contextlib
import math
import os
import time
from pathlib import Path
from typing import Iterator, Optional

from clawmeets.integrations._config_resolve import resolve_skill_config_path
from clawmeets.utils.file_io import FileUtil
from clawmeets.utils.jsonc import parse_jsonc

try:  # flock is POSIX-only; degrade to best-effort (no cross-process lock) elsewhere.
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX runners
    fcntl = None  # type: ignore[assignment]

SKILL_NAME = "ib"

# Market-data-type codes accepted by IB's reqMarketDataType.
MKT_DATA_LIVE = 1
MKT_DATA_DELAYED = 3

# ---------------------------------------------------------------------------
# IB error-code buckets (consumed by _ErrorSink / _classify).
# ---------------------------------------------------------------------------
# Farm-connection / connectivity notices — informational, not failures.
_BENIGN_CODES = frozenset({1102, 2103, 2104, 2105, 2106, 2107, 2108, 2119, 2158})
# The market-data / news subscription is missing (or a competing session stole
# the line). These must become MarketDataNotSubscribedError, not empty data.
_NOT_SUBSCRIBED_CODES = frozenset({354, 10089, 10091, 10167, 10168, 10197})
# Historical-data "no permissions" rides code 162 alongside pacing — the message
# text disambiguates (see _is_permission_162).
_PERMISSION_162 = 162
# Server-side pacing violations — retryable with backoff.
_PACING_CODES = frozenset({420})
# Connectivity lost / cannot connect — fatal for this invocation.
_FATAL_CODES = frozenset({1100, 502, 504})


# ---------------------------------------------------------------------------
# Typed errors
# ---------------------------------------------------------------------------
class IBError(Exception):
    """Base for every error this integration raises."""


class IBConnectionError(IBError):
    """Could not connect to / stay connected to TWS or IB Gateway."""


class MarketDataNotSubscribedError(IBError):
    """A required paid market-data / news subscription is missing.

    Carries the human identifier, which IB subscription is implicated, and the
    raw IB error code(s) so the operator can fix the entitlement. Raised instead
    of returning empty/nan data — the whole point of the silent-empty trap fix.
    """

    def __init__(self, ident: str, detail: str, codes: Optional[list[int]] = None):
        self.ident = ident
        self.detail = detail
        self.codes = codes or []
        super().__init__(
            f"No market-data subscription for {ident!r}: {detail}"
            + (f" (IB codes {self.codes})" if self.codes else "")
        )


class ContractResolutionError(IBError):
    """A symbol / CUSIP did not resolve to exactly one IB contract."""


class IBPacingError(IBError):
    """IB pacing / rate limit would be violated. Retryable after ``retry_after``."""

    def __init__(self, message: str, retry_after: float = 0.0):
        self.retry_after = retry_after
        super().__init__(message)


# ---------------------------------------------------------------------------
# Config (F5 — connection coordinates only, never credentials)
# ---------------------------------------------------------------------------
_CONFIG_DEFAULTS = {
    "host": "127.0.0.1",
    "port": 4001,          # IB Gateway live=4001/paper=4002; TWS live=7496/paper=7497
    "client_id": 17,
    "account": "",         # optional; disambiguates a login; single-account scope
    "market_data_type": MKT_DATA_LIVE,
    "connect_timeout_s": 15,
}


def _load_config(config_file: str = "") -> dict:
    """Resolve + parse ``ib.json`` (host/port/clientId/account only).

    Falls back to ``$CLAWMEETS_AGENT_DIR/skill-hub/configs/ib.json``. A missing
    file is fine — connection defaults (localhost Gateway) are used so a fresh
    install still works. Never contains, and never logs, credentials.
    """
    path = resolve_skill_config_path(SKILL_NAME, explicit=config_file)
    cfg = dict(_CONFIG_DEFAULTS)
    if path:
        raw = FileUtil.read(Path(path), "text")
        if raw:
            parsed = parse_jsonc(raw)
            if not isinstance(parsed, dict):
                raise IBError(f"ib config at {path} is not a JSON object")
            # Only known connection keys are honoured; unknown keys are ignored
            # rather than trusted, keeping this file to pure connection coordinates.
            for key in _CONFIG_DEFAULTS:
                if key in parsed and parsed[key] is not None:
                    cfg[key] = parsed[key]
    return cfg


# ---------------------------------------------------------------------------
# F1 — disk-persisted pacing ledger
# ---------------------------------------------------------------------------
# IB's documented pacing limits for the socket API (historical data is the
# binding constraint): no more than 60 requests in any rolling 10 minutes, no
# identical request within 15s, and no burst of 6+ identical requests within 2s.
HIST_WINDOW_S = 600
HIST_MAX_IN_WINDOW = 60
IDENTICAL_MIN_GAP_S = 15
BURST_WINDOW_S = 2
BURST_MAX = 6
# Short waits we simply sleep through; a longer required wait (e.g. the 10-min
# window is full) is surfaced as a retryable IBPacingError so the caller backs
# off instead of blocking a CLI process for minutes.
MAX_BLOCKING_SLEEP_S = 20.0


def _pacing_ledger_path() -> Optional[Path]:
    agent_dir = os.environ.get("CLAWMEETS_AGENT_DIR")
    if not agent_dir:
        return None
    return Path(agent_dir) / "skill-hub" / "state" / SKILL_NAME / "pacing.json"


class PacingLedger:
    """Cross-process throttle backed by an on-disk timestamp record (F1).

    Because each ``clawmeets ib`` call is its own process, the only place pacing
    state can live is disk. ``gate`` reads the ledger, prunes stale entries,
    waits (or raises) to satisfy IB's limits, then records the new request —
    all under an advisory file lock so concurrent runners share one budget.
    """

    def __init__(self, path: Optional[Path], now=time.time, sleep=time.sleep):
        self._path = path
        self._now = now
        self._sleep = sleep

    def gate(self, signature: str) -> float:
        """Block until issuing a request with ``signature`` respects IB pacing.

        Returns the seconds actually slept. Raises ``IBPacingError`` (with
        ``retry_after``) when the required wait exceeds ``MAX_BLOCKING_SLEEP_S``.
        A ``None`` path (no agent dir) is a best-effort no-op that still records
        nothing — pacing simply isn't enforced in that degraded environment.
        """
        if self._path is None:
            return 0.0
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked() as fh:
            data = self._read(fh)
            now = self._now()
            hist = [t for t in data.get("hist", []) if now - t < HIST_WINDOW_S]
            sigs = {
                s: t for s, t in data.get("sig", {}).items()
                if now - t < max(IDENTICAL_MIN_GAP_S, BURST_WINDOW_S) * 4
            }
            burst = [t for t in data.get("burst", {}).get(signature, []) if now - t < BURST_WINDOW_S]

            wait = 0.0
            if signature in sigs and now - sigs[signature] < IDENTICAL_MIN_GAP_S:
                wait = max(wait, IDENTICAL_MIN_GAP_S - (now - sigs[signature]))
            if len(burst) >= BURST_MAX:
                wait = max(wait, BURST_WINDOW_S - (now - burst[0]))
            if len(hist) >= HIST_MAX_IN_WINDOW:
                wait = max(wait, HIST_WINDOW_S - (now - hist[0]))

            if wait > MAX_BLOCKING_SLEEP_S:
                raise IBPacingError(
                    f"IB pacing budget exhausted; retry in ~{wait:.0f}s "
                    f"({len(hist)} requests in the last {HIST_WINDOW_S // 60} min).",
                    retry_after=wait,
                )
            if wait > 0:
                self._sleep(wait)
                now = self._now()

            hist.append(now)
            burst = [t for t in burst if now - t < BURST_WINDOW_S] + [now]
            sigs[signature] = now
            # Prune the per-signature burst dict: any signature whose timestamps
            # have all aged past BURST_WINDOW_S can no longer constrain a burst,
            # so drop it. Without this the dict grows one key per distinct
            # signature forever and the ledger file bloats unbounded.
            pruned_burst = {
                sig: recent
                for sig, stamps in data.get("burst", {}).items()
                if sig != signature
                for recent in [[t for t in stamps if now - t < BURST_WINDOW_S]]
                if recent
            }
            pruned_burst[signature] = burst
            data = {
                "hist": hist[-HIST_MAX_IN_WINDOW * 2:],
                "sig": sigs,
                "burst": pruned_burst,
            }
            self._write(fh, data)
            return wait

    # -- ledger file I/O (operates on an already-locked handle) --
    @staticmethod
    def _read(fh) -> dict:
        fh.seek(0)
        raw = fh.read()
        if not raw.strip():
            return {}
        try:
            import json
            return json.loads(raw)
        except ValueError:
            return {}

    @staticmethod
    def _write(fh, data: dict) -> None:
        import json
        fh.seek(0)
        fh.truncate()
        fh.write(json.dumps(data))
        fh.flush()
        os.fsync(fh.fileno())

    @contextlib.contextmanager
    def _locked(self):
        # Open r+ (create if absent), take an exclusive advisory lock, ensure
        # the file is 0600 (it records only timestamps, but stays private).
        assert self._path is not None
        if not self._path.exists():
            self._path.touch(mode=0o600)
        else:
            with contextlib.suppress(OSError):
                self._path.chmod(0o600)
        fh = open(self._path, "r+", encoding="utf-8")
        try:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            yield fh
        finally:
            if fcntl is not None:
                with contextlib.suppress(OSError):
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            fh.close()


# ---------------------------------------------------------------------------
# F2/F3 — central error sink
# ---------------------------------------------------------------------------
def _is_permission_162(text: str) -> bool:
    """A code-162 that is a permission problem, not a pacing violation.

    Note: we deliberately do NOT match ``"hmds"`` here — IB's HMDS server emits
    code 162 for *both* a missing entitlement and a legitimately-empty window
    (``HMDS query returned no data``). Keying off ``"hmds"`` swallowed the
    no-data case as a permission error (F3); that outcome is now split out by
    ``_is_no_data_162`` and surfaced as an ``empty_window`` success.
    """
    low = (text or "").lower()
    return "market data permission" in low or "no market data" in low


def _is_no_data_162(text: str) -> bool:
    """A code-162 that means the request was valid but the window held no data.

    IB's HMDS server returns ``HMDS query returned no data`` for a thin/illiquid
    bond, an off-hours request, or a series that simply has no bars (e.g. YIELD
    on a rarely-traded instrument). This is the F3 THIRD outcome — a success
    with zero bars, distinct from a missing subscription (``not_subscribed``)
    and a pacing violation (``pacing``).
    """
    return "returned no data" in (text or "").lower()


class _ErrorSink:
    """Collects IB ``errorEvent`` messages for the in-flight request window.

    ib_async surfaces missing-subscription conditions as an ``errorEvent`` while
    the returned Ticker/bars/headlines stay nan/empty — no exception. Since the
    CLI issues one request at a time, we clear the sink, run the request, then
    inspect what fired to convert a silent nan into a typed failure. Fatal codes
    are raised eagerly by the caller after the window.
    """

    def __init__(self):
        self.records: list[tuple[int, int, str]] = []

    def handle(self, reqId=-1, errorCode=-1, errorString="", *_extra) -> None:
        # Signature is tolerant of ib_async version drift (later versions append
        # a contract / advancedOrderReject arg). Benign farm notices are dropped.
        try:
            code = int(errorCode)
        except (TypeError, ValueError):
            return
        if code in _BENIGN_CODES:
            return
        self.records.append((int(reqId) if reqId is not None else -1, code, str(errorString)))

    def clear(self) -> None:
        self.records.clear()

    @property
    def codes(self) -> list[int]:
        return [c for _, c, _ in self.records]

    def raise_if_fatal(self) -> None:
        for _reqId, code, text in self.records:
            if code in _FATAL_CODES:
                raise IBConnectionError(f"IB fatal error {code}: {text}")

    def subscription_problem(self) -> Optional[tuple[list[int], str]]:
        """Return (codes, message) if a not-subscribed condition fired, else None."""
        hits: list[int] = []
        detail = ""
        for _reqId, code, text in self.records:
            if code in _NOT_SUBSCRIBED_CODES or (code == _PERMISSION_162 and _is_permission_162(text)):
                hits.append(code)
                detail = detail or text
        return (hits, detail) if hits else None

    def pacing_problem(self) -> bool:
        for _reqId, code, text in self.records:
            if code in _PACING_CODES or (code == _PERMISSION_162 and not _is_permission_162(text)
                                         and "pacing" in (text or "").lower()):
                return True
        return False

    def empty_window(self) -> Optional[str]:
        """Return the detail of a no-data 162 (valid query, zero bars), else None.

        The F3 THIRD outcome: distinct from ``subscription_problem`` and
        ``pacing_problem`` — the request was entitled and correctly paced, the
        window simply held no data.
        """
        for _reqId, code, text in self.records:
            if code == _PERMISSION_162 and _is_no_data_162(text):
                return text
        return None


# ---------------------------------------------------------------------------
# Connection (read-only, connect-per-call)
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def ib_session(config_file: str = "") -> Iterator[tuple["object", _ErrorSink, dict]]:
    """Yield a connected, **read-only** IB handle plus its error sink and config.

    Connects with ``readonly=True`` — a handshake assertion to TWS/Gateway that
    this client will not place orders. It is a belt-and-suspenders signal, NOT
    the guarantee: the real guarantee is that this package contains no
    order/trade/placement code path at all, enforced by the CI AST-scan
    (``tests/test_ib_readonly_guard.py``). Do not rely on ``readonly=True``
    alone. The error sink is attached to ``errorEvent`` before connecting so
    subscription/permission errors are captured from the first message. The
    connection is always torn down in ``finally``.
    """
    try:
        from ib_async import IB
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise IBError(
            "ib_async is not installed. Add `ib_async` to the runner "
            "environment (pip install ib_async)."
        ) from exc

    cfg = _load_config(config_file)
    sink = _ErrorSink()
    ib = IB()
    ib.errorEvent += sink.handle
    try:
        ib.connect(
            cfg["host"],
            int(cfg["port"]),
            clientId=int(cfg["client_id"]),
            timeout=float(cfg["connect_timeout_s"]),
            readonly=True,
        )
    except Exception as exc:  # ib_async raises assorted asyncio/socket errors
        with contextlib.suppress(Exception):
            ib.disconnect()
        # Do NOT echo host/port in a way that could leak a private topology at
        # info level; a bounded message is enough for the operator.
        raise IBConnectionError(
            "Could not connect to TWS / IB Gateway. Is it running and logged in, "
            "and is the API port + Read-Only API enabled? "
            f"({type(exc).__name__})"
        ) from exc

    # Default LIVE market data; a fetch may switch to delayed only under an
    # explicit allow_delayed flag (F6).
    with contextlib.suppress(Exception):
        ib.reqMarketDataType(int(cfg.get("market_data_type", MKT_DATA_LIVE)))
    try:
        yield ib, sink, cfg
    finally:
        with contextlib.suppress(Exception):
            ib.disconnect()


def _resolve_account(ib, account: str, cfg: dict) -> str:
    """Resolve the single account to query (decision #5 — single account).

    Order: explicit arg > config > the login's sole managed account. If the
    login manages several accounts and none was specified, raise rather than
    guess — this milestone is single-account only.
    """
    explicit = account or cfg.get("account") or ""
    managed = list(ib.managedAccounts() or [])
    if explicit:
        if managed and explicit not in managed:
            raise IBError(f"Account {explicit!r} is not among managed accounts {managed}.")
        return explicit
    if len(managed) == 1:
        return managed[0]
    if not managed:
        return ""  # some gateways report none until account data loads; let IB default
    raise IBError(
        f"Login manages multiple accounts {managed}; pass --account "
        "(single-account scope this milestone)."
    )


# ---------------------------------------------------------------------------
# F4 — contract resolution (equity vs bond)
# ---------------------------------------------------------------------------
def _resolve_contract(ib, ident: str, sec_type: str):
    """Resolve ``ident`` to a conId-pinned IB contract.

    EQUITY (``STK``): ``Stock(ident, "SMART", "USD")`` qualified to a conId.
    BOND: build ``Contract(secType="BOND", secIdType="CUSIP", secId=ident)`` and
    require ``reqContractDetails`` to return exactly one match — 0 or >1 is a
    ``ContractResolutionError`` so an ambiguous CUSIP never silently picks a leg.
    """
    from ib_async import Stock, Contract

    st = (sec_type or "STK").upper()
    if st in ("STK", "EQUITY", "STOCK"):
        contract = Stock(ident, "SMART", "USD")
        qualified = ib.qualifyContracts(contract)
        if not qualified or not getattr(qualified[0], "conId", 0):
            raise ContractResolutionError(f"Could not qualify equity {ident!r}.")
        return qualified[0]

    if st in ("BOND", "BND"):
        contract = Contract(
            secType="BOND", secIdType="CUSIP", secId=ident,
            exchange="SMART", currency="USD",
        )
        details = ib.reqContractDetails(contract)
        conids = {getattr(d.contract, "conId", 0) for d in (details or [])}
        conids.discard(0)
        if not conids:
            raise ContractResolutionError(f"CUSIP {ident!r} did not resolve to any IB bond.")
        if len(conids) > 1:
            raise ContractResolutionError(
                f"CUSIP {ident!r} resolved to multiple bonds {sorted(conids)}; ambiguous."
            )
        return details[0].contract

    raise IBError(f"Unsupported sec_type {sec_type!r}; use STK or BOND.")


# ---------------------------------------------------------------------------
# F4 — bond economics (the single home for equity/bond divergence)
# ---------------------------------------------------------------------------
def _bond_economics(face: float, clean_price: Optional[float],
                    accrued_interest: Optional[float]) -> dict:
    """Normalize bond pricing: quote clean, report accrued separately (decision #4).

    Bonds quote as a **percent of par** ("clean" price, coupon accrual excluded).
    We report the clean price as the quote and surface ``accrued_interest`` as
    its own field — never folded into the quoted price. ``dirty_price`` is
    derived only when accrued is known. Market value uses the clean leg plus
    accrued so a caller can see both. Equity paths never touch this function.
    """
    result = {
        "price_convention": "percent_of_par",
        "clean_price": clean_price,
        "accrued_interest": accrued_interest,
        "dirty_price": None,
        "market_value_clean": None,
        "market_value_dirty": None,
    }
    if clean_price is not None:
        result["market_value_clean"] = face * clean_price / 100.0
        if accrued_interest is not None:
            # accrued is already a cash amount for the held face; add it on.
            result["dirty_price"] = clean_price + (accrued_interest / face * 100.0 if face else 0.0)
            result["market_value_dirty"] = result["market_value_clean"] + accrued_interest
    return result


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _is_nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _clean(x):
    """nan -> None so payloads stay JSON-clean and never imply a real 0/NaN."""
    return None if _is_nan(x) else x


def _contract_row(contract) -> dict:
    """Flatten the read-only identity fields off an IB contract."""
    return {
        "conid": getattr(contract, "conId", None),
        "secType": getattr(contract, "secType", None),
        "symbol": getattr(contract, "symbol", None),
        "cusip": getattr(contract, "secId", None) if getattr(contract, "secType", "") == "BOND" else None,
        "currency": getattr(contract, "currency", None),
        "exchange": getattr(contract, "exchange", None),
        "multiplier": getattr(contract, "multiplier", None) or None,
    }


def _is_bond(contract) -> bool:
    return getattr(contract, "secType", "") == "BOND"


# ---------------------------------------------------------------------------
# Account data (no market-data subscription needed)
# ---------------------------------------------------------------------------
def get_positions(config_file: str = "", account: str = "") -> list[dict]:
    """Return open positions. Account data — no market-data subscription, no trap.

    An empty list means genuinely flat (and the caller says so), never a hidden
    subscription failure. ``secType`` is preserved so callers branch equity/bond.
    """
    with ib_session(config_file) as (ib, sink, cfg):
        acct = _resolve_account(ib, account, cfg)
        ib.sleep(0.25)  # let the account/position snapshot arrive
        rows = []
        for pos in ib.positions(acct) if acct else ib.positions():
            row = _contract_row(pos.contract)
            row["account"] = pos.account
            row["position"] = pos.position          # equity: shares; bond: face value
            row["avgCost"] = pos.avgCost
            row["price_convention"] = "percent_of_par" if _is_bond(pos.contract) else "per_share"
            rows.append(row)
        return rows


def get_cost_basis(config_file: str = "", account: str = "") -> list[dict]:
    """Return cost basis per position from ``avgCost``. Account data — no sub.

    EQUITY: ``cost_basis = avgCost * position``.
    BOND: IB's ``avgCost`` already folds par-scaling/multiplier, and the position
    quantity is the face value; the row is tagged ``percent_of_par`` and routed
    through ``_bond_economics`` so the divergence lives in one place (F4).
    """
    with ib_session(config_file) as (ib, sink, cfg):
        acct = _resolve_account(ib, account, cfg)
        ib.sleep(0.25)
        rows = []
        for pos in ib.positions(acct) if acct else ib.positions():
            row = _contract_row(pos.contract)
            row["account"] = pos.account
            row["position"] = pos.position
            row["avgCost"] = pos.avgCost
            if _is_bond(pos.contract):
                row["price_convention"] = "percent_of_par"
                # avgCost is IB's par-scaled cost per bond * face already; the
                # cash basis is avgCost * position. Accrued at purchase is not
                # reconstructable read-only, so it stays a separate (null) field.
                row["cost_basis"] = pos.avgCost * pos.position
                row["accrued_interest"] = None
            else:
                row["price_convention"] = "per_share"
                row["cost_basis"] = pos.avgCost * pos.position
            rows.append(row)
        return rows


def get_pnl(config_file: str = "", account: str = "",
            conids: Optional[list[int]] = None) -> list[dict]:
    """Return per-position PnL. Realized/daily = account data; unrealized needs live price.

    Uses ``reqPnLSingle`` per position. The unrealized leg needs a live market
    price for that conId; if it comes back nan **and** a not-subscribed code
    fired for the request window, raise ``MarketDataNotSubscribedError`` naming
    the contract — never emit ``unrealizedPnL: null`` silently (F2). Bonds route
    through ``_bond_economics`` for the value leg.
    """
    with ib_session(config_file) as (ib, sink, cfg):
        acct = _resolve_account(ib, account, cfg)
        ib.sleep(0.25)
        positions = list(ib.positions(acct) if acct else ib.positions())
        want = set(conids or [])
        rows = []
        for pos in positions:
            conid = getattr(pos.contract, "conId", None)
            if want and conid not in want:
                continue
            sink.clear()
            single = ib.reqPnLSingle(pos.account, "", conid)
            deadline = time.monotonic() + 6.0
            while time.monotonic() < deadline:
                ib.sleep(0.25)
                if not _is_nan(getattr(single, "unrealizedPnL", None)):
                    break
            sink.raise_if_fatal()
            row = _contract_row(pos.contract)
            row["account"] = pos.account
            row["position"] = pos.position
            row["dailyPnL"] = _clean(getattr(single, "dailyPnL", None))
            row["realizedPnL"] = _clean(getattr(single, "realizedPnL", None))
            unreal = getattr(single, "unrealizedPnL", None)
            market_value = getattr(single, "value", None)
            with contextlib.suppress(Exception):
                ib.cancelPnLSingle(pos.account, "", conid)
            if _is_nan(unreal):
                problem = sink.subscription_problem()
                if problem:
                    codes, detail = problem
                    raise MarketDataNotSubscribedError(
                        row.get("symbol") or str(conid),
                        f"unrealized PnL needs a live price. {detail}", codes,
                    )
                # nan with no explanatory error: still not a silent success.
                row["unrealizedPnL"] = None
                row["unrealized_status"] = "unavailable (no live price / timeout)"
            else:
                row["unrealizedPnL"] = unreal
            if _is_bond(pos.contract):
                # Back out an approximate %-of-par price from the position value
                # (value = face * price/100). Accrued is not separable read-only,
                # so it stays a null field rather than being folded into price.
                clean_px = _clean(market_value / pos.position * 100.0) if market_value and pos.position else None
                row["bond"] = _bond_economics(pos.position, clean_px, None)
            else:
                row["marketValue"] = _clean(market_value)
            rows.append(row)
        return rows


# ---------------------------------------------------------------------------
# Prices (market-data subscription REQUIRED — the silent-empty trap)
# ---------------------------------------------------------------------------
def get_current_price(ident: str, sec_type: str = "STK",
                      allow_delayed: bool = False, config_file: str = "") -> dict:
    """Snapshot current price. Market-data subscription REQUIRED (F2/F6).

    Defaults to LIVE. ``allow_delayed=True`` switches to delayed (type 3) and
    tags the payload ``data_type="delayed"`` so delayed never masquerades as live.
    The trap: an unsubscribed feed does not throw — the Ticker stays nan while IB
    emits 354/10089/10167 on errorEvent. After a bounded deadline we consult the
    captured errors: a not-subscribed code -> ``MarketDataNotSubscribedError``;
    nan with no error -> ``IBError`` (still never a silent empty success).
    """
    with ib_session(config_file) as (ib, sink, cfg):
        contract = _resolve_contract(ib, ident, sec_type)
        data_type = "live"
        if allow_delayed:
            ib.reqMarketDataType(MKT_DATA_DELAYED)
            data_type = "delayed"
        else:
            ib.reqMarketDataType(MKT_DATA_LIVE)

        PacingLedger(_pacing_ledger_path()).gate(f"px:{sec_type}:{ident}")
        sink.clear()
        ticker = ib.reqMktData(contract, "", snapshot=True, regulatorySnapshot=False)
        try:
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                ib.sleep(0.25)
                if not _is_nan(getattr(ticker, "last", None)) or not _is_nan(getattr(ticker, "close", None)):
                    break
            sink.raise_if_fatal()
            last = getattr(ticker, "last", None)
            close = getattr(ticker, "close", None)
            bid = getattr(ticker, "bid", None)
            ask = getattr(ticker, "ask", None)
            if _is_nan(last) and _is_nan(close) and _is_nan(bid) and _is_nan(ask):
                problem = sink.subscription_problem()
                if problem:
                    codes, detail = problem
                    raise MarketDataNotSubscribedError(ident, detail or "price feed not entitled", codes)
                raise IBError(f"No price returned for {ident!r} within deadline (no data / timeout).")

            payload = {
                "ident": ident,
                "contract": _contract_row(contract),
                "data_type": data_type,
                "last": _clean(last),
                "close": _clean(close),
                "bid": _clean(bid),
                "ask": _clean(ask),
            }
            if _is_bond(contract):
                # Bond ticks are % of par (clean). Report clean + accrued separately.
                clean_px = _clean(last) if not _is_nan(last) else _clean(close)
                payload["bond"] = _bond_economics(100.0, clean_px, None)
            return payload
        finally:
            with contextlib.suppress(Exception):
                ib.cancelMktData(contract)


def get_historic_prices(ident: str, sec_type: str = "STK", duration: str = "1 M",
                        bar_size: str = "1 day", what: str = "TRADES",
                        allow_delayed: bool = False, config_file: str = "") -> dict:
    """Historic bars. Market-data subscription REQUIRED; pacing enforced (F1/F2/F3).

    Empty bars resolve to one of THREE distinct outcomes (F3), never a silent
    ``[]``:
      * ``not_subscribed`` — a permission-162 / not-entitled code fired ->
        raises ``MarketDataNotSubscribedError``.
      * ``pacing`` — a pacing-162 / 420 fired -> raises ``IBPacingError``
        (retryable with backoff).
      * ``empty_window`` — the query was valid and correctly paced but the
        window held no data (IB's ``HMDS query returned no data``: a thin/
        illiquid bond, off-hours, or a sparse YIELD series) -> returns
        SUCCESS with ``bars: []`` and ``status="empty_window"``.
    The pacing ledger gates before the request (F1). BOND ``TRADES`` history is
    often sparse; ``YIELD``/``BID_ASK`` are documented alternatives and bars are
    %-of-par for bonds.
    """
    with ib_session(config_file) as (ib, sink, cfg):
        contract = _resolve_contract(ib, ident, sec_type)
        data_type = "live"
        if allow_delayed:
            ib.reqMarketDataType(MKT_DATA_DELAYED)
            data_type = "delayed"
        else:
            ib.reqMarketDataType(MKT_DATA_LIVE)

        signature = f"hist:{sec_type}:{ident}:{what}:{duration}:{bar_size}"
        PacingLedger(_pacing_ledger_path()).gate(signature)
        sink.clear()
        bars = ib.reqHistoricalData(
            contract, endDateTime="", durationStr=duration, barSizeSetting=bar_size,
            whatToShow=what, useRTH=True, formatDate=1,
        )
        sink.raise_if_fatal()
        convention = "percent_of_par" if _is_bond(contract) else "per_share"
        if not bars:
            problem = sink.subscription_problem()
            if problem:
                codes, detail = problem
                raise MarketDataNotSubscribedError(ident, detail or "historical data not entitled", codes)
            if sink.pacing_problem():
                raise IBPacingError(f"Historical pacing violation for {ident!r}; retry with backoff.")
            # F3 THIRD outcome: valid, entitled, correctly-paced query that the
            # window simply had no data for — a SUCCESS with zero bars, NOT a
            # subscription failure. A no-data 162 (HMDS "returned no data") or a
            # clean empty response both land here.
            empty_detail = sink.empty_window()
            return {
                "ident": ident,
                "contract": _contract_row(contract),
                "data_type": data_type,
                "what": what,
                "bar_size": bar_size,
                "duration": duration,
                "price_convention": convention,
                "status": "empty_window",
                "reason": empty_detail or "no data in the requested window",
                "bars": [],
            }

        return {
            "ident": ident,
            "contract": _contract_row(contract),
            "data_type": data_type,
            "what": what,
            "bar_size": bar_size,
            "duration": duration,
            "price_convention": convention,
            "status": "ok",
            "bars": [
                {
                    "date": str(getattr(b, "date", "")),
                    "open": _clean(getattr(b, "open", None)),
                    "high": _clean(getattr(b, "high", None)),
                    "low": _clean(getattr(b, "low", None)),
                    "close": _clean(getattr(b, "close", None)),
                    "volume": _clean(getattr(b, "volume", None)),
                }
                for b in bars
            ],
        }


# ---------------------------------------------------------------------------
# News (per-provider subscription REQUIRED)
# ---------------------------------------------------------------------------
def list_news_providers(config_file: str = "") -> list[dict]:
    """Return news providers the account is subscribed to.

    ``reqNewsProviders`` returns ONLY subscribed providers, so this is itself the
    subscription probe: a provider you expect but don't see = not subscribed.
    """
    with ib_session(config_file) as (ib, sink, cfg):
        ib.sleep(0.25)
        providers = ib.reqNewsProviders() or []
        return [{"code": p.code, "name": p.name} for p in providers]


def get_news_headlines(ident: str, sec_type: str = "STK", lookback_days: int = 7,
                       providers: Optional[list[str]] = None, config_file: str = "") -> dict:
    """Historical news headlines for a contract. Per-provider subscription REQUIRED.

    Requested providers are intersected with the subscribed set up front; a
    requested-but-unsubscribed provider raises ``MarketDataNotSubscribedError``
    rather than silently returning fewer/zero headlines (F2).
    """
    from datetime import datetime, timedelta, timezone

    with ib_session(config_file) as (ib, sink, cfg):
        contract = _resolve_contract(ib, ident, sec_type)
        subscribed = {p.code: p.name for p in (ib.reqNewsProviders() or [])}
        if providers:
            missing = [p for p in providers if p not in subscribed]
            if missing:
                raise MarketDataNotSubscribedError(
                    ident, f"news provider(s) not subscribed: {missing}. "
                    f"Subscribed: {sorted(subscribed)}",
                )
            use = list(providers)
        else:
            use = list(subscribed)
        if not use:
            raise MarketDataNotSubscribedError(ident, "no subscribed news providers on this account")

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback_days)
        fmt = "%Y-%m-%d %H:%M:%S"
        conid = getattr(contract, "conId", 0)
        PacingLedger(_pacing_ledger_path()).gate(f"news:{conid}:{','.join(sorted(use))}")
        sink.clear()
        articles = ib.reqHistoricalNews(
            conId=conid, providerCodes="+".join(use),
            startDateTime=start.strftime(fmt), endDateTime=end.strftime(fmt),
            totalResults=100,
        )
        sink.raise_if_fatal()
        headlines = [
            {
                "time": str(getattr(a, "time", "")),
                "providerCode": getattr(a, "providerCode", None),
                "articleId": getattr(a, "articleId", None),
                "headline": getattr(a, "headline", None),
            }
            for a in (articles or [])
        ]
        return {
            "ident": ident,
            "contract": _contract_row(contract),
            "providers_used": use,
            "lookback_days": lookback_days,
            "headlines": headlines,
        }


def get_news_article(provider_code: str, article_id: str, config_file: str = "") -> dict:
    """Fetch a full news-article body for a headline id. Not-permissioned -> typed error."""
    with ib_session(config_file) as (ib, sink, cfg):
        sink.clear()
        article = ib.reqNewsArticle(provider_code, article_id)
        sink.raise_if_fatal()
        problem = sink.subscription_problem()
        if problem and article is None:
            codes, detail = problem
            raise MarketDataNotSubscribedError(article_id, detail or "news article not entitled", codes)
        return {
            "providerCode": provider_code,
            "articleId": article_id,
            "articleType": getattr(article, "articleType", None),
            "articleText": getattr(article, "articleText", None),
        }


# ---------------------------------------------------------------------------
# Doctor — connectivity + read-only + subscription self-check (no data returned)
# ---------------------------------------------------------------------------
def doctor(config_file: str = "") -> dict:
    """Probe the setup without returning market data.

    Reports connectivity, the connected account id + paper/live flag (so a
    wrong-port connection is caught, F6), whether the API session is read-only,
    and which news providers are subscribed. Never fetches positions or prices.
    """
    report: dict = {"connected": False, "readonly": True}
    try:
        with ib_session(config_file) as (ib, sink, cfg):
            report["connected"] = True
            accounts = list(ib.managedAccounts() or [])
            report["managed_accounts"] = accounts
            # IB paper accounts start with 'D' (DU/DF); live start with 'U'.
            report["account_kind"] = (
                "paper" if accounts and accounts[0].startswith("D")
                else "live" if accounts else "unknown"
            )
            report["market_data_type_default"] = int(cfg.get("market_data_type", MKT_DATA_LIVE))
            report["news_providers"] = [p.code for p in (ib.reqNewsProviders() or [])]
            report["server_version"] = getattr(ib.client, "serverVersion", lambda: None)()
    except IBError as exc:
        report["error"] = str(exc)
    return report
