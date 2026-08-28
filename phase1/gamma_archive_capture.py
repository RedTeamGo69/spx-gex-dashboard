"""Canonical production calculation for forward Gamma archive observations.

Both the scheduled 09:30 capture and the lightweight SPY intraday collector
call this module. Keeping expiration selection and the full derived-level
calculation here prevents the two historical sources from drifting while
leaving all GEX, zero-gamma, wall, parity, rate, dividend, confidence,
freshness, regime, and expected-move formulas in their existing modules.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from phase1.confidence import build_run_confidence
from phase1.config import NY_TZ
from phase1.expected_move import build_expected_move_analysis
import phase1.gex_engine as gex_engine
from phase1.market_clock import get_calendar_snapshot
from phase1.parity import get_reference_spot_details
from phase1.staleness import build_staleness_info
from phase1.ticker_config import get_config


CANONICAL_EXPIRATION_COUNT = 4


@dataclass(frozen=True)
class GammaArchiveCalculation:
    """Already-computed canonical archive inputs; no persistence side effects."""

    captured_at: datetime
    ticker: str
    target_expirations: tuple[str, ...]
    quote: dict
    spot: float
    spot_info: dict
    gex_df: Any
    stats: dict
    all_options: list
    levels: dict | None
    regime_info: dict | None
    staleness_info: dict
    confidence_info: dict
    em_analysis: dict | None


def canonical_gamma_expirations(
    available_expirations: list[str],
    session_date: date,
) -> tuple[str, ...]:
    """Return the production archive universe: nearest four live date slots.

    On a normal SPY session this is the same-day/0DTE expiration plus the next
    three listed expirations. If no same-day contract exists, it is simply the
    next four listed dates. Dates before the New York session are never used.
    """
    session_key = session_date.isoformat()
    return tuple(
        expiration
        for expiration in available_expirations
        if expiration >= session_key
    )[:CANONICAL_EXPIRATION_COUNT]


def _spot_from_full_quote(quote: dict) -> float:
    """Preserve TradierDataClient.get_spot_price's last→close→prevclose rule."""
    for field in ("last", "close", "prevclose"):
        try:
            value = float(quote.get(field) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        if value > 0:
            return value
    return 0.0


def calculate_gamma_archive_observation(
    *,
    client,
    ticker: str,
    available_expirations: list[str],
    captured_at: datetime,
    risk_free_rate: float,
    risk_free_curve: dict | None,
) -> GammaArchiveCalculation:
    """Run the exact production Gamma archive calculation, without saving.

    A single quote response supplies both the parity fallback spot and
    previous close. The nearest chain fetched for parity is reused by the GEX
    engine; the engine fetches only the remaining canonical expirations, and
    expected move reuses that same nearest cached chain.

    An empty GEX result is returned with ``levels=None``. The scheduled job can
    retain its force-mode operational fallback without archiving it, while the
    intraday research collector can fail closed and leave the bucket missing.
    """
    if captured_at.tzinfo is None or captured_at.utcoffset() is None:
        raise ValueError("captured_at must be timezone-aware")
    captured_at = captured_at.astimezone(NY_TZ)
    target_expirations = canonical_gamma_expirations(
        available_expirations, captured_at.date()
    )
    if not target_expirations:
        raise RuntimeError(
            f"No non-expired option expirations available for {ticker} "
            f"on {captured_at.date().isoformat()}"
        )

    quote = client.get_full_quote(ticker)
    quote_spot = _spot_from_full_quote(quote)
    nearest_expiration = target_expirations[0]
    spot_info = get_reference_spot_details(
        ticker=ticker,
        nearest_exp=nearest_expiration,
        get_spot_price_func=lambda _ticker: quote_spot,
        get_chain_cached_func=client.get_chain_cached,
        r=risk_free_rate,
        now=captured_at,
        r_curve=risk_free_curve,
    )
    spot = spot_info["spot"]

    gex_df, stats, all_options, _strike_support, _expiration_support = (
        gex_engine.calculate_all(
            client,
            ticker,
            list(target_expirations),
            spot,
            r=risk_free_rate,
            now=captured_at,
            r_curve=risk_free_curve,
        )
    )

    calendar_snapshot = get_calendar_snapshot(captured_at)
    has_0dte = captured_at.date().isoformat() in target_expirations
    staleness_info = build_staleness_info(
        calendar_snapshot, spot_info, stats, has_0dte=has_0dte
    )
    confidence_info = build_run_confidence(
        stats, spot_info, staleness_info=staleness_info
    )

    if gex_df.empty:
        return GammaArchiveCalculation(
            captured_at=captured_at,
            ticker=ticker,
            target_expirations=target_expirations,
            quote=quote,
            spot=spot,
            spot_info=spot_info,
            gex_df=gex_df,
            stats=stats,
            all_options=all_options,
            levels=None,
            regime_info=None,
            staleness_info=staleness_info,
            confidence_info=confidence_info,
            em_analysis=None,
        )

    dividend_yield = float(get_config(ticker).get("dividend_yield", 0.0) or 0.0)
    levels = gex_engine.find_key_levels(
        gex_df,
        spot,
        all_options=all_options,
        r=risk_free_rate,
        r_curve=risk_free_curve,
        q=dividend_yield,
    )
    regime_info = gex_engine.get_gamma_regime_text(spot, levels["zero_gamma"])

    nearest_chain = client.get_chain_cached(ticker, nearest_expiration)
    calls = nearest_chain.get("calls", []) if nearest_chain.get("status") == "ok" else []
    puts = nearest_chain.get("puts", []) if nearest_chain.get("status") == "ok" else []
    em_analysis = build_expected_move_analysis(
        spot=spot,
        prev_close=float(quote.get("prevclose") or 0.0),
        zero_gamma=levels["zero_gamma"],
        gamma_regime=regime_info.get("regime", "Unknown"),
        calls_0dte=calls,
        puts_0dte=puts,
        market_open=True,
        expiration=nearest_expiration,
        as_of=captured_at.date(),
    )

    return GammaArchiveCalculation(
        captured_at=captured_at,
        ticker=ticker,
        target_expirations=target_expirations,
        quote=quote,
        spot=spot,
        spot_info=spot_info,
        gex_df=gex_df,
        stats=stats,
        all_options=all_options,
        levels=levels,
        regime_info=regime_info,
        staleness_info=staleness_info,
        confidence_info=confidence_info,
        em_analysis=em_analysis,
    )
