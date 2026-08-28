#!/usr/bin/env python3
"""Guaranteed-schedule, lightweight SPY intraday Gamma archive collector.

This entrypoint is intentionally restricted to SPY and the 10:00 through
15:30 New York archive buckets. It performs no Range Finder, model, spread,
historical-data, export, UI, or trading work.
"""
from __future__ import annotations

from datetime import datetime, time, timezone
import logging
import os
import sys

from phase1.data_client import TradierDataClient
from phase1.gamma_archive_capture import calculate_gamma_archive_observation
from phase1.gamma_level_history import (
    gamma_archive_bucket,
    gamma_level_snapshot_kwargs,
    save_gamma_level_snapshot,
)
from phase1.market_clock import now_ny
from phase1.rates import fetch_risk_free_rate


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RECOMMENDED_ARCHIVE_TICKER = "SPY"
FIRST_INTRADAY_BUCKET = time(10, 0)
LAST_INTRADAY_BUCKET = time(15, 30)


def capture_intraday_gamma_archive(
    *,
    captured_at: datetime | None = None,
    environ: dict | None = None,
    client_factory=None,
    rate_loader=None,
    calculation_func=None,
    save_func=None,
    clock=None,
) -> int:
    """Capture the current SPY bucket; return a process-style exit code.

    Time/calendar validation happens before credentials and dependency
    construction, proving invalid runs perform no market-data or database work.
    Test seams are dependency injection only; production has no ticker input and
    therefore cannot fan out beyond the one hard-coded SPY calculation.
    """
    clock = clock or now_ny
    gate_at = captured_at or clock()
    bucket = gamma_archive_bucket(gate_at)
    if bucket is None:
        log.info("No valid NYSE RTH archive bucket — exiting without I/O")
        return 0
    if not (FIRST_INTRADAY_BUCKET <= bucket.bucket_start <= LAST_INTRADAY_BUCKET):
        log.info(
            "Bucket %s is owned by the opening job or outside intraday scope — "
            "exiting without I/O",
            bucket.bucket_start.strftime("%H:%M"),
        )
        return 0

    env = environ if environ is not None else os.environ
    tradier_token = str(env.get("TRADIER_TOKEN", "")).strip()
    if not tradier_token:
        log.error("TRADIER_TOKEN not set — cannot calculate SPY Gamma")
        return 1
    if not str(env.get("DATABASE_URL", "")).strip():
        log.error("DATABASE_URL not set — cannot archive SPY Gamma")
        return 1

    ticker = RECOMMENDED_ARCHIVE_TICKER
    client_factory = client_factory or TradierDataClient
    rate_loader = rate_loader or fetch_risk_free_rate
    calculation_func = calculation_func or calculate_gamma_archive_observation
    save_func = save_func or save_gamma_level_snapshot

    try:
        rate_info = rate_loader(
            str(env.get("FRED_API_KEY", "")).strip(),
            prefer_same_day_cache=True,
            now=gate_at,
        )
        client = client_factory(token=tradier_token)
        available_expirations = client.get_expirations(ticker)
        if not available_expirations:
            log.error("No expirations returned from Tradier for SPY")
            return 1

        # Timestamp immediately before the quote/chain calculation. If rate or
        # expiration setup crossed a bucket boundary, values belong only to the
        # new contemporaneous bucket; never file them under the earlier gate.
        observation_at = captured_at or clock()
        observation_bucket = gamma_archive_bucket(observation_at)
        if observation_bucket is None or not (
            FIRST_INTRADAY_BUCKET
            <= observation_bucket.bucket_start
            <= LAST_INTRADAY_BUCKET
        ):
            log.info("Calculation setup crossed outside intraday RTH — exiting before chain fetch")
            return 0
        bucket = observation_bucket

        calculation = calculation_func(
            client=client,
            ticker=ticker,
            available_expirations=available_expirations,
            captured_at=observation_at,
            risk_free_rate=rate_info["rate"],
            risk_free_curve=rate_info.get("curve"),
        )
    except Exception as exc:
        log.error("SPY Gamma calculation failed: %s", exc)
        return 1

    if calculation.levels is None or calculation.regime_info is None:
        log.warning(
            "SPY Gamma calculation produced no usable chain result; "
            "leaving bucket %s missing",
            bucket.bucket_start.strftime("%H:%M"),
        )
        return 1

    try:
        inserted = save_func(**gamma_level_snapshot_kwargs(
            captured_at=calculation.captured_at,
            ticker=ticker,
            spot=calculation.spot,
            levels=calculation.levels,
            regime_info=calculation.regime_info,
            stats=calculation.stats,
            confidence_info=calculation.confidence_info,
            staleness_info=calculation.staleness_info,
            em_analysis=calculation.em_analysis,
        ))
    except Exception as exc:
        log.warning("Gamma archive save failed: %s", exc)
        return 1

    if inserted:
        log.info(
            "Archived SPY Gamma observation at %s UTC into %s NY bucket",
            calculation.captured_at.astimezone(timezone.utc).isoformat(),
            bucket.bucket_start.strftime("%H:%M"),
        )
    else:
        log.info(
            "SPY Gamma bucket %s already exists; first observation preserved",
            bucket.bucket_start.strftime("%H:%M"),
        )
    return 0


def main() -> None:
    sys.exit(capture_intraday_gamma_archive())


if __name__ == "__main__":
    main()
