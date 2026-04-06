# =============================================================================
# spread_persistence.py
# Spread plan DB logging, outcome tracking, and pretty-printing.
# =============================================================================

import sqlite3
import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)


def init_spread_log_table(conn) -> None:
    """Ensure spread_log table exists.
    Now handled by db.init_all_tables() — kept for backwards compatibility."""
    pass  # Tables created in db.init_all_tables()


def log_spread_plan(
    conn: sqlite3.Connection,
    plan,
    wing_width_used: int = None,
) -> None:
    """Persist a SpreadPlan to spread_log."""
    width = wing_width_used or plan.recommended_width

    call = next((s for s in plan.call_spreads if s.wing_width == width), None)
    put  = next((s for s in plan.put_spreads  if s.wing_width == width), None)

    now = datetime.now(timezone.utc).isoformat()

    conn.execute("""
        INSERT INTO spread_log (
            week_start, generated_at,
            spx_ref_close, point_pct, upper_pct, effective_range_pct,
            call_short, call_long, put_short, put_long,
            wing_width_used, buffer_pct, event_count, gex_flag,
            warnings, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(week_start) DO UPDATE SET
            generated_at        = excluded.generated_at,
            spx_ref_close       = excluded.spx_ref_close,
            point_pct           = excluded.point_pct,
            upper_pct           = excluded.upper_pct,
            effective_range_pct = excluded.effective_range_pct,
            call_short          = excluded.call_short,
            call_long           = excluded.call_long,
            put_short           = excluded.put_short,
            put_long            = excluded.put_long,
            wing_width_used     = excluded.wing_width_used,
            buffer_pct          = excluded.buffer_pct,
            event_count         = excluded.event_count,
            gex_flag            = excluded.gex_flag,
            warnings            = excluded.warnings,
            updated_at          = excluded.updated_at
    """, (
        plan.week_start,
        plan.generated_at,
        plan.spx_ref_close,
        plan.point_pct,
        plan.upper_pct,
        plan.effective_range_pct,
        call.short_strike if call else None,
        call.long_strike  if call else None,
        put.short_strike  if put  else None,
        put.long_strike   if put  else None,
        width,
        plan.buffer_pct,
        plan.event_count,
        plan.gex_flag,
        " | ".join(plan.warnings),
        now,
    ))
    conn.commit()
    log.info(f"Spread plan logged for {plan.week_start}")


def update_outcome(
    conn: sqlite3.Connection,
    week_start: str,
    actual_high: float,
    actual_low: float,
    credit_received: float = None,
) -> str:
    """Fill in the actual outcome after the week expires."""
    row = conn.execute(
        "SELECT call_short, put_short, wing_width_used FROM spread_log WHERE week_start = ?",
        (week_start,)
    ).fetchone()

    if not row:
        log.warning(f"No spread_log entry for {week_start}")
        return "not_found"

    call_short, put_short, width = row
    spx_ref = conn.execute(
        "SELECT spx_ref_close FROM spread_log WHERE week_start = ?", (week_start,)
    ).fetchone()[0]

    actual_range_pct = (actual_high - actual_low) / spx_ref if spx_ref else None
    call_breached    = int(actual_high >= call_short) if call_short else 0
    put_breached     = int(actual_low  <= put_short)  if put_short  else 0

    if call_breached or put_breached:
        if credit_received and width:
            pnl_pts = credit_received - width
            outcome = "partial_loss" if pnl_pts > -width * 0.5 else "full_loss"
        else:
            outcome = "full_loss"
            pnl_pts = None
    else:
        outcome = "full_profit"
        pnl_pts = credit_received if credit_received else None

    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        UPDATE spread_log SET
            actual_high      = ?,
            actual_low       = ?,
            actual_range_pct = ?,
            call_breached    = ?,
            put_breached     = ?,
            outcome          = ?,
            pnl_pts          = ?,
            updated_at       = ?
        WHERE week_start = ?
    """, (
        actual_high, actual_low, actual_range_pct,
        call_breached, put_breached, outcome,
        pnl_pts, now,
        week_start,
    ))
    conn.commit()
    log.info(f"Outcome updated for {week_start}: {outcome}")
    return outcome


def print_spread_plan(plan) -> None:
    """Pretty-print the full spread plan to console."""
    sep = "=" * 70

    print(f"\n{sep}")
    print(f"  WEEKLY SPREAD PLAN  --  Week of {plan.week_start}")
    print(f"  Generated: {plan.generated_at[:19]} UTC")
    print(sep)

    print(f"\n  REFERENCE")
    print(f"    SPX Friday close  : {plan.spx_ref_close:>10,.2f}")
    if plan.spx_ref_open:
        print(f"    SPX Monday open   : {plan.spx_ref_open:>10,.2f}")
    print(f"    VIX implied range : {plan.vix_implied_pct*100:>9.2f}%")

    print(f"\n  FORECAST  ({plan.confidence_level}% CI)")
    print(f"    Point estimate    : +/-{plan.point_pct/2*100:.2f}%  "
          f"({plan.point_pct*100:.2f}% total)")
    print(f"    PI upper bound    :  {plan.upper_pct*100:.2f}%  total range")
    print(f"    Buffer applied    : +{plan.buffer_pct*100:.3f}%  ({plan.buffer_pts:.1f} pts)")
    print(f"    Effective range   :  {plan.effective_range_pct*100:.2f}%  total")
    print(f"    Effective upper   : {plan.effective_upper_px:>10,.2f}")
    print(f"    Effective lower   : {plan.effective_lower_px:>10,.2f}")

    print(f"\n  CONTEXT")
    print(f"    Events this week  : {plan.event_count}  "
          f"(FOMC={plan.has_fomc} CPI={plan.has_cpi} NFP={plan.has_nfp} OPEX={plan.has_opex})")
    print(f"    GEX regime        : {plan.gex_regime}")
    print(f"    Recommended width : {plan.recommended_width} pts")

    if plan.warnings:
        print(f"\n  WARNINGS")
        for w in plan.warnings:
            print(f"    {w}")

    print(f"\n{sep}\n")
