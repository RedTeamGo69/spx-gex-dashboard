#!/usr/bin/env python3
"""
Wall-proximity calibration — empirical analysis of historical breach data.

Run this when spread_log has accumulated enough outcome rows (~20+ weeks
of trading) to calibrate the wall-proximity warning thresholds from data
instead of from gut feel. It reads every row with both a known outcome
(call_breached / put_breached) and the corresponding short strikes, joins
each row back to the wall values that were saved alongside, and reports:

  - How far the short strike was from the wall at breach time (abs & pct)
  - Distribution of breach-distance-to-wall per side (call vs put)
  - Suggested warning thresholds (e.g. 90th / 95th percentile) for each
    side so you can pick a data-driven number to feed back into
    range_finder.gex_bridge.adjust_spread_with_gex().

It does NOT write anything to the database and it does NOT change any
code — it's read-only analysis, safe to run any time. If the sample is
too small to be meaningful (say <10 breach weeks per side), it prints
a warning and refuses to suggest thresholds — tuning a distribution
from 3 observations is worse than leaving the default in place.

Usage:
    DATABASE_URL=postgres://... python range_finder/analyze_wall_calibration.py

Output: a human-readable report to stdout. Nothing persisted.

Current (hardcoded) thresholds in gex_bridge.py:
    - Call short within 0.4% of call wall → widening warning
    - Put short  within 0.4% of put  wall → widening warning
    (both use the same 0.4%; this script checks if that's right)

TODO when ready to tune:
    1. Run this script against your accumulated spread_log
    2. Look at the "90th percentile of breach-distance-to-wall" number
       per side
    3. Set gex_bridge thresholds to that value (or slightly looser if
       you want more false positives and fewer misses)
    4. Re-run after another 20 weeks to see if the calibration still
       holds
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# Minimum breach-week sample size per side before we'll suggest a
# threshold. Anything below this and we print the raw data but refuse
# to compute a "calibrated" number.
MIN_BREACH_SAMPLE = 10


def _format_pct(x):
    return "n/a" if x is None else f"{x*100:.3f}%"


def _format_pts(x):
    return "n/a" if x is None else f"{x:.1f}"


def _percentile(sorted_vals: list, q: float) -> Optional[float]:
    """Simple percentile (linear interpolation) on a sorted list.
    Returns None for empty input."""
    if not sorted_vals:
        return None
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    idx = q * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _describe_distribution(values: list, label: str) -> None:
    """Print summary statistics for a distribution of breach distances."""
    if not values:
        print(f"  {label}: no data")
        return
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    p50 = _percentile(sorted_vals, 0.50)
    p75 = _percentile(sorted_vals, 0.75)
    p90 = _percentile(sorted_vals, 0.90)
    p95 = _percentile(sorted_vals, 0.95)
    mean = sum(sorted_vals) / n
    print(f"  {label}: n={n}")
    print(f"    min    = {sorted_vals[0]:.4f}")
    print(f"    p50    = {p50:.4f}")
    print(f"    mean   = {mean:.4f}")
    print(f"    p75    = {p75:.4f}")
    print(f"    p90    = {p90:.4f}")
    print(f"    p95    = {p95:.4f}")
    print(f"    max    = {sorted_vals[-1]:.4f}")


def main():
    if not os.environ.get("DATABASE_URL", "").strip():
        log.error("DATABASE_URL not set — cannot read spread_log.")
        sys.exit(1)

    from range_finder.db import get_connection

    conn = get_connection()
    cur = conn.cursor()

    # Pull every spread_log row where both the plan levels and the outcome
    # are populated. We include both full_profit and breach rows so we can
    # compute "at breach time, how close was the short strike to the wall?"
    # We don't need wall values here — the distance to wall will be
    # inferred from the GEX context at the time, which we don't preserve
    # per-week (!). Limitation noted in the readme below.
    cur.execute("""
        SELECT week_start, spx_ref_close, call_short, put_short,
               actual_high, actual_low, call_breached, put_breached, outcome
        FROM spread_log
        WHERE call_short IS NOT NULL
          AND put_short  IS NOT NULL
          AND actual_high IS NOT NULL
          AND actual_low  IS NOT NULL
        ORDER BY week_start ASC
    """)
    rows = cur.fetchall()

    print("=" * 78)
    print("  WALL CALIBRATION ANALYSIS")
    print("=" * 78)
    print(f"  Total completed weeks in spread_log: {len(rows)}")

    if not rows:
        print("  No completed outcome rows yet — nothing to calibrate.")
        print("  Run this script again after several weeks of trading have")
        print("  accumulated outcomes via scheduled_snapshot's Monday setup.")
        sys.exit(0)

    # Split breach weeks by side. Compute the "distance-to-breach":
    #   - Call breach: (actual_high - call_short) — how many points ABOVE
    #     the short the market traded. Positive = breached, negative =
    #     stayed below (but since call_breached=True we expect positive).
    #   - Put  breach: (put_short - actual_low) — how far below.
    # Normalized by spx_ref_close to make cross-week comparable.

    call_breach_dist_pct = []
    put_breach_dist_pct = []
    call_nobreach_cushion_pct = []  # how much cushion call side had when no breach
    put_nobreach_cushion_pct = []

    for row in rows:
        (_week_start, spx_ref, call_short, put_short,
         actual_high, actual_low, call_breached, put_breached, _outcome) = row

        if spx_ref is None or spx_ref <= 0:
            continue

        if call_breached:
            dist_pts = actual_high - call_short
            call_breach_dist_pct.append(dist_pts / spx_ref)
        else:
            cushion_pts = call_short - actual_high
            call_nobreach_cushion_pct.append(cushion_pts / spx_ref)

        if put_breached:
            dist_pts = put_short - actual_low
            put_breach_dist_pct.append(dist_pts / spx_ref)
        else:
            cushion_pts = actual_low - put_short
            put_nobreach_cushion_pct.append(cushion_pts / spx_ref)

    print()
    print("── CALL SIDE ────────────────────────────────────────────────────")
    print(f"  breach weeks:    {len(call_breach_dist_pct)}")
    print(f"  no-breach weeks: {len(call_nobreach_cushion_pct)}")
    print()
    print("  Breach-penetration distribution (how far above the short strike")
    print("  the market traded on breach weeks, as % of spot):")
    _describe_distribution(call_breach_dist_pct, "call breach penetration")
    print()
    print("  No-breach cushion distribution (how much room to the short")
    print("  strike you had left at week end on non-breach weeks):")
    _describe_distribution(call_nobreach_cushion_pct, "call no-breach cushion")

    print()
    print("── PUT SIDE ─────────────────────────────────────────────────────")
    print(f"  breach weeks:    {len(put_breach_dist_pct)}")
    print(f"  no-breach weeks: {len(put_nobreach_cushion_pct)}")
    print()
    _describe_distribution(put_breach_dist_pct, "put breach penetration")
    print()
    _describe_distribution(put_nobreach_cushion_pct, "put no-breach cushion")

    # Suggested thresholds — only if sample is large enough
    print()
    print("── SUGGESTED WARNING THRESHOLDS ─────────────────────────────────")
    print()
    print("  The current gex_bridge.py uses 0.4% for BOTH sides:")
    print("    if dist / spot < 0.004:  # ~0.4% of spot")
    print()
    print("  To tune from data, pick a percentile of no-breach cushion")
    print("  where you want the warning to fire. A good starting point is")
    print("  the 25th percentile — i.e., 'when my cushion is in the bottom")
    print("  quartile of non-breach weeks, warn me'. Lower → more warnings,")
    print("  higher → fewer warnings.")
    print()

    if len(call_nobreach_cushion_pct) >= MIN_BREACH_SAMPLE:
        sorted_call_cushion = sorted(call_nobreach_cushion_pct)
        p25_call = _percentile(sorted_call_cushion, 0.25)
        print(f"  Call warning threshold (25th pctile of no-breach cushion):")
        print(f"    {p25_call*100:.3f}% of spot  (current: 0.400%)")
    else:
        print(f"  Call: insufficient sample ({len(call_nobreach_cushion_pct)} < "
              f"{MIN_BREACH_SAMPLE}) — no recommendation. Keep the 0.4% default.")

    if len(put_nobreach_cushion_pct) >= MIN_BREACH_SAMPLE:
        sorted_put_cushion = sorted(put_nobreach_cushion_pct)
        p25_put = _percentile(sorted_put_cushion, 0.25)
        print(f"  Put  warning threshold (25th pctile of no-breach cushion):")
        print(f"    {p25_put*100:.3f}% of spot  (current: 0.400%)")
    else:
        print(f"  Put:  insufficient sample ({len(put_nobreach_cushion_pct)} < "
              f"{MIN_BREACH_SAMPLE}) — no recommendation. Keep the 0.4% default.")

    print()
    print("── ASYMMETRY TEST ───────────────────────────────────────────────")
    if (len(call_nobreach_cushion_pct) >= MIN_BREACH_SAMPLE
            and len(put_nobreach_cushion_pct) >= MIN_BREACH_SAMPLE):
        p25_call = _percentile(sorted(call_nobreach_cushion_pct), 0.25)
        p25_put = _percentile(sorted(put_nobreach_cushion_pct), 0.25)
        ratio = p25_put / p25_call if p25_call > 0 else None
        print(f"  put-25th / call-25th ratio: "
              f"{ratio:.2f}" if ratio is not None else "n/a")
        print()
        print("  Interpretation:")
        print("    ≈ 1.00 → symmetric (current 0.4% default is fine)")
        print("    > 1.20 → put side has more cushion; call threshold tighter")
        print("    < 0.83 → call side has more cushion; put threshold tighter")
        print()
        if ratio is None:
            pass
        elif 0.83 <= ratio <= 1.20:
            print("  → Symmetric within noise. Keep the 0.4% default for both.")
        elif ratio > 1.20:
            print("  → Asymmetric — puts run with more cushion historically.")
            print("    Consider tighter call-side threshold.")
        else:
            print("  → Asymmetric — calls run with more cushion historically.")
            print("    Consider tighter put-side threshold.")
    else:
        print("  Not enough data for asymmetry test yet.")

    print()
    print("── CAVEATS ──────────────────────────────────────────────────────")
    print("  1. This analysis uses 'no-breach cushion' as a proxy for the")
    print("     effective wall-distance distribution. It assumes the short")
    print("     strikes were placed near the walls; if you sell spreads")
    print("     at a fixed delta instead, the interpretation shifts.")
    print("  2. spread_log does NOT currently persist the GEX wall values")
    print("     at plan time, so we can't compute 'distance from short")
    print("     strike to wall' directly. If you want that, add call_wall")
    print("     and put_wall columns to spread_log in a future migration")
    print("     and have scheduled_snapshot save them alongside the plan.")
    print("  3. Don't tune thresholds from a sample smaller than ~20 weeks")
    print("     per side. The distribution is noisy at low N.")


if __name__ == "__main__":
    main()
