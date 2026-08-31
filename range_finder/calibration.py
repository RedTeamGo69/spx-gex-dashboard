# =============================================================================
# calibration.py
# Empirical prediction-interval coverage audit.
#
# The HAR forecast quotes an 80% prediction interval (PI_ALPHA = 0.20) and
# strike placement rides its upper bound — but until this module existed,
# NOTHING ever checked whether the realized weekly range actually lands
# inside that interval ~80% of the time. spread_log stores the forecast
# bounds (point/lower/upper _pct) AND, once scored, the realized range
# (actual_range_pct). This module turns those rows into coverage numbers.
# (The daily/0DTE variant of this audit was removed with the 0DTE finder.)
#
# Interpretation:
#   * one-sided coverage (P[actual <= upper]) audits the RANGE forecast.
#     Nominal for an 80% central interval is 90% one-sided (10% in each
#     tail). NOTE: this is an UPPER BOUND on strike survival, not strike
#     survival itself — no-touch of both band edges implies range <= upper,
#     but a trending week can breach one strike while its total range stays
#     inside the forecast (the open rarely sits mid-range; arcsine law).
#   * strike-touch rate (P[high >= call_short or low <= put_short], from the
#     scored breach flags) is the number that actually matters for condor
#     P&L. It is strictly worse than one-sided coverage by construction;
#     the side-share quantile placement (har_model.
#     estimate_side_share_quantile) exists to close that gap.
#   * two-sided coverage (P[lower <= actual <= upper]) audits the interval
#     as a whole. Nominal is 80%.
#   * buffer breach rate audits the heuristic buffer ON TOP of the PI:
#     P[actual > effective_range] should be ~the tail mass the buffer is
#     believed to absorb.
#
# All functions are pure DataFrame -> dict/DataFrame transforms; the conn
# wrappers only do SELECTs (sqlite-compatible SQL, tested against sqlite).
# Nothing here is called by the cron — reports run on demand.
# =============================================================================

import logging
import math

import pandas as pd

log = logging.getLogger(__name__)

# Below this many scored observations, print the numbers but refuse to call
# them calibration evidence (mirrors analyze_wall_calibration's stance).
MIN_SAMPLE = 15

# Nominal levels implied by PI_ALPHA = 0.20 (an 80% central interval).
NOMINAL_TWO_SIDED = 0.80
NOMINAL_ONE_SIDED = 0.90


# =============================================================================
# LOADERS — thin, read-only
# =============================================================================

def load_completed_forecasts(conn, ticker: str = "SPX") -> pd.DataFrame:
    """spread_log rows that have BOTH a forecast and a scored outcome."""
    df = pd.read_sql_query(
        """
        SELECT week_start, ticker, model_name,
               point_pct, lower_pct, upper_pct, effective_range_pct,
               buffer_pct, event_count,
               actual_high, actual_low, actual_range_pct, outcome,
               call_breached, put_breached,
               wing_width_used, pnl_pts
        FROM spread_log
        WHERE ticker = ?
          AND upper_pct IS NOT NULL
          AND actual_range_pct IS NOT NULL
        ORDER BY week_start ASC
        """,
        conn,
        params=(ticker,),
        parse_dates=["week_start"],
    )
    return df


# =============================================================================
# COVERAGE MATH — pure functions
# =============================================================================

def _binomial_ci(successes: int, n: int, conf: float = 0.90) -> tuple[float, float]:
    """Two-sided binomial CI on a proportion (Wilson score interval).

    Wilson keeps sane bounds at the small n this audit starts life with,
    without needing scipy at import time.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    z = {0.90: 1.6449, 0.95: 1.9600}.get(conf, 1.6449)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def pi_coverage(df: pd.DataFrame, nominal_two_sided: float = NOMINAL_TWO_SIDED) -> dict:
    """Empirical PI coverage from completed forecast rows.

    Expects columns: actual_range_pct, upper_pct, and (optionally, may be
    NULL on legacy rows) lower_pct. Returns one-sided coverage (primary),
    two-sided coverage over the rows that carry lower_pct, Wilson 90% CIs,
    and sample sizes.
    """
    n = len(df)
    if n == 0:
        return {"n": 0, "one_sided": float("nan"), "two_sided": float("nan"),
                "one_sided_ci": (float("nan"), float("nan")),
                "two_sided_ci": (float("nan"), float("nan")),
                "n_two_sided": 0, "nominal_one_sided": NOMINAL_ONE_SIDED,
                "nominal_two_sided": nominal_two_sided,
                "sufficient": False}

    inside_upper = (df["actual_range_pct"] <= df["upper_pct"])
    one_sided = float(inside_upper.mean())

    has_lower = df["lower_pct"].notna() if "lower_pct" in df.columns \
        else pd.Series(False, index=df.index)
    two_df = df[has_lower]
    if len(two_df):
        inside_both = ((two_df["actual_range_pct"] >= two_df["lower_pct"])
                       & (two_df["actual_range_pct"] <= two_df["upper_pct"]))
        two_sided = float(inside_both.mean())
        two_ci = _binomial_ci(int(inside_both.sum()), len(two_df))
    else:
        two_sided = float("nan")
        two_ci = (float("nan"), float("nan"))

    return {
        "n": n,
        "one_sided": one_sided,
        "one_sided_ci": _binomial_ci(int(inside_upper.sum()), n),
        "two_sided": two_sided,
        "two_sided_ci": two_ci,
        "n_two_sided": int(len(two_df)),
        "nominal_one_sided": NOMINAL_ONE_SIDED,
        "nominal_two_sided": nominal_two_sided,
        "sufficient": n >= MIN_SAMPLE,
    }


def strike_touch_summary(df: pd.DataFrame) -> dict:
    """Empirical strike-touch rates from the scored breach flags.

    The honest condor metric: a week counts against the placement when the
    realized high reached the logged call short OR the realized low reached
    the logged put short — regardless of whether the total range stayed
    inside the PI. Only rows where BOTH breach flags were scored are used
    (legacy rows may carry NULLs).
    """
    need = ("call_breached", "put_breached")
    if df.empty or any(c not in df.columns for c in need):
        return {"n": 0, "call_touch": float("nan"), "put_touch": float("nan"),
                "any_touch": float("nan"),
                "any_touch_ci": (float("nan"), float("nan")),
                "sufficient": False}
    scored = df[df["call_breached"].notna() & df["put_breached"].notna()]
    n = len(scored)
    if n == 0:
        return {"n": 0, "call_touch": float("nan"), "put_touch": float("nan"),
                "any_touch": float("nan"),
                "any_touch_ci": (float("nan"), float("nan")),
                "sufficient": False}
    call_t = scored["call_breached"].astype(float) > 0
    put_t = scored["put_breached"].astype(float) > 0
    any_t = call_t | put_t
    return {
        "n": n,
        "call_touch": float(call_t.mean()),
        "put_touch": float(put_t.mean()),
        "any_touch": float(any_t.mean()),
        "any_touch_ci": _binomial_ci(int(any_t.sum()), n),
        "sufficient": n >= MIN_SAMPLE,
    }


def buffer_breach_rate(df: pd.DataFrame) -> dict:
    """How often the realized range blew through PI-upper PLUS the buffer.

    This audits the persisted heuristic buffer. Historical rows may contain
    the legacy GEX adjustment; BUG-06 production rows use the 0.3% base plus
    FOMC/CPI/NFP/OpEx multipliers with GEX disabled pending calibration.
    effective_range_pct = upper_pct + buffer.
    """
    scored = df[df["effective_range_pct"].notna()] \
        if "effective_range_pct" in df.columns else df.iloc[0:0]
    n = len(scored)
    if n == 0:
        return {"n": 0, "breach_rate": float("nan"),
                "breach_ci": (float("nan"), float("nan")), "sufficient": False}
    breached = (scored["actual_range_pct"] > scored["effective_range_pct"])
    return {
        "n": n,
        "breach_rate": float(breached.mean()),
        "breach_ci": _binomial_ci(int(breached.sum()), n),
        "sufficient": n >= MIN_SAMPLE,
    }


def expectancy_summary(df: pd.DataFrame,
                       credit_ratio: "float | None" = None) -> dict:
    """Turn the loss rate into the P&L constraint it implies.

    A coverage or survival number is not an edge. With credit c, width w and
    loss probability p, treating a loss as the full width (the conservative
    convention update_expiration_outcome already stores):

        E = (1-p)*c - p*(w - c) = c - p*w

    so expectancy is zero at c/w = p. The BREAK-EVEN CREDIT RATIO IS THE LOSS
    RATE — which is what makes an otherwise reassuring "70% of weeks survive"
    readable: it demands a 30% credit/width ratio just to break even.

    Two limitations, both structural rather than fixable here:

    * Wins carry no stored P&L. spread_log has pnl_pts, but
      update_expiration_outcome writes NULL on full_profit because the actual
      credit received is not recorded anywhere (spread_persistence.py). So
      realized expectancy cannot be computed from this table — only the
      break-even ratio, which needs no credit data, and a what-if under a
      caller-supplied `credit_ratio`.
    * The loss rate here is TOUCH-based, from the breach flags. XSP weeklys
      are European and cash-settled, so a week can touch a short strike
      intraweek and still settle outside it — a full profit that this counts
      as a loss. spread_log stores actual_high/actual_low but no settlement
      close, so the terminal rate is not recoverable from it;
      target_form_experiment.py computes it from weekly_spx instead. Read
      this number as an upper bound on the true loss rate.

    Returns loss rate, break-even credit ratio, mean width, and (when
    `credit_ratio` is given) expectancy in width-units and in points.
    """
    touch = strike_touch_summary(df)
    n = touch["n"]
    if n == 0:
        return {"n": 0, "loss_rate": float("nan"),
                "breakeven_credit_ratio": float("nan"),
                "mean_width": float("nan"), "credit_ratio": credit_ratio,
                "expectancy_width_units": float("nan"),
                "expectancy_pts": float("nan"), "sufficient": False}

    loss_rate = touch["any_touch"]

    widths = (df["wing_width_used"].dropna()
              if "wing_width_used" in df.columns else pd.Series(dtype=float))
    mean_width = float(widths.mean()) if len(widths) else float("nan")

    out = {
        "n": n,
        "loss_rate": loss_rate,
        "loss_rate_ci": touch["any_touch_ci"],
        "breakeven_credit_ratio": loss_rate,   # E = 0 at c/w = p
        "mean_width": mean_width,
        "credit_ratio": credit_ratio,
        "sufficient": n >= MIN_SAMPLE,
    }

    if credit_ratio is not None:
        per_width = float(credit_ratio) - loss_rate
        out["expectancy_width_units"] = per_width
        out["expectancy_pts"] = (per_width * mean_width
                                 if mean_width == mean_width else float("nan"))
    else:
        out["expectancy_width_units"] = float("nan")
        out["expectancy_pts"] = float("nan")

    return out


def rolling_coverage(df: pd.DataFrame, window: int = 26,
                     date_col: str = "week_start") -> pd.DataFrame:
    """Rolling one-sided coverage — drift detection.

    Returns a frame indexed by date with a `rolling_one_sided` column; NaN
    until `window` observations accumulate.
    """
    if df.empty:
        return pd.DataFrame(columns=["rolling_one_sided"])
    s = (df.set_index(date_col)
           .sort_index()
           .apply(lambda r: float(r["actual_range_pct"] <= r["upper_pct"]),
                  axis=1))
    return s.rolling(window, min_periods=window).mean() \
            .to_frame("rolling_one_sided")


def coverage_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Per-spec coverage breakdown. Legacy rows without a stored spec pool
    under "(unknown)"."""
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["model_name"] = work["model_name"].fillna("(unknown)")
    rows = []
    for name, grp in work.groupby("model_name"):
        cov = pi_coverage(grp)
        rows.append({"model_name": name, "n": cov["n"],
                     "one_sided": cov["one_sided"],
                     "two_sided": cov["two_sided"],
                     "sufficient": cov["sufficient"]})
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


# =============================================================================
# CONN WRAPPERS
# =============================================================================

def weekly_pi_coverage(conn, ticker: str = "SPX") -> dict:
    """One-call weekly coverage summary for the UI caption."""
    df = load_completed_forecasts(conn, ticker=ticker)
    cov = pi_coverage(df)
    cov["buffer"] = buffer_breach_rate(df)
    cov["strike_touch"] = strike_touch_summary(df)
    return cov


# =============================================================================
# CLI REPORT
# =============================================================================
# Usage:
#   DATABASE_URL=postgres://... python -m range_finder.calibration
#
# Read-only. Prints the weekly coverage report, refusing conclusions under
# MIN_SAMPLE observations.

def _fmt_pct(x: float) -> str:
    return "—" if x != x else f"{x:.0%}"


def _print_coverage_block(title: str, cov: dict) -> None:
    print(f"\n  {title}")
    print(f"    scored observations : {cov['n']}")
    if cov["n"] == 0:
        print("    (nothing scored yet — outcomes accumulate one per period)")
        return
    lo, hi = cov["one_sided_ci"]
    print(f"    one-sided coverage  : {_fmt_pct(cov['one_sided'])} "
          f"(CI {_fmt_pct(lo)}-{_fmt_pct(hi)}, nominal "
          f"{_fmt_pct(cov['nominal_one_sided'])})")
    if cov["n_two_sided"]:
        lo2, hi2 = cov["two_sided_ci"]
        print(f"    two-sided coverage  : {_fmt_pct(cov['two_sided'])} "
              f"(CI {_fmt_pct(lo2)}-{_fmt_pct(hi2)}, nominal "
              f"{_fmt_pct(cov['nominal_two_sided'])}, "
              f"n={cov['n_two_sided']})")
    touch = cov.get("strike_touch")
    if touch and touch["n"]:
        lo3, hi3 = touch["any_touch_ci"]
        print(f"    strike-touch rate   : {_fmt_pct(touch['any_touch'])} "
              f"(CI {_fmt_pct(lo3)}-{_fmt_pct(hi3)}, "
              f"call {_fmt_pct(touch['call_touch'])} / "
              f"put {_fmt_pct(touch['put_touch'])}, n={touch['n']}) "
              f"<- the condor number; strictly worse than coverage")
    if not cov["sufficient"]:
        # ASCII only — Windows cp1252 consoles choke on warning glyphs
        print(f"    [!] n < {MIN_SAMPLE} - numbers shown, conclusions refused. "
              "Keep accumulating.")


def main() -> None:
    import os
    import sys

    logging.basicConfig(level=logging.WARNING)

    if not os.environ.get("DATABASE_URL", "").strip():
        log.error("DATABASE_URL not set — cannot read spread_log.")
        sys.exit(1)

    from range_finder.db import get_connection

    conn = get_connection()

    print("=" * 70)
    print("  PREDICTION-INTERVAL CALIBRATION REPORT")
    print("=" * 70)

    df_w = load_completed_forecasts(conn, ticker="SPX")
    cov_w = pi_coverage(df_w)
    _print_coverage_block("WEEKLY (spread_log, SPX)", cov_w)

    buf = buffer_breach_rate(df_w)
    if buf["n"]:
        lo, hi = buf["breach_ci"]
        print(f"    buffer breach rate  : {_fmt_pct(buf['breach_rate'])} "
              f"(CI {_fmt_pct(lo)}-{_fmt_pct(hi)}) - realized range beyond "
              "PI-upper + buffer")

    exp = expectancy_summary(df_w)
    if exp["n"]:
        from range_finder.spread_levels import MIN_CREDIT_RATIO
        need = exp["breakeven_credit_ratio"]
        print(f"\n    loss rate (touch)   : {_fmt_pct(exp['loss_rate'])} "
              f"(n={exp['n']})")
        print(f"    breakeven credit    : {_fmt_pct(need)} of width - below "
              "this the placement loses money however good its coverage looks")
        print(f"    finder credit floor : {_fmt_pct(MIN_CREDIT_RATIO)}"
              + ("  [!] floor is BELOW breakeven at this loss rate"
                 if need == need and MIN_CREDIT_RATIO < need else ""))
        print("    (touch-based, so an upper bound on true loss rate - XSP "
              "settles European; see target_form_experiment.py)")

    by_model = coverage_by_model(df_w)
    if not by_model.empty:
        print("\n    per-spec breakdown:")
        for _, r in by_model.iterrows():
            print(f"      {r['model_name']:22s} n={r['n']:<4d} "
                  f"one-sided={_fmt_pct(r['one_sided'])}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
