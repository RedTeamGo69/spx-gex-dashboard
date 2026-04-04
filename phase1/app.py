from __future__ import annotations

import calendar
from datetime import timedelta
import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import Calendar as TkCalendar

from phase1.config import HEATMAP_EXPS, build_config_snapshot
from phase1.market_clock import now_ny, get_calendar_snapshot
from phase1.data_client import TradierDataClient
from phase1.rates import fetch_risk_free_rate
from phase1.parity import get_reference_spot_details
from phase1.run_metadata import build_run_metadata
import phase1.gex_engine as gex_engine
import phase1.dashboard as dashboard
from phase1.confidence import build_run_confidence
from phase1.staleness import build_staleness_info
from phase1.wall_credibility import build_wall_credibility
from phase1.scenarios import run_scenario_engine
from phase1.expected_move import build_expected_move_analysis


def validate_runtime_inputs(tradier_token: str) -> bool:
    return bool(tradier_token and tradier_token != "YOUR_TOKEN_HERE")


def select_heatmap_exps(avail: list[str], today_str: str, count: int = HEATMAP_EXPS) -> list[str]:
    return [e for e in avail if e >= today_str][:count]


def pick_dates(available_exps):
    result = {"exps": None, "label": None}
    base_now = now_ny()
    today_str = base_now.strftime("%Y-%m-%d")
    dte0 = next((e for e in available_exps if e >= today_str), None)
    tomorrow_str = (base_now + timedelta(days=1)).strftime("%Y-%m-%d")
    dte1 = next((e for e in available_exps if e >= tomorrow_str), None)

    root = tk.Tk()
    root.title("SPX GEX — Pick Expiration")
    root.configure(bg="#1a1a2e")
    root.resizable(False, False)

    style = ttk.Style()
    style.theme_use("clam")

    for s, kw in [
        ("TRadiobutton", {"background": "#1a1a2e", "foreground": "white", "font": ("Segoe UI", 11)}),
        ("TButton", {"font": ("Segoe UI", 11, "bold"), "padding": 6}),
        ("TLabel", {"background": "#1a1a2e", "foreground": "white", "font": ("Segoe UI", 10)}),
        ("Header.TLabel", {"font": ("Segoe UI", 14, "bold"), "background": "#1a1a2e", "foreground": "#00e5ff"}),
    ]:
        style.configure(s, **kw)

    ttk.Label(root, text="SPX Gamma Exposure", style="Header.TLabel").pack(pady=(12, 4))
    ttk.Label(root, text="Select expiration range:").pack(pady=(0, 8))

    mode_var = tk.StringVar(value="0dte")
    rf = tk.Frame(root, bg="#1a1a2e")
    rf.pack(padx=20, anchor="w")

    for txt, val, st in [
        (f"0DTE  ({dte0})" if dte0 else "0DTE  (n/a)", "0dte", "normal" if dte0 else "disabled"),
        (f"Tomorrow  ({dte1})" if dte1 else "Tomorrow  (n/a)", "tomorrow", "normal" if dte1 else "disabled"),
        ("This week", "week", "normal"),
        ("This month", "month", "normal"),
        ("Pick specific date", "single", "normal"),
        ("Pick date range", "range", "normal"),
    ]:
        ttk.Radiobutton(rf, text=txt, variable=mode_var, value=val, state=st).pack(anchor="w", pady=2)

    cf = tk.Frame(root, bg="#1a1a2e")
    ck = dict(
        selectmode="day",
        date_pattern="yyyy-mm-dd",
        year=base_now.year,
        month=base_now.month,
        day=base_now.day,
        background="#2a2a4a",
        foreground="white",
        selectbackground="#00c853",
        selectforeground="black",
        normalbackground="#1a1a2e",
        normalforeground="white",
        headersbackground="#2a2a4a",
        headersforeground="#00e5ff",
        weekendbackground="#1a1a2e",
        weekendforeground="#ff8a80",
        othermonthbackground="#111",
        othermonthforeground="#555",
        font=("Segoe UI", 9),
    )
    lf = ttk.Label(cf, text="From:")
    c1 = TkCalendar(cf, **ck)
    lt = ttk.Label(cf, text="To:")
    ck["selectbackground"] = "#ff1744"
    ck["selectforeground"] = "white"
    c2 = TkCalendar(cf, **ck)

    def omc(*_):
        m = mode_var.get()
        if m == "single":
            cf.pack(padx=20, pady=(8, 4))
            lf.grid(row=0, column=0, sticky="w")
            c1.grid(row=1, column=0, padx=(0, 8))
            lt.grid_remove()
            c2.grid_remove()
        elif m == "range":
            cf.pack(padx=20, pady=(8, 4))
            lf.grid(row=0, column=0, sticky="w", padx=(0, 8))
            c1.grid(row=1, column=0, padx=(0, 8))
            lt.grid(row=0, column=1, sticky="w")
            c2.grid(row=1, column=1)
        else:
            cf.pack_forget()

    mode_var.trace_add("write", omc)
    cf.pack_forget()

    def run():
        m = mode_var.get()
        t = now_ny()

        if m == "0dte":
            if dte0:
                result["exps"], result["label"] = [dte0], dte0
            else:
                messagebox.showerror("Error", "No 0DTE available")
                return

        elif m == "tomorrow":
            if dte1:
                result["exps"], result["label"] = [dte1], dte1
            else:
                messagebox.showerror("Error", "No tomorrow available")
                return

        elif m == "week":
            # days_to_fri: Mon→4, Tue→3, ..., Fri→0 (today), Sat→6, Sun→5
            days_to_fri = (4 - t.weekday()) % 7
            fri = (t + timedelta(days=days_to_fri)).strftime("%Y-%m-%d")
            s = [e for e in available_exps if today_str <= e <= fri]
            if not s:
                messagebox.showerror("Error", "None this week")
                return
            result["exps"], result["label"] = s, f"{s[0]} to {s[-1]}"

        elif m == "month":
            ld = t.replace(day=calendar.monthrange(t.year, t.month)[1]).strftime("%Y-%m-%d")
            s = [e for e in available_exps if today_str <= e <= ld]
            if not s:
                messagebox.showerror("Error", "None this month")
                return
            result["exps"], result["label"] = s, f"{s[0]} to {s[-1]}"

        elif m == "single":
            p = c1.get_date()
            if p in available_exps:
                result["exps"], result["label"] = [p], p
            else:
                n = min(
                    available_exps,
                    key=lambda e: abs(
                        __import__("datetime").datetime.strptime(e, "%Y-%m-%d")
                        - __import__("datetime").datetime.strptime(p, "%Y-%m-%d")
                    ),
                )
                if messagebox.askyesno("", f"No exp on {p}. Use {n}?"):
                    result["exps"], result["label"] = [n], n
                else:
                    return

        elif m == "range":
            a, b = c1.get_date(), c2.get_date()
            if a > b:
                a, b = b, a
            s = [e for e in available_exps if a <= e <= b]
            if not s:
                messagebox.showerror("Error", f"None between {a} and {b}")
                return
            result["exps"], result["label"] = s, f"{a} to {b}"

        root.destroy()

    bf = tk.Frame(root, bg="#1a1a2e")
    bf.pack(pady=(8, 12))
    ttk.Button(bf, text="Run GEX", command=run).pack(side="left", padx=4)
    ttk.Button(bf, text="Cancel", command=root.destroy).pack(side="left", padx=4)

    root.update_idletasks()
    root.geometry(
        f"+{root.winfo_screenwidth() // 2 - root.winfo_width() // 2}"
        f"+{root.winfo_screenheight() // 2 - root.winfo_height() // 2}"
    )
    root.mainloop()
    return result["exps"], result["label"]


def run_app(
    tradier_token: str,
    fred_api_key: str,
    tradier_base_url: str = "https://api.tradier.com/v1",
    debug: bool = False,
    tool_version: str = "v5",
):
    print("=" * 58)
    print(f"  SPX Gamma Exposure (GEX) Calculator  {tool_version}")
    print("  Implied spot | Zero gamma sweep | GEX profile | Hybrid IV mode")
    print("=" * 58, "\n")

    if not validate_runtime_inputs(tradier_token):
        print("ERROR: No Tradier API token configured.")
        print("  Set TRADIER_TOKEN environment variable or edit the launcher file.")
        print("  Get your token at: https://web.tradier.com/user/api")
        return

    client = TradierDataClient(token=tradier_token, base_url=tradier_base_url)
    client.clear_cache()

    run_now = now_ny()
    calendar_snapshot = get_calendar_snapshot(run_now)

    print("Fetching risk-free rate...")
    rfr_info = fetch_risk_free_rate(fred_api_key, debug=debug)
    rfr = rfr_info["rate"]

    print("\nFetching expirations...")
    avail = client.get_expirations("SPX")
    today_str = run_now.strftime("%Y-%m-%d")
    nearest_exp = next((e for e in avail if e >= today_str), avail[0])

    print("\nComputing reference spot...")
    spot_info = get_reference_spot_details(
        ticker="SPX",
        nearest_exp=nearest_exp,
        get_spot_price_func=client.get_spot_price,
        get_chain_cached_func=client.get_chain_cached,
        r=rfr,
        now=run_now,
    )

    spot = spot_info["spot"]
    spot_source = spot_info["source"]

    print(f"  Tradier quote: ${spot_info['tradier_spot']:.2f} (may be delayed)")

    if spot_info["parity_attempted"]:
        print(f"  Parity chain status: {spot_info['parity_chain_status']}")

    if spot_info["implied_spot"] is not None:
        print(f"  Implied spot: ${spot_info['implied_spot']:.2f}")

    if spot_info.get("parity_diagnostics"):
        pdiag = spot_info["parity_diagnostics"]
        cq = pdiag["call_quality"]
        pq = pdiag["put_quality"]

        print(
            f"  Parity quote quality (calls): usable={cq['usable']}/{cq['total']}, "
            f"no2s={cq['no_two_sided_quote']}, crossed={cq['crossed_or_locked']}, "
            f"wide={cq['wide_spread']}, bad_mid={cq['bad_mid']}"
        )
        print(
            f"  Parity quote quality (puts):  usable={pq['usable']}/{pq['total']}, "
            f"no2s={pq['no_two_sided_quote']}, crossed={pq['crossed_or_locked']}, "
            f"wide={pq['wide_spread']}, bad_mid={pq['bad_mid']}"
        )
        print(
            f"  Common usable strikes: {pdiag['common_usable_strikes']}  |  "
            f"Near-spot candidates: {pdiag['near_spot_candidates']}  |  "
            f"Final ATM strikes: {pdiag['final_atm_strikes']}"
        )
        print(
            f"  Relative-band pass: {pdiag['relative_band_pass_count']}  |  "
            f"Hard-filter pass: {pdiag['hard_filter_pass_count']}"
        )
        print(
            f"  Parity method: {pdiag['parity_method']}  |  "
            f"Simple median: {pdiag['simple_median_spot']}  |  "
            f"Weighted median: {pdiag['weighted_median_spot']}  |  "
            f"Weight sum: {pdiag['selected_weight_sum']:.4f}"
        )

    print(f"  Reference spot: ${spot:.2f} ({spot_source})")
    if spot_info.get("expiration_close_ny"):
        print(f"  Nearest exp close NY: {spot_info['expiration_close_ny']}")
    print(f"\n  → Using: ${spot:.2f} ({spot_source})")
    print(f"  {len(avail)} expirations available\n")

    target_exps, date_label = pick_dates(avail)
    if target_exps is None:
        print("Cancelled.")
        return

    heatmap_exps = select_heatmap_exps(avail, today_str, HEATMAP_EXPS)
    config_snapshot = build_config_snapshot()

    print(f"\nSelected: {date_label} ({len(target_exps)} exp)")
    print(f"Heatmap: {len(heatmap_exps)} columns")
    print("Fetching chains...\n")

    gex_df, hm_gex, hm_iv, stats, all_options, strike_support_df, expiration_support_df = gex_engine.calculate_all(
        client, "SPX", target_exps, spot, heatmap_exps, r=rfr, now=run_now
    )

    if gex_df.empty:
        print("No data. All requested target expirations may have failed or had no usable contracts.")
        return

    print("\nComputing key levels...")
    levels = gex_engine.find_key_levels(gex_df, spot, all_options=all_options, r=rfr)

    print("\nComputing GEX profile curve...")
    profile_df = gex_engine.compute_gex_profile_curve(all_options, spot, r=rfr)

    sensitivity_df = gex_engine.compute_zero_gamma_sensitivity(all_options, spot, r=rfr)
    scenarios_df = run_scenario_engine(all_options, base_spot=spot, base_r=rfr)
    has_0dte = any(e == today_str for e in target_exps)
    staleness_info = build_staleness_info(calendar_snapshot, spot_info, stats, has_0dte=has_0dte)
    confidence_info = build_run_confidence(stats, spot_info, staleness_info=staleness_info)
    wall_credibility_info = build_wall_credibility(
        levels=levels,
        strike_support_df=strike_support_df,
        sensitivity_df=sensitivity_df,
        confidence_info=confidence_info,
        staleness_info=staleness_info,
    )    

    regime_info = gex_engine.get_gamma_regime_text(spot, levels["zero_gamma"])

    # ── Expected move analysis (0DTE framework) ──
    print("\nComputing expected move analysis...")
    spx_full_quote = None
    spy_quote = None
    try:
        spx_full_quote = client.get_full_quote("SPX")
    except Exception as e:
        print(f"  Warning: could not fetch full SPX quote: {e}")
    try:
        spy_quote = client.get_full_quote("SPY")
    except Exception:
        pass  # SPY proxy is optional

    prev_close = spx_full_quote["prevclose"] if spx_full_quote else 0.0

    # Get the 0DTE chain for straddle extraction
    dte0_exp = target_exps[0] if target_exps else nearest_exp
    dte0_entry = client.get_chain_cached("SPX", dte0_exp)
    dte0_calls = dte0_entry.get("calls", []) if dte0_entry.get("status") == "ok" else []
    dte0_puts = dte0_entry.get("puts", []) if dte0_entry.get("status") == "ok" else []

    em_analysis = build_expected_move_analysis(
        spot=spot,
        prev_close=prev_close,
        zero_gamma=levels["zero_gamma"],
        gamma_regime=regime_info["regime"],
        calls_0dte=dte0_calls,
        puts_0dte=dte0_puts,
        spy_quote=spy_quote,
        market_open=bool(spot_info.get("market_open")),
    )

    print(f"\n{'─' * 58}")
    print(f"  Spot:              ${spot:.2f}  ({spot_source})")
    print(f"  Pos GEX:           ${levels['call_wall']:.0f}  ({levels['call_wall_gex']:+,.0f})")
    print(f"  Neg GEX:           ${levels['put_wall']:.0f}  ({levels['put_wall_gex']:+,.0f})")
    print(f"  Zero Gamma:        ${levels['zero_gamma']:.2f}  (sweep)")
    print(
        f"  Zero Γ type:       {levels.get('zero_gamma_type', 'unknown')} "
        f"[{levels.get('zero_gamma_method', 'unknown')}]"
    )
    if levels.get("zero_gamma_abs_gex") is not None:
        print(f"  Zero Γ residual:   {levels['zero_gamma_abs_gex']:.4f}")
    print(f"  Gamma Regime:      {regime_info['regime']}  ({regime_info['distance_text']})")

    # Expected move output
    em = em_analysis["expected_move"]
    on = em_analysis["overnight_move"]
    cl = em_analysis["classification"]
    if em["expected_move_pts"] is not None:
        print(f"  {'─' * 44}")
        print(f"  ATM Straddle:      {em['expected_move_pts']:.1f} pts  ({em['expected_move_pct']:.2f}%)")
        print(f"  Expected Range:    ${em['lower_level']:.0f} – ${em['upper_level']:.0f}")
        if on["overnight_move_pts"] is not None:
            print(f"  Overnight Move:    {on['overnight_move_pts']:+.1f} pts  ({on['overnight_move_pct']:+.2f}%)")
            if cl["move_ratio"] is not None:
                print(f"  Move Ratio:        {cl['move_ratio']*100:.0f}% of expected  ({cl['move_ratio_label']})")
        if cl["classification"]:
            print(f"  Session Type:      {cl['classification']}")
            print(f"  Bias:              {cl['bias']}")
            if cl.get("historical_tendencies"):
                print(f"  Tendency:          {cl['historical_tendencies'][0]}")
        if em_analysis.get("spy_proxy"):
            sp = em_analysis["spy_proxy"]
            print(f"  SPY Proxy Move:    {sp['spy_move_pct']:+.2f}% → ~{sp['implied_spx_move_pts']:+.1f} SPX pts")
        print(f"  {'─' * 44}")

    print(f"  Rate:              {rfr*100:.2f}%")
    print(f"  Used options:      {stats['used_option_count']:,}")
    print(f"  Direct IV:         {stats['direct_iv_count']:,}")
    print(f"  Synthetic IV:      {stats['synthetic_iv_count']:,}")
    print(f"  Synthetic accept:  {stats['synthetic_fit_accept_count']:,}")
    print(f"  Synthetic reject:  {stats['synthetic_fit_reject_count']:,}")
    print(f"  No model input:    {stats['no_model_input_count']:,}")
    if stats['synthetic_fit_avg_rel_error'] is not None:
        print(f"  Synth avg fit err: {stats['synthetic_fit_avg_rel_error']*100:.2f}%")
    if stats['synthetic_fit_max_rel_error'] is not None:
        print(f"  Synth max fit err: {stats['synthetic_fit_max_rel_error']*100:.2f}%")
    print(f"  Skipped:           {stats['skipped_count']:,}")
    print(f"  Coverage:          {stats['coverage_ratio']*100:.1f}%")
    print(f"  Failed exps:       {stats['failed_exp_count']}")
    print(f"  Freshness:         {staleness_info['freshness_score']:.1f} / 100 ({staleness_info['freshness_label']})")
    print(
        f"  Wall credibility:  CW {wall_credibility_info['call_wall']['score']:.1f} "
        f"| PW {wall_credibility_info['put_wall']['score']:.1f} "
        f"| ZG {wall_credibility_info['zero_gamma']['score']:.1f}"
    )
    print(f"  Scenarios:         {len(scenarios_df)}")        
    if stats.get("strike_support_avg") is not None:
        print(f"  Strike support:    {stats['strike_support_avg']:.1f} avg  |  Fragile: {stats['fragile_strike_count']}")
    if stats.get("expiration_support_avg") is not None:
        print(f"  Exp support:       {stats['expiration_support_avg']:.1f} avg")

    # Per-expiry zero-gamma
    per_exp_zg = levels.get("per_exp_zero_gamma") or {}
    if per_exp_zg.get("nearest_exp_zero_gamma") is not None:
        print(f"  ZG (0DTE):         ${per_exp_zg['nearest_exp_zero_gamma']:.2f}  ({per_exp_zg['nearest_exp_option_count']} opts)")
    if per_exp_zg.get("other_exp_zero_gamma") is not None:
        print(f"  ZG (other):        ${per_exp_zg['other_exp_zero_gamma']:.2f}  ({per_exp_zg['other_exp_option_count']} opts)")

    # Call/put wall clusters
    for wall_name, cluster_key in [("CW", "call_wall_cluster"), ("PW", "put_wall_cluster")]:
        cluster = levels.get(cluster_key)
        if cluster and cluster.get("is_cluster"):
            print(f"  {wall_name} cluster:       centroid ${cluster['centroid']:.0f}  strikes={cluster['cluster_strikes']}")

    print(f"{'─' * 58}")
    print(f"  NOTE: GEX assumes dealers are net short all options")
    print(f"  (standard convention). Actual positioning varies by")
    print(f"  strike. OI is EOD data — intraday flow not captured.")
    print(f"{'─' * 58}\n")

    run_metadata = build_run_metadata(
        tool_version=tool_version,
        calendar_snapshot=calendar_snapshot,
        risk_free_info=rfr_info,
        spot_info=spot_info,
        stats=stats,
        selected_exps=target_exps,
        heatmap_exps=heatmap_exps,
        config_snapshot=config_snapshot,
        confidence_info=confidence_info,
        sensitivity_rows=sensitivity_df.to_dict(orient="records"),
        strike_support_rows=strike_support_df.to_dict(orient="records"),
        expiration_support_rows=expiration_support_df.to_dict(orient="records"),
        staleness_info=staleness_info,
        wall_credibility_info=wall_credibility_info,
        scenario_rows=scenarios_df.to_dict(orient="records"),
        expected_move_info=em_analysis,
    )

    dashboard.build_dashboard(
        gex_df=gex_df,
        hm_gex=hm_gex,
        hm_iv=hm_iv,
        stats=stats,
        levels=levels,
        profile_df=profile_df,
        sensitivity_df=sensitivity_df,
        strike_support_df=strike_support_df,
        expiration_support_df=expiration_support_df,
        wall_credibility_info=wall_credibility_info,
        scenarios_df=scenarios_df,
        spot=spot,
        spot_source=spot_source,
        date_label=date_label,
        num_exps=len(target_exps),
        spot_info=spot_info,
        run_metadata=run_metadata,
        expected_move_info=em_analysis,
    )
