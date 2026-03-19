from __future__ import annotations

import pandas as pd

import phase1.gex_engine as gex_engine


def build_default_scenarios():
    """
    Default what-if scenarios.
    rate_shock_bps is in basis points.
    """
    return [
        {"name": "Base", "spot_shock_pct": 0.0000, "rate_shock_bps": 0},
        {"name": "Spot -0.50%", "spot_shock_pct": -0.0050, "rate_shock_bps": 0},
        {"name": "Spot -0.25%", "spot_shock_pct": -0.0025, "rate_shock_bps": 0},
        {"name": "Spot +0.25%", "spot_shock_pct": 0.0025, "rate_shock_bps": 0},
        {"name": "Spot +0.50%", "spot_shock_pct": 0.0050, "rate_shock_bps": 0},
        {"name": "Rate -50bp", "spot_shock_pct": 0.0000, "rate_shock_bps": -50},
        {"name": "Rate +50bp", "spot_shock_pct": 0.0000, "rate_shock_bps": 50},
    ]


def run_scenario_engine(all_options, base_spot, base_r, scenario_defs=None):
    """
    Reprice the current normalized option universe under spot/rate shocks
    and recompute call wall / put wall / zero gamma.

    Returns a DataFrame.
    """
    if scenario_defs is None:
        scenario_defs = build_default_scenarios()

    rows = []

    for sc in scenario_defs:
        spot_shock_pct = float(sc["spot_shock_pct"])
        rate_shock_bps = int(sc["rate_shock_bps"])

        shocked_spot = float(base_spot) * (1.0 + spot_shock_pct)
        shocked_r = float(base_r) + (rate_shock_bps / 10000.0)

        scenario_gex_df = gex_engine.compute_strike_gex_from_all_options(
            all_options=all_options,
            spot=shocked_spot,
            r=shocked_r,
        )

        levels = gex_engine.find_key_levels(
            gex_df=scenario_gex_df,
            spot=shocked_spot,
            all_options=all_options,
            r=shocked_r,
        )

        regime_info = gex_engine.get_gamma_regime_text(shocked_spot, levels["zero_gamma"])

        rows.append(
            {
                "scenario": sc["name"],
                "spot_shock_pct": spot_shock_pct,
                "rate_shock_bps": rate_shock_bps,
                "spot": round(float(shocked_spot), 2),
                "rate": round(float(shocked_r), 6),
                "call_wall": round(float(levels["call_wall"]), 2),
                "put_wall": round(float(levels["put_wall"]), 2),
                "zero_gamma": round(float(levels["zero_gamma"]), 2),
                "zero_gamma_type": levels.get("zero_gamma_type", "Unknown"),
                "gamma_regime": regime_info["regime"],
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    base_row = df.iloc[0]
    df["call_wall_move"] = df["call_wall"] - float(base_row["call_wall"])
    df["put_wall_move"] = df["put_wall"] - float(base_row["put_wall"])
    df["zero_gamma_move"] = df["zero_gamma"] - float(base_row["zero_gamma"])

    return df