from __future__ import annotations

import numpy as np
import pandas as pd

from phase1.config import (
    SUPPORT_MAX_SPREAD_FOR_SCORE,
    SUPPORT_HIGH_THRESHOLD,
    SUPPORT_MODERATE_THRESHOLD,
    STRIKE_SUPPORT_W_OI,
    STRIKE_SUPPORT_W_BREADTH,
    STRIKE_SUPPORT_W_CONTRACTS,
    STRIKE_SUPPORT_W_DIRECT_IV,
    STRIKE_SUPPORT_W_SPREAD,
    STRIKE_SUPPORT_W_TWO_SIDED,
    EXP_SUPPORT_W_OI,
    EXP_SUPPORT_W_CONTRACTS,
    EXP_SUPPORT_W_STRIKE_BREADTH,
    EXP_SUPPORT_W_DIRECT_IV,
    EXP_SUPPORT_W_SPREAD,
)


def _safe_mean_numeric(series):
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return float(vals.mean()) if len(vals) else None


def _norm(value, ref):
    if value is None or ref is None or ref <= 0:
        return 0.0
    return float(max(0.0, min(float(value) / float(ref), 1.0)))


def label_support_score(score):
    if score >= SUPPORT_HIGH_THRESHOLD:
        return "High"
    if score >= SUPPORT_MODERATE_THRESHOLD:
        return "Moderate"
    return "Low"


def _score_spread(avg_spread):
    if avg_spread is None:
        return 0.0
    return float(max(0.0, 1.0 - min(avg_spread / SUPPORT_MAX_SPREAD_FOR_SCORE, 1.0)))


def build_strike_support_df(records, selected_exp_count):
    columns = [
        "strike",
        "supporting_expirations",
        "used_contracts",
        "total_oi",
        "call_contracts",
        "put_contracts",
        "direct_iv_count",
        "synthetic_iv_count",
        "avg_spread",
        "net_gex",
        "abs_net_gex",
        "direct_iv_share",
        "support_score",
        "support_label",
    ]

    if not records:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(records)

    grouped = (
        df.groupby("strike", as_index=False)
        .agg(
            supporting_expirations=("expiration", "nunique"),
            used_contracts=("oi", "size"),
            total_oi=("oi", "sum"),
            call_contracts=("is_call", "sum"),
            put_contracts=("is_put", "sum"),
            direct_iv_count=("iv_source", lambda s: int((s == "direct_iv").sum())),
            synthetic_iv_count=("iv_source", lambda s: int((s == "synthetic_iv").sum())),
            avg_spread=("spread", _safe_mean_numeric),
            net_gex=("net_gex", "sum"),
            abs_net_gex=("abs_gex", "sum"),
        )
    )

    oi_ref = float(grouped["total_oi"].quantile(0.90)) if not grouped.empty else 1.0
    if oi_ref <= 0:
        oi_ref = max(float(grouped["total_oi"].max()), 1.0)

    contract_ref = float(grouped["used_contracts"].quantile(0.90)) if not grouped.empty else 1.0
    if contract_ref <= 0:
        contract_ref = max(float(grouped["used_contracts"].max()), 1.0)

    exp_ref = max(float(selected_exp_count), 1.0)

    scores = []
    labels = []
    direct_shares = []

    for _, row in grouped.iterrows():
        direct_share = float(row["direct_iv_count"] / row["used_contracts"]) if row["used_contracts"] > 0 else 0.0
        spread_score = _score_spread(row["avg_spread"])
        breadth_score = _norm(row["supporting_expirations"], exp_ref)
        oi_score = _norm(row["total_oi"], oi_ref)
        contract_score = _norm(row["used_contracts"], contract_ref)
        two_sided_score = 1.0 if row["call_contracts"] > 0 and row["put_contracts"] > 0 else 0.5

        support_score = 100.0 * (
            STRIKE_SUPPORT_W_OI * oi_score
            + STRIKE_SUPPORT_W_BREADTH * breadth_score
            + STRIKE_SUPPORT_W_CONTRACTS * contract_score
            + STRIKE_SUPPORT_W_DIRECT_IV * direct_share
            + STRIKE_SUPPORT_W_SPREAD * spread_score
            + STRIKE_SUPPORT_W_TWO_SIDED * two_sided_score
        )

        direct_shares.append(direct_share)
        scores.append(round(float(support_score), 1))
        labels.append(label_support_score(float(support_score)))

    grouped["direct_iv_share"] = direct_shares
    grouped["support_score"] = scores
    grouped["support_label"] = labels

    grouped = grouped.sort_values(
        ["abs_net_gex", "total_oi", "support_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return grouped[columns]


def build_expiration_support_df(records):
    columns = [
        "expiration",
        "strikes_used",
        "used_contracts",
        "total_oi",
        "direct_iv_count",
        "synthetic_iv_count",
        "avg_spread",
        "abs_net_gex",
        "direct_iv_share",
        "support_score",
        "support_label",
    ]

    if not records:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(records)

    grouped = (
        df.groupby("expiration", as_index=False)
        .agg(
            strikes_used=("strike", "nunique"),
            used_contracts=("oi", "size"),
            total_oi=("oi", "sum"),
            direct_iv_count=("iv_source", lambda s: int((s == "direct_iv").sum())),
            synthetic_iv_count=("iv_source", lambda s: int((s == "synthetic_iv").sum())),
            avg_spread=("spread", _safe_mean_numeric),
            abs_net_gex=("abs_gex", "sum"),
        )
    )

    oi_ref = float(grouped["total_oi"].quantile(0.90)) if not grouped.empty else 1.0
    if oi_ref <= 0:
        oi_ref = max(float(grouped["total_oi"].max()), 1.0)

    contract_ref = float(grouped["used_contracts"].quantile(0.90)) if not grouped.empty else 1.0
    if contract_ref <= 0:
        contract_ref = max(float(grouped["used_contracts"].max()), 1.0)

    strikes_ref = float(grouped["strikes_used"].quantile(0.90)) if not grouped.empty else 1.0
    if strikes_ref <= 0:
        strikes_ref = max(float(grouped["strikes_used"].max()), 1.0)

    scores = []
    labels = []
    direct_shares = []

    for _, row in grouped.iterrows():
        direct_share = float(row["direct_iv_count"] / row["used_contracts"]) if row["used_contracts"] > 0 else 0.0
        spread_score = _score_spread(row["avg_spread"])
        oi_score = _norm(row["total_oi"], oi_ref)
        contract_score = _norm(row["used_contracts"], contract_ref)
        strike_breadth_score = _norm(row["strikes_used"], strikes_ref)

        support_score = 100.0 * (
            EXP_SUPPORT_W_OI * oi_score
            + EXP_SUPPORT_W_CONTRACTS * contract_score
            + EXP_SUPPORT_W_STRIKE_BREADTH * strike_breadth_score
            + EXP_SUPPORT_W_DIRECT_IV * direct_share
            + EXP_SUPPORT_W_SPREAD * spread_score
        )

        direct_shares.append(direct_share)
        scores.append(round(float(support_score), 1))
        labels.append(label_support_score(float(support_score)))

    grouped["direct_iv_share"] = direct_shares
    grouped["support_score"] = scores
    grouped["support_label"] = labels

    grouped = grouped.sort_values("expiration").reset_index(drop=True)
    return grouped[columns]