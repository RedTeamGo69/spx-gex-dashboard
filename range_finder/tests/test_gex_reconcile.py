"""GEX warning reconciliation (gex_bridge.reconcile_gex_warnings).

The dashboard keeps anchored and live gamma observations for research. The
reconciler must compose AT MOST ONE regime message (flip-aware) and state that
GEX is not applied to the live buffer or strikes.
"""
from range_finder.gex_bridge import (
    GEXContext, adjust_spread_with_gex, reconcile_gex_warnings,
)
from range_finder.spread_levels import GEX_NEGATIVE_REGIME_WARNING


WALL_NOTE = "Call short (6000) is only 15 pts below call wall (6015) — consider widening"
ZG_NOTE = "Spot (5980) is very close to zero-gamma (5990, 0.17% away) — regime could flip intraday. Monitor closely."
FOMC_WARNING = "FOMC week — minimum width floor raised to 100 pts; gaps through strikes are possible"


def _regime_messages(warnings: list) -> list:
    """Warnings that talk about the GEX regime (composed by the reconciler)."""
    stems = ("Positive GEX regime", "Negative GEX regime",
             "GEX regime flipped", "Anchored GEX regime", "Live GEX regime")
    return [w for w in warnings if w.startswith(stems)]


# ── agreement cases ───────────────────────────────────────────────────────────

def test_agree_positive_single_message_no_false_adjustment_claim():
    out = reconcile_gex_warnings(
        [FOMC_WARNING], [WALL_NOTE],
        anchored_gex_flag=1, live_regime="Positive Gamma",
    )
    regime = _regime_messages(out)
    assert len(regime) == 1
    assert "agree" in regime[0]
    assert "tightened" not in regime[0]          # the old false claim
    assert "not applied to the live buffer or strikes" in regime[0]
    assert FOMC_WARNING in out and WALL_NOTE in out


def test_agree_negative_replaces_legacy_warning():
    out = reconcile_gex_warnings(
        [FOMC_WARNING, GEX_NEGATIVE_REGIME_WARNING], [],
        anchored_gex_flag=-1, live_regime="Negative Gamma",
    )
    assert GEX_NEGATIVE_REGIME_WARNING not in out
    regime = _regime_messages(out)
    assert len(regime) == 1
    assert "Negative GEX regime" in regime[0] and "agree" in regime[0]


# ── flip cases (the reported bug) ─────────────────────────────────────────────

def test_flip_negative_anchor_positive_live():
    out = reconcile_gex_warnings(
        [GEX_NEGATIVE_REGIME_WARNING], [],
        anchored_gex_flag=-1, live_regime="Positive Gamma",
    )
    regime = _regime_messages(out)
    assert len(regime) == 1
    assert "flipped" in regime[0]
    assert "negative at the Monday anchor" in regime[0]
    assert "now reads positive" in regime[0]
    assert "not applied to the live buffer or strikes" in regime[0]


def test_flip_positive_anchor_negative_live():
    out = reconcile_gex_warnings(
        [], [],
        anchored_gex_flag=1, live_regime="Negative Gamma",
    )
    regime = _regime_messages(out)
    assert len(regime) == 1
    assert "positive at the Monday anchor" in regime[0]
    assert "now reads negative" in regime[0]


def test_flip_never_renders_both_regimes_as_separate_lines():
    """The exact user-visible bug: 'Negative GEX regime' and 'Positive GEX'
    must never coexist as separate warning lines."""
    out = reconcile_gex_warnings(
        [GEX_NEGATIVE_REGIME_WARNING], [WALL_NOTE],
        anchored_gex_flag=-1, live_regime="Positive Gamma",
    )
    assert len(_regime_messages(out)) == 1


# ── unclear / missing sides ───────────────────────────────────────────────────

def test_anchored_negative_live_unclear():
    out = reconcile_gex_warnings(
        [GEX_NEGATIVE_REGIME_WARNING], [],
        anchored_gex_flag=-1, live_regime="transition",
    )
    regime = _regime_messages(out)
    assert len(regime) == 1
    assert "unclear" in regime[0]
    assert "not applied to the live buffer or strikes" in regime[0]


def test_anchored_none_live_positive_advisory():
    # The 0DTE default path: daily matrix has no GEX columns.
    out = reconcile_gex_warnings(
        [], [ZG_NOTE],
        anchored_gex_flag=None, live_regime="Positive Gamma",
        anchor_label="session anchor",
    )
    regime = _regime_messages(out)
    assert len(regime) == 1
    assert regime[0].startswith("Live GEX regime is positive")
    assert "not applied to the live buffer or strikes" in regime[0]


def test_anchored_neutral_live_negative():
    out = reconcile_gex_warnings(
        [], [],
        anchored_gex_flag=0, live_regime="Negative Gamma",
    )
    regime = _regime_messages(out)
    assert len(regime) == 1
    assert "neutral" in regime[0] and "negative" in regime[0]


def test_both_unknown_passthrough_order_preserved():
    out = reconcile_gex_warnings(
        [FOMC_WARNING, "some other warning"], [WALL_NOTE, ZG_NOTE],
        anchored_gex_flag=None, live_regime="unknown",
    )
    assert out == [FOMC_WARNING, "some other warning", WALL_NOTE, ZG_NOTE]


# ── stability ─────────────────────────────────────────────────────────────────

def test_reconcile_is_idempotent():
    once = reconcile_gex_warnings(
        [FOMC_WARNING, GEX_NEGATIVE_REGIME_WARNING], [WALL_NOTE],
        anchored_gex_flag=-1, live_regime="Positive Gamma",
    )
    twice = reconcile_gex_warnings(
        once, [WALL_NOTE],
        anchored_gex_flag=-1, live_regime="Positive Gamma",
    )
    assert twice == once


# ── regression pin on adjust_spread_with_gex ─────────────────────────────────

def test_adjust_spread_with_gex_emits_no_regime_story_notes():
    """Regime narrative moved to reconcile_gex_warnings; the annotation dict
    must no longer claim the buffer was tightened/widened."""
    from range_finder.spread_levels import SpreadPlan

    plan = SpreadPlan(
        week_start="2026-06-29", generated_at="now",
        spx_ref_close=6000.0, spx_ref_open=None,
        point_pct=0.01, upper_pct=0.012, lower_pct=0.008,
        vix_implied_pct=0.011, confidence_level=90,
        buffer_pct=0.003, buffer_pts=18.0, buffer_reason="base",
        effective_range_pct=0.015, effective_upper_px=6090.0,
        effective_lower_px=5910.0,
    )
    ctx = GEXContext(
        spot=6000.0, zero_gamma=5800.0, call_wall=6100.0, put_wall=5900.0,
        gamma_regime="Negative Gamma",
    )
    result = adjust_spread_with_gex(plan, ctx)
    joined = " ".join(result["gex_adjustment_notes"])
    assert "tightened" not in joined
    assert "widened" not in joined
    assert result["gex_regime_flag"] == -1     # regime data still exposed
