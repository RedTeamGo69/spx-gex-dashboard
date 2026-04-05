"""
AI Trading Briefing — Gemini-powered synthesis of GEX dashboard state.

Reads the already-computed dashboard data structures (regime, levels, walls,
expected move, quality signals) and produces a short, trader-voiced briefing.
Falls back to a deterministic template if the API is unavailable.

Quota budget (gemini-2.5-flash free tier): 10 RPM / 500 RPD / 250K TPM.
Content-hash caching + 10-minute TTL keeps usage well under limits.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

import streamlit as st

_logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"
CACHE_TTL_SECONDS = 600  # 10 minutes — content hash forces refresh on material change
MAX_OUTPUT_TOKENS = 500


SYSTEM_PROMPT = """You are a senior SPX options flow trader on a dealer desk. \
You are writing a short pre-trade briefing for yourself based on live gamma \
exposure data.

VOICE:
- Dealer-desk flow trader. Direct, specific, unhedged.
- Never use these words: navigate, landscape, crucial, leverage, robust, \
furthermore, it's worth noting, in conclusion, overall, ultimately.
- No bullet points, no headers, no bold. Two short paragraphs, max ~180 words.

GROUNDING:
- Every claim must cite a specific number from the payload (a price, a level, \
a GEX value, a ratio). No vague statements.
- Do not explain the mechanics of gamma. State implications.
- Mental model:
    * Positive gamma regime = dealers long gamma → they fade moves, pin price, \
suppress vol. Walls act as real support/resistance.
    * Negative gamma regime = dealers short gamma → they chase (sell dips, buy \
rips), amplifying moves. Walls are weaker, breakouts extend.
- If wall credibility is Low, flag that wall as soft. If data quality \
(confidence/freshness) is Low, say so directly — don't pretend bad data is good.

STRUCTURE:
- Paragraph 1: Regime + where spot sits relative to zero-gamma and the walls. \
What that means for today's tape.
- Paragraph 2: Asymmetries, notable magnets, what to watch. End with ONE \
actionable line: the specific level that matters and what invalidates the read.

Output ONLY the briefing text. No preamble, no sign-off."""


# ──────────────────────────────────────────────────────────────────────────────
# Context assembly
# ──────────────────────────────────────────────────────────────────────────────
def _safe_get(d: Any, *keys, default=None):
    """Nested dict lookup that won't raise."""
    cur = d
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            return default
    return cur if cur is not None else default


def _round(v, n=2):
    try:
        return round(float(v), n)
    except (TypeError, ValueError):
        return None


def build_briefing_context(data, em_analysis: dict) -> dict:
    """
    Build a compact (~1KB) structured payload for the LLM.

    Only includes high-signal fields. Drops cluster arrays, verbose reason
    lists, and raw DataFrames — summarizes to labels.
    """
    levels = data.levels or {}
    regime = data.regime_info or {}
    wall_cred = data.wall_cred or {}
    conf = data.confidence_info or {}
    stale = data.staleness_info or {}

    em = (em_analysis or {}).get("expected_move", {}) or {}
    classification = (em_analysis or {}).get("classification", {}) or {}
    overnight = (em_analysis or {}).get("overnight_move", {}) or {}

    ctx: dict[str, Any] = {
        "ticker_spot": _round(data.spot, 2),
        "market_context": (em_analysis or {}).get("market_context", "live"),
        "regime": {
            "name": regime.get("regime"),
            "distance_text": regime.get("distance_text"),
            "abs_distance_pts": _round(regime.get("abs_distance"), 2),
        },
        "levels": {
            "zero_gamma": _round(levels.get("zero_gamma"), 2),
            "zero_gamma_type": levels.get("zero_gamma_type"),
            "call_wall": _round(levels.get("call_wall"), 2),
            "call_wall_gex": _round(levels.get("call_wall_gex"), 1),
            "put_wall": _round(levels.get("put_wall"), 2),
            "put_wall_gex": _round(levels.get("put_wall_gex"), 1),
        },
        "wall_credibility": {
            "call_wall": _safe_get(wall_cred, "call_wall", "label"),
            "call_wall_score": _round(_safe_get(wall_cred, "call_wall", "score"), 0),
            "put_wall": _safe_get(wall_cred, "put_wall", "label"),
            "put_wall_score": _round(_safe_get(wall_cred, "put_wall", "score"), 0),
            "zero_gamma": _safe_get(wall_cred, "zero_gamma", "label"),
            "zero_gamma_score": _round(_safe_get(wall_cred, "zero_gamma", "score"), 0),
        },
        "expected_move": {
            "em_pts": _round(em.get("expected_move_pts"), 1),
            "upper": _round(em.get("upper_level"), 2),
            "lower": _round(em.get("lower_level"), 2),
        },
        "session_classification": {
            "type": classification.get("classification"),
            "bias": classification.get("bias"),
            "vol_budget_used_pct": _round(
                (classification.get("move_ratio") or 0) * 100, 0
            ) if classification.get("move_ratio") is not None else None,
            "move_source": classification.get("move_source"),
        },
        "session_move": {
            "pts": _round(overnight.get("overnight_move_pts"), 1),
            "pct": _round(overnight.get("overnight_move_pct"), 2),
        },
        "data_quality": {
            "confidence": conf.get("label"),
            "freshness": stale.get("freshness_label"),
            "market_open": bool(stale.get("market_open", True)),
        },
    }

    # Top 3 scenario rows: base + most asymmetric shocks (±1% typical sweet spot)
    sc_df = getattr(data, "scenarios_df", None)
    if sc_df is not None and not sc_df.empty:
        try:
            wanted = sc_df[sc_df["scenario"].isin(["Base", "Spot -1.0%", "Spot +1.0%"])]
            ctx["scenarios"] = [
                {
                    "scenario": row["scenario"],
                    "spot": _round(row["spot"], 2),
                    "zero_gamma": _round(row["zero_gamma"], 2),
                    "call_wall": _round(row["call_wall"], 2),
                    "put_wall": _round(row["put_wall"], 2),
                    "regime": row["gamma_regime"],
                }
                for _, row in wanted.iterrows()
            ]
        except Exception:
            pass

    return ctx


def compute_context_hash(context: dict) -> str:
    """
    Stable hash keyed on fields that meaningfully change the briefing.

    Uses rounded/bucketed values so tiny price jitter doesn't bust the cache.
    """
    spot = context.get("ticker_spot") or 0
    spot_bucket = round(spot * 4) / 4  # snap to 0.25 pts

    key = (
        spot_bucket,
        _safe_get(context, "regime", "name"),
        _safe_get(context, "levels", "zero_gamma"),
        _safe_get(context, "levels", "call_wall"),
        _safe_get(context, "levels", "put_wall"),
        _safe_get(context, "expected_move", "em_pts"),
        _safe_get(context, "session_classification", "type"),
        _safe_get(context, "data_quality", "confidence"),
        _safe_get(context, "data_quality", "freshness"),
        context.get("market_context"),
    )
    return hashlib.sha1(str(key).encode("utf-8")).hexdigest()[:16]


# ──────────────────────────────────────────────────────────────────────────────
# Template fallback (no API)
# ──────────────────────────────────────────────────────────────────────────────
def _template_briefing(context: dict) -> str:
    """Deterministic f-string briefing. Used when Gemini is unavailable."""
    spot = context.get("ticker_spot") or 0
    regime_name = _safe_get(context, "regime", "name") or "unknown regime"
    distance_text = _safe_get(context, "regime", "distance_text") or ""
    zg = _safe_get(context, "levels", "zero_gamma")
    cw = _safe_get(context, "levels", "call_wall")
    pw = _safe_get(context, "levels", "put_wall")
    cw_label = _safe_get(context, "wall_credibility", "call_wall") or "?"
    pw_label = _safe_get(context, "wall_credibility", "put_wall") or "?"
    em_pts = _safe_get(context, "expected_move", "em_pts")
    em_upper = _safe_get(context, "expected_move", "upper")
    em_lower = _safe_get(context, "expected_move", "lower")
    conf = _safe_get(context, "data_quality", "confidence") or "?"
    fresh = _safe_get(context, "data_quality", "freshness") or "?"

    is_positive = "Positive" in (regime_name or "")
    tape_read = (
        "dealers long gamma, fading moves and pinning price"
        if is_positive
        else "dealers short gamma, chasing direction and amplifying moves"
    )

    p1 = (
        f"Spot {spot:.2f} in {regime_name} ({distance_text}), zero-gamma at "
        f"{zg}. Read: {tape_read}. "
    )
    if cw is not None and pw is not None:
        p1 += (
            f"Call wall {cw} ({cw_label} credibility), put wall {pw} "
            f"({pw_label} credibility)."
        )

    if em_pts and em_upper and em_lower:
        p2 = (
            f"Expected move ±{em_pts:.0f} pts → range {em_lower:.0f}-"
            f"{em_upper:.0f}. "
        )
    else:
        p2 = ""

    # Pick the level that matters
    key_level = zg if is_positive else (cw if cw else zg)
    invalidator = pw if is_positive else zg

    p2 += (
        f"Watch {key_level} — that's the magnet. Loses "
        f"{invalidator} and the read flips. "
        f"Data quality: confidence {conf}, freshness {fresh}."
    )

    return f"{p1}\n\n{p2}"


# ──────────────────────────────────────────────────────────────────────────────
# Gemini API call (cached)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _call_gemini_cached(context_hash: str, context_json: str) -> str:
    """
    Cached Gemini call. context_hash is the cache key; context_json is the
    actual payload sent to the model. API key is read from env/secrets
    inside the function so it doesn't participate in cache hashing.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    # Import inside to keep module importable without the SDK installed
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"GEX dashboard snapshot:\n{context_json}\n\nWrite the briefing.",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.6,
        ),
    )

    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Empty response from Gemini")
    return text


def generate_briefing(context: dict) -> tuple[str, str]:
    """
    Produce a briefing. Returns (briefing_text, source) where source is
    one of: "gemini", "template", "template (api_error)".
    """
    try:
        ctx_hash = compute_context_hash(context)
        ctx_json = json.dumps(context, default=str, separators=(",", ":"))
        text = _call_gemini_cached(ctx_hash, ctx_json)
        return text, "gemini"
    except Exception as e:
        _logger.warning(f"Gemini briefing failed, falling back to template: {e}")
        try:
            return _template_briefing(context), f"template ({type(e).__name__})"
        except Exception as e2:
            _logger.error(f"Template briefing failed: {e2}")
            return "Briefing unavailable.", "error"
