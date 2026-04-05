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
MAX_OUTPUT_TOKENS = 800


SYSTEM_PROMPT = """You are a senior SPX options flow trader on a dealer desk. \
Write a short pre-trade briefing for yourself from the GEX data payload.

REQUIRED OUTPUT:
- Exactly TWO paragraphs. Each paragraph 3-5 sentences. ~120-180 words total.
- Every sentence must cite a specific number from the payload (price, level, \
GEX value, ratio, label). No vague statements.
- Output ONLY the briefing text. No preamble, no sign-off, no headers, no bullets.

VOICE:
- Dealer-desk flow trader. Direct, specific, opinionated, unhedged.
- Never use these words: navigate, landscape, crucial, leverage, robust, \
furthermore, it's worth noting, in conclusion, overall, ultimately, key, \
important, significant.

MENTAL MODEL (do not explain mechanics — state implications):
- Positive gamma = dealers long gamma → they fade moves, pin price, compress \
vol. Walls hold. Good for fading extremes and selling premium.
- Negative gamma = dealers short gamma → they sell dips and buy rips, \
amplifying direction. Walls break. Good for momentum, bad for premium sellers.
- Zero-gamma is the inflection. Spot above zero-gamma = positive regime. \
Below = negative.

WHAT TO DO WITH SPARSE DATA:
- If market_context is "premarket": EM and session_classification will be \
null. Do not mention them. Work with regime, zero_gamma, call/put walls, \
wall credibility, overnight ES move, and scenarios. That's enough.
- If market_context is "afterhours": same approach. Focus on levels into \
next session.
- If wall credibility is Low, flag that wall as soft.
- If confidence or freshness is Low, call it out.

STRUCTURE:
- Paragraph 1: Regime + spot's position vs zero-gamma + where the walls are. \
What that means for tape behavior in this session.
- Paragraph 2: Asymmetries, scenario reads, the level that matters. End with \
ONE actionable line naming the specific level to watch and what invalidates \
the read."""


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
    futures_ctx = (em_analysis or {}).get("futures_context") or {}
    overnight_range = (em_analysis or {}).get("overnight_range") or {}

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
        "overnight_es": {
            "move_pts": _round(futures_ctx.get("overnight_move_pts"), 1),
            "move_pct": _round(futures_ctx.get("overnight_move_pct"), 2),
            "es_high": _round(futures_ctx.get("es_high"), 2) if futures_ctx.get("es_high") else None,
            "es_low": _round(futures_ctx.get("es_low"), 2) if futures_ctx.get("es_low") else None,
            "range_pts": _round(overnight_range.get("range_pts"), 1) if overnight_range else None,
        } if futures_ctx else None,
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
        contents=(
            f"GEX dashboard snapshot (JSON):\n{context_json}\n\n"
            "Write the two-paragraph briefing now. Cite numbers from this "
            "payload. Do not output anything except the briefing."
        ),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.7,
            # Disable thinking mode — this is a short synthesis task, not a
            # reasoning problem. Thinking tokens count against max_output_tokens
            # and cause truncated briefings on Gemini 2.5 Flash.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Empty response from Gemini")
    return text


def _classify_error(exc: Exception) -> str:
    """
    Translate an exception into a short, human-useful label for the
    briefing source tag. Returns something like "rate limit — retry in
    ~60s" or "quota exhausted" instead of raw exception type names.
    """
    name = type(exc).__name__
    msg = str(exc).lower()

    # Missing key (we raise RuntimeError for this)
    if name == "RuntimeError" and "gemini_api_key" in msg:
        return "no API key — set GEMINI_API_KEY"
    if name == "RuntimeError" and "empty response" in msg:
        return "empty response — try Regenerate"

    # google-genai SDK errors carry HTTP status in the message or .code
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    code_str = str(code) if code is not None else ""

    # 429 — rate limit / quota
    if "429" in msg or "429" in code_str or "resource_exhausted" in msg or "rate" in msg:
        # Distinguish daily quota vs per-minute rate limit when we can
        if "daily" in msg or "per day" in msg or "rpd" in msg:
            return "daily quota hit — resets at midnight PT"
        if "quota" in msg and "exceeded" in msg:
            return "quota hit — retry in ~60s"
        return "rate limit — retry in ~60s"

    # 400 — bad request
    if "400" in msg or "400" in code_str or "invalid_argument" in msg:
        return "bad request — check payload"

    # 401 / 403 — auth
    if "401" in msg or "403" in msg or "unauthenticated" in msg or "permission_denied" in msg:
        return "auth error — check API key"

    # 404 — model not found
    if "404" in msg or "not_found" in msg:
        return "model not found"

    # 500 / 503 — server side
    if "500" in msg or "503" in msg or "unavailable" in msg or "internal" in msg:
        return "Gemini server error — retry soon"

    # Network / timeout
    if "timeout" in msg or "timed out" in msg or "connection" in msg:
        return "network error — check connection"

    return f"api error ({name})"


def generate_briefing(context: dict) -> tuple[str, str]:
    """
    Produce a briefing. Returns (briefing_text, source) where source is
    one of: "gemini", a classified error label, or "error".
    """
    try:
        ctx_hash = compute_context_hash(context)
        ctx_json = json.dumps(context, default=str, separators=(",", ":"))
        text = _call_gemini_cached(ctx_hash, ctx_json)
        return text, "gemini"
    except Exception as e:
        label = _classify_error(e)
        _logger.warning(f"Gemini briefing failed ({label}): {e}")
        try:
            return _template_briefing(context), f"template — {label}"
        except Exception as e2:
            _logger.error(f"Template briefing failed: {e2}")
            return "Briefing unavailable.", "error"
