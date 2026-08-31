"""Production policy for GEX influence on Spread Finder recommendations."""

# BUG-06 mitigation: the current normalized-GEX scale was never calibrated for
# live spread placement. Keep capture and research features active, but do not
# let GEX affect production forecasts, buffers, or strikes until the historical
# study supplies evidence for a replacement transformation.
GEX_LIVE_SPREAD_INFLUENCE_ENABLED = False
GEX_NORMALIZED_FEATURE = "gex_normalized"


def live_spread_feature_columns(feature_cols) -> list[str]:
    """Return model features allowed to drive a live Spread Finder plan."""
    cols = list(feature_cols)
    if GEX_LIVE_SPREAD_INFLUENCE_ENABLED:
        return cols
    return [col for col in cols if col != GEX_NORMALIZED_FEATURE]


def uses_disabled_gex_feature(feature_cols) -> bool:
    """Whether a saved fit contains GEX that production must not consume."""
    return GEX_NORMALIZED_FEATURE in set(feature_cols or []) and not (
        GEX_LIVE_SPREAD_INFLUENCE_ENABLED
    )
