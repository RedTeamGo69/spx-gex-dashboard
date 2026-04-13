# =============================================================================
# model_persistence.py
# HAR model serialization — save/load fitted models to Postgres (with an
# on-disk pickle fallback if the DB save fails transiently).
# =============================================================================

import logging
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import statsmodels.api as sm

log = logging.getLogger(__name__)

# Pickle fallback path — used only if the DB save raises
MODEL_DIR = Path(__file__).parent / "models"

# Schema version for saved model payloads. Bump this whenever the payload
# dict shape changes, or whenever the HAR feature definitions change in a
# way that invalidates previously-fitted models (so load_model refuses to
# return a stale-but-loadable model instead of silently producing wrong
# forecasts). Load will raise a clear error on mismatch and the UI will
# fall through to "refit the model" guidance.
SCHEMA_VERSION = 1


def save_model(
    result: sm.regression.linear_model.RegressionResultsWrapper,
    feature_cols: list[str],
    model_name: str,
    metrics: dict,
    conn=None,
):
    """Save the fitted model + metadata to Postgres (on-disk pickle fallback)."""
    payload = {
        "schema_version":     SCHEMA_VERSION,
        "statsmodels_version": sm.__version__,
        "python_version":     f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "result":             result,
        "feature_cols":       feature_cols,
        "model_name":         model_name,
        "metrics":            metrics,
        "fitted_at":          datetime.now(timezone.utc).isoformat(),
    }

    blob = pickle.dumps(payload)
    now = datetime.now(timezone.utc).isoformat()

    if conn is not None:
        try:
            import psycopg2
            conn.execute("""
                INSERT INTO saved_models (model_name, model_data, fitted_at, updated_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT(model_name) DO UPDATE SET
                    model_data = EXCLUDED.model_data,
                    fitted_at  = EXCLUDED.fitted_at,
                    updated_at = EXCLUDED.updated_at
            """, (model_name, psycopg2.Binary(blob), payload["fitted_at"], now))
            conn.commit()
            log.info(f"Model saved to database: {model_name}")
            return
        except Exception as e:
            log.warning(f"DB save failed, falling back to pickle: {e}")

    # Fallback: pickle to disk
    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / f"{model_name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    log.info(f"Model saved to disk: {path}")


class IncompatibleModelError(Exception):
    """Raised when a loaded model payload doesn't match the current schema.

    The UI catches this as "saved model incompatible" and prompts the user
    to click Generate Forecast to refit. Raising early is strictly better
    than silently returning a model that would produce wrong predictions.
    """


def _validate_payload(payload: dict, source: str) -> None:
    """Raise IncompatibleModelError if the payload schema is a version we
    can't safely load. Missing schema_version is treated as pre-versioning
    legacy (version 0) and accepted with a warning — existing models keep
    working across this commit. Going forward, bumping SCHEMA_VERSION and
    catching the mismatch is how we force a refit when the HAR feature
    definitions change in a way that invalidates old fits."""
    # Sanity-check required fields first — if these are missing the
    # payload is corrupted, not just old.
    for required in ("result", "feature_cols", "model_name"):
        if required not in payload:
            raise IncompatibleModelError(
                f"{source}: saved model payload missing required field "
                f"'{required}' — payload is corrupted or incomplete."
            )

    payload_version = payload.get("schema_version", 0)
    if payload_version == 0:
        log.warning(
            f"{source}: loading legacy pre-versioned model (no schema_version "
            f"field). Current SCHEMA_VERSION={SCHEMA_VERSION}. Consider "
            f"refitting to pick up any feature definition changes."
        )
        return
    if payload_version != SCHEMA_VERSION:
        raise IncompatibleModelError(
            f"{source}: saved model schema_version={payload_version} but "
            f"current SCHEMA_VERSION={SCHEMA_VERSION}. Refit to regenerate."
        )
    # Warn if statsmodels version differs so unpickle quirks are easier
    # to diagnose. Don't raise — minor version bumps are usually
    # backward-compatible and forcing a refit on every upgrade is too
    # aggressive.
    saved_sm = payload.get("statsmodels_version")
    if saved_sm and saved_sm != sm.__version__:
        log.warning(
            f"{source}: saved model fitted with statsmodels {saved_sm}, "
            f"loading with {sm.__version__} — should still work but flag "
            f"any prediction anomalies."
        )


def load_model(model_name: str, conn=None) -> dict:
    """Load a previously saved model from Postgres (on-disk pickle fallback).

    Raises IncompatibleModelError if the payload schema_version doesn't
    match the current SCHEMA_VERSION constant — caller should treat this
    as "model needs refitting" and prompt the user accordingly.
    """
    if conn is not None:
        try:
            cur = conn.execute(
                "SELECT model_data, fitted_at FROM saved_models WHERE model_name = %s",
                (model_name,)
            )
            row = cur.fetchone()
            if row:
                blob = row[0]
                if isinstance(blob, memoryview):
                    blob = bytes(blob)
                payload = pickle.loads(blob)
                _validate_payload(payload, f"DB:{model_name}")
                log.info(f"Model loaded from database: {model_name}  (fitted {payload['fitted_at']})")
                return payload
        except IncompatibleModelError:
            # Don't swallow — let the caller decide what to do
            raise
        except Exception as e:
            log.warning(f"DB load failed, trying pickle fallback: {e}")

    # Fallback: pickle from disk
    path = MODEL_DIR / f"{model_name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No saved model found for {model_name}")

    with open(path, "rb") as f:
        payload = pickle.load(f)

    _validate_payload(payload, f"disk:{path}")
    log.info(f"Model loaded from disk: {path}  (fitted {payload['fitted_at']})")
    return payload
