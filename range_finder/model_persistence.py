# =============================================================================
# model_persistence.py
# HAR model serialization — save/load fitted models to Postgres or pickle.
# =============================================================================

import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import statsmodels.api as sm

log = logging.getLogger(__name__)

# Model save path
MODEL_DIR = Path(__file__).parent / "models"


def save_model(
    result: sm.regression.linear_model.RegressionResultsWrapper,
    feature_cols: list[str],
    model_name: str,
    metrics: dict,
    conn=None,
):
    """Save the fitted model + metadata to database (or pickle fallback)."""
    payload = {
        "result":       result,
        "feature_cols": feature_cols,
        "model_name":   model_name,
        "metrics":      metrics,
        "fitted_at":    datetime.now(timezone.utc).isoformat(),
    }

    blob = pickle.dumps(payload)
    now = datetime.now(timezone.utc).isoformat()

    if conn is not None:
        try:
            from range_finder.db import get_backend
            if get_backend() == "postgres":
                import psycopg2
                conn.execute("""
                    INSERT INTO saved_models (model_name, model_data, fitted_at, updated_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT(model_name) DO UPDATE SET
                        model_data = EXCLUDED.model_data,
                        fitted_at  = EXCLUDED.fitted_at,
                        updated_at = EXCLUDED.updated_at
                """, (model_name, psycopg2.Binary(blob), payload["fitted_at"], now))
            else:
                conn.execute("""
                    INSERT INTO saved_models (model_name, model_data, fitted_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(model_name) DO UPDATE SET
                        model_data = excluded.model_data,
                        fitted_at  = excluded.fitted_at,
                        updated_at = excluded.updated_at
                """, (model_name, blob, payload["fitted_at"], now))
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


def load_model(model_name: str, conn=None) -> dict:
    """Load a previously saved model from database (or pickle fallback)."""
    if conn is not None:
        try:
            from range_finder.db import get_backend
            if get_backend() == "postgres":
                cur = conn.execute(
                    "SELECT model_data, fitted_at FROM saved_models WHERE model_name = %s",
                    (model_name,)
                )
            else:
                cur = conn.execute(
                    "SELECT model_data, fitted_at FROM saved_models WHERE model_name = ?",
                    (model_name,)
                )
            row = cur.fetchone()
            if row:
                blob = row[0]
                if isinstance(blob, memoryview):
                    blob = bytes(blob)
                payload = pickle.loads(blob)
                log.info(f"Model loaded from database: {model_name}  (fitted {payload['fitted_at']})")
                return payload
        except Exception as e:
            log.warning(f"DB load failed, trying pickle fallback: {e}")

    # Fallback: pickle from disk
    path = MODEL_DIR / f"{model_name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No saved model found for {model_name}")

    with open(path, "rb") as f:
        payload = pickle.load(f)

    log.info(f"Model loaded from disk: {path}  (fitted {payload['fitted_at']})")
    return payload
