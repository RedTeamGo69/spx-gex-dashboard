"""Transactional study repository; SQLite is used only by isolated fixtures.

Production uses a direct psycopg2 connection, not the legacy autocommit wrapper.
A model's four tiers and its inputs commit together. Different models can fail
independently. A logical slot prevents a refit/version/expiration replacing it.
"""
from contextlib import contextmanager
from datetime import date, datetime
import json
from pathlib import Path
import sqlite3

from .config import COHORT, MODELS, UNIVERSE, digest, json_text
from range_finder.recommendations import TIER_KEYS


class Store:
    def __init__(self, connection, *, admission_clock=None):
        self.conn = connection
        self.sqlite = isinstance(connection, sqlite3.Connection)
        self.admission_clock = admission_clock  # deterministic isolated DB tests only

    @classmethod
    def postgres(cls, url):
        import psycopg2
        if not url:
            raise ValueError("FORWARD_TEST_DATABASE_URL or DATABASE_URL is required")
        conn = psycopg2.connect(url, connect_timeout=15)
        conn.autocommit = False
        return cls(conn)

    def close(self):
        self.conn.close()

    def execute(self, sql, params=()):
        cur = self.conn.cursor()
        cur.execute(sql if self.sqlite else sql.replace("?", "%s"), params)
        return cur

    @contextmanager
    def transaction(self):
        try:
            if self.sqlite:
                self.conn.execute("BEGIN IMMEDIATE")
            yield
            self.conn.commit()
        except BaseException:
            self.conn.rollback()
            raise

    def query(self, sql, params=()):
        try:
            cur = self.execute(sql, params)
            names = [d[0] for d in cur.description]
            rows = [dict(zip(names, row)) for row in cur.fetchall()]
            cur.close()
            # Never leave a read idle in transaction, which pins Neon awake.
            self.conn.commit()
            return rows
        except BaseException:
            self.conn.rollback()
            raise

    def legacy_weekly(self, ticker):
        """Reuse existing history depth without changing any legacy rows.

        The interactive side-share quantile uses all persisted weekly bars,
        while model fitting uses a six-year read window. Preserve both rules.
        """
        import pandas as pd
        if ticker == "SPX":
            rows = self.query("SELECT * FROM weekly_spx ORDER BY week_start")
        else:
            rows = self.query("SELECT * FROM weekly_underlying WHERE ticker=? ORDER BY week_start", (ticker,))
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows).set_index("week_start")
        frame.index = pd.to_datetime(frame.index)
        if ticker != "SPX":
            frame = frame.rename(columns={"open":"spx_open", "high":"spx_high", "low":"spx_low", "close":"spx_close",
                                           "volume":"spx_volume", "return_pct":"spx_return", "vol_proxy_close":"vix_close"})
        return frame

    def migrate(self):
        root = Path(__file__).resolve().parents[2] / "migrations"
        with self.transaction():
            for stmt in (root / "001_forward_test.sql").read_text().split(";"):
                if stmt.strip():
                    self.execute(stmt)
            if self.sqlite:
                for table in ("ft_studies", "ft_forecasts", "ft_inputs", "ft_observations", "ft_scores"):
                    for op in ("UPDATE", "DELETE"):
                        self.execute(f"CREATE TRIGGER IF NOT EXISTS {table}_{op} BEFORE {op} ON {table} "
                                     "BEGIN SELECT RAISE(ABORT, 'Forward study evidence is immutable'); END")
                identity = " OR ".join(f"NEW.{k} IS NOT OLD.{k}" for k in
                                      ("slot_id", "study_id", "week_start", "ticker", "model", "model_version", "cohort"))
                self.execute("CREATE TRIGGER IF NOT EXISTS ft_slots_identity BEFORE UPDATE ON ft_slots WHEN "
                             + identity + " OR (OLD.status IN ('captured','missed') AND NEW.status <> OLD.status) "
                             "BEGIN SELECT RAISE(ABORT, 'Forward study slot identity is immutable'); END")
            else:
                self.execute((root / "002_forward_test_immutability.sql").read_text())

    def register(self, study_id, start_week, now, config):
        date.fromisoformat(start_week)
        with self.transaction():
            self.execute("INSERT INTO ft_studies VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING",
                         (study_id, start_week, now.isoformat(), json_text(config)))
        existing = self.query("SELECT * FROM ft_studies WHERE study_id=?", (study_id,))[0]
        if existing["start_week"] != start_week or json.loads(existing["config_json"]) != json.loads(json_text(config)):
            raise ValueError("Study already exists with a different start/configuration")

    def seed_week(self, study_id, week, model_version, now, *, cohort=COHORT):
        with self.transaction():
            for ticker in UNIVERSE:
                for model in MODELS:
                    slot_id = digest([study_id, week, ticker, model])
                    self.execute("INSERT INTO ft_slots VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?) "
                                 "ON CONFLICT DO NOTHING", (slot_id, study_id, week, ticker, model,
                                 model_version, cohort, "pending", now.isoformat()))

    def slots(self, study_id, week):
        return self.query("SELECT * FROM ft_slots WHERE study_id=? AND week_start=? ORDER BY ticker, model",
                          (study_id, week))

    def failure(self, slot, status, error, now):
        with self.transaction():
            self.execute("UPDATE ft_slots SET status=?, error=?, attempts=attempts+1, updated_at=? "
                         "WHERE slot_id=? AND status NOT IN ('captured','missed')",
                         (status, error[:1000], now.isoformat(), slot["slot_id"]))

    def freeze(self, slot, inputs, forecasts, now, week):
        if not week.admits(now) or any(not week.admits(datetime.fromisoformat(f['available_at'])) for f in forecasts):
            raise ValueError("Missed capture window; retrospective admission prohibited")
        if {f["tier"] for f in forecasts} != set(TIER_KEYS) or len(forecasts) != 4:
            raise ValueError("Capture must contain all four tiers exactly once")
        with self.transaction():
            # Row lock serializes competing runners; uniqueness is the final
            # backstop. Never use an application 'check then upsert' race.
            lock = "" if self.sqlite else " FOR UPDATE"
            row = self.execute("SELECT status FROM ft_slots WHERE slot_id=?" + lock,
                               (slot["slot_id"],)).fetchone()
            if row[0] in ("captured", "missed"):
                return False
            if not self.sqlite:
                db_now = self.admission_clock() if self.admission_clock else self.execute("SELECT clock_timestamp()").fetchone()[0]
                if not week.admits(db_now):
                    raise ValueError("Database admission time outside capture window")
            inputs = dict(inputs)
            if "raw_inputs" in inputs:
                shared = {"ticker": inputs["ticker"], "raw_inputs": inputs.pop("raw_inputs"),
                          "sources": inputs["sources"]}
                source_id = digest(shared)
                self.execute("INSERT INTO ft_inputs VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
                             (source_id, now.isoformat(), json_text(shared)))
                inputs["source_inputs_id"] = source_id
            input_id = digest(inputs)
            self.execute("INSERT INTO ft_inputs VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
                         (input_id, now.isoformat(), json_text(inputs)))
            for f in forecasts:
                fid = digest([slot["slot_id"], f["tier"]])
                self.execute("INSERT INTO ft_forecasts VALUES (?, ?, ?, ?, ?, ?, ?)",
                             (fid, slot["slot_id"], f["tier"], f["expiration"],
                              f["available_at"], input_id, json_text(f)))
            self.execute("UPDATE ft_slots SET status='captured', error=NULL, attempts=attempts+1, updated_at=? WHERE slot_id=?",
                         (now.isoformat(), slot["slot_id"]))
            # Large snapshots or a slow database must not finish their writes
            # after the deadline merely because the initial row lock was timely.
            # Reject the whole transaction, including its shared inputs.
            if not self.sqlite:
                db_finished = self.admission_clock() if self.admission_clock else self.execute("SELECT clock_timestamp()").fetchone()[0]
                if not week.admits(db_finished):
                    raise ValueError("Database admission time outside capture window after writes")
        return True

    def append_observation(self, study_id, week, ticker, session, payload, now):
        with self.transaction():
            if not self.sqlite:
                key = int(digest([study_id, week, ticker, session])[:15], 16)
                self.execute("SELECT pg_advisory_xact_lock(?)", (key,))
            old = self.execute("SELECT observation_id, payload_json, revision FROM ft_observations "
                               "WHERE study_id=? AND week_start=? AND ticker=? AND session_date=? "
                               "ORDER BY revision DESC LIMIT 1", (study_id, week, ticker, session)).fetchone()
            content = json_text(payload)
            if old and old[1] == content:
                return old[0]
            revision = old[2] + 1 if old else 1
            oid = digest([study_id, week, ticker, session, revision, payload])
            self.execute("INSERT INTO ft_observations VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                         (oid, study_id, week, ticker, session, now.isoformat(), content, revision))
        return oid

    def observations(self, study_id, week, ticker):
        rows = self.query("SELECT * FROM ft_observations WHERE study_id=? AND week_start=? AND ticker=? "
                          "ORDER BY revision", (study_id, week, ticker))
        latest = {}
        for row in rows:
            latest[row["session_date"]] = {**json.loads(row["payload_json"]),
                                          "observation_id": row["observation_id"], "collected_at": row["collected_at"]}
        return latest

    def observation_checks(self, study_id, week, ticker):
        rows = self.query("SELECT * FROM ft_observation_checks WHERE study_id=? AND week_start=? AND ticker=?",
                          (study_id, week, ticker))
        return {r['session_date']: r for r in rows}

    def checked_observation(self, study_id, week, ticker, session, now, status, error=None):
        with self.transaction():
            self.execute("INSERT INTO ft_observation_checks VALUES (?, ?, ?, ?, ?, ?, ?) "
                         "ON CONFLICT(study_id, week_start, ticker, session_date) DO UPDATE SET "
                         "checked_at=excluded.checked_at, status=excluded.status, error=excluded.error",
                         (study_id, week, ticker, session, now.isoformat(), status, error))

    def forecasts(self, study_id, week=None, *, active_since=None):
        sql = "SELECT f.*, s.study_id, s.week_start, s.ticker, s.model, s.model_version, s.cohort FROM ft_forecasts f JOIN ft_slots s ON s.slot_id=f.slot_id WHERE s.study_id=?"
        if active_since:
            sql += " AND (s.week_start>=? OR NOT EXISTS (SELECT 1 FROM ft_scores c WHERE c.forecast_id=f.forecast_id AND c.finalized=1 AND c.revision=(SELECT MAX(c2.revision) FROM ft_scores c2 WHERE c2.forecast_id=f.forecast_id)))"
            return self.query(sql, (study_id, active_since))
        return self.query(sql + (" AND s.week_start=?" if week else ""), (study_id, week) if week else (study_id,))

    def score_heads(self, forecast_ids):
        if not forecast_ids:
            return {}
        marks = ','.join('?' for _ in forecast_ids)
        rows = self.query("SELECT * FROM ft_scores c WHERE forecast_id IN (" + marks + ") AND revision="
                          "(SELECT MAX(revision) FROM ft_scores c2 WHERE c2.forecast_id=c.forecast_id)", tuple(forecast_ids))
        return {r['forecast_id']: r for r in rows}

    def append_score(self, fid, result, version, now):
        with self.transaction():
            if not self.sqlite:
                self.execute("SELECT forecast_id FROM ft_forecasts WHERE forecast_id=? FOR UPDATE", (fid,))
            old = self.execute("SELECT payload_json, scorer_version, revision FROM ft_scores WHERE forecast_id=? "
                               "ORDER BY revision DESC LIMIT 1", (fid,)).fetchone()
            content = json_text(result)
            if old and old[:2] == (content, version):
                return False
            revision = old[2] + 1 if old else 1
            sid = digest([fid, version, revision, result])
            finalized = int(result.get('status') == 'final' or result.get('reconciliation_status') == 'exhausted_after_14_days')
            self.execute("INSERT INTO ft_scores VALUES (?, ?, ?, ?, ?, ?, ?)",
                         (sid, fid, version, now.isoformat(), content, revision, finalized))
        return True

    def run_record(self, run_id, study_id, started, finished, status, details):
        with self.transaction():
            self.execute("INSERT INTO ft_runs VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(run_id) DO UPDATE SET "
                         "finished_at=excluded.finished_at, status=excluded.status, payload_json=excluded.payload_json",
                         (run_id, study_id, started.isoformat(), finished.isoformat() if finished else None,
                          status, json_text(details)))
