-- Explicit, additive migration. Never called by Streamlit/init_all_tables.
CREATE TABLE IF NOT EXISTS ft_studies (
    study_id TEXT PRIMARY KEY, start_week TEXT NOT NULL,
    created_at TEXT NOT NULL, config_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ft_slots (
    slot_id TEXT PRIMARY KEY, study_id TEXT NOT NULL REFERENCES ft_studies(study_id),
    week_start TEXT NOT NULL, ticker TEXT NOT NULL, model TEXT NOT NULL,
    model_version TEXT NOT NULL, cohort TEXT NOT NULL,
    status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT, updated_at TEXT NOT NULL,
    UNIQUE(study_id, week_start, ticker, model)
);
CREATE TABLE IF NOT EXISTS ft_inputs (
    input_id TEXT PRIMARY KEY, captured_at TEXT NOT NULL, payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ft_forecasts (
    forecast_id TEXT PRIMARY KEY, slot_id TEXT NOT NULL REFERENCES ft_slots(slot_id),
    tier TEXT NOT NULL, expiration TEXT NOT NULL, available_at TEXT NOT NULL,
    input_id TEXT NOT NULL REFERENCES ft_inputs(input_id), payload_json TEXT NOT NULL,
    UNIQUE(slot_id, tier)
);
CREATE TABLE IF NOT EXISTS ft_observations (
    observation_id TEXT PRIMARY KEY, study_id TEXT NOT NULL REFERENCES ft_studies(study_id),
    week_start TEXT NOT NULL, ticker TEXT NOT NULL, session_date TEXT NOT NULL,
    collected_at TEXT NOT NULL, payload_json TEXT NOT NULL, revision INTEGER NOT NULL,
    UNIQUE(study_id, week_start, ticker, session_date, revision)
);
CREATE INDEX IF NOT EXISTS ft_observations_lookup ON ft_observations(study_id, week_start, ticker, session_date, collected_at);
-- Mutable check metadata prevents repeated unchanged vendor responses from
-- forcing another fetch on every retry. Evidence itself remains append-only.
CREATE TABLE IF NOT EXISTS ft_observation_checks (
    study_id TEXT NOT NULL REFERENCES ft_studies(study_id), week_start TEXT NOT NULL,
    ticker TEXT NOT NULL, session_date TEXT NOT NULL, checked_at TEXT NOT NULL,
    status TEXT NOT NULL, error TEXT,
    PRIMARY KEY(study_id, week_start, ticker, session_date)
);
CREATE TABLE IF NOT EXISTS ft_scores (
    score_id TEXT PRIMARY KEY, forecast_id TEXT NOT NULL REFERENCES ft_forecasts(forecast_id),
    scorer_version TEXT NOT NULL, scored_at TEXT NOT NULL, payload_json TEXT NOT NULL, revision INTEGER NOT NULL,
    finalized INTEGER NOT NULL DEFAULT 0,
    UNIQUE(forecast_id, revision)
);
CREATE INDEX IF NOT EXISTS ft_scores_lookup ON ft_scores(forecast_id, scored_at);
CREATE TABLE IF NOT EXISTS ft_runs (
    run_id TEXT PRIMARY KEY, study_id TEXT NOT NULL REFERENCES ft_studies(study_id),
    started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL, payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ft_runs_latest ON ft_runs(started_at DESC);
CREATE INDEX IF NOT EXISTS ft_runs_health ON ft_runs(status, started_at DESC);
