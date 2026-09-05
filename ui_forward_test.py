"""Read-only Forward Test view. No quotes, model refits, or schema changes."""
import json
import os
from pathlib import Path
import subprocess

import pandas as pd
import streamlit as st

from range_finder.forward_test.results import load_results, metrics, scoreboard, build_workbook
from range_finder.forward_test.config import UNIVERSE, MODELS, COHORT


@st.cache_data(ttl=600, max_entries=4, show_spinner=False)
def load_snapshot():
    from range_finder.forward_test.store import Store
    url = os.environ.get("FORWARD_TEST_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not url:
        url = st.secrets.get("DATABASE_URL", "")
    store = Store.postgres(url)
    try:
        studies = store.query("SELECT study_id, start_week FROM ft_studies ORDER BY created_at DESC")
        records = [r for s in studies for r in load_results(store, s["study_id"])]
        runs = store.query("SELECT * FROM ft_runs ORDER BY started_at DESC LIMIT 30")
        last_ok = store.query("SELECT * FROM ft_runs WHERE status='ok' AND finished_at IS NOT NULL ORDER BY started_at DESC LIMIT 1")
        if last_ok and all(r['run_id'] != last_ok[0]['run_id'] for r in runs):
            runs.extend(last_ok)
        return records, runs, studies
    finally:
        store.close()


@st.cache_data(show_spinner=False)
def deployed_revision():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parent, text=True, timeout=5,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return 'Unavailable'


def render_forward_test(rows=None, runs=None, studies=None):
    st.title("Forward Test")
    st.caption("Frozen weekly predictions for SPX, SPY, AAPL and AMD. Observational results; no trades are submitted.")
    if rows is None:
        try:
            rows, runs, studies = load_snapshot()
        except Exception as exc:
            st.info("Forward Test is not available yet. Apply the additive migrations and register the study using the activation instructions.")
            st.caption(f"Database check: {type(exc).__name__}. If already activated, check the database connection and the Weekly Spread Finder Forward Test workflow.")
            return
    runs = runs or []
    studies = studies or []
    if st.button("Refresh results"):
        load_snapshot.clear()
        st.rerun()
    if not rows:
        st.info("No frozen predictions yet. Registered studies are waiting for an eligible opening-week capture.")
    for study in studies:
        st.caption(f"Registered study: {study['study_id']} · First eligible week: {study['start_week']}")
    filtered = list(rows)
    keys = [("study_id", "Study"), ("cohort", "Capture cohort"), ("ticker", "Ticker"),
            ("model", "Model"), ("tier_label", "Tier"), ("model_version", "Model version"), ("week_start", "Week")]
    for i in range(0, len(keys), 3):
        for column, (key, label) in zip(st.columns(min(3, len(keys) - i)), keys[i:i + 3]):
            with column:
                options = {str(r[key]) for r in rows if r.get(key) is not None}
                # Registration is visible before the first capture without
                # inventing pending forecasts or seeding synthetic slots.
                defaults = {'study_id': [s['study_id'] for s in studies],
                            'week_start': [s['start_week'] for s in studies],
                            'ticker': UNIVERSE, 'model': MODELS, 'cohort': [COHORT],
                            'tier_label': ['Lower PI', 'Point Estimate', '80% PI Upper', 'Effective (+buffer)']}
                options = sorted(options.union(defaults.get(key, [])))
                selected = st.multiselect(label, options, key=f"ft_filter_{key}")
                if selected:
                    filtered = [r for r in filtered if str(r.get(key)) in selected]
    summary = metrics(filtered)
    def pct(value):
        return "—" if value is None else f"{value:.1%}"
    columns = st.columns(4)
    for col, label, value, count in zip(columns,
            ("Weekly close hit rate", "Breach rate", "Breach recovery rate", "Average range / reference"),
            (summary["close_hit_rate"], summary["breach_rate"], summary["recovery_rate"], summary["average_width_ratio"]),
            (summary["close_n"], summary["path_n"], summary["recovery_n"], summary["width_n"])):
        with col:
            st.metric(label, pct(value))
            st.caption(f"Eligible tier records: {count}")
    st.caption(f"Put-side close failures: {summary['put_failures']} · Call-side close failures: {summary['call_failures']} · "
               f"Pending: {summary['pending']} · Missed: {summary['missed']} · "
               f"Unavailable: {summary['unavailable']} · Incomplete/invalid: {summary['incomplete']}")
    if len({r.get("model_version") for r in filtered if r.get("forecast_id")}) > 1:
        st.caption("Overview includes multiple model versions. The comparison table keeps versions and cohorts separate.")
    st.caption("Close statistics require a valid final close. Breach statistics require complete prospective coverage. "
               "Recovery rate counts inside closes among records with a verified breach and valid close/path data.")
    st.subheader("Model comparison")
    if filtered:
        comparison = pd.DataFrame(scoreboard(filtered))
        comparison['tier'] = comparison['tier'].map(dict(zip(
            ('lower_pi', 'point', 'pi_upper', 'effective'),
            ('Lower PI', 'Point Estimate', '80% PI Upper', 'Effective (+buffer)'))))
        comparison_fields = ['ticker', 'model', 'tier', 'close_hit_rate', 'close_n', 'breach_rate', 'path_n',
                             'recovery_rate', 'recovery_n', 'average_width_ratio', 'put_failures', 'call_failures',
                             'pending', 'missed', 'unavailable', 'incomplete', 'cohort', 'model_version', 'study_id']
        st.dataframe(comparison.reindex(columns=comparison_fields), hide_index=True)
        fields = ["week_start", "ticker", "model", "tier_label", "put_short", "call_short", "final_close",
                  "classification", "put_breach", "call_breach", "earlier_close_outside", "returned_inside",
                  "close_on_boundary", "available_at", "expiration", "settlement_status", "error"]
        st.subheader("Predictions and results")
        st.dataframe(pd.DataFrame(filtered).reindex(columns=fields), hide_index=True)
    st.download_button("Download Excel results", build_workbook(filtered),
                       file_name="Gamma_Lens_Forward_Test.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.subheader("Capture and scoring health")
    st.caption(f"App revision: {deployed_revision()}")
    good = next((r for r in runs if r["status"] == "ok" and r.get("finished_at")), None)
    st.caption(f"Latest successful run: {good['finished_at'] if good else 'No successful run recorded'}")
    last_capture = max((r['available_at'] for r in rows if r.get('available_at')), default='None recorded')
    last_score = max((r['scored_at'] for r in rows if r.get('scored_at')), default='None recorded')
    st.caption(f"Latest frozen forecast: {last_capture} · Latest saved score: {last_score}")
    if runs:
        latest = runs[0]
        st.write(f"Latest run: {latest['status']} · {latest['started_at']}")
        if latest["status"] == "running":
            st.warning("The latest run has no completion record. Check the workflow for interruption or timeout.")
        details = json.loads(latest["payload_json"])
        for error in details.get("errors", [])[:12]:
            st.warning(error)
        with st.expander("Recent runs"):
            st.dataframe(pd.DataFrame(runs).drop(columns=["payload_json"], errors="ignore"), hide_index=True)
    st.caption("SPX uses SPXW PM-settled contracts. Stock/ETF close results are theoretical expiration proxies. "
               "Official settlement, when verified, is stored separately. A temporary breach is not a weekly-close failure.")
