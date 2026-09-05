"""Single unattended invocation: capture if admissible, reconcile, then score."""
from datetime import date, datetime, timedelta
import json
import os
from uuid import uuid4

from range_finder.trading_week import NY, trading_week
from .capture import capture_model
from .config import (DATA_READY_MINUTES, MAX_CAPTURE_ATTEMPTS, RECONCILE_DAYS,
                     SCORER_VERSION, UNIVERSE, methodology)
from .scoring import score_forecast
from .provider import valid_ohlc


def run_study(store, provider, study_id, *, clock, model_version=None):
    started = clock()
    studies = store.query("SELECT * FROM ft_studies WHERE study_id=?", (study_id,))
    if not studies:
        raise ValueError("Study not registered; run the explicit activation command")
    run_id = uuid4().hex
    details = {"captured_models": 0, "observations_checked": 0, "scores_checked": 0, "errors": []}
    details['scheduler'] = {"run_id": os.environ.get('GITHUB_RUN_ID'), "run_attempt": os.environ.get('GITHUB_RUN_ATTEMPT'),
                            "commit": os.environ.get('GITHUB_SHA'), "event": os.environ.get('GITHUB_EVENT_NAME', 'local CLI')}
    store.run_record(run_id, study_id, started, None, "running", details)
    try:
        from .config import COHORT
        cohort = json.loads(studies[0]['config_json']).get('cohort', COHORT)
        version = model_version or methodology()[0]
        current = trading_week(started.astimezone(NY).date())
        week_date = date.fromisoformat(studies[0]["start_week"])
        if week_date.weekday() != 0:
            raise ValueError("Study start_week must be a Monday label")
        # Reconcile scheduler outages with explicit missed slots, never with
        # reconstructed historical forecasts. No historical market API calls.
        known_weeks = {r['week_start'] for r in store.query("SELECT DISTINCT week_start FROM ft_slots WHERE study_id=?", (study_id,))}
        unfinished = {r['week_start'] for r in store.query("SELECT DISTINCT week_start FROM ft_slots WHERE study_id=? AND status NOT IN ('captured','missed')", (study_id,))}
        while week_date <= current.monday:
            week = trading_week(week_date)
            ws = str(week_date)
            if ws in known_weeks and ws not in unfinished:
                week_date += timedelta(days=7)
                continue
            if ws not in known_weeks:
                store.seed_week(study_id, ws, version if week_date == current.monday else "not_captured", clock(), cohort=cohort)
            slots = store.slots(study_id, ws)
            if clock() >= week.capture_end:
                for slot in slots:
                    if slot["status"] not in ("captured", "missed"):
                        reason = "Capture window missed; no retrospective admission"
                        if slot["error"]:
                            reason += ": " + slot["error"]
                        store.failure(slot, "missed", reason, clock())
                        details["errors"].append(f"{slot['ticker']}/{slot['model']}/{ws}: {reason}")
            elif week.admits(clock()):
                for ticker in UNIVERSE:
                    pending = [s for s in slots if s["ticker"] == ticker and s["status"] != "captured"
                               and s["attempts"] < MAX_CAPTURE_ATTEMPTS]
                    if not pending:
                        continue
                    try:
                        prepared = provider.prepare(ticker, week)
                    except Exception as exc:
                        for slot in pending:
                            store.failure(slot, "unavailable", str(exc), clock())
                        details["errors"].append(f"{ticker}: {exc}")
                        continue
                    for slot in pending:
                        try:
                            if slot["model_version"] != version:
                                raise ValueError("Methodology changed during capture week; existing slots retain their version")
                            inputs, forecasts = capture_model(prepared, slot["model"], week, version, clock())
                            # Availability is when computation actually finishes,
                            # never the opening anchor or invocation start time.
                            ready = clock()
                            for f in forecasts:
                                f["available_at"] = ready.isoformat()
                                f["captured_at"] = ready.isoformat()
                            if store.freeze(slot, inputs, forecasts, ready, week):
                                details["captured_models"] += 1
                        except Exception as exc:
                            store.failure(slot, "unavailable", str(exc), clock())
                            details["errors"].append(f"{ticker}/{slot['model']}: {exc}")
            week_date += timedelta(days=7)

        active_since = str(started.astimezone(NY).date() - timedelta(days=RECONCILE_DAYS + 4))
        frozen = store.forecasts(study_id, active_since=active_since)
        score_heads = store.score_heads([f['forecast_id'] for f in frozen])
        groups = {}
        for f in frozen:
            groups.setdefault((f["week_start"], f["ticker"]), []).append(f)
        for (ws, ticker), forecasts in groups.items():
            week = trading_week(date.fromisoformat(ws))
            now = clock()
            # Price corrections have a bounded automatic reconciliation period.
            # Old incomplete rows remain visible; an explicit reconcile command
            # can be added after source repair without touching frozen forecasts.
            expired = now > week.evaluation_close + timedelta(days=RECONCILE_DAYS)
            observations = store.observations(study_id, ws, ticker)
            checks = store.observation_checks(study_id, ws, ticker)
            changed = False
            first_available = min(datetime.fromisoformat(f["available_at"]) for f in forecasts)
            for session in week.sessions:
                if expired:
                    break
                if now < session.close + timedelta(minutes=DATA_READY_MINUTES):
                    continue
                old = observations.get(str(session.day))
                checked = checks.get(str(session.day))
                if checked and datetime.fromisoformat(checked["checked_at"]).astimezone(NY).date() == now.astimezone(NY).date():
                    continue
                # Re-fetch incomplete sessions daily; complete sessions get a
                # four-calendar-day correction window, then stay cold. Final-day
                # settlement remains independently eligible for reconciliation.
                if old and valid_ohlc(old.get("daily")) and session.day != first_available.astimezone(NY).date():
                    age = (now.astimezone(NY).date() - session.day).days
                    if age > 4 and (ticker != "SPX" or session != week.sessions[-1]):
                        continue
                try:
                    payload = provider.observe(ticker, session,
                        first_available if session.day == first_available.astimezone(NY).date() else None)
                    if (payload.get("ticker") != ticker or payload.get("session") != str(session.day)
                            or not valid_ohlc(payload.get("daily"))
                            or payload["daily"].get("date") != str(session.day)):
                        raise ValueError("Finalized daily bar missing/invalid; retained prior evidence, will retry")
                    oid = store.append_observation(study_id, ws, ticker, str(session.day), payload, clock())
                    changed |= not old or old.get('observation_id') != oid
                    store.checked_observation(study_id, ws, ticker, str(session.day), clock(), "ok")
                    details["observations_checked"] += 1
                    if payload.get("minute_error"):
                        details["errors"].append(f"{ticker}/{session.day}: {payload['minute_error']}")
                except Exception as exc:
                    store.checked_observation(study_id, ws, ticker, str(session.day), clock(), "unavailable", str(exc))
                    details["errors"].append(f"{ticker}/{session.day}: {exc}")
            observations = store.observations(study_id, ws, ticker)
            for f in forecasts:
                prior = score_heads.get(f['forecast_id'])
                if prior and not changed and prior['scorer_version'] == SCORER_VERSION:
                    previous = json.loads(prior['payload_json'])
                    due_transition = previous['status'] == 'pending' and now >= week.evaluation_close + timedelta(minutes=DATA_READY_MINUTES)
                    expired_transition = expired and not prior['finalized']
                    if not due_transition and not expired_transition:
                        continue
                score = score_forecast(json.loads(f["payload_json"]), week, observations, clock())
                if expired and score["status"] != "final":
                    score["reconciliation_status"] = "exhausted_after_14_days"
                store.append_score(f["forecast_id"], score, SCORER_VERSION, clock())
                details["scores_checked"] += 1
        status = "partial" if details["errors"] else "ok"
        store.run_record(run_id, study_id, started, clock(), status, details)
        return details
    except BaseException as exc:
        details["errors"].append(f"{type(exc).__name__}: {exc}")
        store.run_record(run_id, study_id, started, clock(), "failed", details)
        raise
