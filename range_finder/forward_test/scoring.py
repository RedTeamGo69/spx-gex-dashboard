"""Close success and path events are separate, with independent eligibility.

Minute highs/lows locate a breach only to an interval. The first minute may
straddle availability: its full-bar extremes can prove absence of a breach,
but an outside extreme alone cannot prove a prospective breach. A later
unambiguous observation can establish that side's event. Missing intervals
never establish absence. Whole-week daily extremes are labeled separately.
"""
from datetime import datetime, timedelta

from range_finder.trading_week import NY
from .config import DATA_READY_MINUTES
from .provider import valid_ohlc, finite_price


def score_forecast(forecast, week, observations, now):
    available = datetime.fromisoformat(forecast["available_at"])
    put, call = forecast["put_short"], forecast["call_short"]
    out = {"close_eligible": False, "path_eligible": False, "close_inside": None,
           "put_breach": None, "call_breach": None, "either_breach": None, "both_breached": None,
           "put_touch": None, "call_touch": None, "close_on_boundary": None,
           "earlier_close_outside": None, "returned_inside": None, "final_close": None,
           "observed_high": None, "observed_low": None, "whole_week_high": None, "whole_week_low": None,
           "first_breach_session": None, "first_breach_timestamp": None,
           "first_breach_interval": None, "first_breach_is_earliest_verified": False,
           "window_start": available.isoformat(), "window_end": week.evaluation_close.isoformat(),
           "classification": "Pending", "status": "pending", "missing_sessions": [],
           "path_limitations": [], "sources": [], "observation_ids": [],
           "settlement_status": "unverified", "settlement_value": None, "settlement_inside": None}
    if not (finite_price(put) and finite_price(call) and put < call and week.admits(available)
            and forecast["expiration"] == str(week.sessions[-1].day)):
        return {**out, "classification": "Invalid forecast data", "status": "invalid"}
    complete_window = now >= week.evaluation_close + timedelta(minutes=DATA_READY_MINUTES)
    path_complete = True
    evidence = {"put": [], "call": []}
    touches = {"put": False, "call": False}
    touch_known = {"put": True, "call": True}
    boundary_unknown = {"put": False, "call": False}
    path_bars, daily_bars = [], []
    earlier_closes, missing_earlier_close = [], False

    def add_bar(bar, day, start=None, end=None):
        path_bars.append(bar)
        for side, breached, touched in (("put", float(bar["low"]) < put, float(bar["low"]) <= put),
                                        ("call", float(bar["high"]) > call, float(bar["high"]) >= call)):
            touches[side] |= touched
            if breached:
                evidence[side].append({"session": str(day), "start": start, "end": end})

    for session in week.sessions:
        day = str(session.day)
        obs = observations.get(day)
        due = session.close + timedelta(minutes=DATA_READY_MINUTES)
        if session.close <= available:
            continue
        if not obs or obs.get("ticker") != forecast["ticker"] or obs.get("session") != day:
            path_complete = False
            out["missing_sessions"].append(day)
            if session != week.sessions[-1]:
                missing_earlier_close = True
            continue
        out["observation_ids"].append(obs.get("observation_id"))
        out["sources"].append(obs.get("source"))
        collected = datetime.fromisoformat(obs["collected_at"])
        if collected < due or collected > now:
            path_complete = False
            out["missing_sessions"].append(day)
            if session != week.sessions[-1]:
                missing_earlier_close = True
            continue
        bar = obs.get("daily")
        daily_valid = valid_ohlc(bar) and bar.get("date") == day
        if daily_valid:
            daily_bars.append(bar)
            close = float(bar["close"])
            if session == week.sessions[-1] and complete_window:
                out.update(close_eligible=True, final_close=close,
                           close_inside=put <= close <= call, close_on_boundary=close in (put, call))
                settlement = obs.get("settlement", {})
                if (settlement.get("status") == "verified" and finite_price(settlement.get("value"))
                        and settlement.get("expiration") == forecast["expiration"]
                        and settlement.get("root") == forecast["contract_root"] and settlement.get("source")):
                    value = float(settlement["value"])
                    out.update(settlement_status="verified", settlement_value=value,
                               settlement_inside=put <= value <= call, settlement_source=settlement["source"])
            elif session != week.sessions[-1]:
                earlier_closes.append(not (put <= close <= call))
        else:
            path_complete = False
            out["path_limitations"].append(f"{day}: missing/invalid daily OHLC")
            if session != week.sessions[-1]:
                missing_earlier_close = True
        if session.open >= available:
            if daily_valid:
                add_bar(bar, session.day)
            continue

        # Only the first partially observed session requires minute history.
        start = available.replace(second=0, microsecond=0)
        minute_map = {}
        duplicate_minutes = set()
        invalid_minutes = False
        for minute in obs.get("minutes", []):
            try:
                timestamp = datetime.fromisoformat(minute["start"])
                if timestamp.tzinfo is None:
                    invalid_minutes = True
                    continue
                if timestamp in minute_map or timestamp.second or timestamp.microsecond or not valid_ohlc(minute):
                    invalid_minutes = True
                if timestamp in minute_map:
                    duplicate_minutes.add(timestamp)
                minute_map[timestamp] = minute
            except (KeyError, TypeError, ValueError):
                invalid_minutes = True
        if invalid_minutes:
            path_complete = False
            out["path_limitations"].append(f"{day}: invalid or duplicate minute bars")
        for timestamp in duplicate_minutes:
            minute_map.pop(timestamp)
        cursor = start
        while cursor < session.close:
            minute = minute_map.get(cursor)
            end = cursor + timedelta(minutes=1)
            if not minute or not valid_ohlc(minute):
                path_complete = False
            elif cursor < available:
                # Superset bounds wholly inside imply no breach in the partial
                # minute. Outside/equality could be pre-publication, so do not
                # assign an event unless later in-window evidence establishes it.
                boundary_unknown["put"] |= float(minute["low"]) < put
                boundary_unknown["call"] |= float(minute["high"]) > call
                touch_known["put"] &= float(minute["low"]) > put
                touch_known["call"] &= float(minute["high"]) < call
                # Even the close is the last trade *somewhere* in the minute,
                # not a timestamped trade at its end. It may precede availability.
            else:
                add_bar(minute, session.day, cursor.isoformat(), end.isoformat())
            cursor = end
        if len([t for t in minute_map if start <= t < session.close]) != int((session.close - start).total_seconds() / 60):
            out["path_limitations"].append(f"{day}: minute coverage incomplete")

    if path_bars:
        out["observed_high"] = max(float(b["high"]) for b in path_bars)
        out["observed_low"] = min(float(b["low"]) for b in path_bars)
    out["observed_extremes_scope"] = "verified post-availability intervals; excludes partial-minute extremes"
    if daily_bars:
        out["whole_week_high"] = max(float(b["high"]) for b in daily_bars)
        out["whole_week_low"] = min(float(b["low"]) for b in daily_bars)
    out["whole_week_complete"] = len(daily_bars) == len(week.sessions)
    out["earlier_close_outside"] = True if any(earlier_closes) else (None if missing_earlier_close else False)
    for side in ("put", "call"):
        certain = bool(evidence[side])
        known_absent = path_complete and complete_window and not boundary_unknown[side]
        out[f"{side}_breach"] = True if certain else (False if known_absent else None)
        out[f"{side}_touch"] = True if touches[side] else (False if known_absent and touch_known[side] else None)
        if boundary_unknown[side] and not certain:
            out["path_limitations"].append(f"{side}: first minute straddles forecast availability")
    pb, cb = out["put_breach"], out["call_breach"]
    out["either_breach"] = True if pb is True or cb is True else (False if pb is False and cb is False else None)
    out["both_breached"] = False if pb is False or cb is False else (True if pb is True and cb is True else None)
    out["path_eligible"] = path_complete and complete_window and pb is not None and cb is not None
    events = evidence["put"] + evidence["call"]
    if events:
        first = min(events, key=lambda e: (e["session"], e["start"] or ""))
        out["first_breach_session"] = first["session"]
        out["first_breach_interval"] = {"start": first["start"], "end": first["end"]} if first["start"] else None
        out["first_breach_is_earliest_verified"] = out["path_eligible"] and not any(boundary_unknown.values())
        # Never manufacture an exact timestamp from an OHLC interval.
    if out["close_eligible"]:
        close = out["final_close"]
        if close < put:
            out["classification"] = "Closed below put strike"
        elif close > call:
            out["classification"] = "Closed above call strike"
        elif not out["path_eligible"]:
            out["classification"] = "Closed inside; incomplete path data"
        elif out["either_breach"]:
            side = "both-side" if out["both_breached"] else ("put-side" if pb else "call-side")
            out["classification"] = f"Closed inside after {side} breach"
        else:
            out["classification"] = "Closed inside, no strict breach"
        out["returned_inside"] = bool(out["close_inside"]) if out["either_breach"] is True else (False if out["either_breach"] is False else None)
        out["status"] = "final" if out["path_eligible"] else "incomplete"
    elif complete_window:
        out.update(classification="Incomplete or invalid final close data", status="incomplete")
    out["sources"] = sorted(set(filter(None, out["sources"])))
    return out
