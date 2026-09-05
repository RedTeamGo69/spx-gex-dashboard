"""Reproduce cross-source history discrepancies from immutable local evidence.

This is a release diagnostic, not a price adapter or a forecast admission gate.
Agreement does not establish correctness or interchangeable session conventions.
No rows or prices are repaired, selected, imported, or passed to a model.
"""
import argparse
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from range_finder.forward_test.config import MODELS, UNIVERSE
from range_finder.forward_test.provider import valid_ohlc

FIELDS = ('open', 'high', 'low', 'close')
# Diagnostic display alignment only: half a cent plus float32 representation.
# This is NOT an admission tolerance and never changes an input price.
ALIGNMENT_TOLERANCE = 0.0051


def records_frame(rows):
    if not rows:
        return pd.DataFrame(columns=FIELDS, index=pd.DatetimeIndex([]))
    frame = pd.DataFrame(rows).set_index('date')
    frame.index = pd.to_datetime(frame.index)
    return frame.apply(pd.to_numeric, errors='coerce').sort_index()


def yahoo_frame(payload):
    result = payload['chart']['result'][0]
    dates = pd.to_datetime(result['timestamp'], unit='s', utc=True)
    dates = dates.tz_convert('America/New_York').tz_localize(None).normalize()
    # Do not use adjclose, auto_adjust, repair, or silently remove appended bars.
    return pd.DataFrame(result['indicators']['quote'][0], index=dates), result


def finite(value):
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def compare(left, right, fields=FIELDS):
    """Report every discrepancy and both original values; never pick a winner."""
    if right is None or right.empty:
        return {'status': 'UNAVAILABLE_REFERENCE', 'differences': [],
                'left_only': [str(d.date()) for d in left.index], 'right_only': []}
    if left.index.has_duplicates or right.index.has_duplicates:
        return {'status': 'UNAVAILABLE_DUPLICATE_LABELS', 'differences': []}
    overlap = left.index.intersection(right.index)
    differences = []
    for day in overlap:
        for field in fields:
            a, b = finite(left.at[day, field]), finite(right.at[day, field])
            if a is None or b is None or abs(a-b) > ALIGNMENT_TOLERANCE:
                differences.append({'date': str(day.date()), 'field': field,
                                    'left': a, 'right': b,
                                    'delta_left_minus_right': None if a is None or b is None else a-b})
    left_only = [str(d.date()) for d in left.index.difference(right.index)]
    right_only = [str(d.date()) for d in right.index.difference(left.index)]
    return {'status': 'CONTRADICTORY' if differences else
            'INCOMPLETE_OVERLAP' if left_only or right_only else 'NUMERICALLY_ALIGNED_NOT_CERTIFIED',
            'overlap_rows': len(overlap), 'differing_rows': len({r['date'] for r in differences}),
            'differences': differences, 'left_only': left_only, 'right_only': right_only}


def coverage(frame, start, end, cadence='daily'):
    sessions = mcal.get_calendar('NYSE').valid_days(start_date=start, end_date=end).tz_localize(None)
    expected = sessions if cadence == 'daily' else pd.DatetimeIndex(
        (sessions-pd.to_timedelta(sessions.weekday, unit='D')).unique())
    missing = [str(d.date()) for d in expected.difference(frame.index)]
    extra = [str(d.date()) for d in frame.index.difference(expected)]
    invalid = [str(d.date()) for d, row in frame.iterrows() if not valid_ohlc(row.to_dict())]
    return {'rows': len(frame), 'expected_rows': len(expected), 'missing': missing,
            'extra': extra, 'duplicates': bool(frame.index.has_duplicates), 'invalid_ohlc': invalid,
            'integrity_and_coverage_pass': not (missing or extra or invalid or frame.index.has_duplicates)}


def proposed_compatibility(candidate, reference):
    """A matching number cannot override unknown/incompatible source semantics."""
    if not reference:
        return 'UNAVAILABLE_REFERENCE'
    for key in ('instrument', 'session', 'adjustment', 'share_basis'):
        if not candidate.get(key) or not reference.get(key):
            return 'UNAVAILABLE_CONVENTION_UNVERIFIED'
        if candidate[key] != reference[key]:
            return 'UNAVAILABLE_INCOMPATIBLE_' + key.upper()
    return 'CONVENTIONS_MATCH_NOT_PRICE_CERTIFICATION'


def weekly_from_daily(frame):
    # For comparison only; coverage is reported separately. In particular,
    # pandas' missing-value aggregation must never masquerade as repaired data.
    return frame.resample('W-MON', closed='left', label='left').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})


def dividend_comparison(tradier, yahoo, dividends):
    evidence = []
    for item in dividends.to_dict('records'):
        exdate = pd.Timestamp(item['ex_dividend_date'])
        dates = tradier.index[tradier.index < exdate]
        if not len(dates) or exdate not in tradier.index:
            continue
        day = dates[-1]
        if day not in yahoo.index:
            continue
        amount = float(item['split_adjusted_cash_amount'])
        delta = float(yahoo.at[day, 'close']-tradier.at[day, 'close'])
        evidence.append({'prior_session': str(day.date()), 'ex_date': str(exdate.date()),
                         'cash_dividend_in_current_share_basis': amount,
                         'tradier_close': float(tradier.at[day, 'close']),
                         'yahoo_quote_close': float(yahoo.at[day, 'close']),
                         'difference': delta, 'matches_dividend': abs(delta-amount) <= ALIGNMENT_TOLERANCE})
    return evidence


def analyze(root):
    def read(name):
        return json.loads((root/name).read_text(encoding='utf-8'))
    summary = read('summary.json')
    start, end = pd.Timestamp(summary['required_start']), pd.Timestamp(summary['end'])
    manifest = read('manifest.json')
    for entry in manifest:
        if sha256((root/entry['file']).read_bytes()).hexdigest() != entry['sanitized_sha256']:
            raise ValueError('Evidence hash mismatch: '+entry['file'])
    report = {'mode': 'read_only_evidence_analysis_no_forecasts',
              'required_start': str(start.date()), 'last_completed_session': str(end.date()),
              'diagnostic_alignment_tolerance': ALIGNMENT_TOLERANCE,
              'admission_policy_changed': False, 'accepted_price_replacements': 0,
              'release_status': 'UNAVAILABLE_SOURCE_POLICY_NOT_CERTIFIED',
              'manifest': manifest, 'tickers': {},
              'files': {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(root.iterdir()) if p.is_file()}}
    for ticker in UNIVERSE:
        daily_raw = read(f'tradier-{ticker}-daily.json')['history']['day']
        weekly_raw = read(f'tradier-{ticker}-weekly.json')['history']['day']
        daily, weekly = records_frame(daily_raw), records_frame(weekly_raw)
        try:
            yahoo, yresult = yahoo_frame(read(f'yahoo-{ticker}-1d.json'))
            yw, _ = yahoo_frame(read(f'yahoo-{ticker}-1wk.json'))
        except (OSError, KeyError, IndexError, TypeError, ValueError) as exc:
            report['tickers'][ticker] = {
                'models': {model: 'UNAVAILABLE_REFERENCE' for model in MODELS},
                'reference_error': type(exc).__name__,
                'daily_tradier': coverage(daily, start, end),
                'accepted_price_replacements': 0}
            continue
        legacy = records_frame(read(f'legacy-{ticker}.json'))
        legacy = legacy.rename(columns={f'spx_{k}': k for k in FIELDS})
        item = report['tickers'][ticker] = {
            'models': {model: 'UNAVAILABLE_SOURCE_POLICY_NOT_CERTIFIED' for model in MODELS},
            'yahoo_identity': {k: yresult['meta'].get(k) for k in
                               ('symbol', 'instrumentType', 'exchangeName', 'currency', 'exchangeTimezoneName')},
            'daily_tradier': coverage(daily, start, end),
            'tradier_to_yahoo_compatibility': proposed_compatibility(
                {'instrument': ticker, 'session': None, 'adjustment': None, 'share_basis': None},
                {'instrument': ticker, 'session': 'regular', 'adjustment': 'split_only',
                 'share_basis': 'current'}),
            'daily_yahoo_required': coverage(yahoo.loc[start:end], start, end),
            'weekly_tradier_full': coverage(weekly, weekly.index.min(), end, 'weekly'),
            'weekly_yahoo_endpoint_full': coverage(yw, weekly.index.min(), end, 'weekly'),
            'daily_tradier_vs_yahoo': compare(daily, yahoo.loc[start:end]),
            'weekly_tradier_vs_yahoo_daily_aggregation': compare(weekly, weekly_from_daily(yahoo)),
            'weekly_yahoo_endpoint_vs_daily_aggregation': compare(yw, weekly_from_daily(yahoo)),
            'weekly_tradier_vs_own_daily_aggregation_required': compare(
                weekly.loc[daily.index.min():], weekly_from_daily(daily)),
        }
        # These are the actual older rows prepare() prepends, affecting HAR
        # warm-up and pooled side share. Do not shorten them to the fit window.
        older = legacy.loc[legacy.index < weekly.loc[start:].index.min(), list(FIELDS)]
        item['consumed_legacy_older'] = {'rows': len(older),
            'integrity': coverage(older, older.index.min(), older.index.max()+pd.Timedelta(days=4), 'weekly')
            if not older.empty else None,
            'vs_fresh_tradier': compare(older, weekly.reindex(older.index)),
            'vs_yahoo_daily_aggregation': compare(older, weekly_from_daily(yahoo).reindex(older.index))}
        item['narrow_rechecks'] = []
        for raw in daily_raw:
            if not valid_ohlc(raw):
                path = root/f"tradier-{ticker}-single-{raw['date']}.json"
                narrow = read(path.name)['history']['day'] if path.exists() else None
                if isinstance(narrow, dict): narrow = [narrow]
                item['narrow_rechecks'].append({'original': raw, 'individual_response': narrow,
                    'identical': narrow == [raw]})
        if ticker == 'SPX':
            fred = pd.DataFrame(read('fred-SP500.json')['observations']).set_index('date')
            fred.index = pd.to_datetime(fred.index)
            fred = fred.rename(columns={'value': 'close'}).apply(pd.to_numeric, errors='coerce')
            # FRED explicitly marks non-observation dates '.', so align to the
            # exchange calendar and report absent sessions instead of ffill.
            fred = fred.reindex(daily.index)
            item['daily_tradier_close_vs_fred'] = compare(daily, fred, ('close',))
            item['daily_yahoo_close_vs_fred'] = compare(yahoo.loc[start:end], fred, ('close',))
        else:
            chunks = [root/f'massive-{ticker}-{offset}.csv' for offset in (0, 500, 1000)]
            if all(p.exists() for p in chunks):
                massive = pd.concat([pd.read_csv(p) for p in chunks])
                massive.index = pd.DatetimeIndex(pd.to_datetime(massive.t, unit='ms', utc=True)).tz_convert(
                    'America/New_York').tz_localize(None).normalize()
                massive = massive.rename(columns=dict(zip(('o','h','l','c'), FIELDS)))
                item['massive_vs_yahoo_full_required'] = compare(massive, yahoo.loc[start:end])
                item['massive_vs_tradier_full_required'] = compare(massive, daily)
                item['massive_coverage_full_required'] = coverage(massive, start, end)
                mw = weekly_from_daily(massive)
                item['massive_weekly_vs_tradier_overlap'] = compare(mw, weekly.reindex(mw.index))
                item['massive_session_convention'] = 'Qualifying aggregate trades; regular-session equivalence not certified'
        if ticker == 'SPY':
            dividends = pd.read_csv(StringIO(read('massive-spy-dividends-tool.json')['structuredContent']['result']))
            item['dividend_differences'] = dividend_comparison(daily, yahoo, dividends)
        elif ticker == 'AAPL' and (root/'massive-aapl_dividends.csv').exists():
            item['dividend_differences'] = dividend_comparison(daily, yahoo, pd.read_csv(root/'massive-aapl_dividends.csv'))
            item['split_events_yahoo'] = yresult.get('events', {}).get('splits', {})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.evidence_dir)
    # Do not overwrite earlier conclusions or mutate the raw evidence archive.
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(result, output, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('release_status', 'accepted_price_replacements')}))
    return 1  # Numeric agreement alone is never a release certificate.


if __name__ == '__main__':
    raise SystemExit(main())
