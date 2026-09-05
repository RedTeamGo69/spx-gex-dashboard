"""Generate fixture-only end-to-end evidence in a new, local output directory.

Usage: python scripts/demo_forward_test.py --output-dir <empty-directory>
No environment credentials are read for database connections or market calls.
"""
import argparse
from datetime import date, timedelta
import json
import logging
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from range_finder.forward_test.results import load_results, metrics, build_workbook
from range_finder.forward_test.runner import run_study
from range_finder.forward_test.store import Store
from range_finder.tests.forward_fixtures import Clock, FixtureProvider
from range_finder.trading_week import trading_week


class DemoProvider(FixtureProvider):
    def observe(self, ticker, session, first_available=None):
        obs = super().observe(ticker, session, first_available)
        week = trading_week(session.day)
        ref = obs['daily']['close']
        if session == week.sessions[1] and ticker in ('SPY', 'AAPL'):
            obs['daily']['low'] = ref * .8
            if ticker == 'AAPL':
                obs['daily']['high'] = ref * 1.2
                obs['daily']['close'] = ref * .85
        if session == week.sessions[-1] and ticker == 'AMD':
            obs['daily']['close'] = ref * 1.2
            obs['daily']['high'] = ref * 1.22
        return obs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    database = output / 'fixture.sqlite'
    if database.exists():
        raise FileExistsError('Use a new output directory; fixture evidence is never overwritten')
    for name in ('range_finder.har_model', 'range_finder.spread_levels', 'range_finder.feature_builder'):
        logging.getLogger(name).setLevel(logging.WARNING)
    week = trading_week(date(2026, 9, 7))
    clock = Clock(week.capture_start + timedelta(seconds=17))
    store = Store(sqlite3.connect(database, isolation_level=None))
    try:
        store.migrate()
        store.register('DEMO_ONLY', str(week.monday), clock(), {'cohort':'deterministic_fixture'})
        provider = DemoProvider(clock)
        capture = run_study(store, provider, 'DEMO_ONLY', clock=clock, model_version='fixture-v1')
        frozen = store.forecasts('DEMO_ONLY')
        retry = run_study(store, provider, 'DEMO_ONLY', clock=clock, model_version='fixture-v2')
        assert store.forecasts('DEMO_ONLY') == frozen and retry['captured_models'] == 0
        clock.value = week.evaluation_close + timedelta(hours=3)
        scoring = run_study(store, provider, 'DEMO_ONLY', clock=clock, model_version='fixture-v1')
        rows = load_results(store, 'DEMO_ONLY')
        summary = {'cohort':'deterministic_fixture', 'database':'isolated SQLite',
                   'capture':capture, 'duplicate_trigger':retry, 'scoring':scoring,
                   'metrics':metrics(rows), 'forecast_rows':len(rows),
                   'classes':sorted({r['classification'] for r in rows}),
                   'inputs_bytes':store.query('SELECT SUM(LENGTH(payload_json)) AS n FROM ft_inputs')[0]['n'],
                   'observation_bytes':store.query('SELECT SUM(LENGTH(payload_json)) AS n FROM ft_observations')[0]['n']}
        (output / 'fixture-summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        (output / 'fixture-rows.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        (output / 'fixture-results.xlsx').write_bytes(build_workbook(rows))
        print(json.dumps(summary, indent=2))
    finally:
        store.close()


if __name__ == '__main__':
    main()
