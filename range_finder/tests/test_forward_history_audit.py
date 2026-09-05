"""Recorded vendor defects remain evidence, never repaired production bars."""
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path

import pandas as pd
import pytest

from range_finder.feature_builder import compute_hv_windows
from range_finder.forward_test.provider import valid_ohlc
from scripts.analyze_forward_history import (analyze, compare, coverage, dividend_comparison,
    proposed_compatibility, records_frame, yahoo_frame)

EVIDENCE = json.loads((Path(__file__).parent/'fixtures/forward_history_defects.json').read_text())


def pair(ticker):
    item = EVIDENCE['tickers'][ticker]
    return records_frame(item['tradier']), records_frame(item['yahoo_quote'])


def test_all_eight_spy_invalid_closes_are_cash_dividend_discrepancies_not_repairs():
    tradier, yahoo = pair('SPY')
    invalid = [day for day, row in tradier.iterrows() if not valid_ohlc(row.to_dict())]
    assert len(invalid) == 8
    dividends = pd.read_csv(StringIO(EVIDENCE['spy_dividends_csv']))
    # These eight ex-dates need not be in the narrow regression excerpt.
    # Use their recorded preceding sessions explicitly, without manufacturing bars.
    for day in invalid:
        following = dividends[pd.to_datetime(dividends.ex_dividend_date) > day].iloc[0]
        assert abs(yahoo.at[day, 'close']-tradier.at[day, 'close']-
                   following.split_adjusted_cash_amount) < .0051
    before = tradier.copy(deep=True)
    result = compare(tradier, yahoo, ('close',))
    assert result['status'] == 'CONTRADICTORY'
    assert set(str(d.date()) for d in invalid) <= {r['date'] for r in result['differences']}
    pd.testing.assert_frame_equal(before, tradier)


def test_spx_independently_matching_close_does_not_fabricate_missing_ohlc():
    tradier, yahoo = pair('SPX')
    day = pd.Timestamp('2026-03-19')
    # FRED SP500 also returned 6606.49; this corroborates only the consumed close.
    assert tradier.at[day, 'close'] == 6606.49
    assert abs(yahoo.at[day, 'close']-6606.49) < .0051
    assert not valid_ohlc(tradier.loc[day].to_dict())
    assert compare(tradier.loc[[day]], yahoo.loc[[day]])['status'] == 'CONTRADICTORY'


@pytest.mark.parametrize('ticker', ['SPX','AAPL','AMD'])
def test_internally_valid_sources_can_disagree_on_consumed_prices(ticker):
    tradier, yahoo = pair(ticker)
    day = pd.Timestamp('2026-08-31' if ticker == 'SPX' else '2023-09-11')
    assert valid_ohlc(tradier.loc[day].to_dict())
    assert valid_ohlc(yahoo.loc[day].to_dict())
    result = compare(tradier.loc[[day]], yahoo.loc[[day]])
    assert result['status'] == 'CONTRADICTORY'
    assert any(d['field'] == 'open' for d in result['differences'])


def test_actual_spy_close_difference_changes_hv_but_unused_ohl_does_not():
    tradier, yahoo = pair('SPY')
    tradier, yahoo = tradier.loc['2026-05-18':], yahoo.loc['2026-05-18':]
    th = compute_hv_windows(tradier[['close']])
    yh = compute_hv_windows(yahoo[['close']])
    day = pd.Timestamp('2026-06-15')
    assert abs(th.at[day,'hv5']/th.at[day,'hv20'] - yh.at[day,'hv5']/yh.at[day,'hv20']) > .001
    changed = tradier.copy()
    changed[['open','high','low']] = 99999
    pd.testing.assert_frame_equal(th, compute_hv_windows(changed[['close']]))


def test_unavailable_reference_and_adjustment_mismatch_cannot_certify_a_fallback():
    tradier, _ = pair('SPY')
    assert compare(tradier, None)['status'] == 'UNAVAILABLE_REFERENCE'
    candidate = {'instrument':'SPY','session':'regular','adjustment':'split_only','share_basis':'current'}
    assert proposed_compatibility(candidate, None) == 'UNAVAILABLE_REFERENCE'
    assert proposed_compatibility(candidate, {**candidate, 'adjustment':'split_and_dividend'}) == 'UNAVAILABLE_INCOMPATIBLE_ADJUSTMENT'
    assert proposed_compatibility(candidate, {**candidate, 'session':'extended'}) == 'UNAVAILABLE_INCOMPATIBLE_SESSION'
    assert proposed_compatibility(candidate, {**candidate, 'share_basis':None}) == 'UNAVAILABLE_CONVENTION_UNVERIFIED'
    assert proposed_compatibility(candidate, candidate) == 'CONVENTIONS_MATCH_NOT_PRICE_CERTIFICATION'


def test_future_missing_and_duplicate_sessions_are_reported_without_dropping_rows():
    tradier, _ = pair('SPY')
    frame = tradier.loc['2026-06-22':'2026-06-26'].copy()
    assert coverage(frame, '2026-06-22', '2026-06-26')['integrity_and_coverage_pass']
    missing = frame.drop(pd.Timestamp('2026-06-24'))
    assert coverage(missing, '2026-06-22', '2026-06-26')['missing'] == ['2026-06-24']
    extra = pd.concat([frame, tradier.loc[['2026-06-29']]])
    assert coverage(extra, '2026-06-22', '2026-06-26')['extra'] == ['2026-06-29']
    assert len(extra) == 6
    duplicate = pd.concat([frame, frame.iloc[[-1]]])
    assert coverage(duplicate, '2026-06-22', '2026-06-26')['duplicates']
    assert compare(duplicate, frame)['status'] == 'UNAVAILABLE_DUPLICATE_LABELS'


def test_yahoo_appended_quote_is_retained_and_adjclose_is_never_consumed():
    payload = {'chart':{'result':[{'timestamp': [1788177600,1788552000],
        'indicators':{'quote':[{'open':[100,101],'high':[103,104],
            'low':[99,100],'close':[102,103]}], 'adjclose':[{'adjclose':[1,2]}]}}]}}
    frame, _ = yahoo_frame(payload)
    assert frame.close.tolist() == [102,103]
    result = coverage(frame, '2026-08-31','2026-09-04','weekly')
    assert result['extra'] == ['2026-09-04']
    assert not result['integrity_and_coverage_pass']


def test_comparisons_preserve_original_prices_and_source_provenance():
    before = deepcopy(EVIDENCE)
    for ticker in EVIDENCE['tickers']:
        compare(*pair(ticker))
    assert EVIDENCE == before
    for metadata in EVIDENCE['sources'].values():
        assert len(metadata['raw_sha256']) == len(metadata['sanitized_sha256']) == 64
        assert metadata['retrieved_at'].endswith('+00:00')
        assert 'api_key' not in metadata['params']


def test_dividend_comparison_respects_holiday_previous_session():
    tradier, yahoo = pair('SPY')
    dividends = pd.read_csv(StringIO(EVIDENCE['spy_dividends_csv']))
    result = dividend_comparison(tradier, yahoo, dividends)
    item = next(r for r in result if r['ex_date'] == '2026-06-18')
    assert item['prior_session'] == '2026-06-17' and item['matches_dividend']


def test_changed_archived_body_is_rejected_before_comparison(tmp_path):
    (tmp_path/'summary.json').write_text(json.dumps({'required_start':'2020-08-08','end':'2026-09-04'}))
    (tmp_path/'manifest.json').write_text(json.dumps([{'file':'response.json','sanitized_sha256':'0'*64}]))
    (tmp_path/'response.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='Evidence hash mismatch: response.json'):
        analyze(tmp_path)


def test_missing_yahoo_archive_reports_unavailable_models_without_substitution(tmp_path, monkeypatch):
    from scripts import analyze_forward_history as module
    monkeypatch.setattr(module, 'UNIVERSE', ('SPY',))
    (tmp_path/'summary.json').write_text(json.dumps({'required_start':'2026-06-22','end':'2026-06-26'}))
    (tmp_path/'manifest.json').write_text('[]')
    rows = [r for r in EVIDENCE['tickers']['SPY']['tradier'] if '2026-06-22' <= r['date'] <= '2026-06-26']
    for cadence in ('daily','weekly'):
        (tmp_path/f'tradier-SPY-{cadence}.json').write_text(json.dumps({'history':{'day':rows}}))
    result = analyze(tmp_path)
    assert set(result['tickers']['SPY']['models'].values()) == {'UNAVAILABLE_REFERENCE'}
    assert result['tickers']['SPY']['daily_tradier']['integrity_and_coverage_pass']
    assert result['accepted_price_replacements'] == 0
