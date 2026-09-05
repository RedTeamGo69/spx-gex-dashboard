"""Source identity and freshness gates exercised without network traffic."""
from datetime import date, timedelta

import pytest
import numpy as np
import pandas as pd

from range_finder.forward_test.provider import TradierProvider
from range_finder.forward_test.capture import capture_model, contract_chain
from range_finder.tests.forward_fixtures import Clock, prepared_fixture
from range_finder.trading_week import trading_week


@pytest.mark.parametrize('ticker,bar',[
    ('SPX',{'date':'2026-03-19','open':'NaN','high':'NaN','low':'NaN','close':6606.49,'volume':0}),
    ('SPY',{'date':'2026-06-17','open':751.29,'high':752.15,'low':739.22,'close':739.05648,'volume':85945227}),
])
def test_prepare_rejects_observed_vendor_history_defects(monkeypatch,ticker,bar):
    week=trading_week(date(2026,9,7)); clock=Clock(week.capture_start)
    provider=TradierProvider('fixture',clock=clock)
    valid={'date':'2026-08-31','open':100.,'high':105.,'low':95.,'close':102.,'volume':0}
    monkeypatch.setattr(provider,'history',lambda ticker,start,end,interval='daily':[valid] if interval=='weekly' else [bar])
    try:
        with pytest.raises(ValueError,match='Invalid '+ticker+' history OHLC'):
            provider.prepare(ticker,week)
    finally:provider.close()


def test_stale_and_mismatched_quotes_are_rejected():
    week=trading_week(date(2026,9,7))
    clock=Clock(week.capture_start)
    provider=TradierProvider('fixture-not-a-token',clock=clock)
    try:
        quote={'symbol':'SPY','open':600.,'last':601.,'trade_date':int(clock().timestamp()*1000)}
        provider.__dict__['quotes']={'SPY':quote}
        assert provider.checked_quote('SPY',week.sessions[0].day)[0]['open']==600.
        quote['trade_date']-=600_000
        with pytest.raises(ValueError,match='Stale'):
            provider.checked_quote('SPY',week.sessions[0].day)
        quote['symbol']='SPX'
        with pytest.raises(ValueError,match='Missing'):
            provider.checked_quote('SPY',week.sessions[0].day)
    finally:provider.close()


def test_declared_delay_is_recorded_and_bounded():
    week=trading_week(date(2026,9,7)); clock=Clock(week.capture_start)
    provider=TradierProvider('fixture',clock=clock,declared_delay_seconds=900)
    provider.__dict__['quotes']={'SPY':{'symbol':'SPY','open':600.,'last':601.,'trade_date':int((clock()-timedelta(minutes=15)).timestamp()*1000)}}
    assert provider.checked_quote('SPY',week.sessions[0].day)
    provider.close()


def test_spx_requires_pm_series_and_full_contract_identity():
    week=trading_week(date(2026,9,14))  # third Friday
    clock=Clock(week.capture_start); prepared=prepared_fixture('SPX',week,clock)
    chain,root=contract_chain(prepared['chain'],'SPX',prepared['expiration'])
    assert root=='SPXW'
    for side in ('calls','puts'):
        for option in prepared['chain'][side]:
            option['root']='SPX'
    with pytest.raises(ValueError,match='SPXW'):
        contract_chain(prepared['chain'],'SPX',prepared['expiration'])


@pytest.mark.parametrize('field,value',[('symbol',None),('expiration_date','2026-09-18'),('contract_size',10),('root','AAPL1')])
def test_unknown_adjusted_or_wrong_date_contracts_cannot_be_fabricated(field,value):
    week=trading_week(date(2026,9,7)); clock=Clock(week.capture_start)
    p=prepared_fixture('AAPL',week,clock)
    for side in ('calls','puts'):
        for opt in p['chain'][side]:opt[field]=value
    with pytest.raises(ValueError,match='verified standard'):
        capture_model(p,'M2_vix',week,'v1',clock())


def test_missing_chain_tier_is_unavailable_and_m4_source_failure_is_explicit():
    week=trading_week(date(2026,9,7)); clock=Clock(week.capture_start)
    p=prepared_fixture('AMD',week,clock)
    p['source_errors']={'macro':'fixture missing FRED'}
    with pytest.raises(ValueError,match='M4_full input source unavailable'):
        capture_model(p,'M4_full',week,'v1',clock())
    assert len(capture_model(p,'M2_vix',week,'v1',clock())[1])==4
    p['chain']['puts']=[]
    with pytest.raises(ValueError,match='Missing verified'):
        capture_model(p,'M2_vix',week,'v1',clock())


def test_timesales_session_and_timestamp_filter(monkeypatch):
    week=trading_week(date(2026,11,23)); session=week.sessions[-1]
    clock=Clock(session.close+timedelta(hours=3))
    provider=TradierProvider('fixture',clock=clock)
    monkeypatch.setattr(provider,'history',lambda *a,**k:[{'date':str(session.day),'open':100.,'high':105.,'low':95.,'close':102.}])
    calls=[]
    def fetch(endpoint,**params):
        calls.append((endpoint,params))
        return {'series':{'data':[{'timestamp':int(when.timestamp()),'open':100.,'high':101.,'low':99.,'close':100.}
                                 for when in (session.open-timedelta(minutes=1),session.open,session.close)]}}
    monkeypatch.setattr(provider,'_get',fetch)
    obs=provider.observe('SPY',session,session.open)
    assert len(obs['minutes'])==1
    assert calls[0][1]['session_filter']=='open' and calls[0][1]['end'].endswith('13:00')
    assert obs['settlement']['status']=='unverified'
    provider.close()


def test_stale_volatility_source_does_not_become_current_week_feature(monkeypatch):
    from range_finder.forward_test import provider as module
    from range_finder import data_collector
    week=trading_week(date(2026,9,7)); clock=Clock(week.capture_start)
    previous=trading_week(week.monday-timedelta(days=7)).sessions[-1].day
    days=pd.date_range(previous-timedelta(days=90),previous,freq='B')
    daily=pd.DataFrame({k:20. for k in ('open','high','low','close')},index=days)
    monkeypatch.setattr(module,'fetch_cboe_index_history',lambda index:daily.iloc[:-1] if index=='VIX3M' else daily)
    monkeypatch.setattr(data_collector,'fetch_fred_macro',lambda **k:pd.DataFrame({'yield_spread':.5,'fed_funds':4.},index=days))
    provider=TradierProvider('fixture',clock=clock)
    assert 'VIX3M' in provider.common[3] and 'VIX' not in provider.common[3]
    provider.close()
    monkeypatch.setattr(module,'fetch_cboe_index_history',lambda index:daily.iloc[:-1])
    provider=TradierProvider('fixture',clock=clock)
    with pytest.raises(ValueError,match='VIX history unavailable'):
        _=provider.common
    provider.close()


def test_prepare_uses_completed_primary_history_and_rejects_session_gap(monkeypatch):
    import pandas_market_calendars as mcal
    from range_finder.forward_test import provider as module
    from range_finder import data_collector
    from range_finder.cboe_data import resample_cboe_weekly
    week=trading_week(date(2026,9,7)); clock=Clock(week.capture_start)
    previous=trading_week(week.monday-timedelta(days=7)).sessions[-1].day
    start=week.monday-timedelta(days=int(6*365.25)+30)
    days=mcal.get_calendar('NYSE').valid_days(start_date=start,end_date=previous).tz_localize(None)
    close=600+np.sin(np.arange(len(days))*.4)*20
    daily=pd.DataFrame({'open':close-1,'high':close+2,'low':close-2,'close':close,'volume':10000.},index=days)
    weekly=resample_cboe_weekly(daily)
    def records(frame):
        return [{'date':str(d.date()),**r.to_dict()} for d,r in frame.iterrows()]
    provider=TradierProvider('fixture',clock=clock)
    monkeypatch.setattr(provider,'history',lambda ticker,start,end,interval='daily':records(weekly if interval=='weekly' else daily))
    monkeypatch.setattr(module,'fetch_cboe_index_history',lambda index:daily/30)
    monkeypatch.setattr(data_collector,'fetch_fred_macro',lambda **k:pd.DataFrame({'yield_spread':.5,'fed_funds':4.},index=days))
    monkeypatch.setattr(provider,'_get',lambda *a,**k:{'quotes':{'quote':[
        {'symbol':ticker,'open':value,'last':value,'trade_date':int(clock().timestamp()*1000)} for ticker,value in [('SPY',600.),('VIX',20.)]]}})
    fixture=prepared_fixture('SPY',week,clock)
    monkeypatch.setattr(provider.client,'get_expirations',lambda ticker:[fixture['expiration']])
    monkeypatch.setattr(provider.client,'get_chain_once',lambda *a:{'status':'ok',**fixture['chain']})
    prepared=provider.prepare('SPY',week)
    assert prepared['reference']==600 and prepared['anchor_at']==week.sessions[0].open.isoformat()
    target=prepared['features'].loc[pd.Timestamp(week.monday)]
    assert pd.isna(target['log_range']) and target['path_source_week']==pd.Timestamp(week.monday-timedelta(days=7))
    assert all(r['date']<str(week.monday) for r in prepared['raw_inputs']['weekly'])
    # A single missing ordinary weekday must not silently shorten HV windows.
    daily.drop(daily.index[-4],inplace=True)
    with pytest.raises(ValueError,match='missing 1 exchange session'):
        provider.prepare('SPY',week)
    provider.close()
