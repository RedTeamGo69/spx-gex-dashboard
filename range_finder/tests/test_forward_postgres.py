"""Real PostgreSQL gates; explicit isolated test URL only, never DATABASE_URL."""
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
import json
import os
from urllib.parse import urlparse
from uuid import uuid4

import pytest

from range_finder.forward_test.store import Store
from range_finder.forward_test.runner import run_study
from range_finder.forward_test.results import load_results, metrics, build_workbook
from range_finder.tests.forward_fixtures import Clock, FixtureProvider, prepared_fixture
from range_finder.forward_test.capture import capture_model
from range_finder.trading_week import trading_week


@pytest.fixture
def pg_store():
    url=os.environ.get('FORWARD_TEST_TEST_DATABASE_URL')
    if not url:
        pytest.skip('Set FORWARD_TEST_TEST_DATABASE_URL to an isolated local *_test database')
    parsed=urlparse(url)
    if parsed.hostname not in ('127.0.0.1','localhost') or not parsed.path.endswith('_test'):
        pytest.fail('Refusing to migrate anything except a local isolated *_test database')
    import psycopg2
    from psycopg2 import sql
    conn=psycopg2.connect(url)
    schema='ft_fixture_'+uuid4().hex
    with conn.cursor() as cur:
        cur.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
        cur.execute(sql.SQL('SET search_path TO {}').format(sql.Identifier(schema)))
    conn.commit()
    clock=Clock(trading_week(date(2026,9,7)).capture_start+timedelta(seconds=10))
    store=Store(conn,admission_clock=clock)
    store.migrate()
    yield store,clock,url,schema
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))
    conn.commit()
    store.close()


def test_postgres_full_lifecycle_rollback_and_immutability(pg_store):
    store,clock,_,_=pg_store
    week=trading_week(clock().date())
    store.migrate()  # idempotent DDL
    store.register('postgres-fixture',str(week.monday),clock(),{'fixture':True})
    provider=FixtureProvider(clock)
    result=run_study(store,provider,'postgres-fixture',clock=clock,model_version='fixture-v1')
    assert result['captured_models']==16 and not result['errors']
    assert len(store.forecasts('postgres-fixture'))==64
    import psycopg2
    for table in ('ft_forecasts','ft_inputs','ft_studies'):
        with pytest.raises(psycopg2.Error,match='immutable'):
            with store.transaction():
                store.execute(f'DELETE FROM {table}')
    clock.value=week.evaluation_close+timedelta(hours=3)
    run_study(store,provider,'postgres-fixture',clock=clock,model_version='fixture-v1')
    records=load_results(store,'postgres-fixture')
    assert metrics(records)['close_n']==64 and metrics(records)['path_n']==64
    assert build_workbook(records)[:2]==b'PK'
    for table in ('ft_observations','ft_scores'):
        with pytest.raises(psycopg2.Error,match='immutable'):
            with store.transaction(): store.execute(f'DELETE FROM {table}')
    # Scoring updates append, including corrections back to the first result.
    f=records[0]
    store.append_score(f['forecast_id'],{'corrected':1},'test',clock())
    store.append_score(f['forecast_id'],{'corrected':2},'test',clock())
    store.append_score(f['forecast_id'],{'corrected':1},'test',clock())
    assert json.loads(store.query('SELECT payload_json FROM ft_scores WHERE forecast_id=? ORDER BY revision DESC LIMIT 1',(f['forecast_id'],))[0]['payload_json'])=={'corrected':1}


def test_postgres_concurrent_winner_and_atomic_failure(pg_store):
    store,clock,url,schema=pg_store
    week=trading_week(clock().date())
    store.register('x',str(week.monday),clock(),{'fixture':True})
    store.seed_week('x',str(week.monday),'v1',clock())
    slot=store.slots('x',str(week.monday))[0]
    inputs,forecasts=capture_model(prepared_fixture(slot['ticker'],week,clock),slot['model'],week,'v1',clock())
    # An in-window application timestamp cannot bypass the DB's later clock.
    in_window=clock()
    clock.value=week.capture_end
    with pytest.raises(ValueError,match='Database admission time'):
        store.freeze(slot,inputs,forecasts,in_window,week)
    clock.value=in_window
    assert store.query('SELECT count(*) AS n FROM ft_forecasts')[0]['n']==0
    checks=iter([in_window,week.capture_end])
    store.admission_clock=lambda:next(checks)
    with pytest.raises(ValueError,match='after writes'):
        store.freeze(slot,inputs,forecasts,in_window,week)
    store.admission_clock=clock
    assert store.query('SELECT count(*) AS n FROM ft_forecasts')[0]['n']==0
    assert store.query('SELECT count(*) AS n FROM ft_inputs')[0]['n']==0
    assert store.slots('x',str(week.monday))[0]['status']=='pending'
    bad=[dict(f) for f in forecasts]
    bad[-1]['expiration']=None
    import psycopg2
    with pytest.raises(psycopg2.Error):
        store.freeze(slot,inputs,bad,clock(),week)
    assert store.query('SELECT count(*) AS n FROM ft_forecasts')[0]['n']==0
    assert store.query('SELECT count(*) AS n FROM ft_inputs')[0]['n']==0
    def capture(_):
        from psycopg2 import sql
        conn=psycopg2.connect(url)
        with conn.cursor() as cur:cur.execute(sql.SQL('SET search_path TO {}').format(sql.Identifier(schema)))
        conn.commit()
        concurrent=Store(conn,admission_clock=clock)
        try:return concurrent.freeze(slot,inputs,forecasts,clock(),week)
        finally:concurrent.close()
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(capture,range(4)))==1
    assert len(store.forecasts('x'))==4
