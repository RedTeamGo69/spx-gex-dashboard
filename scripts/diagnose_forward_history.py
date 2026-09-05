"""Read-only history evidence; no fits, database writes or price repair.

Archive response bodies (credentials removed), request conventions and hashes.
The output directory must be new, so earlier source evidence is not overwritten.
"""
import argparse
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import sys
import tomllib
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import requests
from curl_cffi import requests as curl_requests

from range_finder.forward_test.config import UNIVERSE, utcnow
from range_finder.forward_test.provider import TradierProvider, valid_ohlc, frame_records
from range_finder.forward_test.store import Store
from range_finder.trading_week import NY, trading_week


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--secrets-file', default='.streamlit/secrets.toml')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    cfg = tomllib.loads(Path(args.secrets_file).read_text())
    manifest = []

    def save(name, response, params, source):
        body = response.content
        sanitized = body.decode('utf-8')
        for key, value in cfg.items():
            if isinstance(value, str) and value:
                sanitized = sanitized.replace(value, '[REDACTED]')
        target = output / (name + '.json')
        target.write_bytes(sanitized.encode('utf-8'))
        manifest.append({'file':target.name, 'source':source,
            'url':response.url.split('?')[0], 'params':{k:v for k,v in params.items() if k!='api_key'},
            'retrieved_at':utcnow().isoformat(), 'http_status':response.status_code,
            'raw_sha256':hashlib.sha256(body).hexdigest(),
            'sanitized_sha256':hashlib.sha256(sanitized.encode()).hexdigest()})
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        if response.status_code != 200:
            raise ValueError(f'{source} HTTP {response.status_code}')
        return json.loads(sanitized)

    now = utcnow()
    target = trading_week(now.astimezone(NY).date())
    if now >= target.capture_end:
        target = trading_week(target.monday+timedelta(days=7))
    previous = trading_week(target.monday-timedelta(days=7)).sessions[-1].day
    start = target.monday-timedelta(days=int(6*365.25)+30)
    provider = TradierProvider(cfg['TRADIER_TOKEN'], clock=utcnow)
    store = Store.postgres(cfg['DATABASE_URL'])
    store.conn.set_session(readonly=True)
    yahoo = curl_requests.Session(impersonate='chrome')
    summary = {'mode':'read_only_no_forecasts', 'required_start':str(start),
               'end':str(previous), 'tickers':{}, 'errors':[]}
    try:
        for ticker in UNIVERSE:
            item = summary['tickers'][ticker] = {}
            older = store.legacy_weekly(ticker)
            (output/f'legacy-{ticker}.json').write_text(json.dumps(frame_records(older), indent=2), encoding='utf-8')
            older_start = min(start, older.index.min().date()) if not older.empty else start
            item['legacy_weekly_rows'] = len(older)
            item['legacy_first_week'] = str(older_start)
            rows = {}
            for interval in ('daily','weekly'):
                begin = start if interval=='daily' else older_start
                params = {'symbol':ticker,'interval':interval,'start':str(begin),'end':str(previous)}
                response = provider.http.get(provider.client.base_url+'/markets/history', params=params, timeout=20)
                data = save(f'tradier-{ticker}-{interval}',response,params,'Tradier Brokerage history')
                values = (data.get('history') or {}).get('day') or []
                if isinstance(values,dict): values=[values]
                rows[interval] = values
                item[interval] = {'rows':len(values),'invalid':[r for r in values if not valid_ohlc(r)]}
            # Exact re-requests preserve the provider's raw date labels. Weekly
            # labels are checked against the full range rather than normalized.
            for bar in item['daily']['invalid']:
                params = {'symbol':ticker,'interval':'daily','start':bar['date'],'end':bar['date']}
                response = provider.http.get(provider.client.base_url+'/markets/history',params=params,timeout=20)
                save(f"tradier-{ticker}-single-{bar['date']}",response,params,'Tradier Brokerage history')
                week = trading_week(datetime.fromisoformat(bar['date']).date())
                params.update(interval='weekly',start=str(week.monday),end=str(week.sessions[-1].day))
                response = provider.http.get(provider.client.base_url+'/markets/history',params=params,timeout=20)
                save(f"tradier-{ticker}-week-{week.monday}",response,params,'Tradier Brokerage history')
            symbol = '^GSPC' if ticker=='SPX' else ticker
            for interval in ('1d','1wk'):
                params = {'period1':int(datetime.combine(older_start,datetime.min.time(),NY).timestamp()),
                    'period2':int(datetime.combine(previous+timedelta(days=1),datetime.min.time(),NY).timestamp()),
                    'interval':interval,'includePrePost':'false','events':'div,splits,capitalGains'}
                try:
                    response = yahoo.get('https://query2.finance.yahoo.com/v8/finance/chart/'+quote(symbol,safe=''),
                                         params=params,timeout=20)
                    data = save(f'yahoo-{ticker}-{interval}',response,params,'Yahoo chart; raw quote OHLC, no auto-adjust or repair')
                    result = data['chart']['result'][0]
                    item['yahoo_'+interval] = {'bars':len(result.get('timestamp',[])),
                                               'symbol':result['meta']['symbol']}
                except Exception as exc:
                    summary['errors'].append(f'{ticker} Yahoo {interval}: {type(exc).__name__}')
            print(json.dumps({'ticker':ticker,**item}),flush=True)
        params={'series_id':'SP500','api_key':cfg['FRED_API_KEY'],'file_type':'json',
                'observation_start':str(start),'observation_end':str(previous)}
        response=requests.get('https://api.stlouisfed.org/fred/series/observations',params=params,timeout=20)
        fred=save('fred-SP500',response,params,'FRED SP500; S&P DJI daily price-index close')
        summary['fred_sp500_rows']=len(fred.get('observations',[]))
    finally:
        provider.close(); store.close(); yahoo.close()
        (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(str(output))


if __name__=='__main__':
    main()
