from phase1.data_client import TradierDataClient


def test_parse_iv_from_greeks_handles_percent_style():
    client = TradierDataClient(token="dummy")
    greeks = {"mid_iv": 25.0}
    iv = client._parse_iv_from_greeks(greeks)
    assert iv == 0.25


def test_parse_iv_from_greeks_handles_decimal_style():
    client = TradierDataClient(token="dummy")
    greeks = {"mid_iv": 0.22}
    iv = client._parse_iv_from_greeks(greeks)
    assert iv == 0.22


def test_get_chain_cached_uses_cache(monkeypatch):
    client = TradierDataClient(token="dummy")

    call_count = {"n": 0}

    def fake_get_chain_with_retry(ticker, expiration, retries=0, sleep_sec=0):
        call_count["n"] += 1
        return {"status": "ok", "calls": [], "puts": [], "error": None}

    monkeypatch.setattr(client, "get_chain_with_retry", fake_get_chain_with_retry)

    r1 = client.get_chain_cached("SPX", "2026-03-20")
    r2 = client.get_chain_cached("SPX", "2026-03-20")

    assert r1["status"] == "ok"
    assert r2["status"] == "ok"
    assert call_count["n"] == 1


def test_clear_cache_empties_cache(monkeypatch):
    client = TradierDataClient(token="dummy")

    def fake_get_chain_with_retry(ticker, expiration, retries=0, sleep_sec=0):
        return {"status": "ok", "calls": [], "puts": [], "error": None}

    monkeypatch.setattr(client, "get_chain_with_retry", fake_get_chain_with_retry)

    client.get_chain_cached("SPX", "2026-03-20")
    assert len(client.chain_cache) == 1

    client.clear_cache()
    assert len(client.chain_cache) == 0
