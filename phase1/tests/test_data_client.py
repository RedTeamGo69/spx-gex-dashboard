import phase1.data_client as data_client_mod
from phase1.data_client import TradierDataClient

import pytest


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_get_spot_price_handles_unmatched_symbol(monkeypatch):
    """Tradier returns unmatched_symbols (no 'quote' key) for any symbol it
    can't price. The old bare subscript raised KeyError('quote') → the
    'Engine error: quote' the user saw. It must now degrade to 0.0."""
    client = TradierDataClient(token="dummy")
    payload = {"quotes": {"unmatched_symbols": {"symbol": "ZZZZ"}}}
    monkeypatch.setattr(data_client_mod.requests, "get",
                        lambda *a, **k: _FakeResponse(payload))

    assert client.get_spot_price("ZZZZ") == 0.0


@pytest.mark.parametrize(
    ("quote", "expected"),
    [
        ({"last": 5030.25, "close": 5012.5, "prevclose": 4999}, 5030.25),
        ({"last": None, "close": 5012.5, "prevclose": 4999}, 5012.5),
        ({"last": 0, "close": None, "prevclose": 4999}, 4999.0),
        ({"last": -1, "close": 0, "prevclose": None}, 0.0),
    ],
)
def test_resolve_quote_spot_uses_one_positive_fallback_chain(quote, expected):
    assert data_client_mod.resolve_quote_spot(quote) == expected


def test_normalized_after_hours_quote_keeps_valid_close():
    normalized = TradierDataClient._normalize_quote(
        {"last": None, "close": 5012.5, "prevclose": 4999}, "SPX",
    )

    assert normalized["last"] == 0.0
    assert data_client_mod.resolve_quote_spot(normalized) == 5012.5


def test_get_full_quotes_handles_unmatched_symbol(monkeypatch):
    client = TradierDataClient(token="dummy")
    payload = {"quotes": {"unmatched_symbols": {"symbol": "ZZZZ"}}}
    monkeypatch.setattr(data_client_mod.requests, "get",
                        lambda *a, **k: _FakeResponse(payload))

    assert client.get_full_quotes(["ZZZZ"]) == {}


def test_get_full_quote_degrades_to_zeroed_quote(monkeypatch):
    client = TradierDataClient(token="dummy")
    payload = {"quotes": {"unmatched_symbols": {"symbol": "ZZZZ"}}}
    monkeypatch.setattr(data_client_mod.requests, "get",
                        lambda *a, **k: _FakeResponse(payload))

    q = client.get_full_quote("ZZZZ")
    assert q["symbol"] == "ZZZZ"
    assert q["last"] == 0.0


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
