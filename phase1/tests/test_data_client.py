from phase1.data_client import PublicDataClient


def test_parse_iv_from_greeks_normal_decimal():
    """Non-0DTE IV in decimal format (0.22 = 22%) passes through unchanged."""
    greeks = {"impliedVolatility": "0.22"}
    iv = PublicDataClient._parse_iv_from_greeks(greeks, is_0dte=False)
    assert iv == 0.22


def test_parse_iv_from_greeks_small_decimal():
    """Standard decimal IV passes through."""
    greeks = {"impliedVolatility": "0.15"}
    iv = PublicDataClient._parse_iv_from_greeks(greeks, is_0dte=False)
    assert iv == 0.15


def test_parse_iv_from_greeks_0dte_inflated_zeroed():
    """0DTE inflated IV (>3.0) gets zeroed out to trigger synthetic fallback."""
    greeks = {"impliedVolatility": "2.6322"}
    iv = PublicDataClient._parse_iv_from_greeks(greeks, is_0dte=True)
    assert iv == 0.0


def test_parse_iv_from_greeks_non_0dte_high_divided():
    """Non-0DTE IV > 3.0 is divided by 100 (defensive handling)."""
    greeks = {"impliedVolatility": "25.0"}
    iv = PublicDataClient._parse_iv_from_greeks(greeks, is_0dte=False)
    assert iv == 0.25


def test_parse_iv_from_greeks_0dte_boundary():
    """0DTE IV of exactly 1.5 passes through (threshold is > 1.5)."""
    greeks = {"impliedVolatility": "1.5"}
    iv = PublicDataClient._parse_iv_from_greeks(greeks, is_0dte=True)
    assert iv == 1.5


def test_parse_iv_from_greeks_0dte_above_boundary():
    """0DTE IV just above 1.5 gets zeroed out."""
    greeks = {"impliedVolatility": "1.51"}
    iv = PublicDataClient._parse_iv_from_greeks(greeks, is_0dte=True)
    assert iv == 0.0


def test_parse_iv_from_greeks_non_0dte_high_iv_ok():
    """Non-0DTE IV between 1.5 and 3.0 passes through (not inflated)."""
    greeks = {"impliedVolatility": "2.5"}
    iv = PublicDataClient._parse_iv_from_greeks(greeks, is_0dte=False)
    assert iv == 2.5


def test_parse_iv_from_greeks_empty():
    """Empty greeks returns 0.0."""
    assert PublicDataClient._parse_iv_from_greeks({}) == 0.0
    assert PublicDataClient._parse_iv_from_greeks(None) == 0.0


def test_get_chain_cached_uses_cache(monkeypatch):
    client = PublicDataClient(secret_key="dummy")

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
    client = PublicDataClient(secret_key="dummy")

    def fake_get_chain_with_retry(ticker, expiration, retries=0, sleep_sec=0):
        return {"status": "ok", "calls": [], "puts": [], "error": None}

    monkeypatch.setattr(client, "get_chain_with_retry", fake_get_chain_with_retry)

    client.get_chain_cached("SPX", "2026-03-20")
    assert len(client.chain_cache) == 1

    client.clear_cache()
    assert len(client.chain_cache) == 0
