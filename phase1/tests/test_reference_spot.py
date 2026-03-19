from datetime import datetime
from zoneinfo import ZoneInfo

from phase1.parity import get_reference_spot_details

NY = ZoneInfo("America/New_York")


def test_market_closed_forces_tradier(monkeypatch):
    def fake_get_spot_price(_ticker):
        return 5000.0

    def fake_get_chain_cached(_ticker, _exp):
        raise AssertionError("Chain should not be fetched when market is closed")

    import phase1.parity as parity_mod
    monkeypatch.setattr(parity_mod, "is_cash_market_open", lambda now=None: False)

    details = get_reference_spot_details(
        ticker="SPX",
        nearest_exp="2026-03-20",
        get_spot_price_func=fake_get_spot_price,
        get_chain_cached_func=fake_get_chain_cached,
        r=0.0,
        now=datetime(2026, 3, 14, 12, 0, tzinfo=NY),
    )

    assert details["spot"] == 5000.0
    assert details["source"] == "tradier (forced, market closed)"
    assert details["parity_attempted"] is False
    assert details["parity_diagnostics"] is None


def test_market_open_can_use_implied(monkeypatch):
    def fake_get_spot_price(_ticker):
        return 5000.0

    def fake_get_chain_cached(_ticker, _exp):
        return {
            "status": "ok",
            "calls": [
                {"strike": 4990, "bid": 22.0, "ask": 22.4},
                {"strike": 5000, "bid": 16.0, "ask": 16.4},
                {"strike": 5010, "bid": 11.0, "ask": 11.4},
            ],
            "puts": [
                {"strike": 4990, "bid": 12.0, "ask": 12.4},
                {"strike": 5000, "bid": 16.0, "ask": 16.4},
                {"strike": 5010, "bid": 21.0, "ask": 21.4},
            ],
        }

    import phase1.parity as parity_mod
    monkeypatch.setattr(parity_mod, "is_cash_market_open", lambda now=None: True)

    details = get_reference_spot_details(
        ticker="SPX",
        nearest_exp="2026-03-20",
        get_spot_price_func=fake_get_spot_price,
        get_chain_cached_func=fake_get_chain_cached,
        r=0.0,
        now=datetime(2026, 3, 20, 10, 0, tzinfo=NY),
    )

    assert details["parity_attempted"] is True
    assert details["parity_chain_status"] == "ok"
    assert details["source"].startswith("implied")
    assert details["implied_spot"] is not None
    assert details["expiration_close_ny"] is not None

    diag = details["parity_diagnostics"]
    assert diag is not None
    assert diag["call_quality"]["usable"] == 3
    assert diag["put_quality"]["usable"] == 3
    assert diag["common_usable_strikes"] == 3
    assert diag["final_atm_strikes"] == 3
