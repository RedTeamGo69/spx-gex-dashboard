from phase1.parity import compute_implied_spot


def test_parity_uses_median_and_ignores_outlier():
    tradier_spot = 5000.0

    calls = [
        {"strike": 4990, "bid": 22.0, "ask": 22.4},
        {"strike": 5000, "bid": 16.0, "ask": 16.4},
        {"strike": 5010, "bid": 11.0, "ask": 11.4},
        {"strike": 5020, "bid": 8.0, "ask": 8.4},
        {"strike": 5030, "bid": 20.0, "ask": 20.4},  # noisy outlier
    ]

    puts = [
        {"strike": 4990, "bid": 12.0, "ask": 12.4},
        {"strike": 5000, "bid": 16.0, "ask": 16.4},
        {"strike": 5010, "bid": 21.0, "ask": 21.4},
        {"strike": 5020, "bid": 28.0, "ask": 28.4},
        {"strike": 5030, "bid": 45.0, "ask": 45.4},  # noisy outlier pair
    ]

    spot, source = compute_implied_spot(calls, puts, tradier_spot, r=0.0, T=0.0)

    assert source.startswith("implied weighted median")
    assert abs(spot - 5000.0) < 20.0


def test_parity_falls_back_if_not_enough_valid_quotes():
    tradier_spot = 5000.0

    calls = [{"strike": 5000, "bid": 0.0, "ask": 0.0}]
    puts = [{"strike": 5000, "bid": 0.0, "ask": 0.0}]

    spot, source = compute_implied_spot(calls, puts, tradier_spot, r=0.0, T=0.0)

    assert spot == tradier_spot
    assert "tradier" in source
